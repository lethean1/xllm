#include "hccl_weight_transfer.h"

#include <glog/logging.h>

#include "util/net.h"  // 假设你有获取 local ip 的工具

namespace xllm {

class WeightTransferServiceImpl : public xllm::proto::WeightTransferService {
 public:
  explicit WeightTransferServiceImpl(HcclWeightTransfer* hccl_weight_transfer)
      : hccl_weight_transfer_(hccl_weight_transfer) {}

  void InitComm(google::protobuf::RpcController* controller,
                const xllm::proto::InitCommRequest* request,
                xllm::proto::InitCommResponse* response,
                google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);

    std::string remote_addr = request->addr();
    std::string root_info_str = request->root_info();

    std::thread([this, remote_addr, root_info_str]() {
      LOG(INFO) << "Sender Async Thread: Start Waiting for Receiver to join "
                   "HCCL group...";

      hccl_weight_transfer_->handle_init_comm(remote_addr,
                                              root_info_str.data());

      LOG(INFO) << "Sender Async Thread: HCCL Init DONE! Handshake complete.";
    }).detach();
    response->set_success(true);
    LOG(INFO)
        << "Sender: RootInfo generated and sent back. Async Init triggered.";
  }

  void GetLayerMeta(google::protobuf::RpcController* controller,
                    const xllm::proto::GetLayerMetaRequest* request,
                    xllm::proto::GetLayerMetaResponse* response,
                    google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    auto tensors =
        hccl_weight_transfer_->get_registered_tensors(request->layer_id());

    for (const auto& t : tensors) {
      auto* meta = response->add_metas();
      meta->set_dtype(static_cast<int32_t>(t.scalar_type()));
      for (int i = 0; i < t.dim(); ++i) {
        meta->add_shape(t.size(i));
      }
    }
  }

  void TriggerSend(google::protobuf::RpcController* controller,
                   const xllm::proto::TriggerSendRequest* request,
                   xllm::proto::TriggerSendResponse* response,
                   google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    hccl_weight_transfer_->process_send_request(request->layer_id());
    response->set_success(true);
  }

 private:
  HcclWeightTransfer* hccl_weight_transfer_;
};

HcclWeightTransfer::HcclWeightTransfer(int32_t device_id, int32_t listen_port)
    : device_id_(device_id), listen_port_(listen_port) {
  aclrtSetDevice(device_id_);
  aclrtCreateStream(&stream_);

  std::string ip = net::get_local_ip_addr();
  local_addr_ = ip + ":" + std::to_string(listen_port);
  hccl_thread_pool_ = std::make_shared<hccl_transfer::ThreadPool>();
  rpc_thread_pool_ = std::make_shared<hccl_transfer::ThreadPool>();
}

HcclWeightTransfer::~HcclWeightTransfer() {
  if (server_.IsRunning()) server_.Stop(0);
  server_.Join();
  if (hccl_comm_) HcclCommDestroy(hccl_comm_);
  aclrtDestroyStream(stream_);
}

void HcclWeightTransfer::register_layer(
    int32_t layer_id,
    const std::vector<at::Tensor>& tensors) {
  LOG(INFO) << "layer_id: " << layer_id;
  layer_registry_[layer_id] = tensors;
}

std::vector<at::Tensor> HcclWeightTransfer::get_registered_tensors(
    int32_t layer_id) {
  if (layer_registry_.find(layer_id) == layer_registry_.end()) {
    return {};
  }
  return layer_registry_[layer_id];
}

void HcclWeightTransfer::start_serving() {
  service_ = std::make_unique<WeightTransferServiceImpl>(this);
  if (server_.AddService(service_.get(), brpc::SERVER_DOESNT_OWN_SERVICE) !=
      0) {
    LOG(ERROR) << "Failed to add service to server";
  }
  brpc::ServerOptions options;
  if (server_.Start(listen_port_, &options) != 0) {
    LOG(ERROR) << "Failed to start Brpc rpc server";
  }
  LOG(INFO) << "Weight Transfer Server started on " << local_addr_;
}

bool HcclWeightTransfer::handle_init_comm(const std::string& remote_addr,
                                          const void* root_info_ptr) {
  if (is_comm_initialized_) return true;

  LOG(INFO) << "Sender: Initializing HCCL Comm with Receiver " << remote_addr;
  aclrtSetDevice(device_id_);

  HcclRootInfo root_info;
  memcpy(&root_info, root_info_ptr, sizeof(HcclRootInfo));

  auto ret = HcclCommInitRootInfo(2, &root_info, 1, &hccl_comm_);
  if (ret != HCCL_SUCCESS) {
    LOG(ERROR) << "HcclCommInitRootInfo failed: " << ret;
    return false;
  }
  is_comm_initialized_ = true;
  return true;
}

void HcclWeightTransfer::process_send_request(int32_t layer_id) {
  int wait_retry = 0;
  while (!is_comm_initialized_) {
    if (wait_retry % 100 == 0) {  // 每 1秒打印一次日志
      LOG(WARNING)
          << "Sender: Waiting for HCCL Init to finish before sending layer "
          << layer_id << "...";
    }
    usleep(10000);  // 睡眠 10ms
    wait_retry++;

    if (wait_retry > 2000) {  // 等待超过 20秒 则放弃，防止死锁
      LOG(ERROR) << "Sender: FATAL TIMEOUT waiting for HCCL Init.";
      return;
    }
  }

  aclrtSetDevice(device_id_);
  auto tensors = get_registered_tensors(layer_id);
  if (tensors.empty()) {
    LOG(ERROR) << "Request to send unknown layer " << layer_id;
    return;
  }

  auto promise = std::make_shared<std::promise<bool>>();
  std::future<bool> future = promise->get_future();

  hccl_thread_pool_->schedule([&]() mutable {
    aclrtSetDevice(device_id_);
    // LOG(INFO) << "[Sender Thread] Task started for Layer " << layer_id;

    for (size_t i = 0; i < tensors.size(); ++i) {
      const auto& tensor = tensors[i];
      size_t nbytes = tensor.nbytes();

      auto hccl_ret = HcclSend(tensor.data_ptr(),
                               nbytes,
                               HCCL_DATA_TYPE_UINT8,
                               0,
                               hccl_comm_,
                               stream_);

      if (hccl_ret != HCCL_SUCCESS) {
        LOG(ERROR) << "[Sender Thread] HcclSend Failed at index " << i;
        promise->set_value(false);
        return;
      }
    }
    auto sync_ret = aclrtSynchronizeStream(stream_);

    if (sync_ret != ACL_SUCCESS) {
      promise->set_value(false);
    } else {
      promise->set_value(true);
    }
  });
  bool result = future.get();

  if (!result) {
    LOG(ERROR) << "Sender process failed for layer " << layer_id;
  }
}

bool HcclWeightTransfer::connect_to_remote(const std::string& remote_addr) {
  aclrtSetDevice(device_id_);
  channel_ = std::make_unique<brpc::Channel>();
  brpc::ChannelOptions options;
  options.timeout_ms = 5000;
  options.connect_timeout_ms = 2000;
  options.max_retry = 3;

  // 1. 初始化 Channel
  if (channel_->Init(remote_addr.c_str(), &options) != 0) {
    LOG(ERROR) << "BRPC Channel init failed";
    return false;
  }
  stub_ =
      std::make_unique<xllm::proto::WeightTransferService_Stub>(channel_.get());

  // 2. 准备 HCCL Root Info
  aclrtSetDevice(device_id_);
  HcclRootInfo root_info;
  auto ret = HcclGetRootInfo(&root_info);
  if (ret != HCCL_SUCCESS) {
    LOG(ERROR) << "HcclGetRootInfo failed";
    return false;
  }

  // 3. 【关键修改】循环尝试连接 (Poling)
  // Sender 可能正在加载权重，我们需要一直试，直到它端口打开
  int max_wait_retries = 100;  // 最多等 100 次
  bool connect_success = false;

  xllm::proto::InitCommRequest req;
  xllm::proto::InitCommResponse resp;
  req.set_addr(local_addr_);
  req.set_root_info(&root_info, sizeof(HcclRootInfo));

  for (int i = 0; i < max_wait_retries; ++i) {
    brpc::Controller cntl;

    // 尝试发起 RPC
    stub_->InitComm(&cntl, &req, &resp, nullptr);

    if (!cntl.Failed()) {
      // RPC 成功，说明连上了！
      if (resp.success()) {
        LOG(INFO) << "Receiver: Successfully connected to Sender at "
                  << remote_addr;
        connect_success = true;
        break;
      } else {
        LOG(ERROR) << "Receiver: Connected, but Sender returned logic error.";
        return false;
      }
    }

    // 既然是 Connection refused，说明对方还没起，打印日志并等待
    if (i % 5 == 0) {
      LOG(WARNING) << "Receiver: Waiting for Sender (" << remote_addr
                   << ") to come online... (Attempt " << i + 1 << "/"
                   << max_wait_retries << ")";
    }

    // 睡眠 1 秒后再试
    sleep(1);
  }

  if (!connect_success) {
    LOG(ERROR)
        << "Receiver: FATAL - Timed out waiting for Sender to start after "
        << max_wait_retries << " seconds.";
    return false;
  }

  // 4. 接收端初始化 HCCL (Rank 1)
  LOG(INFO) << "Receiver: Initializing HCCL Comm (Rank 1)...";
  ret = HcclCommInitRootInfo(2, &root_info, 0, &hccl_comm_);
  if (ret != HCCL_SUCCESS) {
    LOG(ERROR) << "HcclCommInitRootInfo (Client) failed: " << ret;
    return false;
  }

  is_comm_initialized_ = true;
  return true;
}

bool HcclWeightTransfer::pull_layer(int32_t layer_id,
                                    std::vector<at::Tensor>& local_tensors) {
  if (!is_comm_initialized_) return false;

  brpc::Controller cntl_meta;
  xllm::proto::GetLayerMetaRequest req_meta;
  xllm::proto::GetLayerMetaResponse resp_meta;
  req_meta.set_layer_id(layer_id);

  stub_->GetLayerMeta(&cntl_meta, &req_meta, &resp_meta, nullptr);
  if (cntl_meta.Failed()) {
    LOG(ERROR) << "GetLayerMeta failed: " << cntl_meta.ErrorText();
    return false;
  }

  int tensor_count = resp_meta.metas_size();
  local_tensors.resize(tensor_count);

  aclrtSetDevice(device_id_);

  for (int i = 0; i < tensor_count; ++i) {
    const auto& meta = resp_meta.metas(i);
    std::vector<int64_t> shape;
    for (int64_t d : meta.shape()) shape.push_back(d);

    auto options = torch::TensorOptions()
                       .dtype(static_cast<at::ScalarType>(meta.dtype()))
                       .device("npu:" + std::to_string(device_id_));

    local_tensors[i] = torch::empty(shape, options);
  }

  rpc_thread_pool_->schedule([this, layer_id]() {
    brpc::Controller cntl_trig;
    xllm::proto::TriggerSendRequest req_trig;
    xllm::proto::TriggerSendResponse resp_trig;
    req_trig.set_layer_id(layer_id);

    stub_->TriggerSend(&cntl_trig, &req_trig, &resp_trig, nullptr);

    if (cntl_trig.Failed() || !resp_trig.success()) {
      LOG(ERROR) << "TriggerSend failed: " << cntl_trig.ErrorText();
    }
  });
  auto promise = std::make_shared<std::promise<bool>>();
  std::future<bool> future = promise->get_future();

  hccl_thread_pool_->schedule([&]() mutable {
    aclError ret;
    ret = aclrtSetDevice(device_id_);

    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "[Receiver Thread] SetContext Failed: " << ret;
      promise->set_value(false);
      return;
    }

    for (size_t i = 0; i < local_tensors.size(); ++i) {
      auto& tensor = local_tensors[i];
      auto hccl_ret = HcclRecv(tensor.data_ptr(),
                               tensor.nbytes(),
                               HCCL_DATA_TYPE_UINT8,
                               1,
                               hccl_comm_,
                               stream_);

      if (hccl_ret != HCCL_SUCCESS) {
        LOG(ERROR) << "[Receiver Thread] HcclRecv Failed.";
        promise->set_value(false);
        return;
      }
    }

    auto sync_ret = aclrtSynchronizeStream(stream_);

    promise->set_value(sync_ret == ACL_SUCCESS);
  });

  bool result = future.get();
  if (!result) {
    LOG(ERROR) << "Push layer failed!";
  }

  return result;
}

}  // namespace xllm