include(cc_binary)
include(cc_library)
include(cc_test)


cc_library(
  NAME
    weight_transfer
  HDRS
    hccl_weight_transfer.h
  SRCS
    hccl_weight_transfer.cpp
  DEPS
    :common
    glog::glog
    torch
    $<$<BOOL:${USE_NPU}>:torch_npu>
    $<$<BOOL:${USE_NPU}>:hccl_transfer>
    proto::xllm_proto
    :util
    brpc
)
