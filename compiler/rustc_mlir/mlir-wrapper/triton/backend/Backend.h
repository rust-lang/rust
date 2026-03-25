/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRITON_BACKEND_H
#define TRITON_BACKEND_H

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "llvm/IR/Module.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#define CHECK_RESULT(result, msg)                                              \
  if (failed(result)) {                                                        \
    llvm::errs() << msg << "\n";                                               \
    return result;                                                             \
  }

namespace mlir {
namespace triton {

using namespace mlir::triton::gpu;
using namespace mlir::triton::instrument;

enum Error {
  InvalidPass,
  InvalidLLVMModule,
};

enum Language {
  TRITON,
  GLUON,
};

enum MlirPass {
  // common
  sccp,
  symbol_dce,
  inliner,
  canonicalizer,
  cse,
  licm,
  print_ir,

  // ttir
  ttir_combine,
  ttir_reorder_broadcast,
  ttir_rewrite_tensor_pointer,
  ttir_rewrite_tensor_descriptor_to_pointer,
  ttir_loop_unroll,
  ttir_triton_licm,
  ttir_loop_aware_cse,
  ttir_convert_to_ttgpuir,

  // ttgpuir
  ttgpuir_coalesce,
  ttgpuir_optimize_thread_locality,
  ttgpuir_hoist_tmem_alloc,
  ttgpuir_assign_latencies,
  ttgpuir_schedule_loops,
  ttgpuir_pipeline,
  ttgpuir_warp_specialize,
  ttgpuir_prefetch,
  ttgpuir_accelerate_matmul,
  ttgpuir_reorder_instructions,
  ttgpuir_f32_dot_tc,
  ttgpuir_optimize_dot_operands,
  ttgpuir_remove_layout_conversions,
  ttgpuir_reduce_data_duplication,
  ttgpuir_allocate_warp_groups,
  ttgpuir_allocate_shared_memory,
  ttgpuir_allocate_global_scratch_memory,
  ttgpuir_combine_tensor_select_and_if,
  ttgpuir_optimize_accumulator_init,
  ttgpuir_fuse_nested_loops,
  ttgpuir_coalesce_async_copy,
  ttgpuir_concurrency_sanitizer,
  ttgpuir_optimize_partition_warps,
  ttgpuir_allocate_shared_memory_nv,

  // convert
  scf_to_cf,
  cf_to_llvmir,
  index_to_llvmir,
  arith_to_llvmir,
  nvvm_to_llvm,

  // count
  llvmir_di_scope,
  llvmir_di_local_variable,

  // gluon
  gluon_resolve_auto_encodings,
  gluon_canonicalizer,
  gluon_inliner,
  gluon_infer_coalesced_encodings,
};

class Backend {
public:
  Backend(std::string target);

  virtual ~Backend();

  const std::optional<Error> getLastError() const { return m_last_error; };

  const std::string &getLastErrorString() const { return m_last_error_string; };

  virtual void loadDialects(MLIRContext &context) = 0;

  virtual LogicalResult applyPasses(MLIRContext &context, ModuleOp module,
                                    Language language);

  virtual LogicalResult makeTTIR(MLIRContext &context, ModuleOp module) = 0;

  virtual LogicalResult makeTTGIR(MLIRContext &context, ModuleOp module) = 0;

  virtual LogicalResult gluonToTTGIR(MLIRContext &context, ModuleOp module) = 0;

  virtual LogicalResult makeLLIR(MLIRContext &context, ModuleOp module) = 0;

  virtual LogicalResult makeLLVMIR(MLIRContext &context, ModuleOp module) = 0;

  virtual LogicalResult makeASM(MLIRContext &context, ModuleOp module) = 0;

  virtual LogicalResult makeBIN(MLIRContext &context, ModuleOp module) = 0;

  void printIR(std::string stage, ModuleOp module);

  const char *getLLIR() const { return m_llir.c_str(); }

  const char *getTTIR() const { return m_ttir.c_str(); }

  const char *getTTGIR() const { return m_ttgir.c_str(); }

  const char *getLLVMIR() const { return m_llvmir.c_str(); }

  const char *getASM() const { return m_asm.c_str(); }

  const char *getBIN() const { return m_bin.c_str(); }

protected:
  std::string m_target;
  std::optional<Error> m_last_error;
  std::string m_last_error_string = "";

  std::string m_ttir;
  std::string m_ttgir;
  std::string m_llir;
  std::string m_llvmir;

  std::string m_asm;
  std::string m_bin; // utf-8 encoded string

  virtual std::optional<Error> addPass(PassManager &pm, MlirPass pass);

  virtual std::optional<Error> addPass(PassManager &pm, MlirPass pass,
                                       int arg0);

  virtual std::optional<Error> addPass(PassManager &pm, MlirPass pass,
                                       bool arg0);

  virtual std::optional<Error> addPass(PassManager &pm, MlirPass pass, int arg0,
                                       bool arg1);

  virtual std::optional<Error> addPass(PassManager &pm, MlirPass pass,
                                       const std::string &arg0, int arg1,
                                       int arg2, int arg3);

private:
  // triton passes
  std::unordered_map<MlirPass, std::unique_ptr<Pass> (*)()> m_pass_fns = {
      // common
      {MlirPass::sccp, createSCCPPass},
      {MlirPass::symbol_dce, createSymbolDCEPass},
      {MlirPass::inliner, createInlinerPass},
      {MlirPass::canonicalizer, createCanonicalizerPass},
      {MlirPass::cse, createCSEPass},
      {MlirPass::licm, createLoopInvariantCodeMotionPass},

      // ttir
      {MlirPass::ttir_combine, createTritonCombineOps},
      {MlirPass::ttir_reorder_broadcast, createTritonReorderBroadcast},
      {MlirPass::ttir_rewrite_tensor_pointer, createTritonRewriteTensorPointer},
      {MlirPass::ttir_rewrite_tensor_descriptor_to_pointer,
       createTritonRewriteTensorDescriptorToPointer},
      {MlirPass::ttir_loop_unroll, createTritonLoopUnroll},
      {MlirPass::ttir_triton_licm, createTritonLoopInvariantCodeMotion},
      {MlirPass::ttir_loop_aware_cse, createTritonLoopAwareCSE},

      // ttgpuir
      {MlirPass::ttgpuir_coalesce, createTritonGPUCoalesce},
      {MlirPass::ttgpuir_optimize_thread_locality,
       createTritonGPUOptimizeThreadLocality},
      {MlirPass::ttgpuir_schedule_loops, createTritonGPUScheduleLoops},
      {MlirPass::ttgpuir_prefetch, createTritonGPUPrefetch},
      {MlirPass::ttgpuir_accelerate_matmul, createTritonGPUAccelerateMatmul},
      {MlirPass::ttgpuir_reorder_instructions,
       createTritonGPUReorderInstructions},
      {MlirPass::ttgpuir_remove_layout_conversions,
       createTritonGPURemoveLayoutConversions},
      {MlirPass::ttgpuir_reduce_data_duplication,
       createTritonGPUReduceDataDuplication},
      {MlirPass::ttgpuir_allocate_warp_groups,
       createTritonGPUAllocateWarpGroups},
      {MlirPass::ttgpuir_allocate_shared_memory, createAllocateSharedMemory},
      {MlirPass::ttgpuir_allocate_global_scratch_memory,
       createTritonGPUGlobalScratchAllocationPass},
      {MlirPass::ttgpuir_combine_tensor_select_and_if,
       createTritonGPUCombineTensorSelectAndIf},
      {MlirPass::ttgpuir_optimize_accumulator_init,
       createTritonGPUOptimizeAccumulatorInit},
      {MlirPass::ttgpuir_fuse_nested_loops, createTritonGPUFuseNestedLoops},
      {MlirPass::ttgpuir_coalesce_async_copy, createTritonGPUCoalesceAsyncCopy},
      {MlirPass::ttgpuir_concurrency_sanitizer,
       createTritonInstrumentConcurrencySanitizer},
      {MlirPass::ttgpuir_optimize_partition_warps,
       createTritonGPUOptimizePartitionWarps},

      // convert
      {MlirPass::scf_to_cf, createSCFToControlFlowPass},
      {MlirPass::cf_to_llvmir, createConvertControlFlowToLLVMPass},
      {MlirPass::index_to_llvmir, createConvertIndexToLLVMPass},
      {MlirPass::arith_to_llvmir, createArithToLLVMConversionPass},
      {MlirPass::nvvm_to_llvm, createConvertNVVMToLLVMPass},

      // llvmir
      {MlirPass::llvmir_di_scope, createLLVMDIScope},
      {MlirPass::llvmir_di_local_variable, createLLVMDILocalVariable},

      // gluon
      {MlirPass::gluon_resolve_auto_encodings,
       gluon::createGluonResolveAutoEncodingsPass},
      {MlirPass::gluon_canonicalizer, gluon::createGluonCanonicalize},
      {MlirPass::gluon_inliner, gluon::createGluonInline},
      {MlirPass::gluon_infer_coalesced_encodings,
       gluon::createGluonInferCoalescedEncodingsPass},
  };
};

} // namespace triton
} // namespace mlir

#endif /*! TRITON_BACKEND_H */
