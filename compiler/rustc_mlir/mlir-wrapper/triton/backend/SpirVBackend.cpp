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

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Pass/PassManager.h"

#include "spirv/include/Conversion/TritonGPUToSPIRV/TritonGPUToSPIRVPass.h"
#include "spirv/include/Target/SPIRV/SPIRVTranslation.h"

#include "SpirVBackend.h"

using namespace mlir;
using namespace triton;

SpirVBackend::SpirVBackend(std::string target, SpirVOptions options)
    : Backend(target), m_options(options) {}

SpirVBackend::~SpirVBackend() {
  // nop
}

void SpirVBackend::loadDialects(MLIRContext &context) {
  DialectRegistry registry;
  registry.insert<mlir::spirv::SPIRVDialect>();
  context.appendDialectRegistry(registry);
}

LogicalResult SpirVBackend::makeTTIR(MLIRContext &context, ModuleOp module) {
  PassManager pm(&context);
  auto op = module.getOperation();

  addPass(pm, MlirPass::inliner);
  addPass(pm, MlirPass::ttir_rewrite_tensor_pointer);
  addPass(pm, MlirPass::ttir_rewrite_tensor_descriptor_to_pointer);
  addPass(pm, MlirPass::canonicalizer);
  addPass(pm, MlirPass::ttir_combine);
  addPass(pm, MlirPass::ttir_reorder_broadcast);
  addPass(pm, MlirPass::cse);
  addPass(pm, MlirPass::symbol_dce);
  addPass(pm, MlirPass::ttir_loop_unroll);

  return pm.run(op);
}

LogicalResult SpirVBackend::makeTTGIR(MLIRContext &context, ModuleOp module) {
  PassManager pm(&context);
  auto op = module.getOperation();

  std::string capability_str = std::string("spirv:").append(m_target);

  addPass(pm, MlirPass::ttir_convert_to_ttgpuir, capability_str,
          m_options.num_warps, m_options.threads_per_warp, m_options.num_ctas);

  addPass(pm, MlirPass::ttgpuir_coalesce);
  addPass(pm, MlirPass::ttgpuir_remove_layout_conversions);
  addPass(pm, MlirPass::ttgpuir_optimize_thread_locality);
  addPass(pm, MlirPass::ttgpuir_accelerate_matmul);
  addPass(pm, MlirPass::ttgpuir_remove_layout_conversions);
  addPass(pm, MlirPass::ttgpuir_optimize_dot_operands, /*emuTF32=*/false);
  addPass(pm, MlirPass::ttir_loop_aware_cse);
  addPass(pm, MlirPass::ttir_triton_licm);
  addPass(pm, MlirPass::ttgpuir_prefetch);
  addPass(pm, MlirPass::ttgpuir_reduce_data_duplication);
  addPass(pm, MlirPass::ttgpuir_reorder_instructions);
  addPass(pm, MlirPass::ttir_loop_aware_cse);
  addPass(pm, MlirPass::symbol_dce);
  addPass(pm, MlirPass::sccp);
  addPass(pm, MlirPass::cse);
  addPass(pm, MlirPass::canonicalizer);

  return pm.run(op);
}

LogicalResult SpirVBackend::gluonToTTGIR(MLIRContext &context,
                                         ModuleOp module) {
  PassManager pm(&context);
  auto op = module.getOperation();

  addPass(pm, MlirPass::gluon_inliner);
  addPass(pm, MlirPass::gluon_infer_coalesced_encodings);
  addPass(pm, MlirPass::gluon_resolve_auto_encodings);
  addPass(pm, MlirPass::canonicalizer);
  addPass(pm, MlirPass::sccp);
  addPass(pm, MlirPass::ttir_loop_aware_cse);
  addPass(pm, MlirPass::gluon_canonicalizer);

  return pm.run(op);
}

LogicalResult SpirVBackend::makeLLIR(MLIRContext &context, ModuleOp module) {
  PassManager pm(&context);
  auto op = module.getOperation();

  addPass(pm, MlirPass::ttgpuir_allocate_shared_memory);
  addPass(pm, MlirPass::ttgpuir_allocate_global_scratch_memory);
  addPass(pm, MlirPass::scf_to_cf);

  // TritonGPU → SPIRV dialect conversion
  pm.addPass(createConvertTritonGPUToSPIRVPass(m_options.capability));

  addPass(pm, MlirPass::canonicalizer);
  addPass(pm, MlirPass::cse);
  addPass(pm, MlirPass::symbol_dce);
  if (!m_options.disable_line_info) {
    addPass(pm, MlirPass::llvmir_di_scope);
  }

  return pm.run(op);
}

LogicalResult SpirVBackend::makeLLVMIR(MLIRContext &context, ModuleOp module) {
  // For SPIRV, the "LLVM IR" stage is not applicable. The final binary is
  // produced by generateSPIRV() which calls translateTritonGPUToSPIRVIR.
  // This method is intentionally left as a no-op so that the base class
  // applyPasses() pipeline remains compatible.
  return LogicalResult::success();
}

LogicalResult SpirVBackend::makeASM(MLIRContext &context, ModuleOp module) {
  std::string spirvIR =
      translateTritonGPUToSPIRVIR(module, m_options.capability);
  if (spirvIR.empty()) {
    llvm::errs() << "Failed to translate TritonGPU module to SPIRV IR\n";
    return LogicalResult::failure();
  }

  // Assemble the text IR into a binary SPIR-V blob and store as m_bin,
  // and keep the disassembled text in m_asm for inspection.
  m_asm = spirvIR;

  llvm::SmallVector<char> binaryBuf;
  {
    llvm::raw_svector_ostream os(binaryBuf);
    auto result = assembleSPIRV(spirvIR, os);
    if (failed(result)) {
      llvm::errs() << "Failed to assemble SPIRV binary\n";
      return LogicalResult::failure();
    }
  }
  m_bin = std::string(binaryBuf.data(), binaryBuf.size());

  return LogicalResult::success();
}

LogicalResult SpirVBackend::makeBIN(MLIRContext &context, ModuleOp module) {
  return LogicalResult::success();
}
