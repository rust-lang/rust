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

#include <iostream>
#include <vector>

#include "Backend.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace triton {

/// Replace every `cf.cond_br %c, ^bb, ^bb` with `cf.br ^bb` in the module.
///
/// Triton's makeTTIR canonicalization can merge two identical successor blocks
/// into one, creating degenerate `cf.cond_br %c, ^bb, ^bb` operations.  The
/// standard MLIR canonicalization pass is expected to fold these, but in
/// practice it sometimes misses them when the degenerate form is created late
/// in the same pass run.  This explicit pass ensures the fold always happens
/// before makeTTGIR, which crashes on degenerate cond_brs.
static void foldDegenerateCondBranches(ModuleOp module) {
  std::vector<cf::CondBranchOp> toFold;
  module.walk([&](cf::CondBranchOp condBr) {
    if (condBr.getTrueDest() == condBr.getFalseDest()) {
      auto trueOps = condBr.getTrueDestOperands();
      auto falseOps = condBr.getFalseDestOperands();
      if (trueOps.size() == falseOps.size() &&
          std::equal(trueOps.begin(), trueOps.end(), falseOps.begin())) {
        toFold.push_back(condBr);
      }
    }
  });
  for (auto condBr : toFold) {
    OpBuilder builder(condBr);
    builder.create<cf::BranchOp>(condBr.getLoc(), condBr.getTrueDest(),
                                 condBr.getTrueDestOperands());
    condBr.erase();
  }
}

Backend::Backend(std::string target) : m_target(target) {
  // nop
}

Backend::~Backend() {
  // nop
}

LogicalResult Backend::applyPasses(MLIRContext &context, ModuleOp module,
                                   Language language) {
  auto m_result = LogicalResult::success();

  m_ttir.clear();
  m_ttgir.clear();
  m_llir.clear();
  m_llvmir.clear();
  m_asm.clear();

  if (language == Language::TRITON) {
    m_result = makeTTIR(context, module);
    CHECK_RESULT(m_result, "Failed to make TTIR module. Aborting translation.");

    // Triton's canonicalization inside makeTTIR can produce degenerate
    // cf.cond_br %c, ^bb, ^bb operations (both arms the same block).
    // Fold these to cf.br ^bb before makeTTGIR sees them.
    foldDegenerateCondBranches(module);
    // Clean up any dead values created by the fold (e.g., unused comparison
    // results after a degenerate cond_br is replaced with an unconditional br).
    // NOTE: SimplifyPassThroughCondBranch in the canonicalizer can collapse
    // pass-through trampolines into cond_br successors, which may recreate
    // degenerate cond_brs.  Run the fold a second time to catch those.
    {
      PassManager cleanup_pm(&context);
      cleanup_pm.addPass(createCanonicalizerPass());
      auto r = cleanup_pm.run(module.getOperation());
      (void)r; // best-effort
    }
    foldDegenerateCondBranches(module);

    llvm::raw_string_ostream ttir_os(m_ttir);
    module.print(ttir_os);
    ttir_os.flush();

    m_result = makeTTGIR(context, module);
    CHECK_RESULT(m_result,
                 "Failed to make TTGIR module. Aborting translation.");

    llvm::raw_string_ostream ttgir_os(m_ttgir);
    module.print(ttgir_os);
  } else {
    m_result = gluonToTTGIR(context, module);
    CHECK_RESULT(m_result, "Failed to convert GLUON module to TTGIR module. "
                           "Aborting translation.");

    llvm::raw_string_ostream ttgir_os(m_ttgir);
    module.print(ttgir_os);
  }

  m_result = makeLLIR(context, module);
  CHECK_RESULT(m_result, "Failed to make LLIR module. Aborting translation.");

  llvm::raw_string_ostream llir_os(m_llir);
  module.print(llir_os);

  m_result = makeLLVMIR(context, module);
  CHECK_RESULT(m_result, "Failed to make LLVMIR module. Aborting translation.");

  return LogicalResult::success();
}

void Backend::printIR(std::string stage, ModuleOp module) {
  llvm::outs() << "--------------------------------\n";
  llvm::outs() << "Stage: " << stage << "\n";
  module.print(llvm::outs());
  llvm::outs() << "\n--------------------------------\n";
}

std::optional<Error> Backend::addPass(PassManager &pm, MlirPass pass) {
  auto pass_fn = m_pass_fns.find(pass);
  if (pass_fn == m_pass_fns.end()) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid triton pass";
    return m_last_error;
  }

  pm.addPass(pass_fn->second());
  return std::nullopt;
}

std::optional<Error> Backend::addPass(PassManager &pm, MlirPass pass,
                                      int arg0) {
  m_last_error = std::nullopt;
  m_last_error_string = "";

  switch (pass) {
  case MlirPass::ttgpuir_assign_latencies:
    pm.addPass(createTritonGPUAssignLatencies({arg0}));
    break;

  case MlirPass::ttgpuir_warp_specialize:
    pm.addPass(createTritonGPUAutomaticWarpSpecialization({arg0}));
    break;

  default:
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid triton pass";
    break;
  }

  return m_last_error;
}

std::optional<Error> Backend::addPass(PassManager &pm, MlirPass pass,
                                      bool arg0) {
  if (pass != MlirPass::ttgpuir_optimize_dot_operands) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid triton pass";
    return m_last_error;
  }

  pm.addPass(createTritonGPUOptimizeDotOperands({arg0}));
  return std::nullopt;
}

std::optional<Error> Backend::addPass(PassManager &pm, MlirPass pass, int arg0,
                                      bool arg1) {
  if (pass != MlirPass::ttgpuir_pipeline) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid triton pass";
    return m_last_error;
  }

  pm.addPass(createTritonGPUPipeline({arg0, arg1}));
  return std::nullopt;
}

std::optional<Error> Backend::addPass(PassManager &pm, MlirPass pass,
                                      const std::string &arg0, int arg1,
                                      int arg2, int arg3) {
  if (pass != MlirPass::ttir_convert_to_ttgpuir) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid triton pass";
    return m_last_error;
  }

  pm.addPass(createConvertTritonToTritonGPU({arg0, arg1, arg2, arg3}));
  return std::nullopt;
}
} // namespace triton
} // namespace mlir
