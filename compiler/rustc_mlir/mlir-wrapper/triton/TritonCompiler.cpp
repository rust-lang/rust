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

#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/MLIRContext.h"

#include "TritonCompiler.h"
#include "backend/CudaBackend.h"
#include "backend/SpirVBackend.h"

using namespace std;

namespace mlir {
namespace triton {

TritonCompiler::TritonCompiler(MLIRContext *context, std::string target,
                               CompileOptions options)
    : Compiler(context, target, options) {
  switch (options.backend) {
  case TargetBackend_Spirv:
    backend = new SpirVBackend(target, SpirVOptions());
    break;
  case TargetBackend_Cuda:
  default:
    backend = new CudaBackend(target, options.data.cuda);
    break;
  }
  backend->loadDialects(*context);
}

TritonCompiler::~TritonCompiler() { delete backend; }

LogicalResult TritonCompiler::compile(ModuleOp mlir_module) {
  auto result = applyTritonPasses(mlir_module);
  if (failed(result)) {
    llvm::errs() << "Failed to apply Triton passes. Aborting translation.\n";
    return result;
  }

  // Generate ASM and BIN from the backend
  auto asmResult = backend->makeASM(*context, mlir_module);
  if (failed(asmResult)) {
    llvm::errs() << "Failed to generate ASM from backend.\n";
    return asmResult;
  }

  auto binResult = backend->makeBIN(*context, mlir_module);
  if (failed(binResult)) {
    llvm::errs() << "Failed to generate BIN from backend.\n";
    return binResult;
  }

  return LogicalResult::success();
}

const char *TritonCompiler::getLLIR() const { return backend->getLLIR(); }
const char *TritonCompiler::getTTIR() const { return backend->getTTIR(); }
const char *TritonCompiler::getTTGIR() const { return backend->getTTGIR(); }
const char *TritonCompiler::getLLVMIR() const { return backend->getLLVMIR(); }
const char *TritonCompiler::getASM() const { return backend->getASM(); }
const char *TritonCompiler::getBIN() const { return backend->getBIN(); }

LogicalResult TritonCompiler::applyTritonPasses(ModuleOp mlir_module) {
  auto result = backend->applyPasses(*context, mlir_module, Language::TRITON);
  if (failed(result)) {
    llvm::errs() << "Failed to apply Triton passes. Aborting translation.\n";
  }

  return result;
}

} // namespace triton
} // namespace mlir
