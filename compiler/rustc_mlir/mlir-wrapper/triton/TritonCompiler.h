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

#ifndef TRITON_COMPILER_H
#define TRITON_COMPILER_H

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include "../Compiler.h"
#include "backend/Backend.h"

namespace mlir {
namespace triton {

class TritonCompiler : public Compiler {
public:
  TritonCompiler(MLIRContext *context, std::string target, std::string options);

  virtual ~TritonCompiler() override;

  virtual LogicalResult compile(ModuleOp mlir_module) override;

  /// Return the output string from the last successful compile. The pointer
  /// remains valid until the next successful compile or compiler destruction.
  const char *getLLIR() const;
  const char *getTTIR() const;
  const char *getTTGIR() const;
  const char *getLLVMIR() const;
  const char *getASM() const;
  const char *getBIN() const;

private:
  LogicalResult applyTritonPasses(ModuleOp mlir_module);

  Backend *backend;
};

} // namespace triton
} // namespace mlir

#endif /* TRITON_COMPILER_H */
