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

#ifndef COMPILER_H
#define COMPILER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace triton {

class Compiler {
public:
  Compiler(MLIRContext *context, std::string target, std::string options)
      : context(context), target(target), options(options) {};

  virtual ~Compiler() = default;

  virtual LogicalResult compile(ModuleOp mlir_module) = 0;

protected:
  std::string target;
  std::string options;
  MLIRContext *context;
};

} // namespace triton
} // namespace mlir

#endif /* COMPILER_H */
