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

#ifndef TRITON_SPIRV_BACKEND_H
#define TRITON_SPIRV_BACKEND_H

#include <any>
#include <map>
#include <optional>
#include <string>

#include "Backend.h"

namespace mlir {
namespace triton {

struct SpirVOptions {
  int num_warps = 4;
  int num_ctas = 1;
  int num_stages = 3;
  int threads_per_warp = 32;
  std::map<std::string, std::string> extern_libs = {};
  bool debug = false;
  std::string backend_name = "spirv";
  bool disable_line_info = false;
  // filename of a user-defined IR (*.{ttir|ttgir|llir})
  std::optional<std::string> ir_override = std::nullopt;
  std::map<std::string, std::any> capability = {};
};

class SpirVBackend : public Backend {
public:
  SpirVBackend(std::string target, SpirVOptions options);

  virtual ~SpirVBackend();

  virtual void loadDialects(MLIRContext &context) override;

  virtual LogicalResult makeTTIR(MLIRContext &context,
                                 ModuleOp module) override;

  virtual LogicalResult makeTTGIR(MLIRContext &context,
                                  ModuleOp module) override;

  virtual LogicalResult gluonToTTGIR(MLIRContext &context,
                                     ModuleOp module) override;

  virtual LogicalResult makeLLIR(MLIRContext &context,
                                 ModuleOp module) override;

  virtual LogicalResult makeLLVMIR(MLIRContext &context,
                                   ModuleOp module) override;

  virtual LogicalResult makeASM(MLIRContext &context, ModuleOp module) override;

  virtual LogicalResult makeBIN(MLIRContext &context, ModuleOp module) override;

private:
  SpirVOptions m_options;
};

} // namespace triton
} // namespace mlir

#endif /*! TRITON_SPIRV_BACKEND_H */
