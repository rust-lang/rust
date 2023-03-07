//===- PreserveNVVM.h - Mark NVVM attributes for preservation.  -------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains createPreserveNVVM, a transformation pass that marks
// calls to __nv_* functions, marking them as noinline as implementing the llvm
// intrinsic.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"

namespace llvm {
class FunctionPass;
}

llvm::FunctionPass *createPreserveNVVMPass(bool Begin);

class PreserveNVVMNewPM final
    : public llvm::AnalysisInfoMixin<PreserveNVVMNewPM> {
  friend struct llvm::AnalysisInfoMixin<PreserveNVVMNewPM>;

private:
  bool Begin;
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::PreservedAnalyses;
  PreserveNVVMNewPM(bool Begin) : Begin(Begin) {}

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};
