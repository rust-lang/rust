//===- InstructionBatcher.h
//--------------------------------------------------===//
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
// This file contains an instruction visitor InstructionBatcher that generates
// the batches all LLVM instructions.
//
//===----------------------------------------------------------------------===//

#ifndef INSTRUCTION_BATCHER_H_
#define INSTRUCTION_BATCHER_H_

#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Value.h"

#include "llvm/ADT/SmallPtrSet.h"

#include "EnzymeLogic.h"

class InstructionBatcher final : public llvm::InstVisitor<InstructionBatcher> {
public:
  bool hasError;
  InstructionBatcher(
      llvm::Function *oldFunc, llvm::Function *newFunc, unsigned width,
      llvm::ValueMap<const llvm::Value *, std::vector<llvm::Value *>>
          &vectorizedValues,
      llvm::ValueMap<const llvm::Value *, llvm::WeakTrackingVH>
          &originalToNewFn,
      llvm::SmallPtrSetImpl<llvm::Value *> &toVectorize, EnzymeLogic &Logic);

private:
  llvm::ValueMap<const llvm::Value *, std::vector<llvm::Value *>>
      &vectorizedValues;
  llvm::ValueMap<const llvm::Value *, llvm::WeakTrackingVH> &originalToNewFn;
  llvm::SmallPtrSetImpl<llvm::Value *> &toVectorize;
  unsigned width;
  EnzymeLogic &Logic;

private:
  llvm::Value *getNewOperand(unsigned int i, llvm::Value *op);

public:
  void visitInstruction(llvm::Instruction &inst);

  void visitPHINode(llvm::PHINode &phi);

  void visitSwitchInst(llvm::SwitchInst &inst);

  void visitBranchInst(llvm::BranchInst &branch);

  void visitReturnInst(llvm::ReturnInst &ret);

  void visitCallInst(llvm::CallInst &call);
};

#endif
