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

#include "llvm/IR/InstVisitor.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/Support/Casting.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "GradientUtils.h"

using namespace llvm;

class InstructionBatcher : public llvm::InstVisitor<InstructionBatcher> {
public:
  InstructionBatcher(
      Function *oldFunc, Function *newFunc, unsigned width,
      ValueMap<const Value *, std::vector<Value *>> &vectorizedValues,
      ValueToValueMapTy &originalToNewFn, SmallPtrSetImpl<Value *> &toVectorize,
      EnzymeLogic &Logic)
      : vectorizedValues(vectorizedValues), originalToNewFn(originalToNewFn),
        toVectorize(toVectorize), width(width), Logic(Logic) {}

private:
  ValueMap<const Value *, std::vector<Value *>> &vectorizedValues;
  ValueToValueMapTy &originalToNewFn;
  SmallPtrSetImpl<Value *> &toVectorize;
  unsigned width;
  EnzymeLogic &Logic;

private:
  Value *getNewOperand(unsigned int i, llvm::Value *op) {
    if (auto meta = dyn_cast<MetadataAsValue>(op)) {
      auto md = meta->getMetadata();
      if (auto val = dyn_cast<ValueAsMetadata>(md))
        return MetadataAsValue::get(
            op->getContext(),
            ValueAsMetadata::get(getNewOperand(i, val->getValue())));
    }

    if (isa<ConstantData>(op)) {
      return op;
    } else if (isa<Function>(op)) {
      return op;
    } else if (isa<GlobalValue>(op)) {
      // TODO: !!!
    } else if (toVectorize.count(op) != 0) {
      auto found = vectorizedValues.find(op);
      assert(found != vectorizedValues.end());
      return found->second[i];
    } else {
      auto found = originalToNewFn.find(op);
      assert(found != originalToNewFn.end());
      return found->second;
    }
  }

public:
  void visitInstruction(llvm::Instruction &inst) {
    auto found = vectorizedValues.find(&inst);
    assert(found != vectorizedValues.end());
    auto placeholders = found->second;
    Instruction *placeholder = cast<Instruction>(placeholders[0]);

    for (unsigned i = 1; i < width; ++i) {
      ValueToValueMapTy vmap;
      Instruction *new_inst = placeholder->clone();
      vmap[placeholder] = new_inst;

      for (unsigned j = 0; j < inst.getNumOperands(); ++j) {
        Value *op = inst.getOperand(j);

        // Don't allow writing vectors to global memory, loading and splatting a
        // global is fine though.
        if (isa<GlobalValue>(op) && !isa<ConstantData>(op) &&
            inst.mayWriteToMemory() && toVectorize.count(op) != 0) {
          // TODO: handle buffer access
          EmitFailure("GlobalValueCannotBeVectorized", inst.getDebugLoc(),
                      &inst, "global variables have to be scalar values", inst);
          llvm_unreachable("vectorized control flow is not allowed");
        }

        if (auto meta = dyn_cast<MetadataAsValue>(op))
          if (!isa<ValueAsMetadata>(meta->getMetadata()))
            continue;

        Value *new_op = getNewOperand(i, op);
        vmap[placeholder->getOperand(j)] = new_op;
      }

      if (placeholders.size() == width) {
        // Instructions which return a value
        Instruction *placeholder = cast<Instruction>(placeholders[i]);
        assert(!placeholder->getType()->isVoidTy());

        ReplaceInstWithInst(placeholder, new_inst);
        vectorizedValues[&inst][i] = new_inst;
      } else if (placeholders.size() == 1) {
        // Instructions which don't return a value
        assert(placeholder->getType()->isVoidTy());

        Instruction *insertionPoint = placeholder->getNextNode()
                                          ? placeholder->getNextNode()
                                          : placeholder;
        IRBuilder<> Builder2(insertionPoint);
        Builder2.SetCurrentDebugLocation(DebugLoc());
        Builder2.Insert(new_inst);
        vectorizedValues[&inst].push_back(new_inst);
      } else {
        llvm_unreachable("Unexpected number of values in mapping");
      }

      RemapInstruction(new_inst, vmap, RF_NoModuleLevelChanges);

      if (!inst.getType()->isVoidTy() && inst.hasName())
        new_inst->setName(inst.getName() + Twine(i));
    }
  }

  void visitPHINode(PHINode &phi) {
    PHINode *placeholder = cast<PHINode>(vectorizedValues[&phi][0]);

    for (unsigned i = 1; i < width; ++i) {
      ValueToValueMapTy vmap;
      Instruction *new_phi = placeholder->clone();
      vmap[placeholder] = new_phi;

      for (unsigned j = 0; j < phi.getNumIncomingValues(); ++j) {
        Value *orig_block = phi.getIncomingBlock(j);
        BasicBlock *new_block = cast<BasicBlock>(originalToNewFn[orig_block]);
        Value *orig_val = phi.getIncomingValue(j);
        Value *new_val = getNewOperand(i, orig_val);

        vmap[placeholder->getIncomingValue(j)] = new_val;
        vmap[new_block] = new_block;
      }

      RemapInstruction(new_phi, vmap, RF_NoModuleLevelChanges);
      Instruction *placeholder = cast<Instruction>(vectorizedValues[&phi][i]);
      ReplaceInstWithInst(placeholder, new_phi);
      new_phi->setName(phi.getName());
      vectorizedValues[&phi][i] = new_phi;
    }
  }

  void visitSwitchInst(llvm::SwitchInst &inst) {
    // TODO: runtime check
    EmitFailure("SwitchConditionCannotBeVectorized", inst.getDebugLoc(), &inst,
                "switch conditions have to be scalar values", inst);
    llvm_unreachable("vectorized control flow is not allowed");
  }

  void visitBranchInst(llvm::BranchInst &branch) {
    // TODO: runtime check
    EmitFailure("BranchConditionCannotBeVectorized", branch.getDebugLoc(),
                &branch, "branch conditions have to be scalar values", branch);
    llvm_unreachable("vectorized control flow is not allowed");
  }

  void visitReturnInst(llvm::ReturnInst &ret) {
    auto found = originalToNewFn.find(ret.getParent());
    assert(found != originalToNewFn.end());
    BasicBlock *nBB = dyn_cast<BasicBlock>(&*found->second);
    IRBuilder<> Builder2 = IRBuilder<>(nBB);
    Builder2.SetCurrentDebugLocation(DebugLoc());
    ReturnInst *placeholder = cast<ReturnInst>(nBB->getTerminator());
    SmallVector<Value *, 4> rets;

    for (unsigned j = 0; j < ret.getNumOperands(); ++j) {
      Value *op = ret.getOperand(j);
      for (unsigned i = 0; i < width; ++i) {
        Value *new_op = getNewOperand(i, op);
        rets.push_back(new_op);
      }
    }

    if (ret.getNumOperands() != 0) {
      auto ret = Builder2.CreateAggregateRet(rets.data(), width);
      ret->setDebugLoc(placeholder->getDebugLoc());
      placeholder->eraseFromParent();
    }
  }

  void visitCallInst(llvm::CallInst &call) {
    auto found = vectorizedValues.find(&call);
    assert(found != vectorizedValues.end());
    auto placeholders = found->second;
    Instruction *placeholder = cast<Instruction>(placeholders[0]);
    IRBuilder<> Builder2(placeholder);
    Builder2.SetCurrentDebugLocation(DebugLoc());
    Function *orig_func = getFunctionFromCall(&call);

    bool isDefined = !orig_func->isDeclaration();

    if (!isDefined)
      return visitInstruction(call);

    SmallVector<Value *, 4> args;
    SmallVector<BATCH_TYPE, 4> arg_types;
#if LLVM_VERSION_MAJOR >= 14
    for (unsigned j = 0; j < call.arg_size(); ++j) {
#else
    for (unsigned j = 0; j < call.getNumArgOperands(); ++j) {
#endif
      Value *op = call.getArgOperand(j);

      if (toVectorize.count(op) != 0) {
        Type *aggTy = GradientUtils::getShadowType(op->getType(), width);
        Value *agg = UndefValue::get(aggTy);
        for (unsigned i = 0; i < width; i++) {
          auto found = vectorizedValues.find(op);
          assert(found != vectorizedValues.end());
          Value *new_op = found->second[i];
          Builder2.CreateInsertValue(agg, new_op, {i});
        }
        args.push_back(agg);
        arg_types.push_back(BATCH_TYPE::VECTOR);
      } else if (isa<ConstantData>(op)) {
        args.push_back(op);
        arg_types.push_back(BATCH_TYPE::SCALAR);
      } else {
        auto found = originalToNewFn.find(op);
        assert(found != originalToNewFn.end());
        Value *arg = found->second;
        args.push_back(arg);
        arg_types.push_back(BATCH_TYPE::SCALAR);
      }
    }

    BATCH_TYPE ret_type = orig_func->getReturnType()->isVoidTy()
                              ? BATCH_TYPE::SCALAR
                              : BATCH_TYPE::VECTOR;

    Function *new_func =
        Logic.CreateBatch(orig_func, width, arg_types, ret_type);
    CallInst *new_call = Builder2.CreateCall(new_func->getFunctionType(),
                                             new_func, args, call.getName());

    new_call->setDebugLoc(placeholder->getDebugLoc());

    if (!call.getType()->isVoidTy()) {
      for (unsigned i = 0; i < width; ++i) {
        Instruction *placeholder = dyn_cast<Instruction>(placeholders[i]);
        ExtractValueInst *ret = ExtractValueInst::Create(
            new_call, {i},
            "unwrap" + (call.hasName() ? "." + call.getName() + Twine(i) : ""));
        ReplaceInstWithInst(placeholder, ret);
        vectorizedValues[&call][i] = ret;
      }
    } else {
      placeholder->replaceAllUsesWith(new_call);
      placeholder->eraseFromParent();
    }
  }
};
