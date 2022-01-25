//===- GradientUtils.cpp - Helper class and utilities for AD     ---------===//
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
// This file define two helper classes GradientUtils and subclass
// DiffeGradientUtils. These classes contain utilities for managing the cache,
// recomputing statements, and in the case of DiffeGradientUtils, managing
// adjoint values and shadow pointers.
//
//===----------------------------------------------------------------------===//

#include <algorithm>

#include <llvm/Config/llvm-config.h>

#include "DifferentialUseAnalysis.h"
#include "EnzymeLogic.h"
#include "FunctionUtils.h"
#include "GradientUtils.h"
#include "LibraryFuncs.h"

#include "llvm/IR/GlobalValue.h"

#include "llvm/IR/Constants.h"

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Support/AMDGPUMetadata.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"

std::map<std::string, std::function<llvm::Value *(IRBuilder<> &, CallInst *,
                                                  ArrayRef<Value *>)>>
    shadowHandlers;
std::map<std::string,
         std::function<llvm::CallInst *(IRBuilder<> &, Value *, Function *)>>
    shadowErasers;

std::map<
    std::string,
    std::pair<std::function<void(IRBuilder<> &, CallInst *, GradientUtils &,
                                 Value *&, Value *&, Value *&)>,
              std::function<void(IRBuilder<> &, CallInst *,
                                 DiffeGradientUtils &, Value *)>>>
    customCallHandlers;

std::map<std::string, std::function<void(IRBuilder<> &, CallInst *,
                                         GradientUtils &, Value *&, Value *&)>>
    customFwdCallHandlers;

extern "C" {
llvm::cl::opt<bool>
    EnzymeNewCache("enzyme-new-cache", cl::init(true), cl::Hidden,
                   cl::desc("Use new cache decision algorithm"));

llvm::cl::opt<bool> EnzymeMinCutCache("enzyme-mincut-cache", cl::init(true),
                                      cl::Hidden,
                                      cl::desc("Use Enzyme Mincut algorithm"));

llvm::cl::opt<bool> EnzymeLoopInvariantCache(
    "enzyme-loop-invariant-cache", cl::init(true), cl::Hidden,
    cl::desc("Attempt to hoist cache outside of loop"));

llvm::cl::opt<bool> EnzymeInactiveDynamic(
    "enzyme-inactive-dynamic", cl::init(true), cl::Hidden,
    cl::desc("Force wholy inactive dynamic loops to have 0 iter reverse pass"));

llvm::cl::opt<bool>
    EnzymeSharedForward("enzyme-shared-forward", cl::init(false), cl::Hidden,
                        cl::desc("Forward Shared Memory from definitions"));

llvm::cl::opt<bool>
    EnzymeRegisterReduce("enzyme-register-reduce", cl::init(false), cl::Hidden,
                         cl::desc("Reduce the amount of register reduce"));
llvm::cl::opt<bool>
    EnzymeSpeculatePHIs("enzyme-speculate-phis", cl::init(false), cl::Hidden,
                        cl::desc("Speculatively execute phi computations"));
llvm::cl::opt<bool> EnzymeFreeInternalAllocations(
    "enzyme-free-internal-allocations", cl::init(true), cl::Hidden,
    cl::desc("Always free internal allocations (disable if allocation needs "
             "access outside)"));
}

Value *GradientUtils::unwrapM(Value *const val, IRBuilder<> &BuilderM,
                              const ValueToValueMapTy &available,
                              UnwrapMode unwrapMode, BasicBlock *scope,
                              bool permitCache) {
  assert(val);
  assert(val->getName() != "<badref>");
  assert(val->getType());

  for (auto pair : available) {
    assert(pair.first);
    assert(pair.second);
    assert(pair.first->getType());
    assert(pair.second->getType());
    assert(pair.first->getType() == pair.second->getType());
  }

  if (isa<LoadInst>(val) &&
      cast<LoadInst>(val)->getMetadata("enzyme_mustcache")) {
    return val;
  }

  if (available.count(val)) {
    auto avail = available.lookup(val);
    assert(avail->getType());
    if (avail->getType() != val->getType()) {
      llvm::errs() << "val: " << *val << "\n";
      llvm::errs() << "available[val]: " << *available.lookup(val) << "\n";
    }
    assert(available.lookup(val)->getType() == val->getType());
    return available.lookup(val);
  }

  if (auto inst = dyn_cast<Instruction>(val)) {
    // if (inst->getParent() == &newFunc->getEntryBlock()) {
    //  return inst;
    //}
    if (inst->getParent()->getParent() == newFunc &&
        isOriginalBlock(*BuilderM.GetInsertBlock())) {
      if (BuilderM.GetInsertBlock()->size() &&
          BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
        if (DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
          // llvm::errs() << "allowed " << *inst << "from domination\n";
          assert(inst->getType() == val->getType());
          return inst;
        }
      } else {
        if (DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
          // llvm::errs() << "allowed " << *inst << "from block domination\n";
          assert(inst->getType() == val->getType());
          return inst;
        }
      }
    }
  }

  std::pair<Value *, BasicBlock *> idx = std::make_pair(val, scope);
  // assert(!val->getName().startswith("$tapeload"));
  if (permitCache) {
    auto found0 = unwrap_cache.find(BuilderM.GetInsertBlock());
    if (found0 != unwrap_cache.end()) {
      auto found1 = found0->second.find(idx.first);
      if (found1 != found0->second.end()) {
        auto found2 = found1->second.find(idx.second);
        if (found2 != found1->second.end()) {

          auto cachedValue = found2->second;
          if (cachedValue == nullptr) {
            found1->second.erase(idx.second);
            if (found1->second.size() == 0) {
              found0->second.erase(idx.first);
            }
          } else {
            if (cachedValue->getType() != val->getType()) {
              llvm::errs() << "newFunc: " << *newFunc << "\n";
              llvm::errs() << "val: " << *val << "\n";
              llvm::errs() << "unwrap_cache[cidx]: " << *cachedValue << "\n";
            }
            assert(cachedValue->getType() == val->getType());
            return cachedValue;
          }
        }
      }
    }
  }

  if (this->mode == DerivativeMode::ReverseModeGradient)
    if (auto inst = dyn_cast<Instruction>(val)) {
      if (unwrapMode == UnwrapMode::LegalFullUnwrap) {
        // TODO this isOriginal is a bottleneck, the new mapping of
        // knownRecompute should be precomputed and maintained to lookup instead
        Instruction *orig = isOriginal(inst);
        // If a given value has been chosen to be cached, do not compute the
        // operands to unwrap it, instead simply emit a placeholder to be
        // replaced by the cache load later. This placeholder should only be
        // returned when the original value would be recomputed (e.g. this
        // function would not return null). Since this case assumes everything
        // can be recomputed, simply return the placeholder.
        if (orig && knownRecomputeHeuristic.find(orig) !=
                        knownRecomputeHeuristic.end()) {
          if (!knownRecomputeHeuristic[orig]) {
            assert(inst->getParent()->getParent() == newFunc);
            auto placeholder = BuilderM.CreatePHI(
                val->getType(), 0, val->getName() + "_krcLFUreplacement");
            assert(permitCache);
            unwrappedLoads[placeholder] = inst;
            return unwrap_cache[BuilderM.GetInsertBlock()][idx.first]
                               [idx.second] = placeholder;
          }
        }
      } else if (unwrapMode == UnwrapMode::AttemptFullUnwrapWithLookup) {
        // TODO this isOriginal is a bottleneck, the new mapping of
        // knownRecompute should be precomputed and maintained to lookup instead
        Instruction *orig = isOriginal(inst);
        // If a given value has been chosen to be cached, do not compute the
        // operands to unwrap it, instead simply emit a placeholder to be
        // replaced by the cache load later. This placeholder should only be
        // returned when the original value would be recomputed (e.g. this
        // function would not return null). See note below about the condition
        // as applied to this case.
        if (orig && knownRecomputeHeuristic.find(orig) !=
                        knownRecomputeHeuristic.end()) {
          if (!knownRecomputeHeuristic[orig]) {
            // Note that this logic (original load must dominate or
            // alternatively be in the reverse block) is only valid iff when
            // applicable (here if in split mode), an uncacheable load cannot be
            // hoisted outside of a loop to be used as a loop limit. This
            // optimization is currently done in the combined mode (e.g. if a
            // load isn't modified between a prior insertion point and the
            // actual load, it is legal to recompute).
            if (!isOriginalBlock(*BuilderM.GetInsertBlock()) ||
                DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
              assert(inst->getParent()->getParent() == newFunc);
              auto placeholder = BuilderM.CreatePHI(
                  val->getType(), 0, val->getName() + "_krcAFUWLreplacement");
              assert(permitCache);
              unwrappedLoads[placeholder] = inst;
              return unwrap_cache[BuilderM.GetInsertBlock()][idx.first]
                                 [idx.second] = placeholder;
            }
          }
        }
      } else if (unwrapMode != UnwrapMode::LegalFullUnwrapNoTapeReplace) {
        // TODO this isOriginal is a bottleneck, the new mapping of
        // knownRecompute should be precomputed and maintained to lookup instead

        // If a given value has been chosen to be cached, do not compute the
        // operands to unwrap it if it is not legal to do so. This prevents the
        // creation of unused versions of the instruction's operand, which may
        // be assumed to never be used and thus cause an error when they are
        // inadvertantly cached.
        Value *orig = isOriginal(val);
        if (orig && knownRecomputeHeuristic.find(orig) !=
                        knownRecomputeHeuristic.end()) {
          if (!knownRecomputeHeuristic[orig]) {
            if (!legalRecompute(orig, available, &BuilderM))
              return nullptr;

            assert(isa<LoadInst>(orig) == isa<LoadInst>(val));
          }
        }
      }
    }

#define getOpFullest(Builder, vtmp, frominst, check)                           \
  ({                                                                           \
    Value *v = vtmp;                                                           \
    BasicBlock *origParent = frominst;                                         \
    Value *___res;                                                             \
    if (unwrapMode == UnwrapMode::LegalFullUnwrap ||                           \
        unwrapMode == UnwrapMode::LegalFullUnwrapNoTapeReplace ||              \
        unwrapMode == UnwrapMode::AttemptFullUnwrap ||                         \
        unwrapMode == UnwrapMode::AttemptFullUnwrapWithLookup) {               \
      if (v == val)                                                            \
        ___res = nullptr;                                                      \
      else                                                                     \
        ___res = unwrapM(v, Builder, available, unwrapMode, origParent,        \
                         permitCache);                                         \
      if (!___res && unwrapMode == UnwrapMode::AttemptFullUnwrapWithLookup) {  \
        bool noLookup = false;                                                 \
        if (auto opinst = dyn_cast<Instruction>(v))                            \
          if (isOriginalBlock(*Builder.GetInsertBlock())) {                    \
            if (!DT.dominates(opinst, &*Builder.GetInsertPoint()))             \
              noLookup = true;                                                 \
          }                                                                    \
        if (origParent)                                                        \
          if (auto opinst = dyn_cast<Instruction>(v)) {                        \
            v = fixLCSSA(opinst, origParent);                                  \
          }                                                                    \
        if (!noLookup)                                                         \
          ___res = lookupM(v, Builder, available, v != val);                   \
      }                                                                        \
      if (___res)                                                              \
        assert(___res->getType() == v->getType() && "uw");                     \
    } else {                                                                   \
      if (origParent)                                                          \
        if (auto opinst = dyn_cast<Instruction>(v)) {                          \
          v = fixLCSSA(opinst, origParent);                                    \
        }                                                                      \
      assert(unwrapMode == UnwrapMode::AttemptSingleUnwrap);                   \
      ___res = lookupM(v, Builder, available, v != val);                       \
      if (___res && ___res->getType() != v->getType()) {                       \
        llvm::errs() << *newFunc << "\n";                                      \
        llvm::errs() << " v = " << *v << " res = " << *___res << "\n";         \
      }                                                                        \
      if (___res)                                                              \
        assert(___res->getType() == v->getType() && "lu");                     \
    }                                                                          \
    ___res;                                                                    \
  })
#define getOpFull(Builder, vtmp, frominst)                                     \
  getOpFullest(Builder, vtmp, frominst, true)
#define getOpUnchecked(vtmp)                                                   \
  ({                                                                           \
    BasicBlock *parent = scope;                                                \
    getOpFullest(BuilderM, vtmp, parent, false);                               \
  })
#define getOp(vtmp)                                                            \
  ({                                                                           \
    BasicBlock *parent = scope;                                                \
    if (parent == nullptr)                                                     \
      if (auto originst = dyn_cast<Instruction>(val))                          \
        parent = originst->getParent();                                        \
    getOpFullest(BuilderM, vtmp, parent, true);                                \
  })

  if (isa<Argument>(val) || isa<Constant>(val)) {
    return val;
  } else if (isa<AllocaInst>(val)) {
    return val;
#if LLVM_VERSION_MAJOR >= 10
  } else if (auto op = dyn_cast<FreezeInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateFreeze(op0, op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
    }
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
#endif
  } else if (auto op = dyn_cast<CastInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateCast(op->getOpcode(), op0, op->getDestTy(),
                                        op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<ExtractValueInst>(val)) {
    auto op0 = getOp(op->getAggregateOperand());
    if (op0 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateExtractValue(op0, op->getIndices(),
                                                op->getName() + "_unwrap");
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<InsertValueInst>(val)) {
    auto op0 = getOp(op->getAggregateOperand());
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getInsertedValueOperand());
    if (op1 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateInsertValue(op0, op1, op->getIndices(),
                                               op->getName() + "_unwrap");
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<ExtractElementInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
    auto toreturn =
        BuilderM.CreateExtractElement(op0, op1, op->getName() + "_unwrap");
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<InsertElementInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
    auto op2 = getOp(op->getOperand(2));
    if (op2 == nullptr)
      goto endCheck;
    auto toreturn =
        BuilderM.CreateInsertElement(op0, op1, op2, op->getName() + "_unwrap");
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<ShuffleVectorInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
#if LLVM_VERSION_MAJOR >= 11
    auto toreturn = BuilderM.CreateShuffleVector(
        op0, op1, op->getShuffleMaskForBitcode(), op->getName() + "'_unwrap");
#else
    auto toreturn = BuilderM.CreateShuffleVector(op0, op1, op->getOperand(2),
                                                 op->getName() + "'_unwrap");
#endif
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<BinaryOperator>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
    if (op0->getType() != op1->getType()) {
      llvm::errs() << " op: " << *op << " op0: " << *op0 << " op1: " << *op1
                   << " p0: " << *op->getOperand(0)
                   << "  p1: " << *op->getOperand(1) << "\n";
    }
    assert(op0->getType() == op1->getType());
    auto toreturn = BuilderM.CreateBinOp(op->getOpcode(), op0, op1,
                                         op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<ICmpInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateICmp(op->getPredicate(), op0, op1,
                                        op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<FCmpInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateFCmp(op->getPredicate(), op0, op1,
                                        op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
#if LLVM_VERSION_MAJOR >= 9
  } else if (isa<FPMathOperator>(val) &&
             cast<FPMathOperator>(val)->getOpcode() == Instruction::FNeg) {
    auto op = cast<FPMathOperator>(val);
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateFNeg(op0, op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() !=
          cast<Instruction>(val)->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
#endif
  } else if (auto op = dyn_cast<SelectInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
    auto op2 = getOp(op->getOperand(2));
    if (op2 == nullptr)
      goto endCheck;
    auto toreturn =
        BuilderM.CreateSelect(op0, op1, op2, op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto inst = dyn_cast<GetElementPtrInst>(val)) {
    auto ptr = getOp(inst->getPointerOperand());
    if (ptr == nullptr)
      goto endCheck;
    SmallVector<Value *, 4> ind;
    // llvm::errs() << "inst: " << *inst << "\n";
    for (unsigned i = 0; i < inst->getNumIndices(); ++i) {
      Value *a = inst->getOperand(1 + i);
      auto op = getOp(a);
      if (op == nullptr)
        goto endCheck;
      ind.push_back(op);
    }
#if LLVM_VERSION_MAJOR > 7
    auto toreturn = BuilderM.CreateGEP(
        cast<PointerType>(inst->getPointerOperandType())->getElementType(), ptr,
        ind, inst->getName() + "_unwrap");
#else
    auto toreturn = BuilderM.CreateGEP(ptr, ind, inst->getName() + "_unwrap");
#endif
    if (isa<GetElementPtrInst>(toreturn))
      cast<GetElementPtrInst>(toreturn)->setIsInBounds(inst->isInBounds());
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(inst);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != inst->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto load = dyn_cast<LoadInst>(val)) {
    if (load->getMetadata("enzyme_noneedunwrap"))
      return load;

    bool legalMove = unwrapMode == UnwrapMode::LegalFullUnwrap ||
                     unwrapMode == UnwrapMode::LegalFullUnwrapNoTapeReplace;
    if (!legalMove) {
      BasicBlock *parent = nullptr;
      if (isOriginalBlock(*BuilderM.GetInsertBlock()))
        parent = BuilderM.GetInsertBlock();
      if (!parent ||
          LI.getLoopFor(parent) == LI.getLoopFor(load->getParent()) ||
          DT.dominates(load, parent)) {
        legalMove = legalRecompute(load, available, &BuilderM);
      } else {
        legalMove =
            legalRecompute(load, available, &BuilderM, /*reverse*/ false,
                           /*legalRecomputeCache*/ false);
      }
    }
    if (!legalMove) {
      auto &warnMap = UnwrappedWarnings[load];
      if (!warnMap.count(BuilderM.GetInsertBlock())) {
        EmitWarning("UncacheableUnwrap", load->getDebugLoc(),
                    load->getParent()->getParent(), load->getParent(),
                    "Load cannot be unwrapped ", *load, " in ",
                    BuilderM.GetInsertBlock()->getName(), " - ",
                    BuilderM.GetInsertBlock()->getParent()->getName(), " mode ",
                    unwrapMode);
        warnMap.insert(BuilderM.GetInsertBlock());
      }
      goto endCheck;
    }

    Value *pidx = getOp(load->getOperand(0));

    if (pidx == nullptr) {
      goto endCheck;
    }

    if (pidx->getType() != load->getOperand(0)->getType()) {
      llvm::errs() << "load: " << *load << "\n";
      llvm::errs() << "load->getOperand(0): " << *load->getOperand(0) << "\n";
      llvm::errs() << "idx: " << *pidx << " unwrapping: " << *val
                   << " mode=" << unwrapMode << "\n";
    }
    assert(pidx->getType() == load->getOperand(0)->getType());

#if LLVM_VERSION_MAJOR > 7
    auto toreturn = BuilderM.CreateLoad(
        cast<PointerType>(pidx->getType())->getElementType(), pidx,
        load->getName() + "_unwrap");
#else
    auto toreturn = BuilderM.CreateLoad(pidx, load->getName() + "_unwrap");
#endif
    toreturn->copyIRFlags(load);
    unwrappedLoads[toreturn] = load;
    if (toreturn->getParent()->getParent() != load->getParent()->getParent())
      toreturn->setDebugLoc(nullptr);
    else
      toreturn->setDebugLoc(getNewFromOriginal(load->getDebugLoc()));
#if LLVM_VERSION_MAJOR >= 10
    toreturn->setAlignment(load->getAlign());
#else
    toreturn->setAlignment(load->getAlignment());
#endif
    toreturn->setVolatile(load->isVolatile());
    toreturn->setOrdering(load->getOrdering());
    toreturn->setSyncScopeID(load->getSyncScopeID());
    if (toreturn->getParent()->getParent() != load->getParent()->getParent())
      toreturn->setDebugLoc(nullptr);
    else
      toreturn->setDebugLoc(getNewFromOriginal(load->getDebugLoc()));
    toreturn->setMetadata(LLVMContext::MD_tbaa,
                          load->getMetadata(LLVMContext::MD_tbaa));
    toreturn->setMetadata(LLVMContext::MD_invariant_group,
                          load->getMetadata(LLVMContext::MD_invariant_group));
    // TODO adding to cache only legal if no alias of any future writes
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<CallInst>(val)) {

    bool legalMove = unwrapMode == UnwrapMode::LegalFullUnwrap ||
                     unwrapMode == UnwrapMode::LegalFullUnwrapNoTapeReplace;
    if (!legalMove) {
      legalMove = legalRecompute(op, available, &BuilderM);
    }
    if (!legalMove)
      goto endCheck;

    std::vector<Value *> args;
#if LLVM_VERSION_MAJOR >= 14
    for (unsigned i = 0; i < op->arg_size(); ++i)
#else
    for (unsigned i = 0; i < op->getNumArgOperands(); ++i)
#endif
    {
      args.emplace_back(getOp(op->getArgOperand(i)));
      if (args[i] == nullptr)
        goto endCheck;
    }

#if LLVM_VERSION_MAJOR >= 11
    Value *fn = getOp(op->getCalledOperand());
#else
    Value *fn = getOp(op->getCalledValue());
#endif
    if (fn == nullptr)
      goto endCheck;

    auto toreturn =
        cast<CallInst>(BuilderM.CreateCall(op->getFunctionType(), fn, args));
    toreturn->copyIRFlags(op);
    toreturn->setAttributes(op->getAttributes());
    toreturn->setCallingConv(op->getCallingConv());
    toreturn->setTailCallKind(op->getTailCallKind());
    if (toreturn->getParent()->getParent() == op->getParent()->getParent())
      toreturn->setDebugLoc(getNewFromOriginal(op->getDebugLoc()));
    else
      toreturn->setDebugLoc(nullptr);
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    unwrappedLoads[toreturn] = val;
    return toreturn;
  } else if (auto phi = dyn_cast<PHINode>(val)) {
    if (phi->getNumIncomingValues() == 0) {
      // This is a placeholder shadow for a load, rather than falling
      // back to the uncached variant, use the proper procedure for
      // an inverted load
      if (auto dli = dyn_cast_or_null<LoadInst>(hasUninverted(phi))) {
        // Almost identical code to unwrap load (replacing use of shadow
        // where appropriate)
        if (dli->getMetadata("enzyme_noneedunwrap"))
          return dli;

        bool legalMove = unwrapMode == UnwrapMode::LegalFullUnwrap ||
                         unwrapMode == UnwrapMode::LegalFullUnwrapNoTapeReplace;
        if (!legalMove) {
          // TODO actually consider whether this is legal to move to the new
          // location, rather than recomputable anywhere
          legalMove = legalRecompute(dli, available, &BuilderM);
        }
        if (!legalMove) {
          auto &warnMap = UnwrappedWarnings[phi];
          if (!warnMap.count(BuilderM.GetInsertBlock())) {
            EmitWarning("UncacheableUnwrap", dli->getDebugLoc(),
                        dli->getParent()->getParent(), dli->getParent(),
                        "Differential Load cannot be unwrapped ", *dli, " in ",
                        BuilderM.GetInsertBlock()->getName(), " mode ",
                        unwrapMode);
            warnMap.insert(BuilderM.GetInsertBlock());
          }
          return nullptr;
        }

        Value *pidx = nullptr;

        if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
          pidx = invertPointerM(dli->getOperand(0), BuilderM);
        } else {
          pidx =
              lookupM(invertPointerM(dli->getOperand(0), BuilderM), BuilderM);
        }

        if (pidx == nullptr)
          goto endCheck;

        if (pidx->getType() != dli->getOperand(0)->getType()) {
          llvm::errs() << "dli: " << *dli << "\n";
          llvm::errs() << "dli->getOperand(0): " << *dli->getOperand(0) << "\n";
          llvm::errs() << "pidx: " << *pidx << "\n";
        }
        assert(pidx->getType() == dli->getOperand(0)->getType());
#if LLVM_VERSION_MAJOR > 7
        auto toreturn = BuilderM.CreateLoad(
            cast<PointerType>(pidx->getType())->getElementType(), pidx,
            phi->getName() + "_unwrap");
#else
        auto toreturn = BuilderM.CreateLoad(pidx, phi->getName() + "_unwrap");
#endif
        if (auto newi = dyn_cast<Instruction>(toreturn)) {
          newi->copyIRFlags(dli);
          unwrappedLoads[toreturn] = dli;
        }
#if LLVM_VERSION_MAJOR >= 10
        toreturn->setAlignment(dli->getAlign());
#else
        toreturn->setAlignment(dli->getAlignment());
#endif
        toreturn->setVolatile(dli->isVolatile());
        toreturn->setOrdering(dli->getOrdering());
        toreturn->setSyncScopeID(dli->getSyncScopeID());
        toreturn->setDebugLoc(getNewFromOriginal(dli->getDebugLoc()));
        toreturn->setMetadata(LLVMContext::MD_tbaa,
                              dli->getMetadata(LLVMContext::MD_tbaa));
        toreturn->setMetadata(
            LLVMContext::MD_invariant_group,
            dli->getMetadata(LLVMContext::MD_invariant_group));
        // TODO adding to cache only legal if no alias of any future writes
        if (permitCache)
          unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] =
              toreturn;
        assert(val->getType() == toreturn->getType());
        return toreturn;
      }
      goto endCheck;
    }
    assert(phi->getNumIncomingValues() != 0);

    // If requesting loop bound and are requesting the total size.
    // Rather than generating a new lcssa variable, use the existing loop exact
    // bound var
    BasicBlock *ivctx = scope;
    if (!ivctx)
      ivctx = BuilderM.GetInsertBlock();
    if (newFunc == ivctx->getParent() && !isOriginalBlock(*ivctx)) {
      ivctx = originalForReverseBlock(*ivctx);
    }
    if ((ivctx == phi->getParent() || DT.dominates(phi, ivctx)) &&
        (!isOriginalBlock(*BuilderM.GetInsertBlock()) ||
         DT.dominates(phi, &*BuilderM.GetInsertPoint()))) {
      LoopContext lc;
      bool loopVar = false;
      if (getContext(phi->getParent(), lc) && lc.var == phi) {
        loopVar = true;
      } else {
        Value *V = nullptr;
        bool legal = true;
        for (auto &val : phi->incoming_values()) {
          if (isa<UndefValue>(val))
            continue;
          if (V == nullptr)
            V = val;
          else if (V != val) {
            legal = false;
            break;
          }
        }
        if (legal) {
          if (auto I = dyn_cast_or_null<PHINode>(V)) {
            if (getContext(I->getParent(), lc) && lc.var == I) {
              loopVar = true;
            }
          }
        }
      }
      if (loopVar) {
        if (!lc.dynamic) {
          Value *lim = getOp(lc.trueLimit);
          if (lim) {
            unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] =
                lim;
            return lim;
          }
        } else if (unwrapMode == UnwrapMode::AttemptFullUnwrapWithLookup &&
                   reverseBlocks.size() > 0) {
          // Must be in a reverse pass fashion for a lookup to index bound to be
          // legal
          assert(/*ReverseLimit*/ reverseBlocks.size() > 0);
          LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                            lc.preheader);
          Value *lim = lookupValueFromCache(
              /*forwardPass*/ false, BuilderM, lctx,
              getDynamicLoopLimit(LI.getLoopFor(lc.header)),
              /*isi1*/ false);
          unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = lim;
          return lim;
        }
      }
    }

    auto parent = phi->getParent();

    // Don't attempt to unroll a loop induction variable in other
    // circumstances
    auto &LLI = Logic.PPC.FAM.getResult<LoopAnalysis>(*parent->getParent());
    if (LLI.isLoopHeader(parent)) {
      assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
      goto endCheck;
    }
    for (auto &val : phi->incoming_values()) {
      if (isPotentialLastLoopValue(val, parent, LLI)) {
        assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
        goto endCheck;
      }
    }

    if (phi->getNumIncomingValues() == 1) {
      assert(phi->getIncomingValue(0) != phi);
      auto toreturn = getOpUnchecked(phi->getIncomingValue(0));
      if (toreturn == nullptr || toreturn == phi) {
        assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
        goto endCheck;
      }
      assert(val->getType() == toreturn->getType());
      return toreturn;
    }

    std::set<BasicBlock *> targetToPreds;
    // Map of function edges to list of values possible
    std::map<std::pair</*pred*/ BasicBlock *, /*successor*/ BasicBlock *>,
             std::set<BasicBlock *>>
        done;
    {
      std::deque<std::tuple<
          std::pair</*pred*/ BasicBlock *, /*successor*/ BasicBlock *>,
          BasicBlock *>>
          Q; // newblock, target

      for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
        Q.push_back(
            std::make_pair(std::make_pair(phi->getIncomingBlock(i), parent),
                           phi->getIncomingBlock(i)));
        targetToPreds.insert(phi->getIncomingBlock(i));
      }

      for (std::tuple<
               std::pair</*pred*/ BasicBlock *, /*successor*/ BasicBlock *>,
               BasicBlock *>
               trace;
           Q.size() > 0;) {
        trace = Q.front();
        Q.pop_front();
        auto edge = std::get<0>(trace);
        auto block = edge.first;
        auto target = std::get<1>(trace);

        if (done[edge].count(target))
          continue;
        done[edge].insert(target);

        Loop *blockLoop = LI.getLoopFor(block);

        for (BasicBlock *Pred : predecessors(block)) {
          // Don't go up the backedge as we can use the last value if desired
          // via lcssa
          if (blockLoop && blockLoop->getHeader() == block &&
              blockLoop == LI.getLoopFor(Pred))
            continue;

          Q.push_back(
              std::tuple<std::pair<BasicBlock *, BasicBlock *>, BasicBlock *>(
                  std::make_pair(Pred, block), target));
        }
      }
    }

    std::set<BasicBlock *> blocks;
    for (auto pair : done) {
      const auto &edge = pair.first;
      blocks.insert(edge.first);
    }

    if (targetToPreds.size() == 3) {
      for (auto block : blocks) {
        std::set<BasicBlock *> foundtargets;
        std::set<BasicBlock *> uniqueTargets;
        for (BasicBlock *succ : successors(block)) {
          auto edge = std::make_pair(block, succ);
          for (BasicBlock *target : done[edge]) {
            if (foundtargets.find(target) != foundtargets.end()) {
              goto rnextpair;
            }
            foundtargets.insert(target);
            if (done[edge].size() == 1)
              uniqueTargets.insert(target);
          }
        }
        if (foundtargets.size() != 3)
          goto rnextpair;
        if (uniqueTargets.size() != 1)
          goto rnextpair;

        {
          BasicBlock *subblock = nullptr;
          for (auto block2 : blocks) {
            std::set<BasicBlock *> seen2;
            for (BasicBlock *succ : successors(block2)) {
              auto edge = std::make_pair(block2, succ);
              if (done[edge].size() != 1) {
                // llvm::errs() << " -- failed from noonesize\n";
                goto nextblock;
              }
              for (BasicBlock *target : done[edge]) {
                if (seen2.find(target) != seen2.end()) {
                  // llvm::errs() << " -- failed from not uniqueTargets\n";
                  goto nextblock;
                }
                seen2.insert(target);
                if (foundtargets.find(target) == foundtargets.end()) {
                  // llvm::errs() << " -- failed from not unknown target\n";
                  goto nextblock;
                }
                if (uniqueTargets.find(target) != uniqueTargets.end()) {
                  // llvm::errs() << " -- failed from not same target\n";
                  goto nextblock;
                }
              }
            }
            if (seen2.size() != 2) {
              // llvm::errs() << " -- failed from not 2 seen\n";
              goto nextblock;
            }
            subblock = block2;
            break;
          nextblock:;
          }

          if (subblock == nullptr)
            goto rnextpair;

          {
            auto bi1 = cast<BranchInst>(block->getTerminator());

            auto cond1 = getOp(bi1->getCondition());
            if (cond1 == nullptr) {
              assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
              goto endCheck;
            }
            auto bi2 = cast<BranchInst>(subblock->getTerminator());
            auto cond2 = getOp(bi2->getCondition());
            if (cond2 == nullptr) {
              assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
              goto endCheck;
            }

            BasicBlock *oldB = BuilderM.GetInsertBlock();
            if (BuilderM.GetInsertPoint() != oldB->end()) {
              assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
              goto endCheck;
            }

            auto found = reverseBlockToPrimal.find(oldB);
            if (found == reverseBlockToPrimal.end()) {
              assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
              goto endCheck;
            }
            BasicBlock *fwd = found->second;

            SmallVector<BasicBlock *, 2> predBlocks;
            predBlocks.push_back(bi2->getSuccessor(0));
            predBlocks.push_back(bi2->getSuccessor(1));
            for (int i = 0; i < 2; i++) {
              auto edge = std::make_pair(block, bi1->getSuccessor(i));
              if (done[edge].size() == 1) {
                predBlocks.push_back(bi1->getSuccessor(i));
              }
            }

            SmallVector<Value *, 2> vals;

            SmallVector<BasicBlock *, 2> blocks;
            SmallVector<BasicBlock *, 2> endingBlocks;

            BasicBlock *last = oldB;

            BasicBlock *bret = BasicBlock::Create(
                val->getContext(), oldB->getName() + "_phimerge", newFunc);

            for (size_t i = 0; i < predBlocks.size(); i++) {
              BasicBlock *valparent = (i < 2) ? subblock : block;
              assert(done.find(std::make_pair(valparent, predBlocks[i])) !=
                     done.end());
              assert(done[std::make_pair(valparent, predBlocks[i])].size() ==
                     1);
              blocks.push_back(BasicBlock::Create(
                  val->getContext(), oldB->getName() + "_phirc", newFunc));
              blocks[i]->moveAfter(last);
              last = blocks[i];
              reverseBlocks[fwd].push_back(blocks[i]);
              reverseBlockToPrimal[blocks[i]] = fwd;
              IRBuilder<> B(blocks[i]);

              unwrap_cache[blocks[i]] = unwrap_cache[oldB];
              lookup_cache[blocks[i]] = lookup_cache[oldB];
              auto PB = *done[std::make_pair(valparent, predBlocks[i])].begin();

              if (auto inst = dyn_cast<Instruction>(
                      phi->getIncomingValueForBlock(PB))) {
                // Recompute the phi computation with the conditional if:
                // 1) the instruction may reat from memory AND does not
                //    dominate the current insertion point (thereby
                //    potentially making such recomputation without the
                //    condition illegal)
                // 2) the value is a call or load and option is set to not
                //    speculatively recompute values within a phi
                BasicBlock *nextScope = PB;
                // if (inst->getParent() == nextScope) nextScope =
                // phi->getParent();
                if ((inst->mayReadFromMemory() &&
                     !DT.dominates(inst->getParent(), phi->getParent())) ||
                    (!EnzymeSpeculatePHIs &&
                     (isa<CallInst>(inst) || isa<LoadInst>(inst))))
                  vals.push_back(getOpFull(B, inst, nextScope));
                else
                  vals.push_back(getOpFull(BuilderM, inst, nextScope));
              } else
                vals.push_back(
                    getOpFull(BuilderM, phi->getIncomingValueForBlock(PB), PB));

              if (!vals[i]) {
                for (size_t j = 0; j <= i; j++) {
                  reverseBlocks[fwd].erase(std::find(reverseBlocks[fwd].begin(),
                                                     reverseBlocks[fwd].end(),
                                                     blocks[j]));
                  reverseBlockToPrimal.erase(blocks[j]);
                  unwrap_cache.erase(blocks[j]);
                  lookup_cache.erase(blocks[j]);
                  SmallVector<Instruction *, 4> toErase;
                  for (auto &I : *blocks[j]) {
                    toErase.push_back(&I);
                  }
                  for (auto I : toErase) {
                    erase(I);
                  }
                }
                bret->eraseFromParent();
                for (size_t j = 0; j <= i; j++) {
                  blocks[j]->eraseFromParent();
                };
                assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
                goto endCheck;
              }
              assert(val->getType() == vals[i]->getType());
              B.CreateBr(bret);
              endingBlocks.push_back(B.GetInsertBlock());
            }

            bret->moveAfter(last);

            BasicBlock *bsplit = BasicBlock::Create(
                val->getContext(), oldB->getName() + "_phisplt", newFunc);
            bsplit->moveAfter(oldB);
            BuilderM.CreateCondBr(
                cond1,
                (done[std::make_pair(block, bi1->getSuccessor(0))].size() == 1)
                    ? blocks[2]
                    : bsplit,
                (done[std::make_pair(block, bi1->getSuccessor(1))].size() == 1)
                    ? blocks[2]
                    : bsplit);

            BuilderM.SetInsertPoint(bsplit);
            BuilderM.CreateCondBr(cond2, blocks[0], blocks[1]);

            BuilderM.SetInsertPoint(bret);
            reverseBlocks[fwd].push_back(bret);
            reverseBlockToPrimal[bret] = fwd;
            auto toret = BuilderM.CreatePHI(val->getType(), vals.size());
            for (size_t i = 0; i < vals.size(); i++)
              toret->addIncoming(vals[i], endingBlocks[i]);
            assert(val->getType() == toret->getType());
            if (permitCache) {
              unwrap_cache[bret][idx.first][idx.second] = toret;
            }
            unwrappedLoads[toret] = val;
            unwrap_cache[bret] = unwrap_cache[oldB];
            lookup_cache[bret] = lookup_cache[oldB];
            return toret;
          }
        }
      rnextpair:;
      }
    }

    Instruction *equivalentTerminator = nullptr;
    for (auto block : blocks) {
      std::set<BasicBlock *> foundtargets;
      for (BasicBlock *succ : successors(block)) {
        auto edge = std::make_pair(block, succ);
        if (done[edge].size() != 1) {
          goto nextpair;
        }
        BasicBlock *target = *done[edge].begin();
        if (foundtargets.find(target) != foundtargets.end()) {
          goto nextpair;
        }
        foundtargets.insert(target);
      }
      if (foundtargets.size() != targetToPreds.size()) {
        goto nextpair;
      }

      if (block == parent || DT.dominates(block, parent)) {
        equivalentTerminator = block->getTerminator();
        goto fast;
      }

    nextpair:;
    }
    goto endCheck;

  fast:;
    assert(equivalentTerminator);

    if (isa<BranchInst>(equivalentTerminator) ||
        isa<SwitchInst>(equivalentTerminator)) {
      BasicBlock *oldB = BuilderM.GetInsertBlock();

      BasicBlock *fwd = oldB;
      if (!isOriginalBlock(*fwd)) {
        auto found = reverseBlockToPrimal.find(oldB);
        if (found == reverseBlockToPrimal.end()) {
          assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
          goto endCheck;
        }
        fwd = found->second;
      }

      SmallVector<BasicBlock *, 2> predBlocks;
      Value *cond = nullptr;
      if (auto branch = dyn_cast<BranchInst>(equivalentTerminator)) {
        cond = getOp(branch->getCondition());
        predBlocks.push_back(branch->getSuccessor(0));
        predBlocks.push_back(branch->getSuccessor(1));
      } else {
        auto SI = cast<SwitchInst>(equivalentTerminator);
        cond = getOp(SI->getCondition());
        predBlocks.push_back(SI->getDefaultDest());
        for (auto scase : SI->cases()) {
          predBlocks.push_back(scase.getCaseSuccessor());
        }
      }

      if (cond == nullptr) {
        assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
        goto endCheck;
      }

      SmallVector<Value *, 2> vals;

      SmallVector<BasicBlock *, 2> blocks;
      SmallVector<BasicBlock *, 2> endingBlocks;

      BasicBlock *last = oldB;

      BasicBlock *bret = BasicBlock::Create(
          val->getContext(), oldB->getName() + "_phimerge", newFunc);

      for (size_t i = 0; i < predBlocks.size(); i++) {
        assert(done.find(std::make_pair(equivalentTerminator->getParent(),
                                        predBlocks[i])) != done.end());
        assert(done[std::make_pair(equivalentTerminator->getParent(),
                                   predBlocks[i])]
                   .size() == 1);
        BasicBlock *PB = *done[std::make_pair(equivalentTerminator->getParent(),
                                              predBlocks[i])]
                              .begin();
        blocks.push_back(BasicBlock::Create(
            val->getContext(), oldB->getName() + "_phirc", newFunc));
        blocks[i]->moveAfter(last);
        last = blocks[i];
        if (reverseBlocks.size() > 0) {
          reverseBlocks[fwd].push_back(blocks[i]);
          reverseBlockToPrimal[blocks[i]] = fwd;
        }
        IRBuilder<> B(blocks[i]);

        unwrap_cache[blocks[i]] = unwrap_cache[oldB];
        lookup_cache[blocks[i]] = lookup_cache[oldB];

        if (auto inst =
                dyn_cast<Instruction>(phi->getIncomingValueForBlock(PB))) {
          // Recompute the phi computation with the conditional if:
          // 1) the instruction may reat from memory AND does not dominate
          //    the current insertion point (thereby potentially making such
          //    recomputation without the condition illegal)
          // 2) the value is a call or load and option is set to not
          //    speculatively recompute values within a phi
          BasicBlock *nextScope = PB;
          // if (inst->getParent() == nextScope) nextScope = phi->getParent();
          if ((inst->mayReadFromMemory() &&
               !DT.dominates(inst->getParent(), phi->getParent())) ||
              (!EnzymeSpeculatePHIs &&
               (isa<CallInst>(inst) || isa<LoadInst>(inst))))
            vals.push_back(getOpFull(B, inst, nextScope));
          else
            vals.push_back(getOpFull(BuilderM, inst, nextScope));
        } else
          vals.push_back(phi->getIncomingValueForBlock(PB));

        if (!vals[i]) {
          for (size_t j = 0; j <= i; j++) {
            if (reverseBlocks.size() > 0) {
              reverseBlocks[fwd].erase(std::find(reverseBlocks[fwd].begin(),
                                                 reverseBlocks[fwd].end(),
                                                 blocks[j]));
              reverseBlockToPrimal.erase(blocks[j]);
            }
            unwrap_cache.erase(blocks[j]);
            lookup_cache.erase(blocks[j]);
            SmallVector<Instruction *, 4> toErase;
            for (auto &I : *blocks[j]) {
              toErase.push_back(&I);
            }
            for (auto I : toErase) {
              erase(I);
            }
          }
          bret->eraseFromParent();
          for (size_t j = 0; j <= i; j++) {
            blocks[j]->eraseFromParent();
          };
          assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
          goto endCheck;
        }
        assert(val->getType() == vals[i]->getType());
        B.CreateBr(bret);
        endingBlocks.push_back(B.GetInsertBlock());
      }

      // Fast path to not make a split block if no additional instructions
      // were made in the two blocks
      if (isa<BranchInst>(equivalentTerminator) && blocks[0]->size() == 1 &&
          blocks[1]->size() == 1) {
        for (size_t j = 0; j < blocks.size(); j++) {
          if (reverseBlocks.size() > 0) {
            reverseBlocks[fwd].erase(std::find(reverseBlocks[fwd].begin(),
                                               reverseBlocks[fwd].end(),
                                               blocks[j]));
            reverseBlockToPrimal.erase(blocks[j]);
          }
          unwrap_cache.erase(blocks[j]);
          lookup_cache.erase(blocks[j]);
          SmallVector<Instruction *, 4> toErase;
          for (auto &I : *blocks[j]) {
            toErase.push_back(&I);
          }
          for (auto I : toErase) {
            erase(I);
          }
        }
        bret->eraseFromParent();
        for (size_t j = 0; j < blocks.size(); j++) {
          blocks[j]->eraseFromParent();
        };
        Value *toret = BuilderM.CreateSelect(cond, vals[0], vals[1],
                                             phi->getName() + "_unwrap");
        if (permitCache) {
          unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] =
              toret;
        }
        if (auto instRet = dyn_cast<Instruction>(toret))
          unwrappedLoads[instRet] = val;
        return toret;
      }

      if (BuilderM.GetInsertPoint() != oldB->end()) {
        for (size_t j = 0; j < blocks.size(); j++) {
          if (reverseBlocks.size() > 0) {
            reverseBlocks[fwd].erase(std::find(reverseBlocks[fwd].begin(),
                                               reverseBlocks[fwd].end(),
                                               blocks[j]));
            reverseBlockToPrimal.erase(blocks[j]);
          }
          unwrap_cache.erase(blocks[j]);
          lookup_cache.erase(blocks[j]);
          SmallVector<Instruction *, 4> toErase;
          for (auto &I : *blocks[j]) {
            toErase.push_back(&I);
          }
          for (auto I : toErase) {
            erase(I);
          }
        }
        bret->eraseFromParent();
        for (size_t j = 0; j < blocks.size(); j++) {
          blocks[j]->eraseFromParent();
        };
        assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
        goto endCheck;
      }

      bret->moveAfter(last);
      if (isa<BranchInst>(equivalentTerminator)) {
        BuilderM.CreateCondBr(cond, blocks[0], blocks[1]);
      } else {
        auto SI = cast<SwitchInst>(equivalentTerminator);
        auto NSI = BuilderM.CreateSwitch(cond, blocks[0], SI->getNumCases());
        size_t idx = 1;
        for (auto scase : SI->cases()) {
          NSI->addCase(scase.getCaseValue(), blocks[idx]);
          idx++;
        }
      }
      BuilderM.SetInsertPoint(bret);
      reverseBlocks[fwd].push_back(bret);
      reverseBlockToPrimal[bret] = fwd;
      auto toret = BuilderM.CreatePHI(val->getType(), vals.size());
      for (size_t i = 0; i < vals.size(); i++)
        toret->addIncoming(vals[i], endingBlocks[i]);
      assert(val->getType() == toret->getType());
      if (permitCache) {
        unwrap_cache[bret][idx.first][idx.second] = toret;
      }
      unwrap_cache[bret] = unwrap_cache[oldB];
      lookup_cache[bret] = lookup_cache[oldB];
      unwrappedLoads[toret] = val;
      return toret;
    }
    assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
    goto endCheck;
  }

endCheck:
  assert(val);
  if (unwrapMode == UnwrapMode::LegalFullUnwrap ||
      unwrapMode == UnwrapMode::LegalFullUnwrapNoTapeReplace ||
      unwrapMode == UnwrapMode::AttemptFullUnwrapWithLookup) {
    assert(val->getName() != "<badref>");
    Value *nval = val;
    if (auto opinst = dyn_cast<Instruction>(nval))
      if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
        if (!DT.dominates(opinst, &*BuilderM.GetInsertPoint())) {
          if (unwrapMode != UnwrapMode::AttemptFullUnwrapWithLookup) {
            llvm::errs() << " oldF: " << *oldFunc << "\n";
            llvm::errs() << " opParen: " << *opinst->getParent()->getParent()
                         << "\n";
            llvm::errs() << " newF: " << *newFunc << "\n";
            llvm::errs() << " - blk: " << *BuilderM.GetInsertBlock();
            llvm::errs() << " opInst: " << *opinst << " mode=" << unwrapMode
                         << "\n";
          }
          assert(unwrapMode == UnwrapMode::AttemptFullUnwrapWithLookup);
          return nullptr;
        }
      }
    if (scope)
      if (auto opinst = dyn_cast<Instruction>(nval)) {
        nval = fixLCSSA(opinst, scope);
      }
    auto toreturn =
        lookupM(nval, BuilderM, available, /*tryLegalRecomputeCheck*/ false);
    assert(val->getType() == toreturn->getType());
    return toreturn;
  }

  if (auto inst = dyn_cast<Instruction>(val)) {
    if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
      if (BuilderM.GetInsertBlock()->size() &&
          BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
        if (DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
          assert(inst->getType() == val->getType());
          return inst;
        }
      } else {
        if (DT.dominates(inst, BuilderM.GetInsertBlock())) {
          assert(inst->getType() == val->getType());
          return inst;
        }
      }
    }
    assert(val->getName() != "<badref>");
    auto &warnMap = UnwrappedWarnings[inst];
    if (!warnMap.count(BuilderM.GetInsertBlock())) {
      EmitWarning("NoUnwrap", inst->getDebugLoc(), oldFunc, inst->getParent(),
                  "Cannot unwrap ", *val, " in ",
                  BuilderM.GetInsertBlock()->getName());
      warnMap.insert(BuilderM.GetInsertBlock());
    }
  }
  return nullptr;
}

Value *GradientUtils::cacheForReverse(IRBuilder<> &BuilderQ, Value *malloc,
                                      int idx, bool ignoreType, bool replace) {
  assert(malloc);
  assert(BuilderQ.GetInsertBlock()->getParent() == newFunc);
  assert(isOriginalBlock(*BuilderQ.GetInsertBlock()));
  if (mode == DerivativeMode::ReverseModeCombined) {
    assert(!tape);
    return malloc;
  }

  if (auto CI = dyn_cast<CallInst>(malloc)) {
    if (auto F = CI->getCalledFunction()) {
      assert(F->getName() != "omp_get_thread_num");
    }
  }

  if (malloc->getType()->isTokenTy()) {
    llvm::errs() << " oldFunc: " << *oldFunc << "\n";
    llvm::errs() << " newFunc: " << *newFunc << "\n";
    llvm::errs() << " malloc: " << *malloc << "\n";
  }
  assert(!malloc->getType()->isTokenTy());

  if (tape) {
    if (idx >= 0 && !tape->getType()->isStructTy()) {
      llvm::errs() << "cacheForReverse incorrect tape type: " << *tape
                   << " idx: " << idx << "\n";
    }
    assert(idx < 0 || tape->getType()->isStructTy());
    if (idx >= 0 &&
        (unsigned)idx >= cast<StructType>(tape->getType())->getNumElements()) {
      llvm::errs() << "oldFunc: " << *oldFunc << "\n";
      llvm::errs() << "newFunc: " << *newFunc << "\n";
      if (malloc)
        llvm::errs() << "malloc: " << *malloc << "\n";
      llvm::errs() << "tape: " << *tape << "\n";
      llvm::errs() << "idx: " << idx << "\n";
    }
    assert(idx < 0 ||
           (unsigned)idx < cast<StructType>(tape->getType())->getNumElements());
    Value *ret =
        (idx < 0) ? tape : BuilderQ.CreateExtractValue(tape, {(unsigned)idx});

    if (ret->getType()->isEmptyTy()) {
      if (auto inst = dyn_cast_or_null<Instruction>(malloc)) {
        if (!ignoreType) {
          if (inst->getType() != ret->getType()) {
            llvm::errs() << "oldFunc: " << *oldFunc << "\n";
            llvm::errs() << "newFunc: " << *newFunc << "\n";
            llvm::errs() << "inst==malloc: " << *inst << "\n";
            llvm::errs() << "ret: " << *ret << "\n";
          }
          assert(inst->getType() == ret->getType());
          if (replace)
            inst->replaceAllUsesWith(UndefValue::get(ret->getType()));
        }
        if (replace)
          erase(inst);
      }
      Type *retType = ret->getType();
      if (replace)
        if (auto ri = dyn_cast<Instruction>(ret))
          erase(ri);
      return UndefValue::get(retType);
    }

    LimitContext ctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                     BuilderQ.GetInsertBlock());
    if (auto inst = dyn_cast<Instruction>(malloc))
      ctx = LimitContext(/*ReverseLimit*/ reverseBlocks.size() > 0,
                         inst->getParent());
    if (auto found = findInMap(scopeMap, malloc)) {
      ctx = found->second;
    }
    assert(isOriginalBlock(*ctx.Block));

    bool inLoop;
    if (ctx.ForceSingleIteration) {
      inLoop = true;
      ctx.ForceSingleIteration = false;
    } else {
      LoopContext lc;
      inLoop = getContext(ctx.Block, lc);
    }

    if (!inLoop) {
      if (malloc)
        ret->setName(malloc->getName() + "_fromtape");
      if (omp) {
        Value *tid = ompThreadId();
#if LLVM_VERSION_MAJOR > 7
        Value *tPtr = BuilderQ.CreateInBoundsGEP(
            cast<PointerType>(ret->getType())->getElementType(), ret,
            ArrayRef<Value *>(tid));
#else
        Value *tPtr = BuilderQ.CreateInBoundsGEP(ret, ArrayRef<Value *>(tid));
#endif
        ret = BuilderQ.CreateLoad(
            cast<PointerType>(ret->getType())->getElementType(), tPtr);
      }
    } else {
      if (idx >= 0)
        erase(cast<Instruction>(ret));
      IRBuilder<> entryBuilder(inversionAllocs);
      entryBuilder.setFastMathFlags(getFast());
      ret = (idx < 0) ? tape
                      : entryBuilder.CreateExtractValue(tape, {(unsigned)idx});

      Type *innerType = ret->getType();
      for (size_t i = 0,
                  limit = getSubLimits(
                              /*inForwardPass*/ true, nullptr,
                              LimitContext(
                                  /*ReverseLimit*/ reverseBlocks.size() > 0,
                                  BuilderQ.GetInsertBlock()))
                              .size();
           i < limit; ++i) {
        if (!isa<PointerType>(innerType)) {
          llvm::errs() << "mod: "
                       << *BuilderQ.GetInsertBlock()->getParent()->getParent()
                       << "\n";
          llvm::errs() << "fn: " << *BuilderQ.GetInsertBlock()->getParent()
                       << "\n";
          llvm::errs() << "bq insertblock: " << *BuilderQ.GetInsertBlock()
                       << "\n";
          llvm::errs() << "ret: " << *ret << " type: " << *ret->getType()
                       << "\n";
          llvm::errs() << "innerType: " << *innerType << "\n";
          if (malloc)
            llvm::errs() << " malloc: " << *malloc << " i=" << i
                         << " / lim = " << limit << "\n";
        }
        assert(isa<PointerType>(innerType));
        innerType = cast<PointerType>(innerType)->getElementType();
      }

      assert(malloc);
      if (!ignoreType) {
        if (EfficientBoolCache && malloc->getType()->isIntegerTy() &&
            cast<IntegerType>(malloc->getType())->getBitWidth() == 1 &&
            innerType != ret->getType()) {
          assert(innerType == Type::getInt8Ty(malloc->getContext()));
        } else {
          if (innerType != malloc->getType()) {
            llvm::errs() << *oldFunc << "\n";
            llvm::errs() << *newFunc << "\n";
            llvm::errs() << "innerType: " << *innerType << "\n";
            llvm::errs() << "malloc->getType(): " << *malloc->getType() << "\n";
            llvm::errs() << "ret: " << *ret << " - " << *ret->getType() << "\n";
            llvm::errs() << "malloc: " << *malloc << "\n";
            assert(0 && "illegal loop cache type");
            llvm_unreachable("illegal loop cache type");
          }
        }
      }

      LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                        BuilderQ.GetInsertBlock());
      AllocaInst *cache = createCacheForScope(
          lctx, innerType, "mdyncache_fromtape", true, false);
      assert(malloc);
      bool isi1 = !ignoreType && malloc->getType()->isIntegerTy() &&
                  cast<IntegerType>(malloc->getType())->getBitWidth() == 1;
      assert(isa<PointerType>(cache->getType()));
      assert(cast<PointerType>(cache->getType())->getElementType() ==
             ret->getType());
      entryBuilder.CreateStore(ret, cache);

      auto v = lookupValueFromCache(/*forwardPass*/ true, BuilderQ, lctx, cache,
                                    isi1);
      if (!ignoreType && malloc) {
        assert(v->getType() == malloc->getType());
      }
      insert_or_assign(scopeMap, v,
                       std::make_pair(AssertingVH<AllocaInst>(cache), ctx));
      ret = cast<Instruction>(v);
    }

    if (malloc && !isa<UndefValue>(malloc)) {
      if (!ignoreType) {
        if (malloc->getType() != ret->getType()) {
          llvm::errs() << *oldFunc << "\n";
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << *malloc << "\n";
          llvm::errs() << *ret << "\n";
        }
        assert(malloc->getType() == ret->getType());
      }

      if (replace) {
        auto found = newToOriginalFn.find(malloc);
        if (found != newToOriginalFn.end()) {
          Value *orig = found->second;
          originalToNewFn[orig] = ret;
          newToOriginalFn.erase(malloc);
          newToOriginalFn[ret] = orig;
        }
      }

      if (auto found = findInMap(scopeMap, malloc)) {
        // There already exists an alloaction for this, we should fully remove
        // it
        if (!inLoop) {

          // Remove stores into
          SmallVector<Instruction *, 3> stores(
              scopeInstructions[found->first].begin(),
              scopeInstructions[found->first].end());
          scopeInstructions.erase(found->first);
          for (int i = stores.size() - 1; i >= 0; i--) {
            erase(stores[i]);
          }

          std::vector<User *> users;
          for (auto u : found->first->users()) {
            users.push_back(u);
          }
          for (auto u : users) {
            if (auto li = dyn_cast<LoadInst>(u)) {
              IRBuilder<> lb(li);
              if (replace) {
                auto replacewith =
                    (idx < 0) ? tape
                              : lb.CreateExtractValue(tape, {(unsigned)idx});
                li->replaceAllUsesWith(replacewith);
              } else {
                auto phi =
                    lb.CreatePHI(li->getType(), 0, li->getName() + "_cfrphi");
                unwrappedLoads[phi] = malloc;
                li->replaceAllUsesWith(phi);
              }
              erase(li);
            } else {
              llvm::errs() << "newFunc: " << *newFunc << "\n";
              llvm::errs() << "malloc: " << *malloc << "\n";
              llvm::errs() << "scopeMap[malloc]: " << *found->first << "\n";
              llvm::errs() << "u: " << *u << "\n";
              assert(0 && "illegal use for out of loop scopeMap1");
            }
          }

          {
            AllocaInst *preerase = found->first;
            scopeMap.erase(malloc);
            erase(preerase);
          }
        } else {
          // Remove stores into
          SmallVector<Instruction *, 3> stores(
              scopeInstructions[found->first].begin(),
              scopeInstructions[found->first].end());
          scopeInstructions.erase(found->first);
          for (int i = stores.size() - 1; i >= 0; i--) {
            erase(stores[i]);
          }

          // Remove allocations for scopealloc since it is already allocated
          // by the augmented forward pass
          SmallVector<CallInst *, 3> allocs(scopeAllocs[found->first].begin(),
                                            scopeAllocs[found->first].end());
          scopeAllocs.erase(found->first);
          for (auto allocinst : allocs) {
            CastInst *cast = nullptr;
            StoreInst *store = nullptr;
            for (auto use : allocinst->users()) {
              if (auto ci = dyn_cast<CastInst>(use)) {
                assert(cast == nullptr);
                cast = ci;
              }
              if (auto si = dyn_cast<StoreInst>(use)) {
                if (si->getValueOperand() == allocinst) {
                  assert(store == nullptr);
                  store = si;
                }
              }
            }
            if (cast) {
              assert(store == nullptr);
              for (auto use : cast->users()) {
                if (auto si = dyn_cast<StoreInst>(use)) {
                  if (si->getValueOperand() == cast) {
                    assert(store == nullptr);
                    store = si;
                  }
                }
              }
            }

            Instruction *storedinto =
                cast ? (Instruction *)cast : (Instruction *)allocinst;
            for (auto use : storedinto->users()) {
              if (auto si = dyn_cast<StoreInst>(use))
                erase(si);
            }

            if (cast)
              erase(cast);
            erase(allocinst);
          }

          // Remove frees
          SmallVector<CallInst *, 3> tofree(scopeFrees[found->first].begin(),
                                            scopeFrees[found->first].end());
          scopeFrees.erase(found->first);
          for (auto freeinst : tofree) {
            // This deque contains a list of operations
            // we can erasing upon erasing the free (and so on).
            // Since multiple operations can have the same operand,
            // this deque can contain the same value multiple times.
            // To remedy this we use a tracking value handle which will
            // be set to null when erased.
            std::deque<WeakTrackingVH> ops = {freeinst->getArgOperand(0)};
            erase(freeinst);

            while (ops.size()) {
              auto z = dyn_cast_or_null<Instruction>(ops[0]);
              ops.pop_front();
              if (z && z->getNumUses() == 0) {
                for (unsigned i = 0; i < z->getNumOperands(); ++i) {
                  ops.push_back(z->getOperand(i));
                }
                erase(z);
              }
            }
          }

          // uses of the alloc
          std::vector<User *> users;
          for (auto u : found->first->users()) {
            users.push_back(u);
          }
          for (auto u : users) {
            if (auto li = dyn_cast<LoadInst>(u)) {
              // even with replace off, this can be replaced
              // as since we're in a loop this load is a load of cache
              // not of the final value (thereby overwriting the new
              // inst
              IRBuilder<> lb(li);
              auto replacewith =
                  (idx < 0) ? tape
                            : lb.CreateExtractValue(tape, {(unsigned)idx});
              li->replaceAllUsesWith(replacewith);
              erase(li);
            } else {
              llvm::errs() << "newFunc: " << *newFunc << "\n";
              llvm::errs() << "malloc: " << *malloc << "\n";
              llvm::errs() << "scopeMap[malloc]: " << *found->first << "\n";
              llvm::errs() << "u: " << *u << "\n";
              assert(0 && "illegal use for out of loop scopeMap2");
            }
          }

          AllocaInst *preerase = found->first;
          scopeMap.erase(malloc);
          if (replace)
            erase(preerase);
        }
      }
      if (!ignoreType && replace)
        cast<Instruction>(malloc)->replaceAllUsesWith(ret);
      ret->takeName(malloc);
      if (replace)
        erase(cast<Instruction>(malloc));
    }
    return ret;
  } else {
    assert(malloc);
    assert(!ignoreType);

    assert(idx >= 0 && (unsigned)idx == addedTapeVals.size());

    if (isa<UndefValue>(malloc)) {
      addedTapeVals.push_back(malloc);
      return malloc;
    }

    LimitContext ctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                     BuilderQ.GetInsertBlock());
    if (auto inst = dyn_cast<Instruction>(malloc))
      ctx = LimitContext(/*ReverseLimit*/ reverseBlocks.size() > 0,
                         inst->getParent());
    if (auto found = findInMap(scopeMap, malloc)) {
      ctx = found->second;
    }

    bool inLoop;

    if (ctx.ForceSingleIteration) {
      inLoop = true;
      ctx.ForceSingleIteration = false;
    } else {
      LoopContext lc;
      inLoop = getContext(ctx.Block, lc);
    }

    if (!inLoop) {
      Value *toStoreInTape = malloc;
      if (omp) {
        Value *numThreads = ompNumThreads();
        Value *tid = ompThreadId();
        IRBuilder<> entryBuilder(inversionAllocs);

        Constant *byteSizeOfType = ConstantInt::get(
            numThreads->getType(),
            (newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(
                 malloc->getType()) +
             7) /
                8,
            false);

        auto firstallocation = cast<Instruction>(CallInst::CreateMalloc(
            inversionAllocs, numThreads->getType(), malloc->getType(),
            byteSizeOfType, numThreads, nullptr,
            malloc->getName() + "_malloccache"));
        if (firstallocation->getParent() == nullptr) {
          inversionAllocs->getInstList().push_back(firstallocation);
        }

        if (auto inst = dyn_cast<Instruction>(malloc)) {
          entryBuilder.SetInsertPoint(inst->getNextNode());
        }
#if LLVM_VERSION_MAJOR > 7
        Value *tPtr = entryBuilder.CreateInBoundsGEP(
            firstallocation->getType()->getPointerElementType(),
            firstallocation, ArrayRef<Value *>(tid));
#else
        Value *tPtr = entryBuilder.CreateInBoundsGEP(firstallocation,
                                                     ArrayRef<Value *>(tid));
#endif
        entryBuilder.CreateStore(malloc, tPtr);
        toStoreInTape = firstallocation;
      }
      addedTapeVals.push_back(toStoreInTape);
      return malloc;
    }

    ensureLookupCached(cast<Instruction>(malloc),
                       /*shouldFree=*/reverseBlocks.size() > 0);
    auto found2 = scopeMap.find(malloc);
    assert(found2 != scopeMap.end());
    assert(found2->second.first);

    Value *toadd;
    toadd = scopeAllocs[found2->second.first][0];
    for (auto u : toadd->users()) {
      if (auto ci = dyn_cast<CastInst>(u)) {
        toadd = ci;
      }
    }

    // llvm::errs() << " malloc: " << *malloc << "\n";
    // llvm::errs() << " toadd: " << *toadd << "\n";
    Type *innerType = toadd->getType();
    for (size_t
             i = 0,
             limit = getSubLimits(
                         /*inForwardPass*/ true, nullptr,
                         LimitContext(/*ReverseLimit*/ reverseBlocks.size() > 0,
                                      BuilderQ.GetInsertBlock()))
                         .size();
         i < limit; ++i) {
      innerType = cast<PointerType>(innerType)->getElementType();
    }
    assert(!ignoreType);
    if (EfficientBoolCache && malloc->getType()->isIntegerTy() &&
        toadd->getType() != innerType &&
        cast<IntegerType>(malloc->getType())->getBitWidth() == 1) {
      assert(innerType == Type::getInt8Ty(toadd->getContext()));
    } else {
      if (innerType != malloc->getType()) {
        llvm::errs() << "oldFunc:" << *oldFunc << "\n";
        llvm::errs() << "newFunc: " << *newFunc << "\n";
        llvm::errs() << " toadd: " << *toadd << "\n";
        llvm::errs() << "innerType: " << *innerType << "\n";
        llvm::errs() << "malloc: " << *malloc << "\n";
      }
      assert(innerType == malloc->getType());
    }
    addedTapeVals.push_back(toadd);
    return malloc;
  }
  llvm::errs()
      << "Fell through on cacheForReverse. This should never happen.\n";
  assert(false);
}

/// Given an edge from BB to branchingBlock get the corresponding block to
/// branch to in the reverse pass
BasicBlock *GradientUtils::getReverseOrLatchMerge(BasicBlock *BB,
                                                  BasicBlock *branchingBlock) {
  assert(BB);
  // BB should be a forward pass block, assert that
  if (reverseBlocks.find(BB) == reverseBlocks.end()) {
    llvm::errs() << *oldFunc << "\n";
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << "BB: " << *BB << "\n";
    llvm::errs() << "branchingBlock: " << *branchingBlock << "\n";
  }
  assert(reverseBlocks.find(BB) != reverseBlocks.end());
  LoopContext lc;
  bool inLoop = getContext(BB, lc);

  LoopContext branchingContext;
  bool inLoopContext = getContext(branchingBlock, branchingContext);

  if (!inLoop)
    return reverseBlocks[BB].front();

  auto tup = std::make_tuple(BB, branchingBlock);
  if (newBlocksForLoop_cache.find(tup) != newBlocksForLoop_cache.end())
    return newBlocksForLoop_cache[tup];

  if (inLoop && inLoopContext && branchingBlock == lc.header &&
      lc.header == branchingContext.header) {
    BasicBlock *incB = BasicBlock::Create(
        BB->getContext(), "inc" + reverseBlocks[lc.header].front()->getName(),
        BB->getParent());
    incB->moveAfter(reverseBlocks[lc.header].back());

    IRBuilder<> tbuild(incB);

#if LLVM_VERSION_MAJOR > 7
    Value *av = tbuild.CreateLoad(
        cast<PointerType>(lc.antivaralloc->getType())->getElementType(),
        lc.antivaralloc);
#else
    Value *av = tbuild.CreateLoad(lc.antivaralloc);
#endif
    Value *sub = tbuild.CreateAdd(av, ConstantInt::get(av->getType(), -1), "",
                                  /*NUW*/ false, /*NSW*/ true);
    tbuild.CreateStore(sub, lc.antivaralloc);
    tbuild.CreateBr(reverseBlocks[BB].front());
    return newBlocksForLoop_cache[tup] = incB;
  }

  if (inLoop) {
    auto L = LI.getLoopFor(BB);
    auto latches = getLatches(L, lc.exitBlocks);

    if (std::find(latches.begin(), latches.end(), BB) != latches.end() &&
        std::find(lc.exitBlocks.begin(), lc.exitBlocks.end(), branchingBlock) !=
            lc.exitBlocks.end()) {
      BasicBlock *incB = BasicBlock::Create(
          BB->getContext(),
          "merge" + reverseBlocks[lc.header].front()->getName() + "_" +
              branchingBlock->getName(),
          BB->getParent());
      incB->moveAfter(reverseBlocks[branchingBlock].back());

      IRBuilder<> tbuild(reverseBlocks[branchingBlock].back());

      Value *lim = nullptr;
      if (lc.dynamic && assumeDynamicLoopOfSizeOne(L)) {
        lim = ConstantInt::get(lc.var->getType(), 0);
      } else if (lc.dynamic) {
        // Must be in a reverse pass fashion for a lookup to index bound to be
        // legal
        assert(/*ReverseLimit*/ reverseBlocks.size() > 0);
        LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                          lc.preheader);
        lim =
            lookupValueFromCache(/*forwardPass*/ false, tbuild, lctx,
                                 getDynamicLoopLimit(LI.getLoopFor(lc.header)),
                                 /*isi1*/ false);
      } else {
        lim = lookupM(lc.trueLimit, tbuild);
      }

      tbuild.SetInsertPoint(incB);
      tbuild.CreateStore(lim, lc.antivaralloc);
      tbuild.CreateBr(reverseBlocks[BB].front());

      return newBlocksForLoop_cache[tup] = incB;
    }
  }

  return newBlocksForLoop_cache[tup] = reverseBlocks[BB].front();
}

void GradientUtils::forceContexts() {
  for (auto BB : originalBlocks) {
    LoopContext lc;
    getContext(BB, lc);
  }
}

bool GradientUtils::legalRecompute(const Value *val,
                                   const ValueToValueMapTy &available,
                                   IRBuilder<> *BuilderM, bool reverse,
                                   bool legalRecomputeCache) const {
  if (available.count(val)) {
    return true;
  }

  if (auto phi = dyn_cast<PHINode>(val)) {
    if (auto uiv = hasUninverted(val)) {
      if (auto dli = dyn_cast_or_null<LoadInst>(uiv)) {
        return legalRecompute(
            dli, available, BuilderM,
            reverse); // TODO ADD && !TR.intType(getOriginal(dli),
                      // /*mustfind*/false).isPossibleFloat();
      }
      if (phi->getNumIncomingValues() == 0) {
        return false;
      }
    }

    if (phi->getNumIncomingValues() == 0) {
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *phi << "\n";
    }
    assert(phi->getNumIncomingValues() != 0);
    auto parent = phi->getParent();
    if (parent->getParent() == newFunc) {
      if (LI.isLoopHeader(parent)) {
        return false;
      }
      for (auto &val : phi->incoming_values()) {
        if (isPotentialLastLoopValue(val, parent, LI))
          return false;
      }
      return true;
    } else if (parent->getParent() == oldFunc) {
      if (OrigLI.isLoopHeader(parent)) {
        return false;
      }
      for (auto &val : phi->incoming_values()) {
        if (isPotentialLastLoopValue(val, parent, OrigLI))
          return false;
      }
      return true;
    } else {
      return false;
    }

    // if (SE.isSCEVable(phi->getType())) {
    // auto scev =
    // const_cast<GradientUtils*>(this)->SE.getSCEV(const_cast<Value*>(val));
    // llvm::errs() << "phi: " << *val << " scev: " << *scev << "\n";
    //}
    return false;
  }

  if (isa<Instruction>(val) &&
      cast<Instruction>(val)->getMetadata("enzyme_mustcache")) {
    return false;
  }

  // If this is a load from cache already, dont force a cache of this
  if (legalRecomputeCache && isa<LoadInst>(val) &&
      CacheLookups.count(cast<LoadInst>(val))) {
    return true;
  }

  // TODO consider callinst here

  if (auto li = dyn_cast<Instruction>(val)) {

    const IntrinsicInst *II;
    if (isa<LoadInst>(li) ||
        ((II = dyn_cast<IntrinsicInst>(li)) &&
         (II->getIntrinsicID() == Intrinsic::nvvm_ldu_global_i ||
          II->getIntrinsicID() == Intrinsic::nvvm_ldu_global_p ||
          II->getIntrinsicID() == Intrinsic::nvvm_ldu_global_f ||
          II->getIntrinsicID() == Intrinsic::nvvm_ldg_global_i ||
          II->getIntrinsicID() == Intrinsic::nvvm_ldg_global_p ||
          II->getIntrinsicID() == Intrinsic::nvvm_ldg_global_f ||
          II->getIntrinsicID() == Intrinsic::masked_load))) {
      // If this is an already unwrapped value, legal to recompute again.
      if (unwrappedLoads.find(li) != unwrappedLoads.end())
        return legalRecompute(unwrappedLoads.find(li)->second, available,
                              BuilderM, reverse);

      const Instruction *orig = nullptr;
      if (li->getParent()->getParent() == oldFunc) {
        orig = li;
      } else if (li->getParent()->getParent() == newFunc) {
        orig = isOriginal(li);
        // todo consider when we pass non original queries
        if (orig && !isa<LoadInst>(orig)) {
          return legalRecompute(orig, available, BuilderM, reverse,
                                legalRecomputeCache);
        }
      } else {
        llvm::errs() << " newFunc: " << *newFunc << "\n";
        llvm::errs() << " parent: " << *li->getParent()->getParent() << "\n";
        llvm::errs() << " li: " << *li << "\n";
        assert(0 && "illegal load legalRecopmute query");
      }

      if (orig) {
        assert(can_modref_map);
        auto found = can_modref_map->find(const_cast<Instruction *>(orig));
        if (found == can_modref_map->end()) {
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << *oldFunc << "\n";
          llvm::errs() << "can_modref_map:\n";
          for (auto &pair : *can_modref_map) {
            llvm::errs() << " + " << *pair.first << ": " << pair.second
                         << " of func "
                         << pair.first->getParent()->getParent()->getName()
                         << "\n";
          }
          llvm::errs() << "couldn't find in can_modref_map: " << *li << " - "
                       << *orig << " in fn: "
                       << orig->getParent()->getParent()->getName();
        }
        assert(found != can_modref_map->end());
        if (!found->second)
          return true;
        // if insertion block of this function:
        BasicBlock *fwdBlockIfReverse = nullptr;
        if (BuilderM) {
          fwdBlockIfReverse = BuilderM->GetInsertBlock();
          if (!reverse) {
            auto found = reverseBlockToPrimal.find(BuilderM->GetInsertBlock());
            if (found != reverseBlockToPrimal.end()) {
              fwdBlockIfReverse = found->second;
              reverse = true;
            }
          }
          if (fwdBlockIfReverse->getParent() != oldFunc)
            fwdBlockIfReverse =
                cast_or_null<BasicBlock>(isOriginal(fwdBlockIfReverse));
        }
        if (mode == DerivativeMode::ReverseModeCombined && fwdBlockIfReverse) {
          if (reverse) {
            bool failed = false;
            allFollowersOf(
                const_cast<Instruction *>(orig), [&](Instruction *I) -> bool {
                  if (I->mayWriteToMemory() &&
                      writesToMemoryReadBy(
                          OrigAA,
                          /*maybeReader*/ const_cast<Instruction *>(orig),
                          /*maybeWriter*/ I)) {
                    failed = true;
                    EmitWarning("UncacheableLoad", orig->getDebugLoc(), oldFunc,
                                orig->getParent(), "Load must be recomputed ",
                                *orig, " in reverse_",
                                BuilderM->GetInsertBlock()->getName(),
                                " due to ", *I);
                    return /*earlyBreak*/ true;
                  }
                  return /*earlyBreak*/ false;
                });
            if (!failed)
              return true;
          } else {
            Instruction *origStart = &*BuilderM->GetInsertPoint();
            do {
              if (Instruction *og = isOriginal(origStart)) {
                origStart = og;
                break;
              }
              origStart = origStart->getNextNode();
            } while (true);
            if (OrigDT.dominates(origStart, const_cast<Instruction *>(orig))) {
              bool failed = false;

              allInstructionsBetween(
                  const_cast<GradientUtils *>(this)->LI, origStart,
                  const_cast<Instruction *>(orig), [&](Instruction *I) -> bool {
                    if (I->mayWriteToMemory() &&
                        writesToMemoryReadBy(
                            OrigAA,
                            /*maybeReader*/ const_cast<Instruction *>(orig),
                            /*maybeWriter*/ I)) {
                      failed = true;
                      EmitWarning("UncacheableLoad", orig->getDebugLoc(),
                                  oldFunc, orig->getParent(),
                                  "Load must be recomputed ", *orig, " in ",
                                  BuilderM->GetInsertBlock()->getName(),
                                  " due to ", *I);
                      return /*earlyBreak*/ true;
                    }
                    return /*earlyBreak*/ false;
                  });
              if (!failed)
                return true;
            }
          }
        }
        return false;
      } else {
        if (auto dli = dyn_cast_or_null<LoadInst>(hasUninverted(li))) {
          return legalRecompute(dli, available, BuilderM, reverse);
        }

        // TODO mark all the explicitly legal nodes (caches, etc)
        return true;
        llvm::errs() << *li << " orig: " << orig
                     << " parent: " << li->getParent()->getParent()->getName()
                     << "\n";
        llvm_unreachable("unknown load to redo!");
      }
    }
  }

  if (auto ci = dyn_cast<CallInst>(val)) {
    if (auto called = ci->getCalledFunction()) {
      auto n = called->getName();
      if (n == "lgamma" || n == "lgammaf" || n == "lgammal" ||
          n == "lgamma_r" || n == "lgammaf_r" || n == "lgammal_r" ||
          n == "__lgamma_r_finite" || n == "__lgammaf_r_finite" ||
          n == "__lgammal_r_finite" || isMemFreeLibMFunction(n) ||
          n.startswith("enzyme_wrapmpi$$") || n == "omp_get_thread_num" ||
          n == "omp_get_max_threads") {
        return true;
      }
    }
  }

  if (auto inst = dyn_cast<Instruction>(val)) {
    if (inst->mayReadOrWriteMemory()) {
      return false;
    }
  }

  return true;
}

//! Given the option to recompute a value or re-use an old one, return true if
//! it is faster to recompute this value from scratch
bool GradientUtils::shouldRecompute(const Value *val,
                                    const ValueToValueMapTy &available,
                                    IRBuilder<> *BuilderM) {
  if (available.count(val))
    return true;
  // TODO: remake such that this returns whether a load to a cache is more
  // expensive than redoing the computation.

  // If this is a load from cache already, just reload this
  if (isa<LoadInst>(val) &&
      cast<LoadInst>(val)->getMetadata("enzyme_fromcache"))
    return true;

  if (!isa<Instruction>(val))
    return true;

  const Instruction *inst = cast<Instruction>(val);

  if (TapesToPreventRecomputation.count(inst))
    return false;

  if (knownRecomputeHeuristic.find(inst) != knownRecomputeHeuristic.end()) {
    return knownRecomputeHeuristic[inst];
  }
  if (auto OrigInst = isOriginal(inst)) {
    if (knownRecomputeHeuristic.find(OrigInst) !=
        knownRecomputeHeuristic.end()) {
      return knownRecomputeHeuristic[OrigInst];
    }
  }

  if (isa<CastInst>(val) || isa<GetElementPtrInst>(val))
    return true;

  if (EnzymeNewCache && !EnzymeMinCutCache) {
    // if this has operands that need to be loaded and haven't already been
    // loaded
    // TODO, just cache this
    for (auto &op : inst->operands()) {
      if (!legalRecompute(op, available, BuilderM)) {

        // If this is a load from cache already, dont force a cache of this
        if (isa<LoadInst>(op) && CacheLookups.count(cast<LoadInst>(op)))
          continue;

        // If a previously cached this operand, don't let it trigger the
        // heuristic for caching this value instead.
        if (scopeMap.find(op) != scopeMap.end())
          continue;

        // If the actually uncacheable operand is in a different loop scope
        // don't cache this value instead as it may require more memory
        LoopContext lc1;
        LoopContext lc2;
        bool inLoop1 =
            getContext(const_cast<Instruction *>(inst)->getParent(), lc1);
        bool inLoop2 = getContext(cast<Instruction>(op)->getParent(), lc2);
        if (inLoop1 != inLoop2 || (inLoop1 && (lc1.header != lc2.header))) {
          continue;
        }

        // If a placeholder phi for inversion (and we know from above not
        // recomputable)
        if (!isa<PHINode>(op) &&
            dyn_cast_or_null<LoadInst>(hasUninverted(op))) {
          goto forceCache;
        }

        // Even if cannot recompute (say a phi node), don't force a reload if it
        // is possible to just use this instruction from forward pass without
        // issue
        if (auto i2 = dyn_cast<Instruction>(op)) {
          if (!i2->mayReadOrWriteMemory()) {
            LoopContext lc;
            bool inLoop = const_cast<GradientUtils *>(this)->getContext(
                i2->getParent(), lc);
            if (!inLoop) {
              // TODO upgrade this to be all returns that this could enter from
              BasicBlock *orig = isOriginal(i2->getParent());
              assert(orig);
              bool legal = BlocksDominatingAllReturns.count(orig);
              if (legal) {
                continue;
              }
            }
          }
        }
      forceCache:;
        EmitWarning("ChosenCache", inst->getDebugLoc(), oldFunc,
                    inst->getParent(), "Choosing to cache use ", *inst,
                    " due to ", *op);
        return false;
      }
    }
  }

  if (auto op = dyn_cast<IntrinsicInst>(val)) {
    if (!op->mayReadOrWriteMemory())
      return true;
    switch (op->getIntrinsicID()) {
    case Intrinsic::sin:
    case Intrinsic::cos:
    case Intrinsic::exp:
    case Intrinsic::log:
    case Intrinsic::nvvm_ldu_global_i:
    case Intrinsic::nvvm_ldu_global_p:
    case Intrinsic::nvvm_ldu_global_f:
    case Intrinsic::nvvm_ldg_global_i:
    case Intrinsic::nvvm_ldg_global_p:
    case Intrinsic::nvvm_ldg_global_f:
      return true;
    default:
      return false;
    }
  }

  if (auto ci = dyn_cast<CallInst>(val)) {
    if (auto called = ci->getCalledFunction()) {
      auto n = called->getName();
      if (n == "lgamma" || n == "lgammaf" || n == "lgammal" ||
          n == "lgamma_r" || n == "lgammaf_r" || n == "lgammal_r" ||
          n == "__lgamma_r_finite" || n == "__lgammaf_r_finite" ||
          n == "__lgammal_r_finite" || n == "tanh" || n == "tanhf" ||
          n == "__pow_finite" || n == "__fd_sincos_1" ||
          isMemFreeLibMFunction(n) || n == "julia.pointer_from_objref" ||
          n.startswith("enzyme_wrapmpi$$") || n == "omp_get_thread_num" ||
          n == "omp_get_max_threads") {
        return true;
      }
    }
  }

  // cache a call, assuming its longer to run that
  if (isa<CallInst>(val)) {
    llvm::errs() << " caching call: " << *val << "\n";
    // cast<CallInst>(val)->getCalledFunction()->dump();
    return false;
  }

  return true;
}

GradientUtils *GradientUtils::CreateFromClone(
    EnzymeLogic &Logic, Function *todiff, TargetLibraryInfo &TLI,
    TypeAnalysis &TA, DIFFE_TYPE retType,
    const std::vector<DIFFE_TYPE> &constant_args, bool returnUsed,
    std::map<AugmentedStruct, int> &returnMapping, bool omp) {
  assert(!todiff->empty());

  // Since this is forward pass this should always return the tape (at index 0)
  returnMapping[AugmentedStruct::Tape] = 0;

  int returnCount = 0;

  if (returnUsed) {
    assert(!todiff->getReturnType()->isEmptyTy());
    assert(!todiff->getReturnType()->isVoidTy());
    returnMapping[AugmentedStruct::Return] = returnCount + 1;
    ++returnCount;
  }

  // We don't need to differentially return something that we know is not a
  // pointer (or somehow needed for shadow analysis)
  if (retType == DIFFE_TYPE::DUP_ARG || retType == DIFFE_TYPE::DUP_NONEED) {
    assert(!todiff->getReturnType()->isEmptyTy());
    assert(!todiff->getReturnType()->isVoidTy());
    assert(!todiff->getReturnType()->isFPOrFPVectorTy());
    returnMapping[AugmentedStruct::DifferentialReturn] = returnCount + 1;
    ++returnCount;
  }

  ReturnType returnValue;
  if (returnCount == 0)
    returnValue = ReturnType::Tape;
  else if (returnCount == 1)
    returnValue = ReturnType::TapeAndReturn;
  else if (returnCount == 2)
    returnValue = ReturnType::TapeAndTwoReturns;
  else
    llvm_unreachable("illegal number of elements in augmented return struct");

  ValueToValueMapTy invertedPointers;
  SmallPtrSet<Instruction *, 4> constants;
  SmallPtrSet<Instruction *, 20> nonconstant;
  SmallPtrSet<Value *, 2> returnvals;
  ValueToValueMapTy originalToNew;

  SmallPtrSet<Value *, 4> constant_values;
  SmallPtrSet<Value *, 4> nonconstant_values;

  auto newFunc = Logic.PPC.CloneFunctionWithReturns(
      DerivativeMode::ReverseModePrimal, /* width */ 1, todiff,
      invertedPointers, constant_args, constant_values, nonconstant_values,
      returnvals,
      /*returnValue*/ returnValue, retType,
      "fakeaugmented_" + todiff->getName(), &originalToNew,
      /*diffeReturnArg*/ false, /*additionalArg*/ nullptr);

  auto res = new GradientUtils(
      Logic, newFunc, todiff, TLI, TA, invertedPointers, constant_values,
      nonconstant_values, retType, originalToNew,
      DerivativeMode::ReverseModePrimal, /* width */ 1, omp);
  return res;
}

DiffeGradientUtils *DiffeGradientUtils::CreateFromClone(
    EnzymeLogic &Logic, DerivativeMode mode, unsigned width, Function *todiff,
    TargetLibraryInfo &TLI, TypeAnalysis &TA, DIFFE_TYPE retType,
    bool diffeReturnArg, const std::vector<DIFFE_TYPE> &constant_args,
    ReturnType returnValue, Type *additionalArg, bool omp) {
  assert(!todiff->empty());
  assert(mode == DerivativeMode::ReverseModeGradient ||
         mode == DerivativeMode::ReverseModeCombined ||
         mode == DerivativeMode::ForwardMode);
  ValueToValueMapTy invertedPointers;
  SmallPtrSet<Instruction *, 4> constants;
  SmallPtrSet<Instruction *, 20> nonconstant;
  SmallPtrSet<Value *, 2> returnvals;
  ValueToValueMapTy originalToNew;

  SmallPtrSet<Value *, 4> constant_values;
  SmallPtrSet<Value *, 4> nonconstant_values;

  std::string prefix;

  switch (mode) {
  case DerivativeMode::ForwardMode:
  case DerivativeMode::ForwardModeSplit:
    prefix = "fwddiffe";
    if (width > 1)
      prefix += std::to_string(width);
    break;
  case DerivativeMode::ReverseModeCombined:
  case DerivativeMode::ReverseModeGradient:
    prefix = "diffe";
    break;
  case DerivativeMode::ReverseModePrimal:
    llvm_unreachable("invalid DerivativeMode: ReverseModePrimal\n");
  }

  auto newFunc = Logic.PPC.CloneFunctionWithReturns(
      mode, width, todiff, invertedPointers, constant_args, constant_values,
      nonconstant_values, returnvals, returnValue, retType,
      prefix + todiff->getName(), &originalToNew,
      /*diffeReturnArg*/ diffeReturnArg, additionalArg);
  auto res = new DiffeGradientUtils(
      Logic, newFunc, todiff, TLI, TA, invertedPointers, constant_values,
      nonconstant_values, retType, originalToNew, mode, width, omp);
  return res;
}

Constant *GradientUtils::GetOrCreateShadowConstant(
    EnzymeLogic &Logic, TargetLibraryInfo &TLI, TypeAnalysis &TA,
    Constant *oval, DerivativeMode mode, unsigned width, bool AtomicAdd,
    bool PostOpt) {
  if (isa<ConstantPointerNull>(oval)) {
    return oval;
  } else if (isa<UndefValue>(oval)) {
    return oval;
  } else if (isa<ConstantInt>(oval)) {
    return oval;
  } else if (auto CD = dyn_cast<ConstantDataArray>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumElements(); i < len; i++) {
      Vals.push_back(
          GetOrCreateShadowConstant(Logic, TLI, TA, CD->getElementAsConstant(i),
                                    mode, width, AtomicAdd, PostOpt));
    }
    return ConstantArray::get(CD->getType(), Vals);
  } else if (auto CD = dyn_cast<ConstantArray>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Vals.push_back(GetOrCreateShadowConstant(
          Logic, TLI, TA, CD->getOperand(i), mode, width, AtomicAdd, PostOpt));
    }
    return ConstantArray::get(CD->getType(), Vals);
  } else if (auto CD = dyn_cast<ConstantStruct>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Vals.push_back(GetOrCreateShadowConstant(
          Logic, TLI, TA, CD->getOperand(i), mode, width, AtomicAdd, PostOpt));
    }
    return ConstantStruct::get(CD->getType(), Vals);
  } else if (auto CD = dyn_cast<ConstantVector>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Vals.push_back(GetOrCreateShadowConstant(
          Logic, TLI, TA, CD->getOperand(i), mode, width, AtomicAdd, PostOpt));
    }
    return ConstantVector::get(Vals);
  } else if (auto F = dyn_cast<Function>(oval)) {
    return GetOrCreateShadowFunction(Logic, TLI, TA, F, mode, width, AtomicAdd,
                                     PostOpt);
  } else if (auto arg = dyn_cast<ConstantExpr>(oval)) {
    auto C = GetOrCreateShadowConstant(Logic, TLI, TA, arg->getOperand(0), mode,
                                       width, AtomicAdd, PostOpt);
    if (arg->isCast() || arg->getOpcode() == Instruction::GetElementPtr) {
      SmallVector<Constant *, 8> NewOps;
      for (unsigned i = 0, e = arg->getNumOperands(); i != e; ++i)
        NewOps.push_back(i == 0 ? C : arg->getOperand(i));
      return arg->getWithOperands(NewOps);
    }
  } else if (auto arg = dyn_cast<GlobalVariable>(oval)) {
    if (arg->getName() == "_ZTVN10__cxxabiv120__si_class_type_infoE" ||
        arg->getName() == "_ZTVN10__cxxabiv117__class_type_infoE")
      return arg;

    if (hasMetadata(arg, "enzyme_shadow")) {
      auto md = arg->getMetadata("enzyme_shadow");
      if (!isa<MDTuple>(md)) {
        llvm::errs() << *arg << "\n";
        llvm::errs() << *md << "\n";
        assert(0 && "cannot compute with global variable that doesn't have "
                    "marked shadow global");
        report_fatal_error(
            "cannot compute with global variable that doesn't "
            "have marked shadow global (metadata incorrect type)");
      }
      auto md2 = cast<MDTuple>(md);
      assert(md2->getNumOperands() == 1);
      auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
      return gvemd->getValue();
    }

    auto Arch = llvm::Triple(arg->getParent()->getTargetTriple()).getArch();
    int SharedAddrSpace = Arch == Triple::amdgcn
                              ? (int)AMDGPU::HSAMD::AddressSpaceQualifier::Local
                              : 3;
    int AddrSpace = cast<PointerType>(arg->getType())->getAddressSpace();
    if ((Arch == Triple::nvptx || Arch == Triple::nvptx64 ||
         Arch == Triple::amdgcn) &&
        AddrSpace == SharedAddrSpace) {
      assert(0 && "shared memory not handled in meta global");
    }

    // Create global variable locally if not externally visible
    if (arg->isConstant() || arg->hasInternalLinkage() ||
        arg->hasPrivateLinkage() ||
        (arg->hasExternalLinkage() && arg->hasInitializer())) {
      Type *type = cast<PointerType>(arg->getType())->getElementType();
      auto shadow = new GlobalVariable(
          *arg->getParent(), type, arg->isConstant(), arg->getLinkage(),
          arg->getInitializer()
              ? GetOrCreateShadowConstant(Logic, TLI, TA,
                                          cast<Constant>(arg->getOperand(0)),
                                          mode, width, AtomicAdd, PostOpt)
              : Constant::getNullValue(type),
          arg->getName() + "_shadow", arg, arg->getThreadLocalMode(),
          arg->getType()->getAddressSpace(), arg->isExternallyInitialized());
      arg->setMetadata("enzyme_shadow",
                       MDTuple::get(shadow->getContext(),
                                    {ConstantAsMetadata::get(shadow)}));
#if LLVM_VERSION_MAJOR >= 11
      shadow->setAlignment(arg->getAlign());
#else
      shadow->setAlignment(arg->getAlignment());
#endif
      shadow->setUnnamedAddr(arg->getUnnamedAddr());
      return shadow;
    }
  }
  llvm::errs() << " unknown constant to create shadow of: " << *oval << "\n";
  llvm_unreachable("unknown constant to create shadow of");
}

Constant *GradientUtils::GetOrCreateShadowFunction(
    EnzymeLogic &Logic, TargetLibraryInfo &TLI, TypeAnalysis &TA, Function *fn,
    DerivativeMode mode, unsigned width, bool AtomicAdd, bool PostOpt) {
  //! Todo allow tape propagation
  //  Note that specifically this should _not_ be called with topLevel=true
  //  (since it may not be valid to always assume we can recompute the
  //  augmented primal) However, in the absence of a way to pass tape data
  //  from an indirect augmented (and also since we dont presently allow
  //  indirect augmented calls), topLevel MUST be true otherwise subcalls will
  //  not be able to lookup the augmenteddata/subdata (triggering an assertion
  //  failure, among much worse)
  bool isRealloc = false;
  if (fn->empty()) {
    if (hasMetadata(fn, "enzyme_callwrapper")) {
      auto md = fn->getMetadata("enzyme_callwrapper");
      if (!isa<MDTuple>(md)) {
        llvm::errs() << *fn << "\n";
        llvm::errs() << *md << "\n";
        assert(0 && "callwrapper of incorrect type");
        report_fatal_error("callwrapper of incorrect type");
      }
      auto md2 = cast<MDTuple>(md);
      assert(md2->getNumOperands() == 1);
      auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
      fn = cast<Function>(gvemd->getValue());
    } else {
      auto oldfn = fn;
      fn = Function::Create(oldfn->getFunctionType(), Function::InternalLinkage,
                            "callwrap_" + oldfn->getName(), oldfn->getParent());
      BasicBlock *entry = BasicBlock::Create(fn->getContext(), "entry", fn);
      IRBuilder<> B(entry);
      SmallVector<Value *, 4> args;
      for (auto &a : fn->args())
        args.push_back(&a);
      auto res = B.CreateCall(oldfn, args);
      if (fn->getReturnType()->isVoidTy())
        B.CreateRetVoid();
      else
        B.CreateRet(res);
      oldfn->setMetadata(
          "enzyme_callwrapper",
          MDTuple::get(oldfn->getContext(), {ConstantAsMetadata::get(fn)}));
      if (oldfn->getName() == "realloc")
        isRealloc = true;
    }
  }
  std::map<Argument *, bool> uncacheable_args;
  FnTypeInfo type_args(fn);
  if (isRealloc) {
    llvm::errs() << "warning: assuming realloc only creates pointers\n";
    type_args.Return.insert({-1, -1}, BaseType::Pointer);
  }

  // conservatively assume that we can only cache existing floating types
  // (i.e. that all args are uncacheable)
  std::vector<DIFFE_TYPE> types;
  for (auto &a : fn->args()) {
    uncacheable_args[&a] = !a.getType()->isFPOrFPVectorTy();
    type_args.Arguments.insert(std::pair<Argument *, TypeTree>(&a, {}));
    type_args.KnownValues.insert(
        std::pair<Argument *, std::set<int64_t>>(&a, {}));
    DIFFE_TYPE typ;
    if (a.getType()->isFPOrFPVectorTy()) {
      typ = DIFFE_TYPE::OUT_DIFF;
    } else if (a.getType()->isIntegerTy() &&
               cast<IntegerType>(a.getType())->getBitWidth() < 16) {
      typ = DIFFE_TYPE::CONSTANT;
    } else if (a.getType()->isVoidTy() || a.getType()->isEmptyTy()) {
      typ = DIFFE_TYPE::CONSTANT;
    } else {
      typ = DIFFE_TYPE::DUP_ARG;
    }
    types.push_back(typ);
  }

  DIFFE_TYPE retType = fn->getReturnType()->isFPOrFPVectorTy()
                           ? DIFFE_TYPE::OUT_DIFF
                           : DIFFE_TYPE::DUP_ARG;
  if (fn->getReturnType()->isVoidTy() || fn->getReturnType()->isEmptyTy() ||
      (fn->getReturnType()->isIntegerTy() &&
       cast<IntegerType>(fn->getReturnType())->getBitWidth() < 16))
    retType = DIFFE_TYPE::CONSTANT;

  switch (mode) {
  case DerivativeMode::ForwardMode: {
    Constant *newf =
        Logic.CreateForwardDiff(fn, retType, types, TLI, TA, false, mode, width,
                                nullptr, type_args, uncacheable_args);

    if (!newf)
      newf = UndefValue::get(fn->getType());

    std::string globalname = ("_enzyme_forward_" + fn->getName() + "'").str();
    auto GV = fn->getParent()->getNamedValue(globalname);

    if (GV == nullptr) {
      GV = new GlobalVariable(*fn->getParent(), newf->getType(), true,
                              GlobalValue::LinkageTypes::InternalLinkage, newf,
                              globalname);
    }

    return ConstantExpr::getPointerCast(GV, fn->getType());
  }
  case DerivativeMode::ReverseModeCombined:
  case DerivativeMode::ReverseModeGradient:
  case DerivativeMode::ReverseModePrimal: {
    // TODO re atomic add consider forcing it to be atomic always as fallback if
    // used in a parallel context
    auto &augdata = Logic.CreateAugmentedPrimal(
        fn, retType, /*constant_args*/ types, TLI, TA,
        /*returnUsed*/ !fn->getReturnType()->isEmptyTy() &&
            !fn->getReturnType()->isVoidTy(),
        type_args, uncacheable_args, /*forceAnonymousTape*/ true, AtomicAdd,
        PostOpt);
    Constant *newf = Logic.CreatePrimalAndGradient(
        (ReverseCacheKey){.todiff = fn,
                          .retType = retType,
                          .constant_args = types,
                          .uncacheable_args = uncacheable_args,
                          .returnUsed = false,
                          .shadowReturnUsed = false,
                          .mode = DerivativeMode::ReverseModeGradient,
                          .width = width,
                          .freeMemory = true,
                          .AtomicAdd = AtomicAdd,
                          .additionalType =
                              Type::getInt8PtrTy(fn->getContext()),
                          .typeInfo = type_args},
        TLI, TA,
        /*map*/ &augdata);
    if (!newf)
      newf = UndefValue::get(fn->getType());
    auto cdata = ConstantStruct::get(
        StructType::get(newf->getContext(),
                        {augdata.fn->getType(), newf->getType()}),
        {augdata.fn, newf});
    std::string globalname = ("_enzyme_reverse_" + fn->getName() + "'").str();
    auto GV = fn->getParent()->getNamedValue(globalname);

    if (GV == nullptr) {
      GV = new GlobalVariable(*fn->getParent(), cdata->getType(), true,
                              GlobalValue::LinkageTypes::InternalLinkage, cdata,
                              globalname);
    }
    return ConstantExpr::getPointerCast(GV, fn->getType());
  }
  default: {
    report_fatal_error("Invalid derivative mode");
  }
  }
}

Value *GradientUtils::invertPointerM(Value *const oval, IRBuilder<> &BuilderM,
                                     bool nullShadow) {
  assert(oval);
  if (auto inst = dyn_cast<Instruction>(oval)) {
    assert(inst->getParent()->getParent() == oldFunc);
  }
  if (auto arg = dyn_cast<Argument>(oval)) {
    assert(arg->getParent() == oldFunc);
  }

  if (isa<ConstantPointerNull>(oval)) {
    return oval;
  } else if (isa<UndefValue>(oval)) {
    return oval;
  } else if (isa<ConstantInt>(oval)) {
    return oval;
  } else if (auto CD = dyn_cast<ConstantDataArray>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumElements(); i < len; i++) {
      Vals.push_back(cast<Constant>(
          invertPointerM(CD->getElementAsConstant(i), BuilderM)));
    }
    return ConstantArray::get(CD->getType(), Vals);
  } else if (auto CD = dyn_cast<ConstantArray>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Value *val = invertPointerM(CD->getOperand(i), BuilderM);
      Vals.push_back(cast<Constant>(val));
    }
    return ConstantArray::get(CD->getType(), Vals);
  } else if (auto CD = dyn_cast<ConstantStruct>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Vals.push_back(
          cast<Constant>(invertPointerM(CD->getOperand(i), BuilderM)));
    }
    return ConstantStruct::get(CD->getType(), Vals);
  } else if (auto CD = dyn_cast<ConstantVector>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Vals.push_back(
          cast<Constant>(invertPointerM(CD->getOperand(i), BuilderM)));
    }
    return ConstantVector::get(Vals);
  } else if (isa<ConstantData>(oval) && nullShadow) {
    return Constant::getNullValue(oval->getType());
  }

  if (isConstantValue(oval)) {
    // NOTE, this is legal and the correct resolution, however, our activity
    // analysis honeypot no longer exists
    return getNewFromOriginal(oval);
  }
  assert(!isConstantValue(oval));

  auto M = oldFunc->getParent();
  assert(oval);

  {
    auto ifound = invertedPointers.find(oval);
    if (ifound != invertedPointers.end()) {
      return &*ifound->second;
    }
  }

  if (isa<Argument>(oval) && cast<Argument>(oval)->hasByValAttr()) {
    IRBuilder<> bb(inversionAllocs);
    AllocaInst *antialloca = bb.CreateAlloca(
        cast<PointerType>(oval->getType())->getElementType(),
        cast<PointerType>(oval->getType())->getPointerAddressSpace(), nullptr,
        oval->getName() + "'ipa");
    invertedPointers.insert(std::make_pair(
        (const Value *)oval, InvertedPointerVH(this, antialloca)));
    auto dst_arg =
        bb.CreateBitCast(antialloca, Type::getInt8PtrTy(oval->getContext()));
    auto val_arg = ConstantInt::get(Type::getInt8Ty(oval->getContext()), 0);
    auto len_arg = ConstantInt::get(
        Type::getInt64Ty(oval->getContext()),
        M->getDataLayout().getTypeAllocSizeInBits(
            cast<PointerType>(oval->getType())->getElementType()) /
            8);
    auto volatile_arg = ConstantInt::getFalse(oval->getContext());

#if LLVM_VERSION_MAJOR == 6
    auto align_arg = ConstantInt::get(Type::getInt32Ty(oval->getContext()),
                                      antialloca->getAlignment());
    Value *args[] = {dst_arg, val_arg, len_arg, align_arg, volatile_arg};
#else
    Value *args[] = {dst_arg, val_arg, len_arg, volatile_arg};
#endif
    Type *tys[] = {dst_arg->getType(), len_arg->getType()};
    cast<CallInst>(bb.CreateCall(
        Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args));
    return antialloca;
  } else if (auto arg = dyn_cast<GlobalVariable>(oval)) {
    if (!hasMetadata(arg, "enzyme_shadow")) {

      if ((mode == DerivativeMode::ReverseModeCombined ||
           mode == DerivativeMode::ForwardMode) &&
          arg->getType()->getPointerAddressSpace() == 0) {
        assert(my_TR);
        auto CT = my_TR->query(arg)[{-1, -1}];
        // Can only localy replace a global variable if it is
        // known not to contain a pointer, which may be initialized
        // outside of this function to contain other memory which
        // will not have a shadow within the current function.
        if (CT.isKnown() && CT != BaseType::Pointer) {
          bool seen = false;
          MemoryLocation
#if LLVM_VERSION_MAJOR >= 12
              Loc = MemoryLocation(oval, LocationSize::beforeOrAfterPointer());
#elif LLVM_VERSION_MAJOR >= 9
              Loc = MemoryLocation(oval, LocationSize::unknown());
#else
              Loc = MemoryLocation(oval, MemoryLocation::UnknownSize);
#endif
          for (CallInst *CI : originalCalls) {
            if (isa<IntrinsicInst>(CI))
              continue;
            if (!isConstantInstruction(CI)) {
              Function *F = getFunctionFromCall(CI);
              if (F && (isMemFreeLibMFunction(F->getName()) ||
                        F->getName() == "__fd_sincos_1")) {
                continue;
              }
              if (llvm::isModOrRefSet(OrigAA.getModRefInfo(CI, Loc))) {
                seen = true;
                llvm::errs() << " cannot shadow-inline global " << *oval
                             << " due to " << *CI << "\n";
                goto endCheck;
              }
            }
          }
        endCheck:;
          if (!seen) {
            IRBuilder<> bb(inversionAllocs);
            AllocaInst *antialloca = bb.CreateAlloca(
                arg->getValueType(), arg->getType()->getPointerAddressSpace(),
                nullptr, arg->getName() + "'ipa");
            invertedPointers.insert(std::make_pair(
                (const Value *)oval, InvertedPointerVH(this, antialloca)));

            if (arg->getAlignment()) {
#if LLVM_VERSION_MAJOR >= 10
              antialloca->setAlignment(Align(arg->getAlignment()));
#else
              antialloca->setAlignment(arg->getAlignment());
#endif
            }

            auto dst_arg = bb.CreateBitCast(
                antialloca, Type::getInt8PtrTy(arg->getContext()));
            auto val_arg =
                ConstantInt::get(Type::getInt8Ty(arg->getContext()), 0);
            auto len_arg = ConstantInt::get(
                Type::getInt64Ty(arg->getContext()),
                M->getDataLayout().getTypeAllocSizeInBits(arg->getValueType()) /
                    8);
            auto volatile_arg = ConstantInt::getFalse(oval->getContext());

#if LLVM_VERSION_MAJOR == 6
            auto align_arg =
                ConstantInt::get(Type::getInt32Ty(oval->getContext()),
                                 antialloca->getAlignment());
            Value *args[] = {dst_arg, val_arg, len_arg, align_arg,
                             volatile_arg};
#else
            Value *args[] = {dst_arg, val_arg, len_arg, volatile_arg};
#endif
            Type *tys[] = {dst_arg->getType(), len_arg->getType()};
            auto memset = cast<CallInst>(bb.CreateCall(
                Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args));
#if LLVM_VERSION_MAJOR >= 10
            if (arg->getAlignment()) {
              memset->addParamAttr(
                  0, Attribute::getWithAlignment(arg->getContext(),
                                                 Align(arg->getAlignment())));
            }
#else
            if (arg->getAlignment() != 0) {
              memset->addParamAttr(
                  0, Attribute::getWithAlignment(arg->getContext(),
                                                 arg->getAlignment()));
            }
#endif
            memset->addParamAttr(0, Attribute::NonNull);
            assert(antialloca->getType() == arg->getType());
            return antialloca;
          }
        }
      }

      auto Arch =
          llvm::Triple(newFunc->getParent()->getTargetTriple()).getArch();
      int SharedAddrSpace =
          Arch == Triple::amdgcn
              ? (int)AMDGPU::HSAMD::AddressSpaceQualifier::Local
              : 3;
      int AddrSpace = cast<PointerType>(arg->getType())->getAddressSpace();
      if ((Arch == Triple::nvptx || Arch == Triple::nvptx64 ||
           Arch == Triple::amdgcn) &&
          AddrSpace == SharedAddrSpace) {
        llvm::errs() << "warning found shared memory\n";
        //#if LLVM_VERSION_MAJOR >= 11
        Type *type = cast<PointerType>(arg->getType())->getElementType();
        // TODO this needs initialization by entry
        auto shadow = new GlobalVariable(
            *arg->getParent(), type, arg->isConstant(), arg->getLinkage(),
            UndefValue::get(type), arg->getName() + "_shadow", arg,
            arg->getThreadLocalMode(), arg->getType()->getAddressSpace(),
            arg->isExternallyInitialized());
        arg->setMetadata("enzyme_shadow",
                         MDTuple::get(shadow->getContext(),
                                      {ConstantAsMetadata::get(shadow)}));
        shadow->setMetadata("enzyme_internalshadowglobal",
                            MDTuple::get(shadow->getContext(), {}));
#if LLVM_VERSION_MAJOR >= 11
        shadow->setAlignment(arg->getAlign());
#else
        shadow->setAlignment(arg->getAlignment());
#endif
        shadow->setUnnamedAddr(arg->getUnnamedAddr());
        invertedPointers.insert(std::make_pair(
            (const Value *)oval, InvertedPointerVH(this, shadow)));
        return shadow;
      }

      // Create global variable locally if not externally visible
      if (arg->hasInternalLinkage() || arg->hasPrivateLinkage() ||
          (arg->hasExternalLinkage() && arg->hasInitializer())) {
        Type *type = cast<PointerType>(arg->getType())->getElementType();
        IRBuilder<> B(inversionAllocs);
        auto shadow = new GlobalVariable(
            *arg->getParent(), type, arg->isConstant(), arg->getLinkage(),
            arg->getInitializer()
                ? cast<Constant>(invertPointerM(arg->getInitializer(), B,
                                                /*nullShadow*/ true))
                : Constant::getNullValue(type),
            arg->getName() + "_shadow", arg, arg->getThreadLocalMode(),
            arg->getType()->getAddressSpace(), arg->isExternallyInitialized());
        arg->setMetadata("enzyme_shadow",
                         MDTuple::get(shadow->getContext(),
                                      {ConstantAsMetadata::get(shadow)}));
#if LLVM_VERSION_MAJOR >= 11
        shadow->setAlignment(arg->getAlign());
#else
        shadow->setAlignment(arg->getAlignment());
#endif
        shadow->setUnnamedAddr(arg->getUnnamedAddr());
        invertedPointers.insert(std::make_pair(
            (const Value *)oval, InvertedPointerVH(this, shadow)));
        return shadow;
      }

      llvm::errs() << *oldFunc->getParent() << "\n";
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *arg << "\n";
      assert(0 && "cannot compute with global variable that doesn't have "
                  "marked shadow global");
      report_fatal_error("cannot compute with global variable that doesn't "
                         "have marked shadow global");
    }
    auto md = arg->getMetadata("enzyme_shadow");
    if (!isa<MDTuple>(md)) {
      llvm::errs() << *arg << "\n";
      llvm::errs() << *md << "\n";
      assert(0 && "cannot compute with global variable that doesn't have "
                  "marked shadow global");
      report_fatal_error("cannot compute with global variable that doesn't "
                         "have marked shadow global (metadata incorrect type)");
    }
    auto md2 = cast<MDTuple>(md);
    assert(md2->getNumOperands() == 1);
    auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
    auto cs = gvemd->getValue();

    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, cs)));
    return cs;
  } else if (auto fn = dyn_cast<Function>(oval)) {
    return GetOrCreateShadowFunction(Logic, TLI, TA, fn, mode, width,
                                     AtomicAdd);
  } else if (auto arg = dyn_cast<CastInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *invertOp = invertPointerM(arg->getOperand(0), bb);
    Value *shadow = bb.CreateCast(arg->getOpcode(), invertOp, arg->getDestTy(),
                                  arg->getName() + "'ipc");
    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<ConstantExpr>(oval)) {
    IRBuilder<> bb(inversionAllocs);
    auto ip = invertPointerM(arg->getOperand(0), bb);
    if (arg->isCast()) {
      if (auto PT = dyn_cast<PointerType>(arg->getType())) {
        if (isConstantValue(arg->getOperand(0)) &&
            PT->getElementType()->isFunctionTy()) {
          goto end;
        }
      }
      if (auto C = dyn_cast<Constant>(ip))
        return ConstantExpr::getCast(arg->getOpcode(), C, arg->getType());
      else {
        Value *shadow =
            bb.CreateCast((Instruction::CastOps)arg->getOpcode(), ip,
                          arg->getType(), arg->getName() + "'ipc");
        invertedPointers.insert(std::make_pair(
            (const Value *)oval, InvertedPointerVH(this, shadow)));
        return shadow;
      }
    } else if (arg->getOpcode() == Instruction::GetElementPtr) {
      if (auto C = dyn_cast<Constant>(ip)) {
        SmallVector<Constant *, 8> NewOps;
        for (unsigned i = 0, e = arg->getNumOperands(); i != e; ++i)
          NewOps.push_back(i == 0 ? C : arg->getOperand(i));
        return arg->getWithOperands(NewOps);
      } else {
        SmallVector<Value *, 4> invertargs;
        for (unsigned i = 0; i < arg->getNumOperands() - 1; ++i) {
          Value *b = getNewFromOriginal(arg->getOperand(1 + i));
          invertargs.push_back(b);
        }
        // TODO mark this the same inbounds as the original
#if LLVM_VERSION_MAJOR > 7
        Value *shadow =
            bb.CreateGEP(cast<PointerType>(ip->getType())->getElementType(), ip,
                         invertargs, arg->getName() + "'ipg");
#else
        Value *shadow = bb.CreateGEP(ip, invertargs, arg->getName() + "'ipg");
#endif
        invertedPointers.insert(std::make_pair(
            (const Value *)oval, InvertedPointerVH(this, shadow)));
        return shadow;
      }
    } else {
      llvm::errs() << *arg << "\n";
      assert(0 && "unhandled");
    }
    goto end;
  } else if (auto arg = dyn_cast<ExtractValueInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *shadow =
        bb.CreateExtractValue(invertPointerM(arg->getOperand(0), bb),
                              arg->getIndices(), arg->getName() + "'ipev");
    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<InsertValueInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *shadow =
        bb.CreateInsertValue(invertPointerM(arg->getOperand(0), bb),
                             invertPointerM(arg->getOperand(1), bb),
                             arg->getIndices(), arg->getName() + "'ipiv");
    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<ExtractElementInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *shadow = bb.CreateExtractElement(
        invertPointerM(arg->getVectorOperand(), bb),
        getNewFromOriginal(arg->getIndexOperand()), arg->getName() + "'ipee");
    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<InsertElementInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *op0 = arg->getOperand(0);
    Value *op1 = arg->getOperand(1);
    Value *op2 = arg->getOperand(2);
    Value *shadow = bb.CreateInsertElement(
        invertPointerM(op0, bb), invertPointerM(op1, bb),
        getNewFromOriginal(op2), arg->getName() + "'ipie");
    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<ShuffleVectorInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *op0 = arg->getOperand(0);
    Value *op1 = arg->getOperand(1);
#if LLVM_VERSION_MAJOR >= 11
    Value *shadow = bb.CreateShuffleVector(
        invertPointerM(op0, bb), invertPointerM(op1, bb),
        arg->getShuffleMaskForBitcode(), arg->getName() + "'ipsv");
#else
    Value *shadow =
        bb.CreateShuffleVector(invertPointerM(op0, bb), invertPointerM(op1, bb),
                               arg->getOperand(2), arg->getName() + "'ipsv");
#endif
    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<SelectInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *shadow = bb.CreateSelect(getNewFromOriginal(arg->getCondition()),
                                    invertPointerM(arg->getTrueValue(), bb),
                                    invertPointerM(arg->getFalseValue(), bb),
                                    arg->getName() + "'ipse");
    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<LoadInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *op0 = arg->getOperand(0);
#if LLVM_VERSION_MAJOR > 7
    auto li = bb.CreateLoad(
        cast<PointerType>(arg->getPointerOperandType())->getElementType(),
        invertPointerM(op0, bb), arg->getName() + "'ipl");
#else
    auto li = bb.CreateLoad(invertPointerM(op0, bb), arg->getName() + "'ipl");
#endif
    li->copyIRFlags(arg);
#if LLVM_VERSION_MAJOR >= 10
    li->setAlignment(arg->getAlign());
#else
    li->setAlignment(arg->getAlignment());
#endif
    li->setDebugLoc(getNewFromOriginal(arg->getDebugLoc()));
    li->setVolatile(arg->isVolatile());
    li->setOrdering(arg->getOrdering());
    li->setSyncScopeID(arg->getSyncScopeID());
    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, li)));
    return li;
  } else if (auto arg = dyn_cast<BinaryOperator>(oval)) {
    if (arg->getOpcode() == Instruction::FAdd)
      return getNewFromOriginal(arg);

    if (!arg->getType()->isIntOrIntVectorTy()) {
      llvm::errs() << *oval << "\n";
    }
    assert(arg->getType()->isIntOrIntVectorTy());
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *val0 = nullptr;
    Value *val1 = nullptr;

    val0 = invertPointerM(arg->getOperand(0), bb);
    val1 = invertPointerM(arg->getOperand(1), bb);
    assert(val0->getType() == val1->getType());
    auto li = bb.CreateBinOp(arg->getOpcode(), val0, val1, arg->getName());
    if (auto BI = dyn_cast<BinaryOperator>(li))
      BI->copyIRFlags(arg);
    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, li)));
    return li;
  } else if (auto arg = dyn_cast<GetElementPtrInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    SmallVector<Value *, 4> invertargs;
    for (unsigned i = 0; i < arg->getNumIndices(); ++i) {
      Value *b = getNewFromOriginal(arg->getOperand(1 + i));
      invertargs.push_back(b);
    }
#if LLVM_VERSION_MAJOR > 7
    auto shadow = bb.CreateGEP(
        cast<PointerType>(arg->getPointerOperandType())->getElementType(),
        invertPointerM(arg->getPointerOperand(), bb), invertargs,
        arg->getName() + "'ipg");
#else
    auto shadow = bb.CreateGEP(invertPointerM(arg->getPointerOperand(), bb),
                               invertargs, arg->getName() + "'ipg");
#endif
    if (auto gep = dyn_cast<GetElementPtrInst>(shadow))
      gep->setIsInBounds(arg->isInBounds());
    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto inst = dyn_cast<AllocaInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(inst));
    Value *asize = getNewFromOriginal(inst->getArraySize());
    AllocaInst *antialloca = bb.CreateAlloca(
        inst->getAllocatedType(), inst->getType()->getPointerAddressSpace(),
        asize, inst->getName() + "'ipa");
    invertedPointers.insert(std::make_pair(
        (const Value *)oval, InvertedPointerVH(this, antialloca)));
    if (inst->getAlignment()) {
#if LLVM_VERSION_MAJOR >= 10
      antialloca->setAlignment(Align(inst->getAlignment()));
#else
      antialloca->setAlignment(inst->getAlignment());
#endif
    }

    if (auto ci = dyn_cast<ConstantInt>(asize)) {
      if (ci->isOne()) {
        auto st = bb.CreateStore(
            Constant::getNullValue(inst->getAllocatedType()), antialloca);
        if (inst->getAlignment()) {
#if LLVM_VERSION_MAJOR >= 10
          st->setAlignment(Align(inst->getAlignment()));
#else
          st->setAlignment(inst->getAlignment());
#endif
        }
        return antialloca;
      } else {
        // TODO handle alloca of size > 1
      }
    }

    auto dst_arg =
        bb.CreateBitCast(antialloca, Type::getInt8PtrTy(oval->getContext()));
    auto val_arg = ConstantInt::get(Type::getInt8Ty(oval->getContext()), 0);
    auto len_arg = bb.CreateMul(
        bb.CreateZExtOrTrunc(asize, Type::getInt64Ty(oval->getContext())),
        ConstantInt::get(Type::getInt64Ty(oval->getContext()),
                         M->getDataLayout().getTypeAllocSizeInBits(
                             inst->getAllocatedType()) /
                             8),
        "", true, true);
    auto volatile_arg = ConstantInt::getFalse(oval->getContext());

#if LLVM_VERSION_MAJOR == 6
    auto align_arg = ConstantInt::get(Type::getInt32Ty(oval->getContext()),
                                      antialloca->getAlignment());
    Value *args[] = {dst_arg, val_arg, len_arg, align_arg, volatile_arg};
#else
    Value *args[] = {dst_arg, val_arg, len_arg, volatile_arg};
#endif
    Type *tys[] = {dst_arg->getType(), len_arg->getType()};
    auto memset = cast<CallInst>(bb.CreateCall(
        Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args));
#if LLVM_VERSION_MAJOR >= 10
    if (inst->getAlignment()) {
      memset->addParamAttr(
          0, Attribute::getWithAlignment(inst->getContext(),
                                         Align(inst->getAlignment())));
    }
#else
    if (inst->getAlignment() != 0) {
      memset->addParamAttr(0, Attribute::getWithAlignment(
                                  inst->getContext(), inst->getAlignment()));
    }
#endif
    memset->addParamAttr(0, Attribute::NonNull);
    return antialloca;
  } else if (auto phi = dyn_cast<PHINode>(oval)) {

    if (phi->getNumIncomingValues() == 0) {
      dumpMap(invertedPointers);
      assert(0 && "illegal iv of phi");
    }
    std::map<Value *, std::set<BasicBlock *>> mapped;
    for (unsigned int i = 0; i < phi->getNumIncomingValues(); ++i) {
      mapped[phi->getIncomingValue(i)].insert(phi->getIncomingBlock(i));
    }

    if (false && mapped.size() == 1) {
      return invertPointerM(phi->getIncomingValue(0), BuilderM);
    }
#if 0
     else if (false && mapped.size() == 2) {
         IRBuilder <> bb(phi);
         auto which = bb.CreatePHI(Type::getInt1Ty(phi->getContext()), phi->getNumIncomingValues());
         //TODO this is not recursive

         int cnt = 0;
         Value* vals[2];
         for(auto v : mapped) {
            assert( cnt <= 1 );
            vals[cnt] = v.first;
            for (auto b : v.second) {
                which->addIncoming(ConstantInt::get(which->getType(), cnt), b);
            }
            ++cnt;
         }
         auto result = BuilderM.CreateSelect(which, invertPointerM(vals[1], BuilderM), invertPointerM(vals[0], BuilderM));
         return result;
     }
#endif

    else {
      auto NewV = getNewFromOriginal(phi);
      IRBuilder<> bb(NewV);
      // Note if the original phi node get's scev'd in NewF, it may
      // no longer be a phi and we need a new place to insert this phi
      // Note that if scev'd this can still be a phi with 0 incoming indicating
      // an unnecessary value to be replaced
      // TODO consider allowing the inverted pointer to become a scev
      if (!isa<PHINode>(NewV) ||
          cast<PHINode>(NewV)->getNumIncomingValues() == 0) {
        bb.SetInsertPoint(bb.GetInsertBlock(), bb.GetInsertBlock()->begin());
      }
      auto which = bb.CreatePHI(phi->getType(), phi->getNumIncomingValues());
      invertedPointers.insert(
          std::make_pair((const Value *)oval, InvertedPointerVH(this, which)));

      for (unsigned int i = 0; i < phi->getNumIncomingValues(); ++i) {
        IRBuilder<> pre(
            cast<BasicBlock>(getNewFromOriginal(phi->getIncomingBlock(i)))
                ->getTerminator());
        Value *val = invertPointerM(phi->getIncomingValue(i), pre);
        which->addIncoming(val, cast<BasicBlock>(getNewFromOriginal(
                                    phi->getIncomingBlock(i))));
      }
      return which;
    }
  }

end:;
  assert(BuilderM.GetInsertBlock());
  assert(BuilderM.GetInsertBlock()->getParent());
  assert(oval);

  llvm::errs() << *newFunc->getParent() << "\n";
  llvm::errs() << "fn:" << *newFunc << "\noval=" << *oval
               << " icv=" << isConstantValue(oval) << "\n";
  for (auto z : invertedPointers) {
    llvm::errs() << "available inversion for " << *z.first << " of "
                 << *z.second << "\n";
  }
  assert(0 && "cannot find deal with ptr that isnt arg");
  report_fatal_error("cannot find deal with ptr that isnt arg");
}

Value *GradientUtils::lookupM(Value *val, IRBuilder<> &BuilderM,
                              const ValueToValueMapTy &incoming_available,
                              bool tryLegalRecomputeCheck) {

  assert(mode == DerivativeMode::ReverseModePrimal ||
         mode == DerivativeMode::ReverseModeGradient ||
         mode == DerivativeMode::ReverseModeCombined);

  assert(val->getName() != "<badref>");
  if (isa<Constant>(val)) {
    return val;
  }
  if (isa<BasicBlock>(val)) {
    return val;
  }
  if (isa<Function>(val)) {
    return val;
  }
  if (isa<UndefValue>(val)) {
    return val;
  }
  if (isa<Argument>(val)) {
    return val;
  }
  if (isa<MetadataAsValue>(val)) {
    return val;
  }
  if (isa<InlineAsm>(val)) {
    return val;
  }

  if (!isa<Instruction>(val)) {
    llvm::errs() << *val << "\n";
  }

  auto inst = cast<Instruction>(val);
  assert(inst->getName() != "<badref>");
  if (inversionAllocs && inst->getParent() == inversionAllocs) {
    return val;
  }
  assert(inst->getParent()->getParent() == newFunc);
  assert(BuilderM.GetInsertBlock()->getParent() == newFunc);

  bool reduceRegister = false;

  if (EnzymeRegisterReduce) {
    if (auto II = dyn_cast<IntrinsicInst>(inst)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::nvvm_ldu_global_i:
      case Intrinsic::nvvm_ldu_global_p:
      case Intrinsic::nvvm_ldu_global_f:
      case Intrinsic::nvvm_ldg_global_i:
      case Intrinsic::nvvm_ldg_global_p:
      case Intrinsic::nvvm_ldg_global_f:
        reduceRegister = true;
        break;
      default:
        break;
      }
    }
    if (auto LI = dyn_cast<LoadInst>(inst)) {
      auto Arch =
          llvm::Triple(newFunc->getParent()->getTargetTriple()).getArch();
      unsigned int SharedAddrSpace =
          Arch == Triple::amdgcn
              ? (int)AMDGPU::HSAMD::AddressSpaceQualifier::Local
              : 3;
      if (cast<PointerType>(LI->getPointerOperand()->getType())
              ->getAddressSpace() == SharedAddrSpace) {
        reduceRegister |= tryLegalRecomputeCheck &&
                          legalRecompute(LI, incoming_available, &BuilderM) &&
                          shouldRecompute(LI, incoming_available, &BuilderM);
      }
    }
    if (!inst->mayReadOrWriteMemory()) {
      reduceRegister |= tryLegalRecomputeCheck &&
                        legalRecompute(inst, incoming_available, &BuilderM) &&
                        shouldRecompute(inst, incoming_available, &BuilderM);
    }
    if (this->isOriginalBlock(*BuilderM.GetInsertBlock()))
      reduceRegister = false;
  }

  if (!reduceRegister) {
    if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
      if (BuilderM.GetInsertBlock()->size() &&
          BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
        Instruction *use = &*BuilderM.GetInsertPoint();
        while (isa<PHINode>(use))
          use = use->getNextNode();
        if (DT.dominates(inst, use)) {
          return inst;
        } else {
          llvm::errs() << *BuilderM.GetInsertBlock()->getParent() << "\n";
          llvm::errs() << "didn't dominate inst: " << *inst
                       << "  point: " << *BuilderM.GetInsertPoint()
                       << "\nbb: " << *BuilderM.GetInsertBlock() << "\n";
        }
      } else {
        if (inst->getParent() == BuilderM.GetInsertBlock() ||
            DT.dominates(inst, BuilderM.GetInsertBlock())) {
          // allowed from block domination
          return inst;
        } else {
          llvm::errs() << *BuilderM.GetInsertBlock()->getParent() << "\n";
          llvm::errs() << "didn't dominate inst: " << *inst
                       << "\nbb: " << *BuilderM.GetInsertBlock() << "\n";
        }
      }
      // This is a reverse block
    } else if (BuilderM.GetInsertBlock() != inversionAllocs) {
      // Something in the entry (or anything that dominates all returns, doesn't
      // need caching)

      BasicBlock *orig = isOriginal(inst->getParent());
      assert(orig);

      // TODO upgrade this to be all returns that this could enter from
      bool legal = BlocksDominatingAllReturns.count(orig);
      if (legal) {

        BasicBlock *forwardBlock =
            isOriginal(originalForReverseBlock(*BuilderM.GetInsertBlock()));
        assert(forwardBlock);

        // Don't allow this if we're not definitely using the last iteration of
        // this value
        //   + either because the value isn't in a loop
        //   + or because the forward of the block usage location isn't in a
        //   loop (thus last iteration)
        //   + or because the loop nests share no ancestry

        bool loopLegal = true;
        for (Loop *idx = OrigLI.getLoopFor(orig); idx != nullptr;
             idx = idx->getParentLoop()) {
          for (Loop *fdx = OrigLI.getLoopFor(forwardBlock); fdx != nullptr;
               fdx = fdx->getParentLoop()) {
            if (idx == fdx) {
              loopLegal = false;
              break;
            }
          }
        }

        if (loopLegal) {
          return inst;
        }
      }
    }
  }

  if (lookup_cache[BuilderM.GetInsertBlock()].find(val) !=
      lookup_cache[BuilderM.GetInsertBlock()].end()) {
    auto result = lookup_cache[BuilderM.GetInsertBlock()][val];
    if (result == nullptr) {
      lookup_cache[BuilderM.GetInsertBlock()].erase(val);
    } else {
      assert(result);
      assert(result->getType());
      result = BuilderM.CreateBitCast(result, val->getType());
      assert(result->getType() == inst->getType());
      return result;
    }
  }

  ValueToValueMapTy available;
  for (auto pair : incoming_available) {
    assert(pair.first->getType() == pair.second->getType());
    available[pair.first] = pair.second;
  }

  {
    BasicBlock *forwardPass = BuilderM.GetInsertBlock();
    if (forwardPass != inversionAllocs && !isOriginalBlock(*forwardPass)) {
      forwardPass = originalForReverseBlock(*forwardPass);
    }
    LoopContext lc;
    bool inLoop = getContext(forwardPass, lc);

    if (inLoop) {
      bool first = true;
      for (LoopContext idx = lc;; getContext(idx.parent->getHeader(), idx)) {
        if (available.count(idx.var) == 0) {
          if (!isOriginalBlock(*BuilderM.GetInsertBlock())) {
#if LLVM_VERSION_MAJOR > 7
            available[idx.var] =
                BuilderM.CreateLoad(idx.var->getType(), idx.antivaralloc);
#else
            available[idx.var] = BuilderM.CreateLoad(idx.antivaralloc);
#endif
          } else {
            available[idx.var] = idx.var;
          }
        }
        if (!first && idx.var == inst)
          return available[idx.var];
        if (first) {
          first = false;
        }
        if (idx.parent == nullptr)
          break;
      }
    }
  }

  if (available.count(inst)) {
    assert(available[inst]->getType() == inst->getType());
    return available[inst];
  }

  // If requesting loop bound and not available from index per above
  // we must be requesting the total size. Rather than generating
  // a new lcssa variable, use the existing loop exact bound var
  {
    LoopContext lc;
    bool loopVar = false;
    if (getContext(inst->getParent(), lc) && lc.var == inst) {
      loopVar = true;
    } else if (auto phi = dyn_cast<PHINode>(inst)) {
      Value *V = nullptr;
      bool legal = true;
      for (auto &val : phi->incoming_values()) {
        if (isa<UndefValue>(val))
          continue;
        if (V == nullptr)
          V = val;
        else if (V != val) {
          legal = false;
          break;
        }
      }
      if (legal) {
        if (auto I = dyn_cast_or_null<PHINode>(V)) {
          if (getContext(I->getParent(), lc) && lc.var == I) {
            loopVar = true;
          }
        }
      }
    }
    if (loopVar) {
      Value *lim = nullptr;
      if (lc.dynamic) {
        // Must be in a reverse pass fashion for a lookup to index bound to be
        // legal
        assert(/*ReverseLimit*/ reverseBlocks.size() > 0);
        LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                          lc.preheader);
        lim =
            lookupValueFromCache(/*forwardPass*/ false, BuilderM, lctx,
                                 getDynamicLoopLimit(LI.getLoopFor(lc.header)),
                                 /*isi1*/ false);
      } else {
        lim = lookupM(lc.trueLimit, BuilderM);
      }
      lookup_cache[BuilderM.GetInsertBlock()][val] = lim;
      return lim;
    }
  }

  Instruction *prelcssaInst = inst;

  assert(inst->getName() != "<badref>");
  val = fixLCSSA(inst, BuilderM.GetInsertBlock());
  if (isa<UndefValue>(val)) {
    llvm::errs() << *oldFunc << "\n";
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << *BuilderM.GetInsertBlock() << "\n";
    llvm::errs() << *val << " inst " << *inst << "\n";
  }
  inst = cast<Instruction>(val);
  assert(prelcssaInst->getType() == inst->getType());
  assert(!this->isOriginalBlock(*BuilderM.GetInsertBlock()));

  // Update index and caching per lcssa
  if (lookup_cache[BuilderM.GetInsertBlock()].find(val) !=
      lookup_cache[BuilderM.GetInsertBlock()].end()) {
    auto result = lookup_cache[BuilderM.GetInsertBlock()][val];
    if (result == nullptr) {
      lookup_cache[BuilderM.GetInsertBlock()].erase(val);
    } else {
      assert(result);
      assert(result->getType());
      result = BuilderM.CreateBitCast(result, val->getType());
      assert(result->getType() == inst->getType());
      return result;
    }
  }

  // TODO consider call as part of
  bool lrc = false, src = false;
  if (tryLegalRecomputeCheck &&
      (lrc = legalRecompute(prelcssaInst, available, &BuilderM))) {
    if ((src = shouldRecompute(prelcssaInst, available, &BuilderM))) {
      auto op = unwrapM(prelcssaInst, BuilderM, available,
                        UnwrapMode::AttemptSingleUnwrap);
      if (op) {
        assert(op);
        assert(op->getType());
        if (auto load_op = dyn_cast<LoadInst>(prelcssaInst)) {
          if (auto new_op = dyn_cast<LoadInst>(op)) {
            MDNode *invgroup =
                load_op->getMetadata(LLVMContext::MD_invariant_group);
            if (invgroup == nullptr) {
              invgroup = MDNode::getDistinct(load_op->getContext(), {});
              load_op->setMetadata(LLVMContext::MD_invariant_group, invgroup);
            }
            new_op->setMetadata(LLVMContext::MD_invariant_group, invgroup);
          }
        }
        assert(op->getType() == inst->getType());
        if (!reduceRegister)
          lookup_cache[BuilderM.GetInsertBlock()][val] = op;
        return op;
      }
    } else {
      if (isa<LoadInst>(prelcssaInst)) {
      }
    }
  }

  if (auto li = dyn_cast<LoadInst>(inst))
    if (auto origInst = dyn_cast_or_null<LoadInst>(isOriginal(inst))) {
#if LLVM_VERSION_MAJOR >= 12
      auto liobj = getUnderlyingObject(li->getPointerOperand(), 100);
#else
      auto liobj = GetUnderlyingObject(
          li->getPointerOperand(), oldFunc->getParent()->getDataLayout(), 100);
#endif

#if LLVM_VERSION_MAJOR >= 12
      auto orig_liobj = getUnderlyingObject(origInst->getPointerOperand(), 100);
#else
      auto orig_liobj =
          GetUnderlyingObject(origInst->getPointerOperand(),
                              oldFunc->getParent()->getDataLayout(), 100);
#endif

      if (scopeMap.find(inst) == scopeMap.end()) {
        for (auto pair : scopeMap) {
          if (auto li2 = dyn_cast<LoadInst>(const_cast<Value *>(pair.first))) {

#if LLVM_VERSION_MAJOR >= 12
            auto li2obj = getUnderlyingObject(li2->getPointerOperand(), 100);
#else
            auto li2obj =
                GetUnderlyingObject(li2->getPointerOperand(),
                                    oldFunc->getParent()->getDataLayout(), 100);
#endif

            if (liobj == li2obj && DT.dominates(li2, li)) {
              auto orig2 = isOriginal(li2);
              if (!orig2)
                continue;

              bool failed = false;

              // llvm::errs() << "found potential candidate loads: oli:"
              //             << *origInst << " oli2: " << *orig2 << "\n";

              auto scev1 = SE.getSCEV(li->getPointerOperand());
              auto scev2 = SE.getSCEV(li2->getPointerOperand());
              // llvm::errs() << " scev1: " << *scev1 << " scev2: " << *scev2
              //             << "\n";

              allInstructionsBetween(
                  OrigLI, orig2, origInst, [&](Instruction *I) -> bool {
                    if (I->mayWriteToMemory() &&
                        writesToMemoryReadBy(OrigAA, /*maybeReader*/ origInst,
                                             /*maybeWriter*/ I)) {
                      failed = true;
                      // llvm::errs() << "FAILED: " << *I << "\n";
                      return /*earlyBreak*/ true;
                    }
                    return /*earlyBreak*/ false;
                  });
              if (failed)
                continue;

              if (auto ar1 = dyn_cast<SCEVAddRecExpr>(scev1)) {
                if (auto ar2 = dyn_cast<SCEVAddRecExpr>(scev2)) {
                  if (ar1->getStart() != SE.getCouldNotCompute() &&
                      ar1->getStart() == ar2->getStart() &&
                      ar1->getStepRecurrence(SE) != SE.getCouldNotCompute() &&
                      ar1->getStepRecurrence(SE) ==
                          ar2->getStepRecurrence(SE)) {

                    LoopContext l1;
                    getContext(ar1->getLoop()->getHeader(), l1);
                    LoopContext l2;
                    getContext(ar2->getLoop()->getHeader(), l2);
                    if (l1.dynamic || l2.dynamic)
                      continue;

                    // TODO IF len(ar2) >= len(ar1) then we can replace li with
                    // li2
                    if (SE.getSCEV(l1.trueLimit) != SE.getCouldNotCompute() &&
                        SE.getSCEV(l1.trueLimit) == SE.getSCEV(l2.trueLimit)) {
                      // llvm::errs()
                      //    << " step1: " << *ar1->getStepRecurrence(SE)
                      //    << " step2: " << *ar2->getStepRecurrence(SE) <<
                      //    "\n";

                      inst = li2;
                      break;
                    }
                  }
                }
              }
            }
          }
        }

        auto scev1 = OrigSE.getSCEV(origInst->getPointerOperand());

        auto Arch =
            llvm::Triple(newFunc->getParent()->getTargetTriple()).getArch();
        unsigned int SharedAddrSpace =
            Arch == Triple::amdgcn
                ? (int)AMDGPU::HSAMD::AddressSpaceQualifier::Local
                : 3;
        if (EnzymeSharedForward && scev1 != OrigSE.getCouldNotCompute() &&
            cast<PointerType>(orig_liobj->getType())->getAddressSpace() ==
                SharedAddrSpace) {
          Value *resultValue = nullptr;
          ValueToValueMapTy newavail;
          for (const auto &pair : available) {
            assert(pair.first->getType() == pair.second->getType());
            newavail[pair.first] = pair.second;
          }
          allDomPredecessorsOf(origInst, OrigDT, [&](Instruction *pred) {
            if (auto SI = dyn_cast<StoreInst>(pred)) {
              // auto NewSI = cast<StoreInst>(getNewFromOriginal(SI));
#if LLVM_VERSION_MAJOR >= 12
              auto si2obj = getUnderlyingObject(SI->getPointerOperand(), 100);
#else
              auto si2obj =
                GetUnderlyingObject(SI->getPointerOperand(),
                                    oldFunc->getParent()->getDataLayout(), 100);
#endif

              if (si2obj != orig_liobj)
                return false;

              bool lastStore = true;
              bool interveningSync = false;
              allInstructionsBetween(
                  OrigLI, SI, origInst, [&](Instruction *potentialAlias) {
                    if (!potentialAlias->mayWriteToMemory())
                      return false;
                    if (!writesToMemoryReadBy(OrigAA, origInst, potentialAlias))
                      return false;

                    if (auto II = dyn_cast<IntrinsicInst>(potentialAlias)) {
                      if (II->getIntrinsicID() == Intrinsic::nvvm_barrier0 ||
                          II->getIntrinsicID() == Intrinsic::amdgcn_s_barrier) {
                        interveningSync =
                            DT.dominates(SI, II) && DT.dominates(II, origInst);
                        allUnsyncdPredecessorsOf(
                            II,
                            [&](Instruction *mid) {
                              if (!mid->mayWriteToMemory())
                                return false;

                              if (mid == SI)
                                return false;

                              if (!writesToMemoryReadBy(OrigAA, origInst,
                                                        mid)) {
                                return false;
                              }
                              lastStore = false;
                              return true;
                            },
                            [&]() {
                              // if gone past entry
                              if (mode != DerivativeMode::ReverseModeCombined) {
                                lastStore = false;
                              }
                            });
                        if (!lastStore)
                          return true;
                        else
                          return false;
                      }
                    }

                    lastStore = false;
                    return true;
                  });

              if (!lastStore)
                return false;

              auto scev2 = OrigSE.getSCEV(SI->getPointerOperand());
              bool legal = scev1 == scev2;
              if (auto ar2 = dyn_cast<SCEVAddRecExpr>(scev2)) {
                if (auto ar1 = dyn_cast<SCEVAddRecExpr>(scev1)) {
                  if (ar2->getStart() != OrigSE.getCouldNotCompute() &&
                      ar1->getStart() == ar2->getStart() &&
                      ar2->getStepRecurrence(OrigSE) !=
                          OrigSE.getCouldNotCompute() &&
                      ar1->getStepRecurrence(OrigSE) ==
                          ar2->getStepRecurrence(OrigSE)) {

                    LoopContext l1;
                    getContext(getNewFromOriginal(ar1->getLoop()->getHeader()),
                               l1);
                    LoopContext l2;
                    getContext(getNewFromOriginal(ar2->getLoop()->getHeader()),
                               l2);
                    if (!l1.dynamic && !l2.dynamic) {
                      // TODO IF len(ar2) >= len(ar1) then we can replace li
                      // with li2
                      if (l1.trueLimit == l2.trueLimit) {
                        const Loop *L1 = ar1->getLoop();
                        while (L1) {
                          if (L1 == ar2->getLoop())
                            return false;
                          L1 = L1->getParentLoop();
                        }
                        newavail[l2.var] = available[l1.var];
                        legal = true;
                      }
                    }
                  }
                }
              }
              if (!legal) {
                Value *sval = SI->getPointerOperand();
                Value *lval = origInst->getPointerOperand();
                while (auto CI = dyn_cast<CastInst>(sval))
                  sval = CI->getOperand(0);
                while (auto CI = dyn_cast<CastInst>(lval))
                  lval = CI->getOperand(0);
                if (auto sgep = dyn_cast<GetElementPtrInst>(sval)) {
                  if (auto lgep = dyn_cast<GetElementPtrInst>(lval)) {
                    if (sgep->getPointerOperand() ==
                        lgep->getPointerOperand()) {
                      SmallVector<Value *, 3> svals;
                      for (auto &v : sgep->indices()) {
                        Value *q = v;
                        while (auto CI = dyn_cast<CastInst>(q))
                          q = CI->getOperand(0);
                        svals.push_back(q);
                      }
                      SmallVector<Value *, 3> lvals;
                      for (auto &v : lgep->indices()) {
                        Value *q = v;
                        while (auto CI = dyn_cast<CastInst>(q))
                          q = CI->getOperand(0);
                        lvals.push_back(q);
                      }
                      ValueToValueMapTy ThreadLookup;
                      bool legal = true;
                      for (size_t i = 0; i < svals.size(); i++) {
                        auto ss = OrigSE.getSCEV(svals[i]);
                        auto ls = OrigSE.getSCEV(lvals[i]);
                        if (cast<IntegerType>(ss->getType())->getBitWidth() >
                            cast<IntegerType>(ls->getType())->getBitWidth()) {
                          ls = OrigSE.getZeroExtendExpr(ls, ss->getType());
                        }
                        if (cast<IntegerType>(ss->getType())->getBitWidth() <
                            cast<IntegerType>(ls->getType())->getBitWidth()) {
                          ls = OrigSE.getTruncateExpr(ls, ss->getType());
                        }
                        if (ls != ss) {
                          if (auto II = dyn_cast<IntrinsicInst>(svals[i])) {
                            switch (II->getIntrinsicID()) {
                            case Intrinsic::nvvm_read_ptx_sreg_tid_x:
                            case Intrinsic::nvvm_read_ptx_sreg_tid_y:
                            case Intrinsic::nvvm_read_ptx_sreg_tid_z:
                            case Intrinsic::amdgcn_workitem_id_x:
                            case Intrinsic::amdgcn_workitem_id_y:
                            case Intrinsic::amdgcn_workitem_id_z:
                              ThreadLookup[getNewFromOriginal(II)] =
                                  BuilderM.CreateZExtOrTrunc(
                                      lookupM(getNewFromOriginal(lvals[i]),
                                              BuilderM, available),
                                      II->getType());
                              break;
                            default:
                              legal = false;
                              break;
                            }
                          } else {
                            legal = false;
                            break;
                          }
                        }
                      }
                      if (legal) {
                        for (auto pair : newavail) {
                          assert(pair.first->getType() ==
                                 pair.second->getType());
                          ThreadLookup[pair.first] = pair.second;
                        }
                        Value *recomp = unwrapM(
                            getNewFromOriginal(SI->getValueOperand()), BuilderM,
                            ThreadLookup, UnwrapMode::AttemptFullUnwrap,
                            /*scope*/ nullptr,
                            /*permitCache*/ false);
                        if (recomp) {
                          resultValue = recomp;
                          return true;
                          ;
                        }
                      }
                    }
                  }
                }
              }
              if (!legal)
                return false;
              return true;
            }
            return false;
          });

          if (resultValue) {
            if (resultValue->getType() != val->getType())
              resultValue = BuilderM.CreateBitCast(resultValue, val->getType());
            return resultValue;
          }
        }
      }

      auto loadSize = (li->getParent()
                           ->getParent()
                           ->getParent()
                           ->getDataLayout()
                           .getTypeAllocSizeInBits(li->getType()) +
                       7) /
                      8;

      // this is guarded because havent told cacheForReverse how to move
      if (mode == DerivativeMode::ReverseModeCombined)
        if (!li->isVolatile() && EnzymeLoopInvariantCache) {
          if (auto AI = dyn_cast<AllocaInst>(liobj)) {
            assert(isa<AllocaInst>(orig_liobj));
            if (auto AT = dyn_cast<ArrayType>(AI->getAllocatedType()))
              if (auto GEP =
                      dyn_cast<GetElementPtrInst>(li->getPointerOperand())) {
                if (GEP->getPointerOperand() == AI) {
                  LoopContext l1;
                  if (!getContext(li->getParent(), l1))
                    goto noSpeedCache;

                  BasicBlock *ctx = l1.preheader;

                  auto origPH = cast_or_null<BasicBlock>(isOriginal(ctx));
                  assert(origPH);
                  if (OrigPDT.dominates(origPH, origInst->getParent())) {
                    goto noSpeedCache;
                  }

                  Instruction *origTerm = origPH->getTerminator();
                  if (!origTerm)
                    llvm::errs() << *origTerm << "\n";
                  assert(origTerm);
                  IRBuilder<> OB(origTerm);
                  LoadInst *tmpload = OB.CreateLoad(AT, orig_liobj, "'tmpload");

                  bool failed = false;
                  allInstructionsBetween(
                      OrigLI, &*origTerm, origInst,
                      [&](Instruction *I) -> bool {
                        if (I->mayWriteToMemory() &&
                            writesToMemoryReadBy(OrigAA,
                                                 /*maybeReader*/ tmpload,
                                                 /*maybeWriter*/ I)) {
                          failed = true;
                          return /*earlyBreak*/ true;
                        }
                        return /*earlyBreak*/ false;
                      });
                  if (failed) {
                    tmpload->eraseFromParent();
                    goto noSpeedCache;
                  }
                  while (Loop *L = LI.getLoopFor(ctx)) {
                    BasicBlock *nctx = L->getLoopPreheader();
                    assert(nctx);
                    bool failed = false;
                    auto origPH = cast_or_null<BasicBlock>(isOriginal(nctx));
                    assert(origPH);
                    if (OrigPDT.dominates(origPH, origInst->getParent())) {
                      break;
                    }
                    Instruction *origTerm = origPH->getTerminator();
                    allInstructionsBetween(
                        OrigLI, &*origTerm, origInst,
                        [&](Instruction *I) -> bool {
                          if (I->mayWriteToMemory() &&
                              writesToMemoryReadBy(OrigAA,
                                                   /*maybeReader*/ tmpload,
                                                   /*maybeWriter*/ I)) {
                            failed = true;
                            return /*earlyBreak*/ true;
                          }
                          return /*earlyBreak*/ false;
                        });
                    if (failed)
                      break;
                    ctx = nctx;
                  }

                  tmpload->eraseFromParent();

                  IRBuilder<> v(ctx->getTerminator());
                  bool isi1 = false;

                  AllocaInst *cache = nullptr;

                  LoopContext tmp;
                  bool forceSingleIter = false;
                  if (!getContext(ctx, tmp)) {
                    forceSingleIter = true;
                  }
                  LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                                    ctx, forceSingleIter);

                  if (auto found = findInMap(scopeMap, (Value *)AI)) {
                    cache = found->first;
                  } else {
                    // if freeing reverseblocks must exist
                    assert(reverseBlocks.size());
                    cache = createCacheForScope(lctx, AT, li->getName(),
                                                /*shouldFree*/ true,
                                                /*allocate*/ true);
                    assert(cache);
                    scopeMap.insert(
                        std::make_pair(AI, std::make_pair(cache, lctx)));

                    v.setFastMathFlags(getFast());
                    assert(isOriginalBlock(*v.GetInsertBlock()));
                    Value *outer = getCachePointer(/*inForwardPass*/ true, v,
                                                   lctx, cache, isi1,
                                                   /*storeinstorecache*/ true);

                    auto ld = v.CreateLoad(AT, AI);
                    if (AI->getAlignment()) {
#if LLVM_VERSION_MAJOR >= 10
                      ld->setAlignment(Align(AI->getAlignment()));
#else
                      ld->setAlignment(AI->getAlignment());
#endif
                    }
                    scopeInstructions[cache].push_back(ld);
                    auto st = v.CreateStore(ld, outer);
                    auto bsize = newFunc->getParent()
                                     ->getDataLayout()
                                     .getTypeAllocSizeInBits(AT) /
                                 8;
                    if ((bsize & (bsize - 1)) == 0) {
#if LLVM_VERSION_MAJOR >= 10
                      st->setAlignment(Align(bsize));
#else
                      st->setAlignment(bsize);
#endif
                    }
                    scopeInstructions[cache].push_back(st);
                  }

                  assert(!isOriginalBlock(*BuilderM.GetInsertBlock()));
                  Value *outer = getCachePointer(/*inForwardPass*/ false,
                                                 BuilderM, lctx, cache, isi1,
                                                 /*storeinstorecache*/ true);
                  SmallVector<Value *, 2> idxs;
                  for (auto &idx : GEP->indices()) {
                    idxs.push_back(lookupM(idx, BuilderM, available,
                                           tryLegalRecomputeCheck));
                  }

#if LLVM_VERSION_MAJOR > 7
                  auto cptr = BuilderM.CreateGEP(
                      cast<PointerType>(outer->getType())->getElementType(),
                      outer, idxs);
#else
                  auto cptr = BuilderM.CreateGEP(outer, idxs);
#endif
                  cast<GetElementPtrInst>(cptr)->setIsInBounds(true);

                  // Retrieve the actual result
                  auto result = loadFromCachePointer(BuilderM, cptr, cache);

                  assert(result->getType() == inst->getType());
                  lookup_cache[BuilderM.GetInsertBlock()][val] = result;
                  return result;
                }
              }
          }

          auto scev1 = SE.getSCEV(li->getPointerOperand());
          // Store in memcpy opt
          Value *lim = nullptr;
          BasicBlock *ctx = nullptr;
          Value *start = nullptr;
          Value *offset = nullptr;
          if (auto ar1 = dyn_cast<SCEVAddRecExpr>(scev1)) {
            if (auto step =
                    dyn_cast<SCEVConstant>(ar1->getStepRecurrence(SE))) {
              if (step->getAPInt() != loadSize)
                goto noSpeedCache;

              LoopContext l1;
              getContext(ar1->getLoop()->getHeader(), l1);

              if (l1.dynamic)
                goto noSpeedCache;

              offset = available[l1.var];
              ctx = l1.preheader;

              IRBuilder<> v(ctx->getTerminator());

              auto origPH = cast_or_null<BasicBlock>(isOriginal(ctx));
              assert(origPH);
              if (OrigPDT.dominates(origPH, origInst->getParent())) {
                goto noSpeedCache;
              }

              lim = unwrapM(l1.trueLimit, v,
                            /*available*/ ValueToValueMapTy(),
                            UnwrapMode::AttemptFullUnwrapWithLookup);
              if (!lim) {
                goto noSpeedCache;
              }
              lim = v.CreateAdd(lim, ConstantInt::get(lim->getType(), 1), "",
                                true, true);

              std::vector<Instruction *> toErase;
              {
#if LLVM_VERSION_MAJOR >= 12
                SCEVExpander Exp(SE,
                                 ctx->getParent()->getParent()->getDataLayout(),
                                 "enzyme");
#else
                fake::SCEVExpander Exp(
                    SE, ctx->getParent()->getParent()->getDataLayout(),
                    "enzyme");
#endif
                Exp.setInsertPoint(l1.header->getTerminator());
                Value *start0 = Exp.expandCodeFor(
                    ar1->getStart(), li->getPointerOperand()->getType());
                start = unwrapM(start0, v,
                                /*available*/ ValueToValueMapTy(),
                                UnwrapMode::AttemptFullUnwrapWithLookup);
                std::set<Value *> todo = {start0};
                while (todo.size()) {
                  Value *now = *todo.begin();
                  todo.erase(now);
                  if (Instruction *inst = dyn_cast<Instruction>(now)) {
                    if (inst != start && inst->getNumUses() == 0 &&
                        Exp.isInsertedInstruction(inst)) {
                      for (auto &op : inst->operands()) {
                        todo.insert(op);
                      }
                      toErase.push_back(inst);
                    }
                  }
                }
              }
              for (auto a : toErase)
                erase(a);

              if (!start)
                goto noSpeedCache;

              Instruction *origTerm = origPH->getTerminator();

              bool failed = false;
              allInstructionsBetween(
                  OrigLI, &*origTerm, origInst, [&](Instruction *I) -> bool {
                    if (I->mayWriteToMemory() &&
                        writesToMemoryReadBy(OrigAA, /*maybeReader*/ origInst,
                                             /*maybeWriter*/ I)) {
                      failed = true;
                      return /*earlyBreak*/ true;
                    }
                    return /*earlyBreak*/ false;
                  });
              if (failed)
                goto noSpeedCache;
            }
          }

          if (ctx && lim && start && offset) {
            Value *firstLim = lim;
            Value *firstStart = start;
            while (Loop *L = LI.getLoopFor(ctx)) {
              BasicBlock *nctx = L->getLoopPreheader();
              assert(nctx);
              bool failed = false;
              auto origPH = cast_or_null<BasicBlock>(isOriginal(nctx));
              assert(origPH);
              if (OrigPDT.dominates(origPH, origInst->getParent())) {
                break;
              }
              Instruction *origTerm = origPH->getTerminator();
              allInstructionsBetween(
                  OrigLI, &*origTerm, origInst, [&](Instruction *I) -> bool {
                    if (I->mayWriteToMemory() &&
                        writesToMemoryReadBy(OrigAA, /*maybeReader*/ origInst,
                                             /*maybeWriter*/ I)) {
                      failed = true;
                      return /*earlyBreak*/ true;
                    }
                    return /*earlyBreak*/ false;
                  });
              if (failed)
                break;
              IRBuilder<> nv(nctx->getTerminator());
              Value *nlim = unwrapM(firstLim, nv,
                                    /*available*/ ValueToValueMapTy(),
                                    UnwrapMode::AttemptFullUnwrapWithLookup);
              if (!nlim)
                break;
              Value *nstart = unwrapM(firstStart, nv,
                                      /*available*/ ValueToValueMapTy(),
                                      UnwrapMode::AttemptFullUnwrapWithLookup);
              if (!nstart)
                break;
              lim = nlim;
              start = nstart;
              ctx = nctx;
            }
            IRBuilder<> v(ctx->getTerminator());
            bool isi1 = val->getType()->isIntegerTy() &&
                        cast<IntegerType>(li->getType())->getBitWidth() == 1;

            AllocaInst *cache = nullptr;

            LoopContext tmp;
            bool forceSingleIter = false;
            if (!getContext(ctx, tmp)) {
              forceSingleIter = true;
            } else if (auto inst = dyn_cast<Instruction>(lim)) {
              if (inst->getParent() == ctx ||
                  !DT.dominates(inst->getParent(), ctx)) {
                forceSingleIter = true;
              }
            }
            LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0, ctx,
                              forceSingleIter);

            if (auto found = findInMap(scopeMap, (Value *)inst)) {
              cache = found->first;
            } else {
              // if freeing reverseblocks must exist
              assert(reverseBlocks.size());
              cache = createCacheForScope(lctx, li->getType(), li->getName(),
                                          /*shouldFree*/ true,
                                          /*allocate*/ true, /*extraSize*/ lim);
              assert(cache);
              scopeMap.insert(
                  std::make_pair(inst, std::make_pair(cache, lctx)));

              v.setFastMathFlags(getFast());
              assert(isOriginalBlock(*v.GetInsertBlock()));
              Value *outer =
                  getCachePointer(/*inForwardPass*/ true, v, lctx, cache, isi1,
                                  /*storeinstorecache*/ true,
                                  /*extraSize*/ lim);

              auto dst_arg = v.CreateBitCast(
                  outer,
                  Type::getInt8PtrTy(
                      inst->getContext(),
                      cast<PointerType>(outer->getType())->getAddressSpace()));
              scopeInstructions[cache].push_back(cast<Instruction>(dst_arg));
              auto src_arg = v.CreateBitCast(
                  start,
                  Type::getInt8PtrTy(
                      inst->getContext(),
                      cast<PointerType>(start->getType())->getAddressSpace()));
              auto len_arg =
                  v.CreateMul(ConstantInt::get(lim->getType(), loadSize), lim,
                              "", true, true);
              if (Instruction *I = dyn_cast<Instruction>(len_arg))
                scopeInstructions[cache].push_back(I);
              auto volatile_arg = ConstantInt::getFalse(inst->getContext());

              Value *nargs[] = {dst_arg, src_arg, len_arg, volatile_arg};

              Type *tys[] = {dst_arg->getType(), src_arg->getType(),
                             len_arg->getType()};

              auto memcpyF = Intrinsic::getDeclaration(newFunc->getParent(),
                                                       Intrinsic::memcpy, tys);
              auto mem = cast<CallInst>(v.CreateCall(memcpyF, nargs));

              mem->addParamAttr(0, Attribute::NonNull);
              mem->addParamAttr(1, Attribute::NonNull);

              auto bsize =
                  newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(
                      li->getType()) /
                  8;
              if ((bsize & (bsize - 1)) == 0) {
#if LLVM_VERSION_MAJOR >= 10
                mem->addParamAttr(0, Attribute::getWithAlignment(
                                         memcpyF->getContext(), Align(bsize)));
#else
                mem->addParamAttr(0, Attribute::getWithAlignment(
                                         memcpyF->getContext(), bsize));
#endif
              }

#if LLVM_VERSION_MAJOR >= 11
              mem->addParamAttr(1, Attribute::getWithAlignment(
                                       memcpyF->getContext(), li->getAlign()));
#elif LLVM_VERSION_MAJOR >= 10
              if (li->getAlign())
                mem->addParamAttr(
                    1, Attribute::getWithAlignment(memcpyF->getContext(),
                                                   li->getAlign().getValue()));
#else
              if (li->getAlignment())
                mem->addParamAttr(
                    1, Attribute::getWithAlignment(memcpyF->getContext(),
                                                   li->getAlignment()));
#endif

              scopeInstructions[cache].push_back(mem);
            }

            assert(!isOriginalBlock(*BuilderM.GetInsertBlock()));
            Value *result = lookupValueFromCache(
                /*isForwardPass*/ false, BuilderM, lctx, cache, isi1,
                /*extraSize*/ lim, offset);
            assert(result->getType() == inst->getType());
            lookup_cache[BuilderM.GetInsertBlock()][val] = result;

            if (scopeMap.find(inst) == scopeMap.end())
              EmitWarning("Uncacheable", inst->getDebugLoc(), newFunc,
                          inst->getParent(), "Caching instruction ", *inst,
                          " legalRecompute: ", lrc, " shouldRecompute: ", src,
                          " tryLegalRecomputeCheck: ", tryLegalRecomputeCheck);
            return result;
          }
        }
    noSpeedCache:;
    }

  if (scopeMap.find(inst) == scopeMap.end())
    EmitWarning("Uncacheable", inst->getDebugLoc(), newFunc, inst->getParent(),
                "Caching instruction ", *inst, " legalRecompute: ", lrc,
                " shouldRecompute: ", src,
                " tryLegalRecomputeCheck: ", tryLegalRecomputeCheck);
  ensureLookupCached(inst);
  bool isi1 = inst->getType()->isIntegerTy() &&
              cast<IntegerType>(inst->getType())->getBitWidth() == 1;
  assert(!isOriginalBlock(*BuilderM.GetInsertBlock()));
  auto found = findInMap(scopeMap, (Value *)inst);
  Value *result = lookupValueFromCache(/*isForwardPass*/ false, BuilderM,
                                       found->second, found->first, isi1);
  if (result->getType() != inst->getType()) {
    llvm::errs() << "newFunc: " << *newFunc << "\n";
    llvm::errs() << "result: " << *result << "\n";
    llvm::errs() << "inst: " << *inst << "\n";
    llvm::errs() << "val: " << *val << "\n";
  }
  assert(result->getType() == inst->getType());
  lookup_cache[BuilderM.GetInsertBlock()][val] = result;
  assert(result);
  if (result->getType() != val->getType()) {
    result = BuilderM.CreateBitCast(result, val->getType());
  }
  assert(result->getType() == val->getType());
  assert(result->getType());
  return result;
}

//! Given a map of edges we could have taken to desired target, compute a value
//! that determines which target should be branched to
//  This function attempts to determine an equivalent condition from earlier in
//  the code and use that if possible, falling back to creating a phi node of
//  which edge was taken if necessary This function can be used in two ways:
//   * If replacePHIs is null (usual case), this function does the branch
//   * If replacePHIs isn't null, do not perform the branch and instead replace
//   the PHI's with the derived condition as to whether we should branch to a
//   particular target
void GradientUtils::branchToCorrespondingTarget(
    BasicBlock *ctx, IRBuilder<> &BuilderM,
    const std::map<BasicBlock *,
                   std::vector<std::pair</*pred*/ BasicBlock *,
                                         /*successor*/ BasicBlock *>>>
        &targetToPreds,
    const std::map<BasicBlock *, PHINode *> *replacePHIs) {
  assert(targetToPreds.size() > 0);
  if (replacePHIs) {
    if (replacePHIs->size() == 0)
      return;

    for (auto x : *replacePHIs) {
      assert(targetToPreds.find(x.first) != targetToPreds.end());
    }
  }

  if (targetToPreds.size() == 1) {
    if (replacePHIs == nullptr) {
      if (!(BuilderM.GetInsertBlock()->size() == 0 ||
            !isa<BranchInst>(BuilderM.GetInsertBlock()->back()))) {
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << *BuilderM.GetInsertBlock() << "\n";
      }
      assert(BuilderM.GetInsertBlock()->size() == 0 ||
             !isa<BranchInst>(BuilderM.GetInsertBlock()->back()));
      BuilderM.CreateBr(targetToPreds.begin()->first);
    } else {
      for (auto pair : *replacePHIs) {
        pair.second->replaceAllUsesWith(
            ConstantInt::getTrue(pair.second->getContext()));
        pair.second->eraseFromParent();
      }
    }
    return;
  }

  // Map of function edges to list of targets this can branch to we have
  std::map<std::pair</*pred*/ BasicBlock *, /*successor*/ BasicBlock *>,
           std::set<BasicBlock *>>
      done;
  {
    std::deque<
        std::tuple<std::pair</*pred*/ BasicBlock *, /*successor*/ BasicBlock *>,
                   BasicBlock *>>
        Q; // newblock, target

    for (auto pair : targetToPreds) {
      for (auto pred_edge : pair.second) {
        Q.push_back(std::make_pair(pred_edge, pair.first));
      }
    }

    for (std::tuple<
             std::pair</*pred*/ BasicBlock *, /*successor*/ BasicBlock *>,
             BasicBlock *>
             trace;
         Q.size() > 0;) {
      trace = Q.front();
      Q.pop_front();
      auto edge = std::get<0>(trace);
      auto block = edge.first;
      auto target = std::get<1>(trace);

      if (done[edge].count(target))
        continue;
      done[edge].insert(target);

      Loop *blockLoop = LI.getLoopFor(block);

      for (BasicBlock *Pred : predecessors(block)) {
        // Don't go up the backedge as we can use the last value if desired via
        // lcssa
        if (blockLoop && blockLoop->getHeader() == block &&
            blockLoop == LI.getLoopFor(Pred))
          continue;

        Q.push_back(
            std::tuple<std::pair<BasicBlock *, BasicBlock *>, BasicBlock *>(
                std::make_pair(Pred, block), target));
      }
    }
  }

  IntegerType *T = (targetToPreds.size() == 2)
                       ? Type::getInt1Ty(BuilderM.getContext())
                       : Type::getInt8Ty(BuilderM.getContext());

  Instruction *equivalentTerminator = nullptr;

  std::set<BasicBlock *> blocks;
  for (auto pair : done) {
    const auto &edge = pair.first;
    blocks.insert(edge.first);
  }

  if (targetToPreds.size() == 3) {
    for (auto block : blocks) {
      std::set<BasicBlock *> foundtargets;
      std::set<BasicBlock *> uniqueTargets;
      for (BasicBlock *succ : successors(block)) {
        auto edge = std::make_pair(block, succ);
        for (BasicBlock *target : done[edge]) {
          if (foundtargets.find(target) != foundtargets.end()) {
            goto rnextpair;
          }
          foundtargets.insert(target);
          if (done[edge].size() == 1)
            uniqueTargets.insert(target);
        }
      }
      if (foundtargets.size() != 3)
        goto rnextpair;
      if (uniqueTargets.size() != 1)
        goto rnextpair;

      {
        BasicBlock *subblock = nullptr;
        for (auto block2 : blocks) {
          std::set<BasicBlock *> seen2;
          for (BasicBlock *succ : successors(block2)) {
            auto edge = std::make_pair(block2, succ);
            if (done[edge].size() != 1) {
              // llvm::errs() << " -- failed from noonesize\n";
              goto nextblock;
            }
            for (BasicBlock *target : done[edge]) {
              if (seen2.find(target) != seen2.end()) {
                // llvm::errs() << " -- failed from not uniqueTargets\n";
                goto nextblock;
              }
              seen2.insert(target);
              if (foundtargets.find(target) == foundtargets.end()) {
                // llvm::errs() << " -- failed from not unknown target\n";
                goto nextblock;
              }
              if (uniqueTargets.find(target) != uniqueTargets.end()) {
                // llvm::errs() << " -- failed from not same target\n";
                goto nextblock;
              }
            }
          }
          if (seen2.size() != 2) {
            // llvm::errs() << " -- failed from not 2 seen\n";
            goto nextblock;
          }
          subblock = block2;
          break;
        nextblock:;
        }

        if (subblock == nullptr)
          goto rnextpair;

        if (!isa<BranchInst>(block->getTerminator()))
          goto rnextpair;

        if (!isa<BranchInst>(subblock->getTerminator()))
          goto rnextpair;

        {
          if (!isa<BranchInst>(block->getTerminator())) {
            llvm::errs() << *block << "\n";
          }
          auto bi1 = cast<BranchInst>(block->getTerminator());

          auto cond1 = lookupM(bi1->getCondition(), BuilderM);
          auto bi2 = cast<BranchInst>(subblock->getTerminator());
          auto cond2 = lookupM(bi2->getCondition(), BuilderM);

          if (replacePHIs == nullptr) {
            BasicBlock *staging =
                BasicBlock::Create(oldFunc->getContext(), "staging", newFunc);
            auto stagingIfNeeded = [&](BasicBlock *B) {
              auto edge = std::make_pair(block, B);
              if (done[edge].size() == 1) {
                return *done[edge].begin();
              } else {
                return staging;
              }
            };
            BuilderM.CreateCondBr(cond1, stagingIfNeeded(bi1->getSuccessor(0)),
                                  stagingIfNeeded(bi1->getSuccessor(1)));
            BuilderM.SetInsertPoint(staging);
            BuilderM.CreateCondBr(
                cond2,
                *done[std::make_pair(subblock, bi2->getSuccessor(0))].begin(),
                *done[std::make_pair(subblock, bi2->getSuccessor(1))].begin());
          } else {
            Value *otherBranch = nullptr;
            for (unsigned i = 0; i < 2; ++i) {
              Value *val = cond1;
              if (i == 1)
                val = BuilderM.CreateNot(val, "anot1_");
              auto edge = std::make_pair(block, bi1->getSuccessor(i));
              if (done[edge].size() == 1) {
                auto found = replacePHIs->find(*done[edge].begin());
                if (found == replacePHIs->end())
                  continue;
                if (&*BuilderM.GetInsertPoint() == found->second) {
                  if (found->second->getNextNode())
                    BuilderM.SetInsertPoint(found->second->getNextNode());
                  else
                    BuilderM.SetInsertPoint(found->second->getParent());
                }
                found->second->replaceAllUsesWith(val);
                found->second->eraseFromParent();
              } else {
                otherBranch = val;
              }
            }

            for (unsigned i = 0; i < 2; ++i) {
              auto edge = std::make_pair(subblock, bi2->getSuccessor(i));
              auto found = replacePHIs->find(*done[edge].begin());
              if (found == replacePHIs->end())
                continue;

              Value *val = cond2;
              if (i == 1)
                val = BuilderM.CreateNot(val, "bnot1_");
              val = BuilderM.CreateAnd(val, otherBranch,
                                       "andVal" + std::to_string(i));
              if (&*BuilderM.GetInsertPoint() == found->second) {
                if (found->second->getNextNode())
                  BuilderM.SetInsertPoint(found->second->getNextNode());
                else
                  BuilderM.SetInsertPoint(found->second->getParent());
              }
              found->second->replaceAllUsesWith(val);
              found->second->eraseFromParent();
            }
          }

          return;
        }
      }
    rnextpair:;
    }
  }

  BasicBlock *forwardBlock = BuilderM.GetInsertBlock();

  if (!isOriginalBlock(*forwardBlock)) {
    forwardBlock = originalForReverseBlock(*forwardBlock);
  }

  for (auto block : blocks) {
    std::set<BasicBlock *> foundtargets;
    for (BasicBlock *succ : successors(block)) {
      auto edge = std::make_pair(block, succ);
      if (done[edge].size() != 1) {
        goto nextpair;
      }
      BasicBlock *target = *done[edge].begin();
      if (foundtargets.find(target) != foundtargets.end()) {
        goto nextpair;
      }
      foundtargets.insert(target);
    }
    if (foundtargets.size() != targetToPreds.size()) {
      goto nextpair;
    }

    if (forwardBlock == block || DT.dominates(block, forwardBlock)) {
      equivalentTerminator = block->getTerminator();
      goto fast;
    }

  nextpair:;
  }
  goto nofast;

fast:;
  assert(equivalentTerminator);

  if (auto branch = dyn_cast<BranchInst>(equivalentTerminator)) {
    BasicBlock *block = equivalentTerminator->getParent();
    assert(branch->getCondition());

    assert(branch->getCondition()->getType() == T);

    if (replacePHIs == nullptr) {
      assert(BuilderM.GetInsertBlock()->size() == 0 ||
             !isa<BranchInst>(BuilderM.GetInsertBlock()->back()));
      BuilderM.CreateCondBr(
          lookupM(branch->getCondition(), BuilderM),
          *done[std::make_pair(block, branch->getSuccessor(0))].begin(),
          *done[std::make_pair(block, branch->getSuccessor(1))].begin());
    } else {
      for (auto pair : *replacePHIs) {
        Value *phi = lookupM(branch->getCondition(), BuilderM);
        Value *val = nullptr;
        if (pair.first ==
            *done[std::make_pair(block, branch->getSuccessor(0))].begin()) {
          val = phi;
        } else if (pair.first ==
                   *done[std::make_pair(block, branch->getSuccessor(1))]
                        .begin()) {
          val = BuilderM.CreateNot(phi);
        } else {
          llvm::errs() << *pair.first->getParent() << "\n";
          llvm::errs() << *pair.first << "\n";
          llvm::errs() << *branch << "\n";
          llvm_unreachable("unknown successor for replacephi");
        }
        if (&*BuilderM.GetInsertPoint() == pair.second) {
          if (pair.second->getNextNode())
            BuilderM.SetInsertPoint(pair.second->getNextNode());
          else
            BuilderM.SetInsertPoint(pair.second->getParent());
        }
        pair.second->replaceAllUsesWith(val);
        pair.second->eraseFromParent();
      }
    }
  } else if (auto si = dyn_cast<SwitchInst>(equivalentTerminator)) {
    BasicBlock *block = equivalentTerminator->getParent();

    IRBuilder<> pbuilder(equivalentTerminator);
    pbuilder.setFastMathFlags(getFast());

    if (replacePHIs == nullptr) {
      SwitchInst *swtch = BuilderM.CreateSwitch(
          lookupM(si->getCondition(), BuilderM),
          *done[std::make_pair(block, si->getDefaultDest())].begin());
      for (auto switchcase : si->cases()) {
        swtch->addCase(
            switchcase.getCaseValue(),
            *done[std::make_pair(block, switchcase.getCaseSuccessor())]
                 .begin());
      }
    } else {
      for (auto pair : *replacePHIs) {
        Value *cas = si->findCaseDest(pair.first);
        Value *val = nullptr;
        Value *phi = lookupM(si->getCondition(), BuilderM);
        if (cas) {
          val = BuilderM.CreateICmpEQ(cas, phi);
        } else {
          // default case
          val = ConstantInt::getFalse(pair.second->getContext());
          for (auto switchcase : si->cases()) {
            val = BuilderM.CreateOr(
                val, BuilderM.CreateICmpEQ(switchcase.getCaseValue(), phi));
          }
          val = BuilderM.CreateNot(val);
        }
        if (&*BuilderM.GetInsertPoint() == pair.second) {
          if (pair.second->getNextNode())
            BuilderM.SetInsertPoint(pair.second->getNextNode());
          else
            BuilderM.SetInsertPoint(pair.second->getParent());
        }
        pair.second->replaceAllUsesWith(val);
        pair.second->eraseFromParent();
      }
    }
  } else {
    llvm::errs() << "unknown equivalent terminator\n";
    llvm::errs() << *equivalentTerminator << "\n";
    llvm_unreachable("unknown equivalent terminator");
  }
  return;

nofast:;

  // if freeing reverseblocks must exist
  assert(reverseBlocks.size());
  LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0, ctx);
  AllocaInst *cache = createCacheForScope(lctx, T, "", /*shouldFree*/ true);
  std::vector<BasicBlock *> targets;
  {
    size_t idx = 0;
    std::map<BasicBlock * /*storingblock*/,
             std::map<ConstantInt * /*target*/,
                      std::vector<BasicBlock *> /*predecessors*/>>
        storing;
    for (const auto &pair : targetToPreds) {
      for (auto pred : pair.second) {
        storing[pred.first][ConstantInt::get(T, idx)].push_back(pred.second);
      }
      targets.push_back(pair.first);
      ++idx;
    }
    assert(targets.size() > 0);

    for (const auto &pair : storing) {
      IRBuilder<> pbuilder(pair.first);

      if (pair.first->getTerminator())
        pbuilder.SetInsertPoint(pair.first->getTerminator());

      pbuilder.setFastMathFlags(getFast());

      Value *tostore = ConstantInt::get(T, 0);

      if (pair.second.size() == 1) {
        tostore = pair.second.begin()->first;
      } else {
        assert(0 && "multi exit edges not supported");
        exit(1);
        // for(auto targpair : pair.second) {
        //     tostore = pbuilder.CreateOr(tostore, pred);
        //}
      }
      storeInstructionInCache(lctx, pbuilder, tostore, cache);
    }
  }

  bool isi1 = T->isIntegerTy() && cast<IntegerType>(T)->getBitWidth() == 1;
  Value *which = lookupValueFromCache(
      /*forwardPass*/ isOriginalBlock(*BuilderM.GetInsertBlock()), BuilderM,
      LimitContext(/*reversePass*/ reverseBlocks.size() > 0, ctx), cache, isi1);
  assert(which);
  assert(which->getType() == T);

  if (replacePHIs == nullptr) {
    if (targetToPreds.size() == 2) {
      assert(BuilderM.GetInsertBlock()->size() == 0 ||
             !isa<BranchInst>(BuilderM.GetInsertBlock()->back()));
      BuilderM.CreateCondBr(which, /*true*/ targets[1], /*false*/ targets[0]);
    } else {
      assert(targets.size() > 0);
      auto swit =
          BuilderM.CreateSwitch(which, targets.back(), targets.size() - 1);
      for (unsigned i = 0; i < targets.size() - 1; ++i) {
        swit->addCase(ConstantInt::get(T, i), targets[i]);
      }
    }
  } else {
    for (unsigned i = 0; i < targets.size(); ++i) {
      auto found = replacePHIs->find(targets[i]);
      if (found == replacePHIs->end())
        continue;

      Value *val = nullptr;
      if (targets.size() == 2 && i == 0) {
        val = BuilderM.CreateNot(which);
      } else if (targets.size() == 2 && i == 1) {
        val = which;
      } else {
        val = BuilderM.CreateICmpEQ(ConstantInt::get(T, i), which);
      }
      if (&*BuilderM.GetInsertPoint() == found->second) {
        if (found->second->getNextNode())
          BuilderM.SetInsertPoint(found->second->getNextNode());
        else
          BuilderM.SetInsertPoint(found->second->getParent());
      }
      found->second->replaceAllUsesWith(val);
      found->second->eraseFromParent();
    }
  }
  return;
}

void GradientUtils::computeMinCache(
    TypeResults &TR,
    const SmallPtrSetImpl<BasicBlock *> &guaranteedUnreachable) {
  if (EnzymeMinCutCache) {
    SmallPtrSet<Value *, 4> Recomputes;

    std::map<UsageKey, bool> FullSeen;
    std::map<UsageKey, bool> OneLevelSeen;

    ValueToValueMapTy Available;

    std::map<Loop *, std::set<Instruction *>> LoopAvail;

    for (BasicBlock &BB : *oldFunc) {
      if (guaranteedUnreachable.count(&BB))
        continue;
      auto L = OrigLI.getLoopFor(&BB);

      auto invariant = [&](Value *V) {
        if (isa<Constant>(V))
          return true;
        if (isa<Argument>(V))
          return true;
        if (auto I = dyn_cast<Instruction>(V)) {
          if (!L->contains(OrigLI.getLoopFor(I->getParent())))
            return true;
        }
        return false;
      };
      for (Instruction &I : BB) {
        if (auto PN = dyn_cast<PHINode>(&I)) {
          if (!OrigLI.isLoopHeader(&BB))
            continue;
          if (PN->getType()->isIntegerTy()) {
            bool legal = true;
            SmallPtrSet<Instruction *, 4> Increment;
            for (auto B : PN->blocks()) {
              if (OrigLI.getLoopFor(B) == L) {
                if (auto BO = dyn_cast<BinaryOperator>(
                        PN->getIncomingValueForBlock(B))) {
                  if (BO->getOpcode() == BinaryOperator::Add) {
                    if ((BO->getOperand(0) == PN &&
                         invariant(BO->getOperand(1))) ||
                        (BO->getOperand(1) == PN &&
                         invariant(BO->getOperand(0)))) {
                      Increment.insert(BO);
                    } else {
                      legal = false;
                    }
                  } else if (BO->getOpcode() == BinaryOperator::Sub) {
                    if (BO->getOperand(0) == PN &&
                        invariant(BO->getOperand(1))) {
                      Increment.insert(BO);
                    } else {
                      legal = false;
                    }
                  } else {
                    legal = false;
                  }
                } else {
                  legal = false;
                }
              }
            }
            if (legal) {
              LoopAvail[L].insert(PN);
              for (auto I : Increment)
                LoopAvail[L].insert(I);
            }
          }
        } else if (auto CI = dyn_cast<CallInst>(&I)) {
          Function *F = getFunctionFromCall(CI);
          if (F && isAllocationFunction(*F, TLI))
            Available[CI] = CI;
        }
      }
    }

    SmallPtrSet<Instruction *, 3> NewLoopBoundReq;
    {
      std::deque<Instruction *> LoopBoundRequirements;

      for (auto &context : loopContexts) {
        for (auto val : {context.second.maxLimit, context.second.trueLimit}) {
          if (val)
            if (auto inst = dyn_cast<Instruction>(&*val)) {
              LoopBoundRequirements.push_back(inst);
            }
        }
      }
      SmallPtrSet<Instruction *, 3> Seen;
      while (LoopBoundRequirements.size()) {
        Instruction *val = LoopBoundRequirements.front();
        LoopBoundRequirements.pop_front();
        if (NewLoopBoundReq.count(val))
          continue;
        if (Seen.count(val))
          continue;
        Seen.insert(val);
        if (auto orig = isOriginal(val)) {
          NewLoopBoundReq.insert(orig);
        } else {
          for (auto &op : val->operands()) {
            if (auto inst = dyn_cast<Instruction>(op)) {
              LoopBoundRequirements.push_back(inst);
            }
          }
        }
      }
      for (auto inst : NewLoopBoundReq) {
        OneLevelSeen[UsageKey(inst, ValueType::Primal)] = true;
        FullSeen[UsageKey(inst, ValueType::Primal)] = true;
      }
    }

    auto minCutMode = (mode == DerivativeMode::ReverseModePrimal)
                          ? DerivativeMode::ReverseModeGradient
                          : mode;

    for (BasicBlock &BB : *oldFunc) {
      if (guaranteedUnreachable.count(&BB))
        continue;
      ValueToValueMapTy Available2;
      for (auto a : Available)
        Available2[a.first] = a.second;
      for (Loop *L = OrigLI.getLoopFor(&BB); L != nullptr;
           L = L->getParentLoop()) {
        for (auto v : LoopAvail[L]) {
          Available2[v] = v;
        }
      }
      for (Instruction &I : BB) {

        if (!legalRecompute(&I, Available2, nullptr)) {
          if (is_value_needed_in_reverse<ValueType::Primal>(
                  TR, this, &I, minCutMode, FullSeen, guaranteedUnreachable)) {
            bool oneneed = is_value_needed_in_reverse<ValueType::Primal,
                                                      /*OneLevel*/ true>(
                TR, this, &I, minCutMode, OneLevelSeen, guaranteedUnreachable);
            if (oneneed) {
              knownRecomputeHeuristic[&I] = false;
            } else
              Recomputes.insert(&I);
          }
        }
      }
    }

    SmallPtrSet<Value *, 4> Intermediates;
    SmallPtrSet<Value *, 4> Required;

    Intermediates.clear();
    Required.clear();
    std::deque<Value *> todo(Recomputes.begin(), Recomputes.end());

    while (todo.size()) {
      Value *V = todo.front();
      todo.pop_front();
      if (Intermediates.count(V))
        continue;
      if (!is_value_needed_in_reverse<ValueType::Primal>(
              TR, this, V, minCutMode, FullSeen, guaranteedUnreachable)) {
        continue;
      }
      if (!Recomputes.count(V)) {
        ValueToValueMapTy Available2;
        for (auto a : Available)
          Available2[a.first] = a.second;
        for (Loop *L = OrigLI.getLoopFor(cast<Instruction>(V)->getParent());
             L != nullptr; L = L->getParentLoop()) {
          for (auto v : LoopAvail[L]) {
            Available2[v] = v;
          }
        }
        if (!legalRecompute(V, Available2, nullptr)) {
          // if not legal to recompute, we would've already explicitly marked
          // this for caching if it was needed in reverse pass
          continue;
        }
      }
      Intermediates.insert(V);
      if (is_value_needed_in_reverse<ValueType::Primal, /*OneLevel*/ true>(
              TR, this, V, minCutMode, OneLevelSeen, guaranteedUnreachable)) {
        Required.insert(V);
      } else {
        for (auto V2 : V->users()) {
          todo.push_back(V2);
        }
      }
    }

    SmallPtrSet<Value *, 5> MinReq;
    minCut(oldFunc->getParent()->getDataLayout(), OrigLI, Recomputes,
           Intermediates, Required, MinReq);
    SmallPtrSet<Value *, 5> NeedGraph;
    for (Value *V : MinReq)
      NeedGraph.insert(V);
    for (Value *V : Required)
      todo.push_back(V);
    while (todo.size()) {
      Value *V = todo.front();
      todo.pop_front();
      if (NeedGraph.count(V))
        continue;
      NeedGraph.insert(V);
      if (auto I = dyn_cast<Instruction>(V))
        for (auto &V2 : I->operands()) {
          if (Intermediates.count(V2))
            todo.push_back(V2);
        }
    }

    for (auto V : Intermediates) {
      knownRecomputeHeuristic[V] = !MinReq.count(V);
      if (!NeedGraph.count(V)) {
        unnecessaryIntermediates.insert(cast<Instruction>(V));
      }
    }
  }
}

void InvertedPointerVH::deleted() {
  llvm::errs() << *gutils->oldFunc << "\n";
  llvm::errs() << *gutils->newFunc << "\n";
  gutils->dumpPointers();
  llvm::errs() << **this << "\n";
  assert(0 && "erasing something in invertedPointers map");
}

void SubTransferHelper(GradientUtils *gutils, DerivativeMode mode,
                       Type *secretty, Intrinsic::ID intrinsic,
                       unsigned dstalign, unsigned srcalign, unsigned offset,
                       bool dstConstant, Value *shadow_dst, bool srcConstant,
                       Value *shadow_src, Value *length, Value *isVolatile,
                       llvm::CallInst *MTI, bool allowForward) {
  // TODO offset
  if (secretty) {
    // no change to forward pass if represents floats
    if (mode == DerivativeMode::ReverseModeGradient ||
        mode == DerivativeMode::ReverseModeCombined) {
      IRBuilder<> Builder2(MTI->getParent());
      gutils->getReverseBuilder(Builder2, /*original*/ true);

      // If the src is constant simply zero d_dst and don't propagate to d_src
      // (which thus == src and may be illegal)
      if (srcConstant) {
        SmallVector<Value *, 4> args;
        args.push_back(gutils->lookupM(shadow_dst, Builder2));
        if (args[0]->getType()->isIntegerTy())
          args[0] = Builder2.CreateIntToPtr(
              args[0], Type::getInt8PtrTy(MTI->getContext()));
        args.push_back(ConstantInt::get(Type::getInt8Ty(MTI->getContext()), 0));
        args.push_back(gutils->lookupM(length, Builder2));
#if LLVM_VERSION_MAJOR <= 6
        args.push_back(ConstantInt::get(Type::getInt32Ty(MTI->getContext()),
                                        max(1U, dstalign)));
#endif
        args.push_back(ConstantInt::getFalse(MTI->getContext()));

        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memsetIntr = Intrinsic::getDeclaration(
            MTI->getParent()->getParent()->getParent(), Intrinsic::memset, tys);
        auto cal = Builder2.CreateCall(memsetIntr, args);
        cal->setCallingConv(memsetIntr->getCallingConv());
        if (dstalign != 0) {
#if LLVM_VERSION_MAJOR >= 10
          cal->addParamAttr(0, Attribute::getWithAlignment(MTI->getContext(),
                                                           Align(dstalign)));
#else
          cal->addParamAttr(
              0, Attribute::getWithAlignment(MTI->getContext(), dstalign));
#endif
        }

      } else {
        SmallVector<Value *, 4> args;
        auto dsto = gutils->lookupM(shadow_dst, Builder2);
        if (dsto->getType()->isIntegerTy())
          dsto = Builder2.CreateIntToPtr(
              dsto, Type::getInt8PtrTy(dsto->getContext()));
        unsigned dstaddr =
            cast<PointerType>(dsto->getType())->getAddressSpace();
        auto secretpt = PointerType::get(secretty, dstaddr);
        if (offset != 0) {
#if LLVM_VERSION_MAJOR > 7
          dsto = Builder2.CreateConstInBoundsGEP1_64(
              cast<PointerType>(dsto->getType())->getElementType(), dsto,
              offset);
#else
          dsto = Builder2.CreateConstInBoundsGEP1_64(dsto, offset);
#endif
        }
        args.push_back(Builder2.CreatePointerCast(dsto, secretpt));
        auto srco = gutils->lookupM(shadow_src, Builder2);
        if (srco->getType()->isIntegerTy())
          srco = Builder2.CreateIntToPtr(
              srco, Type::getInt8PtrTy(srco->getContext()));
        unsigned srcaddr =
            cast<PointerType>(srco->getType())->getAddressSpace();
        secretpt = PointerType::get(secretty, srcaddr);
        if (offset != 0) {
#if LLVM_VERSION_MAJOR > 7
          srco = Builder2.CreateConstInBoundsGEP1_64(
              cast<PointerType>(srco->getType())->getElementType(), srco,
              offset);
#else
          srco = Builder2.CreateConstInBoundsGEP1_64(srco, offset);
#endif
        }
        args.push_back(Builder2.CreatePointerCast(srco, secretpt));
        args.push_back(Builder2.CreateUDiv(
            gutils->lookupM(length, Builder2),

            ConstantInt::get(length->getType(),
                             Builder2.GetInsertBlock()
                                     ->getParent()
                                     ->getParent()
                                     ->getDataLayout()
                                     .getTypeAllocSizeInBits(secretty) /
                                 8)));

        auto dmemcpy = ((intrinsic == Intrinsic::memcpy)
                            ? getOrInsertDifferentialFloatMemcpy
                            : getOrInsertDifferentialFloatMemmove)(
            *MTI->getParent()->getParent()->getParent(), secretty, dstalign,
            srcalign, dstaddr, srcaddr);
        Builder2.CreateCall(dmemcpy, args);
      }
    }
  } else {

    // if represents pointer or integer type then only need to modify forward
    // pass with the copy
    if (allowForward && (mode == DerivativeMode::ReverseModePrimal ||
                         mode == DerivativeMode::ReverseModeCombined)) {

      // It is questionable how the following case would even occur, but if
      // the dst is constant, we shouldn't do anything extra
      if (dstConstant) {
        return;
      }

      SmallVector<Value *, 4> args;
      IRBuilder<> BuilderZ(gutils->getNewFromOriginal(MTI));

      // If src is inactive, then we should copy from the regular pointer
      // (i.e. suppose we are copying constant memory representing dimensions
      // into a tensor)
      //  to ensure that the differential tensor is well formed for use
      //  OUTSIDE the derivative generation (as enzyme doesn't need this), we
      //  should also perform the copy onto the differential. Future
      //  Optimization (not implemented): If dst can never escape Enzyme code,
      //  we may omit this copy.
      // no need to update pointers, even if dst is active
      auto dsto = shadow_dst;
      if (dsto->getType()->isIntegerTy())
        dsto = BuilderZ.CreateIntToPtr(dsto,
                                       Type::getInt8PtrTy(MTI->getContext()));
      if (offset != 0) {
#if LLVM_VERSION_MAJOR > 7
        dsto = BuilderZ.CreateConstInBoundsGEP1_64(
            cast<PointerType>(dsto->getType())->getElementType(), dsto, offset);
#else
        dsto = BuilderZ.CreateConstInBoundsGEP1_64(dsto, offset);
#endif
      }
      args.push_back(dsto);
      auto srco = shadow_src;
      if (srco->getType()->isIntegerTy())
        srco = BuilderZ.CreateIntToPtr(srco,
                                       Type::getInt8PtrTy(MTI->getContext()));
      if (offset != 0) {
#if LLVM_VERSION_MAJOR > 7
        srco = BuilderZ.CreateConstInBoundsGEP1_64(
            cast<PointerType>(srco->getType())->getElementType(), srco, offset);
#else
        srco = BuilderZ.CreateConstInBoundsGEP1_64(srco, offset);
#endif
      }
      args.push_back(srco);

      args.push_back(length);
#if LLVM_VERSION_MAJOR <= 6
      args.push_back(ConstantInt::get(Type::getInt32Ty(MTI->getContext()),
                                      max(1U, min(srcalign, dstalign))));
#endif
      args.push_back(isVolatile);

      //#if LLVM_VERSION_MAJOR >= 7
      Type *tys[] = {args[0]->getType(), args[1]->getType(),
                     args[2]->getType()};
      //#else
      // Type *tys[] = {args[0]->getType(), args[1]->getType(),
      // args[2]->getType(), args[3]->getType()}; #endif

      auto memtransIntr = Intrinsic::getDeclaration(
          gutils->newFunc->getParent(), intrinsic, tys);
      auto cal = BuilderZ.CreateCall(memtransIntr, args);
      cal->setAttributes(MTI->getAttributes());
      cal->setCallingConv(memtransIntr->getCallingConv());
      cal->setTailCallKind(MTI->getTailCallKind());

      if (dstalign != 0) {
#if LLVM_VERSION_MAJOR >= 10
        cal->addParamAttr(
            0, Attribute::getWithAlignment(MTI->getContext(), Align(dstalign)));
#else
        cal->addParamAttr(
            0, Attribute::getWithAlignment(MTI->getContext(), dstalign));
#endif
      }
      if (srcalign != 0) {
#if LLVM_VERSION_MAJOR >= 10
        cal->addParamAttr(
            1, Attribute::getWithAlignment(MTI->getContext(), Align(srcalign)));
#else
        cal->addParamAttr(
            1, Attribute::getWithAlignment(MTI->getContext(), srcalign));
#endif
      }
    }
  }
}
