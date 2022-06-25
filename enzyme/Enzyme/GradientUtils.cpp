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
#include "TypeAnalysis/TBAA.h"

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
    EnzymeRuntimeActivityCheck("enzyme-runtime-activity", cl::init(false),
                               cl::Hidden,
                               cl::desc("Perform runtime activity checks"));

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

llvm::cl::opt<bool>
    EnzymeRematerialize("enzyme-rematerialize", cl::init(true), cl::Hidden,
                        cl::desc("Rematerialize allocations/shadows in the "
                                 "reverse rather than caching"));
}

unsigned int MD_ToCopy[5] = {LLVMContext::MD_dbg, LLVMContext::MD_tbaa,
                             LLVMContext::MD_tbaa_struct, LLVMContext::MD_range,
                             LLVMContext::MD_nonnull};

Value *GradientUtils::unwrapM(Value *const val, IRBuilder<> &BuilderM,
                              const ValueToValueMapTy &available,
                              UnwrapMode unwrapMode, BasicBlock *scope,
                              bool permitCache) {
  assert(val);
  assert(val->getName() != "<badref>");
  assert(val->getType());

  for (auto pair : available) {
    assert(pair.first);
    assert(pair.first->getType());
    if (pair.second) {
      assert(pair.second->getType());
      assert(pair.first->getType() == pair.second->getType());
    }
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
    if (inversionAllocs && inst->getParent() == inversionAllocs) {
      return val;
    }
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
    assert(!TapesToPreventRecomputation.count(inst));
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

  if (this->mode == DerivativeMode::ReverseModeGradient ||
      this->mode == DerivativeMode::ForwardModeSplit ||
      this->mode == DerivativeMode::ReverseModeCombined)
    if (auto inst = dyn_cast<Instruction>(val)) {
      if (inst->getParent()->getParent() == newFunc) {
        if (unwrapMode == UnwrapMode::LegalFullUnwrap &&
            this->mode != DerivativeMode::ReverseModeCombined) {
          // TODO this isOriginal is a bottleneck, the new mapping of
          // knownRecompute should be precomputed and maintained to lookup
          // instead
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
              unwrappedLoads[placeholder] = inst;
              SmallVector<Metadata *, 1> avail;
              for (auto pair : available)
                if (pair.second)
                  avail.push_back(MDNode::get(
                      placeholder->getContext(),
                      {ValueAsMetadata::get(const_cast<Value *>(pair.first)),
                       ValueAsMetadata::get(pair.second)}));
              placeholder->setMetadata(
                  "enzyme_available",
                  MDNode::get(placeholder->getContext(), avail));
              if (!permitCache)
                return placeholder;
              return unwrap_cache[BuilderM.GetInsertBlock()][idx.first]
                                 [idx.second] = placeholder;
            }
          }
        } else if (unwrapMode == UnwrapMode::AttemptFullUnwrapWithLookup) {
          // TODO this isOriginal is a bottleneck, the new mapping of
          // knownRecompute should be precomputed and maintained to lookup
          // instead
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
              if (mode == DerivativeMode::ReverseModeCombined) {
                // Don't unnecessarily cache a value if the caching
                // heuristic says we should preserve this precise (and not
                // an lcssa wrapped) value
                if (!isOriginalBlock(*BuilderM.GetInsertBlock())) {
                  Value *nval = inst;
                  if (scope)
                    nval = fixLCSSA(inst, scope);
                  if (nval == inst)
                    goto endCheck;
                }
              } else {
                // Note that this logic (original load must dominate or
                // alternatively be in the reverse block) is only valid iff when
                // applicable (here if in split mode), an uncacheable load
                // cannot be hoisted outside of a loop to be used as a loop
                // limit. This optimization is currently done in the combined
                // mode (e.g. if a load isn't modified between a prior insertion
                // point and the actual load, it is legal to recompute).
                if (!isOriginalBlock(*BuilderM.GetInsertBlock()) ||
                    DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
                  assert(inst->getParent()->getParent() == newFunc);
                  auto placeholder = BuilderM.CreatePHI(
                      val->getType(), 0,
                      val->getName() + "_krcAFUWLreplacement");
                  unwrappedLoads[placeholder] = inst;
                  SmallVector<Metadata *, 1> avail;
                  for (auto pair : available)
                    if (pair.second)
                      avail.push_back(
                          MDNode::get(placeholder->getContext(),
                                      {ValueAsMetadata::get(
                                           const_cast<Value *>(pair.first)),
                                       ValueAsMetadata::get(pair.second)}));
                  placeholder->setMetadata(
                      "enzyme_available",
                      MDNode::get(placeholder->getContext(), avail));
                  if (!permitCache)
                    return placeholder;
                  return unwrap_cache[BuilderM.GetInsertBlock()][idx.first]
                                     [idx.second] = placeholder;
                }
              }
            }
          }
        } else if (unwrapMode != UnwrapMode::LegalFullUnwrapNoTapeReplace &&
                   mode != DerivativeMode::ReverseModeCombined) {
          // TODO this isOriginal is a bottleneck, the new mapping of
          // knownRecompute should be precomputed and maintained to lookup
          // instead

          // If a given value has been chosen to be cached, do not compute the
          // operands to unwrap it if it is not legal to do so. This prevents
          // the creation of unused versions of the instruction's operand, which
          // may be assumed to never be used and thus cause an error when they
          // are inadvertantly cached.
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
        auto found = available.find(v);                                        \
        if (found != available.end() && !found->second)                        \
          noLookup = true;                                                     \
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
      auto found = available.find(v);                                          \
      assert(found == available.end() || found->second);                       \
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
        inst->getPointerOperandType()->getPointerElementType(), ptr, ind,
        inst->getName() + "_unwrap");
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
    auto toreturn =
        BuilderM.CreateLoad(pidx->getType()->getPointerElementType(), pidx,
                            load->getName() + "_unwrap");
#else
    auto toreturn = BuilderM.CreateLoad(pidx, load->getName() + "_unwrap");
#endif
    toreturn->copyMetadata(*load, MD_ToCopy);
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

    SmallVector<Value *, 4> args;
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
          pidx = lookupM(invertPointerM(dli->getOperand(0), BuilderM), BuilderM,
                         available);
        }

        if (pidx == nullptr)
          goto endCheck;

        if (pidx->getType() != getShadowType(dli->getOperand(0)->getType())) {
          llvm::errs() << "dli: " << *dli << "\n";
          llvm::errs() << "dli->getOperand(0): " << *dli->getOperand(0) << "\n";
          llvm::errs() << "pidx: " << *pidx << "\n";
        }
        assert(pidx->getType() == getShadowType(dli->getOperand(0)->getType()));

        Value *toreturn = applyChainRule(
            dli->getType(), BuilderM,
            [&](Value *pidx) {
#if LLVM_VERSION_MAJOR > 7
              auto toreturn = BuilderM.CreateLoad(dli->getType(), pidx,
                                                  phi->getName() + "_unwrap");
#else
              auto toreturn =
                  BuilderM.CreateLoad(pidx, phi->getName() + "_unwrap");
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
              return toreturn;
            },
            pidx);

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
              /*isi1*/ false, available);
          unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = lim;
          return lim;
        }
      }
    }

    auto parent = phi->getParent();

    // Don't attempt to unroll a loop induction variable in other
    // circumstances
    auto &LLI = Logic.PPC.FAM.getResult<LoopAnalysis>(*parent->getParent());
    std::set<BasicBlock *> prevIteration;
    if (LLI.isLoopHeader(parent)) {
      if (phi->getNumIncomingValues() != 2) {
        assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
        goto endCheck;
      }
      auto L = LLI.getLoopFor(parent);
      for (auto PH : predecessors(parent)) {
        if (L->contains(PH))
          prevIteration.insert(PH);
      }
      if (prevIteration.size() && !legalRecompute(phi, available, &BuilderM)) {
        assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
        goto endCheck;
      }
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

        if (DT.dominates(block, phi->getParent()))
          continue;

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

    BasicBlock *oldB = BuilderM.GetInsertBlock();
    if (BuilderM.GetInsertPoint() != oldB->end()) {
      assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
      goto endCheck;
    }

    BasicBlock *fwd = oldB;
    bool inReverseBlocks = false;
    if (!isOriginalBlock(*fwd)) {
      auto found = reverseBlockToPrimal.find(oldB);
      if (found == reverseBlockToPrimal.end()) {
        assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
        goto endCheck;
      }
      fwd = found->second;
      inReverseBlocks =
          std::find(reverseBlocks[fwd].begin(), reverseBlocks[fwd].end(),
                    oldB) != reverseBlocks[fwd].end();
    }

    auto eraseBlocks = [&](ArrayRef<BasicBlock *> blocks, BasicBlock *bret) {
      SmallVector<BasicBlock *, 2> revtopo;
      {
        SmallPtrSet<BasicBlock *, 2> seen;
        std::function<void(BasicBlock *)> dfs = [&](BasicBlock *B) {
          if (seen.count(B))
            return;
          seen.insert(B);
          if (B->getTerminator())
            for (auto S : successors(B))
              if (!seen.count(S))
                dfs(S);
          revtopo.push_back(B);
        };
        for (auto B : blocks)
          dfs(B);
        if (!seen.count(bret))
          revtopo.insert(revtopo.begin(), bret);
      }

      SmallVector<Instruction *, 4> toErase;
      for (auto B : revtopo) {
        if (B == bret)
          continue;
        for (auto &I : llvm::reverse(*B)) {
          toErase.push_back(&I);
        }
        unwrap_cache.erase(B);
        lookup_cache.erase(B);
        if (reverseBlocks.size() > 0) {
          auto tfwd = reverseBlockToPrimal[B];
          assert(tfwd);
          auto rfound = reverseBlocks.find(tfwd);
          assert(rfound != reverseBlocks.end());
          auto &tlst = rfound->second;
          auto found = std::find(tlst.begin(), tlst.end(), B);
          if (found != tlst.end())
            tlst.erase(found);
          reverseBlockToPrimal.erase(B);
        }
      }
      for (auto I : toErase) {
        erase(I);
      }
      for (auto B : revtopo)
        B->eraseFromParent();
    };

    if (targetToPreds.size() == 3) {
      for (auto block : blocks) {
        if (!DT.dominates(block, phi->getParent()))
          continue;
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
            {
              // The second split block must not have a parent with an edge
              // to a block other than to itself, which can reach any of its
              // two targets.
              // TODO verify this
              for (auto P : predecessors(block2)) {
                for (auto S : successors(P)) {
                  if (S == block2)
                    continue;
                  auto edge = std::make_pair(P, S);
                  if (done.find(edge) != done.end()) {
                    for (auto target : done[edge]) {
                      if (foundtargets.find(target) != foundtargets.end() &&
                          uniqueTargets.find(target) == uniqueTargets.end())
                        goto nextblock;
                    }
                  }
                }
              }
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
            }
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

            SmallVector<BasicBlock *, 3> predBlocks = {bi2->getSuccessor(0),
                                                       bi2->getSuccessor(1)};
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
              if (inReverseBlocks)
                reverseBlocks[fwd].push_back(blocks[i]);
              reverseBlockToPrimal[blocks[i]] = fwd;
              IRBuilder<> B(blocks[i]);

              for (auto pair : unwrap_cache[oldB])
                unwrap_cache[blocks[i]].insert(pair);
              for (auto pair : lookup_cache[oldB])
                lookup_cache[blocks[i]].insert(pair);
              auto PB = *done[std::make_pair(valparent, predBlocks[i])].begin();

              if (auto inst = dyn_cast<Instruction>(
                      phi->getIncomingValueForBlock(PB))) {
                // Recompute the phi computation with the conditional if:
                // 1) the instruction may read from memory AND does not
                //    dominate the current insertion point (thereby
                //    potentially making such recomputation without the
                //    condition illegal)
                // 2) the value is a call or load and option is set to not
                //    speculatively recompute values within a phi
                //            OR
                // 3) the value comes from a previous iteration.
                BasicBlock *nextScope = PB;
                // if (inst->getParent() == nextScope) nextScope =
                // phi->getParent();
                if (prevIteration.count(PB)) {
                  assert(0 && "tri block prev iteration unhandled");
                } else if ((inst->mayReadFromMemory() &&
                            !DT.dominates(inst->getParent(),
                                          phi->getParent())) ||
                           (!EnzymeSpeculatePHIs &&
                            (isa<CallInst>(inst) || isa<LoadInst>(inst))))
                  vals.push_back(getOpFull(B, inst, nextScope));
                else
                  vals.push_back(getOpFull(BuilderM, inst, nextScope));
              } else
                vals.push_back(
                    getOpFull(BuilderM, phi->getIncomingValueForBlock(PB), PB));

              if (!vals[i]) {
                eraseBlocks(blocks, bret);
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
            if (inReverseBlocks)
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
            for (auto pair : unwrap_cache[oldB])
              unwrap_cache[bret].insert(pair);
            for (auto pair : lookup_cache[oldB])
              lookup_cache[bret].insert(pair);
            return toret;
          }
        }
      rnextpair:;
      }
    }

    Instruction *equivalentTerminator = nullptr;

    if (prevIteration.size() == 1) {
      if (phi->getNumIncomingValues() == 2) {

        ValueToValueMapTy prevAvailable;
        for (const auto &pair : available)
          prevAvailable.insert(pair);
        LoopContext ctx;
        getContext(parent, ctx);
        Value *prevIdx;
        if (prevAvailable.count(ctx.var))
          prevIdx = prevAvailable[ctx.var];
        else {
          if (!isOriginalBlock(*BuilderM.GetInsertBlock())) {
#if LLVM_VERSION_MAJOR > 7
            prevIdx = BuilderM.CreateLoad(ctx.var->getType(), ctx.antivaralloc);
#else
            prevIdx = BuilderM.CreateLoad(ctx.antivaralloc);
#endif
          } else {
            prevIdx = ctx.var;
          }
        }
        // Prevent recursive unroll.
        prevAvailable[phi] = nullptr;
        SmallVector<Value *, 2> vals;

        SmallVector<BasicBlock *, 2> blocks;
        SmallVector<BasicBlock *, 2> endingBlocks;
        BasicBlock *last = oldB;

        BasicBlock *bret = BasicBlock::Create(
            val->getContext(), oldB->getName() + "_phimerge", newFunc);

        SmallVector<BasicBlock *, 2> preds(predecessors(phi->getParent()));

        for (auto tup : llvm::enumerate(preds)) {
          auto i = tup.index();
          BasicBlock *PB = tup.value();
          blocks.push_back(BasicBlock::Create(
              val->getContext(), oldB->getName() + "_phirc", newFunc));
          blocks[i]->moveAfter(last);
          last = blocks[i];
          if (reverseBlocks.size() > 0) {
            if (inReverseBlocks)
              reverseBlocks[fwd].push_back(blocks[i]);
            reverseBlockToPrimal[blocks[i]] = fwd;
          }
          IRBuilder<> B(blocks[i]);

          if (!prevIteration.count(PB)) {
            for (auto pair : unwrap_cache[oldB])
              unwrap_cache[blocks[i]].insert(pair);
            for (auto pair : lookup_cache[oldB])
              lookup_cache[blocks[i]].insert(pair);
          }

          if (auto inst =
                  dyn_cast<Instruction>(phi->getIncomingValueForBlock(PB))) {
            // Recompute the phi computation with the conditional if:
            // 1) the instruction may reat from memory AND does not dominate
            //    the current insertion point (thereby potentially making such
            //    recomputation without the condition illegal)
            // 2) the value is a call or load and option is set to not
            //    speculatively recompute values within a phi
            //                OR
            // 3) the value comes from a previous iteration.
            BasicBlock *nextScope = PB;
            // if (inst->getParent() == nextScope) nextScope = phi->getParent();
            if (prevIteration.count(PB)) {
              prevAvailable[ctx.incvar] = prevIdx;
              prevAvailable[ctx.var] =
                  B.CreateSub(prevIdx, ConstantInt::get(prevIdx->getType(), 1),
                              "", /*NUW*/ true, /*NSW*/ false);
              Value *___res;
              if (unwrapMode == UnwrapMode::LegalFullUnwrap ||
                  unwrapMode == UnwrapMode::LegalFullUnwrapNoTapeReplace ||
                  unwrapMode == UnwrapMode::AttemptFullUnwrap ||
                  unwrapMode == UnwrapMode::AttemptFullUnwrapWithLookup) {
                ___res = unwrapM(inst, B, prevAvailable, unwrapMode, nextScope,
                                 /*permitCache*/ false);
                if (!___res &&
                    unwrapMode == UnwrapMode::AttemptFullUnwrapWithLookup) {
                  bool noLookup = false;
                  if (isOriginalBlock(*B.GetInsertBlock())) {
                    if (!DT.dominates(inst, &*B.GetInsertPoint()))
                      noLookup = true;
                  }
                  Value *v = fixLCSSA(inst, nextScope);
                  if (!noLookup)
                    ___res = lookupM(v, B, prevAvailable, v != val);
                }
                if (___res)
                  assert(___res->getType() == inst->getType() && "uw");
              } else {
                Value *v = fixLCSSA(inst, nextScope);
                assert(unwrapMode == UnwrapMode::AttemptSingleUnwrap);
                ___res = lookupM(v, B, prevAvailable, v != val);
                if (___res && ___res->getType() != v->getType()) {
                  llvm::errs() << *newFunc << "\n";
                  llvm::errs() << " v = " << *v << " res = " << *___res << "\n";
                }
                if (___res)
                  assert(___res->getType() == inst->getType() && "lu");
              }
              vals.push_back(___res);
            } else if ((inst->mayReadFromMemory() &&
                        !DT.dominates(inst->getParent(), phi->getParent())) ||
                       (!EnzymeSpeculatePHIs &&
                        (isa<CallInst>(inst) || isa<LoadInst>(inst))))
              vals.push_back(getOpFull(B, inst, nextScope));
            else
              vals.push_back(getOpFull(BuilderM, inst, nextScope));
          } else
            vals.push_back(phi->getIncomingValueForBlock(PB));

          if (!vals[i]) {
            eraseBlocks(blocks, bret);
            assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
            goto endCheck;
          }
          assert(val->getType() == vals[i]->getType());
          B.CreateBr(bret);
          endingBlocks.push_back(B.GetInsertBlock());
        }

        // Coming from a previous iteration is equivalent to the current
        // iteration at zero.
        Value *cond;
        if (prevIteration.count(preds[0]))
          cond = BuilderM.CreateICmpNE(prevIdx,
                                       ConstantInt::get(prevIdx->getType(), 0));
        else
          cond = BuilderM.CreateICmpEQ(prevIdx,
                                       ConstantInt::get(prevIdx->getType(), 0));

        if (blocks[0]->size() == 1 && blocks[1]->size() == 1) {
          eraseBlocks(blocks, bret);
          Value *toret = BuilderM.CreateSelect(cond, vals[0], vals[1],
                                               phi->getName() + "_unwrap");
          if (permitCache) {
            unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] =
                toret;
          }
          if (auto instRet = dyn_cast<Instruction>(toret)) {
            unwrappedLoads[instRet] = val;
          }
          return toret;
        }

        bret->moveAfter(last);
        BuilderM.CreateCondBr(cond, blocks[0], blocks[1]);

        BuilderM.SetInsertPoint(bret);
        if (inReverseBlocks)
          reverseBlocks[fwd].push_back(bret);
        reverseBlockToPrimal[bret] = fwd;
        auto toret = BuilderM.CreatePHI(val->getType(), vals.size());
        for (size_t i = 0; i < vals.size(); i++)
          toret->addIncoming(vals[i], endingBlocks[i]);
        assert(val->getType() == toret->getType());
        if (permitCache) {
          unwrap_cache[bret][idx.first][idx.second] = toret;
        }
        for (auto pair : unwrap_cache[oldB])
          unwrap_cache[bret].insert(pair);
        for (auto pair : lookup_cache[oldB])
          lookup_cache[bret].insert(pair);
        unwrappedLoads[toret] = val;
        return toret;
      }
    }
    if (prevIteration.size() != 0) {
      llvm::errs() << "prev iteration: " << *phi << "\n";
      assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
      goto endCheck;
    }

    for (auto block : blocks) {
      if (!DT.dominates(block, phi->getParent()))
        continue;
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

      if (DT.dominates(block, parent)) {
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

      SmallVector<BasicBlock *, 2> predBlocks;
      Value *cond = nullptr;
      if (auto branch = dyn_cast<BranchInst>(equivalentTerminator)) {
        cond = branch->getCondition();
        predBlocks.push_back(branch->getSuccessor(0));
        predBlocks.push_back(branch->getSuccessor(1));
      } else {
        auto SI = cast<SwitchInst>(equivalentTerminator);
        cond = SI->getCondition();
        predBlocks.push_back(SI->getDefaultDest());
        for (auto scase : SI->cases()) {
          predBlocks.push_back(scase.getCaseSuccessor());
        }
      }
      cond = getOp(cond);
      if (!cond) {
        assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
        goto endCheck;
      }

      SmallVector<Value *, 2> vals;

      SmallVector<BasicBlock *, 2> blocks;
      SmallVector<BasicBlock *, 2> endingBlocks;

      BasicBlock *last = oldB;

      assert(prevIteration.size() == 0);

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
          if (inReverseBlocks)
            reverseBlocks[fwd].push_back(blocks[i]);
          reverseBlockToPrimal[blocks[i]] = fwd;
        }
        IRBuilder<> B(blocks[i]);

        for (auto pair : unwrap_cache[oldB])
          unwrap_cache[blocks[i]].insert(pair);
        for (auto pair : lookup_cache[oldB])
          lookup_cache[blocks[i]].insert(pair);

        if (auto inst =
                dyn_cast<Instruction>(phi->getIncomingValueForBlock(PB))) {
          // Recompute the phi computation with the conditional if:
          // 1) the instruction may reat from memory AND does not dominate
          //    the current insertion point (thereby potentially making such
          //    recomputation without the condition illegal)
          // 2) the value is a call or load and option is set to not
          //    speculatively recompute values within a phi
          //                OR
          // 3) the value comes from a previous iteration.
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
          eraseBlocks(blocks, bret);
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
        eraseBlocks(blocks, bret);
        Value *toret = BuilderM.CreateSelect(cond, vals[0], vals[1],
                                             phi->getName() + "_unwrap");
        if (permitCache) {
          unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] =
              toret;
        }
        if (auto instRet = dyn_cast<Instruction>(toret)) {
          unwrappedLoads[instRet] = val;
        }
        return toret;
      }

      if (BuilderM.GetInsertPoint() != oldB->end()) {
        eraseBlocks(blocks, bret);
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
      if (inReverseBlocks)
        reverseBlocks[fwd].push_back(bret);
      reverseBlockToPrimal[bret] = fwd;
      auto toret = BuilderM.CreatePHI(val->getType(), vals.size());
      for (size_t i = 0; i < vals.size(); i++)
        toret->addIncoming(vals[i], endingBlocks[i]);
      assert(val->getType() == toret->getType());
      if (permitCache) {
        unwrap_cache[bret][idx.first][idx.second] = toret;
      }
      for (auto pair : unwrap_cache[oldB])
        unwrap_cache[bret].insert(pair);
      for (auto pair : lookup_cache[oldB])
        lookup_cache[bret].insert(pair);
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
        Value *tPtr =
            BuilderQ.CreateInBoundsGEP(ret->getType()->getPointerElementType(),
                                       ret, ArrayRef<Value *>(tid));
#else
        Value *tPtr = BuilderQ.CreateInBoundsGEP(ret, ArrayRef<Value *>(tid));
#endif
        ret =
            BuilderQ.CreateLoad(ret->getType()->getPointerElementType(), tPtr);
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
        innerType = innerType->getPointerElementType();
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
      AllocaInst *cache =
          createCacheForScope(lctx, innerType, "mdyncache_fromtape",
                              ((DiffeGradientUtils *)this)->FreeMemory, false);
      assert(malloc);
      bool isi1 = !ignoreType && malloc->getType()->isIntegerTy() &&
                  cast<IntegerType>(malloc->getType())->getBitWidth() == 1;
      assert(isa<PointerType>(cache->getType()));
      assert(cache->getType()->getPointerElementType() == ret->getType());
      entryBuilder.CreateStore(ret, cache);

      auto v = lookupValueFromCache(/*forwardPass*/ true, BuilderQ, lctx, cache,
                                    isi1, /*available*/ ValueToValueMapTy());
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

          SmallVector<User *, 4> users;
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
                if (li->getType() != replacewith->getType()) {
                  llvm::errs() << " oldFunc: " << *oldFunc << "\n";
                  llvm::errs() << " newFunc: " << *newFunc << "\n";
                  llvm::errs() << " malloc: " << *malloc << "\n";
                  llvm::errs() << " li: " << *li << "\n";
                  llvm::errs() << " u: " << *u << "\n";
                  llvm::errs() << " replacewith: " << *replacewith
                               << " idx=" << idx << " - tape=" << *tape << "\n";
                }
                assert(li->getType() == replacewith->getType());
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
              if (z && z->getNumUses() == 0 && !z->isUsedByMetadata()) {
                for (unsigned i = 0; i < z->getNumOperands(); ++i) {
                  ops.push_back(z->getOperand(i));
                }
                erase(z);
              }
            }
          }

          // uses of the alloc
          SmallVector<User *, 4> users;
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

    ensureLookupCached(
        cast<Instruction>(malloc),
        /*shouldFree=*/reverseBlocks.size() > 0,
        /*scope*/ nullptr,
        cast<Instruction>(malloc)->getMetadata(LLVMContext::MD_tbaa));
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
      innerType = innerType->getPointerElementType();
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
  assert(reverseBlocks.find(branchingBlock) != reverseBlocks.end());
  LoopContext lc;
  bool inLoop = getContext(BB, lc);

  LoopContext branchingContext;
  bool inLoopContext = getContext(branchingBlock, branchingContext);

  if (!inLoop)
    return reverseBlocks[BB].front();

  auto tup = std::make_tuple(BB, branchingBlock);
  if (newBlocksForLoop_cache.find(tup) != newBlocksForLoop_cache.end())
    return newBlocksForLoop_cache[tup];

  if (inLoop) {
    // If we're reversing a latch edge.
    bool incEntering = inLoopContext && branchingBlock == lc.header &&
                       lc.header == branchingContext.header;

    auto L = LI.getLoopFor(BB);
    auto latches = getLatches(L, lc.exitBlocks);
    // If we're reverseing a loop exit.
    bool exitEntering =
        std::find(latches.begin(), latches.end(), BB) != latches.end() &&
        std::find(lc.exitBlocks.begin(), lc.exitBlocks.end(), branchingBlock) !=
            lc.exitBlocks.end();

    // If we're re-entering a loop, prepare a loop-level forward pass to
    // rematerialize any loop-scope rematerialization.
    if (incEntering || exitEntering) {
      SmallPtrSet<Instruction *, 1> loopRematerializations;
      SmallPtrSet<Instruction *, 1> loopReallocations;
      SmallPtrSet<Instruction *, 1> loopShadowReallocations;
      SmallPtrSet<Instruction *, 1> loopShadowRematerializations;
      Loop *origLI = nullptr;
      for (auto pair : rematerializableAllocations) {
        if (pair.second.LI &&
            getNewFromOriginal(pair.second.LI->getHeader()) == L->getHeader()) {
          bool rematerialized = false;
          std::map<UsageKey, bool> Seen;
          for (auto pair : knownRecomputeHeuristic)
            if (!pair.second)
              Seen[UsageKey(pair.first, ValueType::Primal)] = false;

          if (is_value_needed_in_reverse<ValueType::Primal>(
                  this, pair.first, mode, Seen, notForAnalysis)) {
            rematerialized = true;
          }
          if (rematerialized) {
            if (auto inst = dyn_cast<Instruction>(pair.first))
              if (pair.second.LI->contains(inst->getParent())) {
                loopReallocations.insert(inst);
              }
            for (auto I : pair.second.stores)
              loopRematerializations.insert(I);
            origLI = pair.second.LI;
          }
        }
      }
      for (auto pair : backwardsOnlyShadows) {
        if (pair.second.LI &&
            getNewFromOriginal(pair.second.LI->getHeader()) == L->getHeader()) {
          if (!pair.second.primalInitialize) {
            if (auto inst = dyn_cast<Instruction>(pair.first)) {
              if (pair.second.LI->contains(inst->getParent())) {
                loopShadowReallocations.insert(inst);
                for (auto I : pair.second.stores)
                  loopShadowRematerializations.insert(I);
                origLI = pair.second.LI;
              }
            }
          }
        }
      }
      BasicBlock *resumeblock = reverseBlocks[BB].front();
      if (loopRematerializations.size() != 0 || loopReallocations.size() != 0 ||
          loopShadowRematerializations.size() != 0 ||
          loopShadowReallocations.size() != 0) {
        auto found = rematerializedLoops_cache.find(L);
        if (found != rematerializedLoops_cache.end()) {
          resumeblock = found->second;
        } else {
          BasicBlock *enterB = BasicBlock::Create(
              BB->getContext(), "remat_enter", BB->getParent());
          rematerializedLoops_cache[L] = enterB;
          std::map<BasicBlock *, BasicBlock *> origToNewForward;
          for (auto B : origLI->getBlocks()) {
            BasicBlock *newB = BasicBlock::Create(
                B->getContext(),
                "remat_" + lc.header->getName() + "_" + B->getName(),
                BB->getParent());
            origToNewForward[B] = newB;
            reverseBlockToPrimal[newB] = getNewFromOriginal(B);
          }

          ValueToValueMapTy available;

          {
            IRBuilder<> NB(enterB);
            NB.CreateBr(origToNewForward[origLI->getHeader()]);
          }

          std::function<void(Loop *, bool)> handleLoop = [&](Loop *OL,
                                                             bool subLoop) {
            if (subLoop) {
              auto Header = OL->getHeader();
              IRBuilder<> NB(origToNewForward[Header]);
              LoopContext flc;
              getContext(getNewFromOriginal(Header), flc);

              auto iv = NB.CreatePHI(flc.var->getType(), 2, "fiv");
              auto inc = NB.CreateAdd(iv, ConstantInt::get(iv->getType(), 1));

              for (auto PH : predecessors(Header)) {
                if (notForAnalysis.count(PH))
                  continue;

                if (OL->contains(PH))
                  iv->addIncoming(inc, origToNewForward[PH]);
                else
                  iv->addIncoming(ConstantInt::get(iv->getType(), 0),
                                  origToNewForward[PH]);
              }
              available[flc.var] = iv;
              available[flc.incvar] = inc;
            }
            for (auto SL : OL->getSubLoops())
              handleLoop(SL, /*subLoop*/ true);
          };
          handleLoop(origLI, /*subLoop*/ false);

          for (auto B : origLI->getBlocks()) {
            auto newB = origToNewForward[B];
            IRBuilder<> NB(newB);

            // TODO fill available with relevant IV's surrounding and
            // IV's of inner loop phi's

            for (auto &I : *B) {
              // Only handle store, memset, and julia.write_barrier
              if (loopRematerializations.count(&I)) {
                if (auto SI = dyn_cast<StoreInst>(&I)) {
                  auto ts = NB.CreateStore(
                      lookupM(getNewFromOriginal(SI->getValueOperand()), NB,
                              available),
                      lookupM(getNewFromOriginal(SI->getPointerOperand()), NB,
                              available));
                  ts->copyMetadata(*SI, MD_ToCopy);
#if LLVM_VERSION_MAJOR >= 10
                  ts->setAlignment(SI->getAlign());
#else
                  ts->setAlignment(SI->getAlignment());
#endif
                  ts->setVolatile(SI->isVolatile());
                  ts->setOrdering(SI->getOrdering());
                  ts->setSyncScopeID(SI->getSyncScopeID());
                  ts->setDebugLoc(getNewFromOriginal(I.getDebugLoc()));
                } else if (auto CI = dyn_cast<CallInst>(&I)) {
                  Function *called = getFunctionFromCall(CI);
                  assert(called);
                  if (called->getName() == "julia.write_barrier" ||
                      isa<MemSetInst>(&I) || isa<MemTransferInst>(&I)) {

                    // TODO
                    SmallVector<Value *, 2> args;
#if LLVM_VERSION_MAJOR >= 14
                    for (auto &arg : CI->args())
#else
                    for (auto &arg : CI->arg_operands())
#endif
                      args.push_back(
                          lookupM(getNewFromOriginal(arg), NB, available));

                    SmallVector<ValueType, 2> BundleTypes(args.size(),
                                                          ValueType::Primal);

                    auto Defs = getInvertedBundles(CI, BundleTypes, NB,
                                                   /*lookup*/ true, available);
                    auto cal = NB.CreateCall(called, args, Defs);
                    cal->setAttributes(CI->getAttributes());
                    cal->setCallingConv(CI->getCallingConv());
                    cal->setTailCallKind(CI->getTailCallKind());
                    cal->setDebugLoc(getNewFromOriginal(I.getDebugLoc()));
                  } else {
                    assert(isDeallocationFunction(*called, TLI));
                    continue;
                  }
                } else {
                  assert(0 && "unhandlable loop rematerialization instruction");
                }
              } else if (loopReallocations.count(&I)) {
                LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                                  &newFunc->getEntryBlock());

                auto inst = getNewFromOriginal((Value *)&I);

                auto found = scopeMap.find(inst);
                if (found == scopeMap.end()) {
                  AllocaInst *cache =
                      createCacheForScope(lctx, inst->getType(),
                                          inst->getName(), /*shouldFree*/ true);
                  assert(cache);
                  found = insert_or_assign(
                      scopeMap, inst,
                      std::pair<AssertingVH<AllocaInst>, LimitContext>(cache,
                                                                       lctx));
                }
                auto cache = found->second.first;
                if (auto MD = hasMetadata(&I, "enzyme_fromstack")) {
                  auto replacement = NB.CreateAlloca(
                      Type::getInt8Ty(I.getContext()),
                      lookupM(getNewFromOriginal(I.getOperand(0)), NB,
                              available));
                  auto Alignment = cast<ConstantInt>(cast<ConstantAsMetadata>(
                                                         MD->getOperand(0))
                                                         ->getValue())
                                       ->getLimitedValue();
#if LLVM_VERSION_MAJOR >= 10
                  replacement->setAlignment(Align(Alignment));
#else
                  replacement->setAlignment(Alignment);
#endif
                  replacement->setDebugLoc(getNewFromOriginal(I.getDebugLoc()));
                  storeInstructionInCache(lctx, NB, replacement, cache);
                } else if (auto CI = dyn_cast<CallInst>(&I)) {
                  SmallVector<Value *, 2> args;
#if LLVM_VERSION_MAJOR >= 14
                  for (auto &arg : CI->args())
#else
                  for (auto &arg : CI->arg_operands())
#endif
                    args.push_back(
                        lookupM(getNewFromOriginal(arg), NB, available));

                  SmallVector<ValueType, 2> BundleTypes(args.size(),
                                                        ValueType::Primal);

                  auto Defs = getInvertedBundles(CI, BundleTypes, NB,
                                                 /*lookup*/ true, available);
                  auto cal = NB.CreateCall(CI->getCalledFunction(), args, Defs);
                  cal->copyMetadata(*CI, MD_ToCopy);
                  cal->setName("remat_" + CI->getName());
                  cal->setAttributes(CI->getAttributes());
                  cal->setCallingConv(CI->getCallingConv());
                  cal->setTailCallKind(CI->getTailCallKind());
                  cal->setDebugLoc(getNewFromOriginal(I.getDebugLoc()));
                  storeInstructionInCache(lctx, NB, cal, cache);
                } else {
                  llvm::errs() << " realloc: " << I << "\n";
                  llvm_unreachable("Unknown loop reallocation");
                }
              }
              if (loopShadowRematerializations.count(&I)) {
                if (auto SI = dyn_cast<StoreInst>(&I)) {
                  Value *orig_ptr = SI->getPointerOperand();
                  Value *orig_val = SI->getValueOperand();
                  Type *valType = orig_val->getType();
                  assert(!isConstantValue(orig_ptr));

                  auto &DL = newFunc->getParent()->getDataLayout();

                  bool constantval = isConstantValue(orig_val) ||
                                     parseTBAA(I, DL).Inner0().isIntegral();

                  // TODO allow recognition of other types that could contain
                  // pointers [e.g. {void*, void*} or <2 x i64> ]
                  auto storeSize = DL.getTypeSizeInBits(valType) / 8;

                  //! Storing a floating point value
                  Type *FT = nullptr;
                  if (valType->isFPOrFPVectorTy()) {
                    FT = valType->getScalarType();
                  } else if (!valType->isPointerTy()) {
                    if (looseTypeAnalysis) {
                      auto fp = TR.firstPointer(storeSize, orig_ptr,
                                                /*errifnotfound*/ false,
                                                /*pointerIntSame*/ true);
                      if (fp.isKnown()) {
                        FT = fp.isFloat();
                      } else if (isa<ConstantInt>(orig_val) ||
                                 valType->isIntOrIntVectorTy()) {
                        llvm::errs()
                            << "assuming type as integral for store: " << I
                            << "\n";
                        FT = nullptr;
                      } else {
                        TR.firstPointer(storeSize, orig_ptr,
                                        /*errifnotfound*/ true,
                                        /*pointerIntSame*/ true);
                        llvm::errs()
                            << "cannot deduce type of store " << I << "\n";
                        assert(0 && "cannot deduce");
                      }
                    } else {
                      FT = TR.firstPointer(storeSize, orig_ptr,
                                           /*errifnotfound*/ true,
                                           /*pointerIntSame*/ true)
                               .isFloat();
                    }
                  }
                  if (!FT) {
                    Value *valueop = nullptr;
                    if (constantval) {
                      Value *val =
                          lookupM(getNewFromOriginal(orig_val), NB, available);
                      valueop = val;
                      if (getWidth() > 1) {
                        Value *array =
                            UndefValue::get(getShadowType(val->getType()));
                        for (unsigned i = 0; i < getWidth(); ++i) {
                          array = NB.CreateInsertValue(array, val, {i});
                        }
                        valueop = array;
                      }
                    } else {
                      valueop =
                          lookupM(invertPointerM(orig_val, NB), NB, available);
                    }
#if LLVM_VERSION_MAJOR >= 10
                    auto align = SI->getAlign();
#else
                    auto align = SI->getAlignment();
#endif
                    setPtrDiffe(orig_ptr, valueop, NB, align, SI->isVolatile(),
                                SI->getOrdering(), SI->getSyncScopeID(),
                                /*mask*/ nullptr);
                  }
                  // TODO shadow memtransfer
                } else if (auto MS = dyn_cast<MemSetInst>(&I)) {
                  if (!isConstantValue(MS->getArgOperand(0))) {
                    Value *args[4] = {
                        lookupM(invertPointerM(MS->getArgOperand(0), NB), NB,
                                available),
                        lookupM(getNewFromOriginal(MS->getArgOperand(1)), NB,
                                available),
                        lookupM(getNewFromOriginal(MS->getArgOperand(2)), NB,
                                available),
                        lookupM(getNewFromOriginal(MS->getArgOperand(3)), NB,
                                available)};

                    ValueType BundleTypes[4] = {
                        ValueType::Shadow, ValueType::Primal, ValueType::Primal,
                        ValueType::Primal};
                    auto Defs = getInvertedBundles(MS, BundleTypes, NB,
                                                   /*lookup*/ true, available);
                    auto cal =
                        NB.CreateCall(MS->getCalledFunction(), args, Defs);
                    cal->copyMetadata(*MS, MD_ToCopy);
                    cal->setAttributes(MS->getAttributes());
                    cal->setCallingConv(MS->getCallingConv());
                    cal->setTailCallKind(MS->getTailCallKind());
                    cal->setDebugLoc(getNewFromOriginal(I.getDebugLoc()));
                  }
                } else if (auto CI = dyn_cast<CallInst>(&I)) {
                  Function *called = getFunctionFromCall(CI);
                  assert(called);
                  if (called->getName() == "julia.write_barrier") {

                    // TODO
                    SmallVector<Value *, 2> args;
#if LLVM_VERSION_MAJOR >= 14
                    for (auto &arg : CI->args())
#else
                    for (auto &arg : CI->arg_operands())
#endif
                      if (!isConstantValue(arg))
                        args.push_back(
                            lookupM(invertPointerM(arg, NB), NB, available));

                    if (args.size()) {
                      SmallVector<ValueType, 2> BundleTypes(args.size(),
                                                            ValueType::Primal);

                      auto Defs =
                          getInvertedBundles(CI, BundleTypes, NB,
                                             /*lookup*/ true, available);
                      auto cal = NB.CreateCall(called, args, Defs);
                      cal->setAttributes(CI->getAttributes());
                      cal->setCallingConv(CI->getCallingConv());
                      cal->setTailCallKind(CI->getTailCallKind());
                      cal->setDebugLoc(getNewFromOriginal(I.getDebugLoc()));
                    }
                  } else {
                    assert(isDeallocationFunction(*called, TLI));
                    continue;
                  }
                } else {
                  assert(
                      0 &&
                      "unhandlable loop shadow rematerialization instruction");
                }
              } else if (loopShadowReallocations.count(&I)) {

                LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                                  &newFunc->getEntryBlock());
                auto ipfound = invertedPointers.find(&I);
                PHINode *placeholder = cast<PHINode>(&*ipfound->second);

                auto found = scopeMap.find(placeholder);
                if (found == scopeMap.end()) {
                  AllocaInst *cache = createCacheForScope(
                      lctx, placeholder->getType(), placeholder->getName(),
                      /*shouldFree*/ true);
                  assert(cache);
                  found = insert_or_assign(
                      scopeMap, (Value *&)placeholder,
                      std::pair<AssertingVH<AllocaInst>, LimitContext>(cache,
                                                                       lctx));
                }
                auto cache = found->second.first;
                Value *anti = nullptr;

                if (auto orig = dyn_cast<CallInst>(&I)) {
                  Function *called = getFunctionFromCall(orig);
                  assert(called);

                  auto dbgLoc = getNewFromOriginal(orig)->getDebugLoc();

                  SmallVector<Value *, 8> args;
#if LLVM_VERSION_MAJOR >= 14
                  for (auto &arg : orig->args())
#else
                  for (auto &arg : orig->arg_operands())
#endif
                  {
                    args.push_back(lookupM(getNewFromOriginal(arg), NB));
                  }

                  placeholder->setName("");
                  if (shadowHandlers.find(called->getName().str()) !=
                      shadowHandlers.end()) {

                    anti =
                        shadowHandlers[called->getName().str()](NB, orig, args);
                  } else {
                    auto rule = [&]() {
#if LLVM_VERSION_MAJOR >= 11
                      Value *anti = NB.CreateCall(
                          orig->getFunctionType(), orig->getCalledOperand(),
                          args, orig->getName() + "'mi");
#else
                      Value *anti = NB.CreateCall(orig->getCalledValue(), args,
                                                  orig->getName() + "'mi");
#endif
                      cast<CallInst>(anti)->setAttributes(
                          orig->getAttributes());
                      cast<CallInst>(anti)->setCallingConv(
                          orig->getCallingConv());
                      cast<CallInst>(anti)->setTailCallKind(
                          orig->getTailCallKind());
                      cast<CallInst>(anti)->setDebugLoc(
                          getNewFromOriginal(I.getDebugLoc()));

#if LLVM_VERSION_MAJOR >= 14
                      cast<CallInst>(anti)->addAttributeAtIndex(
                          AttributeList::ReturnIndex, Attribute::NoAlias);
                      cast<CallInst>(anti)->addAttributeAtIndex(
                          AttributeList::ReturnIndex, Attribute::NonNull);
#else
                      cast<CallInst>(anti)->addAttribute(
                          AttributeList::ReturnIndex, Attribute::NoAlias);
                      cast<CallInst>(anti)->addAttribute(
                          AttributeList::ReturnIndex, Attribute::NonNull);
#endif
                      return anti;
                    };

                    anti = applyChainRule(orig->getType(), NB, rule);

                    if (auto MD = hasMetadata(orig, "enzyme_fromstack")) {
                      auto rule = [&](Value *anti) {
                        AllocaInst *replacement = NB.CreateAlloca(
                            Type::getInt8Ty(orig->getContext()), args[0]);
                        replacement->takeName(anti);
                        auto Alignment =
                            cast<ConstantInt>(
                                cast<ConstantAsMetadata>(MD->getOperand(0))
                                    ->getValue())
                                ->getLimitedValue();
#if LLVM_VERSION_MAJOR >= 10
                        replacement->setAlignment(Align(Alignment));
#else
                        replacement->setAlignment(Alignment);
#endif
                        replacement->setDebugLoc(
                            getNewFromOriginal(I.getDebugLoc()));
                        return replacement;
                      };

                      Value *replacement = applyChainRule(
                          Type::getInt8Ty(orig->getContext()), NB, rule, anti);

                      replaceAWithB(cast<Instruction>(anti), replacement);
                      erase(cast<Instruction>(anti));
                      anti = replacement;
                    }

                    applyChainRule(
                        NB,
                        [&](Value *anti) {
                          zeroKnownAllocation(NB, anti, args, *called, TLI);
                        },
                        anti);
                  }
                } else {
                  llvm_unreachable("Unknown shadow rematerialization value");
                }
                assert(anti);
                storeInstructionInCache(lctx, NB, anti, cache);
              }
            }

            llvm::SmallPtrSet<llvm::BasicBlock *, 8> origExitBlocks;
            getExitBlocks(origLI, origExitBlocks);
            // Remap a branch to the header to enter the incremented
            // reverse of that block.
            auto remap = [&](BasicBlock *rB) {
              // Remap of an exit branch is to go to the reverse
              // exiting block.
              if (origExitBlocks.count(rB)) {
                return reverseBlocks[getNewFromOriginal(B)].front();
              }
              // Reverse of an incrementing branch is go to the
              // reverse of the branching block.
              if (rB == origLI->getHeader())
                return reverseBlocks[getNewFromOriginal(B)].front();
              return origToNewForward[rB];
            };

            // TODO clone terminator
            auto TI = B->getTerminator();
            assert(TI);
            if (notForAnalysis.count(B)) {
              NB.CreateUnreachable();
            } else if (auto BI = dyn_cast<BranchInst>(TI)) {
              if (BI->isUnconditional())
                NB.CreateBr(remap(BI->getSuccessor(0)));
              else
                NB.CreateCondBr(lookupM(getNewFromOriginal(BI->getCondition()),
                                        NB, available),
                                remap(BI->getSuccessor(0)),
                                remap(BI->getSuccessor(1)));
            } else if (auto SI = dyn_cast<SwitchInst>(TI)) {
              auto NSI = NB.CreateSwitch(
                  lookupM(getNewFromOriginal(BI->getCondition()), NB,
                          available),
                  remap(SI->getDefaultDest()));
              for (auto cas : SI->cases()) {
                NSI->addCase(cas.getCaseValue(), remap(cas.getCaseSuccessor()));
              }
            } else {
              assert(isa<UnreachableInst>(TI));
              NB.CreateUnreachable();
            }
            // Fixup phi nodes that may have their predecessors now changed by
            // the phi unwrapping
            if (!notForAnalysis.count(B) &&
                NB.GetInsertBlock() != origToNewForward[B]) {
              for (auto S0 : successors(B)) {
                if (!origToNewForward.count(S0))
                  continue;
                auto S = origToNewForward[S0];
                assert(S);
                for (auto I = S->begin(), E = S->end(); I != E; ++I) {
                  PHINode *orig = dyn_cast<PHINode>(&*I);
                  if (orig == nullptr)
                    break;
                  for (unsigned Op = 0, NumOps = orig->getNumOperands();
                       Op != NumOps; ++Op)
                    if (orig->getIncomingBlock(Op) == origToNewForward[B])
                      orig->setIncomingBlock(Op, NB.GetInsertBlock());
                }
              }
            }
          }
          resumeblock = enterB;
        }
      }

      if (incEntering) {
        BasicBlock *incB = BasicBlock::Create(
            BB->getContext(),
            "inc" + reverseBlocks[lc.header].front()->getName(),
            BB->getParent());
        incB->moveAfter(reverseBlocks[lc.header].back());

        IRBuilder<> tbuild(incB);

#if LLVM_VERSION_MAJOR > 7
        Value *av = tbuild.CreateLoad(lc.var->getType(), lc.antivaralloc);
#else
        Value *av = tbuild.CreateLoad(lc.antivaralloc);
#endif
        Value *sub =
            tbuild.CreateAdd(av, ConstantInt::get(av->getType(), -1), "",
                             /*NUW*/ false, /*NSW*/ true);
        tbuild.CreateStore(sub, lc.antivaralloc);
        tbuild.CreateBr(resumeblock);
        return newBlocksForLoop_cache[tup] = incB;
      } else {
        assert(exitEntering);
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
          lim = lookupValueFromCache(
              /*forwardPass*/ false, tbuild, lctx,
              getDynamicLoopLimit(LI.getLoopFor(lc.header)),
              /*isi1*/ false, /*available*/ ValueToValueMapTy());
        } else {
          lim = lookupM(lc.trueLimit, tbuild);
        }

        tbuild.SetInsertPoint(incB);
        tbuild.CreateStore(lim, lc.antivaralloc);
        tbuild.CreateBr(resumeblock);

        return newBlocksForLoop_cache[tup] = incB;
      }
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
  {
    auto found = available.find(val);
    if (found != available.end()) {
      if (found->second)
        return true;
      else {
        return false;
      }
    }
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
    struct {
      Function *func;
      const LoopInfo &FLI;
    } options[2] = {{newFunc, LI}, {oldFunc, OrigLI}};
    for (const auto &tup : options) {
      if (parent->getParent() == tup.func) {
        for (auto &val : phi->incoming_values()) {
          if (isPotentialLastLoopValue(val, parent, tup.FLI)) {
            return false;
          }
        }
        if (tup.FLI.isLoopHeader(parent)) {
          // Currently can only recompute header
          // with two incoming values
          if (phi->getNumIncomingValues() != 2)
            return false;
          auto L = tup.FLI.getLoopFor(parent);

          // Only recomputable if non recursive.
          SmallPtrSet<Instruction *, 2> seen;
          SmallVector<Instruction *, 1> todo;
          for (auto PH : predecessors(parent)) {
            // Prior iterations must be recomputable without
            // this value.
            if (L->contains(PH)) {
              if (auto I =
                      dyn_cast<Instruction>(phi->getIncomingValueForBlock(PH)))
                if (L->contains(I->getParent()))
                  todo.push_back(I);
            }
          }

          while (todo.size()) {
            auto cur = todo.back();
            todo.pop_back();
            if (seen.count(cur))
              continue;
            seen.insert(cur);
            if (cur == phi)
              return false;
            for (auto &op : cur->operands()) {
              if (auto I = dyn_cast<Instruction>(op)) {
                if (L->contains(I->getParent()))
                  todo.push_back(I);
              }
            }
          }
        }
        return true;
      }
    }
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
      if (called->hasFnAttribute("enzyme_math"))
        n = called->getFnAttribute("enzyme_math").getValueAsString();
      Intrinsic::ID ID = Intrinsic::not_intrinsic;
      if (called->hasFnAttribute("enzyme_shouldrecompute") ||
          isMemFreeLibMFunction(n, &ID) || n == "lgamma_r" ||
          n == "lgammaf_r" || n == "lgammal_r" || n == "__lgamma_r_finite" ||
          n == "__lgammaf_r_finite" || n == "__lgammal_r_finite" ||
          n == "tanh" || n == "tanhf" || n == "__pow_finite" ||
          n == "__fd_sincos_1" || n == "julia.pointer_from_objref" ||
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
      if (called->hasFnAttribute("enzyme_math"))
        n = called->getFnAttribute("enzyme_math").getValueAsString();
      Intrinsic::ID ID = Intrinsic::not_intrinsic;
      if (called->hasFnAttribute("enzyme_shouldrecompute") ||
          isMemFreeLibMFunction(n, &ID) || n == "lgamma_r" ||
          n == "lgammaf_r" || n == "lgammal_r" || n == "__lgamma_r_finite" ||
          n == "__lgammaf_r_finite" || n == "__lgammal_r_finite" ||
          n == "tanh" || n == "tanhf" || n == "__pow_finite" ||
          n == "__fd_sincos_1" || n == "julia.pointer_from_objref" ||
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
    EnzymeLogic &Logic, unsigned width, Function *todiff,
    TargetLibraryInfo &TLI, TypeAnalysis &TA, FnTypeInfo &oldTypeInfo,
    DIFFE_TYPE retType, ArrayRef<DIFFE_TYPE> constant_args, bool returnUsed,
    bool shadowReturnUsed, std::map<AugmentedStruct, int> &returnMapping,
    bool omp) {
  assert(!todiff->empty());
  Function *oldFunc = todiff;

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
  if (shadowReturnUsed) {
    assert(retType == DIFFE_TYPE::DUP_ARG || retType == DIFFE_TYPE::DUP_NONEED);
    assert(!todiff->getReturnType()->isEmptyTy());
    assert(!todiff->getReturnType()->isVoidTy());
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

  std::string prefix = "fakeaugmented";
  if (width > 1)
    prefix += std::to_string(width);
  prefix += "_";
  prefix += todiff->getName().str();

  auto newFunc = Logic.PPC.CloneFunctionWithReturns(
      DerivativeMode::ReverseModePrimal, /* width */ width, oldFunc,
      invertedPointers, constant_args, constant_values, nonconstant_values,
      returnvals,
      /*returnValue*/ returnValue, retType, prefix, &originalToNew,
      /*diffeReturnArg*/ false, /*additionalArg*/ nullptr);

  // Convert uncacheable args from the input function to the preprocessed
  // function

  FnTypeInfo typeInfo(oldFunc);
  {
    auto toarg = todiff->arg_begin();
    auto olarg = oldFunc->arg_begin();
    for (; toarg != todiff->arg_end(); ++toarg, ++olarg) {

      {
        auto fd = oldTypeInfo.Arguments.find(toarg);
        assert(fd != oldTypeInfo.Arguments.end());
        typeInfo.Arguments.insert(
            std::pair<Argument *, TypeTree>(olarg, fd->second));
      }

      {
        auto cfd = oldTypeInfo.KnownValues.find(toarg);
        assert(cfd != oldTypeInfo.KnownValues.end());
        typeInfo.KnownValues.insert(
            std::pair<Argument *, std::set<int64_t>>(olarg, cfd->second));
      }
    }
    typeInfo.Return = oldTypeInfo.Return;
  }

  TypeResults TR = TA.analyzeFunction(typeInfo);
  assert(TR.getFunction() == oldFunc);

  auto res = new GradientUtils(
      Logic, newFunc, oldFunc, TLI, TA, TR, invertedPointers, constant_values,
      nonconstant_values, retType, originalToNew,
      DerivativeMode::ReverseModePrimal, /* width */ width, omp);
  return res;
}

DiffeGradientUtils *DiffeGradientUtils::CreateFromClone(
    EnzymeLogic &Logic, DerivativeMode mode, unsigned width, Function *todiff,
    TargetLibraryInfo &TLI, TypeAnalysis &TA, FnTypeInfo &oldTypeInfo,
    DIFFE_TYPE retType, bool diffeReturnArg, ArrayRef<DIFFE_TYPE> constant_args,
    ReturnType returnValue, Type *additionalArg, bool omp) {
  assert(!todiff->empty());
  Function *oldFunc = todiff;
  assert(mode == DerivativeMode::ReverseModeGradient ||
         mode == DerivativeMode::ReverseModeCombined ||
         mode == DerivativeMode::ForwardMode ||
         mode == DerivativeMode::ForwardModeSplit);
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
    break;
  case DerivativeMode::ReverseModeCombined:
  case DerivativeMode::ReverseModeGradient:
    prefix = "diffe";
    break;
  case DerivativeMode::ReverseModePrimal:
    llvm_unreachable("invalid DerivativeMode: ReverseModePrimal\n");
  }

  if (width > 1)
    prefix += std::to_string(width);

  auto newFunc = Logic.PPC.CloneFunctionWithReturns(
      mode, width, oldFunc, invertedPointers, constant_args, constant_values,
      nonconstant_values, returnvals, returnValue, retType,
      prefix + oldFunc->getName(), &originalToNew,
      /*diffeReturnArg*/ diffeReturnArg, additionalArg);

  // Convert uncacheable args from the input function to the preprocessed
  // function

  FnTypeInfo typeInfo(oldFunc);
  {
    auto toarg = todiff->arg_begin();
    auto olarg = oldFunc->arg_begin();
    for (; toarg != todiff->arg_end(); ++toarg, ++olarg) {

      {
        auto fd = oldTypeInfo.Arguments.find(toarg);
        assert(fd != oldTypeInfo.Arguments.end());
        typeInfo.Arguments.insert(
            std::pair<Argument *, TypeTree>(olarg, fd->second));
      }

      {
        auto cfd = oldTypeInfo.KnownValues.find(toarg);
        assert(cfd != oldTypeInfo.KnownValues.end());
        typeInfo.KnownValues.insert(
            std::pair<Argument *, std::set<int64_t>>(olarg, cfd->second));
      }
    }
    typeInfo.Return = oldTypeInfo.Return;
  }

  TypeResults TR = TA.analyzeFunction(typeInfo);
  assert(TR.getFunction() == oldFunc);

  auto res = new DiffeGradientUtils(
      Logic, newFunc, oldFunc, TLI, TA, TR, invertedPointers, constant_values,
      nonconstant_values, retType, originalToNew, mode, width, omp);

  return res;
}

Constant *GradientUtils::GetOrCreateShadowConstant(
    EnzymeLogic &Logic, TargetLibraryInfo &TLI, TypeAnalysis &TA,
    Constant *oval, DerivativeMode mode, unsigned width, bool AtomicAdd) {
  if (isa<ConstantPointerNull>(oval)) {
    return oval;
  } else if (isa<UndefValue>(oval)) {
    return oval;
  } else if (isa<ConstantInt>(oval)) {
    return oval;
  } else if (auto CD = dyn_cast<ConstantDataArray>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumElements(); i < len; i++) {
      Vals.push_back(GetOrCreateShadowConstant(
          Logic, TLI, TA, CD->getElementAsConstant(i), mode, width, AtomicAdd));
    }
    return ConstantArray::get(CD->getType(), Vals);
  } else if (auto CD = dyn_cast<ConstantArray>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Vals.push_back(GetOrCreateShadowConstant(
          Logic, TLI, TA, CD->getOperand(i), mode, width, AtomicAdd));
    }
    return ConstantArray::get(CD->getType(), Vals);
  } else if (auto CD = dyn_cast<ConstantStruct>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Vals.push_back(GetOrCreateShadowConstant(
          Logic, TLI, TA, CD->getOperand(i), mode, width, AtomicAdd));
    }
    return ConstantStruct::get(CD->getType(), Vals);
  } else if (auto CD = dyn_cast<ConstantVector>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Vals.push_back(GetOrCreateShadowConstant(
          Logic, TLI, TA, CD->getOperand(i), mode, width, AtomicAdd));
    }
    return ConstantVector::get(Vals);
  } else if (auto F = dyn_cast<Function>(oval)) {
    return GetOrCreateShadowFunction(Logic, TLI, TA, F, mode, width, AtomicAdd);
  } else if (auto arg = dyn_cast<ConstantExpr>(oval)) {
    auto C = GetOrCreateShadowConstant(Logic, TLI, TA, arg->getOperand(0), mode,
                                       width, AtomicAdd);
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
      Type *type = arg->getType()->getPointerElementType();
      auto shadow = new GlobalVariable(
          *arg->getParent(), type, arg->isConstant(), arg->getLinkage(),
          Constant::getNullValue(type), arg->getName() + "_shadow", arg,
          arg->getThreadLocalMode(), arg->getType()->getAddressSpace(),
          arg->isExternallyInitialized());
      arg->setMetadata("enzyme_shadow",
                       MDTuple::get(shadow->getContext(),
                                    {ConstantAsMetadata::get(shadow)}));
#if LLVM_VERSION_MAJOR >= 11
      shadow->setAlignment(arg->getAlign());
#else
      shadow->setAlignment(arg->getAlignment());
#endif
      shadow->setUnnamedAddr(arg->getUnnamedAddr());
      if (arg->getInitializer())
        shadow->setInitializer(GetOrCreateShadowConstant(
            Logic, TLI, TA, cast<Constant>(arg->getOperand(0)), mode, width,
            AtomicAdd));
      return shadow;
    }
  }
  llvm::errs() << " unknown constant to create shadow of: " << *oval << "\n";
  llvm_unreachable("unknown constant to create shadow of");
}

Constant *GradientUtils::GetOrCreateShadowFunction(
    EnzymeLogic &Logic, TargetLibraryInfo &TLI, TypeAnalysis &TA, Function *fn,
    DerivativeMode mode, unsigned width, bool AtomicAdd) {
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
      typ = mode == DerivativeMode::ForwardMode ? DIFFE_TYPE::DUP_ARG
                                                : DIFFE_TYPE::OUT_DIFF;
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

  DIFFE_TYPE retType = fn->getReturnType()->isFPOrFPVectorTy() &&
                               mode != DerivativeMode::ForwardMode
                           ? DIFFE_TYPE::OUT_DIFF
                           : DIFFE_TYPE::DUP_ARG;
  if (fn->getReturnType()->isVoidTy() || fn->getReturnType()->isEmptyTy() ||
      (fn->getReturnType()->isIntegerTy() &&
       cast<IntegerType>(fn->getReturnType())->getBitWidth() < 16))
    retType = DIFFE_TYPE::CONSTANT;

  switch (mode) {
  case DerivativeMode::ForwardMode: {
    Constant *newf = Logic.CreateForwardDiff(
        fn, retType, types, TA, false, mode, /*freeMemory*/ true, width,
        nullptr, type_args, uncacheable_args, /*augmented*/ nullptr);

    assert(newf);

    std::string prefix = "_enzyme_forward";

    if (width > 1) {
      prefix += std::to_string(width);
    }

    std::string globalname = (prefix + "_" + fn->getName() + "'").str();
    auto GV = fn->getParent()->getNamedValue(globalname);

    if (GV == nullptr) {
      GV = new GlobalVariable(*fn->getParent(), newf->getType(), true,
                              GlobalValue::LinkageTypes::InternalLinkage, newf,
                              globalname);
    }

    return ConstantExpr::getPointerCast(GV, fn->getType());
  }
  case DerivativeMode::ForwardModeSplit: {
    auto &augdata = Logic.CreateAugmentedPrimal(
        fn, retType, /*constant_args*/ types, TA,
        /*returnUsed*/ !fn->getReturnType()->isEmptyTy() &&
            !fn->getReturnType()->isVoidTy(),
        /*shadowReturnUsed*/ false, type_args, uncacheable_args,
        /*forceAnonymousTape*/ true, width, AtomicAdd);
    Constant *newf = Logic.CreateForwardDiff(
        fn, retType, types, TA, false, mode, /*freeMemory*/ true, width,
        nullptr, type_args, uncacheable_args, /*augmented*/ &augdata);

    assert(newf);

    std::string prefix = "_enzyme_forwardsplit";

    if (width > 1) {
      prefix += std::to_string(width);
    }

    auto cdata = ConstantStruct::get(
        StructType::get(newf->getContext(),
                        {augdata.fn->getType(), newf->getType()}),
        {augdata.fn, newf});

    std::string globalname = (prefix + "_" + fn->getName() + "'").str();
    auto GV = fn->getParent()->getNamedValue(globalname);

    if (GV == nullptr) {
      GV = new GlobalVariable(*fn->getParent(), cdata->getType(), true,
                              GlobalValue::LinkageTypes::InternalLinkage, cdata,
                              globalname);
    }

    return ConstantExpr::getPointerCast(GV, fn->getType());
  }
  case DerivativeMode::ReverseModeCombined:
  case DerivativeMode::ReverseModeGradient:
  case DerivativeMode::ReverseModePrimal: {
    // TODO re atomic add consider forcing it to be atomic always as fallback if
    // used in a parallel context
    bool returnUsed =
        !fn->getReturnType()->isEmptyTy() && !fn->getReturnType()->isVoidTy();
    bool shadowReturnUsed = returnUsed && (retType == DIFFE_TYPE::DUP_ARG ||
                                           retType == DIFFE_TYPE::DUP_NONEED);
    auto &augdata = Logic.CreateAugmentedPrimal(
        fn, retType, /*constant_args*/ types, TA, returnUsed, shadowReturnUsed,
        type_args, uncacheable_args, /*forceAnonymousTape*/ true, width,
        AtomicAdd);
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
        TA,
        /*map*/ &augdata);
    assert(newf);
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
    return applyChainRule(oval->getType(), BuilderM, [&]() { return oval; });
  } else if (isa<UndefValue>(oval)) {
    if (nullShadow)
      return Constant::getNullValue(getShadowType(oval->getType()));
    return applyChainRule(oval->getType(), BuilderM, [&]() { return oval; });
  } else if (isa<ConstantInt>(oval)) {
    if (nullShadow)
      return Constant::getNullValue(getShadowType(oval->getType()));
    return applyChainRule(oval->getType(), BuilderM, [&]() { return oval; });
  } else if (auto CD = dyn_cast<ConstantDataArray>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumElements(); i < len; i++) {
      Value *val = invertPointerM(CD->getElementAsConstant(i), BuilderM);
      Vals.push_back(cast<Constant>(val));
    }
    auto rule = [&CD](ArrayRef<Constant *> Vals) {
      return ConstantArray::get(CD->getType(), Vals);
    };
    return applyChainRule(CD->getType(), Vals, BuilderM, rule);
  } else if (auto CD = dyn_cast<ConstantArray>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Value *val = invertPointerM(CD->getOperand(i), BuilderM);
      Vals.push_back(cast<Constant>(val));
    }

    auto rule = [&CD](ArrayRef<Constant *> Vals) {
      return ConstantArray::get(CD->getType(), Vals);
    };

    return applyChainRule(CD->getType(), Vals, BuilderM, rule);
  } else if (auto CD = dyn_cast<ConstantStruct>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Vals.push_back(
          cast<Constant>(invertPointerM(CD->getOperand(i), BuilderM)));
    }

    auto rule = [&CD](ArrayRef<Constant *> Vals) {
      return ConstantStruct::get(CD->getType(), Vals);
    };
    return applyChainRule(CD->getType(), Vals, BuilderM, rule);
  } else if (auto CD = dyn_cast<ConstantVector>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Vals.push_back(
          cast<Constant>(invertPointerM(CD->getOperand(i), BuilderM)));
    }

    auto rule = [](ArrayRef<Constant *> Vals) {
      return ConstantVector::get(Vals);
    };

    return applyChainRule(CD->getType(), Vals, BuilderM, rule);
  } else if (isa<ConstantData>(oval) && nullShadow) {
    auto rule = [&oval]() { return Constant::getNullValue(oval->getType()); };

    return applyChainRule(oval->getType(), BuilderM, rule);
  }

  if (isConstantValue(oval)) {
    // NOTE, this is legal and the correct resolution, however, our activity
    // analysis honeypot no longer exists

    // Nulling the shadow for a constant is only necessary if any of the data
    // could contain a float (e.g. should not be applied to pointers).
    if (nullShadow) {
      auto CT = TR.query(oval)[{-1}];
      if (!CT.isKnown() || CT.isFloat()) {
        return Constant::getNullValue(getShadowType(oval->getType()));
      }
    }

    if (isa<ConstantExpr>(oval)) {
      auto rule = [&oval]() { return oval; };
      return applyChainRule(oval->getType(), BuilderM, rule);
    }

    Value *newval = getNewFromOriginal(oval);

    auto rule = [&]() { return newval; };

    return applyChainRule(oval->getType(), BuilderM, rule);
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

    auto rule1 = [&]() {
      AllocaInst *antialloca = bb.CreateAlloca(
          oval->getType()->getPointerElementType(),
          cast<PointerType>(oval->getType())->getPointerAddressSpace(), nullptr,
          oval->getName() + "'ipa");

      auto dst_arg =
          bb.CreateBitCast(antialloca, Type::getInt8PtrTy(oval->getContext()));
      auto val_arg = ConstantInt::get(Type::getInt8Ty(oval->getContext()), 0);
      auto len_arg =
          ConstantInt::get(Type::getInt64Ty(oval->getContext()),
                           M->getDataLayout().getTypeAllocSizeInBits(
                               oval->getType()->getPointerElementType()) /
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
    };

    Value *antialloca = applyChainRule(oval->getType(), bb, rule1);

    invertedPointers.insert(std::make_pair(
        (const Value *)oval, InvertedPointerVH(this, antialloca)));

    return antialloca;
  } else if (auto arg = dyn_cast<GlobalVariable>(oval)) {
    if (!hasMetadata(arg, "enzyme_shadow")) {

      if ((mode == DerivativeMode::ReverseModeCombined ||
           mode == DerivativeMode::ForwardMode) &&
          arg->getType()->getPointerAddressSpace() == 0) {
        auto CT = TR.query(arg)[{-1, -1}];
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
            Type *allocaTy = arg->getValueType();

            auto rule1 = [&]() {
              AllocaInst *antialloca = bb.CreateAlloca(
                  allocaTy, arg->getType()->getPointerAddressSpace(), nullptr,
                  arg->getName() + "'ipa");
              if (arg->getAlignment()) {
#if LLVM_VERSION_MAJOR >= 10
                antialloca->setAlignment(Align(arg->getAlignment()));
#else
                antialloca->setAlignment(arg->getAlignment());
#endif
              }
              return antialloca;
            };

            Value *antialloca = applyChainRule(arg->getType(), bb, rule1);

            invertedPointers.insert(std::make_pair(
                (const Value *)oval, InvertedPointerVH(this, antialloca)));

            auto rule2 = [&](Value *antialloca) {
              auto dst_arg = bb.CreateBitCast(
                  antialloca, Type::getInt8PtrTy(arg->getContext()));
              auto val_arg =
                  ConstantInt::get(Type::getInt8Ty(arg->getContext()), 0);
              auto len_arg =
                  ConstantInt::get(Type::getInt64Ty(arg->getContext()),
                                   M->getDataLayout().getTypeAllocSizeInBits(
                                       arg->getValueType()) /
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
              assert((width > 1 && antialloca->getType() ==
                                       ArrayType::get(arg->getType(), width)) ||
                     antialloca->getType() == arg->getType());
              return antialloca;
            };

            return applyChainRule(arg->getType(), bb, rule2, antialloca);
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
        Type *type = arg->getType()->getPointerElementType();
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
      //  If a variable is constant, for forward mode it will also
      //  only be read, so invert initializing is fine.
      //  For reverse mode, any floats will be +='d into, but never
      //  read, and any pointers will be used as expected. The never
      //  read means even if two globals for floats, that's fine.
      //  As long as the pointers point to equivalent places (which
      //  they should from the same initialization), it is also ok.
      if (arg->hasInternalLinkage() || arg->hasPrivateLinkage() ||
          (arg->hasExternalLinkage() && arg->hasInitializer()) ||
          arg->isConstant()) {
        Type *elemTy = arg->getType()->getPointerElementType();
        IRBuilder<> B(inversionAllocs);

        auto rule = [&]() {
          auto shadow = new GlobalVariable(
              *arg->getParent(), elemTy, arg->isConstant(), arg->getLinkage(),
              Constant::getNullValue(elemTy), arg->getName() + "_shadow", arg,
              arg->getThreadLocalMode(), arg->getType()->getAddressSpace(),
              arg->isExternallyInitialized());
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
        };

        Value *shadow = applyChainRule(oval->getType(), BuilderM, rule);

        if (arg->getInitializer()) {
          applyChainRule(
              BuilderM,
              [&](Value *shadow, Value *ip) {
                cast<GlobalVariable>(shadow)->setInitializer(
                    cast<Constant>(ip));
              },
              shadow,
              invertPointerM(arg->getInitializer(), B, /*nullShadow*/ true));
        }

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
    auto cs = cast<Constant>(gvemd->getValue());

    if (width > 1) {
      SmallVector<Constant *, 2> Vals;
      for (unsigned i = 0; i < width; ++i) {

        Constant *idxs[] = {
            ConstantInt::get(Type::getInt32Ty(cs->getContext()), 0),
            ConstantInt::get(Type::getInt32Ty(cs->getContext()), i)};
        Constant *elem = ConstantExpr::getInBoundsGetElementPtr(
            cs->getType()->getPointerElementType(), cs, idxs);
        Vals.push_back(elem);
      }

      auto agg = ConstantArray::get(
          cast<ArrayType>(getShadowType(arg->getType())), Vals);

      invertedPointers.insert(
          std::make_pair((const Value *)oval, InvertedPointerVH(this, agg)));
      return agg;
    } else {
      invertedPointers.insert(
          std::make_pair((const Value *)oval, InvertedPointerVH(this, cs)));
      return cs;
    }
  } else if (auto fn = dyn_cast<Function>(oval)) {
    Constant *shadow =
        GetOrCreateShadowFunction(Logic, TLI, TA, fn, mode, width, AtomicAdd);
    if (width > 1) {
      SmallVector<Constant *, 3> arr;
      for (unsigned i = 0; i < width; ++i) {
        arr.push_back(shadow);
      }
      ArrayType *arrTy = ArrayType::get(shadow->getType(), width);
      shadow = ConstantArray::get(arrTy, arr);
    }
    return shadow;
  } else if (auto arg = dyn_cast<CastInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *invertOp = invertPointerM(arg->getOperand(0), bb);
    Type *shadowTy = arg->getDestTy();

    auto rule = [&](Value *invertOp) {
      return bb.CreateCast(arg->getOpcode(), invertOp, shadowTy,
                           arg->getName() + "'ipc");
    };

    Value *shadow = applyChainRule(shadowTy, bb, rule, invertOp);

    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<ConstantExpr>(oval)) {
    IRBuilder<> bb(inversionAllocs);
    auto ip = invertPointerM(arg->getOperand(0), bb);

    if (arg->isCast()) {
      if (auto PT = dyn_cast<PointerType>(arg->getType())) {
        if (isConstantValue(arg->getOperand(0)) &&
            PT->getPointerElementType()->isFunctionTy()) {
          goto end;
        }
      }
      if (auto C = dyn_cast<Constant>(ip)) {
        auto rule = [&](Value *ip) {
          return ConstantExpr::getCast(arg->getOpcode(), C, arg->getType());
        };

        return applyChainRule(arg->getType(), bb, rule, ip);

      } else {
        auto rule = [&](Value *ip) {
          return bb.CreateCast((Instruction::CastOps)arg->getOpcode(), ip,
                               arg->getType(), arg->getName() + "'ipc");
        };

        Value *shadow = applyChainRule(arg->getType(), bb, rule, ip);

        invertedPointers.insert(std::make_pair(
            (const Value *)oval, InvertedPointerVH(this, shadow)));

        return shadow;
      }
    } else if (arg->getOpcode() == Instruction::GetElementPtr) {
      if (auto C = dyn_cast<Constant>(ip)) {
        auto rule = [&arg, &C]() {
          SmallVector<Constant *, 8> NewOps;
          for (unsigned i = 0, e = arg->getNumOperands(); i != e; ++i)
            NewOps.push_back(i == 0 ? C : arg->getOperand(i));
          return cast<Value>(arg->getWithOperands(NewOps));
        };

        return applyChainRule(arg->getType(), bb, rule);
      } else {
        SmallVector<Value *, 4> invertargs;
        for (unsigned i = 0; i < arg->getNumOperands() - 1; ++i) {
          Value *b = getNewFromOriginal(arg->getOperand(1 + i));
          invertargs.push_back(b);
        }

        auto rule = [&bb, &arg, &invertargs](Value *ip) {
// TODO mark this the same inbounds as the original
#if LLVM_VERSION_MAJOR > 7
          return bb.CreateGEP(ip->getType()->getPointerElementType(), ip,
                              invertargs, arg->getName() + "'ipg");
#else
          return bb.CreateGEP(ip, invertargs, arg->getName() + "'ipg");
#endif
        };

        Value *shadow = applyChainRule(arg->getType(), bb, rule, ip);

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
    auto ip = invertPointerM(arg->getOperand(0), bb);

    auto rule = [&bb, &arg](Value *ip) {
      return bb.CreateExtractValue(ip, arg->getIndices(),
                                   arg->getName() + "'ipev");
    };

    Value *shadow = applyChainRule(arg->getType(), bb, rule, ip);

    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<InsertValueInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    auto ip0 = invertPointerM(arg->getOperand(0), bb, nullShadow);
    auto ip1 = invertPointerM(arg->getOperand(1), bb, nullShadow);

    auto rule = [&bb, &arg](Value *ip0, Value *ip1) {
      return bb.CreateInsertValue(ip0, ip1, arg->getIndices(),
                                  arg->getName() + "'ipiv");
    };

    Value *shadow = applyChainRule(arg->getType(), bb, rule, ip0, ip1);

    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<ExtractElementInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    auto ip = invertPointerM(arg->getVectorOperand(), bb);

    auto rule = [&](Value *ip) {
      return bb.CreateExtractElement(ip,
                                     getNewFromOriginal(arg->getIndexOperand()),
                                     arg->getName() + "'ipee");
      ;
    };

    Value *shadow = applyChainRule(arg->getType(), bb, rule, ip);

    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<InsertElementInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *op0 = arg->getOperand(0);
    Value *op1 = arg->getOperand(1);
    Value *op2 = arg->getOperand(2);
    auto ip0 = invertPointerM(op0, bb, nullShadow);
    auto ip1 = invertPointerM(op1, bb, nullShadow);

    auto rule = [&](Value *ip0, Value *ip1) {
      return bb.CreateInsertElement(ip0, ip1, getNewFromOriginal(op2),
                                    arg->getName() + "'ipie");
    };

    Value *shadow = applyChainRule(arg->getType(), bb, rule, ip0, ip1);

    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<ShuffleVectorInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *op0 = arg->getOperand(0);
    Value *op1 = arg->getOperand(1);
    auto ip0 = invertPointerM(op0, bb);
    auto ip1 = invertPointerM(op1, bb);

    auto rule = [&bb, &arg](Value *ip0, Value *ip1) {
#if LLVM_VERSION_MAJOR >= 11
      return bb.CreateShuffleVector(ip0, ip1, arg->getShuffleMaskForBitcode(),
                                    arg->getName() + "'ipsv");
#else
      return bb.CreateShuffleVector(ip0, ip1, arg->getOperand(2),
                                    arg->getName() + "'ipsv");
#endif
    };

    Value *shadow = applyChainRule(arg->getType(), bb, rule, ip0, ip1);

    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<SelectInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    bb.setFastMathFlags(getFast());
    Value *shadow = applyChainRule(
        arg->getType(), bb,
        [&](Value *tv, Value *fv) {
          return bb.CreateSelect(getNewFromOriginal(arg->getCondition()), tv,
                                 fv, arg->getName() + "'ipse");
        },
        invertPointerM(arg->getTrueValue(), bb, nullShadow),
        invertPointerM(arg->getFalseValue(), bb, nullShadow));
    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<LoadInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *op0 = arg->getOperand(0);
    Value *ip = invertPointerM(op0, bb);

    auto rule = [&](Value *ip) {
#if LLVM_VERSION_MAJOR > 7
      auto li =
          bb.CreateLoad(arg->getPointerOperandType()->getPointerElementType(),
                        ip, arg->getName() + "'ipl");
#else
      auto li = bb.CreateLoad(ip, arg->getName() + "'ipl");
#endif
      li->copyMetadata(*arg, MD_ToCopy);
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
      return li;
    };

    Value *li = applyChainRule(arg->getType(), bb, rule, ip);

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

    auto rule = [&bb, &arg](Value *val0, Value *val1) {
      auto li = bb.CreateBinOp(arg->getOpcode(), val0, val1, arg->getName());
      if (auto BI = dyn_cast<BinaryOperator>(li))
        BI->copyIRFlags(arg);
      return li;
    };

    Value *li = applyChainRule(arg->getType(), bb, rule, val0, val1);

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
    Value *ip = invertPointerM(arg->getPointerOperand(), bb);

    auto rule = [&](Value *ip) {
#if LLVM_VERSION_MAJOR > 7
      auto shadow = bb.CreateGEP(ip->getType()->getPointerElementType(), ip,
                                 invertargs, arg->getName() + "'ipg");
#else
      auto shadow = bb.CreateGEP(ip, invertargs, arg->getName() + "'ipg");
#endif

      if (auto gep = dyn_cast<GetElementPtrInst>(shadow))
        gep->setIsInBounds(arg->isInBounds());

      return shadow;
    };

    Value *shadow = applyChainRule(arg->getType(), bb, rule, ip);

    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto inst = dyn_cast<AllocaInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(inst));
    Value *asize = getNewFromOriginal(inst->getArraySize());

    auto rule1 = [&]() {
      AllocaInst *antialloca = bb.CreateAlloca(
          inst->getAllocatedType(), inst->getType()->getPointerAddressSpace(),
          asize, inst->getName() + "'ipa");
      if (inst->getAlignment()) {
#if LLVM_VERSION_MAJOR >= 10
        antialloca->setAlignment(Align(inst->getAlignment()));
#else
        antialloca->setAlignment(inst->getAlignment());
#endif
      }
      return antialloca;
    };

    Value *antialloca = applyChainRule(oval->getType(), bb, rule1);

    invertedPointers.insert(std::make_pair(
        (const Value *)oval, InvertedPointerVH(this, antialloca)));

    if (auto ci = dyn_cast<ConstantInt>(asize)) {
      if (ci->isOne()) {

        auto rule = [&](Value *antialloca) {
          StoreInst *st = bb.CreateStore(
              Constant::getNullValue(inst->getAllocatedType()), antialloca);
          if (inst->getAlignment()) {
#if LLVM_VERSION_MAJOR >= 10
            cast<StoreInst>(st)->setAlignment(Align(inst->getAlignment()));
#else
            cast<StoreInst>(st)->setAlignment(inst->getAlignment());
#endif
          }
        };

        applyChainRule(bb, rule, antialloca);

        return antialloca;
      } else {
        // TODO handle alloca of size > 1
      }
    }

    auto rule2 = [&](Value *antialloca) {
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
    };

    applyChainRule(bb, rule2, antialloca);

    return antialloca;
  } else if (auto II = dyn_cast<IntrinsicInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(II));
    bb.setFastMathFlags(getFast());
    switch (II->getIntrinsicID()) {
    default:
      goto end;
    case Intrinsic::nvvm_ldu_global_i:
    case Intrinsic::nvvm_ldu_global_p:
    case Intrinsic::nvvm_ldu_global_f:
    case Intrinsic::nvvm_ldg_global_i:
    case Intrinsic::nvvm_ldg_global_p:
    case Intrinsic::nvvm_ldg_global_f: {
      return applyChainRule(
          II->getType(), bb,
          [&](Value *ptr) {
            Value *args[] = {ptr};
            auto li = bb.CreateCall(II->getCalledFunction(), args);
            li->copyMetadata(*II, MD_ToCopy);
            li->setDebugLoc(getNewFromOriginal(II->getDebugLoc()));
            return li;
          },
          invertPointerM(II->getArgOperand(0), bb));
    case Intrinsic::masked_load:
      return applyChainRule(
          II->getType(), bb,
          [&](Value *ptr, Value *defaultV) {
            Value *args[] = {ptr, getNewFromOriginal(II->getArgOperand(1)),
                             getNewFromOriginal(II->getArgOperand(2)),
                             defaultV};
            auto li = bb.CreateCall(II->getCalledFunction(), args);
            li->copyMetadata(*II, MD_ToCopy);
            li->setDebugLoc(getNewFromOriginal(II->getDebugLoc()));
            return li;
          },
          invertPointerM(II->getArgOperand(0), bb),
          invertPointerM(II->getArgOperand(3), bb, nullShadow));
    }
    }
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
      bb.setFastMathFlags(getFast());
      // Note if the original phi node get's scev'd in NewF, it may
      // no longer be a phi and we need a new place to insert this phi
      // Note that if scev'd this can still be a phi with 0 incoming indicating
      // an unnecessary value to be replaced
      // TODO consider allowing the inverted pointer to become a scev
      if (!isa<PHINode>(NewV) ||
          cast<PHINode>(NewV)->getNumIncomingValues() == 0) {
        bb.SetInsertPoint(bb.GetInsertBlock(), bb.GetInsertBlock()->begin());
      }

      Type *shadowTy = getShadowType(phi->getType());
      PHINode *which = bb.CreatePHI(shadowTy, phi->getNumIncomingValues());
      which->setDebugLoc(getNewFromOriginal(phi->getDebugLoc()));

      invertedPointers.insert(
          std::make_pair((const Value *)oval, InvertedPointerVH(this, which)));

      for (unsigned int i = 0; i < phi->getNumIncomingValues(); ++i) {
        IRBuilder<> pre(
            cast<BasicBlock>(getNewFromOriginal(phi->getIncomingBlock(i)))
                ->getTerminator());
        Value *val = invertPointerM(phi->getIncomingValue(i), pre, nullShadow);
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

  if (CustomErrorHandler) {
    std::string str;
    raw_string_ostream ss(str);
    ss << "cannot find shadow for " << *oval;
    CustomErrorHandler(str.c_str(), wrap(oval), ErrorType::NoShadow, this);
  }

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
      if (!orig) {
        llvm::errs() << "oldFunc: " << *oldFunc << "\n";
        llvm::errs() << "newFunc: " << *newFunc << "\n";
        llvm::errs() << "insertBlock: " << *BuilderM.GetInsertBlock() << "\n";
        llvm::errs() << "instP: " << *inst->getParent() << "\n";
        llvm::errs() << "inst: " << *inst << "\n";
      }
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
    if (pair.second)
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
                                 /*isi1*/ false, available);
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
        if (op->getType() != inst->getType()) {
          llvm::errs() << " op: " << *op << " inst: " << *inst << "\n";
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

                  if (auto found = findInMap(scopeMap, (Value *)liobj)) {
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
                    Value *outer = getCachePointer(
                        /*inForwardPass*/ true, v, lctx, cache, isi1,
                        /*storeinstorecache*/ true,
                        /*available*/ ValueToValueMapTy(),
                        /*extraSize*/ nullptr);

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
                  Value *outer = getCachePointer(
                      /*inForwardPass*/ false, BuilderM, lctx, cache, isi1,
                      /*storeinstorecache*/ true, available,
                      /*extraSize*/ nullptr);
                  SmallVector<Value *, 2> idxs;
                  for (auto &idx : GEP->indices()) {
                    idxs.push_back(lookupM(idx, BuilderM, available,
                                           tryLegalRecomputeCheck));
                  }

#if LLVM_VERSION_MAJOR > 7
                  auto cptr = BuilderM.CreateGEP(
                      outer->getType()->getPointerElementType(), outer, idxs);
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

              SmallVector<Instruction *, 4> toErase;
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
                                  /*available*/ ValueToValueMapTy(),
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
                /*isForwardPass*/ false, BuilderM, lctx, cache, isi1, available,
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

  BasicBlock *scope = inst->getParent();
  if (auto origInst = isOriginal(inst)) {
    auto found = rematerializableAllocations.find(origInst);
    if (found != rematerializableAllocations.end())
      if (found->second.LI && found->second.LI->contains(origInst)) {
        bool cacheWholeAllocation = false;
        if (knownRecomputeHeuristic.count(origInst)) {
          if (!knownRecomputeHeuristic[origInst]) {
            cacheWholeAllocation = true;
          }
        }
        // If not caching whole allocation and rematerializing the allocation
        // within the loop, force an entry-level scope so there is no need
        // to cache.
        if (!cacheWholeAllocation)
          scope = &newFunc->getEntryBlock();
      }
  } else {
    for (auto pair : backwardsOnlyShadows) {
      if (auto pinst = dyn_cast<Instruction>(pair.first))
        if (!pair.second.primalInitialize && pair.second.LI &&
            pair.second.LI->contains(pinst->getParent())) {
          auto found = invertedPointers.find(pair.first);
          if (found != invertedPointers.end() && found->second == inst) {
            scope = &newFunc->getEntryBlock();

            // Prevent the phi node from being stored into the cache by creating
            // it before the ensureLookupCached.
            if (scopeMap.find(inst) == scopeMap.end()) {
              LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                                scope);

              AllocaInst *cache = createCacheForScope(
                  lctx, inst->getType(), inst->getName(), /*shouldFree*/ true);
              assert(cache);
              insert_or_assign(scopeMap, (Value *&)inst,
                               std::pair<AssertingVH<AllocaInst>, LimitContext>(
                                   cache, lctx));
            }
            break;
          }
        }
    }
  }

  ensureLookupCached(inst, /*shouldFree*/ true, scope,
                     inst->getMetadata(LLVMContext::MD_tbaa));
  bool isi1 = inst->getType()->isIntegerTy() &&
              cast<IntegerType>(inst->getType())->getBitWidth() == 1;
  assert(!isOriginalBlock(*BuilderM.GetInsertBlock()));
  auto found = findInMap(scopeMap, (Value *)inst);
  Value *result =
      lookupValueFromCache(/*isForwardPass*/ false, BuilderM, found->second,
                           found->first, isi1, available);
  if (auto LI2 = dyn_cast<LoadInst>(result))
    if (auto LI1 = dyn_cast<LoadInst>(inst))
      LI2->copyMetadata(*LI1, MD_ToCopy);
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

      // If this block dominates the context, don't go back up as any
      // predecessors won't contain the conditions.
      if (DT.dominates(block, ctx))
        continue;

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

  // llvm::errs() << "\n\n<DONE = " << ctx->getName() << ">\n";
  for (auto pair : done) {
    const auto &edge = pair.first;
    blocks.insert(edge.first);
    // llvm::errs() << " edge  (" << edge.first->getName() << ", "
    //             << edge.second->getName() << ") : [";
    // for (auto s : pair.second)
    //  llvm::errs() << s->getName() << ",";
    // llvm::errs() << "]\n";
  }
  // llvm::errs() << "</DONE>\n";

  if (targetToPreds.size() == 3) {
    // Try `block` as a potential first split point.
    for (auto block : blocks) {
      {
        // The original split block must not have a parent with an edge
        // to a block other than to itself, which can reach any targets.
        if (!DT.dominates(block, ctx))
          continue;

        // For all successors and thus edges (block, succ):
        // 1) Ensure that no successors have overlapping potential
        // destinations (a list of destinations previously seen is in
        // foundtargets).
        // 2) The block branches to all 3 destinations (foundTargets==3)
        std::set<BasicBlock *> foundtargets;
        // 3) The unique target split off from the others is stored in
        //   uniqueTarget.
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

        // Only handle cases where the split was due to a conditional
        // branch. This branch, `bi`, splits off uniqueTargets[0] from
        // the remainder of foundTargets.
        auto bi1 = dyn_cast<BranchInst>(block->getTerminator());
        if (!bi1)
          goto rnextpair;

        {
          // Find a second block `subblock` which splits the two merged
          // targets from each other.
          BasicBlock *subblock = nullptr;
          for (auto block2 : blocks) {
            {
              // The second split block must not have a parent with an edge
              // to a block other than to itself, which can reach any of its two
              // targets.
              // TODO verify this
              for (auto P : predecessors(block2)) {
                for (auto S : successors(P)) {
                  if (S == block2)
                    continue;
                  auto edge = std::make_pair(P, S);
                  if (done.find(edge) != done.end()) {
                    for (auto target : done[edge]) {
                      if (foundtargets.find(target) != foundtargets.end() &&
                          uniqueTargets.find(target) == uniqueTargets.end()) {
                        goto nextblock;
                      }
                    }
                  }
                }
              }

              // Again, a successful split must have unique targets.
              std::set<BasicBlock *> seen2;
              for (BasicBlock *succ : successors(block2)) {
                auto edge = std::make_pair(block2, succ);
                // Since there are only two targets, a successful split
                // condition has only 1 target per successor of block2.
                if (done[edge].size() != 1) {
                  goto nextblock;
                }
                for (BasicBlock *target : done[edge]) {
                  // block2 has non-unique targets.
                  if (seen2.find(target) != seen2.end()) {
                    goto nextblock;
                  }
                  seen2.insert(target);
                  // block2 has a target which is not part of the two needing
                  // to be split. The two needing to be split is equal to
                  //    foundtargets-uniqueTargets.
                  if (foundtargets.find(target) == foundtargets.end()) {
                    goto nextblock;
                  }
                  if (uniqueTargets.find(target) != uniqueTargets.end()) {
                    goto nextblock;
                  }
                }
              }
              // If we didn't find two valid successors, continue.
              if (seen2.size() != 2) {
                // llvm::errs() << " -- failed from not 2 seen\n";
                goto nextblock;
              }
              subblock = block2;
              break;
            }
          nextblock:;
          }

          // If no split block was found, try again.
          if (subblock == nullptr)
            goto rnextpair;

          // This branch, `bi2`, splits off the two blocks in
          // (foundTargets-uniqueTargets) from each other.
          auto bi2 = dyn_cast<BranchInst>(subblock->getTerminator());
          if (!bi2)
            goto rnextpair;

          // Condition cond1 splits off uniqueTargets[0] from
          // the remainder of foundTargets.
          auto cond1 = lookupM(bi1->getCondition(), BuilderM);

          // Condition cond2 splits off the two blocks in
          // (foundTargets-uniqueTargets) from each other.
          auto cond2 = lookupM(bi2->getCondition(), BuilderM);

          if (replacePHIs == nullptr) {
            BasicBlock *staging =
                BasicBlock::Create(oldFunc->getContext(), "staging", newFunc);
            auto stagingIfNeeded = [&](BasicBlock *B) {
              auto edge = std::make_pair(block, B);
              if (done[edge].size() == 1) {
                return *done[edge].begin();
              } else {
                assert(done[edge].size() == 2);
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
    {
      // The original split block must not have a parent with an edge
      // to a block other than to itself, which can reach any targets.
      if (!DT.dominates(block, ctx))
        for (auto P : predecessors(block)) {
          for (auto S : successors(P)) {
            if (S == block)
              continue;
            auto edge = std::make_pair(P, S);
            if (done.find(edge) != done.end() && done[edge].size())
              goto nextpair;
          }
        }

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
      if (!(BuilderM.GetInsertBlock()->size() == 0 ||
            !isa<BranchInst>(BuilderM.GetInsertBlock()->back()))) {
        llvm::errs() << "newFunc : " << *newFunc << "\n";
        llvm::errs() << "blk : " << *BuilderM.GetInsertBlock() << "\n";
      }
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
        Value *cas = nullptr;
        for (auto c : si->cases()) {
          if (pair.first ==
              *done[std::make_pair(block, c.getCaseSuccessor())].begin()) {
            cas = c.getCaseValue();
            break;
          }
        }
        if (cas == nullptr) {
          assert(pair.first ==
                 *done[std::make_pair(block, si->getDefaultDest())].begin());
        }
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
  SmallVector<BasicBlock *, 4> targets;
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
      LimitContext(/*reversePass*/ reverseBlocks.size() > 0, ctx), cache, isi1,
      /*available*/ ValueToValueMapTy());
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
                  this, &I, minCutMode, FullSeen, guaranteedUnreachable)) {
            bool oneneed = is_value_needed_in_reverse<ValueType::Primal,
                                                      /*OneLevel*/ true>(
                this, &I, minCutMode, OneLevelSeen, guaranteedUnreachable);
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
              this, V, minCutMode, FullSeen, guaranteedUnreachable)) {
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
              this, V, minCutMode, OneLevelSeen, guaranteedUnreachable)) {
        Required.insert(V);
      } else {
        for (auto V2 : V->users()) {
          if (auto Inst = dyn_cast<Instruction>(V2))
            for (auto pair : rematerializableAllocations) {
              if (pair.second.stores.count(Inst)) {
                todo.push_back(pair.first);
              }
            }
          todo.push_back(V2);
        }
      }
    }

    SmallPtrSet<Value *, 5> MinReq;
    minCut(oldFunc->getParent()->getDataLayout(), OrigLI, Recomputes,
           Intermediates, Required, MinReq, rematerializableAllocations);
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
                       llvm::CallInst *MTI, bool allowForward,
                       bool shadowsLookedUp, bool backwardsShadow) {
  // TODO offset
  if (secretty) {
    // no change to forward pass if represents floats
    if (mode == DerivativeMode::ReverseModeGradient ||
        mode == DerivativeMode::ReverseModeCombined ||
        mode == DerivativeMode::ForwardModeSplit) {
      IRBuilder<> Builder2(MTI);
      if (mode == DerivativeMode::ForwardModeSplit)
        gutils->getForwardBuilder(Builder2);
      else
        gutils->getReverseBuilder(Builder2);

      // If the src is constant simply zero d_dst and don't propagate to d_src
      // (which thus == src and may be illegal)
      if (srcConstant) {
        // Don't zero in forward mode.
        if (mode != DerivativeMode::ForwardModeSplit) {

          Value *args[] = {
            shadowsLookedUp ? shadow_dst
                            : gutils->lookupM(shadow_dst, Builder2),
            ConstantInt::get(Type::getInt8Ty(MTI->getContext()), 0),
            gutils->lookupM(length, Builder2),
#if LLVM_VERSION_MAJOR <= 6
            ConstantInt::get(Type::getInt32Ty(MTI->getContext()),
                             max(1U, dstalign)),
#endif
            ConstantInt::getFalse(MTI->getContext())
          };

          if (args[0]->getType()->isIntegerTy())
            args[0] = Builder2.CreateIntToPtr(
                args[0], Type::getInt8PtrTy(MTI->getContext()));

          Type *tys[] = {args[0]->getType(), args[2]->getType()};
          auto memsetIntr = Intrinsic::getDeclaration(
              MTI->getParent()->getParent()->getParent(), Intrinsic::memset,
              tys);
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
        }

      } else {
        auto dsto =
            (shadowsLookedUp || mode == DerivativeMode::ForwardModeSplit)
                ? shadow_dst
                : gutils->lookupM(shadow_dst, Builder2);
        if (dsto->getType()->isIntegerTy())
          dsto = Builder2.CreateIntToPtr(
              dsto, Type::getInt8PtrTy(dsto->getContext()));
        unsigned dstaddr =
            cast<PointerType>(dsto->getType())->getAddressSpace();
        if (offset != 0) {
#if LLVM_VERSION_MAJOR > 7
          dsto = Builder2.CreateConstInBoundsGEP1_64(
              dsto->getType()->getPointerElementType(), dsto, offset);
#else
          dsto = Builder2.CreateConstInBoundsGEP1_64(dsto, offset);
#endif
        }
        auto srco =
            (shadowsLookedUp || mode == DerivativeMode::ForwardModeSplit)
                ? shadow_src
                : gutils->lookupM(shadow_src, Builder2);
        if (mode != DerivativeMode::ForwardModeSplit)
          dsto = Builder2.CreatePointerCast(
              dsto, PointerType::get(secretty, dstaddr));
        if (srco->getType()->isIntegerTy())
          srco = Builder2.CreateIntToPtr(
              srco, Type::getInt8PtrTy(srco->getContext()));
        unsigned srcaddr =
            cast<PointerType>(srco->getType())->getAddressSpace();
        if (offset != 0) {
#if LLVM_VERSION_MAJOR > 7
          srco = Builder2.CreateConstInBoundsGEP1_64(
              srco->getType()->getPointerElementType(), srco, offset);
#else
          srco = Builder2.CreateConstInBoundsGEP1_64(srco, offset);
#endif
        }
        if (mode != DerivativeMode::ForwardModeSplit)
          srco = Builder2.CreatePointerCast(
              srco, PointerType::get(secretty, srcaddr));

        if (mode == DerivativeMode::ForwardModeSplit) {
#if LLVM_VERSION_MAJOR >= 11
          MaybeAlign dalign;
          if (dstalign)
            dalign = MaybeAlign(dstalign);
          MaybeAlign salign;
          if (srcalign)
            salign = MaybeAlign(srcalign);
#else
          auto dalign = dstalign;
          auto salign = srcalign;
#endif

          if (intrinsic == Intrinsic::memmove) {
            Builder2.CreateMemMove(dsto, dalign, srco, salign, length);
          } else {
            Builder2.CreateMemCpy(dsto, dalign, srco, salign, length);
          }
        } else {
          Value *args[]{
              Builder2.CreatePointerCast(dsto,
                                         PointerType::get(secretty, dstaddr)),
              Builder2.CreatePointerCast(srco,
                                         PointerType::get(secretty, srcaddr)),
              Builder2.CreateUDiv(
                  gutils->lookupM(length, Builder2),
                  ConstantInt::get(length->getType(),
                                   Builder2.GetInsertBlock()
                                           ->getParent()
                                           ->getParent()
                                           ->getDataLayout()
                                           .getTypeAllocSizeInBits(secretty) /
                                       8))};

          auto dmemcpy = ((intrinsic == Intrinsic::memcpy)
                              ? getOrInsertDifferentialFloatMemcpy
                              : getOrInsertDifferentialFloatMemmove)(
              *MTI->getParent()->getParent()->getParent(), secretty, dstalign,
              srcalign, dstaddr, srcaddr);
          Builder2.CreateCall(dmemcpy, args);
        }
      }
    }
  } else {

    // if represents pointer or integer type then only need to modify forward
    // pass with the copy
    if ((allowForward && (mode == DerivativeMode::ReverseModePrimal ||
                          mode == DerivativeMode::ReverseModeCombined)) ||
        (backwardsShadow && mode == DerivativeMode::ReverseModeGradient)) {
      assert(!shadowsLookedUp);

      // It is questionable how the following case would even occur, but if
      // the dst is constant, we shouldn't do anything extra
      if (dstConstant) {
        return;
      }

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
            dsto->getType()->getPointerElementType(), dsto, offset);
#else
        dsto = BuilderZ.CreateConstInBoundsGEP1_64(dsto, offset);
#endif
      }
      auto srco = shadow_src;
      if (srco->getType()->isIntegerTy())
        srco = BuilderZ.CreateIntToPtr(srco,
                                       Type::getInt8PtrTy(MTI->getContext()));
      if (offset != 0) {
#if LLVM_VERSION_MAJOR > 7
        srco = BuilderZ.CreateConstInBoundsGEP1_64(
            srco->getType()->getPointerElementType(), srco, offset);
#else
        srco = BuilderZ.CreateConstInBoundsGEP1_64(srco, offset);
#endif
      }
      Value *args[] = {
        dsto,
        srco,
        length,
#if LLVM_VERSION_MAJOR <= 6
        ConstantInt::get(Type::getInt32Ty(MTI->getContext()),
                         max(1U, min(srcalign, dstalign))),
#endif
        isVolatile
      };

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
