//===- GradientUtils.h - Helper class and utilities for AD       ---------===//
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
// This file declares two helper classes GradientUtils and subclass
// DiffeGradientUtils. These classes contain utilities for managing the cache,
// recomputing statements, and in the case of DiffeGradientUtils, managing
// adjoint values and shadow pointers.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_GUTILS_H_
#define ENZYME_GUTILS_H_

#include <algorithm>
#include <deque>
#include <map>

#include <llvm/Config/llvm-config.h>

#include "ActivityAnalysis.h"
#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"
#include "Utils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"

#include "llvm/IR/Dominators.h"

#include "MustExitScalarEvolution.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/Casting.h"

#include "llvm/Transforms/Utils/ValueMapper.h"

#include "ActivityAnalysis.h"
#include "CacheUtility.h"
#include "EnzymeLogic.h"
#include "LibraryFuncs.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#include "llvm-c/Core.h"

extern std::map<std::string, std::function<llvm::Value *(
                                 IRBuilder<> &, CallInst *, ArrayRef<Value *>)>>
    shadowHandlers;

class GradientUtils;
class DiffeGradientUtils;
extern std::map<
    std::string,
    std::pair<std::function<void(llvm::IRBuilder<> &, llvm::CallInst *,
                                 GradientUtils &, llvm::Value *&,
                                 llvm::Value *&, llvm::Value *&)>,
              std::function<void(llvm::IRBuilder<> &, llvm::CallInst *,
                                 DiffeGradientUtils &, llvm::Value *)>>>
    customCallHandlers;

extern std::map<
    std::string,
    std::function<void(llvm::IRBuilder<> &, llvm::CallInst *, GradientUtils &,
                       llvm::Value *&, llvm::Value *&)>>
    customFwdCallHandlers;

extern "C" {
extern llvm::cl::opt<bool> EnzymeInactiveDynamic;
extern llvm::cl::opt<bool> EnzymeFreeInternalAllocations;
}

struct InvertedPointerConfig : ValueMapConfig<const llvm::Value *> {
  typedef GradientUtils *ExtraData;
  static void onDelete(ExtraData gutils, const llvm::Value *old);
};

class InvertedPointerVH : public llvm::CallbackVH {
public:
  GradientUtils *gutils;
  InvertedPointerVH(GradientUtils *gutils) : gutils(gutils) {}
  InvertedPointerVH(GradientUtils *gutils, llvm::Value *V)
      : InvertedPointerVH(gutils) {
    setValPtr(V);
  }
  void deleted() override final;

  void allUsesReplacedWith(Value *new_value) override final {
    setValPtr(new_value);
  }
  virtual ~InvertedPointerVH() {}
};

static bool isPotentialLastLoopValue(Value *val, const BasicBlock *loc,
                                     const LoopInfo &LI) {
  if (Instruction *inst = dyn_cast<Instruction>(val)) {
    const Loop *InstLoop = LI.getLoopFor(inst->getParent());
    if (InstLoop == nullptr) {
      return false;
    }
    for (const Loop *L = LI.getLoopFor(loc); L; L = L->getParentLoop()) {
      if (L == InstLoop)
        return false;
    }
    return true;
  }
  return false;
}

enum class AugmentedStruct;
class GradientUtils : public CacheUtility {
public:
  EnzymeLogic &Logic;
  bool AtomicAdd;
  DerivativeMode mode;
  llvm::Function *oldFunc;
  llvm::ValueMap<const Value *, InvertedPointerVH> invertedPointers;
  DominatorTree &OrigDT;
  PostDominatorTree &OrigPDT;
  LoopInfo &OrigLI;
  ScalarEvolution &OrigSE;

  // (Original) Blocks which dominate all returns
  SmallPtrSet<BasicBlock *, 4> BlocksDominatingAllReturns;

  SmallPtrSet<BasicBlock *, 4> notForAnalysis;
  std::shared_ptr<ActivityAnalyzer> ATA;
  SmallVector<BasicBlock *, 12> originalBlocks;

  // Allocations which are known to always be freed before the
  // reverse, to the list of frees that must apply to this allocation.
  ValueMap<const CallInst *, SmallPtrSet<const CallInst *, 1>>
      allocationsWithGuaranteedFree;

  // Frees which can always be eliminated as the post dominate
  // an allocation (which will itself be freed).
  SmallPtrSet<const CallInst *, 1> postDominatingFrees;

  // Deallocations that should be kept in the forward pass because
  // they deallocation memory which isn't necessary for the reverse
  // pass
  SmallPtrSet<const CallInst *, 1> forwardDeallocations;

  // Map of primal block to corresponding block(s) in reverse
  std::map<BasicBlock *, std::vector<BasicBlock *>> reverseBlocks;
  // Map of block in reverse to corresponding primal block
  std::map<BasicBlock *, BasicBlock *> reverseBlockToPrimal;

  // A set of tape extractions to enforce a cache of
  // rather than attempting to recompute.
  SmallPtrSet<Instruction *, 4> TapesToPreventRecomputation;

  ValueMap<PHINode *, WeakTrackingVH> fictiousPHIs;
  ValueToValueMapTy originalToNewFn;
  ValueToValueMapTy newToOriginalFn;
  std::vector<CallInst *> originalCalls;

  SmallPtrSet<Instruction *, 4> unnecessaryIntermediates;

  const std::map<Instruction *, bool> *can_modref_map;

  Value *getNewIfOriginal(Value *originst) const {
    assert(originst);
    auto f = originalToNewFn.find(originst);
    if (f == originalToNewFn.end()) {
      return originst;
    }
    assert(f != originalToNewFn.end());
    if (f->second == nullptr) {
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *originst << "\n";
    }
    assert(f->second);
    return f->second;
  }

  Value *tid;
  Value *ompThreadId() {
    if (tid)
      return tid;
    IRBuilder<> B(inversionAllocs);

    auto FT = FunctionType::get(Type::getInt64Ty(B.getContext()),
                                ArrayRef<Type *>(), false);
    AttributeList AL;
#if LLVM_VERSION_MAJOR >= 14
    AL = AL.addAttributeAtIndex(B.getContext(), AttributeList::FunctionIndex,
                                Attribute::AttrKind::ReadOnly);
#else
    AL = AL.addAttribute(B.getContext(), AttributeList::FunctionIndex,
                         Attribute::AttrKind::ReadOnly);
#endif
    return tid = B.CreateCall(newFunc->getParent()->getOrInsertFunction(
               "omp_get_thread_num", FT, AL));
  }
  Value *numThreads;
  Value *ompNumThreads() {
    if (numThreads)
      return numThreads;
    IRBuilder<> B(inversionAllocs);

    auto FT = FunctionType::get(Type::getInt64Ty(B.getContext()),
                                ArrayRef<Type *>(), false);
    AttributeList AL;
#if LLVM_VERSION_MAJOR >= 14
    AL = AL.addAttributeAtIndex(B.getContext(), AttributeList::FunctionIndex,
                                Attribute::AttrKind::ReadOnly);
#else
    AL = AL.addAttribute(B.getContext(), AttributeList::FunctionIndex,
                         Attribute::AttrKind::ReadOnly);
#endif
    return numThreads = B.CreateCall(newFunc->getParent()->getOrInsertFunction(
               "omp_get_max_threads", FT, AL));
  }

  Value *getOrInsertTotalMultiplicativeProduct(Value *val, LoopContext &lc) {
    // TODO optimize if val is invariant to loopContext
    assert(val->getType()->isFPOrFPVectorTy());
    for (auto &I : *lc.header) {
      if (auto PN = dyn_cast<PHINode>(&I)) {
        if (PN->getType() != val->getType())
          continue;
        Value *ival = PN->getIncomingValueForBlock(lc.preheader);
        if (auto CDV = dyn_cast<ConstantDataVector>(ival)) {
          if (CDV->isSplat())
            ival = CDV->getSplatValue();
        }
        if (auto C = dyn_cast<ConstantFP>(ival)) {
          if (!C->isExactlyValue(
                  APFloat(C->getType()->getFltSemantics(), "1"))) {
            continue;
          }
        } else
          continue;
        for (auto IB : PN->blocks()) {
          if (IB == lc.preheader)
            continue;

          if (auto BO =
                  dyn_cast<BinaryOperator>(PN->getIncomingValueForBlock(IB))) {
            if (BO->getOpcode() != BinaryOperator::FMul)
              goto continueOutermost;
            if (BO->getOperand(0) == PN && BO->getOperand(1) == val)
              return BO;
            if (BO->getOperand(1) == PN && BO->getOperand(0) == val)
              return BO;
          } else
            goto continueOutermost;
        }
      } else
        break;
    continueOutermost:;
    }

    IRBuilder<> lbuilder(lc.header, lc.header->begin());
    auto PN = lbuilder.CreatePHI(val->getType(), 2);
    Constant *One = ConstantFP::get(val->getType()->getScalarType(), "1");
    if (VectorType *VTy = dyn_cast<VectorType>(val->getType())) {
#if LLVM_VERSION_MAJOR >= 11
      One = ConstantVector::getSplat(VTy->getElementCount(), One);
#else
      One = ConstantVector::getSplat(VTy->getNumElements(), One);
#endif
    }
    PN->addIncoming(One, lc.preheader);
    lbuilder.SetInsertPoint(lc.header->getFirstNonPHI());
    if (auto inst = dyn_cast<Instruction>(val)) {
      if (DT.dominates(PN, inst))
        lbuilder.SetInsertPoint(inst->getNextNode());
    }
    Value *red = lbuilder.CreateFMul(PN, val);
    for (auto pred : predecessors(lc.header)) {
      if (pred == lc.preheader)
        continue;
      PN->addIncoming(red, pred);
    }
    return red;
  }

  Value *getOrInsertConditionalIndex(Value *val, LoopContext &lc,
                                     bool pickTrue) {
    assert(val->getType()->isIntOrIntVectorTy(1));
    // TODO optimize if val is invariant to loopContext
    for (auto &I : *lc.header) {
      if (auto PN = dyn_cast<PHINode>(&I)) {
        if (PN->getNumIncomingValues() == 0)
          continue;
        if (PN->getType() != lc.incvar->getType())
          continue;
        Value *ival = PN->getIncomingValueForBlock(lc.preheader);
        if (auto C = dyn_cast<Constant>(ival)) {
          if (!C->isNullValue()) {
            continue;
          }
        } else
          continue;
        for (auto IB : PN->blocks()) {
          if (IB == lc.preheader)
            continue;

          if (auto SI =
                  dyn_cast<SelectInst>(PN->getIncomingValueForBlock(IB))) {
            if (SI->getCondition() != val)
              goto continueOutermost;
            if (pickTrue && SI->getFalseValue() == PN) {
              // TODO handle vector of
              if (SI->getTrueValue() == lc.incvar)
                return SI;
            }
            if (!pickTrue && SI->getTrueValue() == PN) {
              // TODO handle vector of
              if (SI->getFalseValue() == lc.incvar)
                return SI;
            }
          } else
            goto continueOutermost;
        }
      } else
        break;
    continueOutermost:;
    }

    IRBuilder<> lbuilder(lc.header, lc.header->begin());
    auto PN = lbuilder.CreatePHI(lc.incvar->getType(), 2);
    Constant *Zero =
        Constant::getNullValue(lc.incvar->getType()->getScalarType());
    PN->addIncoming(Zero, lc.preheader);
    lbuilder.SetInsertPoint(lc.incvar->getNextNode());
    Value *red = lc.incvar;
    if (VectorType *VTy = dyn_cast<VectorType>(val->getType())) {
#if LLVM_VERSION_MAJOR >= 12
      red = lbuilder.CreateVectorSplat(VTy->getElementCount(), red);
#else
      red = lbuilder.CreateVectorSplat(VTy->getNumElements(), red);
#endif
    }
    if (auto inst = dyn_cast<Instruction>(val)) {
      if (DT.dominates(PN, inst))
        lbuilder.SetInsertPoint(inst->getNextNode());
    }
    assert(red->getType() == PN->getType());
    red = lbuilder.CreateSelect(val, pickTrue ? red : PN, pickTrue ? PN : red);
    for (auto pred : predecessors(lc.header)) {
      if (pred == lc.preheader)
        continue;
      PN->addIncoming(red, pred);
    }
    return red;
  }

  bool assumeDynamicLoopOfSizeOne(llvm::Loop *L) const override {
    if (!EnzymeInactiveDynamic)
      return false;
    auto OL = OrigLI.getLoopFor(isOriginal(L->getHeader()));
    assert(OL);
    for (auto OB : OL->getBlocks()) {
      for (auto &OI : *OB) {
        if (!isConstantInstruction(&OI))
          return false;
      }
    }
    return true;
  }

  llvm::DebugLoc getNewFromOriginal(const llvm::DebugLoc L) const {
    if (L.get() == nullptr)
      return nullptr;
    if (!oldFunc->getSubprogram())
      return L;
    assert(originalToNewFn.hasMD());
    auto opt = originalToNewFn.getMappedMD(L.getAsMDNode());
    if (!opt.hasValue())
      return L;
    assert(opt.hasValue());
    return llvm::DebugLoc(cast<MDNode>(*opt.getPointer()));
  }

  Value *getNewFromOriginal(const Value *originst) const {
    assert(originst);
    if (isa<ConstantData>(originst))
      return const_cast<Value *>(originst);
    auto f = originalToNewFn.find(originst);
    if (f == originalToNewFn.end()) {
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
      dumpMap(originalToNewFn, [&](const Value *const &v) -> bool {
        if (isa<Instruction>(originst))
          return isa<Instruction>(v);
        if (isa<BasicBlock>(originst))
          return isa<BasicBlock>(v);
        if (isa<Function>(originst))
          return isa<Function>(v);
        if (isa<Argument>(originst))
          return isa<Argument>(v);
        if (isa<Constant>(originst))
          return isa<Constant>(v);
        return true;
      });
      llvm::errs() << *originst << "\n";
    }
    assert(f != originalToNewFn.end());
    if (f->second == nullptr) {
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *originst << "\n";
    }
    assert(f->second);
    return f->second;
  }
  Instruction *getNewFromOriginal(const Instruction *newinst) const {
    auto ninst = getNewFromOriginal((Value *)newinst);
    if (!isa<Instruction>(ninst)) {
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *ninst << " - " << *newinst << "\n";
    }
    return cast<Instruction>(ninst);
  }
  BasicBlock *getNewFromOriginal(const BasicBlock *newinst) const {
    return cast<BasicBlock>(getNewFromOriginal((Value *)newinst));
  }

  Value *hasUninverted(const Value *inverted) const {
    for (auto v : invertedPointers) {
      if (v.second == inverted)
        return const_cast<Value *>(v.first);
    }
    return nullptr;
  }
  BasicBlock *getOriginalFromNew(const BasicBlock *newinst) const {
    assert(newinst->getParent() == newFunc);
    auto found = newToOriginalFn.find(newinst);
    assert(found != newToOriginalFn.end());
    return cast<BasicBlock>(found->second);
  }

  Value *isOriginal(const Value *newinst) const {
    if (isa<Constant>(newinst) || isa<UndefValue>(newinst))
      return const_cast<Value *>(newinst);
    if (auto arg = dyn_cast<Argument>(newinst)) {
      assert(arg->getParent() == newFunc);
    }
    if (auto inst = dyn_cast<Instruction>(newinst)) {
      assert(inst->getParent()->getParent() == newFunc);
    }
    auto found = newToOriginalFn.find(newinst);
    if (found == newToOriginalFn.end())
      return nullptr;
    return found->second;
  }

  Instruction *isOriginal(const Instruction *newinst) const {
    return cast_or_null<Instruction>(isOriginal((const Value *)newinst));
  }
  BasicBlock *isOriginal(const BasicBlock *newinst) const {
    return cast_or_null<BasicBlock>(isOriginal((const Value *)newinst));
  }

  void computeGuaranteedFrees(
      const llvm::SmallPtrSetImpl<BasicBlock *> &oldUnreachable) {
    for (auto &BB : *oldFunc) {
      if (oldUnreachable.count(&BB))
        continue;
      for (auto &I : BB) {
        auto CI = dyn_cast<CallInst>(&I);
        if (!CI)
          continue;

        Function *called = getFunctionFromCall(CI);
        if (!called)
          continue;
        if (isDeallocationFunction(*called, TLI)) {

          llvm::Value *val = CI->getArgOperand(0);
          while (auto cast = dyn_cast<CastInst>(val))
            val = cast->getOperand(0);

          if (auto dc = dyn_cast<CallInst>(val)) {
            if (dc->getCalledFunction() &&
                isAllocationFunction(*dc->getCalledFunction(), TLI)) {

              bool hasPDFree = false;
              if (dc->getParent() == CI->getParent() ||
                  OrigPDT.dominates(CI->getParent(), dc->getParent())) {
                hasPDFree = true;
              }

              if (hasPDFree) {
                allocationsWithGuaranteedFree[dc].insert(CI);
              }
            }
          }
        }
        if (isAllocationFunction(*called, TLI)) {
          if (hasMetadata(CI, "enzyme_fromstack")) {
            allocationsWithGuaranteedFree[CI].insert(CI);
          }
        }
      }
    }
  }

private:
  SmallVector<WeakTrackingVH, 4> addedTapeVals;
  unsigned tapeidx;
  Value *tape;

  std::map<BasicBlock *,
           ValueMap<Value *, std::map<BasicBlock *, WeakTrackingVH>>>
      unwrap_cache;
  std::map<BasicBlock *, ValueMap<Value *, WeakTrackingVH>> lookup_cache;

public:
  BasicBlock *addReverseBlock(BasicBlock *currentBlock, Twine name,
                              bool forkCache = true) {
    assert(reverseBlocks.size());
    auto found = reverseBlockToPrimal.find(currentBlock);
    assert(found != reverseBlockToPrimal.end());

    std::vector<BasicBlock *> &vec = reverseBlocks[found->second];
    assert(vec.size());
    assert(vec.back() == currentBlock);

    BasicBlock *rev =
        BasicBlock::Create(currentBlock->getContext(), name, newFunc);
    rev->moveAfter(currentBlock);
    vec.push_back(rev);
    reverseBlockToPrimal[rev] = found->second;
    if (forkCache) {
      for (auto pair : unwrap_cache[currentBlock])
        unwrap_cache[rev].insert(pair);
      for (auto pair : lookup_cache[currentBlock])
        lookup_cache[rev].insert(pair);
    }
    return rev;
  }

public:
  bool legalRecompute(const Value *val, const ValueToValueMapTy &available,
                      IRBuilder<> *BuilderM, bool reverse = false,
                      bool legalRecomputeCache = true) const;
  std::map<const Value *, bool> knownRecomputeHeuristic;
  bool shouldRecompute(const Value *val, const ValueToValueMapTy &available,
                       IRBuilder<> *BuilderM);

  ValueMap<const Instruction *, AssertingReplacingVH> unwrappedLoads;
  void replaceAWithB(Value *A, Value *B, bool storeInCache = false) override {
    if (A == B)
      return;
    assert(A->getType() == B->getType());

    if (auto iA = dyn_cast<Instruction>(A)) {
      if (unwrappedLoads.find(iA) != unwrappedLoads.end()) {
        auto iB = cast<Instruction>(B);
        unwrappedLoads[iB] = unwrappedLoads[iA];
        unwrappedLoads.erase(iA);
      }
    }

    // Check that the replacement doesn't already exist in the mapping
    // thereby resulting in a conflict.
    {
      auto found = newToOriginalFn.find(A);
      if (found != newToOriginalFn.end()) {
        auto foundB = newToOriginalFn.find(B);
        assert(foundB == newToOriginalFn.end());
      }
    }

    CacheUtility::replaceAWithB(A, B, storeInCache);
  }

  void erase(Instruction *I) override {
    assert(I);
    if (I->getParent()->getParent() != newFunc) {
      llvm::errs() << "newFunc: " << *newFunc << "\n";
      llvm::errs() << "paren: " << *I->getParent()->getParent() << "\n";
      llvm::errs() << "I: " << *I << "\n";
    }
    assert(I->getParent()->getParent() == newFunc);

    // not original, should not contain
    assert(!invertedPointers.count(I));
    // not original, should not contain
    assert(!originalToNewFn.count(I));

    originalToNewFn.erase(I);
    {
      auto found = newToOriginalFn.find(I);
      if (found != newToOriginalFn.end()) {
        Value *orig = found->second;
        newToOriginalFn.erase(found);
        originalToNewFn.erase(orig);
      }
    }
    {
      auto found = UnwrappedWarnings.find(I);
      if (found != UnwrappedWarnings.end()) {
        UnwrappedWarnings.erase(found);
      }
    }
    unwrappedLoads.erase(I);

    for (auto &pair : unwrap_cache) {
      if (pair.second.find(I) != pair.second.end())
        pair.second.erase(I);
    }

    for (auto &pair : lookup_cache) {
      if (pair.second.find(I) != pair.second.end())
        pair.second.erase(I);
    }
    CacheUtility::erase(I);
  }
  // TODO consider invariant group and/or valueInvariant group

  void setTape(Value *newtape) {
    assert(tape == nullptr);
    assert(newtape != nullptr);
    assert(tapeidx == 0);
    assert(addedTapeVals.size() == 0);
    tape = newtape;
  }

  void dumpPointers() {
    llvm::errs() << "invertedPointers:\n";
    for (auto a : invertedPointers) {
      llvm::errs() << "   invertedPointers[" << *a.first << "] = " << *a.second
                   << "\n";
    }
    llvm::errs() << "end invertedPointers\n";
  }

  Value *createAntiMalloc(CallInst *orig, unsigned idx) {
    assert(orig->getParent()->getParent() == oldFunc);
    auto found = invertedPointers.find(orig);
    PHINode *placeholder = cast<PHINode>(&*found->second);

    assert(placeholder->getParent()->getParent() == newFunc);
    placeholder->setName("");
    IRBuilder<> bb(placeholder);

    Function *Fn = orig->getCalledFunction();

#if LLVM_VERSION_MAJOR >= 11
    if (auto castinst = dyn_cast<ConstantExpr>(orig->getCalledOperand()))
#else
    if (auto castinst = dyn_cast<ConstantExpr>(orig->getCalledValue()))
#endif
    {
      if (castinst->isCast())
        if (auto fn = dyn_cast<Function>(castinst->getOperand(0)))
          Fn = fn;
    }
    assert(Fn);

    SmallVector<Value *, 8> args;
#if LLVM_VERSION_MAJOR >= 14
    for (auto &arg : orig->args())
#else
    for (auto &arg : orig->arg_operands())
#endif
    {
      args.push_back(getNewFromOriginal(arg));
    }

    if (shadowHandlers.find(Fn->getName().str()) != shadowHandlers.end()) {
      bb.SetInsertPoint(placeholder);
      Value *anti = placeholder;

      if (mode != DerivativeMode::ReverseModeGradient) {
        anti = shadowHandlers[Fn->getName().str()](bb, orig, args);

        invertedPointers.erase(found);
        bb.SetInsertPoint(placeholder);

        replaceAWithB(placeholder, anti);
        erase(placeholder);
      }

      if (auto inst = dyn_cast<Instruction>(anti))
        bb.SetInsertPoint(inst);

      anti = cacheForReverse(bb, anti, idx);
      invertedPointers.insert(
          std::make_pair(orig, InvertedPointerVH(this, anti)));
      return anti;
    }

#if LLVM_VERSION_MAJOR >= 11
    Value *anti =
        bb.CreateCall(orig->getFunctionType(), orig->getCalledOperand(), args,
                      orig->getName() + "'mi");
#else
    Value *anti =
        bb.CreateCall(orig->getCalledValue(), args, orig->getName() + "'mi");
#endif
    cast<CallInst>(anti)->setAttributes(orig->getAttributes());
    cast<CallInst>(anti)->setCallingConv(orig->getCallingConv());
    cast<CallInst>(anti)->setTailCallKind(orig->getTailCallKind());
    cast<CallInst>(anti)->setDebugLoc(getNewFromOriginal(orig->getDebugLoc()));

#if LLVM_VERSION_MAJOR >= 14
    cast<CallInst>(anti)->addAttributeAtIndex(AttributeList::ReturnIndex,
                                              Attribute::NoAlias);
    cast<CallInst>(anti)->addAttributeAtIndex(AttributeList::ReturnIndex,
                                              Attribute::NonNull);
#else
    cast<CallInst>(anti)->addAttribute(AttributeList::ReturnIndex,
                                       Attribute::NoAlias);
    cast<CallInst>(anti)->addAttribute(AttributeList::ReturnIndex,
                                       Attribute::NonNull);
#endif
    unsigned derefBytes = 0;
    if (Fn->getName() == "malloc" || Fn->getName() == "_Znwm") {
      if (auto ci = dyn_cast<ConstantInt>(args[0])) {
        derefBytes = ci->getLimitedValue();
        CallInst *cal = cast<CallInst>(getNewFromOriginal(orig));
#if LLVM_VERSION_MAJOR >= 14
        cast<CallInst>(anti)->addDereferenceableRetAttr(ci->getLimitedValue());
        cal->addDereferenceableRetAttr(ci->getLimitedValue());
        AttrBuilder B(Fn->getContext());
        B.addDereferenceableOrNullAttr(ci->getLimitedValue());
        cast<CallInst>(anti)->setAttributes(
            cast<CallInst>(anti)->getAttributes().addRetAttributes(
                orig->getContext(), B));
        cal->setAttributes(
            cal->getAttributes().addRetAttributes(orig->getContext(), B));
        cal->addAttributeAtIndex(AttributeList::ReturnIndex,
                                 Attribute::NoAlias);
        cal->addAttributeAtIndex(AttributeList::ReturnIndex,
                                 Attribute::NonNull);
#else
        cast<CallInst>(anti)->addDereferenceableAttr(
            llvm::AttributeList::ReturnIndex, ci->getLimitedValue());
        cal->addDereferenceableAttr(llvm::AttributeList::ReturnIndex,
                                    ci->getLimitedValue());
        cast<CallInst>(anti)->addDereferenceableOrNullAttr(
            llvm::AttributeList::ReturnIndex, ci->getLimitedValue());
        cal->addDereferenceableOrNullAttr(llvm::AttributeList::ReturnIndex,
                                          ci->getLimitedValue());
        cal->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
        cal->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
#endif
      }
    }

    invertedPointers.erase(found);
    bb.SetInsertPoint(placeholder->getNextNode());
    replaceAWithB(placeholder, anti);
    erase(placeholder);

    anti = cacheForReverse(bb, anti, idx);
    invertedPointers.insert(
        std::make_pair((const Value *)orig, InvertedPointerVH(this, anti)));

    if (tape == nullptr) {
      if (Fn->getName() == "julia.gc_alloc_obj") {
        Type *tys[] = {
            PointerType::get(StructType::get(orig->getContext()), 10)};
        FunctionType *FT =
            FunctionType::get(Type::getVoidTy(orig->getContext()), tys, true);
        bb.CreateCall(oldFunc->getParent()->getOrInsertFunction(
                          "julia.write_barrier", FT),
                      anti);
      }

      if (Fn->getName() == "swift_allocObject") {
        EmitFailure(
            "SwiftShadowAllocation", orig->getDebugLoc(), orig,
            "Haven't implemented shadow allocator for `swift_allocObject`",
            *orig);
        return anti;
      }

      Value *dst_arg = anti;

      dst_arg = bb.CreateBitCast(
          dst_arg,
          Type::getInt8PtrTy(orig->getContext(),
                             anti->getType()->getPointerAddressSpace()));

      auto val_arg = ConstantInt::get(Type::getInt8Ty(orig->getContext()), 0);
      Value *size;
      // todo check if this memset is legal and if a write barrier is needed
      if (Fn->getName() == "julia.gc_alloc_obj") {
        size = args[1];
      } else {
        size = args[0];
      }
      auto len_arg =
          bb.CreateZExtOrTrunc(size, Type::getInt64Ty(orig->getContext()));
      auto volatile_arg = ConstantInt::getFalse(orig->getContext());

#if LLVM_VERSION_MAJOR == 6
      auto align_arg =
          ConstantInt::get(Type::getInt32Ty(orig->getContext()), 1);
      Value *nargs[] = {dst_arg, val_arg, len_arg, align_arg, volatile_arg};
#else
      Value *nargs[] = {dst_arg, val_arg, len_arg, volatile_arg};
#endif

      Type *tys[] = {dst_arg->getType(), len_arg->getType()};

      auto memset = cast<CallInst>(
          bb.CreateCall(Intrinsic::getDeclaration(newFunc->getParent(),
                                                  Intrinsic::memset, tys),
                        nargs));
      // memset->addParamAttr(0, Attribute::getWithAlignment(Context,
      // inst->getAlignment()));
      memset->addParamAttr(0, Attribute::NonNull);
      if (derefBytes) {
#if LLVM_VERSION_MAJOR >= 14
        memset->addDereferenceableParamAttr(0, derefBytes);
        memset->setAttributes(
            memset->getAttributes().addDereferenceableOrNullParamAttr(
                memset->getContext(), 0, derefBytes));
#else
        memset->addDereferenceableAttr(llvm::AttributeList::FirstArgIndex,
                                       derefBytes);
        memset->addDereferenceableOrNullAttr(llvm::AttributeList::FirstArgIndex,
                                             derefBytes);
#endif
      }
    }

    return anti;
  }

  int getIndex(std::pair<Instruction *, CacheType> idx,
               std::map<std::pair<Instruction *, CacheType>, int> &mapping) {
    if (tape) {
      if (mapping.find(idx) == mapping.end()) {
        llvm::errs() << "oldFunc: " << *oldFunc << "\n";
        llvm::errs() << "newFunc: " << *newFunc << "\n";
        llvm::errs() << " <mapping>\n";
        for (auto &p : mapping) {
          llvm::errs() << "   idx: " << *p.first.first << ", " << p.first.second
                       << " pos=" << p.second << "\n";
        }
        llvm::errs() << " </mapping>\n";

        if (mapping.find(idx) == mapping.end()) {
          llvm::errs() << "idx: " << *idx.first << ", " << idx.second << "\n";
          assert(0 && "could not find index in mapping");
        }
      }
      return mapping[idx];
    } else {
      if (mapping.find(idx) != mapping.end()) {
        return mapping[idx];
      }
      mapping[idx] = tapeidx;
      ++tapeidx;
      return mapping[idx];
    }
  }

  Value *cacheForReverse(IRBuilder<> &BuilderQ, Value *malloc, int idx,
                         bool ignoreType = false, bool replace = true);

  const SmallVectorImpl<WeakTrackingVH> &getTapeValues() const {
    return addedTapeVals;
  }

public:
  AAResults &OrigAA;
  TypeAnalysis &TA;
  bool omp;

private:
  unsigned width;

public:
  unsigned getWidth() { return width; }

public:
  GradientUtils(EnzymeLogic &Logic, Function *newFunc_, Function *oldFunc_,
                TargetLibraryInfo &TLI_, TypeAnalysis &TA_,
                ValueToValueMapTy &invertedPointers_,
                const SmallPtrSetImpl<Value *> &constantvalues_,
                const SmallPtrSetImpl<Value *> &activevals_,
                DIFFE_TYPE ReturnActivity, ValueToValueMapTy &originalToNewFn_,
                DerivativeMode mode, unsigned width, bool omp)
      : CacheUtility(TLI_, newFunc_), Logic(Logic), mode(mode),
        oldFunc(oldFunc_), invertedPointers(),
        OrigDT(Logic.PPC.FAM.getResult<llvm::DominatorTreeAnalysis>(*oldFunc_)),
        OrigPDT(Logic.PPC.FAM.getResult<llvm::PostDominatorTreeAnalysis>(
            *oldFunc_)),
        OrigLI(Logic.PPC.FAM.getResult<llvm::LoopAnalysis>(*oldFunc_)),
        OrigSE(
            Logic.PPC.FAM.getResult<llvm::ScalarEvolutionAnalysis>(*oldFunc_)),
        notForAnalysis(getGuaranteedUnreachable(oldFunc_)),
        ATA(new ActivityAnalyzer(Logic.PPC,
                                 Logic.PPC.getAAResultsFromFunction(oldFunc_),
                                 notForAnalysis, TLI_, constantvalues_,
                                 activevals_, ReturnActivity)),
        tid(nullptr), numThreads(nullptr),
        OrigAA(Logic.PPC.getAAResultsFromFunction(oldFunc_)), TA(TA_), omp(omp),
        width(width) {
    if (oldFunc_->getSubprogram()) {
      assert(originalToNewFn_.hasMD());
    }

    for (BasicBlock &BB : *oldFunc) {
      for (Instruction &I : BB) {
        if (auto CI = dyn_cast<CallInst>(&I)) {
          originalCalls.push_back(CI);
        }
      }
    }

    originalToNewFn.getMDMap() = originalToNewFn_.getMDMap();

    if (oldFunc_->getSubprogram()) {
      assert(originalToNewFn.hasMD());
    }
#if LLVM_VERSION_MAJOR <= 6
    OrigPDT.recalculate(*oldFunc_);
#endif
    for (auto pair : invertedPointers_) {
      invertedPointers.insert(std::make_pair(
          (const Value *)pair.first, InvertedPointerVH(this, pair.second)));
    }
    originalToNewFn.insert(originalToNewFn_.begin(), originalToNewFn_.end());
    for (BasicBlock &oBB : *oldFunc) {
      for (Instruction &oI : oBB) {
        newToOriginalFn[originalToNewFn[&oI]] = &oI;
      }
      newToOriginalFn[originalToNewFn[&oBB]] = &oBB;
    }
    for (Argument &oArg : oldFunc->args()) {
      newToOriginalFn[originalToNewFn[&oArg]] = &oArg;
    }
    for (BasicBlock &BB : *newFunc) {
      originalBlocks.emplace_back(&BB);
    }
    tape = nullptr;
    tapeidx = 0;
    assert(originalBlocks.size() > 0);

    SmallVector<BasicBlock *, 4> ReturningBlocks;
    for (BasicBlock &BB : *oldFunc) {
      if (isa<ReturnInst>(BB.getTerminator()))
        ReturningBlocks.push_back(&BB);
    }
    for (BasicBlock &BB : *oldFunc) {
      bool legal = true;
      for (auto BRet : ReturningBlocks) {
        if (!(BRet == &BB || OrigDT.dominates(&BB, BRet))) {
          legal = false;
          break;
        }
      }
      if (legal)
        BlocksDominatingAllReturns.insert(&BB);
    }
  }

public:
  static GradientUtils *
  CreateFromClone(EnzymeLogic &Logic, Function *todiff, TargetLibraryInfo &TLI,
                  TypeAnalysis &TA, DIFFE_TYPE retType,
                  const std::vector<DIFFE_TYPE> &constant_args, bool returnUsed,
                  std::map<AugmentedStruct, int> &returnMapping, bool omp);

#if LLVM_VERSION_MAJOR >= 10
  void setPtrDiffe(Value *ptr, Value *newval, IRBuilder<> &BuilderM,
                   MaybeAlign align, bool isVolatile, AtomicOrdering ordering,
                   SyncScope::ID syncScope, Value *mask = nullptr)
#else
  void setPtrDiffe(Value *ptr, Value *newval, IRBuilder<> &BuilderM,
                   unsigned align, bool isVolatile, AtomicOrdering ordering,
                   SyncScope::ID syncScope, Value *mask = nullptr)
#endif
  {
    if (auto inst = dyn_cast<Instruction>(ptr)) {
      assert(inst->getParent()->getParent() == oldFunc);
    }
    if (auto arg = dyn_cast<Argument>(ptr)) {
      assert(arg->getParent() == oldFunc);
    }

    ptr = invertPointerM(ptr, BuilderM);
    if (!isOriginalBlock(*BuilderM.GetInsertBlock()))
      ptr = lookupM(ptr, BuilderM);

    if (!mask) {
      auto ts = BuilderM.CreateStore(newval, ptr);
      if (align)
#if LLVM_VERSION_MAJOR >= 10
        ts->setAlignment(*align);
#else
        ts->setAlignment(align);
#endif
      ts->setVolatile(isVolatile);
      ts->setOrdering(ordering);
      ts->setSyncScopeID(syncScope);
    } else {
      if (!isOriginalBlock(*BuilderM.GetInsertBlock()))
        mask = lookupM(mask, BuilderM);
      Type *tys[] = {newval->getType(), ptr->getType()};
      auto F = Intrinsic::getDeclaration(oldFunc->getParent(),
                                         Intrinsic::masked_store, tys);
      assert(align);
#if LLVM_VERSION_MAJOR >= 10
      Value *alignv =
          ConstantInt::get(Type::getInt32Ty(ptr->getContext()), align->value());
#else
      Value *alignv =
          ConstantInt::get(Type::getInt32Ty(ptr->getContext()), align);
#endif
      Value *args[] = {newval, ptr, alignv, mask};
      BuilderM.CreateCall(F, args)->setCallingConv(F->getCallingConv());
    }
  }

private:
  BasicBlock *originalForReverseBlock(BasicBlock &BB2) const {
    auto found = reverseBlockToPrimal.find(&BB2);
    if (found == reverseBlockToPrimal.end()) {
      llvm::errs() << "newFunc: " << *newFunc << "\n";
      llvm::errs() << BB2 << "\n";
    }
    assert(found != reverseBlockToPrimal.end());
    return found->second;
  }

public:
  //! This cache stores blocks we may insert as part of getReverseOrLatchMerge
  //! to handle inverse iv iteration
  //  As we don't want to create redundant blocks, we use this convenient cache
  std::map<std::tuple<BasicBlock *, BasicBlock *>, BasicBlock *>
      newBlocksForLoop_cache;
  BasicBlock *getReverseOrLatchMerge(BasicBlock *BB,
                                     BasicBlock *branchingBlock);

  void forceContexts();

  void
  computeMinCache(TypeResults &TR,
                  const SmallPtrSetImpl<BasicBlock *> &guaranteedUnreachable);

  bool isOriginalBlock(const BasicBlock &BB) const {
    for (auto A : originalBlocks) {
      if (A == &BB)
        return true;
    }
    return false;
  }

  void eraseFictiousPHIs() {
    std::vector<std::pair<PHINode *, Value *>> phis;
    for (auto pair : fictiousPHIs)
      phis.emplace_back(pair.first, pair.second);
    fictiousPHIs.clear();

    for (auto pair : phis) {
      auto pp = pair.first;
      if (pp->getNumUses() != 0) {
        llvm::errs() << "mod:" << *oldFunc->getParent() << "\n";
        llvm::errs() << "oldFunc:" << *oldFunc << "\n";
        llvm::errs() << "newFunc:" << *newFunc << "\n";
        llvm::errs() << " pp: " << *pp << " of " << *pair.second << "\n";
      }
      assert(pp->getNumUses() == 0);
      pp->replaceAllUsesWith(UndefValue::get(pp->getType()));
      erase(pp);
    }
    fictiousPHIs.clear();
  }

  TypeResults *my_TR;
  void forceActiveDetection(TypeResults &TR) {
    my_TR = &TR;
    for (auto &Arg : oldFunc->args()) {
      ATA->isConstantValue(TR, &Arg);
    }

    for (BasicBlock &BB : *oldFunc) {
      for (Instruction &I : BB) {
        bool const_inst = ATA->isConstantInstruction(TR, &I);
        bool const_value = ATA->isConstantValue(TR, &I);

        if (EnzymePrintActivity)
          llvm::errs() << I << " cv=" << const_value << " ci=" << const_inst
                       << "\n";
      }
    }
  }

  bool isConstantValue(Value *val) const {
    if (auto inst = dyn_cast<Instruction>(val)) {
      assert(inst->getParent()->getParent() == oldFunc);
      return ATA->isConstantValue(*my_TR, val);
    }

    if (auto arg = dyn_cast<Argument>(val)) {
      assert(arg->getParent() == oldFunc);
      return ATA->isConstantValue(*my_TR, val);
    }

    //! Functions must be false so we can replace function with augmentation,
    //! fallback to analysis
    if (isa<Function>(val) || isa<InlineAsm>(val) || isa<Constant>(val) ||
        isa<UndefValue>(val) || isa<MetadataAsValue>(val)) {
      // llvm::errs() << "calling icv on: " << *val << "\n";
      return ATA->isConstantValue(*my_TR, val);
    }

    if (auto gv = dyn_cast<GlobalVariable>(val)) {
      if (hasMetadata(gv, "enzyme_shadow"))
        return false;
      if (auto md = gv->getMetadata("enzyme_activity_value")) {
        auto res = cast<MDString>(md->getOperand(0))->getString();
        if (res == "const")
          return true;
        if (res == "active")
          return false;
      }
      if (EnzymeNonmarkedGlobalsInactive)
        return true;
      goto err;
    }
    if (isa<GlobalValue>(val)) {
      if (EnzymeNonmarkedGlobalsInactive)
        return true;
      goto err;
    }

  err:;
    llvm::errs() << *oldFunc << "\n";
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << *val << "\n";
    llvm::errs() << "  unknown did status attribute\n";
    assert(0 && "bad");
    exit(1);
  }

  bool isConstantInstruction(const Instruction *inst) const {
    assert(inst->getParent()->getParent() == oldFunc);
    return ATA->isConstantInstruction(*my_TR, const_cast<Instruction *>(inst));
  }

  bool getContext(llvm::BasicBlock *BB, LoopContext &lc) {
    return CacheUtility::getContext(BB, lc,
                                    /*ReverseLimit*/ reverseBlocks.size() > 0);
  }

  void forceAugmentedReturns(
      TypeResults &TR,
      const SmallPtrSetImpl<BasicBlock *> &guaranteedUnreachable) {
    assert(TR.getFunction() == oldFunc);

    for (BasicBlock &oBB : *oldFunc) {
      // Don't create derivatives for code that results in termination
      if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end())
        continue;

      LoopContext loopContext;
      getContext(cast<BasicBlock>(getNewFromOriginal(&oBB)), loopContext);

      for (Instruction &I : oBB) {
        Instruction *inst = &I;

        if (inst->getType()->isEmptyTy())
          continue;

        if (inst->getType()->isFPOrFPVectorTy())
          continue; //! op->getType()->isPointerTy() &&
                    //! !op->getType()->isIntegerTy()) {

        if (!TR.query(inst).Inner0().isPossiblePointer())
          continue;

        if (isa<LoadInst>(inst)) {
          IRBuilder<> BuilderZ(inst);
          getForwardBuilder(BuilderZ);

          PHINode *anti = BuilderZ.CreatePHI(inst->getType(), 1,
                                             inst->getName() + "'il_phi");
          invertedPointers.insert(std::make_pair(
              (const Value *)inst, InvertedPointerVH(this, anti)));
          continue;
        }

        if (!isa<CallInst>(inst)) {
          continue;
        }

        if (isa<IntrinsicInst>(inst)) {
          continue;
        }

        if (isConstantValue(inst)) {
          continue;
        }

        CallInst *op = cast<CallInst>(inst);
        Function *called = op->getCalledFunction();

        if (called && isCertainPrintOrFree(called)) {
          continue;
        }

        IRBuilder<> BuilderZ(inst);
        getForwardBuilder(BuilderZ);

        PHINode *anti =
            BuilderZ.CreatePHI(op->getType(), 1, op->getName() + "'ip_phi");
        invertedPointers.insert(
            std::make_pair((const Value *)inst, InvertedPointerVH(this, anti)));

        if (called && isAllocationFunction(*called, TLI)) {
          anti->setName(op->getName() + "'mi");
        }
      }
    }
  }

private:
  // For a given value, a list of basic blocks where an unwrap to has already
  // produced a warning.
  std::map<llvm::Instruction *, std::set<llvm::BasicBlock *>> UnwrappedWarnings;

public:
  /// if full unwrap, don't just unwrap this instruction, but also its operands,
  /// etc
  Value *unwrapM(Value *const val, IRBuilder<> &BuilderM,
                 const ValueToValueMapTy &available, UnwrapMode unwrapMode,
                 BasicBlock *scope = nullptr,
                 bool permitCache = true) override final;

  void ensureLookupCached(Instruction *inst, bool shouldFree = true) {
    assert(inst);
    if (scopeMap.find(inst) != scopeMap.end())
      return;
    if (shouldFree)
      assert(reverseBlocks.size());
    LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                      inst->getParent());

    AllocaInst *cache =
        createCacheForScope(lctx, inst->getType(), inst->getName(), shouldFree);
    assert(cache);
    Value *Val = inst;
    insert_or_assign(
        scopeMap, Val,
        std::pair<AssertingVH<AllocaInst>, LimitContext>(cache, lctx));
    storeInstructionInCache(lctx, inst, cache);
  }

  std::map<Instruction *, ValueMap<BasicBlock *, WeakTrackingVH>> lcssaFixes;
  std::map<PHINode *, WeakTrackingVH> lcssaPHIToOrig;
  Value *fixLCSSA(Instruction *inst, BasicBlock *forwardBlock,
                  bool legalInBlock = false) {
    assert(inst->getName() != "<badref>");

    if (auto lcssaPHI = dyn_cast<PHINode>(inst)) {
      auto found = lcssaPHIToOrig.find(lcssaPHI);
      if (found != lcssaPHIToOrig.end())
        inst = cast<Instruction>(found->second);
    }

    if (inst->getParent() == inversionAllocs)
      return inst;

    if (!isOriginalBlock(*forwardBlock)) {
      forwardBlock = originalForReverseBlock(*forwardBlock);
    }

    bool containsLastLoopValue =
        isPotentialLastLoopValue(inst, forwardBlock, LI);

    // If the instruction cannot represent a loop value, return the original
    // instruction if it either is guaranteed to be available within the block,
    // or it is not needed to guaranteed availability.
    if (!containsLastLoopValue) {
      if (!legalInBlock)
        return inst;
      if (forwardBlock == inst->getParent() || DT.dominates(inst, forwardBlock))
        return inst;
    }

    // llvm::errs() << " inst: " << *inst << "\n";
    // llvm::errs() << " seen: " << *inst->getParent() << "\n";
    assert(inst->getParent() != inversionAllocs);
    assert(isOriginalBlock(*inst->getParent()));

    if (lcssaFixes.find(inst) == lcssaFixes.end()) {
      lcssaFixes[inst][inst->getParent()] = inst;
      SmallPtrSet<BasicBlock *, 4> seen;
      std::deque<BasicBlock *> todo = {inst->getParent()};
      while (todo.size()) {
        BasicBlock *cur = todo.front();
        todo.pop_front();
        if (seen.count(cur))
          continue;
        seen.insert(cur);
        for (auto Succ : successors(cur)) {
          todo.push_back(Succ);
        }
      }
      for (auto &BB : *inst->getParent()->getParent()) {
        if (!seen.count(&BB) || (inst->getParent() != &BB &&
                                 DT.dominates(&BB, inst->getParent()))) {
          lcssaFixes[inst][&BB] = UndefValue::get(inst->getType());
        }
      }
    }

    if (lcssaFixes[inst].find(forwardBlock) != lcssaFixes[inst].end()) {
      return lcssaFixes[inst][forwardBlock];
    }

    // TODO replace forwardBlock with the first block dominated by inst,
    // that dominates (or is) forwardBlock to ensuring maximum reuse
    IRBuilder<> lcssa(&forwardBlock->front());
    auto lcssaPHI =
        lcssa.CreatePHI(inst->getType(), 1, inst->getName() + "!manual_lcssa");
    lcssaFixes[inst][forwardBlock] = lcssaPHI;
    lcssaPHIToOrig[lcssaPHI] = inst;
    for (auto pred : predecessors(forwardBlock)) {
      Value *val = nullptr;
      if (inst->getParent() == pred || DT.dominates(inst, pred)) {
        val = inst;
      }
      if (val == nullptr) {
        val = fixLCSSA(inst, pred, /*legalInBlock*/ true);
        assert(val->getType() == inst->getType());
      }
      assert(val->getType() == inst->getType());
      lcssaPHI->addIncoming(val, pred);
    }

    SmallPtrSet<Value *, 2> vals;
    SmallVector<Value *, 2> todo(lcssaPHI->incoming_values().begin(),
                                 lcssaPHI->incoming_values().end());
    while (todo.size()) {
      Value *v = todo.back();
      todo.pop_back();
      if (v == lcssaPHI)
        continue;
      vals.insert(v);
    }
    assert(vals.size() > 0);

    if (vals.size() > 1) {
      todo.append(vals.begin(), vals.end());
      vals.clear();
      while (todo.size()) {
        Value *v = todo.back();
        todo.pop_back();

        if (auto PN = dyn_cast<PHINode>(v))
          if (lcssaPHIToOrig.find(PN) != lcssaPHIToOrig.end()) {
            v = lcssaPHIToOrig[PN];
          }
        vals.insert(v);
      }
    }
    assert(vals.size() > 0);
    Value *val = nullptr;
    if (vals.size() == 1)
      val = *vals.begin();

    if (val && (!legalInBlock || !isa<Instruction>(val) ||
                DT.dominates(cast<Instruction>(val), lcssaPHI))) {

      if (!isPotentialLastLoopValue(val, forwardBlock, LI)) {
        bool nonSelfUse = false;
        for (auto u : lcssaPHI->users()) {
          if (u != lcssaPHI) {
            nonSelfUse = true;
            break;
          }
        }
        if (!nonSelfUse) {
          lcssaFixes[inst].erase(forwardBlock);
          while (lcssaPHI->getNumOperands())
            lcssaPHI->removeIncomingValue(lcssaPHI->getNumOperands() - 1,
                                          false);
          lcssaPHIToOrig.erase(lcssaPHI);
          lcssaPHI->eraseFromParent();
        }
        return val;
      }
    }
    return lcssaPHI;
  }

  Value *
  lookupM(Value *val, IRBuilder<> &BuilderM,
          const ValueToValueMapTy &incoming_availalble = ValueToValueMapTy(),
          bool tryLegalRecomputeCheck = true) override;

  Value *invertPointerM(Value *val, IRBuilder<> &BuilderM,
                        bool nullShadow = false);

  static Constant *
  GetOrCreateShadowConstant(EnzymeLogic &Logic, TargetLibraryInfo &TLI,
                            TypeAnalysis &TA, Constant *F, DerivativeMode mode,
                            unsigned width, bool AtomicAdd = true,
                            bool PostOpt = false);

  static Constant *
  GetOrCreateShadowFunction(EnzymeLogic &Logic, TargetLibraryInfo &TLI,
                            TypeAnalysis &TA, Function *F, DerivativeMode mode,
                            unsigned width, bool AtomicAdd = true,
                            bool PostOpt = false);

  void branchToCorrespondingTarget(
      BasicBlock *ctx, IRBuilder<> &BuilderM,
      const std::map<BasicBlock *,
                     std::vector<std::pair</*pred*/ BasicBlock *,
                                           /*successor*/ BasicBlock *>>>
          &targetToPreds,
      const std::map<BasicBlock *, PHINode *> *replacePHIs = nullptr);

  void getReverseBuilder(IRBuilder<> &Builder2, bool original = true) {
    assert(reverseBlocks.size());
    BasicBlock *BB = Builder2.GetInsertBlock();
    if (original)
      BB = getNewFromOriginal(BB);
    assert(reverseBlocks.find(BB) != reverseBlocks.end());
    BasicBlock *BB2 = reverseBlocks[BB].back();
    if (!BB2) {
      llvm::errs() << "oldFunc: " << oldFunc << "\n";
      llvm::errs() << "newFunc: " << newFunc << "\n";
      llvm::errs() << "could not invert " << *BB;
    }
    assert(BB2);

    if (BB2->getTerminator())
      Builder2.SetInsertPoint(BB2->getTerminator());
    else
      Builder2.SetInsertPoint(BB2);
    Builder2.SetCurrentDebugLocation(
        getNewFromOriginal(Builder2.getCurrentDebugLocation()));
    Builder2.setFastMathFlags(getFast());
  }

  void getForwardBuilder(IRBuilder<> &Builder2) {
    Instruction *insert = &*Builder2.GetInsertPoint();
    Instruction *nInsert = getNewFromOriginal(insert);

    assert(nInsert);

    Builder2.SetInsertPoint(getNextNonDebugInstruction(nInsert));
    Builder2.SetCurrentDebugLocation(
        getNewFromOriginal(Builder2.getCurrentDebugLocation()));
    Builder2.setFastMathFlags(getFast());
  }
};

class DiffeGradientUtils : public GradientUtils {
  DiffeGradientUtils(EnzymeLogic &Logic, Function *newFunc_, Function *oldFunc_,
                     TargetLibraryInfo &TLI, TypeAnalysis &TA,
                     ValueToValueMapTy &invertedPointers_,
                     const SmallPtrSetImpl<Value *> &constantvalues_,
                     const SmallPtrSetImpl<Value *> &returnvals_,
                     DIFFE_TYPE ActiveReturn, ValueToValueMapTy &origToNew_,
                     DerivativeMode mode, unsigned width, bool omp)
      : GradientUtils(Logic, newFunc_, oldFunc_, TLI, TA, invertedPointers_,
                      constantvalues_, returnvals_, ActiveReturn, origToNew_,
                      mode, width, omp) {
    assert(reverseBlocks.size() == 0);
    if (mode == DerivativeMode::ForwardMode ||
        mode == DerivativeMode::ForwardModeSplit) {
      return;
    }
    for (BasicBlock *BB : originalBlocks) {
      if (BB == inversionAllocs)
        continue;
      BasicBlock *RBB = BasicBlock::Create(BB->getContext(),
                                           "invert" + BB->getName(), newFunc);
      reverseBlocks[BB].push_back(RBB);
      reverseBlockToPrimal[RBB] = BB;
    }
    assert(reverseBlocks.size() != 0);
  }

public:
  // Whether to free memory in reverse pass or split forward.
  bool FreeMemory;
  ValueMap<const Value *, TrackingVH<AllocaInst>> differentials;
  static DiffeGradientUtils *
  CreateFromClone(EnzymeLogic &Logic, DerivativeMode mode, unsigned width,
                  Function *todiff, TargetLibraryInfo &TLI, TypeAnalysis &TA,
                  DIFFE_TYPE retType, bool diffeReturnArg,
                  const std::vector<DIFFE_TYPE> &constant_args,
                  ReturnType returnValue, Type *additionalArg, bool omp);

private:
  AllocaInst *getDifferential(Value *val) {
    assert(val);
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);
    assert(inversionAllocs);
    if (differentials.find(val) == differentials.end()) {
      IRBuilder<> entryBuilder(inversionAllocs);
      entryBuilder.setFastMathFlags(getFast());
      differentials[val] = entryBuilder.CreateAlloca(val->getType(), nullptr,
                                                     val->getName() + "'de");
      auto Alignment =
          oldFunc->getParent()->getDataLayout().getPrefTypeAlignment(
              val->getType());
#if LLVM_VERSION_MAJOR >= 10
      differentials[val]->setAlignment(Align(Alignment));
#else
      differentials[val]->setAlignment(Alignment);
#endif
      entryBuilder.CreateStore(Constant::getNullValue(val->getType()),
                               differentials[val]);
    }
    assert(cast<PointerType>(differentials[val]->getType())->getElementType() ==
           val->getType());
    return differentials[val];
  }

public:
  Value *diffe(Value *val, IRBuilder<> &BuilderM) {
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);

    if (isConstantValue(val)) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *val << "\n";
      assert(0 && "getting diffe of constant value");
    }
    if (val->getType()->isPointerTy()) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *val << "\n";
    }
    assert(!val->getType()->isPointerTy());
    assert(!val->getType()->isVoidTy());
#if LLVM_VERSION_MAJOR > 7
    return BuilderM.CreateLoad(val->getType(), getDifferential(val));
#else
    return BuilderM.CreateLoad(getDifferential(val));
#endif
  }

  // Returns created select instructions, if any
  std::vector<SelectInst *> addToDiffe(Value *val, Value *dif,
                                       IRBuilder<> &BuilderM, Type *addingType,
                                       ArrayRef<Value *> idxs = {},
                                       Value *mask = nullptr) {
    assert(mode == DerivativeMode::ReverseModeGradient ||
           mode == DerivativeMode::ReverseModeCombined);

    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);

    std::vector<SelectInst *> addedSelects;

    auto faddForNeg = [&](Value *old, Value *inc) {
      if (auto bi = dyn_cast<BinaryOperator>(inc)) {
        if (auto ci = dyn_cast<ConstantFP>(bi->getOperand(0))) {
          if (bi->getOpcode() == BinaryOperator::FSub && ci->isZero()) {
            return BuilderM.CreateFSub(old, bi->getOperand(1));
          }
        }
      }
      return BuilderM.CreateFAdd(old, inc);
    };

    auto faddForSelect = [&](Value *old, Value *dif) -> Value * {
      //! optimize fadd of select to select of fadd
      if (SelectInst *select = dyn_cast<SelectInst>(dif)) {
        if (Constant *ci = dyn_cast<Constant>(select->getTrueValue())) {
          if (ci->isZeroValue()) {
            SelectInst *res = cast<SelectInst>(BuilderM.CreateSelect(
                select->getCondition(), old,
                faddForNeg(old, select->getFalseValue())));
            addedSelects.emplace_back(res);
            return res;
          }
        }
        if (Constant *ci = dyn_cast<Constant>(select->getFalseValue())) {
          if (ci->isZeroValue()) {
            SelectInst *res = cast<SelectInst>(BuilderM.CreateSelect(
                select->getCondition(), faddForNeg(old, select->getTrueValue()),
                old));
            addedSelects.emplace_back(res);
            return res;
          }
        }
      }

      //! optimize fadd of bitcast select to select of bitcast fadd
      if (BitCastInst *bc = dyn_cast<BitCastInst>(dif)) {
        if (SelectInst *select = dyn_cast<SelectInst>(bc->getOperand(0))) {
          if (Constant *ci = dyn_cast<Constant>(select->getTrueValue())) {
            if (ci->isZeroValue()) {
              SelectInst *res = cast<SelectInst>(BuilderM.CreateSelect(
                  select->getCondition(), old,
                  faddForNeg(old, BuilderM.CreateCast(bc->getOpcode(),
                                                      select->getFalseValue(),
                                                      bc->getDestTy()))));
              addedSelects.emplace_back(res);
              return res;
            }
          }
          if (Constant *ci = dyn_cast<Constant>(select->getFalseValue())) {
            if (ci->isZeroValue()) {
              SelectInst *res = cast<SelectInst>(BuilderM.CreateSelect(
                  select->getCondition(),
                  faddForNeg(old, BuilderM.CreateCast(bc->getOpcode(),
                                                      select->getTrueValue(),
                                                      bc->getDestTy())),
                  old));
              addedSelects.emplace_back(res);
              return res;
            }
          }
        }
      }

      // fallback
      return faddForNeg(old, dif);
    };

    if (val->getType()->isPointerTy()) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *val << "\n";
    }
    if (isConstantValue(val)) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *val << "\n";
    }
    assert(!val->getType()->isPointerTy());
    assert(!isConstantValue(val));

    Value *ptr = getDifferential(val);

    if (idxs.size() != 0) {
      SmallVector<Value *, 4> sv;
      sv.push_back(ConstantInt::get(Type::getInt32Ty(val->getContext()), 0));
      for (auto i : idxs)
        sv.push_back(i);
#if LLVM_VERSION_MAJOR > 7
      ptr =
          BuilderM.CreateGEP(ptr->getType()->getPointerElementType(), ptr, sv);
#else
      ptr = BuilderM.CreateGEP(ptr, sv);
#endif
      cast<GetElementPtrInst>(ptr)->setIsInBounds(true);
    }
#if LLVM_VERSION_MAJOR > 7
    Value *old =
        BuilderM.CreateLoad(ptr->getType()->getPointerElementType(), ptr);
#else
    Value *old = BuilderM.CreateLoad(ptr);
#endif

    assert(dif->getType() == old->getType());
    Value *res = nullptr;
    if (old->getType()->isIntOrIntVectorTy()) {
      if (!addingType) {
        llvm::errs() << "module: " << *oldFunc->getParent() << "\n";
        llvm::errs() << "oldFunc: " << *oldFunc << "\n";
        llvm::errs() << "newFunc: " << *newFunc << "\n";
        llvm::errs() << "val: " << *val << " old: " << *old << "\n";
      }
      assert(addingType);
      assert(addingType->isFPOrFPVectorTy());

      auto oldBitSize = oldFunc->getParent()->getDataLayout().getTypeSizeInBits(
          old->getType());
      auto newBitSize =
          oldFunc->getParent()->getDataLayout().getTypeSizeInBits(addingType);

      if (oldBitSize > newBitSize && oldBitSize % newBitSize == 0 &&
          !addingType->isVectorTy()) {
#if LLVM_VERSION_MAJOR >= 11
        addingType =
            VectorType::get(addingType, oldBitSize / newBitSize, false);
#else
        addingType = VectorType::get(addingType, oldBitSize / newBitSize);
#endif
      }

      Value *bcold = BuilderM.CreateBitCast(old, addingType);
      Value *bcdif = BuilderM.CreateBitCast(dif, addingType);

      res = faddForSelect(bcold, bcdif);
      if (SelectInst *select = dyn_cast<SelectInst>(res)) {
        assert(addedSelects.back() == select);
        addedSelects.erase(addedSelects.end() - 1);
        res = BuilderM.CreateSelect(
            select->getCondition(),
            BuilderM.CreateBitCast(select->getTrueValue(), old->getType()),
            BuilderM.CreateBitCast(select->getFalseValue(), old->getType()));
        assert(select->getNumUses() == 0);
      } else {
        res = BuilderM.CreateBitCast(res, old->getType());
      }
      if (!mask) {
        BuilderM.CreateStore(res, ptr);
        // store->setAlignment(align);
      } else {
        Type *tys[] = {res->getType(), ptr->getType()};
        auto F = Intrinsic::getDeclaration(oldFunc->getParent(),
                                           Intrinsic::masked_store, tys);
        auto align = cast<AllocaInst>(ptr)->getAlignment();
        assert(align);
        Value *alignv =
            ConstantInt::get(Type::getInt32Ty(mask->getContext()), align);
        Value *args[] = {res, ptr, alignv, mask};
        BuilderM.CreateCall(F, args);
      }
      return addedSelects;
    } else if (old->getType()->isFPOrFPVectorTy()) {
      // TODO consider adding type
      res = faddForSelect(old, dif);

      if (!mask) {
        BuilderM.CreateStore(res, ptr);
        // store->setAlignment(align);
      } else {
        Type *tys[] = {res->getType(), ptr->getType()};
        auto F = Intrinsic::getDeclaration(oldFunc->getParent(),
                                           Intrinsic::masked_store, tys);
        auto align = cast<AllocaInst>(ptr)->getAlignment();
        assert(align);
        Value *alignv =
            ConstantInt::get(Type::getInt32Ty(mask->getContext()), align);
        Value *args[] = {res, ptr, alignv, mask};
        BuilderM.CreateCall(F, args);
      }
      return addedSelects;
    } else if (auto st = dyn_cast<StructType>(old->getType())) {
      assert(!mask);
      if (mask)
        llvm_unreachable("cannot handle recursive addToDiffe with mask");
      for (unsigned i = 0; i < st->getNumElements(); ++i) {
        // TODO pass in full type tree here and recurse into tree.
        if (st->getElementType(i)->isPointerTy())
          continue;
        Value *v = ConstantInt::get(Type::getInt32Ty(st->getContext()), i);
        SmallVector<Value *, 2> idx2(idxs.begin(), idxs.end());
        idx2.push_back(v);
        auto selects = addToDiffe(
            val, BuilderM.CreateExtractValue(dif, ArrayRef<unsigned>(i)),
            BuilderM, nullptr, idx2);
        for (auto select : selects) {
          addedSelects.push_back(select);
        }
      }
      return addedSelects;
    } else if (auto at = dyn_cast<ArrayType>(old->getType())) {
      assert(!mask);
      if (mask)
        llvm_unreachable("cannot handle recursive addToDiffe with mask");
      if (at->getElementType()->isPointerTy())
        return addedSelects;
      for (unsigned i = 0; i < at->getNumElements(); ++i) {
        // TODO pass in full type tree here and recurse into tree.
        Value *v = ConstantInt::get(Type::getInt32Ty(at->getContext()), i);
        SmallVector<Value *, 2> idx2(idxs.begin(), idxs.end());
        idx2.push_back(v);
        auto selects = addToDiffe(
            val, BuilderM.CreateExtractValue(dif, ArrayRef<unsigned>(i)),
            BuilderM, nullptr, idx2);
        for (auto select : selects) {
          addedSelects.push_back(select);
        }
      }
      return addedSelects;
    } else {
      llvm_unreachable("unknown type to add to diffe");
      exit(1);
    }
  }

  void setDiffe(Value *val, Value *toset, IRBuilder<> &BuilderM) {
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);
    if (isConstantValue(val)) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *val << "\n";
    }
    assert(!isConstantValue(val));
    Value *tostore = getDifferential(val);
    if (toset->getType() !=
        cast<PointerType>(tostore->getType())->getElementType()) {
      llvm::errs() << "toset:" << *toset << "\n";
      llvm::errs() << "tostore:" << *tostore << "\n";
    }
    assert(toset->getType() ==
           cast<PointerType>(tostore->getType())->getElementType());
    BuilderM.CreateStore(toset, tostore);
  }

  void freeCache(llvm::BasicBlock *forwardPreheader,
                 const SubLimitType &sublimits, int i, llvm::AllocaInst *alloc,
                 llvm::ConstantInt *byteSizeOfType, llvm::Value *storeInto,
                 llvm::MDNode *InvariantMD) override {
    if (!FreeMemory)
      return;
    assert(reverseBlocks.find(forwardPreheader) != reverseBlocks.end());
    assert(reverseBlocks[forwardPreheader].size());
    IRBuilder<> tbuild(reverseBlocks[forwardPreheader].back());
    tbuild.setFastMathFlags(getFast());

    // ensure we are before the terminator if it exists
    if (tbuild.GetInsertBlock()->size() &&
        tbuild.GetInsertBlock()->getTerminator()) {
      tbuild.SetInsertPoint(tbuild.GetInsertBlock()->getTerminator());
    }

    ValueToValueMapTy antimap;
    for (int j = sublimits.size() - 1; j >= i; j--) {
      auto &innercontainedloops = sublimits[j].second;
      for (auto riter = innercontainedloops.rbegin(),
                rend = innercontainedloops.rend();
           riter != rend; ++riter) {
        const auto &idx = riter->first;
        if (idx.var) {
#if LLVM_VERSION_MAJOR > 7
          antimap[idx.var] = tbuild.CreateLoad(
              cast<PointerType>(idx.antivaralloc->getType())->getElementType(),
              idx.antivaralloc);
#else
          antimap[idx.var] = tbuild.CreateLoad(idx.antivaralloc);
#endif
        }
      }
    }

    Value *metaforfree =
        unwrapM(storeInto, tbuild, antimap, UnwrapMode::LegalFullUnwrap);
#if LLVM_VERSION_MAJOR > 7
    LoadInst *forfree = cast<LoadInst>(tbuild.CreateLoad(
        cast<PointerType>(metaforfree->getType())->getElementType(),
        metaforfree));
#else
    LoadInst *forfree = cast<LoadInst>(tbuild.CreateLoad(metaforfree));
#endif
    forfree->setMetadata(LLVMContext::MD_invariant_group, InvariantMD);
    forfree->setMetadata(
        LLVMContext::MD_dereferenceable,
        MDNode::get(
            forfree->getContext(),
            ArrayRef<Metadata *>(ConstantAsMetadata::get(byteSizeOfType))));
    forfree->setName("forfree");
    unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
    if ((bsize & (bsize - 1)) == 0) {
#if LLVM_VERSION_MAJOR >= 10
      forfree->setAlignment(Align(bsize));
#else
      forfree->setAlignment(bsize);
#endif
    }
    CallInst *ci = cast<CallInst>(CallInst::CreateFree(
        tbuild.CreatePointerCast(forfree,
                                 Type::getInt8PtrTy(newFunc->getContext())),
        tbuild.GetInsertBlock()));
    if (newFunc->getSubprogram())
      ci->setDebugLoc(DILocation::get(newFunc->getContext(), 0, 0,
                                      newFunc->getSubprogram(), 0));
#if LLVM_VERSION_MAJOR >= 14
    ci->addAttributeAtIndex(AttributeList::FirstArgIndex, Attribute::NonNull);
#else
    ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
#endif
    if (ci->getParent() == nullptr) {
      tbuild.Insert(ci);
    }
    scopeFrees[alloc].insert(ci);
  }

//! align is the alignment that should be specified for load/store to pointer
#if LLVM_VERSION_MAJOR >= 10
  void addToInvertedPtrDiffe(Value *origptr, Value *dif, IRBuilder<> &BuilderM,
                             MaybeAlign align, Value *OrigOffset = nullptr,
                             Value *mask = nullptr)
#else
  void addToInvertedPtrDiffe(Value *origptr, Value *dif, IRBuilder<> &BuilderM,
                             unsigned align, Value *OrigOffset = nullptr,
                             Value *mask = nullptr)
#endif
  {
    if (!(origptr->getType()->isPointerTy()) ||
        !(cast<PointerType>(origptr->getType())->getElementType() ==
          dif->getType())) {
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << "Origptr: " << *origptr << "\n";
      llvm::errs() << "Diff: " << *dif << "\n";
    }
    assert(origptr->getType()->isPointerTy());
    assert(cast<PointerType>(origptr->getType())->getElementType() ==
           dif->getType());

    assert(origptr->getType()->isPointerTy());
    assert(cast<PointerType>(origptr->getType())->getElementType() ==
           dif->getType());

    // const SCEV *S = SE.getSCEV(PN);
    // if (SE.getCouldNotCompute() == S)
    //  continue;

    Value *ptr;

    switch (mode) {
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode:
      ptr = invertPointerM(origptr, BuilderM);
      break;
    case DerivativeMode::ReverseModePrimal:
      assert(false && "Invalid derivative mode (ReverseModePrimal)");
      break;
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined:
      ptr = lookupM(invertPointerM(origptr, BuilderM), BuilderM);
      break;
    }

    assert(ptr);
    if (OrigOffset) {
#if LLVM_VERSION_MAJOR > 7
      ptr = BuilderM.CreateGEP(
          cast<PointerType>(ptr->getType())->getElementType(), ptr,
          lookupM(getNewFromOriginal(OrigOffset), BuilderM));
#else
      ptr = BuilderM.CreateGEP(
          ptr, lookupM(getNewFromOriginal(OrigOffset), BuilderM));
#endif
    }

    auto TmpOrig =
#if LLVM_VERSION_MAJOR >= 12
        getUnderlyingObject(origptr, 100);
#else
        GetUnderlyingObject(origptr, oldFunc->getParent()->getDataLayout(),
                            100);
#endif

    // atomics
    bool Atomic = AtomicAdd;
    auto Arch = llvm::Triple(newFunc->getParent()->getTargetTriple()).getArch();

    // No need to do atomic on local memory for CUDA since it can't be raced
    // upon
    if (isa<AllocaInst>(TmpOrig) &&
        (Arch == Triple::nvptx || Arch == Triple::nvptx64 ||
         Arch == Triple::amdgcn)) {
      Atomic = false;
    }

    if (Atomic) {
      // For amdgcn constant AS is 4 and if the primal is in it we need to cast
      // the derivative value to AS 1
      auto AS = cast<PointerType>(ptr->getType())->getAddressSpace();
      if (Arch == Triple::amdgcn && AS == 4) {
        ptr = BuilderM.CreateAddrSpaceCast(
            ptr, PointerType::get(
                     cast<PointerType>(ptr->getType())->getElementType(), 1));
      }

      assert(!mask);
      if (mask) {
        llvm::errs() << "unhandled masked atomic fadd on llvm version " << *ptr
                     << " " << *dif << " mask: " << *mask << "\n";
        llvm_unreachable("unhandled masked atomic fadd");
      }

      /*
      while (auto ASC = dyn_cast<AddrSpaceCastInst>(ptr)) {
        ptr = ASC->getOperand(0);
      }
      while (auto ASC = dyn_cast<ConstantExpr>(ptr)) {
        if (!ASC->isCast()) break;
        if (ASC->getOpcode() != Instruction::AddrSpaceCast) break;
        ptr = ASC->getOperand(0);
      }
      */
      if (dif->getType()->isIntOrIntVectorTy()) {

        ptr = BuilderM.CreateBitCast(
            ptr, PointerType::get(
                     IntToFloatTy(dif->getType()),
                     cast<PointerType>(ptr->getType())->getAddressSpace()));
        dif = BuilderM.CreateBitCast(dif, IntToFloatTy(dif->getType()));
      }
#if LLVM_VERSION_MAJOR >= 9
      AtomicRMWInst::BinOp op = AtomicRMWInst::FAdd;
      if (auto vt = dyn_cast<VectorType>(dif->getType())) {
#if LLVM_VERSION_MAJOR >= 12
        assert(!vt->getElementCount().isScalable());
        size_t numElems = vt->getElementCount().getKnownMinValue();
#else
        size_t numElems = vt->getNumElements();
#endif
        for (size_t i = 0; i < numElems; ++i) {
          auto vdif = BuilderM.CreateExtractElement(dif, i);
          Value *Idxs[] = {
              ConstantInt::get(Type::getInt64Ty(vt->getContext()), 0),
              ConstantInt::get(Type::getInt32Ty(vt->getContext()), i)};
#if LLVM_VERSION_MAJOR > 7
          auto vptr = BuilderM.CreateGEP(
              cast<PointerType>(ptr->getType())->getElementType(), ptr, Idxs);
#else
          auto vptr = BuilderM.CreateGEP(ptr, Idxs);
#endif
#if LLVM_VERSION_MAJOR >= 13
          BuilderM.CreateAtomicRMW(op, vptr, vdif, align,
                                   AtomicOrdering::Monotonic,
                                   SyncScope::System);
#elif LLVM_VERSION_MAJOR >= 11
          AtomicRMWInst *rmw = BuilderM.CreateAtomicRMW(
              op, vptr, vdif, AtomicOrdering::Monotonic, SyncScope::System);
          if (align)
            rmw->setAlignment(align.getValue());
#else
          BuilderM.CreateAtomicRMW(op, vptr, vdif, AtomicOrdering::Monotonic,
                                   SyncScope::System);
#endif
        }
      } else {
#if LLVM_VERSION_MAJOR >= 13
        BuilderM.CreateAtomicRMW(op, ptr, dif, align, AtomicOrdering::Monotonic,
                                 SyncScope::System);
#elif LLVM_VERSION_MAJOR >= 11
        AtomicRMWInst *rmw = BuilderM.CreateAtomicRMW(
            op, ptr, dif, AtomicOrdering::Monotonic, SyncScope::System);
        if (align)
          rmw->setAlignment(align.getValue());
#else
        BuilderM.CreateAtomicRMW(op, ptr, dif, AtomicOrdering::Monotonic,
                                 SyncScope::System);
#endif
      }
#else
      llvm::errs() << "unhandled atomic fadd on llvm version " << *ptr << " "
                   << *dif << "\n";
      llvm_unreachable("unhandled atomic fadd");
#endif
      return;
    }

    Value *res;
    Value *old;

    if (!mask) {
#if LLVM_VERSION_MAJOR > 7
      auto LI = BuilderM.CreateLoad(dif->getType(), ptr);
#else
      auto LI = BuilderM.CreateLoad(ptr);
#endif
      if (align)
#if LLVM_VERSION_MAJOR >= 10
        LI->setAlignment(*align);
#else
        LI->setAlignment(align);
#endif
      old = LI;
    } else {
      Type *tys[] = {dif->getType(), origptr->getType()};
      auto F = Intrinsic::getDeclaration(oldFunc->getParent(),
                                         Intrinsic::masked_load, tys);
#if LLVM_VERSION_MAJOR >= 10
      Value *alignv = ConstantInt::get(Type::getInt32Ty(mask->getContext()),
                                       align ? align->value() : 0);
#else
      Value *alignv =
          ConstantInt::get(Type::getInt32Ty(mask->getContext()), align);
#endif
      Value *args[] = {lookupM(invertPointerM(origptr, BuilderM), BuilderM),
                       alignv, mask, Constant::getNullValue(dif->getType())};
      old = BuilderM.CreateCall(F, args);
    }

    if (old->getType()->isIntOrIntVectorTy()) {
      res = BuilderM.CreateFAdd(
          BuilderM.CreateBitCast(old, IntToFloatTy(old->getType())),
          BuilderM.CreateBitCast(dif, IntToFloatTy(dif->getType())));
      res = BuilderM.CreateBitCast(res, old->getType());
    } else if (old->getType()->isFPOrFPVectorTy()) {
      res = BuilderM.CreateFAdd(old, dif);
    } else {
      assert(old);
      assert(dif);
      llvm::errs() << *newFunc << "\n"
                   << "cannot handle type " << *old << "\n"
                   << *dif;
      assert(0 && "cannot handle type");
      report_fatal_error("cannot handle type");
    }

    if (!mask) {
      StoreInst *st = BuilderM.CreateStore(res, ptr);
      if (align)
#if LLVM_VERSION_MAJOR >= 10
        st->setAlignment(*align);
#else
        st->setAlignment(align);
#endif
    } else {
      Type *tys[] = {dif->getType(), origptr->getType()};
      auto F = Intrinsic::getDeclaration(oldFunc->getParent(),
                                         Intrinsic::masked_store, tys);
      assert(align);
#if LLVM_VERSION_MAJOR >= 10
      Value *alignv = ConstantInt::get(Type::getInt32Ty(mask->getContext()),
                                       align->value());
#else
      Value *alignv =
          ConstantInt::get(Type::getInt32Ty(mask->getContext()), align);
#endif
      Value *args[] = {res, ptr, alignv, mask};
      BuilderM.CreateCall(F, args);
    }
  }
};

void SubTransferHelper(GradientUtils *gutils, DerivativeMode Mode,
                       Type *secretty, Intrinsic::ID intrinsic,
                       unsigned dstalign, unsigned srcalign, unsigned offset,
                       bool dstConstant, Value *shadow_dst, bool srcConstant,
                       Value *shadow_src, Value *length, Value *isVolatile,
                       llvm::CallInst *MTI, bool allowForward = true);
#endif
