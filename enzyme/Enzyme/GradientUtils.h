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
#include "llvm/Support/Casting.h"

#include "llvm/Transforms/Utils/ValueMapper.h"

#include "llvm/Support/ErrorHandling.h"

#include "ActivityAnalysis.h"
#include "CacheUtility.h"
#include "EnzymeLogic.h"
#include "LibraryFuncs.h"

using namespace llvm;

enum class DerivativeMode { Forward, Reverse, Both };

static inline std::string to_string(DerivativeMode mode) {
  switch (mode) {
  case DerivativeMode::Forward:
    return "Forward";
  case DerivativeMode::Reverse:
    return "Reverse";
  case DerivativeMode::Both:
    return "Both";
  }
  llvm_unreachable("illegal derivative mode");
}

enum class AugmentedStruct;
class GradientUtils : public CacheUtility {
public:
  bool AtomicAdd;
  DerivativeMode mode;
  llvm::Function *oldFunc;
  ValueToValueMapTy invertedPointers;
  DominatorTree OrigDT;
  PostDominatorTree OrigPDT;
  std::shared_ptr<ActivityAnalyzer> ATA;
  LoopInfo OrigLI;
  SmallVector<BasicBlock *, 12> originalBlocks;
  ValueMap<BasicBlock *, BasicBlock *> reverseBlocks;
  SmallVector<PHINode *, 4> fictiousPHIs;
  ValueToValueMapTy originalToNewFn;

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

  llvm::DebugLoc getNewFromOriginal(const llvm::DebugLoc L) const {
    if (L.get() == nullptr) return nullptr;
    if (!oldFunc->getSubprogram()) return L;
    assert(originalToNewFn.hasMD());
    auto opt = originalToNewFn.getMappedMD(L.getAsMDNode());
    if (!opt.hasValue())
      return L;
    assert(opt.hasValue());
    return llvm::DebugLoc(cast<MDNode>(*opt.getPointer()));
  }

  Value *getNewFromOriginal(const Value *originst) const {
    assert(originst);
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
    return cast<Instruction>(getNewFromOriginal((Value *)newinst));
  }

  Value *hasUninverted(const Value *inverted) const {
    for (auto v : invertedPointers) {
      if (v.second == inverted)
        return const_cast<Value *>(v.first);
    }
    return nullptr;
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
    for (auto v : originalToNewFn) {
      if (v.second == newinst)
        return const_cast<Value *>(v.first);
    }
    return nullptr;
  }

  Instruction *isOriginal(const Instruction *newinst) const {
    return cast_or_null<Instruction>(isOriginal((const Value *)newinst));
  }

private:
  SmallVector<Value *, 4> addedTapeVals;
  unsigned tapeidx;
  Value *tape;

  std::map<std::pair<Value *, BasicBlock *>, Value *> unwrap_cache;
  std::map<std::pair<Value *, BasicBlock *>, Value *> lookup_cache;

public:
  bool legalRecompute(const Value *val,
                      const ValueToValueMapTy &available) const;
  bool shouldRecompute(const Value *val,
                       const ValueToValueMapTy &available) const;

  void replaceAWithB(Value *A, Value *B, bool storeInCache = false) {
    for (unsigned i = 0; i < addedTapeVals.size(); ++i) {
      if (addedTapeVals[i] == A) {
        addedTapeVals[i] = B;
      }
    }

    auto found = scopeMap.find(A);
    if (found != scopeMap.end()) {
      insert_or_assign2(scopeMap, B, found->second);

      llvm::AllocaInst *cache = found->second.first;
      if (storeInCache) {
        assert(isa<Instruction>(B));
        if (scopeInstructions.find(cache) != scopeInstructions.end()) {
          for (auto st : scopeInstructions[cache])
            cast<StoreInst>(st)->eraseFromParent();
          scopeInstructions.erase(cache);
          storeInstructionInCache(found->second.second, cast<Instruction>(B),
                                  cache);
        }
      }

      scopeMap.erase(A);
    }

    if (invertedPointers.find(A) != invertedPointers.end()) {
      invertedPointers[B] = invertedPointers[A];
      invertedPointers.erase(A);
    }
    if (auto orig = isOriginal(A)) {
      originalToNewFn[orig] = B;
    }

    A->replaceAllUsesWith(B);
  }

  void erase(Instruction *I) override {
    assert(I);
    invertedPointers.erase(I);
    originalToNewFn.erase(I);
  eraser:
    for (auto v : originalToNewFn) {
      if (v.second == I) {
        originalToNewFn.erase(v.first);
        goto eraser;
      }
    }
    for (auto v : invertedPointers) {
      if (v.second == I) {
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        dumpPointers();
        llvm::errs() << *v.first << "\n";
        llvm::errs() << *I << "\n";
        assert(0 && "erasing something in invertedPointers map");
      }
    }

    {
      std::vector<std::pair<Value *, BasicBlock *>> unwrap_cache_pairs;
      for (auto &a : unwrap_cache) {
        if (a.second == I) {
          unwrap_cache_pairs.push_back(a.first);
        }
        if (a.first.first == I) {
          unwrap_cache_pairs.push_back(a.first);
        }
      }
      for (auto a : unwrap_cache_pairs) {
        unwrap_cache.erase(a);
      }
    }

    {
      std::vector<std::pair<Value *, BasicBlock *>> lookup_cache_pairs;
      for (auto &a : lookup_cache) {
        if (a.second == I) {
          lookup_cache_pairs.push_back(a.first);
        }
        if (a.first.first == I) {
          lookup_cache_pairs.push_back(a.first);
        }
      }
      for (auto a : lookup_cache_pairs) {
        lookup_cache.erase(a);
      }
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
    PHINode *placeholder = cast<PHINode>(invertedPointers[orig]);

    assert(placeholder->getParent()->getParent() == newFunc);
    placeholder->setName("");
    IRBuilder<> bb(placeholder);

    SmallVector<Value *, 8> args;
    for (unsigned i = 0; i < orig->getNumArgOperands(); ++i) {
      args.push_back(getNewFromOriginal(orig->getArgOperand(i)));
    }
    Value *anti =
        bb.CreateCall(orig->getCalledFunction(), args, orig->getName() + "'mi");
    cast<CallInst>(anti)->setAttributes(orig->getAttributes());
    cast<CallInst>(anti)->setCallingConv(orig->getCallingConv());
    cast<CallInst>(anti)->setTailCallKind(orig->getTailCallKind());
    cast<CallInst>(anti)->setDebugLoc(getNewFromOriginal(orig->getDebugLoc()));

    cast<CallInst>(anti)->addAttribute(AttributeList::ReturnIndex,
                                       Attribute::NoAlias);
    cast<CallInst>(anti)->addAttribute(AttributeList::ReturnIndex,
                                       Attribute::NonNull);

    unsigned derefBytes = 0;
    if (orig->getCalledFunction()->getName() == "malloc" ||
        orig->getCalledFunction()->getName() == "_Znwm") {
      if (auto ci = dyn_cast<ConstantInt>(args[0])) {
        derefBytes = ci->getLimitedValue();
        cast<CallInst>(anti)->addDereferenceableAttr(
            llvm::AttributeList::ReturnIndex, ci->getLimitedValue());
        cast<CallInst>(anti)->addDereferenceableOrNullAttr(
            llvm::AttributeList::ReturnIndex, ci->getLimitedValue());
        CallInst *cal = cast<CallInst>(getNewFromOriginal(orig));
        cal->addDereferenceableAttr(llvm::AttributeList::ReturnIndex,
                                    ci->getLimitedValue());
        cal->addDereferenceableOrNullAttr(llvm::AttributeList::ReturnIndex,
                                          ci->getLimitedValue());
        cal->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
        cal->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
      }
    }

    invertedPointers[orig] = anti;
    // assert(placeholder != anti);
    bb.SetInsertPoint(placeholder->getNextNode());
    replaceAWithB(placeholder, anti);
    erase(placeholder);

    anti = cacheForReverse(bb, anti, idx);
    invertedPointers[orig] = anti;

    if (tape == nullptr) {
      auto dst_arg =
          bb.CreateBitCast(anti, Type::getInt8PtrTy(orig->getContext()));
      auto val_arg = ConstantInt::get(Type::getInt8Ty(orig->getContext()), 0);
      auto len_arg =
          bb.CreateZExtOrTrunc(args[0], Type::getInt64Ty(orig->getContext()));
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
        memset->addDereferenceableAttr(llvm::AttributeList::FirstArgIndex,
                                       derefBytes);
        memset->addDereferenceableOrNullAttr(llvm::AttributeList::FirstArgIndex,
                                             derefBytes);
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

  Value *cacheForReverse(IRBuilder<> &BuilderQ, Value *malloc, int idx);

  const SmallVectorImpl<Value *> &getTapeValues() const {
    return addedTapeVals;
  }

public:
  AAResults &AA;
  TypeAnalysis &TA;
  GradientUtils(Function *newFunc_, Function *oldFunc_, TargetLibraryInfo &TLI_,
                TypeAnalysis &TA_, AAResults &AA_,
                ValueToValueMapTy &invertedPointers_,
                const SmallPtrSetImpl<Value *> &constantvalues_,
                const SmallPtrSetImpl<Value *> &activevals_, bool ActiveReturn,
                ValueToValueMapTy &originalToNewFn_, DerivativeMode mode)
      : CacheUtility(TLI_, newFunc_), mode(mode), oldFunc(oldFunc_),
        invertedPointers(), OrigDT(*oldFunc_),
#if LLVM_VERSION_MAJOR >= 7
        OrigPDT(*oldFunc_),
#else
        OrigPDT(),
#endif
        ATA(new ActivityAnalyzer(AA_, TLI_, constantvalues_, activevals_,
                                 ActiveReturn)),
        OrigLI(OrigDT), AA(AA_), TA(TA_) {
    if (oldFunc_->getSubprogram()) {
      assert(originalToNewFn_.hasMD());
    }

    originalToNewFn.getMDMap() = originalToNewFn_.getMDMap();

    if (oldFunc_->getSubprogram()) {
      assert(originalToNewFn.hasMD());
    }
#if LLVM_VERSION_MAJOR <= 6
    OrigPDT.recalculate(*oldFunc_);
#endif
    invertedPointers.insert(invertedPointers_.begin(), invertedPointers_.end());
    originalToNewFn.insert(originalToNewFn_.begin(), originalToNewFn_.end());
    for (BasicBlock &BB : *newFunc) {
      originalBlocks.emplace_back(&BB);
    }
    tape = nullptr;
    tapeidx = 0;
    assert(originalBlocks.size() > 0);
  }

public:
  static GradientUtils *
  CreateFromClone(Function *todiff, TargetLibraryInfo &TLI, TypeAnalysis &TA,
                  AAResults &AA, DIFFE_TYPE retType,
                  const std::vector<DIFFE_TYPE> &constant_args, bool returnUsed,
                  std::map<AugmentedStruct, int> &returnMapping);

  StoreInst *setPtrDiffe(Value *ptr, Value *newval, IRBuilder<> &BuilderM) {
    if (auto inst = dyn_cast<Instruction>(ptr)) {
      assert(inst->getParent()->getParent() == oldFunc);
    }
    if (auto arg = dyn_cast<Argument>(ptr)) {
      assert(arg->getParent() == oldFunc);
    }
    ptr = invertPointerM(ptr, BuilderM);
    return BuilderM.CreateStore(newval, ptr);
  }

private:
  BasicBlock *originalForReverseBlock(BasicBlock &BB2) const {
    assert(reverseBlocks.size() != 0);
    for (auto BB : originalBlocks) {
      auto it = reverseBlocks.find(BB);
      assert(it != reverseBlocks.end());
      if (it->second == &BB2) {
        return BB;
      }
    }
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << BB2 << "\n";
    assert(0 && "could not find original block for given reverse block");
    report_fatal_error("could not find original block for given reverse block");
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

  bool isOriginalBlock(const BasicBlock &BB) const {
    for (auto A : originalBlocks) {
      if (A == &BB)
        return true;
    }
    return false;
  }

  void eraseFictiousPHIs() {
    for (auto pp : fictiousPHIs) {
      if (pp->getNumUses() != 0) {
        llvm::errs() << "mod:" << *oldFunc->getParent() << "\n";
        llvm::errs() << "oldFunc:" << *oldFunc << "\n";
        llvm::errs() << "newFunc:" << *newFunc << "\n";
        llvm::errs() << " pp: " << *pp << "\n";
      }
      assert(pp->getNumUses() == 0);
      pp->replaceAllUsesWith(UndefValue::get(pp->getType()));
      erase(pp);
    }
    fictiousPHIs.clear();
  }

  std::map<llvm::Value *, bool> internal_isConstantValue;
  std::map<const llvm::Instruction *, bool> internal_isConstantInstruction;
  TypeResults *my_TR;
  void forceActiveDetection(AAResults &AA, TypeResults &TR) {
    my_TR = &TR;
    for (auto &Arg : oldFunc->args()) {
      internal_isConstantValue[&Arg] = ATA->isConstantValue(TR, &Arg);
    }

    for (BasicBlock &BB : *oldFunc) {
      for (Instruction &I : BB) {
        bool const_inst = ATA->isConstantInstruction(TR, &I);
        bool const_value = ATA->isConstantValue(TR, &I);

        internal_isConstantValue[&I] = const_value;
        internal_isConstantInstruction[&I] = const_inst;

        // if (printconst)
        // llvm::errs() << I << " cv=" << const_value << " ci=" << const_inst <<
        // "\n";
      }
    }
  }

  bool isConstantValue(Value *val) const {
    if (auto inst = dyn_cast<Instruction>(val)) {
      assert(inst->getParent()->getParent() == oldFunc);
      assert(internal_isConstantValue.find(inst) !=
             internal_isConstantValue.end());
      return internal_isConstantValue.find(inst)->second;
    }

    if (auto arg = dyn_cast<Argument>(val)) {
      assert(arg->getParent() == oldFunc);
      assert(internal_isConstantValue.find(arg) !=
             internal_isConstantValue.end());
      return internal_isConstantValue.find(arg)->second;
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
      if (nonmarkedglobals_inactive)
        return true;
      goto err;
    }
    if (isa<GlobalValue>(val)) {
      if (nonmarkedglobals_inactive)
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
    if (internal_isConstantInstruction.find(inst) ==
        internal_isConstantInstruction.end()) {
      llvm::errs() << *oldFunc << "\n";
      for (auto &pair : internal_isConstantInstruction) {
        llvm::errs() << " constantinst[" << *pair.first << "] = " << pair.second
                     << "\n";
      }
      llvm::errs() << "inst: " << *inst << "\n";
    }
    assert(internal_isConstantInstruction.find(inst) !=
           internal_isConstantInstruction.end());
    return internal_isConstantInstruction.find(inst)->second;
  }

  void forceAugmentedReturns(
      TypeResults &TR,
      const SmallPtrSetImpl<BasicBlock *> &guaranteedUnreachable) {
    assert(TR.info.Function == oldFunc);

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

        Instruction *newi = getNewFromOriginal(inst);

        if (isa<LoadInst>(inst)) {
          IRBuilder<> BuilderZ(getNextNonDebugInstruction(newi));
          BuilderZ.setFastMathFlags(getFast());
          PHINode *anti = BuilderZ.CreatePHI(inst->getType(), 1,
                                             inst->getName() + "'il_phi");
          invertedPointers[inst] = anti;
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

        IRBuilder<> BuilderZ(getNextNonDebugInstruction(newi));
        BuilderZ.setFastMathFlags(getFast());
        PHINode *anti =
            BuilderZ.CreatePHI(op->getType(), 1, op->getName() + "'ip_phi");
        invertedPointers[inst] = anti;

        if (called && isAllocationFunction(*called, TLI)) {
          invertedPointers[inst]->setName(op->getName() + "'mi");
        }
      }
    }
  }

  /// if full unwrap, don't just unwrap this instruction, but also its operands,
  /// etc
  Value *unwrapM(Value *const val, IRBuilder<> &BuilderM,
                 const ValueToValueMapTy &available,
                 UnwrapMode mode) override final;

  void ensureLookupCached(Instruction *inst, bool shouldFree = true) {
    assert(inst);
    if (scopeMap.find(inst) != scopeMap.end())
      return;
    if (shouldFree)
      assert(reverseBlocks.size());
    AllocaInst *cache = createCacheForScope(inst->getParent(), inst->getType(),
                                            inst->getName(), shouldFree);
    assert(cache);
    Value *Val = inst;
    insert_or_assign(scopeMap, Val,
                     std::pair<AllocaInst *, LimitContext>(
                         cache, LimitContext(inst->getParent())));
    storeInstructionInCache(inst->getParent(), inst, cache);
  }

  std::map<Instruction *, std::map<BasicBlock *, Instruction *>> lcssaFixes;
  Instruction *fixLCSSA(Instruction *inst, const IRBuilder<> &BuilderM) {
    assert(inst->getName() != "<badref>");
    LoopContext lc;
    bool inLoop = getContext(inst->getParent(), lc);
    if (inLoop) {
      bool isChildLoop = false;

      BasicBlock *forwardBlock = BuilderM.GetInsertBlock();

      if (!isOriginalBlock(*forwardBlock)) {
        forwardBlock = originalForReverseBlock(*forwardBlock);
      }

      auto builderLoop = LI.getLoopFor(forwardBlock);
      while (builderLoop) {
        if (builderLoop->getHeader() == lc.header) {
          isChildLoop = true;
          break;
        }
        builderLoop = builderLoop->getParentLoop();
      }

      if (!isChildLoop) {
        // llvm::errs() << "manually performing lcssa for instruction" << *inst
        // << " in block " << BuilderM.GetInsertBlock()->getName() << "\n";
        if (!DT.dominates(inst, forwardBlock)) {
          llvm::errs() << *this->newFunc << "\n";
          llvm::errs() << *forwardBlock << "\n";
          llvm::errs() << *BuilderM.GetInsertBlock() << "\n";
          llvm::errs() << *inst << "\n";
        }
        assert(DT.dominates(inst, forwardBlock));

        for (auto pair : lcssaFixes[inst]) {
          if (DT.dominates(pair.first, forwardBlock)) {
            return pair.second;
          }
        }

        // TODO replace toplace with the first block dominated by inst, that
        // dominates (or is) forwardBlock
        //  for ensuring maximum reuse
        BasicBlock *toplace = forwardBlock;

        IRBuilder<> lcssa(&toplace->front());
        auto lcssaPHI = lcssa.CreatePHI(inst->getType(), 1,
                                        inst->getName() + "!manual_lcssa");
        for (auto pred : predecessors(toplace))
          lcssaPHI->addIncoming(inst, pred);

        lcssaFixes[inst][toplace] = lcssaPHI;
        return lcssaPHI;
      }
    }
    return inst;
  }

  Value *
  lookupM(Value *val, IRBuilder<> &BuilderM,
          const ValueToValueMapTy &incoming_availalble = ValueToValueMapTy(),
          bool tryLegalRecomputeCheck = true) override;

  Value *invertPointerM(Value *val, IRBuilder<> &BuilderM);

  void branchToCorrespondingTarget(
      BasicBlock *ctx, IRBuilder<> &BuilderM,
      const std::map<BasicBlock *,
                     std::vector<std::pair</*pred*/ BasicBlock *,
                                           /*successor*/ BasicBlock *>>>
          &targetToPreds,
      const std::map<BasicBlock *, PHINode *> *replacePHIs = nullptr);
};

class DiffeGradientUtils : public GradientUtils {
  DiffeGradientUtils(Function *newFunc_, Function *oldFunc_,
                     TargetLibraryInfo &TLI, TypeAnalysis &TA, AAResults &AA,
                     ValueToValueMapTy &invertedPointers_,
                     const SmallPtrSetImpl<Value *> &constantvalues_,
                     const SmallPtrSetImpl<Value *> &returnvals_,
                     bool ActiveReturn, ValueToValueMapTy &origToNew_,
                     DerivativeMode mode)
      : GradientUtils(newFunc_, oldFunc_, TLI, TA, AA, invertedPointers_,
                      constantvalues_, returnvals_, ActiveReturn, origToNew_,
                      mode) {
    assert(reverseBlocks.size() == 0);
    for (BasicBlock *BB : originalBlocks) {
      if (BB == inversionAllocs)
        continue;
      reverseBlocks[BB] = BasicBlock::Create(BB->getContext(),
                                             "invert" + BB->getName(), newFunc);
    }
    assert(reverseBlocks.size() != 0);
  }

public:
  ValueToValueMapTy differentials;
  static DiffeGradientUtils *
  CreateFromClone(bool topLevel, Function *todiff, TargetLibraryInfo &TLI,
                  TypeAnalysis &TA, AAResults &AA, DIFFE_TYPE retType,
                  const std::vector<DIFFE_TYPE> &constant_args,
                  ReturnType returnValue, Type *additionalArg);

private:
  Value *getDifferential(Value *val) {
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
    }
    if (val->getType()->isPointerTy()) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *val << "\n";
    }
    assert(!val->getType()->isPointerTy());
    assert(!val->getType()->isVoidTy());
    return BuilderM.CreateLoad(getDifferential(val));
  }

  // Returns created select instructions, if any
  std::vector<SelectInst *>
  addToDiffe(Value *val, Value *dif, IRBuilder<> &BuilderM, Type *addingType) {
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
    if (val->getType() != dif->getType()) {
      llvm::errs() << "val: " << *val << " dif: " << *dif << "\n";
    }
    assert(val->getType() == dif->getType());
    auto old = diffe(val, BuilderM);
    assert(val->getType() == old->getType());
    Value *res = nullptr;
    if (val->getType()->isIntOrIntVectorTy()) {
      if (!addingType) {
        llvm::errs() << "module: " << *oldFunc->getParent() << "\n";
        llvm::errs() << "oldFunc: " << *oldFunc << "\n";
        llvm::errs() << "newFunc: " << *newFunc << "\n";
        llvm::errs() << "val: " << *val << "\n";
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
            BuilderM.CreateBitCast(select->getTrueValue(), val->getType()),
            BuilderM.CreateBitCast(select->getFalseValue(), val->getType()));
        assert(select->getNumUses() == 0);
      } else {
        res = BuilderM.CreateBitCast(res, val->getType());
      }
      BuilderM.CreateStore(res, getDifferential(val));
      // store->setAlignment(align);
      return addedSelects;
    } else if (val->getType()->isFPOrFPVectorTy()) {
      // TODO consider adding type
      res = faddForSelect(old, dif);

      BuilderM.CreateStore(res, getDifferential(val));
      // store->setAlignment(align);
      return addedSelects;
    } else if (val->getType()->isStructTy()) {
      auto st = cast<StructType>(val->getType());
      for (unsigned i = 0; i < st->getNumElements(); ++i) {
        Value *v = ConstantInt::get(Type::getInt32Ty(st->getContext()), i);
        SelectInst *addedSelect = addToDiffeIndexed(
            val, BuilderM.CreateExtractValue(dif, {i}), {v}, BuilderM);
        if (addedSelect) {
          addedSelects.push_back(addedSelect);
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

  SelectInst *addToDiffeIndexed(Value *val, Value *dif, ArrayRef<Value *> idxs,
                                IRBuilder<> &BuilderM) {
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);
    assert(!isConstantValue(val));
    SmallVector<Value *, 4> sv;
    sv.push_back(ConstantInt::get(Type::getInt32Ty(val->getContext()), 0));
    for (auto i : idxs)
      sv.push_back(i);
    Value *ptr = BuilderM.CreateGEP(getDifferential(val), sv);
    cast<GetElementPtrInst>(ptr)->setIsInBounds(true);
    Value *old = BuilderM.CreateLoad(ptr);

    Value *res = nullptr;

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

    SelectInst *addedSelect = nullptr;

    //! optimize fadd of select to select of fadd
    // TODO: Handle Selects of ints
    if (SelectInst *select = dyn_cast<SelectInst>(dif)) {
      if (ConstantFP *ci = dyn_cast<ConstantFP>(select->getTrueValue())) {
        if (ci->isZero()) {
          cast<Instruction>(res)->eraseFromParent();
          res = BuilderM.CreateSelect(
              select->getCondition(), old,
              BuilderM.CreateFAdd(old, select->getFalseValue()));
          addedSelect = cast<SelectInst>(res);
          goto endselect;
        }
      }
      if (ConstantFP *ci = dyn_cast<ConstantFP>(select->getFalseValue())) {
        if (ci->isZero()) {
          cast<Instruction>(res)->eraseFromParent();
          res = BuilderM.CreateSelect(
              select->getCondition(),
              BuilderM.CreateFAdd(old, select->getTrueValue()), old);
          addedSelect = cast<SelectInst>(res);
          goto endselect;
        }
      }
    }
  endselect:;

    BuilderM.CreateStore(res, ptr);
    return addedSelect;
  }

  virtual void
  freeCache(llvm::BasicBlock *forwardPreheader, const SubLimitType &sublimits,
            int i, llvm::AllocaInst *alloc, llvm::ConstantInt *byteSizeOfType,
            llvm::Value *storeInto, llvm::MDNode *InvariantMD) override {
    assert(reverseBlocks.find(forwardPreheader) != reverseBlocks.end());
    assert(reverseBlocks[forwardPreheader]);
    IRBuilder<> tbuild(reverseBlocks[forwardPreheader]);
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
        if (idx.var)
          antimap[idx.var] = tbuild.CreateLoad(idx.antivaralloc);
      }
    }

    auto forfree = cast<LoadInst>(tbuild.CreateLoad(
        unwrapM(storeInto, tbuild, antimap, UnwrapMode::LegalFullUnwrap)));
    forfree->setMetadata(LLVMContext::MD_invariant_group, InvariantMD);
    forfree->setMetadata(
        LLVMContext::MD_dereferenceable,
        MDNode::get(forfree->getContext(),
                    {ConstantAsMetadata::get(byteSizeOfType)}));
    forfree->setName("forfreegutils.h");
    unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
    if ((bsize & (bsize - 1)) == 0) {
#if LLVM_VERSION_MAJOR >= 10
      forfree->setAlignment(Align(bsize));
#else
      forfree->setAlignment(bsize);
#endif
    }
    auto ci = cast<CallInst>(CallInst::CreateFree(
        tbuild.CreatePointerCast(forfree,
                                 Type::getInt8PtrTy(newFunc->getContext())),
        tbuild.GetInsertBlock()));
    ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
    if (ci->getParent() == nullptr) {
      tbuild.Insert(ci);
    }
    scopeFrees[alloc].insert(ci);
  }

//! align is the alignment that should be specified for load/store to pointer
#if LLVM_VERSION_MAJOR >= 10
  void addToInvertedPtrDiffe(Value *ptr, Value *dif, IRBuilder<> &BuilderM,
                             MaybeAlign align){
#else
  void addToInvertedPtrDiffe(Value *ptr, Value *dif, IRBuilder<> &BuilderM,
                             unsigned align) {
#endif
      if (!(ptr->getType()->isPointerTy()) ||
          !(cast<PointerType>(ptr->getType())->getElementType() ==
            dif->getType())) {
        llvm::errs() << *oldFunc << "\n";
  llvm::errs() << *newFunc << "\n";
  llvm::errs() << "Ptr: " << *ptr << "\n";
  llvm::errs() << "Diff: " << *dif << "\n";
} assert(ptr->getType()->isPointerTy());
assert(cast<PointerType>(ptr->getType())->getElementType() == dif->getType());

assert(ptr->getType()->isPointerTy());
assert(cast<PointerType>(ptr->getType())->getElementType() == dif->getType());

// const SCEV *S = SE.getSCEV(PN);
// if (SE.getCouldNotCompute() == S)
//  continue;

// atomics
if (AtomicAdd) {
  if (dif->getType()->isIntOrIntVectorTy()) {

    ptr = BuilderM.CreateBitCast(
        ptr,
        PointerType::get(IntToFloatTy(dif->getType()),
                         cast<PointerType>(ptr->getType())->getAddressSpace()));
    dif = BuilderM.CreateBitCast(dif, IntToFloatTy(dif->getType()));
  }
  if (llvm::Triple(newFunc->getParent()->getTargetTriple()).getArch() ==
          Triple::nvptx ||
      llvm::Triple(newFunc->getParent()->getTargetTriple()).getArch() ==
          Triple::nvptx64) {
    if (dif->getType()->isFloatTy()) {
      // auto atomicAdd =
      cast<CallInst>(BuilderM.CreateCall(
          Intrinsic::getDeclaration(newFunc->getParent(),
                                    Intrinsic::nvvm_atomic_add_gen_f_sys,
                                    {dif->getType(), ptr->getType()}),
          {ptr, dif}));
    } else if (dif->getType()->isDoubleTy()) {
      // auto atomicAdd =
      cast<CallInst>(BuilderM.CreateCall(
          Intrinsic::getDeclaration(newFunc->getParent(),
                                    Intrinsic::nvvm_atomic_add_gen_f_sys,
                                    {dif->getType(), ptr->getType()}),
          {ptr, dif}));
    } else {
      llvm::errs() << "unhandled atomic add: " << *ptr << " " << *dif << "\n";
      llvm_unreachable("unhandled atomic add");
    }
  } else {
#if LLVM_VERSION_MAJOR >= 9
    AtomicRMWInst::BinOp op = AtomicRMWInst::FAdd;
#if LLVM_VERSION_MAJOR >= 11
    AtomicRMWInst *rmw = BuilderM.CreateAtomicRMW(
        op, ptr, dif, AtomicOrdering::Monotonic, SyncScope::System);
    if (align)
      rmw->setAlignment(align.getValue());
#else
    BuilderM.CreateAtomicRMW(op, ptr, dif, AtomicOrdering::Monotonic,
                             SyncScope::System);
#endif
#else
        llvm::errs() << "unhandled atomic fadd on llvm version " << *ptr << " "
                     << *dif << "\n";
        llvm_unreachable("unhandled atomic fadd");
#endif
  }
  return;
}

Value *res;
LoadInst *old = BuilderM.CreateLoad(ptr);
#if LLVM_VERSION_MAJOR >= 10
if (align)
  old->setAlignment(align.getValue());
#else
    old->setAlignment(align);
#endif

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
StoreInst *st = BuilderM.CreateStore(res, ptr);
#if LLVM_VERSION_MAJOR >= 10
if (align)
  st->setAlignment(align.getValue());
#else
    st->setAlignment(align);
#endif
}
}
;
#endif
