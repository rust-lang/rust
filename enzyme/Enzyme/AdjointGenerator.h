//===- AdjointGenerator.h - Implementation of Adjoint's of instructions --===//
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
// This file contains an instruction visitor AdjointGenerator that generates
// the corresponding augmented forward pass code, and adjoints for all
// LLVM instructions.
//
//===----------------------------------------------------------------------===//
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Value.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "DifferentialUseAnalysis.h"
#include "EnzymeLogic.h"
#include "FunctionUtils.h"
#include "GradientUtils.h"
#include "LibraryFuncs.h"
#include "TypeAnalysis/TBAA.h"

#define DEBUG_TYPE "enzyme"
using namespace llvm;

// Helper instruction visitor that generates adjoints
template <class AugmentedReturnType = AugmentedReturn *>
class AdjointGenerator
    : public llvm::InstVisitor<AdjointGenerator<AugmentedReturnType>> {
private:
  // Type of code being generated (forward, reverse, or both)
  const DerivativeMode Mode;

  GradientUtils *const gutils;
  ArrayRef<DIFFE_TYPE> constant_args;
  DIFFE_TYPE retType;
  TypeResults &TR = gutils->TR;
  std::function<unsigned(Instruction *, CacheType)> getIndex;
  const std::map<CallInst *, const std::map<Argument *, bool>>
      uncacheable_args_map;
  const SmallPtrSetImpl<Instruction *> *returnuses;
  AugmentedReturnType augmentedReturn;
  const std::map<ReturnInst *, StoreInst *> *replacedReturns;

  const SmallPtrSetImpl<const Value *> &unnecessaryValues;
  const SmallPtrSetImpl<const Instruction *> &unnecessaryInstructions;
  const SmallPtrSetImpl<const Instruction *> &unnecessaryStores;
  const SmallPtrSetImpl<BasicBlock *> &oldUnreachable;
  AllocaInst *dretAlloca;

public:
  AdjointGenerator(
      DerivativeMode Mode, GradientUtils *gutils,
      ArrayRef<DIFFE_TYPE> constant_args, DIFFE_TYPE retType,
      std::function<unsigned(Instruction *, CacheType)> getIndex,
      const std::map<CallInst *, const std::map<Argument *, bool>>
          uncacheable_args_map,
      const SmallPtrSetImpl<Instruction *> *returnuses,
      AugmentedReturnType augmentedReturn,
      const std::map<ReturnInst *, StoreInst *> *replacedReturns,
      const SmallPtrSetImpl<const Value *> &unnecessaryValues,
      const SmallPtrSetImpl<const Instruction *> &unnecessaryInstructions,
      const SmallPtrSetImpl<const Instruction *> &unnecessaryStores,
      const SmallPtrSetImpl<BasicBlock *> &oldUnreachable,
      AllocaInst *dretAlloca)
      : Mode(Mode), gutils(gutils), constant_args(constant_args),
        retType(retType), getIndex(getIndex),
        uncacheable_args_map(uncacheable_args_map), returnuses(returnuses),
        augmentedReturn(augmentedReturn), replacedReturns(replacedReturns),
        unnecessaryValues(unnecessaryValues),
        unnecessaryInstructions(unnecessaryInstructions),
        unnecessaryStores(unnecessaryStores), oldUnreachable(oldUnreachable),
        dretAlloca(dretAlloca) {

    assert(TR.getFunction() == gutils->oldFunc);
    for (auto &pair : TR.analyzer.analysis) {
      if (auto in = dyn_cast<Instruction>(pair.first)) {
        if (in->getParent()->getParent() != gutils->oldFunc) {
          llvm::errs() << "inf: " << *in->getParent()->getParent() << "\n";
          llvm::errs() << "gutils->oldFunc: " << *gutils->oldFunc << "\n";
          llvm::errs() << "in: " << *in << "\n";
        }
        assert(in->getParent()->getParent() == gutils->oldFunc);
      }
    }
  }

  SmallPtrSet<Instruction *, 4> erased;

  void eraseIfUnused(llvm::Instruction &I, bool erase = true,
                     bool check = true) {
    bool used =
        unnecessaryInstructions.find(&I) == unnecessaryInstructions.end();
    if (!used) {
      // if decided to cache a value, preserve it here for later
      // replacement in EnzymeLogic
      auto found = gutils->knownRecomputeHeuristic.find(&I);
      if (found != gutils->knownRecomputeHeuristic.end() && !found->second)
        used = true;
    }
    auto iload = gutils->getNewFromOriginal((Value *)&I);
    if (used && check)
      return;

    PHINode *pn = nullptr;
    if (!I.getType()->isVoidTy() && !I.getType()->isTokenTy() &&
        isa<Instruction>(iload)) {
      IRBuilder<> BuilderZ(cast<Instruction>(iload));
      pn = BuilderZ.CreatePHI(I.getType(), 1,
                              (I.getName() + "_replacementA").str());
      gutils->fictiousPHIs[pn] = &I;
      gutils->replaceAWithB(iload, pn);
    }

    erased.insert(&I);
    if (erase) {
      if (auto inst = dyn_cast<Instruction>(iload)) {
        gutils->erase(inst);
      }
    }
  }

  llvm::Value *MPI_TYPE_SIZE(llvm::Value *DT, IRBuilder<> &B, Type *intType) {
    if (DT->getType()->isIntegerTy())
      DT = B.CreateIntToPtr(DT, Type::getInt8PtrTy(DT->getContext()));

    if (Constant *C = dyn_cast<Constant>(DT)) {
      while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
        C = CE->getOperand(0);
      }
      if (auto GV = dyn_cast<GlobalVariable>(C)) {
        if (GV->getName() == "ompi_mpi_double") {
          return ConstantInt::get(intType, 8, false);
        } else if (GV->getName() == "ompi_mpi_float") {
          return ConstantInt::get(intType, 4, false);
        }
      }
    }
    Type *pargs[] = {Type::getInt8PtrTy(DT->getContext()),
                     PointerType::getUnqual(intType)};
    auto FT = FunctionType::get(intType, pargs, false);
    auto alloc = IRBuilder<>(gutils->inversionAllocs).CreateAlloca(intType);
    llvm::Value *args[] = {DT, alloc};
    if (DT->getType() != pargs[0])
      args[0] = B.CreateBitCast(args[0], pargs[0]);
    AttributeList AL;
    AL = AL.addParamAttribute(DT->getContext(), 0,
                              Attribute::AttrKind::ReadOnly);
    AL = AL.addParamAttribute(DT->getContext(), 0,
                              Attribute::AttrKind::NoCapture);
    AL =
        AL.addParamAttribute(DT->getContext(), 0, Attribute::AttrKind::NoAlias);
    AL =
        AL.addParamAttribute(DT->getContext(), 0, Attribute::AttrKind::NonNull);
    AL = AL.addParamAttribute(DT->getContext(), 1,
                              Attribute::AttrKind::WriteOnly);
    AL = AL.addParamAttribute(DT->getContext(), 1,
                              Attribute::AttrKind::NoCapture);
    AL =
        AL.addParamAttribute(DT->getContext(), 1, Attribute::AttrKind::NoAlias);
    AL =
        AL.addParamAttribute(DT->getContext(), 1, Attribute::AttrKind::NonNull);
#if LLVM_VERSION_MAJOR >= 14
    AL = AL.addAttributeAtIndex(DT->getContext(), AttributeList::FunctionIndex,
                                Attribute::AttrKind::ArgMemOnly);
    AL = AL.addAttributeAtIndex(DT->getContext(), AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoUnwind);
    AL = AL.addAttributeAtIndex(DT->getContext(), AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoFree);
    AL = AL.addAttributeAtIndex(DT->getContext(), AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoSync);
    AL = AL.addAttributeAtIndex(DT->getContext(), AttributeList::FunctionIndex,
                                Attribute::AttrKind::WillReturn);
#elif LLVM_VERSION_MAJOR >= 9
    AL = AL.addAttribute(DT->getContext(), AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoFree);
    AL = AL.addAttribute(DT->getContext(), AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoSync);
    AL = AL.addAttribute(DT->getContext(), AttributeList::FunctionIndex,
                         Attribute::AttrKind::WillReturn);
#endif
#if LLVM_VERSION_MAJOR < 14
    AL = AL.addAttribute(DT->getContext(), AttributeList::FunctionIndex,
                         Attribute::AttrKind::ArgMemOnly);
    AL = AL.addAttribute(DT->getContext(), AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoUnwind);
#endif
    B.CreateCall(
        B.GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
            "MPI_Type_size", FT, AL),
        args);
#if LLVM_VERSION_MAJOR > 7
    return B.CreateLoad(intType, alloc);
#else
    return B.CreateLoad(alloc);
#endif
  }

  // To be double-checked against the functionality needed and the respective
  // implementation in Adjoint-MPI
  llvm::Value *MPI_COMM_RANK(llvm::Value *comm, IRBuilder<> &B, Type *rankTy) {
    Type *pargs[] = {comm->getType(), PointerType::getUnqual(rankTy)};
    auto FT = FunctionType::get(rankTy, pargs, false);
    auto &context = comm->getContext();
    auto alloc = IRBuilder<>(gutils->inversionAllocs).CreateAlloca(rankTy);
    AttributeList AL;
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::ReadOnly);
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::NoCapture);
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::NoAlias);
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::NonNull);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::WriteOnly);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::NoCapture);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::NoAlias);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::NonNull);
#if LLVM_VERSION_MAJOR >= 14
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoUnwind);
#else
    AL = AL.addAttribute(context, AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoUnwind);
#endif
#if LLVM_VERSION_MAJOR >= 14
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoFree);
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoSync);
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::WillReturn);
#elif LLVM_VERSION_MAJOR >= 9
    AL = AL.addAttribute(context, AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoFree);
    AL = AL.addAttribute(context, AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoSync);
    AL = AL.addAttribute(context, AttributeList::FunctionIndex,
                         Attribute::AttrKind::WillReturn);
#endif
    llvm::Value *args[] = {comm, alloc};
    B.CreateCall(
        B.GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
            "MPI_Comm_rank", FT, AL),
        args);
#if LLVM_VERSION_MAJOR > 7
    return B.CreateLoad(rankTy, alloc);
#else
    return B.CreateLoad(alloc);
#endif
  }

  llvm::Value *MPI_COMM_SIZE(llvm::Value *comm, IRBuilder<> &B, Type *rankTy) {
    Type *pargs[] = {comm->getType(), PointerType::getUnqual(rankTy)};
    auto FT = FunctionType::get(rankTy, pargs, false);
    auto &context = comm->getContext();
    auto alloc = IRBuilder<>(gutils->inversionAllocs).CreateAlloca(rankTy);
    AttributeList AL;
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::ReadOnly);
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::NoCapture);
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::NoAlias);
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::NonNull);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::WriteOnly);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::NoCapture);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::NoAlias);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::NonNull);
#if LLVM_VERSION_MAJOR >= 14
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoUnwind);
#else
    AL = AL.addAttribute(context, AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoUnwind);
#endif
#if LLVM_VERSION_MAJOR >= 14
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoFree);
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoSync);
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::WillReturn);
#elif LLVM_VERSION_MAJOR >= 9
    AL = AL.addAttribute(context, AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoFree);
    AL = AL.addAttribute(context, AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoSync);
    AL = AL.addAttribute(context, AttributeList::FunctionIndex,
                         Attribute::AttrKind::WillReturn);
#endif
    llvm::Value *args[] = {comm, alloc};
    B.CreateCall(
        B.GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
            "MPI_Comm_size", FT, AL),
        args);
#if LLVM_VERSION_MAJOR > 7
    return B.CreateLoad(rankTy, alloc);
#else
    return B.CreateLoad(alloc);
#endif
  }

#if LLVM_VERSION_MAJOR >= 10
  void visitFreezeInst(llvm::FreezeInst &inst) {
    eraseIfUnused(inst);
    if (gutils->isConstantInstruction(&inst))
      return;
    Value *orig_op0 = inst.getOperand(0);

    switch (Mode) {
    case DerivativeMode::ReverseModeCombined:
    case DerivativeMode::ReverseModeGradient: {
      IRBuilder<> Builder2(inst.getParent());
      getReverseBuilder(Builder2);

      auto rule = [&](Value *idiff) { return Builder2.CreateFreeze(idiff); };
      Value *idiff = diffe(&inst, Builder2);
      Value *dif1 = applyChainRule(orig_op0->getType(), Builder2, rule, idiff);
      setDiffe(&inst,
               Constant::getNullValue(gutils->getShadowType(inst.getType())),
               Builder2);
      size_t size = 1;
      if (inst.getType()->isSized())
        size = (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                    orig_op0->getType()) +
                7) /
               8;
      addToDiffe(orig_op0, dif1, Builder2, TR.addingType(size, orig_op0));
      return;
    }
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      IRBuilder<> BuilderZ(&inst);
      getForwardBuilder(BuilderZ);

      auto rule = [&](Value *idiff) { return BuilderZ.CreateFreeze(idiff); };
      Value *idiff = diffe(orig_op0, BuilderZ);
      Value *dif1 = applyChainRule(inst.getType(), BuilderZ, rule, idiff);
      setDiffe(&inst, dif1, BuilderZ);
      return;
    }
    case DerivativeMode::ReverseModePrimal:
      return;
    }
  }
#endif

  void visitInstruction(llvm::Instruction &inst) {
    // TODO explicitly handle all instructions rather than using the catch all
    // below

#if LLVM_VERSION_MAJOR >= 10
    if (auto *FPMO = dyn_cast<FPMathOperator>(&inst)) {
      if (FPMO->getOpcode() == Instruction::FNeg) {
        eraseIfUnused(inst);
        if (gutils->isConstantInstruction(&inst))
          return;

        Value *orig_op1 = FPMO->getOperand(0);
        bool constantval1 = gutils->isConstantValue(orig_op1);

        if (constantval1) {
          return;
        }

        switch (Mode) {
        case DerivativeMode::ReverseModeCombined:
        case DerivativeMode::ReverseModeGradient: {
          IRBuilder<> Builder2(inst.getParent());
          getReverseBuilder(Builder2);

          auto rule = [&](Value *idiff) { return Builder2.CreateFNeg(idiff); };
          Value *idiff = diffe(FPMO, Builder2);
          Value *dif1 =
              applyChainRule(orig_op1->getType(), Builder2, rule, idiff);
          setDiffe(
              FPMO,
              Constant::getNullValue(gutils->getShadowType(FPMO->getType())),
              Builder2);
          addToDiffe(orig_op1, dif1, Builder2,
                     dif1->getType()->getScalarType());
          break;
        }
        case DerivativeMode::ForwardModeSplit:
        case DerivativeMode::ForwardMode: {
          IRBuilder<> Builder2(&inst);
          getForwardBuilder(Builder2);

          auto rule = [&Builder2](Value *idiff) {
            return Builder2.CreateFNeg(idiff);
          };

          Value *idiff = diffe(orig_op1, Builder2);
          Value *dif1 = applyChainRule(inst.getType(), Builder2, rule, idiff);

          setDiffe(FPMO, dif1, Builder2);
          break;
        }
        case DerivativeMode::ReverseModePrimal:
          return;
        }
        return;
      }
    }
#endif

    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << *gutils->oldFunc << "\n";
    ss << *gutils->newFunc << "\n";
    ss << "in Mode: " << to_string(Mode) << "\n";
    ss << "cannot handle unknown instruction\n" << inst;
    if (CustomErrorHandler) {
      CustomErrorHandler(ss.str().c_str(), wrap(&inst), ErrorType::NoDerivative,
                         nullptr);
    }
    llvm::errs() << ss.str() << "\n";
    report_fatal_error("unknown instruction");
  }

  // Common function for falling back to the implementation
  // of dual propagation, as available in invertPointerM.
  void forwardModeInvertedPointerFallback(Instruction &I) {
    if (gutils->isConstantValue(&I))
      return;
    auto found = gutils->invertedPointers.find(&I);
    assert(found != gutils->invertedPointers.end());
    auto placeholder = cast<PHINode>(&*found->second);
    gutils->invertedPointers.erase(found);

    if (!is_value_needed_in_reverse<ValueType::Shadow>(gutils, &I, Mode,
                                                       oldUnreachable)) {
      gutils->erase(placeholder);
      return;
    }

    IRBuilder<> Builder2(&I);
    getForwardBuilder(Builder2);

    auto toset = gutils->invertPointerM(&I, Builder2, /*nullShadow*/ true);

    gutils->replaceAWithB(placeholder, toset);
    placeholder->replaceAllUsesWith(toset);
    gutils->erase(placeholder);
    gutils->invertedPointers.insert(
        std::make_pair((const Value *)&I, InvertedPointerVH(gutils, toset)));
    return;
  }

  void visitAllocaInst(llvm::AllocaInst &I) {
    eraseIfUnused(I);
    switch (Mode) {
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(I);
      return;
    }
    default:
      return;
    }
  }
  void visitICmpInst(llvm::ICmpInst &I) { eraseIfUnused(I); }

  void visitFCmpInst(llvm::FCmpInst &I) { eraseIfUnused(I); }

#if LLVM_VERSION_MAJOR >= 10
  void visitLoadLike(llvm::Instruction &I, MaybeAlign alignment,
                     bool constantval, Value *OrigOffset = nullptr,
#else
  void visitLoadLike(llvm::Instruction &I, unsigned alignment, bool constantval,
                     Value *OrigOffset = nullptr,
#endif
                     Value *mask = nullptr, Value *orig_maskInit = nullptr) {
    auto &DL = gutils->newFunc->getParent()->getDataLayout();

    assert(Mode == DerivativeMode::ForwardMode || gutils->can_modref_map);
    assert(Mode == DerivativeMode::ForwardMode ||
           gutils->can_modref_map->find(&I) != gutils->can_modref_map->end());
    bool can_modref = Mode == DerivativeMode::ForwardMode
                          ? false
                          : gutils->can_modref_map->find(&I)->second;

    constantval |= gutils->isConstantValue(&I);

    BasicBlock *parent = I.getParent();
    Type *type = gutils->getShadowType(I.getType());

    auto *newi = dyn_cast<Instruction>(gutils->getNewFromOriginal(&I));

    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeSplit) {
      if (!constantval) {
        auto found = gutils->invertedPointers.find(&I);
        assert(found != gutils->invertedPointers.end());
        Instruction *placeholder = cast<Instruction>(&*found->second);
        assert(placeholder->getType() == type);
        gutils->invertedPointers.erase(found);

        IRBuilder<> BuilderZ(newi);
        // only make shadow where caching needed
        if (!is_value_needed_in_reverse<ValueType::Shadow>(gutils, &I, Mode,
                                                           oldUnreachable)) {
          gutils->erase(placeholder);
          return;
        }

        if (can_modref) {
          if (!I.getType()->isEmptyTy() && !I.getType()->isFPOrFPVectorTy() &&
              TR.query(&I).Inner0().isPossiblePointer()) {
            Value *newip = gutils->cacheForReverse(
                BuilderZ, placeholder, getIndex(&I, CacheType::Shadow));
            assert(newip->getType() == type);
            gutils->invertedPointers.insert(std::make_pair(
                (const Value *)&I, InvertedPointerVH(gutils, newip)));
          } else {
            gutils->erase(placeholder);
          }
        } else {
          Value *newip = gutils->invertPointerM(&I, BuilderZ);
          assert(newip->getType() == type);
          placeholder->replaceAllUsesWith(newip);
          gutils->erase(placeholder);
          gutils->invertedPointers.insert(std::make_pair(
              (const Value *)&I, InvertedPointerVH(gutils, newip)));
        }
      }
      return;
    }

    //! Store inverted pointer loads that need to be cached for use in reverse
    //! pass
    if (!I.getType()->isEmptyTy() && !I.getType()->isFPOrFPVectorTy() &&
        TR.query(&I).Inner0().isPossiblePointer()) {
      auto found = gutils->invertedPointers.find(&I);
      assert(found != gutils->invertedPointers.end());
      Instruction *placeholder = cast<Instruction>(&*found->second);
      assert(placeholder->getType() == type);
      gutils->invertedPointers.erase(found);

      if (!constantval) {
        IRBuilder<> BuilderZ(newi);
        Value *newip = nullptr;

        // TODO: In the case of fwd mode this should be true if the loaded value
        // itself is used as a pointer.
        bool needShadow = is_value_needed_in_reverse<ValueType::Shadow>(
            gutils, &I, Mode, oldUnreachable);

        switch (Mode) {

        case DerivativeMode::ReverseModePrimal:
        case DerivativeMode::ReverseModeCombined: {
          if (!needShadow) {
            gutils->erase(placeholder);
          } else {
            newip = gutils->invertPointerM(&I, BuilderZ);
            assert(newip->getType() == type);
            if (Mode == DerivativeMode::ReverseModePrimal && can_modref &&
                is_value_needed_in_reverse<ValueType::Shadow>(
                    gutils, &I, DerivativeMode::ReverseModeGradient,
                    oldUnreachable)) {
              gutils->cacheForReverse(BuilderZ, newip,
                                      getIndex(&I, CacheType::Shadow));
            }
            placeholder->replaceAllUsesWith(newip);
            gutils->erase(placeholder);
            gutils->invertedPointers.insert(std::make_pair(
                (const Value *)&I, InvertedPointerVH(gutils, newip)));
          }
          break;
        }
        case DerivativeMode::ForwardModeSplit:
        case DerivativeMode::ForwardMode: {
          assert(0 && "impossible branch");
          return;
        }
        case DerivativeMode::ReverseModeGradient: {
          if (!needShadow) {
            gutils->erase(placeholder);
          } else {
            // only make shadow where caching needed
            if (can_modref) {
              newip = gutils->cacheForReverse(BuilderZ, placeholder,
                                              getIndex(&I, CacheType::Shadow));
              assert(newip->getType() == type);
              gutils->invertedPointers.insert(std::make_pair(
                  (const Value *)&I, InvertedPointerVH(gutils, newip)));
            } else {
              newip = gutils->invertPointerM(&I, BuilderZ);
              assert(newip->getType() == type);
              placeholder->replaceAllUsesWith(newip);
              gutils->erase(placeholder);
              gutils->invertedPointers.insert(std::make_pair(
                  (const Value *)&I, InvertedPointerVH(gutils, newip)));
            }
          }
          break;
        }
        }

      } else {
        gutils->erase(placeholder);
      }
    }

    Value *inst = newi;

    //! Store loads that need to be cached for use in reverse pass

    // Only cache value here if caching decision isn't precomputed.
    // Otherwise caching will be done inside EnzymeLogic.cpp at
    // the end of the function jointly.
    if (Mode != DerivativeMode::ForwardMode &&
        !gutils->knownRecomputeHeuristic.count(&I) && can_modref &&
        !gutils->unnecessaryIntermediates.count(&I)) {
      // we can pre initialize all the knownRecomputeHeuristic values to false
      // (not needing) as we may assume that minCutCache already preserves
      // everything it requires.
      std::map<UsageKey, bool> Seen;
      bool primalNeededInReverse = false;
      for (auto pair : gutils->knownRecomputeHeuristic)
        if (!pair.second) {
          Seen[UsageKey(pair.first, ValueType::Primal)] = false;
          if (pair.first == &I)
            primalNeededInReverse = true;
        }
      primalNeededInReverse |= is_value_needed_in_reverse<ValueType::Primal>(
          gutils, &I, Mode, Seen, oldUnreachable);
      if (primalNeededInReverse) {
        IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&I));
        inst = gutils->cacheForReverse(BuilderZ, newi,
                                       getIndex(&I, CacheType::Self));
        assert(inst->getType() == type);

        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ForwardModeSplit) {
          assert(inst != newi);
        } else {
          assert(inst == newi);
        }
      }
    }

    if (Mode == DerivativeMode::ReverseModePrimal)
      return;

    if (constantval)
      return;

    if (nonmarkedglobals_inactiveloads) {
      // Assume that non enzyme_shadow globals are inactive
      //  If we ever store to a global variable, we will error if it doesn't
      //  have a shadow This allows functions who only read global memory to
      //  have their derivative computed Note that this is too aggressive for
      //  general programs as if the global aliases with an argument something
      //  that is written to, then we will have a logical error
      if (auto arg = dyn_cast<GlobalVariable>(I.getOperand(0))) {
        if (!hasMetadata(arg, "enzyme_shadow")) {
          return;
        }
      }
    }

    // Only propagate if instruction is active. The value can be active and not
    // the instruction if the value is a potential pointer. This may not be
    // caught by type analysis is the result does not have a known type.
    if (!gutils->isConstantInstruction(&I)) {
      Type *isfloat = I.getType()->isFPOrFPVectorTy()
                          ? I.getType()->getScalarType()
                          : nullptr;
      if (!isfloat && I.getType()->isIntOrIntVectorTy()) {
        auto LoadSize = DL.getTypeSizeInBits(I.getType()) / 8;
        ConcreteType vd = BaseType::Unknown;
        if (!OrigOffset)
          vd =
              TR.firstPointer(LoadSize, I.getOperand(0),
                              /*errifnotfound*/ false, /*pointerIntSame*/ true);
        if (vd.isKnown())
          isfloat = vd.isFloat();
        else {
          isfloat =
              TR.intType(LoadSize, &I, /*errIfNotFound*/ !looseTypeAnalysis)
                  .isFloat();
        }
      }

      if (isfloat) {
        switch (Mode) {
        case DerivativeMode::ForwardModeSplit:
        case DerivativeMode::ForwardMode: {
          assert(0 && "impossible branch");
          return;
        }
        case DerivativeMode::ReverseModeGradient:
        case DerivativeMode::ReverseModeCombined: {
          IRBuilder<> Builder2(parent);
          getReverseBuilder(Builder2);

          Value *prediff = diffe(&I, Builder2);
          setDiffe(&I, Constant::getNullValue(type), Builder2);

          if (mask && (!gutils->isConstantValue(I.getOperand(0)) ||
                       !gutils->isConstantValue(orig_maskInit)))
            mask = lookup(mask, Builder2);

          if (!gutils->isConstantValue(I.getOperand(0))) {
            BasicBlock *merge = nullptr;
            if (EnzymeRuntimeActivityCheck) {
              Value *shadow = Builder2.CreateICmpNE(
                  lookup(gutils->getNewFromOriginal(I.getOperand(0)), Builder2),
                  lookup(gutils->invertPointerM(I.getOperand(0), Builder2),
                         Builder2));

              BasicBlock *current = Builder2.GetInsertBlock();
              BasicBlock *conditional = gutils->addReverseBlock(
                  current, current->getName() + "_active");
              merge = gutils->addReverseBlock(conditional,
                                              current->getName() + "_amerge");
              Builder2.CreateCondBr(shadow, conditional, merge);
              Builder2.SetInsertPoint(conditional);
            }
            ((DiffeGradientUtils *)gutils)
                ->addToInvertedPtrDiffe(I.getOperand(0), prediff, Builder2,
                                        alignment, OrigOffset, mask);
            if (merge) {
              Builder2.CreateBr(merge);
              Builder2.SetInsertPoint(merge);
            }
          }
          if (mask && !gutils->isConstantValue(orig_maskInit)) {
            addToDiffe(orig_maskInit, prediff, Builder2, isfloat,
                       Builder2.CreateNot(mask));
          }
          break;
        }
        case DerivativeMode::ReverseModePrimal:
          break;
        }
      }
    }
  }

  void visitLoadInst(llvm::LoadInst &LI) {
    // If a load of an omp init argument, don't cache for reverse
    // and don't do any adjoint propagation (assumed integral)
    for (auto U : LI.getPointerOperand()->users()) {
      if (auto CI = dyn_cast<CallInst>(U)) {
        if (auto F = CI->getCalledFunction()) {
          if (F->getName() == "__kmpc_for_static_init_4" ||
              F->getName() == "__kmpc_for_static_init_4u" ||
              F->getName() == "__kmpc_for_static_init_8" ||
              F->getName() == "__kmpc_for_static_init_8u") {
            eraseIfUnused(LI);
            return;
          }
        }
      }
    }

#if LLVM_VERSION_MAJOR >= 10
    auto alignment = LI.getAlign();
#else
    auto alignment = LI.getAlignment();
#endif

    auto &DL = gutils->newFunc->getParent()->getDataLayout();

    bool constantval = parseTBAA(LI, DL).Inner0().isIntegral();
    visitLoadLike(LI, alignment, constantval);
    eraseIfUnused(LI);
  }

  void visitAtomicRMWInst(llvm::AtomicRMWInst &I) {
    if (Mode == DerivativeMode::ForwardMode) {
      IRBuilder<> BuilderZ(&I);
      getForwardBuilder(BuilderZ);
      switch (I.getOperation()) {
      case AtomicRMWInst::FAdd:
      case AtomicRMWInst::FSub: {
        auto rule = [&](Value *ptr, Value *dif) -> Value * {
          if (!gutils->isConstantInstruction(&I)) {
            assert(ptr);
            AtomicRMWInst *rmw = nullptr;
#if LLVM_VERSION_MAJOR >= 13
            rmw = BuilderZ.CreateAtomicRMW(I.getOperation(), ptr, dif,
                                           I.getAlign(), I.getOrdering(),
                                           I.getSyncScopeID());
#elif LLVM_VERSION_MAJOR >= 11
            rmw = BuilderZ.CreateAtomicRMW(I.getOperation(), ptr, dif,
                                           I.getOrdering(), I.getSyncScopeID());
            rmw->setAlignment(I.getAlign());
#else
                               rmw = BuilderZ.CreateAtomicRMW(
                                   I.getOperation(), ptr, dif, I.getOrdering(),
                                   I.getSyncScopeID());
#endif
            rmw->setVolatile(I.isVolatile());
            if (gutils->isConstantValue(&I))
              return Constant::getNullValue(dif->getType());
            else
              return rmw;
          } else {
            assert(gutils->isConstantValue(&I));
            return Constant::getNullValue(dif->getType());
          }
        };

        Value *diff = applyChainRule(
            I.getType(), BuilderZ, rule,
            gutils->isConstantValue(I.getPointerOperand())
                ? nullptr
                : gutils->invertPointerM(I.getPointerOperand(), BuilderZ),
            gutils->isConstantValue(I.getValOperand())
                ? Constant::getNullValue(I.getType())
                : gutils->invertPointerM(I.getValOperand(), BuilderZ));
        if (!gutils->isConstantValue(&I))
          setDiffe(&I, diff, BuilderZ);
        return;
      }
      default:
        break;
      }
    }
    if (!gutils->isConstantInstruction(&I) || !gutils->isConstantValue(&I)) {
      if (looseTypeAnalysis) {
        auto &DL = gutils->newFunc->getParent()->getDataLayout();
        auto valType = I.getValOperand()->getType();
        auto storeSize = DL.getTypeSizeInBits(valType) / 8;
        auto fp = TR.firstPointer(storeSize, I.getPointerOperand(),
                                  /*errifnotfound*/ false,
                                  /*pointerIntSame*/ true);
        if (!fp.isKnown() && valType->isIntOrIntVectorTy()) {
          goto noerror;
        }
      }
      TR.dump();
      llvm::errs() << "oldFunc: " << *gutils->newFunc << "\n";
      llvm::errs() << "I: " << I << "\n";
      assert(0 && "Active atomic inst not handled");
    }
  noerror:;

    if (Mode == DerivativeMode::ReverseModeGradient) {
      eraseIfUnused(I, /*erase*/ true, /*check*/ false);
    }
  }

  void visitStoreInst(llvm::StoreInst &SI) {
    // If a store of an omp init argument, don't delete in reverse
    // and don't do any adjoint propagation (assumed integral)
    for (auto U : SI.getPointerOperand()->users()) {
      if (auto CI = dyn_cast<CallInst>(U)) {
        if (auto F = CI->getCalledFunction()) {
          if (F->getName() == "__kmpc_for_static_init_4" ||
              F->getName() == "__kmpc_for_static_init_4u" ||
              F->getName() == "__kmpc_for_static_init_8" ||
              F->getName() == "__kmpc_for_static_init_8u") {
            return;
          }
        }
      }
    }
#if LLVM_VERSION_MAJOR >= 10
    auto align = SI.getAlign();
#else
    auto align = SI.getAlignment();
#endif

    visitCommonStore(SI, SI.getPointerOperand(), SI.getValueOperand(), align,
                     SI.isVolatile(), SI.getOrdering(), SI.getSyncScopeID(),
                     /*mask=*/nullptr);
    eraseIfUnused(SI);
  }

#if LLVM_VERSION_MAJOR >= 10
  void visitCommonStore(llvm::Instruction &I, Value *orig_ptr, Value *orig_val,
                        MaybeAlign align, bool isVolatile,
                        AtomicOrdering ordering, SyncScope::ID syncScope,
                        Value *mask)
#else
  void visitCommonStore(llvm::Instruction &I, Value *orig_ptr, Value *orig_val,
                        unsigned align, bool isVolatile,
                        AtomicOrdering ordering, SyncScope::ID syncScope,
                        Value *mask)
#endif
  {
    Value *val = gutils->getNewFromOriginal(orig_val);
    Type *valType = orig_val->getType();

    auto &DL = gutils->newFunc->getParent()->getDataLayout();

    if (unnecessaryStores.count(&I)) {
      return;
    }

    if (gutils->isConstantValue(orig_ptr)) {
      return;
    }

    bool constantval = gutils->isConstantValue(orig_val) ||
                       parseTBAA(I, DL).Inner0().isIntegral();

    if (Mode == DerivativeMode::ForwardMode) {
      IRBuilder<> Builder2(&I);
      getForwardBuilder(Builder2);

      Value *diff;
      // TODO type analyze
      if (!constantval)
        diff = gutils->invertPointerM(orig_val, Builder2, /*nullShadow*/ true);
      else if (orig_val->getType()->isPointerTy())
        diff = gutils->invertPointerM(orig_val, Builder2, /*nullShadow*/ false);
      else
        diff = gutils->invertPointerM(orig_val, Builder2, /*nullShadow*/ true);

      gutils->setPtrDiffe(orig_ptr, diff, Builder2, align, isVolatile, ordering,
                          syncScope, mask);
      return;
    }

    // TODO allow recognition of other types that could contain pointers [e.g.
    // {void*, void*} or <2 x i64> ]
    auto storeSize = DL.getTypeSizeInBits(valType) / 8;

    //! Storing a floating point value
    Type *FT = nullptr;
    if (valType->isFPOrFPVectorTy()) {
      FT = valType->getScalarType();
    } else if (!valType->isPointerTy()) {
      if (looseTypeAnalysis) {
        auto fp = TR.firstPointer(storeSize, orig_ptr, /*errifnotfound*/ false,
                                  /*pointerIntSame*/ true);
        if (fp.isKnown()) {
          FT = fp.isFloat();
        } else if (isa<ConstantInt>(orig_val) ||
                   valType->isIntOrIntVectorTy()) {
          llvm::errs() << "assuming type as integral for store: " << I << "\n";
          FT = nullptr;
        } else {
          TR.firstPointer(storeSize, orig_ptr, /*errifnotfound*/ true,
                          /*pointerIntSame*/ true);
          llvm::errs() << "cannot deduce type of store " << I << "\n";
          assert(0 && "cannot deduce");
        }
      } else {
        FT = TR.firstPointer(storeSize, orig_ptr, /*errifnotfound*/ true,
                             /*pointerIntSame*/ true)
                 .isFloat();
      }
    }

    if (FT) {
      //! Only need to update the reverse function
      switch (Mode) {
      case DerivativeMode::ReverseModePrimal:
        break;
      case DerivativeMode::ReverseModeGradient:
      case DerivativeMode::ReverseModeCombined: {
        IRBuilder<> Builder2(I.getParent());
        getReverseBuilder(Builder2);

        if (constantval) {
          gutils->setPtrDiffe(
              orig_ptr, Constant::getNullValue(gutils->getShadowType(valType)),
              Builder2, align, isVolatile, ordering, syncScope, mask);
        } else {
          Value *diff;
          if (!mask) {
            Value *dif1Ptr =
                lookup(gutils->invertPointerM(orig_ptr, Builder2), Builder2);

            auto rule = [&](Value *dif1Ptr) {
#if LLVM_VERSION_MAJOR > 7
              LoadInst *dif1 = Builder2.CreateLoad(
                  dif1Ptr->getType()->getPointerElementType(), dif1Ptr,
                  isVolatile);
#else
              LoadInst *dif1 = Builder2.CreateLoad(dif1Ptr, isVolatile);
#endif
              if (align)
#if LLVM_VERSION_MAJOR >= 10
                dif1->setAlignment(*align);
#else
                dif1->setAlignment(align);
#endif
              dif1->setOrdering(ordering);
              dif1->setSyncScopeID(syncScope);
              return dif1;
            };

            diff = applyChainRule(valType, Builder2, rule, dif1Ptr);
          } else {
            mask = lookup(mask, Builder2);
            Type *tys[] = {valType, orig_ptr->getType()};
            auto F = Intrinsic::getDeclaration(gutils->oldFunc->getParent(),
                                               Intrinsic::masked_load, tys);
#if LLVM_VERSION_MAJOR >= 10
            Value *alignv =
                ConstantInt::get(Type::getInt32Ty(mask->getContext()),
                                 align ? align->value() : 0);
#else
            Value *alignv =
                ConstantInt::get(Type::getInt32Ty(mask->getContext()), align);
#endif
            Value *ip =
                lookup(gutils->invertPointerM(orig_ptr, Builder2), Builder2);

            auto rule = [&](Value *ip) {
              Value *args[] = {ip, alignv, mask,
                               Constant::getNullValue(valType)};
              diff = Builder2.CreateCall(F, args);
              return diff;
            };

            diff = applyChainRule(valType, Builder2, rule, ip);
          }

          gutils->setPtrDiffe(
              orig_ptr, Constant::getNullValue(gutils->getShadowType(valType)),
              Builder2, align, isVolatile, ordering, syncScope, mask);
          addToDiffe(orig_val, diff, Builder2, FT, mask);
        }
        break;
      }
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode: {
        IRBuilder<> Builder2(&I);
        getForwardBuilder(Builder2);

        Type *diffeTy = gutils->getShadowType(valType);

        Value *diff = constantval ? Constant::getNullValue(diffeTy)
                                  : diffe(orig_val, Builder2);
        gutils->setPtrDiffe(orig_ptr, diff, Builder2, align, isVolatile,
                            ordering, syncScope, mask);

        break;
      }
      }

      //! Storing an integer or pointer
    } else {
      //! Only need to update the forward function

      // Don't reproduce mpi null requests
      if (constantval)
        if (Constant *C = dyn_cast<Constant>(orig_val)) {
          while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
            C = CE->getOperand(0);
          }
          if (auto GV = dyn_cast<GlobalVariable>(C)) {
            if (GV->getName() == "ompi_request_null") {
              return;
            }
          }
        }

      bool backwardsShadow = false;
      bool forwardsShadow = true;
      for (auto pair : gutils->backwardsOnlyShadows) {
        if (pair.second.stores.count(&I)) {
          backwardsShadow = true;
          forwardsShadow = pair.second.primalInitialize;
          if (auto inst = dyn_cast<Instruction>(pair.first))
            if (!forwardsShadow && pair.second.LI &&
                pair.second.LI->contains(inst->getParent()))
              backwardsShadow = false;
        }
      }

      if ((Mode == DerivativeMode::ReverseModePrimal && forwardsShadow) ||
          (Mode == DerivativeMode::ReverseModeGradient && backwardsShadow) ||
          (Mode == DerivativeMode::ForwardModeSplit && backwardsShadow) ||
          (Mode == DerivativeMode::ReverseModeCombined &&
           (forwardsShadow || backwardsShadow)) ||
          Mode == DerivativeMode::ForwardMode) {
        IRBuilder<> storeBuilder(gutils->getNewFromOriginal(&I));

        Value *valueop = nullptr;

        if (constantval) {
          valueop = val;
          if (gutils->getWidth() > 1) {
            Value *array =
                UndefValue::get(gutils->getShadowType(val->getType()));
            for (unsigned i = 0; i < gutils->getWidth(); ++i) {
              array = storeBuilder.CreateInsertValue(array, val, {i});
            }
            valueop = array;
          }
        } else {
          valueop = gutils->invertPointerM(orig_val, storeBuilder);
        }
        gutils->setPtrDiffe(orig_ptr, valueop, storeBuilder, align, isVolatile,
                            ordering, syncScope, mask);
      }
    }
  }

  void visitGetElementPtrInst(llvm::GetElementPtrInst &gep) {
    eraseIfUnused(gep);
    switch (Mode) {
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(gep);
      return;
    }
    default:
      return;
    }
  }

  void visitPHINode(llvm::PHINode &phi) {
    eraseIfUnused(phi);

    switch (Mode) {
    case DerivativeMode::ReverseModePrimal:
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
      return;
    }
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(phi);
      return;
    }
    }
  }

  void visitCastInst(llvm::CastInst &I) {
    eraseIfUnused(I);

    switch (Mode) {
    case DerivativeMode::ReverseModePrimal: {
      return;
    }
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
      if (gutils->isConstantInstruction(&I))
        return;

      if (I.getType()->isPointerTy() ||
          I.getOpcode() == CastInst::CastOps::PtrToInt)
        return;

      Value *orig_op0 = I.getOperand(0);
      Value *op0 = gutils->getNewFromOriginal(orig_op0);

      IRBuilder<> Builder2(I.getParent());
      getReverseBuilder(Builder2);

      if (!gutils->isConstantValue(orig_op0)) {
        size_t size = 1;
        if (orig_op0->getType()->isSized())
          size =
              (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                   orig_op0->getType()) +
               7) /
              8;
        Type *FT = TR.addingType(size, orig_op0);
        if (!FT) {
          llvm::errs() << " " << *gutils->oldFunc << "\n";
          TR.dump();
          llvm::errs() << " " << *orig_op0 << "\n";
        }
        assert(FT);

        auto rule = [&](Value *dif) {
          if (I.getOpcode() == CastInst::CastOps::FPTrunc ||
              I.getOpcode() == CastInst::CastOps::FPExt) {
            return Builder2.CreateFPCast(dif, op0->getType());
          } else if (I.getOpcode() == CastInst::CastOps::BitCast) {
            return Builder2.CreateBitCast(dif, op0->getType());
          } else if (I.getOpcode() == CastInst::CastOps::Trunc) {
            // TODO CHECK THIS
            return Builder2.CreateZExt(dif, op0->getType());
          } else {
            std::string s;
            llvm::raw_string_ostream ss(s);
            ss << *I.getParent()->getParent() << "\n" << *I.getParent() << "\n";
            ss << "cannot handle above cast " << I << "\n";
            if (CustomErrorHandler) {
              CustomErrorHandler(ss.str().c_str(), wrap(&I),
                                 ErrorType::NoDerivative, nullptr);
            }
            TR.dump();
            llvm::errs() << ss.str() << "\n";
            report_fatal_error("unknown instruction");
          }
        };

        Value *dif = diffe(&I, Builder2);
        Value *diff = applyChainRule(op0->getType(), Builder2, rule, dif);

        addToDiffe(orig_op0, diff, Builder2, FT);
      }

      Type *diffTy = gutils->getShadowType(I.getType());
      setDiffe(&I, Constant::getNullValue(diffTy), Builder2);

      break;
    }
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(I);
      return;
    }
    }
  }

  void visitSelectInst(llvm::SelectInst &SI) {
    eraseIfUnused(SI);

    switch (Mode) {
    case DerivativeMode::ReverseModePrimal:
      return;
    case DerivativeMode::ReverseModeCombined:
    case DerivativeMode::ReverseModeGradient: {
      if (gutils->isConstantInstruction(&SI))
        return;
      if (SI.getType()->isPointerTy())
        return;
      createSelectInstAdjoint(SI);
      return;
    }
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(SI);
      return;
    }
    }
  }

  void createSelectInstAdjoint(llvm::SelectInst &SI) {
    Value *op0 = gutils->getNewFromOriginal(SI.getOperand(0));
    Value *orig_op1 = SI.getOperand(1);
    Value *op1 = gutils->getNewFromOriginal(orig_op1);
    Value *orig_op2 = SI.getOperand(2);
    Value *op2 = gutils->getNewFromOriginal(orig_op2);

    // TODO fix all the reverse builders
    IRBuilder<> Builder2(SI.getParent());
    getReverseBuilder(Builder2);

    Value *dif1 = nullptr;
    Value *dif2 = nullptr;

    size_t size = 1;
    if (orig_op1->getType()->isSized())
      size = (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                  orig_op1->getType()) +
              7) /
             8;
    // Required loopy phi = [in, BO, BO, ..., BO]
    //  1) phi is only used in this B0
    //  2) BO dominates all latches
    //  3) phi == B0 whenever not coming from preheader [implies 2]
    //  4) [optional but done for ease] one exit to make it easier to
    //  calculation the product at that point
    for (int i = 0; i < 2; i++)
      if (auto P0 = dyn_cast<PHINode>(SI.getOperand(i + 1))) {
        LoopContext lc;
        SmallVector<Instruction *, 4> activeUses;
        for (auto u : P0->users()) {
          if (!gutils->isConstantInstruction(cast<Instruction>(u))) {
            activeUses.push_back(cast<Instruction>(u));
          } else if (retType == DIFFE_TYPE::OUT_DIFF && isa<ReturnInst>(u))
            activeUses.push_back(cast<Instruction>(u));
        }
        if (activeUses.size() == 1 && activeUses[0] == &SI &&
            gutils->getContext(gutils->getNewFromOriginal(P0->getParent()),
                               lc) &&
            gutils->getNewFromOriginal(P0->getParent()) == lc.header) {
          SmallVector<BasicBlock *, 1> Latches;
          gutils->OrigLI.getLoopFor(P0->getParent())->getLoopLatches(Latches);
          bool allIncoming = true;
          for (auto Latch : Latches) {
            if (&SI != P0->getIncomingValueForBlock(Latch)) {
              allIncoming = false;
              break;
            }
          }
          if (allIncoming && lc.exitBlocks.size() == 1) {
            if (!gutils->isConstantValue(SI.getOperand(2 - i))) {
              auto addingType = TR.addingType(size, SI.getOperand(2 - i));
              if (addingType || !looseTypeAnalysis) {
                auto index = gutils->getOrInsertConditionalIndex(
                    gutils->getNewFromOriginal(SI.getOperand(0)), lc, i == 1);
                IRBuilder<> EB(*lc.exitBlocks.begin());
                getReverseBuilder(EB, /*original=*/false);
                Value *inc = lookup(lc.incvar, Builder2);
                if (VectorType *VTy =
                        dyn_cast<VectorType>(SI.getOperand(0)->getType())) {
#if LLVM_VERSION_MAJOR >= 12
                  inc = Builder2.CreateVectorSplat(VTy->getElementCount(), inc);
#else
                  inc = Builder2.CreateVectorSplat(VTy->getNumElements(), inc);
#endif
                }
                Value *dif = Builder2.CreateSelect(
                    Builder2.CreateICmpEQ(gutils->lookupM(index, EB), inc),
                    diffe(&SI, Builder2),
                    Constant::getNullValue(
                        gutils->getShadowType(op1->getType())));
                addToDiffe(SI.getOperand(2 - i), dif, Builder2, addingType);
              }
            }
            return;
          }
        }
      }

    if (!gutils->isConstantValue(orig_op1))
      dif1 = Builder2.CreateSelect(
          lookup(op0, Builder2), diffe(&SI, Builder2),
          Constant::getNullValue(gutils->getShadowType(op1->getType())),
          "diffe" + op1->getName());
    if (!gutils->isConstantValue(orig_op2))
      dif2 = Builder2.CreateSelect(
          lookup(op0, Builder2),
          Constant::getNullValue(gutils->getShadowType(op2->getType())),
          diffe(&SI, Builder2), "diffe" + op2->getName());

    setDiffe(&SI, Constant::getNullValue(gutils->getShadowType(SI.getType())),
             Builder2);
    if (dif1) {
      Type *addingType = TR.addingType(size, orig_op1);
      if (addingType || !looseTypeAnalysis)
        addToDiffe(orig_op1, dif1, Builder2, addingType);
      else
        llvm::errs() << " warning: assuming integral for " << SI << "\n";
    }
    if (dif2) {
      Type *addingType = TR.addingType(size, orig_op2);
      if (addingType || !looseTypeAnalysis)
        addToDiffe(orig_op2, dif2, Builder2, addingType);
      else
        llvm::errs() << " warning: assuming integral for " << SI << "\n";
    }
  }

  void visitExtractElementInst(llvm::ExtractElementInst &EEI) {
    eraseIfUnused(EEI);
    switch (Mode) {
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(EEI);
      return;
    }
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
      if (gutils->isConstantInstruction(&EEI))
        return;
      IRBuilder<> Builder2(EEI.getParent());
      getReverseBuilder(Builder2);

      Value *orig_vec = EEI.getVectorOperand();

      if (!gutils->isConstantValue(orig_vec)) {
        Value *sv[] = {gutils->getNewFromOriginal(EEI.getIndexOperand())};

        size_t size = 1;
        if (EEI.getType()->isSized())
          size =
              (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                   EEI.getType()) +
               7) /
              8;
        ((DiffeGradientUtils *)gutils)
            ->addToDiffe(orig_vec, diffe(&EEI, Builder2), Builder2,
                         TR.addingType(size, &EEI), sv);
      }
      setDiffe(&EEI,
               Constant::getNullValue(gutils->getShadowType(EEI.getType())),
               Builder2);
      return;
    }
    case DerivativeMode::ReverseModePrimal: {
      return;
    }
    }
  }

  void visitInsertElementInst(llvm::InsertElementInst &IEI) {
    eraseIfUnused(IEI);

    switch (Mode) {
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(IEI);
      return;
    }
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
      if (gutils->isConstantInstruction(&IEI))
        return;
      IRBuilder<> Builder2(IEI.getParent());
      getReverseBuilder(Builder2);

      Value *dif1 = diffe(&IEI, Builder2);

      Value *orig_op0 = IEI.getOperand(0);
      Value *orig_op1 = IEI.getOperand(1);
      Value *op1 = gutils->getNewFromOriginal(orig_op1);
      Value *op2 = gutils->getNewFromOriginal(IEI.getOperand(2));

      size_t size0 = 1;
      if (orig_op0->getType()->isSized())
        size0 =
            (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                 orig_op0->getType()) +
             7) /
            8;
      size_t size1 = 1;
      if (orig_op1->getType()->isSized())
        size1 =
            (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                 orig_op1->getType()) +
             7) /
            8;

      if (!gutils->isConstantValue(orig_op0))
        addToDiffe(
            orig_op0,
            Builder2.CreateInsertElement(
                dif1,
                Constant::getNullValue(gutils->getShadowType(op1->getType())),
                lookup(op2, Builder2)),
            Builder2, TR.addingType(size0, orig_op0));

      if (!gutils->isConstantValue(orig_op1))
        addToDiffe(orig_op1,
                   Builder2.CreateExtractElement(dif1, lookup(op2, Builder2)),
                   Builder2, TR.addingType(size1, orig_op1));

      setDiffe(&IEI,
               Constant::getNullValue(gutils->getShadowType(IEI.getType())),
               Builder2);
      return;
    }
    case DerivativeMode::ReverseModePrimal: {
      return;
    }
    }
  }

  void visitShuffleVectorInst(llvm::ShuffleVectorInst &SVI) {
    eraseIfUnused(SVI);

    switch (Mode) {
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(SVI);
      return;
    }
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
      if (gutils->isConstantInstruction(&SVI))
        return;
      IRBuilder<> Builder2(SVI.getParent());
      getReverseBuilder(Builder2);

      auto loaded = diffe(&SVI, Builder2);
#if LLVM_VERSION_MAJOR >= 12
      auto count =
          cast<VectorType>(SVI.getOperand(0)->getType())->getElementCount();
      assert(!count.isScalable());
      size_t l1 = count.getKnownMinValue();
#else
      size_t l1 =
          cast<VectorType>(SVI.getOperand(0)->getType())->getNumElements();
#endif
      uint64_t instidx = 0;

      for (size_t idx : SVI.getShuffleMask()) {
        auto opnum = (idx < l1) ? 0 : 1;
        auto opidx = (idx < l1) ? idx : (idx - l1);
        Value *sv[] = {
            ConstantInt::get(Type::getInt32Ty(SVI.getContext()), opidx)};

        if (!gutils->isConstantValue(SVI.getOperand(opnum))) {
          size_t size = 1;
          if (SVI.getOperand(opnum)->getType()->isSized())
            size = (gutils->newFunc->getParent()
                        ->getDataLayout()
                        .getTypeSizeInBits(SVI.getOperand(opnum)->getType()) +
                    7) /
                   8;
          ((DiffeGradientUtils *)gutils)
              ->addToDiffe(SVI.getOperand(opnum),
                           Builder2.CreateExtractElement(loaded, instidx),
                           Builder2, TR.addingType(size, SVI.getOperand(opnum)),
                           sv);
        }
        ++instidx;
      }
      setDiffe(&SVI,
               Constant::getNullValue(gutils->getShadowType(SVI.getType())),
               Builder2);
      return;
    }
    case DerivativeMode::ReverseModePrimal: {
      return;
    }
    }
  }

  void visitExtractValueInst(llvm::ExtractValueInst &EVI) {
    eraseIfUnused(EVI);

    switch (Mode) {
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(EVI);
      return;
    }
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
      if (gutils->isConstantInstruction(&EVI))
        return;
      if (EVI.getType()->isPointerTy())
        return;
      IRBuilder<> Builder2(EVI.getParent());
      getReverseBuilder(Builder2);

      Value *orig_op0 = EVI.getOperand(0);

      auto prediff = diffe(&EVI, Builder2);

      // todo const
      if (!gutils->isConstantValue(orig_op0)) {
        SmallVector<Value *, 4> sv;
        for (auto i : EVI.getIndices())
          sv.push_back(ConstantInt::get(Type::getInt32Ty(EVI.getContext()), i));
        size_t size = 1;
        if (EVI.getType()->isSized())
          size =
              (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                   EVI.getType()) +
               7) /
              8;
        ((DiffeGradientUtils *)gutils)
            ->addToDiffe(orig_op0, prediff, Builder2, TR.addingType(size, &EVI),
                         sv);
      }

      setDiffe(&EVI,
               Constant::getNullValue(gutils->getShadowType(EVI.getType())),
               Builder2);
      return;
    }
    case DerivativeMode::ReverseModePrimal: {
      return;
    }
    }
  }

  void visitInsertValueInst(llvm::InsertValueInst &IVI) {
    eraseIfUnused(IVI);
    if (gutils->isConstantValue(&IVI))
      return;

    if (Mode == DerivativeMode::ReverseModePrimal)
      return;

    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeSplit) {
      forwardModeInvertedPointerFallback(IVI);
      return;
    }

    bool hasNonPointer = false;
    if (auto st = dyn_cast<StructType>(IVI.getType())) {
      for (unsigned i = 0; i < st->getNumElements(); ++i) {
        if (!st->getElementType(i)->isPointerTy()) {
          hasNonPointer = true;
        }
      }
    } else if (auto at = dyn_cast<ArrayType>(IVI.getType())) {
      if (!at->getElementType()->isPointerTy()) {
        hasNonPointer = true;
      }
    }
    if (!hasNonPointer)
      return;

    bool floatingInsertion = false;
    for (InsertValueInst *iv = &IVI;;) {
      size_t size0 = 1;
      if (iv->getInsertedValueOperand()->getType()->isSized() &&
          (iv->getInsertedValueOperand()->getType()->isIntOrIntVectorTy() ||
           iv->getInsertedValueOperand()->getType()->isFPOrFPVectorTy()))
        size0 =
            (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                 iv->getInsertedValueOperand()->getType()) +
             7) /
            8;
      auto it = TR.intType(size0, iv->getInsertedValueOperand(), false);
      if (it.isFloat() || !it.isKnown()) {
        floatingInsertion = true;
        break;
      }
      Value *val = iv->getAggregateOperand();
      if (gutils->isConstantValue(val))
        break;
      if (auto dc = dyn_cast<InsertValueInst>(val)) {
        iv = dc;
      } else {
        // unsure where this came from, conservatively assume contains float
        floatingInsertion = true;
        break;
      }
    }

    if (!floatingInsertion)
      return;

    // TODO handle pointers
    // TODO type analysis handle structs

    switch (Mode) {
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode:
      assert(0 && "should be handled above");
      return;
    case DerivativeMode::ReverseModeCombined:
    case DerivativeMode::ReverseModeGradient: {
      IRBuilder<> Builder2(IVI.getParent());
      getReverseBuilder(Builder2);

      Value *orig_inserted = IVI.getInsertedValueOperand();
      Value *orig_agg = IVI.getAggregateOperand();

      size_t size0 = 1;
      if (orig_inserted->getType()->isSized())
        size0 =
            (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                 orig_inserted->getType()) +
             7) /
            8;

      if (!gutils->isConstantValue(orig_inserted)) {
        auto it =
            TR.intType(size0, orig_inserted, /*errIfFalse*/ !looseTypeAnalysis);
        Type *flt = it.isFloat();
        if (!it.isKnown()) {
          assert(looseTypeAnalysis);
          if (orig_inserted->getType()->isFPOrFPVectorTy())
            flt = orig_inserted->getType()->getScalarType();
          else if (orig_inserted->getType()->isIntOrIntVectorTy())
            flt = nullptr;
          else
            TR.intType(size0, orig_inserted);
        }
        if (flt) {
          auto rule = [&](Value *prediff) {
            return Builder2.CreateExtractValue(prediff, IVI.getIndices());
          };
          auto prediff = diffe(&IVI, Builder2);
          auto dindex =
              applyChainRule(orig_inserted->getType(), Builder2, rule, prediff);
          addToDiffe(orig_inserted, dindex, Builder2, flt);
        }
      }

      size_t size1 = 1;
      if (orig_agg->getType()->isSized() &&
          (orig_agg->getType()->isIntOrIntVectorTy() ||
           orig_agg->getType()->isFPOrFPVectorTy()))
        size1 =
            (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                 orig_agg->getType()) +
             7) /
            8;

      if (!gutils->isConstantValue(orig_agg)) {
        auto rule = [&](Value *prediff) {
          return Builder2.CreateInsertValue(
              prediff, Constant::getNullValue(orig_inserted->getType()),
              IVI.getIndices());
        };
        auto prediff = diffe(&IVI, Builder2);
        auto dindex =
            applyChainRule(orig_agg->getType(), Builder2, rule, prediff);
        addToDiffe(orig_agg, dindex, Builder2, TR.addingType(size1, orig_agg));
      }

      setDiffe(&IVI,
               Constant::getNullValue(gutils->getShadowType(IVI.getType())),
               Builder2);
      return;
    }
    case DerivativeMode::ReverseModePrimal: {
      return;
    }
    }
  }

  void getReverseBuilder(IRBuilder<> &Builder2, bool original = true) {
    ((GradientUtils *)gutils)->getReverseBuilder(Builder2, original);
  }

  void getForwardBuilder(IRBuilder<> &Builder2) {
    ((GradientUtils *)gutils)->getForwardBuilder(Builder2);
  }

  Value *diffe(Value *val, IRBuilder<> &Builder) {
    assert(Mode != DerivativeMode::ReverseModePrimal);
    return ((DiffeGradientUtils *)gutils)->diffe(val, Builder);
  }

  void setDiffe(Value *val, Value *dif, IRBuilder<> &Builder) {
    assert(Mode != DerivativeMode::ReverseModePrimal);
    ((DiffeGradientUtils *)gutils)->setDiffe(val, dif, Builder);
  }

  /// Unwraps a vector derivative from its internal representation and applies a
  /// function f to each element. Return values of f are collected and wrapped.
  template <typename Func, typename... Args>
  Value *applyChainRule(Type *diffType, IRBuilder<> &Builder, Func rule,
                        Args... args) {
    return ((DiffeGradientUtils *)gutils)
        ->applyChainRule(diffType, Builder, rule, args...);
  }

  /// Unwraps a vector derivative from its internal representation and applies a
  /// function f to each element.
  template <typename Func, typename... Args>
  void applyChainRule(IRBuilder<> &Builder, Func rule, Args... args) {
    ((DiffeGradientUtils *)gutils)->applyChainRule(Builder, rule, args...);
  }

  /// Unwraps an collection of constant vector derivatives from their internal
  /// representations and applies a function f to each element.
  template <typename Func>
  void applyChainRule(ArrayRef<Value *> diffs, IRBuilder<> &Builder,
                      Func rule) {
    ((DiffeGradientUtils *)gutils)->applyChainRule(diffs, Builder, rule);
  }

  bool shouldFree() {
    assert(Mode == DerivativeMode::ReverseModeCombined ||
           Mode == DerivativeMode::ReverseModeGradient ||
           Mode == DerivativeMode::ForwardModeSplit);
    return ((DiffeGradientUtils *)gutils)->FreeMemory;
  }

  SmallVector<SelectInst *, 4> addToDiffe(Value *val, Value *dif,
                                          IRBuilder<> &Builder, Type *T,
                                          Value *mask = nullptr) {
    return ((DiffeGradientUtils *)gutils)
        ->addToDiffe(val, dif, Builder, T, /*idxs*/ {}, mask);
  }

  Value *lookup(Value *val, IRBuilder<> &Builder) {
    return gutils->lookupM(val, Builder);
  }

  void visitBinaryOperator(llvm::BinaryOperator &BO) {
    eraseIfUnused(BO);

    size_t size = 1;
    if (BO.getType()->isSized())
      size = (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                  BO.getType()) +
              7) /
             8;

    if (BO.getType()->isIntOrIntVectorTy() &&
        TR.intType(size, &BO, /*errifnotfound*/ false) == BaseType::Pointer) {
      return;
    }

    switch (Mode) {
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined:
      if (gutils->isConstantInstruction(&BO))
        return;
      createBinaryOperatorAdjoint(BO);
      break;
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeSplit:
      createBinaryOperatorDual(BO);
      break;
    case DerivativeMode::ReverseModePrimal:
      return;
    }
  }

  void createBinaryOperatorAdjoint(llvm::BinaryOperator &BO) {
    IRBuilder<> Builder2(BO.getParent());
    getReverseBuilder(Builder2);

    Value *orig_op0 = BO.getOperand(0);
    Value *orig_op1 = BO.getOperand(1);
    bool constantval0 = gutils->isConstantValue(orig_op0);
    bool constantval1 = gutils->isConstantValue(orig_op1);

    Value *dif0 = nullptr;
    Value *dif1 = nullptr;
    Value *idiff = diffe(&BO, Builder2);

    Type *addingType = BO.getType();

    switch (BO.getOpcode()) {
    case Instruction::FMul: {
      if (!constantval0) {
        Value *op0 = lookup(gutils->getNewFromOriginal(orig_op1), Builder2);
        auto rule = [&](Value *idiff) {
          return Builder2.CreateFMul(idiff, op0,
                                     "m0diffe" + orig_op0->getName());
        };
        dif0 = applyChainRule(orig_op0->getType(), Builder2, rule, idiff);
      }
      if (!constantval1) {
        auto rule = [&](Value *idiff) {
          return Builder2.CreateFMul(
              idiff, lookup(gutils->getNewFromOriginal(orig_op0), Builder2),
              "m1diffe" + orig_op1->getName());
        };
        dif1 = applyChainRule(orig_op1->getType(), Builder2, rule, idiff);
      }
      break;
    }
    case Instruction::FAdd: {
      if (!constantval0)
        dif0 = idiff;
      if (!constantval1)
        dif1 = idiff;
      break;
    }
    case Instruction::FSub: {
      if (!constantval0)
        dif0 = idiff;
      if (!constantval1) {
        auto rule = [&](Value *idiff) { return Builder2.CreateFNeg(idiff); };
        dif1 = applyChainRule(orig_op1->getType(), Builder2, rule, idiff);
      }
      break;
    }
    case Instruction::FDiv: {
      // Required loopy phi = [in, BO, BO, ..., BO]
      //  1) phi is only used in this B0
      //  2) BO dominates all latches
      //  3) phi == B0 whenever not coming from preheader [implies 2]
      //  4) [optional but done for ease] one exit to make it easier to
      //  calculation the product at that point
      if (auto P0 = dyn_cast<PHINode>(orig_op0)) {
        LoopContext lc;
        SmallVector<Instruction *, 4> activeUses;
        for (auto u : P0->users()) {
          if (!gutils->isConstantInstruction(cast<Instruction>(u))) {
            activeUses.push_back(cast<Instruction>(u));
          } else if (retType == DIFFE_TYPE::OUT_DIFF && isa<ReturnInst>(u))
            activeUses.push_back(cast<Instruction>(u));
        }
        if (activeUses.size() == 1 && activeUses[0] == &BO &&
            gutils->getContext(gutils->getNewFromOriginal(P0->getParent()),
                               lc) &&
            gutils->getNewFromOriginal(P0->getParent()) == lc.header) {
          SmallVector<BasicBlock *, 1> Latches;
          gutils->OrigLI.getLoopFor(P0->getParent())->getLoopLatches(Latches);
          bool allIncoming = true;
          for (auto Latch : Latches) {
            if (&BO != P0->getIncomingValueForBlock(Latch)) {
              allIncoming = false;
              break;
            }
          }
          if (allIncoming && lc.exitBlocks.size() == 1) {
            if (!constantval1) {
              IRBuilder<> EB(*lc.exitBlocks.begin());
              getReverseBuilder(EB, /*original=*/false);
              Value *Pstart = P0->getIncomingValueForBlock(
                  gutils->getOriginalFromNew(lc.preheader));
              if (gutils->isConstantValue(Pstart)) {
                Value *lop0 = lookup(gutils->getNewFromOriginal(&BO), EB);
                Value *lop1 =
                    lookup(gutils->getNewFromOriginal(orig_op1), Builder2);
                auto rule = [&](Value *idiff) {
                  return Builder2.CreateFDiv(
                      Builder2.CreateFNeg(Builder2.CreateFMul(idiff, lop0)),
                      lop1);
                };
                dif1 =
                    applyChainRule(orig_op1->getType(), Builder2, rule, idiff);
              } else {
                auto product = gutils->getOrInsertTotalMultiplicativeProduct(
                    gutils->getNewFromOriginal(orig_op1), lc);
                IRBuilder<> EB(*lc.exitBlocks.begin());
                getReverseBuilder(EB, /*original=*/false);
                Value *s = lookup(gutils->getNewFromOriginal(Pstart), Builder2);
                Value *lop0 = lookup(product, EB);
                Value *lop1 =
                    lookup(gutils->getNewFromOriginal(orig_op1), Builder2);
                auto rule = [&](Value *idiff) {
                  return Builder2.CreateFDiv(
                      Builder2.CreateFNeg(Builder2.CreateFMul(
                          s, Builder2.CreateFDiv(idiff, lop0))),
                      lop1);
                };
                dif1 =
                    applyChainRule(orig_op1->getType(), Builder2, rule, idiff);
              }
              addToDiffe(orig_op1, dif1, Builder2, addingType);
            }
            return;
          }
        }
      }
      if (!constantval0) {
        Value *op1 = lookup(gutils->getNewFromOriginal(orig_op1), Builder2);
        auto rule = [&](Value *idiff) {
          return Builder2.CreateFDiv(idiff, op1,
                                     "d0diffe" + orig_op0->getName());
        };
        dif0 = applyChainRule(orig_op0->getType(), Builder2, rule, idiff);
      }
      if (!constantval1) {
        Value *lop1 = lookup(gutils->getNewFromOriginal(orig_op1), Builder2);
        Value *lastdiv = lookup(gutils->getNewFromOriginal(&BO), Builder2);

        auto rule = [&](Value *idiff) {
          return Builder2.CreateFNeg(
              Builder2.CreateFMul(lastdiv, Builder2.CreateFDiv(idiff, lop1)));
        };
        dif1 = applyChainRule(orig_op1->getType(), Builder2, rule, idiff);
      }
      break;
    }
    case Instruction::LShr: {
      if (!constantval0) {
        if (auto ci = dyn_cast<ConstantInt>(orig_op1)) {
          size_t size = 1;
          if (orig_op0->getType()->isSized())
            size = (gutils->newFunc->getParent()
                        ->getDataLayout()
                        .getTypeSizeInBits(orig_op0->getType()) +
                    7) /
                   8;

          if (Type *flt = TR.addingType(size, orig_op0)) {
            auto bits = gutils->newFunc->getParent()
                            ->getDataLayout()
                            .getTypeAllocSizeInBits(flt);
            if (ci->getSExtValue() >= (int64_t)bits &&
                ci->getSExtValue() % bits == 0) {
              auto rule = [&](Value *idiff) {
                return Builder2.CreateShl(idiff, ci);
              };
              dif0 = applyChainRule(orig_op0->getType(), Builder2, rule, idiff);
              addingType = flt;
              goto done;
            }
          }
        }
      }
      goto def;
    }
    case Instruction::And: {
      // If & against 0b10000000000 and a float the result is 0
      auto &dl = gutils->oldFunc->getParent()->getDataLayout();
      auto size = dl.getTypeSizeInBits(BO.getType()) / 8;

      auto FT = TR.query(&BO).IsAllFloat(size);
      auto eFT = FT;
      if (FT)
        for (int i = 0; i < 2; ++i) {
          auto CI = dyn_cast<ConstantInt>(BO.getOperand(i));
          if (CI && dl.getTypeSizeInBits(eFT) ==
                        dl.getTypeSizeInBits(CI->getType())) {
            if (CI->isNegative() && CI->isMinValue(/*signed*/ true)) {
              setDiffe(
                  &BO,
                  Constant::getNullValue(gutils->getShadowType(BO.getType())),
                  Builder2);
              // Derivative is zero, no update
              return;
            }
            if (eFT->isDoubleTy() && CI->getValue() == -134217728) {
              setDiffe(
                  &BO,
                  Constant::getNullValue(gutils->getShadowType(BO.getType())),
                  Builder2);
              // Derivative is zero (equivalent to rounding as just chopping off
              // bits of mantissa), no update
              return;
            }
          }
        }
      goto def;
    }
    case Instruction::Xor: {
      auto &dl = gutils->oldFunc->getParent()->getDataLayout();
      auto size = dl.getTypeSizeInBits(BO.getType()) / 8;

      auto FT = TR.query(&BO).IsAllFloat(size);
      auto eFT = FT;
      // If ^ against 0b10000000000 and a float the result is a float
      if (FT)
        for (int i = 0; i < 2; ++i) {
          auto CI = dyn_cast<ConstantInt>(BO.getOperand(i));
          if (auto CV = dyn_cast<ConstantVector>(BO.getOperand(i))) {
            CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
#if LLVM_VERSION_MAJOR >= 12
            FT = VectorType::get(FT, CV->getType()->getElementCount());
#else
            FT = VectorType::get(FT, CV->getType()->getNumElements());
#endif
          }
          if (auto CV = dyn_cast<ConstantDataVector>(BO.getOperand(i))) {
            CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
#if LLVM_VERSION_MAJOR >= 12
            FT = VectorType::get(FT, CV->getType()->getElementCount());
#else
            FT = VectorType::get(FT, CV->getType()->getNumElements());
#endif
          }
          if (CI && dl.getTypeSizeInBits(eFT) ==
                        dl.getTypeSizeInBits(CI->getType())) {
            if (CI->isNegative() && CI->isMinValue(/*signed*/ true)) {
              setDiffe(
                  &BO,
                  Constant::getNullValue(gutils->getShadowType(BO.getType())),
                  Builder2);
              auto rule = [&](Value *idiff) {
                auto neg =
                    Builder2.CreateFNeg(Builder2.CreateBitCast(idiff, FT));
                return Builder2.CreateBitCast(neg, BO.getType());
              };
              auto bc = applyChainRule(BO.getOperand(1 - i)->getType(),
                                       Builder2, rule, idiff);
              addToDiffe(BO.getOperand(1 - i), bc, Builder2, FT);
              return;
            }
          }

          if (auto CV = dyn_cast<ConstantVector>(BO.getOperand(i))) {
            bool validXor = true;
            if (dl.getTypeSizeInBits(eFT) !=
                dl.getTypeSizeInBits(CV->getOperand(0)->getType()))
              validXor = false;
            for (size_t i = 0, end = CV->getNumOperands(); i < end; ++i) {
              auto CI = dyn_cast<ConstantInt>(CV->getOperand(i))->getValue();
              if (!(CI.isNullValue() || CI.isMinSignedValue())) {
                validXor = false;
              }
            }
            if (validXor) {
              setDiffe(
                  &BO,
                  Constant::getNullValue(gutils->getShadowType(BO.getType())),
                  Builder2);
              auto rule = [&](Value *idiff) {
                Value *V = UndefValue::get(CV->getType());
                for (size_t i = 0, end = CV->getNumOperands(); i < end; ++i) {
                  auto CI =
                      dyn_cast<ConstantInt>(CV->getOperand(i))->getValue();
                  if (CI.isNullValue())
                    V = Builder2.CreateInsertElement(
                        V, Builder2.CreateExtractElement(idiff, i), i);
                  if (CI.isMinSignedValue())
                    V = Builder2.CreateInsertElement(
                        V,
                        Builder2.CreateBitCast(
                            Builder2.CreateFNeg(Builder2.CreateBitCast(
                                Builder2.CreateExtractElement(idiff, i), eFT)),
                            CV->getOperand(i)->getType()),
                        i);
                }
                return V;
              };
              Value *V = applyChainRule(BO.getOperand(1 - i)->getType(),
                                        Builder2, rule, idiff);
              addToDiffe(BO.getOperand(1 - i), V, Builder2, FT);
              return;
            }
          } else if (auto CV = dyn_cast<ConstantDataVector>(BO.getOperand(i))) {
            bool validXor = true;
            if (dl.getTypeSizeInBits(eFT) !=
                dl.getTypeSizeInBits(CV->getElementType()))
              validXor = false;
            for (size_t i = 0, end = CV->getNumElements(); i < end; ++i) {
              auto CI = CV->getElementAsAPInt(i);
              if (!(CI.isNullValue() || CI.isMinSignedValue())) {
                validXor = false;
              }
            }
            if (validXor) {
              setDiffe(
                  &BO,
                  Constant::getNullValue(gutils->getShadowType(BO.getType())),
                  Builder2);

              auto rule = [&](Value *idiff) {
                Value *V = UndefValue::get(CV->getType());
                for (size_t i = 0, end = CV->getNumElements(); i < end; ++i) {
                  auto CI = CV->getElementAsAPInt(i);
                  if (CI.isNullValue())
                    V = Builder2.CreateInsertElement(
                        V, Builder2.CreateExtractElement(idiff, i), i);
                  if (CI.isMinSignedValue())
                    V = Builder2.CreateInsertElement(
                        V,
                        Builder2.CreateBitCast(
                            Builder2.CreateFNeg(Builder2.CreateBitCast(
                                Builder2.CreateExtractElement(idiff, i), eFT)),
                            CV->getElementType()),
                        i);
                }
                return V;
              };
              Value *V = applyChainRule(BO.getOperand(1 - i)->getType(),
                                        Builder2, rule, idiff);
              addToDiffe(BO.getOperand(1 - i), V, Builder2, FT);
              return;
            }
          }
        }
      goto def;
    }
    case Instruction::Or: {
      auto &dl = gutils->oldFunc->getParent()->getDataLayout();
      auto size = dl.getTypeSizeInBits(BO.getType()) / 8;

      auto FT = TR.query(&BO).IsAllFloat(size);
      auto eFT = FT;
      // If & against 0b10000000000 and a float the result is a float
      if (FT)
        for (int i = 0; i < 2; ++i) {
          auto CI = dyn_cast<ConstantInt>(BO.getOperand(i));
          if (auto CV = dyn_cast<ConstantVector>(BO.getOperand(i))) {
            CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
#if LLVM_VERSION_MAJOR >= 12
            FT = VectorType::get(FT, CV->getType()->getElementCount());
#else
            FT = VectorType::get(FT, CV->getType()->getNumElements());
#endif
          }
          if (auto CV = dyn_cast<ConstantDataVector>(BO.getOperand(i))) {
            CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
#if LLVM_VERSION_MAJOR >= 12
            FT = VectorType::get(FT, CV->getType()->getElementCount());
#else
            FT = VectorType::get(FT, CV->getType()->getNumElements());
#endif
          }
          if (CI && dl.getTypeSizeInBits(eFT) ==
                        dl.getTypeSizeInBits(CI->getType())) {
            auto AP = CI->getValue();
            bool validXor = false;
            if (AP.isNullValue()) {
              validXor = true;
            } else if (
                !AP.isNegative() &&
                ((FT->isFloatTy() &&
                  (AP & ~0b01111111100000000000000000000000ULL)
                      .isNullValue()) ||
                 (FT->isDoubleTy() &&
                  (AP &
                   ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                      .isNullValue()))) {
              validXor = true;
            }
            if (validXor) {
              setDiffe(
                  &BO,
                  Constant::getNullValue(gutils->getShadowType(BO.getType())),
                  Builder2);

              auto arg = lookup(
                  gutils->getNewFromOriginal(BO.getOperand(1 - i)), Builder2);

              auto rule = [&](Value *idiff) {
                auto prev = Builder2.CreateOr(arg, BO.getOperand(i));
                prev = Builder2.CreateSub(prev, arg, "", /*NUW*/ true,
                                          /*NSW*/ false);
                uint64_t num = 0;
                if (FT->isFloatTy()) {
                  num = 127ULL << 23;
                } else {
                  assert(FT->isDoubleTy());
                  num = 1023ULL << 52;
                }
                prev = Builder2.CreateAdd(
                    prev, ConstantInt::get(prev->getType(), num, false), "",
                    /*NUW*/ true, /*NSW*/ true);
                prev = Builder2.CreateBitCast(
                    Builder2.CreateFMul(Builder2.CreateBitCast(idiff, FT),
                                        Builder2.CreateBitCast(prev, FT)),
                    prev->getType());
                return prev;
              };

              Value *prev = applyChainRule(BO.getOperand(1 - i)->getType(),
                                           Builder2, rule, idiff);
              addToDiffe(BO.getOperand(1 - i), prev, Builder2, FT);
              return;
            }
          }
        }
      goto def;
    }
    case Instruction::Shl:
    case Instruction::Mul:
    case Instruction::Sub:
    case Instruction::Add: {
      if (looseTypeAnalysis) {
        llvm::errs() << "warning: binary operator is integer and constant: "
                     << BO << "\n";
        // if loose type analysis, assume this integer add is constant
        return;
      }
      goto def;
    }
    default:
    def:;
      std::string s;
      llvm::raw_string_ostream ss(s);
      ss << *gutils->oldFunc->getParent() << "\n";
      ss << *gutils->oldFunc << "\n";
      for (auto &arg : gutils->oldFunc->args()) {
        ss << " constantarg[" << arg << "] = " << gutils->isConstantValue(&arg)
           << " type: " << TR.query(&arg).str() << " - vals: {";
        for (auto v : TR.knownIntegralValues(&arg))
          ss << v << ",";
        ss << "}\n";
      }
      for (auto &BB : *gutils->oldFunc)
        for (auto &I : BB) {
          ss << " constantinst[" << I
             << "] = " << gutils->isConstantInstruction(&I)
             << " val:" << gutils->isConstantValue(&I)
             << " type: " << TR.query(&I).str() << "\n";
        }
      ss << "cannot handle unknown binary operator: " << BO << "\n";
      if (CustomErrorHandler) {
        CustomErrorHandler(ss.str().c_str(), wrap(&BO), ErrorType::NoDerivative,
                           nullptr);
      }
      llvm::errs() << ss.str() << "\n";
      report_fatal_error("unknown binary operator");
    }

  done:;
    if (dif0 || dif1)
      setDiffe(&BO, Constant::getNullValue(gutils->getShadowType(BO.getType())),
               Builder2);
    if (dif0)
      addToDiffe(orig_op0, dif0, Builder2, addingType);
    if (dif1)
      addToDiffe(orig_op1, dif1, Builder2, addingType);
  }

  void createBinaryOperatorDual(llvm::BinaryOperator &BO) {
    if (gutils->isConstantInstruction(&BO)) {
      forwardModeInvertedPointerFallback(BO);
      return;
    }

    IRBuilder<> Builder2(&BO);
    getForwardBuilder(Builder2);

    Value *orig_op0 = BO.getOperand(0);
    Value *orig_op1 = BO.getOperand(1);

    bool constantval0 = gutils->isConstantValue(orig_op0);
    bool constantval1 = gutils->isConstantValue(orig_op1);

    Value *dif[2] = {constantval0 ? nullptr : diffe(orig_op0, Builder2),
                     constantval1 ? nullptr : diffe(orig_op1, Builder2)};

    switch (BO.getOpcode()) {
    case Instruction::FMul: {
      if (!constantval0 && !constantval1) {
        auto rule = [&](Value *dif0, Value *dif1) {
          Value *idiff0 =
              Builder2.CreateFMul(dif0, gutils->getNewFromOriginal(orig_op1));
          Value *idiff1 =
              Builder2.CreateFMul(dif1, gutils->getNewFromOriginal(orig_op0));
          return Builder2.CreateFAdd(idiff0, idiff1);
        };
        Value *diff =
            applyChainRule(BO.getType(), Builder2, rule, dif[0], dif[1]);
        setDiffe(&BO, diff, Builder2);
      } else if (!constantval0) {
        auto rule = [&](Value *dif0) {
          return Builder2.CreateFMul(dif0,
                                     gutils->getNewFromOriginal(orig_op1));
        };
        Value *idiff0 = applyChainRule(BO.getType(), Builder2, rule, dif[0]);
        setDiffe(&BO, idiff0, Builder2);
      } else if (!constantval1) {
        auto rule = [&](Value *dif1) {
          return Builder2.CreateFMul(dif1,
                                     gutils->getNewFromOriginal(orig_op0));
        };
        Value *idiff1 = applyChainRule(BO.getType(), Builder2, rule, dif[1]);
        setDiffe(&BO, idiff1, Builder2);
      }
      break;
    }
    case Instruction::FAdd: {
      if (!constantval0 && !constantval1) {
        auto rule = [&](Value *dif0, Value *dif1) {
          return Builder2.CreateFAdd(dif0, dif1);
        };
        Value *diff =
            applyChainRule(BO.getType(), Builder2, rule, dif[0], dif[1]);
        setDiffe(&BO, diff, Builder2);
      } else if (!constantval0) {
        setDiffe(&BO, dif[0], Builder2);
      } else if (!constantval1) {
        setDiffe(&BO, dif[1], Builder2);
      }
      break;
    }
    case Instruction::FSub: {
      if (!constantval0 && !constantval1) {
        auto rule = [&](Value *dif0, Value *dif1) {
          return Builder2.CreateFAdd(dif0, Builder2.CreateFNeg(dif1));
        };
        Value *diff =
            applyChainRule(BO.getType(), Builder2, rule, dif[0], dif[1]);
        setDiffe(&BO, diff, Builder2);
      } else if (!constantval0) {
        setDiffe(&BO, dif[0], Builder2);
      } else if (!constantval1) {
        auto rule = [&](Value *dif1) { return Builder2.CreateFNeg(dif1); };
        Value *diff = applyChainRule(BO.getType(), Builder2, rule, dif[1]);
        setDiffe(&BO, diff, Builder2);
      }
      break;
    }
    case Instruction::FDiv: {
      Value *idiff3 = nullptr;
      if (!constantval0 && !constantval1) {
        auto rule = [&](Value *dif0, Value *dif1) {
          Value *idiff1 =
              Builder2.CreateFMul(dif0, gutils->getNewFromOriginal(orig_op1));
          Value *idiff2 =
              Builder2.CreateFMul(gutils->getNewFromOriginal(orig_op0), dif1);
          return Builder2.CreateFSub(idiff1, idiff2);
        };
        idiff3 = applyChainRule(BO.getType(), Builder2, rule, dif[0], dif[1]);
      } else if (!constantval0) {
        auto rule = [&](Value *dif0) {
          return Builder2.CreateFMul(dif0,
                                     gutils->getNewFromOriginal(orig_op1));
        };
        idiff3 = applyChainRule(BO.getType(), Builder2, rule, dif[0]);
      } else if (!constantval1) {
        auto rule = [&](Value *dif1) {
          return Builder2.CreateFNeg(
              Builder2.CreateFMul(gutils->getNewFromOriginal(orig_op0), dif1));
        };
        idiff3 = applyChainRule(BO.getType(), Builder2, rule, dif[1]);
      }

      Value *idiff4 = Builder2.CreateFMul(gutils->getNewFromOriginal(orig_op1),
                                          gutils->getNewFromOriginal(orig_op1));

      auto rule = [&](Value *idiff3) {
        return Builder2.CreateFDiv(idiff3, idiff4);
      };

      Value *idiff5 = applyChainRule(BO.getType(), Builder2, rule, idiff3);
      setDiffe(&BO, idiff5, Builder2);

      break;
    }
    case Instruction::And: {
      // If & against 0b10000000000 and a float the result is 0
      auto &dl = gutils->oldFunc->getParent()->getDataLayout();
      auto size = dl.getTypeSizeInBits(BO.getType()) / 8;
      Type *diffTy = gutils->getShadowType(BO.getType());

      auto FT = TR.query(&BO).IsAllFloat(size);
      auto eFT = FT;
      if (FT)
        for (int i = 0; i < 2; ++i) {
          auto CI = dyn_cast<ConstantInt>(BO.getOperand(i));
          if (CI && dl.getTypeSizeInBits(eFT) ==
                        dl.getTypeSizeInBits(CI->getType())) {
            if (CI->isNegative() && CI->isMinValue(/*signed*/ true)) {
              setDiffe(&BO, Constant::getNullValue(diffTy), Builder2);
              // Derivative is zero, no update
              return;
            }
            if (eFT->isDoubleTy() && CI->getValue() == -134217728) {
              setDiffe(&BO, Constant::getNullValue(diffTy), Builder2);
              // Derivative is zero (equivalent to rounding as just chopping off
              // bits of mantissa), no update
              return;
            }
          }
        }
      goto def;
    }
    case Instruction::Xor: {
      auto &dl = gutils->oldFunc->getParent()->getDataLayout();
      auto size = dl.getTypeSizeInBits(BO.getType()) / 8;

      Value *dif[2] = {constantval0 ? nullptr : diffe(orig_op0, Builder2),
                       constantval1 ? nullptr : diffe(orig_op1, Builder2)};

      auto FT = TR.query(&BO).IsAllFloat(size);
      auto eFT = FT;
      // If ^ against 0b10000000000 and a float the result is a float
      if (FT)
        for (int i = 0; i < 2; ++i) {
          auto CI = dyn_cast<ConstantInt>(BO.getOperand(i));
          if (auto CV = dyn_cast<ConstantVector>(BO.getOperand(i))) {
            CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
#if LLVM_VERSION_MAJOR >= 12
            FT = VectorType::get(FT, CV->getType()->getElementCount());
#else
            FT = VectorType::get(FT, CV->getType()->getNumElements());
#endif
          }
          if (auto CV = dyn_cast<ConstantDataVector>(BO.getOperand(i))) {
            CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
#if LLVM_VERSION_MAJOR >= 12
            FT = VectorType::get(FT, CV->getType()->getElementCount());
#else
            FT = VectorType::get(FT, CV->getType()->getNumElements());
#endif
          }
          if (CI && dl.getTypeSizeInBits(eFT) ==
                        dl.getTypeSizeInBits(CI->getType())) {
            if (CI->isNegative() && CI->isMinValue(/*signed*/ true)) {
              assert(dif[1 - i]);
              auto rule = [&](Value *difi) {
                auto neg =
                    Builder2.CreateFNeg(Builder2.CreateBitCast(difi, FT));
                return Builder2.CreateBitCast(neg, BO.getType());
              };

              auto diffe =
                  applyChainRule(BO.getType(), Builder2, rule, dif[1 - i]);
              setDiffe(&BO, diffe, Builder2);
              return;
            }
          }

          if (auto CV = dyn_cast<ConstantVector>(BO.getOperand(i))) {
            bool validXor = true;
            if (dl.getTypeSizeInBits(eFT) !=
                dl.getTypeSizeInBits(CV->getOperand(0)->getType()))
              validXor = false;
            for (size_t i = 0, end = CV->getNumOperands(); i < end; ++i) {
              auto CI = dyn_cast<ConstantInt>(CV->getOperand(i))->getValue();
              if (!(CI.isNullValue() || CI.isMinSignedValue())) {
                validXor = false;
              }
            }
            if (validXor) {
              auto rule = [&](Value *difi) {
                Value *V = UndefValue::get(CV->getType());
                for (size_t j = 0, end = CV->getNumOperands(); j < end; ++j) {
                  auto CI =
                      dyn_cast<ConstantInt>(CV->getOperand(j))->getValue();
                  if (CI.isNullValue())
                    V = Builder2.CreateInsertElement(
                        V, Builder2.CreateExtractElement(difi, j), j);
                  if (CI.isMinSignedValue())
                    V = Builder2.CreateInsertElement(
                        V,
                        Builder2.CreateBitCast(
                            Builder2.CreateFNeg(Builder2.CreateBitCast(
                                Builder2.CreateExtractElement(difi, j), eFT)),
                            CV->getOperand(j)->getType()),
                        j);
                }
                return V;
              };

              auto diffe =
                  applyChainRule(BO.getType(), Builder2, rule, dif[1 - i]);
              setDiffe(&BO, diffe, Builder2);
              return;
            }
          } else if (auto CV = dyn_cast<ConstantDataVector>(BO.getOperand(i))) {
            bool validXor = true;
            if (dl.getTypeSizeInBits(eFT) !=
                dl.getTypeSizeInBits(CV->getElementType()))
              validXor = false;
            for (size_t i = 0, end = CV->getNumElements(); i < end; ++i) {
              auto CI = CV->getElementAsAPInt(i);
              if (!(CI.isNullValue() || CI.isMinSignedValue())) {
                validXor = false;
              }
            }
            if (validXor) {
              auto rule = [&](Value *difi) {
                Value *V = UndefValue::get(CV->getType());
                for (size_t j = 0, end = CV->getNumElements(); j < end; ++j) {
                  auto CI = CV->getElementAsAPInt(j);
                  if (CI.isNullValue())
                    V = Builder2.CreateInsertElement(
                        V, Builder2.CreateExtractElement(difi, j), j);
                  if (CI.isMinSignedValue())
                    V = Builder2.CreateInsertElement(
                        V,
                        Builder2.CreateBitCast(
                            Builder2.CreateFNeg(Builder2.CreateBitCast(
                                Builder2.CreateExtractElement(difi, j), eFT)),
                            CV->getElementType()),
                        j);
                }
                return V;
              };

              auto diffe =
                  applyChainRule(BO.getType(), Builder2, rule, dif[1 - i]);
              setDiffe(&BO, diffe, Builder2);
              return;
            }
          }
        }
      goto def;
    }
    case Instruction::Or: {
      auto &dl = gutils->oldFunc->getParent()->getDataLayout();
      auto size = dl.getTypeSizeInBits(BO.getType()) / 8;

      Value *dif[2] = {constantval0 ? nullptr : diffe(orig_op0, Builder2),
                       constantval1 ? nullptr : diffe(orig_op1, Builder2)};

      auto FT = TR.query(&BO).IsAllFloat(size);
      auto eFT = FT;
      // If & against 0b10000000000 and a float the result is a float
      if (FT)
        for (int i = 0; i < 2; ++i) {
          auto CI = dyn_cast<ConstantInt>(BO.getOperand(i));
          if (auto CV = dyn_cast<ConstantVector>(BO.getOperand(i))) {
            CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
#if LLVM_VERSION_MAJOR >= 12
            FT = VectorType::get(FT, CV->getType()->getElementCount());
#else
            FT = VectorType::get(FT, CV->getType()->getNumElements());
#endif
          }
          if (auto CV = dyn_cast<ConstantDataVector>(BO.getOperand(i))) {
            CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
#if LLVM_VERSION_MAJOR >= 12
            FT = VectorType::get(FT, CV->getType()->getElementCount());
#else
            FT = VectorType::get(FT, CV->getType()->getNumElements());
#endif
          }
          if (CI && dl.getTypeSizeInBits(eFT) ==
                        dl.getTypeSizeInBits(CI->getType())) {
            auto AP = CI->getValue();
            bool validXor = false;
            if (AP.isNullValue()) {
              validXor = true;
            } else if (
                !AP.isNegative() &&
                ((FT->isFloatTy() &&
                  (AP & ~0b01111111100000000000000000000000ULL)
                      .isNullValue()) ||
                 (FT->isDoubleTy() &&
                  (AP &
                   ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                      .isNullValue()))) {
              validXor = true;
            }
            if (validXor) {
              auto rule = [&](Value *difi) {
                auto arg = gutils->getNewFromOriginal(BO.getOperand(1 - i));
                auto prev = Builder2.CreateOr(arg, BO.getOperand(i));
                prev = Builder2.CreateSub(prev, arg, "", /*NUW*/ true,
                                          /*NSW*/ false);
                uint64_t num = 0;
                if (FT->isFloatTy()) {
                  num = 127ULL << 23;
                } else {
                  assert(FT->isDoubleTy());
                  num = 1023ULL << 52;
                }
                prev = Builder2.CreateAdd(
                    prev, ConstantInt::get(prev->getType(), num, false), "",
                    /*NUW*/ true, /*NSW*/ true);
                prev = Builder2.CreateBitCast(
                    Builder2.CreateFMul(Builder2.CreateBitCast(difi, FT),
                                        Builder2.CreateBitCast(prev, FT)),
                    prev->getType());

                return prev;
              };

              auto diffe =
                  applyChainRule(BO.getType(), Builder2, rule, dif[1 - i]);
              setDiffe(&BO, diffe, Builder2);
              return;
            }
          }
        }
      goto def;
    }
    case Instruction::Shl:
    case Instruction::Mul:
    case Instruction::Sub:
    case Instruction::Add: {
      if (looseTypeAnalysis) {
        forwardModeInvertedPointerFallback(BO);
        llvm::errs() << "warning: binary operator is integer and constant: "
                     << BO << "\n";
        // if loose type analysis, assume this integer add is constant
        return;
      }
      goto def;
    }
    default:
    def:;
      std::string s;
      llvm::raw_string_ostream ss(s);
      ss << *gutils->oldFunc->getParent() << "\n";
      ss << *gutils->oldFunc << "\n";
      for (auto &arg : gutils->oldFunc->args()) {
        ss << " constantarg[" << arg << "] = " << gutils->isConstantValue(&arg)
           << " type: " << TR.query(&arg).str() << " - vals: {";
        for (auto v : TR.knownIntegralValues(&arg))
          ss << v << ",";
        ss << "}\n";
      }
      for (auto &BB : *gutils->oldFunc)
        for (auto &I : BB) {
          ss << " constantinst[" << I
             << "] = " << gutils->isConstantInstruction(&I)
             << " val:" << gutils->isConstantValue(&I)
             << " type: " << TR.query(&I).str() << "\n";
        }
      ss << "cannot handle unknown binary operator: " << BO << "\n";
      if (CustomErrorHandler) {
        CustomErrorHandler(ss.str().c_str(), wrap(&BO), ErrorType::NoDerivative,
                           nullptr);
      }
      llvm::errs() << ss.str() << "\n";
      report_fatal_error("unknown binary operator");
      break;
    }
  }

  void visitMemSetInst(llvm::MemSetInst &MS) { visitMemSetCommon(MS); }

  void visitMemSetCommon(llvm::CallInst &MS) {
    eraseIfUnused(MS);

    Value *orig_op0 = MS.getArgOperand(0);
    Value *orig_op1 = MS.getArgOperand(1);

    // TODO this should 1) assert that the value being meset is constant
    //                 2) duplicate the memset for the inverted pointer

    if (gutils->isConstantInstruction(&MS) &&
        Mode != DerivativeMode::ForwardMode) {
      return;
    }

    // If constant destination then no operation needs doing
    if (gutils->isConstantValue(orig_op0)) {
      return;
    }

    if (!gutils->isConstantValue(orig_op1)) {
      std::string s;
      llvm::raw_string_ostream ss(s);
      ss << "couldn't handle non constant inst in memset to "
            "propagate differential to\n"
         << MS;
      if (CustomErrorHandler) {
        CustomErrorHandler(ss.str().c_str(), wrap(&MS), ErrorType::NoDerivative,
                           nullptr);
      }
      llvm::errs() << ss.str() << "\n";
      report_fatal_error("non constant in memset");
    }

    bool backwardsShadow = false;
    bool forwardsShadow = true;
    for (auto pair : gutils->backwardsOnlyShadows) {
      if (pair.second.stores.count(&MS)) {
        backwardsShadow = true;
        forwardsShadow = pair.second.primalInitialize;
        if (auto inst = dyn_cast<Instruction>(pair.first))
          if (!forwardsShadow && pair.second.LI &&
              pair.second.LI->contains(inst->getParent()))
            backwardsShadow = false;
      }
    }

    if ((Mode == DerivativeMode::ReverseModePrimal && forwardsShadow) ||
        (Mode == DerivativeMode::ReverseModeGradient && backwardsShadow) ||
        (Mode == DerivativeMode::ReverseModeCombined &&
         (forwardsShadow && backwardsShadow)) ||
        Mode == DerivativeMode::ForwardMode) {
      IRBuilder<> BuilderZ(&MS);
      getForwardBuilder(BuilderZ);

      bool forwardMode = Mode == DerivativeMode::ForwardMode;

      Value *op0 = gutils->invertPointerM(orig_op0, BuilderZ);
      Value *op1 = gutils->getNewFromOriginal(MS.getArgOperand(1));
      if (!forwardMode)
        op1 = gutils->lookupM(op1, BuilderZ);
      Value *op2 = gutils->getNewFromOriginal(MS.getArgOperand(2));
      if (!forwardMode)
        op2 = gutils->lookupM(op2, BuilderZ);
      Value *op3 = nullptr;
#if LLVM_VERSION_MAJOR >= 14
      if (3 < MS.arg_size())
#else
      if (3 < MS.getNumArgOperands())
#endif
      {
        op3 = gutils->getNewFromOriginal(MS.getOperand(3));
        if (!forwardMode)
          op3 = gutils->lookupM(op3, BuilderZ);
      }

      auto Defs =
          gutils->getInvertedBundles(&MS,
                                     {ValueType::Shadow, ValueType::Primal,
                                      ValueType::Primal, ValueType::Primal},
                                     BuilderZ, /*lookup*/ false);

      applyChainRule(
          BuilderZ,
          [&](Value *op0) {
            SmallVector<Value *, 4> args = {op0, op1, op2};
            if (op3)
              args.push_back(op3);
            auto cal = BuilderZ.CreateCall(MS.getCalledFunction(), args, Defs);
            cal->copyMetadata(MS, MD_ToCopy);
            cal->setAttributes(MS.getAttributes());
            cal->setCallingConv(MS.getCallingConv());
            cal->setTailCallKind(MS.getTailCallKind());
            cal->setDebugLoc(gutils->getNewFromOriginal(MS.getDebugLoc()));
          },
          op0);
    }
  }

  void visitMemTransferInst(llvm::MemTransferInst &MTI) {
#if LLVM_VERSION_MAJOR >= 7
    Value *isVolatile = gutils->getNewFromOriginal(MTI.getOperand(3));
#else
    Value *isVolatile = gutils->getNewFromOriginal(MTI.getOperand(4));
#endif
#if LLVM_VERSION_MAJOR >= 10
    auto srcAlign = MTI.getSourceAlign();
    auto dstAlign = MTI.getDestAlign();
#else
    auto srcAlign = MTI.getSourceAlignment();
    auto dstAlign = MTI.getDestAlignment();
#endif
    visitMemTransferCommon(MTI.getIntrinsicID(), srcAlign, dstAlign, MTI,
                           MTI.getOperand(0), MTI.getOperand(1),
                           gutils->getNewFromOriginal(MTI.getOperand(2)),
                           isVolatile);
  }

#if LLVM_VERSION_MAJOR >= 10
  void visitMemTransferCommon(Intrinsic::ID ID, MaybeAlign srcAlign,
                              MaybeAlign dstAlign, llvm::CallInst &MTI,
                              Value *orig_dst, Value *orig_src, Value *new_size,
                              Value *isVolatile)
#else
  void visitMemTransferCommon(Intrinsic::ID ID, unsigned srcAlign,
                              unsigned dstAlign, llvm::CallInst &MTI,
                              Value *orig_dst, Value *orig_src, Value *new_size,
                              Value *isVolatile)
#endif
  {
    if (gutils->isConstantValue(MTI.getOperand(0))) {
      eraseIfUnused(MTI);
      return;
    }

    if (unnecessaryStores.count(&MTI)) {
      eraseIfUnused(MTI);
      return;
    }

    // copying into nullptr is invalid (not sure why it exists here), but we
    // shouldn't do it in reverse pass or shadow
    if (isa<ConstantPointerNull>(orig_dst) ||
        TR.query(orig_dst).Inner0() == BaseType::Anything) {
      eraseIfUnused(MTI);
      return;
    }

    if (Mode == DerivativeMode::ForwardMode) {
      IRBuilder<> Builder2(&MTI);
      getForwardBuilder(Builder2);
      auto ddst = gutils->invertPointerM(orig_dst, Builder2);
      auto dsrc = gutils->invertPointerM(orig_src, Builder2);

      auto rule = [&](Value *ddst, Value *dsrc) {
        if (ddst->getType()->isIntegerTy())
          ddst = Builder2.CreateIntToPtr(
              ddst, Type::getInt8PtrTy(ddst->getContext()));
        if (dsrc->getType()->isIntegerTy())
          dsrc = Builder2.CreateIntToPtr(
              dsrc, Type::getInt8PtrTy(dsrc->getContext()));
        CallInst *call;
        if (ID == Intrinsic::memmove) {
          call =
              Builder2.CreateMemMove(ddst, dstAlign, dsrc, srcAlign, new_size);
        } else {
          call =
              Builder2.CreateMemCpy(ddst, dstAlign, dsrc, srcAlign, new_size);
        }
        call->setAttributes(MTI.getAttributes());
        call->setMetadata(LLVMContext::MD_tbaa,
                          MTI.getMetadata(LLVMContext::MD_tbaa));
        call->setMetadata(LLVMContext::MD_tbaa_struct,
                          MTI.getMetadata(LLVMContext::MD_tbaa_struct));
        call->setMetadata(LLVMContext::MD_invariant_group,
                          MTI.getMetadata(LLVMContext::MD_invariant_group));
        call->setTailCallKind(MTI.getTailCallKind());
      };

      applyChainRule(Builder2, rule, ddst, dsrc);
      eraseIfUnused(MTI);
      return;
    }

    size_t size = 1;
    if (auto ci = dyn_cast<ConstantInt>(new_size)) {
      size = ci->getLimitedValue();
    }

    // TODO note that we only handle memcpy/etc of ONE type (aka memcpy of {int,
    // double} not allowed)

    // llvm::errs() << *gutils->oldFunc << "\n";
    // TR.dump();
    if (size == 0) {
      llvm::errs() << MTI << "\n";
    }
    assert(size != 0);

    auto &DL = gutils->newFunc->getParent()->getDataLayout();
    auto vd = TR.query(orig_dst).Data0().ShiftIndices(DL, 0, size, 0);
    vd |= TR.query(orig_src).Data0().ShiftIndices(DL, 0, size, 0);

    // llvm::errs() << "MIT: " << MTI << "|size: " << size << " vd: " <<
    // vd.str() << "\n";

    if (!vd.isKnownPastPointer()) {
      if (looseTypeAnalysis) {
        for (auto val : {orig_dst, orig_src}) {
          if (auto CI = dyn_cast<CastInst>(val)) {
            if (auto PT = dyn_cast<PointerType>(CI->getSrcTy())) {
              auto ET = PT->getPointerElementType();
              while (1) {
                if (auto ST = dyn_cast<StructType>(ET)) {
                  if (ST->getNumElements()) {
                    ET = ST->getElementType(0);
                    continue;
                  }
                }
                if (auto AT = dyn_cast<ArrayType>(ET)) {
                  ET = AT->getElementType();
                  continue;
                }
                break;
              }
              if (ET->isFPOrFPVectorTy()) {
                vd = TypeTree(ConcreteType(ET->getScalarType())).Only(0);
                goto known;
              }
              if (ET->isPointerTy()) {
                vd = TypeTree(BaseType::Pointer).Only(0);
                goto known;
              }
              if (ET->isIntOrIntVectorTy()) {
                vd = TypeTree(BaseType::Integer).Only(0);
                goto known;
              }
            }
          }
          if (auto gep = dyn_cast<GetElementPtrInst>(val)) {
            if (auto AT = dyn_cast<ArrayType>(gep->getSourceElementType())) {
              if (AT->getElementType()->isIntegerTy()) {
                vd = TypeTree(BaseType::Integer).Only(0);
                goto known;
              }
            }
          }
        }
        EmitWarning("CannotDeduceType", MTI.getDebugLoc(), gutils->oldFunc,
                    MTI.getParent(), &MTI, "failed to deduce type of copy ",
                    MTI);
        vd = TypeTree(BaseType::Pointer).Only(0);
        goto known;
      }
      if (CustomErrorHandler) {
        std::string str;
        raw_string_ostream ss(str);
        ss << "Cannot deduce type of copy " << MTI;
        CustomErrorHandler(str.c_str(), wrap(&MTI), ErrorType::NoType,
                           &TR.analyzer);
      }
      EmitFailure("CannotDeduceType", MTI.getDebugLoc(), &MTI,
                  "failed to deduce type of copy ", MTI);

      TR.firstPointer(size, orig_dst, /*errifnotfound*/ true,
                      /*pointerIntSame*/ true);
      llvm_unreachable("bad mti");
    }
  known:;

#if LLVM_VERSION_MAJOR >= 10
    unsigned dstalign = dstAlign.valueOrOne().value();
    unsigned srcalign = srcAlign.valueOrOne().value();
#else
    unsigned dstalign = dstAlign;
    unsigned srcalign = srcAlign;
#endif

    unsigned start = 0;

    IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&MTI));

    bool backwardsShadow = false;
    bool forwardsShadow = true;
    for (auto pair : gutils->backwardsOnlyShadows) {
      if (pair.second.stores.count(&MTI)) {
        backwardsShadow = true;
        forwardsShadow = pair.second.primalInitialize;
        if (auto inst = dyn_cast<Instruction>(pair.first))
          if (!forwardsShadow && pair.second.LI &&
              pair.second.LI->contains(inst->getParent()))
            backwardsShadow = false;
      }
    }

    while (1) {
      unsigned nextStart = size;

      auto dt = vd[{-1}];
      for (size_t i = start; i < size; ++i) {
        bool Legal = true;
        dt.checkedOrIn(vd[{(int)i}], /*PointerIntSame*/ true, Legal);
        if (!Legal) {
          nextStart = i;
          break;
        }
      }
      if (!dt.isKnown()) {
        TR.dump();
        llvm::errs() << " vd:" << vd.str() << " start:" << start
                     << " size: " << size << " dt:" << dt.str() << "\n";
      }
      assert(dt.isKnown());

      Value *length = new_size;
      if (nextStart != size) {
        length = ConstantInt::get(new_size->getType(), nextStart);
      }
      if (start != 0)
        length = BuilderZ.CreateSub(
            length, ConstantInt::get(new_size->getType(), start));

      unsigned subdstalign = dstalign;
      // todo make better alignment calculation
      if (dstalign != 0) {
        if (start % dstalign != 0) {
          dstalign = 1;
        }
      }
      unsigned subsrcalign = srcalign;
      // todo make better alignment calculation
      if (srcalign != 0) {
        if (start % srcalign != 0) {
          srcalign = 1;
        }
      }
      IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&MTI));
      Value *shadow_dst = gutils->isConstantValue(orig_dst)
                              ? gutils->getNewFromOriginal(orig_dst)
                              : gutils->invertPointerM(orig_dst, BuilderZ);
      Value *shadow_src = gutils->isConstantValue(orig_src)
                              ? gutils->getNewFromOriginal(orig_src)
                              : gutils->invertPointerM(orig_src, BuilderZ);

      auto rule = [&](Value *shadow_dst, Value *shadow_src) {
        SubTransferHelper(
            gutils, Mode, dt.isFloat(), ID, subdstalign, subsrcalign,
            /*offset*/ start, gutils->isConstantValue(orig_dst), shadow_dst,
            gutils->isConstantValue(orig_src), shadow_src,
            /*length*/ length, /*volatile*/ isVolatile, &MTI,
            /*allowForward*/ forwardsShadow, /*shadowsLookedup*/ false,
            /*backwardsShadow*/ backwardsShadow);
      };

      applyChainRule(BuilderZ, rule, shadow_dst, shadow_src);

      if (nextStart == size)
        break;
      start = nextStart;
    }

    eraseIfUnused(MTI);
  }

  void visitIntrinsicInst(llvm::IntrinsicInst &II) {
    if (II.getIntrinsicID() == Intrinsic::stacksave) {
      eraseIfUnused(II, /*erase*/ true, /*check*/ false);
      return;
    }
    if (II.getIntrinsicID() == Intrinsic::stackrestore ||
        II.getIntrinsicID() == Intrinsic::lifetime_end) {
      eraseIfUnused(II, /*erase*/ true, /*check*/ false);
      return;
    }

    SmallVector<Value *, 2> orig_ops(II.getNumOperands());

    for (unsigned i = 0; i < II.getNumOperands(); ++i) {
      orig_ops[i] = II.getOperand(i);
    }
    handleAdjointForIntrinsic(II.getIntrinsicID(), II, orig_ops);
    if (gutils->knownRecomputeHeuristic.find(&II) !=
        gutils->knownRecomputeHeuristic.end()) {
      if (!gutils->knownRecomputeHeuristic[&II]) {
        CallInst *const newCall =
            cast<CallInst>(gutils->getNewFromOriginal(&II));
        IRBuilder<> BuilderZ(newCall);
        BuilderZ.setFastMathFlags(getFast());

        gutils->cacheForReverse(BuilderZ, newCall,
                                getIndex(&II, CacheType::Self));
      }
    }
    eraseIfUnused(II);
  }

  void handleAdjointForIntrinsic(Intrinsic::ID ID, llvm::Instruction &I,
                                 SmallVectorImpl<Value *> &orig_ops) {

    Module *M = I.getParent()->getParent()->getParent();

    switch (ID) {
    case Intrinsic::nvvm_ldu_global_i:
    case Intrinsic::nvvm_ldu_global_p:
    case Intrinsic::nvvm_ldu_global_f:
    case Intrinsic::nvvm_ldg_global_i:
    case Intrinsic::nvvm_ldg_global_p:
    case Intrinsic::nvvm_ldg_global_f: {
      auto CI = cast<ConstantInt>(I.getOperand(1));
#if LLVM_VERSION_MAJOR >= 10
      visitLoadLike(I, /*Align*/ MaybeAlign(CI->getZExtValue()),
                    /*constantval*/ false);
#else
      visitLoadLike(I, /*Align*/ CI->getZExtValue(), /*constantval*/ false);
#endif
      return;
    }
    default:
      break;
    }

    if (ID == Intrinsic::masked_store) {
      auto align0 = cast<ConstantInt>(I.getOperand(2))->getZExtValue();
#if LLVM_VERSION_MAJOR >= 10
      auto align = MaybeAlign(align0);
#else
      auto align = align0;
#endif
      visitCommonStore(I, /*orig_ptr*/ I.getOperand(1),
                       /*orig_val*/ I.getOperand(0), align,
                       /*isVolatile*/ false, llvm::AtomicOrdering::NotAtomic,
                       SyncScope::SingleThread,
                       /*mask*/ gutils->getNewFromOriginal(I.getOperand(3)));
      return;
    }
    if (ID == Intrinsic::masked_load) {
      auto align0 = cast<ConstantInt>(I.getOperand(1))->getZExtValue();
#if LLVM_VERSION_MAJOR >= 10
      auto align = MaybeAlign(align0);
#else
      auto align = align0;
#endif
      auto &DL = gutils->newFunc->getParent()->getDataLayout();
      bool constantval = parseTBAA(I, DL).Inner0().isIntegral();
      visitLoadLike(I, align, constantval, /*OrigOffset*/ nullptr,
                    /*mask*/ gutils->getNewFromOriginal(I.getOperand(2)),
                    /*orig_maskInit*/ I.getOperand(3));
      return;
    }

    switch (Mode) {
    case DerivativeMode::ReverseModePrimal: {
      switch (ID) {
      case Intrinsic::nvvm_barrier0:
      case Intrinsic::nvvm_barrier0_popc:
      case Intrinsic::nvvm_barrier0_and:
      case Intrinsic::nvvm_barrier0_or:
      case Intrinsic::nvvm_membar_cta:
      case Intrinsic::nvvm_membar_gl:
      case Intrinsic::nvvm_membar_sys:
      case Intrinsic::amdgcn_s_barrier:

      case Intrinsic::prefetch:
      case Intrinsic::dbg_declare:
      case Intrinsic::dbg_value:
#if LLVM_VERSION_MAJOR > 6
      case Intrinsic::dbg_label:
#endif
      case Intrinsic::dbg_addr:
      case Intrinsic::lifetime_start:
      case Intrinsic::assume:
      case Intrinsic::fabs:
#if LLVM_VERSION_MAJOR < 10
      case Intrinsic::x86_sse_max_ss:
      case Intrinsic::x86_sse_max_ps:
      case Intrinsic::x86_sse_min_ss:
      case Intrinsic::x86_sse_min_ps:
#endif
      case Intrinsic::maxnum:
      case Intrinsic::minnum:
      case Intrinsic::log:
      case Intrinsic::log2:
      case Intrinsic::log10:
      case Intrinsic::exp:
      case Intrinsic::exp2:
      case Intrinsic::nvvm_ex2_approx_ftz_f:
      case Intrinsic::nvvm_ex2_approx_f:
      case Intrinsic::nvvm_ex2_approx_d:
      case Intrinsic::copysign:
      case Intrinsic::pow:
      case Intrinsic::powi:
#if LLVM_VERSION_MAJOR >= 12
      case Intrinsic::vector_reduce_fadd:
      case Intrinsic::vector_reduce_fmul:
#elif LLVM_VERSION_MAJOR >= 9
      case Intrinsic::experimental_vector_reduce_v2_fadd:
      case Intrinsic::experimental_vector_reduce_v2_fmul:
#endif
      case Intrinsic::sin:
      case Intrinsic::cos:
      case Intrinsic::floor:
      case Intrinsic::ceil:
      case Intrinsic::trunc:
      case Intrinsic::rint:
      case Intrinsic::nearbyint:
      case Intrinsic::round:
      case Intrinsic::sqrt:
      case Intrinsic::nvvm_sqrt_rn_d:
      case Intrinsic::fmuladd:
      case Intrinsic::fma:
        return;
      default:
        if (gutils->isConstantInstruction(&I))
          return;
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << *gutils->oldFunc << "\n";
        ss << *gutils->newFunc << "\n";
        ss << "cannot handle (augmented) unknown intrinsic\n" << I;
        if (CustomErrorHandler) {
          CustomErrorHandler(ss.str().c_str(), wrap(&I),
                             ErrorType::NoDerivative, nullptr);
        }
        llvm::errs() << ss.str() << "\n";
        report_fatal_error("(augmented) unknown intrinsic");
      }
      return;
    }

    case DerivativeMode::ReverseModeCombined:
    case DerivativeMode::ReverseModeGradient: {

      IRBuilder<> Builder2(I.getParent());
      getReverseBuilder(Builder2);

      Value *vdiff = nullptr;
      if (!gutils->isConstantValue(&I)) {
        vdiff = diffe(&I, Builder2);
        setDiffe(&I, Constant::getNullValue(gutils->getShadowType(I.getType())),
                 Builder2);
      }

      switch (ID) {

      case Intrinsic::nvvm_barrier0_popc:
      case Intrinsic::nvvm_barrier0_and:
      case Intrinsic::nvvm_barrier0_or: {
        SmallVector<Value *, 1> args = {};
        auto cal = cast<CallInst>(Builder2.CreateCall(
            Intrinsic::getDeclaration(M, Intrinsic::nvvm_barrier0), args));
        cal->setCallingConv(
            Intrinsic::getDeclaration(M, Intrinsic::nvvm_barrier0)
                ->getCallingConv());
        cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));
        return;
      }

      case Intrinsic::nvvm_barrier0:
      case Intrinsic::amdgcn_s_barrier:
      case Intrinsic::nvvm_membar_cta:
      case Intrinsic::nvvm_membar_gl:
      case Intrinsic::nvvm_membar_sys: {
        SmallVector<Value *, 1> args = {};
        auto cal = cast<CallInst>(
            Builder2.CreateCall(Intrinsic::getDeclaration(M, ID), args));
        cal->setCallingConv(Intrinsic::getDeclaration(M, ID)->getCallingConv());
        cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));
        return;
      }

      case Intrinsic::assume:
      case Intrinsic::prefetch:
      case Intrinsic::dbg_declare:
      case Intrinsic::dbg_value:
#if LLVM_VERSION_MAJOR > 6
      case Intrinsic::dbg_label:
#endif
      case Intrinsic::dbg_addr:
      case Intrinsic::floor:
      case Intrinsic::ceil:
      case Intrinsic::trunc:
      case Intrinsic::rint:
      case Intrinsic::nearbyint:
      case Intrinsic::round:
        // Derivative of these is zero and requires no modification
        return;

#if LLVM_VERSION_MAJOR >= 9
#if LLVM_VERSION_MAJOR >= 12
      case Intrinsic::vector_reduce_fadd:
#else
      case Intrinsic::experimental_vector_reduce_v2_fadd:
#endif
      {
        if (gutils->isConstantInstruction(&I))
          return;

        if (!gutils->isConstantValue(orig_ops[0])) {
          addToDiffe(orig_ops[0], vdiff, Builder2, orig_ops[0]->getType());
        }
        if (!gutils->isConstantValue(orig_ops[1])) {
          auto und = UndefValue::get(orig_ops[1]->getType());
          auto mask = ConstantAggregateZero::get(VectorType::get(
              Type::getInt32Ty(und->getContext()),
#if LLVM_VERSION_MAJOR >= 11
              cast<VectorType>(und->getType())->getElementCount()));
#else
              cast<VectorType>(und->getType())->getNumElements()));
#endif
          auto rule = [&](Value *vdiff) {
            return Builder2.CreateShuffleVector(
                Builder2.CreateInsertElement(und, vdiff, (uint64_t)0), und,
                mask);
          };
          auto vec =
              applyChainRule(orig_ops[1]->getType(), Builder2, rule, vdiff);
          addToDiffe(orig_ops[1], vec, Builder2, orig_ops[0]->getType());
        }
        return;
      }
#endif

      case Intrinsic::lifetime_start: {
        if (gutils->isConstantInstruction(&I))
          return;
        SmallVector<Value *, 2> args = {
            lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2),
            lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2)};
        Type *tys[] = {args[1]->getType()};
        auto cal = Builder2.CreateCall(
            Intrinsic::getDeclaration(M, Intrinsic::lifetime_end, tys), args);
        cal->setCallingConv(
            Intrinsic::getDeclaration(M, Intrinsic::lifetime_end, tys)
                ->getCallingConv());
        return;
      }

      case Intrinsic::nvvm_sqrt_rn_d:
      case Intrinsic::sqrt: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          SmallVector<Value *, 2> args = {
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2)};

          auto &CI = cast<CallInst>(I);
#if LLVM_VERSION_MAJOR >= 11
          auto *SqrtF = CI.getCalledOperand();
#else
          auto *SqrtF = CI.getCalledValue();
#endif
          assert(SqrtF);
          auto FT =
              cast<FunctionType>(SqrtF->getType()->getPointerElementType());

          auto cal = cast<CallInst>(Builder2.CreateCall(FT, SqrtF, args));
          cal->setCallingConv(CI.getCallingConv());
          cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));

          auto rule = [&](Value *vdiff) {
            return Builder2.CreateFDiv(
                Builder2.CreateFMul(ConstantFP::get(I.getType(), 0.5), vdiff),
                cal);
          };

          Value *dif0 =
              applyChainRule(orig_ops[0]->getType(), Builder2, rule, vdiff);
          Value *cmp = Builder2.CreateFCmpOEQ(
              args[0], Constant::getNullValue(
                           gutils->getShadowType(orig_ops[0]->getType())));
          dif0 = Builder2.CreateSelect(
              cmp,
              Constant::getNullValue(
                  gutils->getShadowType(orig_ops[0]->getType())),
              dif0);

          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        return;
      }

      case Intrinsic::fabs: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          Value *cmp = Builder2.CreateFCmpOLT(
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2),
              ConstantFP::get(orig_ops[0]->getType(), 0));

          auto rule = [&](Value *vdiff) {
            return Builder2.CreateFMul(
                Builder2.CreateSelect(
                    cmp, ConstantFP::get(orig_ops[0]->getType(), -1),
                    ConstantFP::get(orig_ops[0]->getType(), 1)),
                vdiff);
          };
          Value *dif0 =
              applyChainRule(orig_ops[0]->getType(), Builder2, rule, vdiff);
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        return;
      }

#if LLVM_VERSION_MAJOR < 10
      case Intrinsic::x86_sse_max_ss:
      case Intrinsic::x86_sse_max_ps:
#endif
      case Intrinsic::maxnum: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          Value *cmp = Builder2.CreateFCmpOLT(
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2),
              lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2));

          Value *dif0 = Builder2.CreateSelect(
              cmp,
              Constant::getNullValue(
                  gutils->getShadowType(orig_ops[0]->getType())),
              vdiff);
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        if (vdiff && !gutils->isConstantValue(orig_ops[1])) {
          Value *cmp = Builder2.CreateFCmpOLT(
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2),
              lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2));

          Value *dif1 = Builder2.CreateSelect(
              cmp, vdiff,
              Constant::getNullValue(
                  gutils->getShadowType(orig_ops[1]->getType())));
          addToDiffe(orig_ops[1], dif1, Builder2, I.getType());
        }
        return;
      }

#if LLVM_VERSION_MAJOR < 10
      case Intrinsic::x86_sse_min_ss:
      case Intrinsic::x86_sse_min_ps:
#endif
      case Intrinsic::minnum: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          Value *cmp = Builder2.CreateFCmpOLT(
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2),
              lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2));

          Value *dif0 = Builder2.CreateSelect(
              cmp, vdiff,
              Constant::getNullValue(
                  gutils->getShadowType(orig_ops[0]->getType())));
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        if (vdiff && !gutils->isConstantValue(orig_ops[1])) {
          Value *cmp = Builder2.CreateFCmpOLT(
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2),
              lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2));

          Value *dif1 = Builder2.CreateSelect(
              cmp,
              Constant::getNullValue(
                  gutils->getShadowType(orig_ops[1]->getType())),
              vdiff);
          addToDiffe(orig_ops[1], dif1, Builder2, I.getType());
        }
        return;
      }

      case Intrinsic::fmuladd:
      case Intrinsic::fma: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          Value *op1 =
              lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2);
          auto rule = [&](Value *vdiff) {
            return Builder2.CreateFMul(vdiff, op1);
          };
          Value *dif0 =
              applyChainRule(orig_ops[0]->getType(), Builder2, rule, vdiff);
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType()->getScalarType());
        }
        if (vdiff && !gutils->isConstantValue(orig_ops[1])) {
          Value *op0 =
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2);
          auto rule = [&](Value *vdiff) {
            return Builder2.CreateFMul(vdiff, op0);
          };
          Value *dif1 =
              applyChainRule(orig_ops[1]->getType(), Builder2, rule, vdiff);
          addToDiffe(orig_ops[1], dif1, Builder2, I.getType()->getScalarType());
        }
        if (vdiff && !gutils->isConstantValue(orig_ops[2])) {
          addToDiffe(orig_ops[2], vdiff, Builder2,
                     I.getType()->getScalarType());
        }
        return;
      }

      case Intrinsic::log: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          Value *op0 =
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2);
          auto rule = [&](Value *vdiff) {
            return Builder2.CreateFDiv(vdiff, op0);
          };
          Value *dif0 =
              applyChainRule(orig_ops[0]->getType(), Builder2, rule, vdiff);
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        return;
      }

      case Intrinsic::log2: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          Value *op0 =
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2);
          auto rule = [&](Value *vdiff) {
            return Builder2.CreateFDiv(
                vdiff,
                Builder2.CreateFMul(
                    ConstantFP::get(I.getType(), 0.6931471805599453), op0));
          };
          Value *dif0 =
              applyChainRule(orig_ops[0]->getType(), Builder2, rule, vdiff);
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        return;
      }
      case Intrinsic::log10: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          Value *op0 =
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2);
          auto rule = [&](Value *vdiff) {
            return Builder2.CreateFDiv(
                vdiff,
                Builder2.CreateFMul(
                    ConstantFP::get(I.getType(), 2.302585092994046), op0));
          };
          Value *dif0 =
              applyChainRule(orig_ops[0]->getType(), Builder2, rule, vdiff);
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        return;
      }

      case Intrinsic::exp:
      case Intrinsic::exp2:
      case Intrinsic::nvvm_ex2_approx_ftz_f:
      case Intrinsic::nvvm_ex2_approx_f:
      case Intrinsic::nvvm_ex2_approx_d: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          SmallVector<Value *, 2> args = {
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2)};
          SmallVector<Type *, 1> tys;
          if (ID == Intrinsic::exp || ID == Intrinsic::exp2)
            tys.push_back(orig_ops[0]->getType());
          auto ExpF = Intrinsic::getDeclaration(M, ID, tys);
          auto cal = cast<CallInst>(Builder2.CreateCall(ExpF, args));
          cal->setCallingConv(ExpF->getCallingConv());
          cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));

          auto rule = [&](Value *vdiff) {
            Value *dif0 = Builder2.CreateFMul(vdiff, cal);
            if (ID != Intrinsic::exp) {
              dif0 = Builder2.CreateFMul(
                  dif0, ConstantFP::get(I.getType(), 0.6931471805599453));
            }
            return dif0;
          };
          Value *dif0 =
              applyChainRule(orig_ops[0]->getType(), Builder2, rule, vdiff);
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        return;
      }
      case Intrinsic::copysign: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          Type *tys[] = {orig_ops[0]->getType()};
          Function *CopyF =
              Intrinsic::getDeclaration(M, Intrinsic::copysign, tys);

          Value *xsign = nullptr;
          {
            SmallVector<Value *, 2> args = {
                ConstantFP::get(tys[0], 1.0),
                lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2)};

            auto cal = cast<CallInst>(Builder2.CreateCall(CopyF, args));
            cal->setCallingConv(CopyF->getCallingConv());
            cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));
            xsign = cal;
          }

          Value *ysign = nullptr;
          {
            SmallVector<Value *, 2> args = {
                ConstantFP::get(tys[0], 1.0),
                lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2)};

            auto cal = cast<CallInst>(Builder2.CreateCall(CopyF, args));
            cal->setCallingConv(CopyF->getCallingConv());
            cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));
            ysign = cal;
          }
          auto rule = [&](Value *vdiff) {
            return Builder2.CreateFMul(Builder2.CreateFMul(xsign, ysign),
                                       vdiff);
          };
          Value *dif0 =
              applyChainRule(orig_ops[0]->getType(), Builder2, rule, vdiff);
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        return;
      }
      case Intrinsic::powi: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          Value *op0 = gutils->getNewFromOriginal(orig_ops[0]);
          Value *op1 = gutils->getNewFromOriginal(orig_ops[1]);
          Value *nop1 = lookup(op1, Builder2);
          SmallVector<Value *, 2> args = {
              lookup(op0, Builder2),
              Builder2.CreateSub(nop1, ConstantInt::get(op1->getType(), 1))};
          auto &CI = cast<CallInst>(I);
#if LLVM_VERSION_MAJOR >= 11
          auto *PowF = CI.getCalledOperand();
#else
          auto *PowF = CI.getCalledValue();
#endif
          assert(PowF);
          auto FT =
              cast<FunctionType>(PowF->getType()->getPointerElementType());
          auto cal = cast<CallInst>(Builder2.CreateCall(FT, PowF, args));
          cal->setCallingConv(CI.getCallingConv());

          cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));
          Value *op1Lookup = lookup(op1, Builder2);
          auto rule = [&](Value *vdiff) {
            return Builder2.CreateFMul(
                Builder2.CreateFMul(vdiff, cal),
                Builder2.CreateSIToFP(op1Lookup,
                                      op0->getType()->getScalarType()));
          };
          Value *dif0 =
              applyChainRule(orig_ops[0]->getType(), Builder2, rule, vdiff);
          auto cmp =
              Builder2.CreateICmpEQ(ConstantInt::get(nop1->getType(), 0), nop1);
          dif0 = Builder2.CreateSelect(
              cmp, Constant::getNullValue(dif0->getType()), dif0);
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        return;
      }
      case Intrinsic::pow: {
        Type *tys[] = {orig_ops[0]->getType()};
        auto &CI = cast<CallInst>(I);
#if LLVM_VERSION_MAJOR >= 11
        auto *PowF = CI.getCalledOperand();
#else
        auto *PowF = CI.getCalledValue();
#endif
        assert(PowF);
        auto FT = cast<FunctionType>(PowF->getType()->getPointerElementType());

        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {

          Value *op0 = gutils->getNewFromOriginal(orig_ops[0]);
          Value *op1 = gutils->getNewFromOriginal(orig_ops[1]);
          /*
          dif0 = Builder2.CreateFMul(
            Builder2.CreateFMul(vdiff,
              Builder2.CreateFDiv(lookup(&II), lookup(II.getOperand(0)))),
          lookup(II.getOperand(1))
          );
          */
          SmallVector<Value *, 2> args = {
              lookup(op0, Builder2),
              Builder2.CreateFSub(lookup(op1, Builder2),
                                  ConstantFP::get(I.getType(), 1.0))};
          auto cal = cast<CallInst>(Builder2.CreateCall(FT, PowF, args));
          cal->setCallingConv(CI.getCallingConv());
          cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));
          Value *op1Lookup = lookup(op1, Builder2);
          auto rule = [&](Value *vdiff) {
            return Builder2.CreateFMul(Builder2.CreateFMul(vdiff, cal),
                                       op1Lookup);
          };
          Value *dif0 =
              applyChainRule(orig_ops[0]->getType(), Builder2, rule, vdiff);
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }

        if (vdiff && !gutils->isConstantValue(orig_ops[1])) {

          CallInst *cal;
          {
            SmallVector<Value *, 2> args = {
                lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2),
                lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2)};

            cal = cast<CallInst>(Builder2.CreateCall(FT, PowF, args));
            cal->setCallingConv(CI.getCallingConv());

            cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));
          }

          Value *args[] = {
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2)};

          auto rule = [&](Value *vdiff) {
            return Builder2.CreateFMul(
                Builder2.CreateFMul(vdiff, cal),
                Builder2.CreateCall(
                    Intrinsic::getDeclaration(M, Intrinsic::log, tys), args));
          };
          Value *dif1 =
              applyChainRule(orig_ops[1]->getType(), Builder2, rule, vdiff);
          addToDiffe(orig_ops[1], dif1, Builder2, I.getType());
        }
        return;
      }
      case Intrinsic::sin: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          Value *args[] = {
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2)};
          Type *tys[] = {orig_ops[0]->getType()};
          CallInst *cal = cast<CallInst>(Builder2.CreateCall(
              Intrinsic::getDeclaration(M, Intrinsic::cos, tys), args));
          auto rule = [&](Value *vdiff) {
            return Builder2.CreateFMul(vdiff, cal);
          };
          Value *dif0 =
              applyChainRule(orig_ops[0]->getType(), Builder2, rule, vdiff);
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        return;
      }
      case Intrinsic::cos: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          Value *args[] = {
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2)};
          Type *tys[] = {orig_ops[0]->getType()};
          CallInst *cal = cast<CallInst>(Builder2.CreateCall(
              Intrinsic::getDeclaration(M, Intrinsic::sin, tys), args));
          auto rule = [&](Value *vdiff) {
            return Builder2.CreateFMul(vdiff, Builder2.CreateFNeg(cal));
          };
          Value *dif0 =
              applyChainRule(orig_ops[0]->getType(), Builder2, rule, vdiff);
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        return;
      }
      default:
        if (gutils->isConstantInstruction(&I))
          return;

        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << *gutils->oldFunc << "\n";
        ss << *gutils->newFunc << "\n";
        if (Intrinsic::isOverloaded(ID))
#if LLVM_VERSION_MAJOR >= 13
          ss << "cannot handle (reverse) unknown intrinsic\n"
             << Intrinsic::getName(ID, ArrayRef<Type *>(),
                                   gutils->oldFunc->getParent(), nullptr)
             << "\n"
             << I;
#else
          ss << "cannot handle (reverse) unknown intrinsic\n"
             << Intrinsic::getName(ID, ArrayRef<Type *>()) << "\n"
             << I;
#endif
        else
          ss << "cannot handle (reverse) unknown intrinsic\n"
             << Intrinsic::getName(ID) << "\n"
             << I;
        if (CustomErrorHandler) {
          CustomErrorHandler(ss.str().c_str(), wrap(&I),
                             ErrorType::NoDerivative, nullptr);
        }
        llvm::errs() << ss.str() << "\n";
        report_fatal_error("(reverse) unknown intrinsic");
      }
      return;
    }
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {

      IRBuilder<> Builder2(&I);
      getForwardBuilder(Builder2);

      switch (ID) {
#if LLVM_VERSION_MAJOR >= 9
#if LLVM_VERSION_MAJOR >= 12
      case Intrinsic::vector_reduce_fadd:
#else
      case Intrinsic::experimental_vector_reduce_v2_fadd:
#endif
      {
        if (gutils->isConstantInstruction(&I))
          return;

        Type *acctype = gutils->getShadowType(orig_ops[0]->getType());
        Type *vectype = gutils->getShadowType(orig_ops[1]->getType());

        auto accdif = gutils->isConstantValue(orig_ops[0])
                          ? Constant::getNullValue(acctype)
                          : diffe(orig_ops[0], Builder2);

        auto vecdif = gutils->isConstantValue(orig_ops[1])
                          ? Constant::getNullValue(vectype)
                          : diffe(orig_ops[1], Builder2);

#if LLVM_VERSION_MAJOR < 12
        auto vfra = Intrinsic::getDeclaration(
            M, ID, {orig_ops[0]->getType(), orig_ops[1]->getType()});
#else
        auto vfra = Intrinsic::getDeclaration(M, ID, {orig_ops[1]->getType()});
#endif

        auto rule = [&](Value *accdif, Value *vecdif) {
          auto cal = Builder2.CreateCall(vfra, {accdif, vecdif});
          cal->setCallingConv(vfra->getCallingConv());
          cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));
          return cal;
        };

        Value *dif =
            applyChainRule(I.getType(), Builder2, rule, accdif, vecdif);
        setDiffe(&I, dif, Builder2);
        return;
      }
#endif
      case Intrinsic::nvvm_sqrt_rn_d:
      case Intrinsic::sqrt: {
        if (gutils->isConstantInstruction(&I))
          return;

        Value *op = diffe(orig_ops[0], Builder2);
        Type *opType = orig_ops[0]->getType();
        Value *args[1] = {gutils->getNewFromOriginal(orig_ops[0])};
        Type *tys[] = {orig_ops[0]->getType()};

        auto &CI = cast<CallInst>(I);
#if LLVM_VERSION_MAJOR >= 11
        auto *SqrtF = CI.getCalledOperand();
#else
        auto *SqrtF = CI.getCalledValue();
#endif
        assert(SqrtF);
        auto FT = cast<FunctionType>(SqrtF->getType()->getPointerElementType());

        auto rule = [&](Value *op) {
          CallInst *cal = cast<CallInst>(Builder2.CreateCall(FT, SqrtF, args));
          cal->setCallingConv(CI.getCallingConv());
          cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));

          Value *half = ConstantFP::get(orig_ops[0]->getType(), 0.5);
          Value *dif0 = Builder2.CreateFDiv(Builder2.CreateFMul(half, op), cal);

          Value *cmp =
              Builder2.CreateFCmpOEQ(args[0], Constant::getNullValue(tys[0]));
          return Builder2.CreateSelect(cmp, Constant::getNullValue(opType),
                                       dif0);
        };

        Value *dif0 = applyChainRule(I.getType(), Builder2, rule, op);
        setDiffe(&I, dif0, Builder2);
        return;
      }

      case Intrinsic::fabs: {
        if (gutils->isConstantInstruction(&I))
          return;

        Value *op = diffe(orig_ops[0], Builder2);
        Type *ty = orig_ops[0]->getType();

        auto rule = [&](Value *op) {
          Value *cmp =
              Builder2.CreateFCmpOLT(gutils->getNewFromOriginal(orig_ops[0]),
                                     Constant::getNullValue(ty));
          Value *select = Builder2.CreateSelect(cmp, ConstantFP::get(ty, -1),
                                                ConstantFP::get(ty, 1));
          return Builder2.CreateFMul(select, op);
        };

        auto dif0 = applyChainRule(I.getType(), Builder2, rule, op);

        setDiffe(&I, dif0, Builder2);
        return;
      }

#if LLVM_VERSION_MAJOR < 10
      case Intrinsic::x86_sse_max_ss:
      case Intrinsic::x86_sse_max_ps:
#endif
      case Intrinsic::maxnum: {
        if (gutils->isConstantInstruction(&I))
          return;
        Value *op0 = gutils->getNewFromOriginal(orig_ops[0]);
        Value *op1 = gutils->getNewFromOriginal(orig_ops[1]);
        Value *cmp = Builder2.CreateFCmpOLT(op0, op1);

        Type *opType0 = gutils->getShadowType(orig_ops[0]->getType());
        Type *opType1 = gutils->getShadowType(orig_ops[1]->getType());

        Value *diffe0 = gutils->isConstantValue(orig_ops[0])
                            ? Constant::getNullValue(opType0)
                            : diffe(orig_ops[0], Builder2);
        Value *diffe1 = gutils->isConstantValue(orig_ops[1])
                            ? Constant::getNullValue(opType1)
                            : diffe(orig_ops[1], Builder2);

        auto rule = [&](Value *diffe0, Value *diffe1) {
          return Builder2.CreateSelect(cmp, diffe0, diffe1);
        };

        Value *dif =
            applyChainRule(I.getType(), Builder2, rule, diffe0, diffe1);
        setDiffe(&I, dif, Builder2);
        return;
      }

#if LLVM_VERSION_MAJOR < 10
      case Intrinsic::x86_sse_min_ss:
      case Intrinsic::x86_sse_min_ps:
#endif
      case Intrinsic::minnum: {
        if (gutils->isConstantInstruction(&I))
          return;
        Value *op0 = gutils->getNewFromOriginal(orig_ops[0]);
        Value *op1 = gutils->getNewFromOriginal(orig_ops[1]);
        Value *cmp = Builder2.CreateFCmpOLT(op0, op1);

        Type *opType0 = gutils->getShadowType(orig_ops[0]->getType());
        Type *opType1 = gutils->getShadowType(orig_ops[1]->getType());

        Value *diffe0 = gutils->isConstantValue(orig_ops[0])
                            ? Constant::getNullValue(opType0)
                            : diffe(orig_ops[0], Builder2);
        Value *diffe1 = gutils->isConstantValue(orig_ops[1])
                            ? Constant::getNullValue(opType1)
                            : diffe(orig_ops[1], Builder2);

        auto rule = [&](Value *diffe0, Value *diffe1) {
          return Builder2.CreateSelect(cmp, diffe0, diffe1);
        };

        Value *dif =
            applyChainRule(I.getType(), Builder2, rule, diffe0, diffe1);
        setDiffe(&I, dif, Builder2);

        return;
      }

      case Intrinsic::fmuladd:
      case Intrinsic::fma: {
        if (gutils->isConstantInstruction(&I))
          return;

        Value *op0 = gutils->getNewFromOriginal(orig_ops[0]);
        Value *op1 = gutils->getNewFromOriginal(orig_ops[1]);

        Type *opType0 = gutils->getShadowType(orig_ops[0]->getType());
        Type *opType1 = gutils->getShadowType(orig_ops[1]->getType());
        Type *opType2 = gutils->getShadowType(orig_ops[2]->getType());

        Value *dif0 = gutils->isConstantValue(orig_ops[0])
                          ? Constant::getNullValue(opType0)
                          : diffe(orig_ops[0], Builder2);
        Value *dif1 = gutils->isConstantValue(orig_ops[1])
                          ? Constant::getNullValue(opType1)
                          : diffe(orig_ops[1], Builder2);
        Value *dif2 = gutils->isConstantValue(orig_ops[2])
                          ? Constant::getNullValue(opType2)
                          : diffe(orig_ops[2], Builder2);

        auto rule = [&](Value *dif0, Value *dif1, Value *dif2) {
          Value *dif =
              Builder2.CreateFAdd(gutils->isConstantValue(orig_ops[1])
                                      ? Constant::getNullValue(opType1)
                                      : Builder2.CreateFMul(op0, dif1),
                                  gutils->isConstantValue(orig_ops[0])
                                      ? Constant::getNullValue(opType2)
                                      : Builder2.CreateFMul(op1, dif0));
          return Builder2.CreateFAdd(dif, dif2);
        };

        Value *dif =
            applyChainRule(I.getType(), Builder2, rule, dif0, dif1, dif2);
        setDiffe(&I, dif, Builder2);

        return;
      }

      case Intrinsic::log: {
        if (gutils->isConstantInstruction(&I))
          return;
        Value *op = diffe(orig_ops[0], Builder2);
        Value *origOp = gutils->getNewFromOriginal(orig_ops[0]);

        auto rule = [&](Value *op) { return Builder2.CreateFDiv(op, origOp); };

        Value *dif0 = applyChainRule(I.getType(), Builder2, rule, op);
        setDiffe(&I, dif0, Builder2);
        return;
      }

      case Intrinsic::log2: {
        if (gutils->isConstantInstruction(&I))
          return;

        Value *op = diffe(orig_ops[0], Builder2);
        Value *c = ConstantFP::get(I.getType(), 0.6931471805599453);
        Value *origOp = gutils->getNewFromOriginal(orig_ops[0]);
        Value *mul = Builder2.CreateFMul(c, origOp);

        auto rule = [&](Value *op) { return Builder2.CreateFDiv(op, mul); };

        Value *dif0 = applyChainRule(I.getType(), Builder2, rule, op);
        setDiffe(&I, dif0, Builder2);
        return;
      }
      case Intrinsic::log10: {
        if (gutils->isConstantInstruction(&I))
          return;

        Value *op = diffe(orig_ops[0], Builder2);
        Value *c = ConstantFP::get(I.getType(), 2.302585092994046);
        Value *origOp = gutils->getNewFromOriginal(orig_ops[0]);
        Value *mul = Builder2.CreateFMul(c, origOp);

        auto rule = [&](Value *op) { return Builder2.CreateFDiv(op, mul); };

        Value *dif0 = applyChainRule(I.getType(), Builder2, rule, op);
        setDiffe(&I, dif0, Builder2);
        return;
      }

      case Intrinsic::exp:
      case Intrinsic::exp2:
      case Intrinsic::nvvm_ex2_approx_ftz_f:
      case Intrinsic::nvvm_ex2_approx_f:
      case Intrinsic::nvvm_ex2_approx_d: {
        if (gutils->isConstantInstruction(&I))
          return;

        Value *op = diffe(orig_ops[0], Builder2);
        Value *args[1] = {gutils->getNewFromOriginal(orig_ops[0])};
        SmallVector<Type *, 1> tys;
        if (ID == Intrinsic::exp || ID == Intrinsic::exp2)
          tys.push_back(orig_ops[0]->getType());
        auto ExpF = Intrinsic::getDeclaration(M, ID, tys);
        CallInst *cal = Builder2.CreateCall(ExpF, args);
        cal->setCallingConv(ExpF->getCallingConv());
        cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));

        Value *c = ConstantFP::get(I.getType(), 0.6931471805599453);

        auto rule = [&](Value *op) {
          Value *dif0 = Builder2.CreateFMul(op, cal);
          if (ID != Intrinsic::exp)
            dif0 = Builder2.CreateFMul(dif0, c);
          return dif0;
        };

        auto dif0 = applyChainRule(I.getType(), Builder2, rule, op);
        setDiffe(&I, dif0, Builder2);
        return;
      }
      case Intrinsic::copysign: {
        if (gutils->isConstantInstruction(&I))
          return;

        Type *tys[] = {orig_ops[0]->getType()};
        Function *CopyF =
            Intrinsic::getDeclaration(M, Intrinsic::copysign, tys);

        Value *xsign = nullptr;
        {
          Value *args[2] = {ConstantFP::get(tys[0], 1.0),
                            gutils->getNewFromOriginal(orig_ops[0])};

          auto cal = cast<CallInst>(Builder2.CreateCall(CopyF, args));
          cal->setCallingConv(CopyF->getCallingConv());
          cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));
          xsign = cal;
        }

        Value *ysign = nullptr;
        {
          Value *args[2] = {ConstantFP::get(tys[0], 1.0),
                            gutils->getNewFromOriginal(orig_ops[1])};

          auto cal = cast<CallInst>(Builder2.CreateCall(CopyF, args));
          cal->setCallingConv(CopyF->getCallingConv());
          cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));
          ysign = cal;
        }

        Value *op = diffe(orig_ops[0], Builder2);

        auto rule = [&](Value *op) {
          return Builder2.CreateFMul(Builder2.CreateFMul(xsign, ysign), op);
        };

        Value *dif0 = applyChainRule(I.getType(), Builder2, rule, op);
        setDiffe(&I, dif0, Builder2);
        return;
      }
      case Intrinsic::powi: {
        if (gutils->isConstantInstruction(&I))
          return;
        if (!gutils->isConstantValue(orig_ops[0])) {
          Value *op0 = gutils->getNewFromOriginal(orig_ops[0]);
          Value *op1 = gutils->getNewFromOriginal(orig_ops[1]);
          SmallVector<Value *, 2> args = {
              op0,
              Builder2.CreateSub(op1, ConstantInt::get(op1->getType(), 1))};
          auto &CI = cast<CallInst>(I);
#if LLVM_VERSION_MAJOR >= 11
          auto *PowF = CI.getCalledOperand();
#else
          auto *PowF = CI.getCalledValue();
#endif
          assert(PowF);
          auto FT =
              cast<FunctionType>(PowF->getType()->getPointerElementType());
          auto cal = cast<CallInst>(Builder2.CreateCall(FT, PowF, args));
          cal->setCallingConv(CI.getCallingConv());
          cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));

          Value *cast =
              Builder2.CreateSIToFP(op1, op0->getType()->getScalarType());
          Value *op = diffe(orig_ops[0], Builder2);

          Value *cmp = Builder2.CreateICmpEQ(
              ConstantInt::get(args[1]->getType(), 0), op1);
          auto rule = [&](Value *op) {
            return Builder2.CreateSelect(
                cmp, Constant::getNullValue(op->getType()),
                Builder2.CreateFMul(Builder2.CreateFMul(op, cal), cast));
          };

          Value *dif0 = applyChainRule(I.getType(), Builder2, rule, op);
          setDiffe(&I, dif0, Builder2);
        }
        return;
      }
      case Intrinsic::pow: {
        if (gutils->isConstantInstruction(&I))
          return;

        auto &CI = cast<CallInst>(I);
#if LLVM_VERSION_MAJOR >= 11
        auto *PowF = CI.getCalledOperand();
#else
        auto *PowF = CI.getCalledValue();
#endif
        assert(PowF);
        auto FT = cast<FunctionType>(PowF->getType()->getPointerElementType());

        Value *op0 = gutils->getNewFromOriginal(orig_ops[0]);
        Value *op1 = gutils->getNewFromOriginal(orig_ops[1]);

        Value *res =
            Constant::getNullValue(gutils->getShadowType(CI.getType()));

        auto &DL = gutils->newFunc->getParent()->getDataLayout();

        if (!gutils->isConstantValue(orig_ops[0])) {
          Value *args[2] = {
              op0,
              Builder2.CreateFSub(op1, ConstantFP::get(op1->getType(), 1.0))};
          Value *powcall1 = Builder2.CreateCall(FT, PowF, args);
          cast<CallInst>(powcall1)->setCallingConv(CI.getCallingConv());
          cast<CallInst>(powcall1)->setDebugLoc(
              gutils->getNewFromOriginal(I.getDebugLoc()));

          if (powcall1->getType() != op1->getType()) {
            if (DL.getTypeSizeInBits(powcall1->getType()) <
                DL.getTypeSizeInBits(op1->getType()))
              powcall1 = Builder2.CreateFPExt(powcall1, op1->getType());
            else
              powcall1 = Builder2.CreateFPTrunc(powcall1, op1->getType());
          }

          Value *mul = Builder2.CreateFMul(op1, powcall1);
          Value *op = diffe(orig_ops[0], Builder2);

          auto rule = [&](Value *op, Value *res) {
            Value *out = Builder2.CreateFMul(mul, op);

            if (out->getType() != CI.getType()) {
              if (DL.getTypeSizeInBits(out->getType()) <
                  DL.getTypeSizeInBits(CI.getType()))
                out = Builder2.CreateFPExt(out, CI.getType());
              else
                out = Builder2.CreateFPTrunc(out, CI.getType());
            }
            return Builder2.CreateFAdd(res, out);
          };

          res = applyChainRule(I.getType(), Builder2, rule, op, res);
        }
        if (!gutils->isConstantValue(orig_ops[1])) {
          Value *powcall = Builder2.CreateCall(FT, PowF, {op0, op1});
          cast<CallInst>(powcall)->setCallingConv(CI.getCallingConv());
          cast<CallInst>(powcall)->setDebugLoc(
              gutils->getNewFromOriginal(I.getDebugLoc()));

          Type *tys[] = {op0->getType()};
          CallInst *logcall = Builder2.CreateCall(
              Intrinsic::getDeclaration(M, Intrinsic::log, tys), {op0});

          if (powcall->getType() != op0->getType()) {
            if (DL.getTypeSizeInBits(powcall->getType()) <
                DL.getTypeSizeInBits(op0->getType()))
              powcall = Builder2.CreateFPExt(powcall, op0->getType());
            else
              powcall = Builder2.CreateFPTrunc(powcall, op0->getType());
          }

          Value *mul = Builder2.CreateFMul(powcall, logcall);
          Value *op = diffe(orig_ops[1], Builder2);

          auto rule = [&](Value *op, Value *res) {
            Value *out = Builder2.CreateFMul(mul, op);

            if (out->getType() != CI.getType()) {
              if (DL.getTypeSizeInBits(out->getType()) <
                  DL.getTypeSizeInBits(CI.getType()))
                out = Builder2.CreateFPExt(out, CI.getType());
              else
                out = Builder2.CreateFPTrunc(out, CI.getType());
            }
            return Builder2.CreateFAdd(res, out);
          };

          res = applyChainRule(I.getType(), Builder2, rule, op, res);
        }

        setDiffe(&I, res, Builder2);
        return;
      }
      case Intrinsic::sin: {
        if (gutils->isConstantInstruction(&I))
          return;
        Value *args[] = {gutils->getNewFromOriginal(orig_ops[0])};
        Type *tys[] = {orig_ops[0]->getType()};
        Value *cal = Builder2.CreateCall(
            Intrinsic::getDeclaration(M, Intrinsic::cos, tys), args);
        Value *op = diffe(orig_ops[0], Builder2);

        auto rule = [&](Value *op) { return Builder2.CreateFMul(op, cal); };

        Value *dif0 = applyChainRule(I.getType(), Builder2, rule, op);
        setDiffe(&I, dif0, Builder2);
        return;
      }
      case Intrinsic::cos: {
        if (gutils->isConstantInstruction(&I))
          return;

        Value *args[] = {gutils->getNewFromOriginal(orig_ops[0])};
        Type *tys[] = {orig_ops[0]->getType()};
        Value *cal = Builder2.CreateCall(
            Intrinsic::getDeclaration(M, Intrinsic::sin, tys), args);
        cal = Builder2.CreateFNeg(cal);
        Value *op = diffe(orig_ops[0], Builder2);

        auto rule = [&](Value *op) { return Builder2.CreateFMul(op, cal); };

        Value *dif0 = applyChainRule(I.getType(), Builder2, rule, op);
        setDiffe(&I, dif0, Builder2);
        return;
      }
      default:
        if (gutils->isConstantInstruction(&I))
          return;
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << *gutils->oldFunc << "\n";
        ss << *gutils->newFunc << "\n";
        if (Intrinsic::isOverloaded(ID))
#if LLVM_VERSION_MAJOR >= 13
          ss << "cannot handle (forward) unknown intrinsic\n"
             << Intrinsic::getName(ID, ArrayRef<Type *>(),
                                   gutils->oldFunc->getParent(), nullptr)
             << "\n"
             << I;
#else
          ss << "cannot handle (forward) unknown intrinsic\n"
             << Intrinsic::getName(ID, ArrayRef<Type *>()) << "\n"
             << I;
#endif
        else
          ss << "cannot handle (forward) unknown intrinsic\n"
             << Intrinsic::getName(ID) << "\n"
             << I;
        if (CustomErrorHandler) {
          CustomErrorHandler(ss.str().c_str(), wrap(&I),
                             ErrorType::NoDerivative, nullptr);
        }
        llvm::errs() << ss.str() << "\n";
        report_fatal_error("(forward) unknown intrinsic");
      }
      return;
    }
    }
  }

  void visitOMPCall(llvm::CallInst &call) {
    Function *kmpc = call.getCalledFunction();

    if (uncacheable_args_map.find(&call) == uncacheable_args_map.end()) {
      llvm::errs() << " call: " << call << "\n";
      for (auto &pair : uncacheable_args_map) {
        llvm::errs() << " + " << *pair.first << "\n";
      }
    }

    assert(uncacheable_args_map.find(&call) != uncacheable_args_map.end());
    const std::map<Argument *, bool> &uncacheable_args =
        uncacheable_args_map.find(&call)->second;

    IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&call));
    BuilderZ.setFastMathFlags(getFast());

    Function *task = dyn_cast<Function>(call.getArgOperand(2));
    if (task == nullptr && isa<ConstantExpr>(call.getArgOperand(2))) {
      task = dyn_cast<Function>(
          cast<ConstantExpr>(call.getArgOperand(2))->getOperand(0));
    }
    if (task == nullptr) {
      llvm::errs() << "could not derive underlying task from omp call: " << call
                   << "\n";
      llvm_unreachable("could not derive underlying task from omp call");
    }
    if (task->empty()) {
      llvm::errs()
          << "could not derive underlying task contents from omp call: " << call
          << "\n";
      llvm_unreachable(
          "could not derive underlying task contents from omp call");
    }

    auto called = task;
    // bool modifyPrimal = true;

    bool foreignFunction = called == nullptr;

    SmallVector<Value *, 8> args = {0, 0, 0};
    SmallVector<Value *, 8> pre_args = {0, 0, 0};
    std::vector<DIFFE_TYPE> argsInverted = {DIFFE_TYPE::CONSTANT,
                                            DIFFE_TYPE::CONSTANT};
    SmallVector<Instruction *, 4> postCreate;
    SmallVector<Instruction *, 4> userReplace;

    SmallVector<Value *, 4> OutTypes;
    SmallVector<Type *, 4> OutFPTypes;

#if LLVM_VERSION_MAJOR >= 14
    for (unsigned i = 3; i < call.arg_size(); ++i)
#else
    for (unsigned i = 3; i < call.getNumArgOperands(); ++i)
#endif
    {

      auto argi = gutils->getNewFromOriginal(call.getArgOperand(i));

      pre_args.push_back(argi);

      if (Mode != DerivativeMode::ReverseModePrimal) {
        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);
        args.push_back(lookup(argi, Builder2));
      }

      if (gutils->isConstantValue(call.getArgOperand(i)) && !foreignFunction) {
        argsInverted.push_back(DIFFE_TYPE::CONSTANT);
        continue;
      }

      auto argType = argi->getType();

      if (!argType->isFPOrFPVectorTy() &&
          TR.query(call.getArgOperand(i)).Inner0().isPossiblePointer()) {
        DIFFE_TYPE ty = DIFFE_TYPE::DUP_ARG;
        if (argType->isPointerTy()) {
#if LLVM_VERSION_MAJOR >= 12
          auto at = getUnderlyingObject(call.getArgOperand(i), 100);
#else
          auto at = GetUnderlyingObject(
              call.getArgOperand(i),
              gutils->oldFunc->getParent()->getDataLayout(), 100);
#endif
          if (auto arg = dyn_cast<Argument>(at)) {
            if (constant_args[arg->getArgNo()] == DIFFE_TYPE::DUP_NONEED) {
              ty = DIFFE_TYPE::DUP_NONEED;
            }
          }
        }
        argsInverted.push_back(ty);

        if (Mode != DerivativeMode::ReverseModePrimal) {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);
          args.push_back(
              lookup(gutils->invertPointerM(call.getArgOperand(i), Builder2),
                     Builder2));
        }
        pre_args.push_back(
            gutils->invertPointerM(call.getArgOperand(i), BuilderZ));

        // Note sometimes whattype mistakenly says something should be constant
        // [because composed of integer pointers alone]
        assert(whatType(argType, Mode) == DIFFE_TYPE::DUP_ARG ||
               whatType(argType, Mode) == DIFFE_TYPE::CONSTANT);
      } else {
        assert(TR.query(call.getArgOperand(i)).Inner0().isFloat());
        OutTypes.push_back(call.getArgOperand(i));
        OutFPTypes.push_back(argType);
        argsInverted.push_back(DIFFE_TYPE::OUT_DIFF);
        assert(whatType(argType, Mode) == DIFFE_TYPE::OUT_DIFF ||
               whatType(argType, Mode) == DIFFE_TYPE::CONSTANT);
      }
    }

    DIFFE_TYPE subretType = DIFFE_TYPE::CONSTANT;

    Value *tape = nullptr;
    CallInst *augmentcall = nullptr;
    // Value *cachereplace = nullptr;

    // TODO consider reduction of int 0 args
    FnTypeInfo nextTypeInfo(called);

    if (called) {
      std::map<Value *, std::set<int64_t>> intseen;

      TypeTree IntPtr;
      IntPtr.insert({-1, -1}, BaseType::Integer);
      IntPtr.insert({-1}, BaseType::Pointer);

      int argnum = 0;
      for (auto &arg : called->args()) {
        if (argnum <= 1) {
          nextTypeInfo.Arguments.insert(
              std::pair<Argument *, TypeTree>(&arg, IntPtr));
          nextTypeInfo.KnownValues.insert(
              std::pair<Argument *, std::set<int64_t>>(&arg, {0}));
        } else {
          nextTypeInfo.Arguments.insert(std::pair<Argument *, TypeTree>(
              &arg, TR.query(call.getArgOperand(argnum - 2 + 3))));
          nextTypeInfo.KnownValues.insert(
              std::pair<Argument *, std::set<int64_t>>(
                  &arg,
                  TR.knownIntegralValues(call.getArgOperand(argnum - 2 + 3))));
        }

        ++argnum;
      }
      nextTypeInfo.Return = TR.query(&call);
    }

    // llvm::Optional<std::map<std::pair<Instruction*, std::string>, unsigned>>
    // sub_index_map;
    // Optional<int> tapeIdx;
    // Optional<int> returnIdx;
    // Optional<int> differetIdx;

    const AugmentedReturn *subdata = nullptr;
    if (Mode == DerivativeMode::ReverseModeGradient) {
      assert(augmentedReturn);
      if (augmentedReturn) {
        auto fd = augmentedReturn->subaugmentations.find(&call);
        if (fd != augmentedReturn->subaugmentations.end()) {
          subdata = fd->second;
        }
      }
    }

    if (Mode == DerivativeMode::ReverseModePrimal ||
        Mode == DerivativeMode::ReverseModeCombined) {
      if (called) {
        subdata = &gutils->Logic.CreateAugmentedPrimal(
            cast<Function>(called), subretType, argsInverted,
            TR.analyzer.interprocedural, /*return is used*/ false,
            /*shadowReturnUsed*/ false, nextTypeInfo, uncacheable_args, false,
            gutils->getWidth(),
            /*AtomicAdd*/ true,
            /*OpenMP*/ true);
        if (Mode == DerivativeMode::ReverseModePrimal) {
          assert(augmentedReturn);
          auto subaugmentations =
              (std::map<const llvm::CallInst *, AugmentedReturn *>
                   *)&augmentedReturn->subaugmentations;
          insert_or_assign2<const llvm::CallInst *, AugmentedReturn *>(
              *subaugmentations, &call, (AugmentedReturn *)subdata);
        }

        assert(subdata);
        auto newcalled = subdata->fn;

        if (subdata->returns.find(AugmentedStruct::Tape) !=
            subdata->returns.end()) {
          ValueToValueMapTy VMap;
          newcalled = CloneFunction(newcalled, VMap);
          auto tapeArg = newcalled->arg_end();
          tapeArg--;
          SmallVector<std::pair<ssize_t, Value *>, 4> geps;
          SmallPtrSet<Instruction *, 4> gepsToErase;
          for (auto a : tapeArg->users()) {
            if (auto gep = dyn_cast<GetElementPtrInst>(a)) {
              auto idx = gep->idx_begin();
              idx++;
              auto cidx = cast<ConstantInt>(idx->get());
              assert(gep->getNumIndices() == 2);
              SmallPtrSet<StoreInst *, 1> storesToErase;
              for (auto st : gep->users()) {
                auto SI = cast<StoreInst>(st);
                Value *op = SI->getValueOperand();
                storesToErase.insert(SI);
                geps.emplace_back(cidx->getLimitedValue(), op);
              }
              for (auto SI : storesToErase)
                SI->eraseFromParent();
              gepsToErase.insert(gep);
            } else if (auto SI = dyn_cast<StoreInst>(a)) {
              Value *op = SI->getValueOperand();
              gepsToErase.insert(SI);
              geps.emplace_back(-1, op);
            } else {
              llvm::errs() << "unknown tape user: " << a << "\n";
              assert(0 && "unknown tape user");
              llvm_unreachable("unknown tape user");
            }
          }
          for (auto gep : gepsToErase)
            gep->eraseFromParent();
          IRBuilder<> ph(&*newcalled->getEntryBlock().begin());
          tape = UndefValue::get(tapeArg->getType()->getPointerElementType());
          ValueToValueMapTy available;
          auto subarg = newcalled->arg_begin();
          subarg++;
          subarg++;
          for (size_t i = 3; i < pre_args.size(); ++i) {
            available[&*subarg] = pre_args[i];
            subarg++;
          }
          for (auto pair : geps) {
            Value *op = pair.second;
            Value *alloc = op;
            Value *replacement = gutils->unwrapM(op, BuilderZ, available,
                                                 UnwrapMode::LegalFullUnwrap);
            tape =
                pair.first == -1
                    ? replacement
                    : BuilderZ.CreateInsertValue(tape, replacement, pair.first);
            if (auto ci = dyn_cast<CastInst>(alloc)) {
              alloc = ci->getOperand(0);
            }
            if (auto uload = dyn_cast<Instruction>(replacement)) {
              gutils->unwrappedLoads.erase(uload);
              if (auto ci = dyn_cast<CastInst>(replacement)) {
                if (auto ucast = dyn_cast<Instruction>(ci->getOperand(0)))
                  gutils->unwrappedLoads.erase(ucast);
              }
            }
            if (auto ci = dyn_cast<CallInst>(alloc)) {
              if (auto F = ci->getCalledFunction()) {
                // Store cached values
                if (F->getName() == "malloc") {
                  const_cast<AugmentedReturn *>(subdata)
                      ->tapeIndiciesToFree.emplace(pair.first);
                  Value *Idxs[] = {
                      ConstantInt::get(Type::getInt64Ty(tapeArg->getContext()),
                                       0),
                      ConstantInt::get(Type::getInt32Ty(tapeArg->getContext()),
                                       pair.first)};
#if LLVM_VERSION_MAJOR > 7
                  op->replaceAllUsesWith(ph.CreateLoad(
                      op->getType(),
                      pair.first == -1
                          ? tapeArg
                          : ph.CreateInBoundsGEP(
                                tapeArg->getType()->getPointerElementType(),
                                tapeArg, Idxs)));
#else
                  op->replaceAllUsesWith(ph.CreateLoad(
                      pair.first == -1 ? tapeArg
                                       : ph.CreateInBoundsGEP(tapeArg, Idxs)));
#endif
                  cast<Instruction>(op)->eraseFromParent();
                  if (op != alloc)
                    ci->eraseFromParent();
                  continue;
                }
              }
            }
            Value *Idxs[] = {
                ConstantInt::get(Type::getInt64Ty(tapeArg->getContext()), 0),
                ConstantInt::get(Type::getInt32Ty(tapeArg->getContext()),
                                 pair.first)};
#if LLVM_VERSION_MAJOR > 7
            op->replaceAllUsesWith(ph.CreateLoad(
                op->getType(),
                pair.first == -1
                    ? tapeArg
                    : ph.CreateInBoundsGEP(
                          tapeArg->getType()->getPointerElementType(), tapeArg,
                          Idxs)));
#else
            op->replaceAllUsesWith(ph.CreateLoad(
                pair.first == -1 ? tapeArg
                                 : ph.CreateInBoundsGEP(tapeArg, Idxs)));
#endif
            cast<Instruction>(op)->eraseFromParent();
          }
          assert(tape);
          auto alloc =
              IRBuilder<>(gutils->inversionAllocs)
                  .CreateAlloca(tapeArg->getType()->getPointerElementType());
          BuilderZ.CreateStore(tape, alloc);
          pre_args.push_back(alloc);
          assert(tape);
          gutils->cacheForReverse(BuilderZ, tape,
                                  getIndex(&call, CacheType::Tape));
        }

        auto numargs = ConstantInt::get(Type::getInt32Ty(call.getContext()),
                                        pre_args.size() - 3);
        pre_args[0] = gutils->getNewFromOriginal(call.getArgOperand(0));
        pre_args[1] = numargs;
        pre_args[2] = BuilderZ.CreatePointerCast(
            newcalled, kmpc->getFunctionType()->getParamType(2));
        augmentcall =
            BuilderZ.CreateCall(kmpc->getFunctionType(), kmpc, pre_args);
        augmentcall->setCallingConv(call.getCallingConv());
        augmentcall->setDebugLoc(
            gutils->getNewFromOriginal(call.getDebugLoc()));
        BuilderZ.SetInsertPoint(
            gutils->getNewFromOriginal(&call)->getNextNode());
        gutils->getNewFromOriginal(&call)->eraseFromParent();
      } else {
        assert(0 && "unhandled unknown outline");
      }
    }

    if (!subdata) {
      llvm::errs() << *gutils->oldFunc->getParent() << "\n";
      llvm::errs() << *gutils->oldFunc << "\n";
      llvm::errs() << *gutils->newFunc << "\n";
      llvm::errs() << *called << "\n";
      llvm_unreachable("no subdata");
    }

    auto found = subdata->returns.find(AugmentedStruct::DifferentialReturn);
    assert(found == subdata->returns.end());

    found = subdata->returns.find(AugmentedStruct::Return);
    assert(found == subdata->returns.end());

    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined) {
      IRBuilder<> Builder2(call.getParent());
      getReverseBuilder(Builder2);

      if (Mode == DerivativeMode::ReverseModeGradient) {
        BuilderZ.SetInsertPoint(
            gutils->getNewFromOriginal(&call)->getNextNode());
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      }

      Function *newcalled = nullptr;
      if (called) {
        if (subdata->returns.find(AugmentedStruct::Tape) !=
            subdata->returns.end()) {
          if (Mode == DerivativeMode::ReverseModeGradient) {
            if (tape == nullptr)
              tape = BuilderZ.CreatePHI(subdata->tapeType, 0, "tapeArg");
            tape = gutils->cacheForReverse(BuilderZ, tape,
                                           getIndex(&call, CacheType::Tape));
          }
          tape = lookup(tape, Builder2);
          auto alloc = IRBuilder<>(gutils->inversionAllocs)
                           .CreateAlloca(tape->getType());
          Builder2.CreateStore(tape, alloc);
          args.push_back(alloc);
        }

        newcalled = gutils->Logic.CreatePrimalAndGradient(
            (ReverseCacheKey){.todiff = cast<Function>(called),
                              .retType = subretType,
                              .constant_args = argsInverted,
                              .uncacheable_args = uncacheable_args,
                              .returnUsed = false,
                              .shadowReturnUsed = false,
                              .mode = DerivativeMode::ReverseModeGradient,
                              .width = gutils->getWidth(),
                              .freeMemory = true,
                              .AtomicAdd = true,
                              .additionalType =
                                  tape ? PointerType::getUnqual(tape->getType())
                                       : nullptr,
                              .typeInfo = nextTypeInfo},
            TR.analyzer.interprocedural, subdata,
            /*omp*/ true);

        if (subdata->returns.find(AugmentedStruct::Tape) !=
            subdata->returns.end()) {
          auto tapeArg = newcalled->arg_end();
          tapeArg--;
          LoadInst *tape = nullptr;
          for (auto u : tapeArg->users()) {
            assert(!tape);
            if (!isa<LoadInst>(u)) {
              llvm::errs() << " newcalled: " << *newcalled << "\n";
              llvm::errs() << " u: " << *u << "\n";
            }
            tape = cast<LoadInst>(u);
          }
          assert(tape);
          SmallVector<Value *, 4> extracts;
          if (subdata->tapeIndices.size() == 1) {
            assert(subdata->tapeIndices.begin()->second == -1);
            extracts.push_back(tape);
          } else {
            for (auto a : tape->users()) {
              extracts.push_back(a);
            }
          }
          SmallVector<LoadInst *, 4> geps;
          for (auto E : extracts) {
            AllocaInst *AI = nullptr;
            for (auto U : E->users()) {
              if (auto SI = dyn_cast<StoreInst>(U)) {
                assert(SI->getValueOperand() == E);
                AI = cast<AllocaInst>(SI->getPointerOperand());
              }
            }
            if (AI) {
              for (auto U : AI->users()) {
                if (auto LI = dyn_cast<LoadInst>(U)) {
                  geps.push_back(LI);
                }
              }
            }
          }
          size_t freeCount = 0;
          for (auto LI : geps) {
            CallInst *freeCall = nullptr;
            for (auto LU : LI->users()) {
              if (auto CI = dyn_cast<CallInst>(LU)) {
                if (auto F = CI->getCalledFunction()) {
                  if (F->getName() == "free") {
                    freeCall = CI;
                    break;
                  }
                }
              } else if (auto BC = dyn_cast<CastInst>(LU)) {
                for (auto CU : BC->users()) {
                  if (auto CI = dyn_cast<CallInst>(CU)) {
                    if (auto F = CI->getCalledFunction()) {
                      if (F->getName() == "free") {
                        freeCall = CI;
                        break;
                      }
                    }
                  }
                }
                if (freeCall)
                  break;
              }
            }
            if (freeCall) {
              freeCall->eraseFromParent();
              freeCount++;
            }
          }
        }

        Value *OutAlloc = nullptr;
        auto ST = StructType::get(newcalled->getContext(), OutFPTypes);
        if (OutTypes.size()) {
          OutAlloc = IRBuilder<>(gutils->inversionAllocs).CreateAlloca(ST);
          args.push_back(OutAlloc);

          SmallVector<Type *, 3> MetaTypes;
          for (auto P :
               cast<Function>(newcalled)->getFunctionType()->params()) {
            MetaTypes.push_back(P);
          }
          MetaTypes.push_back(PointerType::getUnqual(ST));
          auto FT = FunctionType::get(Type::getVoidTy(newcalled->getContext()),
                                      MetaTypes, false);
#if LLVM_VERSION_MAJOR >= 10
          Function *F =
              Function::Create(FT, GlobalVariable::InternalLinkage,
                               cast<Function>(newcalled)->getName() + "#out",
                               *task->getParent());
#else
          Function *F = Function::Create(
              FT, GlobalVariable::InternalLinkage,
              cast<Function>(newcalled)->getName() + "#out", task->getParent());
#endif
          BasicBlock *entry =
              BasicBlock::Create(newcalled->getContext(), "entry", F);
          IRBuilder<> B(entry);
          SmallVector<Value *, 2> SubArgs;
          for (auto &arg : F->args())
            SubArgs.push_back(&arg);
          Value *cacheArg = SubArgs.back();
          SubArgs.pop_back();
          Value *outdiff = B.CreateCall(newcalled, SubArgs);
          for (size_t ee = 0; ee < OutTypes.size(); ee++) {
            Value *dif = B.CreateExtractValue(outdiff, ee);
            Value *Idxs[] = {
                ConstantInt::get(Type::getInt64Ty(ST->getContext()), 0),
                ConstantInt::get(Type::getInt32Ty(ST->getContext()), ee)};
#if LLVM_VERSION_MAJOR > 7
            Value *ptr = B.CreateInBoundsGEP(
                cacheArg->getType()->getPointerElementType(), cacheArg, Idxs);
#else
            Value *ptr = B.CreateInBoundsGEP(cacheArg, Idxs);
#endif

            if (dif->getType()->isIntOrIntVectorTy()) {

              ptr = B.CreateBitCast(
                  ptr,
                  PointerType::get(
                      IntToFloatTy(dif->getType()),
                      cast<PointerType>(ptr->getType())->getAddressSpace()));
              dif = B.CreateBitCast(dif, IntToFloatTy(dif->getType()));
            }

#if LLVM_VERSION_MAJOR >= 10
            MaybeAlign align;
#elif LLVM_VERSION_MAJOR >= 9
            unsigned align = 0;
#endif

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
                auto vdif = B.CreateExtractElement(dif, i);
                Value *Idxs[] = {
                    ConstantInt::get(Type::getInt64Ty(vt->getContext()), 0),
                    ConstantInt::get(Type::getInt32Ty(vt->getContext()), i)};
#if LLVM_VERSION_MAJOR > 7
                auto vptr = B.CreateInBoundsGEP(
                    ptr->getType()->getPointerElementType(), ptr, Idxs);
#else
                auto vptr = B.CreateInBoundsGEP(ptr, Idxs);
#endif
#if LLVM_VERSION_MAJOR >= 13
                B.CreateAtomicRMW(op, vptr, vdif, align,
                                  AtomicOrdering::Monotonic, SyncScope::System);
#elif LLVM_VERSION_MAJOR >= 11
                AtomicRMWInst *rmw =
                    B.CreateAtomicRMW(op, vptr, vdif, AtomicOrdering::Monotonic,
                                      SyncScope::System);
                if (align)
                  rmw->setAlignment(align.getValue());
#else
                B.CreateAtomicRMW(op, vptr, vdif, AtomicOrdering::Monotonic,
                                  SyncScope::System);
#endif
              }
            } else {
#if LLVM_VERSION_MAJOR >= 13
              B.CreateAtomicRMW(op, ptr, dif, align, AtomicOrdering::Monotonic,
                                SyncScope::System);
#elif LLVM_VERSION_MAJOR >= 11
              AtomicRMWInst *rmw = B.CreateAtomicRMW(
                  op, ptr, dif, AtomicOrdering::Monotonic, SyncScope::System);
              if (align)
                rmw->setAlignment(align.getValue());
#else
              B.CreateAtomicRMW(op, ptr, dif, AtomicOrdering::Monotonic,
                                SyncScope::System);
#endif
            }
#else
            llvm::errs() << "unhandled atomic fadd on llvm version " << *ptr
                         << " " << *dif << "\n";
            llvm_unreachable("unhandled atomic fadd");
#endif
          }
          B.CreateRetVoid();
          newcalled = F;
        }

        auto numargs = ConstantInt::get(Type::getInt32Ty(call.getContext()),
                                        args.size() - 3);
        args[0] =
            lookup(gutils->getNewFromOriginal(call.getArgOperand(0)), Builder2);
        args[1] = numargs;
        args[2] = Builder2.CreatePointerCast(
            newcalled, kmpc->getFunctionType()->getParamType(2));

        CallInst *diffes =
            Builder2.CreateCall(kmpc->getFunctionType(), kmpc, args);
        diffes->setCallingConv(call.getCallingConv());
        diffes->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));

        for (size_t i = 0; i < OutTypes.size(); i++) {

          size_t size = 1;
          if (OutTypes[i]->getType()->isSized())
            size = (gutils->newFunc->getParent()
                        ->getDataLayout()
                        .getTypeSizeInBits(OutTypes[i]->getType()) +
                    7) /
                   8;
          Value *Idxs[] = {
              ConstantInt::get(Type::getInt64Ty(call.getContext()), 0),
              ConstantInt::get(Type::getInt32Ty(call.getContext()), i)};
          ((DiffeGradientUtils *)gutils)
              ->addToDiffe(OutTypes[i],
#if LLVM_VERSION_MAJOR > 7
                           Builder2.CreateLoad(
                               OutFPTypes[i],
                               Builder2.CreateInBoundsGEP(ST, OutAlloc, Idxs)),
#else
                           Builder2.CreateLoad(
                               Builder2.CreateInBoundsGEP(OutAlloc, Idxs)),
#endif
                           Builder2, TR.addingType(size, OutTypes[i]));
        }

        if (tape && shouldFree()) {
          for (auto idx : subdata->tapeIndiciesToFree) {
            auto ci = cast<CallInst>(CallInst::CreateFree(
                Builder2.CreatePointerCast(
                    idx == -1 ? tape : Builder2.CreateExtractValue(tape, idx),
                    Type::getInt8PtrTy(Builder2.getContext())),
                Builder2.GetInsertBlock()));
#if LLVM_VERSION_MAJOR >= 14
            ci->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                    Attribute::NonNull);
#else
            ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
#endif
            if (ci->getParent() == nullptr) {
              Builder2.Insert(ci);
            }
          }
        }
      } else {
        assert(0 && "openmp indirect unhandled");
      }
    }
  }

  void DifferentiableMemCopyFloats(CallInst &call, Value *origArg, Value *dsto,
                                   Value *srco, Value *len_arg,
                                   IRBuilder<> &Builder2,
                                   ArrayRef<OperandBundleDef> ReverseDefs) {
    size_t size = 1;
    if (auto ci = dyn_cast<ConstantInt>(len_arg)) {
      size = ci->getLimitedValue();
    }
    auto &DL = gutils->newFunc->getParent()->getDataLayout();
    auto vd = TR.query(origArg).Data0().ShiftIndices(DL, 0, size, 0);
    if (!vd.isKnownPastPointer()) {
      if (looseTypeAnalysis) {
        if (isa<CastInst>(origArg) &&
            cast<CastInst>(origArg)->getSrcTy()->isPointerTy() &&
            cast<CastInst>(origArg)
                ->getSrcTy()
                ->getPointerElementType()
                ->isFPOrFPVectorTy()) {
          vd = TypeTree(ConcreteType(cast<CastInst>(origArg)
                                         ->getSrcTy()
                                         ->getPointerElementType()
                                         ->getScalarType()))
                   .Only(0);
          goto knownF;
        }
      }
      EmitFailure("CannotDeduceType", call.getDebugLoc(), &call,
                  "failed to deduce type of copy ", call);

      TR.firstPointer(size, origArg, /*errifnotfound*/ true,
                      /*pointerIntSame*/ true);
      llvm_unreachable("bad mti");
    }
  knownF:;
    unsigned start = 0;
    while (1) {
      unsigned nextStart = size;

      auto dt = vd[{-1}];
      for (size_t i = start; i < size; ++i) {
        bool Legal = true;
        dt.checkedOrIn(vd[{(int)i}], /*PointerIntSame*/ true, Legal);
        if (!Legal) {
          nextStart = i;
          break;
        }
      }
      if (!dt.isKnown()) {
        TR.dump();
        llvm::errs() << " vd:" << vd.str() << " start:" << start
                     << " size: " << size << " dt:" << dt.str() << "\n";
      }
      assert(dt.isKnown());

      Value *length = len_arg;
      if (nextStart != size) {
        length = ConstantInt::get(len_arg->getType(), nextStart);
      }
      if (start != 0)
        length = Builder2.CreateSub(
            length, ConstantInt::get(len_arg->getType(), start));

      if (auto secretty = dt.isFloat()) {
        auto offset = start;
        if (dsto->getType()->isIntegerTy())
          dsto = Builder2.CreateIntToPtr(
              dsto, Type::getInt8PtrTy(dsto->getContext()));
        unsigned dstaddr =
            cast<PointerType>(dsto->getType())->getAddressSpace();
        auto secretpt = PointerType::get(secretty, dstaddr);
        if (offset != 0) {
#if LLVM_VERSION_MAJOR > 7
          dsto = Builder2.CreateConstInBoundsGEP1_64(
              dsto->getType()->getPointerElementType(), dsto, offset);
#else
          dsto = Builder2.CreateConstInBoundsGEP1_64(dsto, offset);
#endif
        }
        if (srco->getType()->isIntegerTy())
          srco = Builder2.CreateIntToPtr(
              srco, Type::getInt8PtrTy(dsto->getContext()));
        unsigned srcaddr =
            cast<PointerType>(srco->getType())->getAddressSpace();
        secretpt = PointerType::get(secretty, srcaddr);

        if (offset != 0) {
#if LLVM_VERSION_MAJOR > 7
          srco = Builder2.CreateConstInBoundsGEP1_64(
              srco->getType()->getPointerElementType(), srco, offset);
#else
          srco = Builder2.CreateConstInBoundsGEP1_64(srco, offset);
#endif
        }
        Value *args[3] = {
            Builder2.CreatePointerCast(dsto, secretpt),
            Builder2.CreatePointerCast(srco, secretpt),
            Builder2.CreateUDiv(
                length,

                ConstantInt::get(length->getType(),
                                 Builder2.GetInsertBlock()
                                         ->getParent()
                                         ->getParent()
                                         ->getDataLayout()
                                         .getTypeAllocSizeInBits(secretty) /
                                     8))};

        auto dmemcpy = getOrInsertDifferentialFloatMemcpy(
            *Builder2.GetInsertBlock()->getParent()->getParent(), secretty,
            /*dstalign*/ 1, /*srcalign*/ 1, dstaddr, srcaddr);

        Builder2.CreateCall(dmemcpy, args, ReverseDefs);
      }

      if (nextStart == size)
        break;
      start = nextStart;
    }
  }

  std::string extractBLAS(StringRef in, std::string &prefix,
                          std::string &suffix) {
    std::string extractable[] = {"ddot", "sdot", "dnrm2", "snrm2"};
    std::string prefixes[] = {"", "cblas_", "cublas_"};
    std::string suffixes[] = {"", "_", "_64_"};
    for (auto ex : extractable) {
      for (auto p : prefixes) {
        for (auto s : suffixes) {
          if (in == p + ex + s) {
            prefix = p;
            suffix = s;
            return ex;
          }
        }
      }
    }
    return "";
  }

  bool handleBLAS(llvm::CallInst &call, Function *called, StringRef funcName,
                  StringRef prefix, StringRef suffix,
                  const std::map<Argument *, bool> &uncacheable_args) {
    CallInst *const newCall = cast<CallInst>(gutils->getNewFromOriginal(&call));
    IRBuilder<> BuilderZ(newCall);
    BuilderZ.setFastMathFlags(getFast());
    IRBuilder<> allocationBuilder(gutils->inversionAllocs);
    allocationBuilder.setFastMathFlags(getFast());

    if (funcName == "dnrm2" || funcName == "snrm2") {
      if (!gutils->isConstantInstruction(&call)) {

        Type *innerType;
        std::string dfuncName;
        if (funcName == "dnrm2") {
          innerType = Type::getDoubleTy(call.getContext());
          dfuncName = (prefix + "ddot" + suffix).str();
        } else if (funcName == "snrm2") {
          innerType = Type::getFloatTy(call.getContext());
          dfuncName = (prefix + "sdot" + suffix).str();
        } else {
          assert(false && "Unreachable");
        }

        IntegerType *intType =
            dyn_cast<IntegerType>(call.getOperand(0)->getType());
        bool byRef = false;
        if (!intType) {
          auto PT = cast<PointerType>(call.getOperand(0)->getType());
          if (suffix.contains("64"))
            intType = IntegerType::get(PT->getContext(), 64);
          else
            intType = IntegerType::get(PT->getContext(), 32);
          byRef = true;
        }

        // Non-forward Mode not handled yet
        if (Mode != DerivativeMode::ForwardMode) {
          return false;
        } else {
          auto in_arg = call.getCalledFunction()->arg_begin();
          Argument *n = in_arg;
          in_arg++;
          Argument *x = in_arg;
          in_arg++;
          Argument *xinc = in_arg;

          auto derivcall = gutils->oldFunc->getParent()->getOrInsertFunction(
              dfuncName, innerType, n->getType(), x->getType(), xinc->getType(),
              x->getType(), xinc->getType());

#if LLVM_VERSION_MAJOR >= 9
          if (auto F = dyn_cast<Function>(derivcall.getCallee()))
#else
          if (auto F = dyn_cast<Function>(derivcall))
#endif
          {
            F->addFnAttr(Attribute::ArgMemOnly);
            F->addFnAttr(Attribute::ReadOnly);
            if (byRef) {
              F->addParamAttr(0, Attribute::ReadOnly);
              F->addParamAttr(0, Attribute::NoCapture);
              F->addParamAttr(2, Attribute::ReadOnly);
              F->addParamAttr(2, Attribute::NoCapture);
              F->addParamAttr(4, Attribute::ReadOnly);
              F->addParamAttr(4, Attribute::NoCapture);
            }
            if (call.getArgOperand(1)->getType()->isPointerTy()) {
              F->addParamAttr(1, Attribute::ReadOnly);
              F->addParamAttr(1, Attribute::NoCapture);
              F->addParamAttr(3, Attribute::ReadOnly);
              F->addParamAttr(3, Attribute::NoCapture);
            }
          }

          if (!gutils->isConstantValue(&call)) {
            if (gutils->isConstantValue(call.getOperand(1))) {
              setDiffe(
                  &call,
                  Constant::getNullValue(gutils->getShadowType(call.getType())),
                  BuilderZ);
            } else {
              auto Defs = gutils->getInvertedBundles(
                  &call,
                  {ValueType::Primal, ValueType::Primal, ValueType::Primal},
                  BuilderZ, /*lookup*/ false);

#if LLVM_VERSION_MAJOR >= 11
              auto callval = call.getCalledOperand();
#else
              auto callval = call.getCalledValue();
#endif

              if (auto F = dyn_cast<Function>(callval)) {
                F->addFnAttr(Attribute::ArgMemOnly);
                F->addFnAttr(Attribute::ReadOnly);
                if (byRef) {
                  F->addParamAttr(0, Attribute::ReadOnly);
                  F->addParamAttr(0, Attribute::NoCapture);
                  F->addParamAttr(2, Attribute::ReadOnly);
                  F->addParamAttr(2, Attribute::NoCapture);
                }
                if (call.getArgOperand(1)->getType()->isPointerTy()) {
                  F->addParamAttr(1, Attribute::ReadOnly);
                  F->addParamAttr(1, Attribute::NoCapture);
                }
              }

              Value *args[] = {gutils->getNewFromOriginal(call.getOperand(0)),
                               gutils->getNewFromOriginal(call.getOperand(1)),
                               gutils->getNewFromOriginal(call.getOperand(2))};

#if LLVM_VERSION_MAJOR > 7
              auto norm = BuilderZ.CreateCall(call.getFunctionType(), callval,
                                              args, Defs);
#else
              auto norm = BuilderZ.CreateCall(callval, args, Defs);
#endif

              Value *dval = applyChainRule(
                  call.getType(), BuilderZ,
                  [&](Value *ip) {
                    Value *args1[] = {
                        gutils->getNewFromOriginal(call.getOperand(0)),
                        gutils->getNewFromOriginal(call.getOperand(1)),
                        gutils->getNewFromOriginal(call.getOperand(2)), ip,
                        gutils->getNewFromOriginal(call.getOperand(2))};
                    return BuilderZ.CreateFDiv(
                        BuilderZ.CreateCall(
                            derivcall, args1,
                            gutils->getInvertedBundles(
                                &call,
                                {ValueType::Primal, ValueType::Both,
                                 ValueType::Primal},
                                BuilderZ, /*lookup*/ false)),
                        norm);
                  },
                  gutils->invertPointerM(call.getOperand(1), BuilderZ));
              setDiffe(&call, dval, BuilderZ);
            }
          }
        }

        if (gutils->knownRecomputeHeuristic.find(&call) !=
            gutils->knownRecomputeHeuristic.end()) {
          if (!gutils->knownRecomputeHeuristic[&call]) {
            gutils->cacheForReverse(BuilderZ, newCall,
                                    getIndex(&call, CacheType::Self));
          }
        }
      }

      if (Mode == DerivativeMode::ReverseModeGradient) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      } else {
        eraseIfUnused(call);
      }
      return true;
    }
    if (funcName == "ddot" || funcName == "sdot") {
      if (!gutils->isConstantInstruction(&call)) {
        Type *innerType;
        std::string dfuncName;
        if (funcName == "ddot") {
          innerType = Type::getDoubleTy(call.getContext());
          dfuncName = (prefix + "daxpy" + suffix).str();
        } else if (funcName == "sdot") {
          innerType = Type::getFloatTy(call.getContext());
          dfuncName = (prefix + "saxpy" + suffix).str();
        } else {
          assert(false && "Unreachable");
        }

        Type *castvals[2];
        if (auto PT = dyn_cast<PointerType>(call.getArgOperand(1)->getType()))
          castvals[0] = PT;
        else
          castvals[0] = PointerType::getUnqual(innerType);
        if (auto PT = dyn_cast<PointerType>(call.getArgOperand(3)->getType()))
          castvals[1] = PT;
        else
          castvals[1] = PointerType::getUnqual(innerType);

        IntegerType *intType =
            dyn_cast<IntegerType>(call.getOperand(0)->getType());
        bool byRef = false;
        if (!intType) {
          auto PT = cast<PointerType>(call.getOperand(0)->getType());
          if (suffix.contains("64"))
            intType = IntegerType::get(PT->getContext(), 64);
          else
            intType = IntegerType::get(PT->getContext(), 32);
          byRef = true;
        }

        auto &DL = gutils->oldFunc->getParent()->getDataLayout();

        Value *cacheval;
        auto in_arg = call.getCalledFunction()->arg_begin();
        Argument *countarg = in_arg;
        in_arg++;
        Argument *xfuncarg = in_arg;
        in_arg++;
        Argument *xincarg = in_arg;
        in_arg++;
        Argument *yfuncarg = in_arg;
        in_arg++;
        Argument *yincarg = in_arg;

        bool xcache = !gutils->isConstantValue(call.getArgOperand(3)) &&
                      Mode != DerivativeMode::ForwardMode &&
                      uncacheable_args.find(xfuncarg)->second;
        bool ycache = !gutils->isConstantValue(call.getArgOperand(1)) &&
                      Mode != DerivativeMode::ForwardMode &&
                      uncacheable_args.find(yfuncarg)->second;

        bool countcache = false;
        bool xinccache = false;
        bool yinccache = false;

        SmallVector<Type *, 2> cacheTypes;
        if (byRef) {
          // count must be preserved if overwritten
          if (uncacheable_args.find(countarg)->second) {
            cacheTypes.push_back(intType);
            countcache = true;
          }
          // xinc is needed to be preserved if
          // 1) it is potentially overwritten
          //       AND EITHER
          //     a) x is active (for performing the shadow increment) or
          //     b) we're not caching x and need xinc to compute the derivative
          //        of y
          if (uncacheable_args.find(xincarg)->second &&
              (!gutils->isConstantValue(call.getArgOperand(1)) ||
               (!xcache && !gutils->isConstantValue(call.getArgOperand(3))))) {
            cacheTypes.push_back(intType);
            xinccache = true;
          }
          // Similarly for yinc
          if (uncacheable_args.find(yincarg)->second &&
              (!gutils->isConstantValue(call.getArgOperand(3)) ||
               (!ycache && !gutils->isConstantValue(call.getArgOperand(1))))) {
            cacheTypes.push_back(intType);
            yinccache = true;
          }
        }

        if (xcache)
          cacheTypes.push_back(castvals[0]);

        if (ycache)
          cacheTypes.push_back(castvals[1]);

        Type *cachetype = nullptr;
        switch (cacheTypes.size()) {
        case 0:
          break;
        case 1:
          cachetype = cacheTypes[0];
          break;
        default:
          cachetype = StructType::get(call.getContext(), cacheTypes);
          break;
        }

        if ((Mode == DerivativeMode::ReverseModeCombined ||
             Mode == DerivativeMode::ReverseModePrimal) &&
            cachetype) {

          SmallVector<Value *, 2> cacheValues;
          auto size =
              ConstantInt::get(intType, DL.getTypeSizeInBits(innerType) / 8);

          Value *count = gutils->getNewFromOriginal(call.getArgOperand(0));

          if (byRef) {
            count = BuilderZ.CreatePointerCast(count,
                                               PointerType::getUnqual(intType));
#if LLVM_VERSION_MAJOR > 7
            count = BuilderZ.CreateLoad(intType, count);
#else
            count = BuilderZ.CreateLoad(count);
#endif
            if (countcache)
              cacheValues.push_back(count);
          }

          Value *xinc = gutils->getNewFromOriginal(call.getArgOperand(2));
          if (byRef) {
            xinc = BuilderZ.CreatePointerCast(xinc,
                                              PointerType::getUnqual(intType));
#if LLVM_VERSION_MAJOR > 7
            xinc = BuilderZ.CreateLoad(intType, xinc);
#else
            xinc = BuilderZ.CreateLoad(xinc);
#endif
            if (xinccache)
              cacheValues.push_back(xinc);
          }

          Value *yinc = gutils->getNewFromOriginal(call.getArgOperand(4));
          if (byRef) {
            yinc = BuilderZ.CreatePointerCast(yinc,
                                              PointerType::getUnqual(intType));
#if LLVM_VERSION_MAJOR > 7
            yinc = BuilderZ.CreateLoad(intType, yinc);
#else
            yinc = BuilderZ.CreateLoad(yinc);
#endif
            if (yinccache)
              cacheValues.push_back(yinc);
          }

          if (xcache) {
            auto dmemcpy = getOrInsertMemcpyStrided(
                *gutils->oldFunc->getParent(), cast<PointerType>(castvals[0]),
                size->getType(), 0, 0);
            auto malins = CallInst::CreateMalloc(
                gutils->getNewFromOriginal(&call), size->getType(), innerType,
                size, count, nullptr, "");
            Value *arg = BuilderZ.CreateBitCast(malins, castvals[0]);
            Value *args[4] = {arg,
                              gutils->getNewFromOriginal(call.getArgOperand(1)),
                              count, xinc};

            if (args[1]->getType()->isIntegerTy())
              args[1] = BuilderZ.CreateIntToPtr(args[1], castvals[0]);

            BuilderZ.CreateCall(
                dmemcpy, args,
                gutils->getInvertedBundles(&call,
                                           {ValueType::None, ValueType::Shadow,
                                            ValueType::None, ValueType::None,
                                            ValueType::None},
                                           BuilderZ, /*lookup*/ false));
            cacheValues.push_back(arg);
          }

          if (ycache) {
            auto dmemcpy = getOrInsertMemcpyStrided(
                *gutils->oldFunc->getParent(), cast<PointerType>(castvals[1]),
                size->getType(), 0, 0);
            auto malins = CallInst::CreateMalloc(
                gutils->getNewFromOriginal(&call), size->getType(), innerType,
                size, count, nullptr, "");
            Value *arg = BuilderZ.CreateBitCast(malins, castvals[1]);
            Value *args[4] = {arg,
                              gutils->getNewFromOriginal(call.getArgOperand(3)),
                              count, yinc};

            if (args[1]->getType()->isIntegerTy())
              args[1] = BuilderZ.CreateIntToPtr(args[1], castvals[1]);

            BuilderZ.CreateCall(
                dmemcpy, args,
                gutils->getInvertedBundles(&call,
                                           {ValueType::None, ValueType::None,
                                            ValueType::None, ValueType::Shadow,
                                            ValueType::None},
                                           BuilderZ, /*lookup*/ false));
            cacheValues.push_back(arg);
          }

          if (cacheValues.size() == 1)
            cacheval = cacheValues[0];
          else {
            cacheval = UndefValue::get(cachetype);
            for (auto tup : llvm::enumerate(cacheValues))
              cacheval = BuilderZ.CreateInsertValue(cacheval, tup.value(),
                                                    tup.index());
          }
          gutils->cacheForReverse(BuilderZ, cacheval,
                                  getIndex(&call, CacheType::Tape));
        }

        unsigned cacheidx = 0;
        Value *count = gutils->getNewFromOriginal(call.getArgOperand(0));
        Value *trueXinc = gutils->getNewFromOriginal(call.getArgOperand(2));
        Value *trueYinc = gutils->getNewFromOriginal(call.getArgOperand(4));

        IRBuilder<> Builder2(call.getParent());
        switch (Mode) {
        case DerivativeMode::ReverseModeCombined:
        case DerivativeMode::ReverseModeGradient:
          getReverseBuilder(Builder2);
          break;
        case DerivativeMode::ForwardMode:
        case DerivativeMode::ForwardModeSplit:
          Builder2.SetInsertPoint(BuilderZ.GetInsertBlock(),
                                  BuilderZ.GetInsertPoint());
          Builder2.setFastMathFlags(BuilderZ.getFastMathFlags());
          break;
        case DerivativeMode::ReverseModePrimal:
          break;
        }

        if (Mode == DerivativeMode::ReverseModeCombined ||
            Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ForwardModeSplit) {

          if (cachetype) {
            if (Mode != DerivativeMode::ReverseModeCombined) {
              cacheval = BuilderZ.CreatePHI(cachetype, 0);
            }
            cacheval = gutils->cacheForReverse(
                BuilderZ, cacheval, getIndex(&call, CacheType::Tape));
            if (Mode != DerivativeMode::ForwardModeSplit)
              cacheval = lookup(cacheval, Builder2);
          }

          if (byRef) {
            if (countcache) {
              count = (cacheTypes.size() == 1)
                          ? cacheval
                          : Builder2.CreateExtractValue(cacheval, {cacheidx});
              auto alloc = allocationBuilder.CreateAlloca(intType);
              Builder2.CreateStore(count, alloc);
              count = Builder2.CreatePointerCast(
                  alloc, call.getArgOperand(0)->getType());
              cacheidx++;
            } else {
              if (Mode != DerivativeMode::ForwardModeSplit)
                count = lookup(count, Builder2);
            }

            if (xinccache) {
              trueXinc =
                  (cacheTypes.size() == 1)
                      ? cacheval
                      : Builder2.CreateExtractValue(cacheval, {cacheidx});
              auto alloc = allocationBuilder.CreateAlloca(intType);
              Builder2.CreateStore(trueXinc, alloc);
              trueXinc = Builder2.CreatePointerCast(
                  alloc, call.getArgOperand(0)->getType());
              cacheidx++;
            } else if (!gutils->isConstantValue(call.getArgOperand(1)) ||
                       (!xcache &&
                        !gutils->isConstantValue(call.getArgOperand(3)))) {
              if (Mode != DerivativeMode::ForwardModeSplit)
                trueXinc = lookup(trueXinc, Builder2);
            }

            if (yinccache) {
              trueYinc =
                  (cacheTypes.size() == 1)
                      ? cacheval
                      : Builder2.CreateExtractValue(cacheval, {cacheidx});
              auto alloc = allocationBuilder.CreateAlloca(intType);
              Builder2.CreateStore(trueYinc, alloc);
              trueYinc = Builder2.CreatePointerCast(
                  alloc, call.getArgOperand(0)->getType());
              cacheidx++;
            } else if (!gutils->isConstantValue(call.getArgOperand(3)) ||
                       (!ycache &&
                        !gutils->isConstantValue(call.getArgOperand(1)))) {
              if (Mode != DerivativeMode::ForwardModeSplit)
                trueXinc = lookup(trueXinc, Builder2);
            }
          } else if (Mode != DerivativeMode::ForwardModeSplit) {
            count = lookup(count, Builder2);

            if (!gutils->isConstantValue(call.getArgOperand(1)) ||
                (!xcache && !gutils->isConstantValue(call.getArgOperand(3))))
              trueXinc = lookup(trueXinc, Builder2);

            if (!gutils->isConstantValue(call.getArgOperand(3)) ||
                (!ycache && !gutils->isConstantValue(call.getArgOperand(1))))
              trueYinc = lookup(trueYinc, Builder2);
          }
        }

        Value *xdata = gutils->getNewFromOriginal(call.getArgOperand(1));
        Value *xdata_ptr = nullptr;
        Value *xinc = trueXinc;

        Value *ydata = gutils->getNewFromOriginal(call.getArgOperand(3));
        Value *ydata_ptr = nullptr;
        Value *yinc = trueYinc;

        if (Mode == DerivativeMode::ReverseModeCombined ||
            Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ForwardModeSplit) {

          if (xcache) {
            xdata_ptr = xdata =
                (cacheTypes.size() == 1)
                    ? cacheval
                    : Builder2.CreateExtractValue(cacheval, {cacheidx});
            cacheidx++;
            xinc = ConstantInt::get(intType, 1);
            if (byRef) {
              auto alloc = allocationBuilder.CreateAlloca(intType);
              Builder2.CreateStore(xinc, alloc);
              xinc = Builder2.CreatePointerCast(
                  alloc, call.getArgOperand(0)->getType());
            }
            if (call.getArgOperand(1)->getType()->isIntegerTy())
              xdata = Builder2.CreatePtrToInt(xdata,
                                              call.getArgOperand(1)->getType());
          } else if (!gutils->isConstantValue(call.getArgOperand(3))) {
            xdata = lookup(gutils->getNewFromOriginal(call.getArgOperand(1)),
                           Builder2);
          }
          if (ycache) {
            ydata_ptr = ydata =
                (cacheTypes.size() == 1)
                    ? cacheval
                    : Builder2.CreateExtractValue(cacheval, {cacheidx});
            cacheidx++;
            yinc = ConstantInt::get(intType, 1);
            if (byRef) {
              auto alloc = allocationBuilder.CreateAlloca(intType);
              Builder2.CreateStore(yinc, alloc);
              yinc = Builder2.CreatePointerCast(
                  alloc, call.getArgOperand(0)->getType());
            }
            if (call.getArgOperand(3)->getType()->isIntegerTy())
              ydata = Builder2.CreatePtrToInt(ydata,
                                              call.getArgOperand(3)->getType());
          } else if (!gutils->isConstantValue(call.getArgOperand(1))) {
            ydata = lookup(gutils->getNewFromOriginal(call.getArgOperand(3)),
                           Builder2);
          }
        } else {
          if (call.getArgOperand(1)->getType()->isIntegerTy())
            xdata = Builder2.CreatePtrToInt(xdata,
                                            call.getArgOperand(1)->getType());
          if (call.getArgOperand(3)->getType()->isIntegerTy())
            ydata = Builder2.CreatePtrToInt(ydata,
                                            call.getArgOperand(3)->getType());
        }

        if (Mode == DerivativeMode::ForwardMode ||
            Mode == DerivativeMode::ForwardModeSplit) {

#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif

          Value *dx =
              gutils->isConstantValue(call.getArgOperand(1))
                  ? nullptr
                  : gutils->invertPointerM(call.getArgOperand(1), Builder2);
          Value *dy =
              gutils->isConstantValue(call.getArgOperand(3))
                  ? nullptr
                  : gutils->invertPointerM(call.getArgOperand(3), Builder2);

          Value *dres = applyChainRule(
              call.getType(), Builder2,
              [&](Value *dx, Value *dy) {
                Value *dres = nullptr;
                if (!gutils->isConstantValue(call.getArgOperand(3))) {
                  Value *args1[] = {count, xdata, xinc, dy, trueYinc};

                  auto Defs = gutils->getInvertedBundles(
                      &call,
                      {ValueType::None,
                       xcache ? ValueType::None : ValueType::Primal,
                       ValueType::None, ValueType::Shadow, ValueType::None},
                      Builder2, /*lookup*/ false);

#if LLVM_VERSION_MAJOR > 7
                  dres = Builder2.CreateCall(call.getFunctionType(), callval,
                                             args1, Defs);
#else
                  dres = Builder2.CreateCall(callval, args1, Defs);
#endif
                }
                if (!gutils->isConstantValue(call.getArgOperand(1))) {
                  Value *args1[] = {count, ydata, yinc, dx, trueXinc};

                  auto Defs = gutils->getInvertedBundles(
                      &call,
                      {ValueType::None, ValueType::Shadow, ValueType::None,
                       ycache ? ValueType::None : ValueType::Primal,
                       ValueType::None},
                      Builder2, /*lookup*/ false);

#if LLVM_VERSION_MAJOR > 7
                  Value *secondcall = Builder2.CreateCall(
                      call.getFunctionType(), callval, args1, Defs);
#else
                  Value *secondcall = Builder2.CreateCall(callval, args1, Defs);
#endif
                  if (dres)
                    dres = Builder2.CreateFAdd(dres, secondcall);
                  else
                    dres = secondcall;
                }
                return dres;
              },
              dx, dy);
          setDiffe(&call, dres, Builder2);
        }

        if (Mode == DerivativeMode::ReverseModeCombined ||
            Mode == DerivativeMode::ReverseModeGradient) {
          Value *dif = diffe(&call, Builder2);
          Value *alloc = nullptr;
          if (byRef) {
            alloc = allocationBuilder.CreateAlloca(innerType);
          }
          auto derivcall = gutils->oldFunc->getParent()->getOrInsertFunction(
              dfuncName, Builder2.getVoidTy(), call.getArgOperand(0)->getType(),
              byRef ? PointerType::getUnqual(call.getType()) : call.getType(),
              call.getArgOperand(1)->getType(),
              call.getArgOperand(0)->getType(),
              call.getArgOperand(3)->getType(),
              call.getArgOperand(0)->getType());
#if LLVM_VERSION_MAJOR >= 9
          if (auto F = dyn_cast<Function>(derivcall.getCallee()))
#else
          if (auto F = dyn_cast<Function>(derivcall))
#endif
          {
            F->addFnAttr(Attribute::ArgMemOnly);
            if (byRef) {
              F->addParamAttr(0, Attribute::ReadOnly);
              F->addParamAttr(0, Attribute::NoCapture);
              F->addParamAttr(3, Attribute::ReadOnly);
              F->addParamAttr(3, Attribute::NoCapture);
              F->addParamAttr(5, Attribute::ReadOnly);
              F->addParamAttr(5, Attribute::NoCapture);
            }
            if (call.getArgOperand(1)->getType()->isPointerTy()) {
              F->addParamAttr(2, Attribute::ReadOnly);
              F->addParamAttr(2, Attribute::NoCapture);
            }
            if (call.getArgOperand(3)->getType()->isPointerTy()) {
              F->addParamAttr(4, Attribute::NoCapture);
            }
          }

          // Vector Mode not handled yet
          Value *dx = gutils->isConstantValue(call.getArgOperand(1))
                          ? nullptr
                          : lookup(gutils->invertPointerM(call.getArgOperand(1),
                                                          Builder2),
                                   Builder2);
          Value *dy = gutils->isConstantValue(call.getArgOperand(3))
                          ? nullptr
                          : lookup(gutils->invertPointerM(call.getArgOperand(3),
                                                          Builder2),
                                   Builder2);

          applyChainRule(
              Builder2,
              [&](Value *dx, Value *dy, Value *dif) {
                if (byRef) {
                  Builder2.CreateStore(dif, alloc);
                  dif = alloc;
                }
                if (!gutils->isConstantValue(call.getArgOperand(3))) {
                  Value *args1[6] = {count, dif, xdata, xinc, dy, trueYinc};
                  Builder2.CreateCall(
                      derivcall, args1,
                      gutils->getInvertedBundles(
                          &call,
                          {ValueType::None,
                           xcache ? ValueType::None : ValueType::Primal,
                           ValueType::None, ValueType::Shadow, ValueType::None},
                          Builder2, /*lookup*/ true));
                }
                if (!gutils->isConstantValue(call.getArgOperand(1))) {
                  Value *args2[6] = {count, dif, ydata, yinc, dx, trueXinc};
                  Builder2.CreateCall(
                      derivcall, args2,
                      gutils->getInvertedBundles(
                          &call,
                          {ValueType::None, ValueType::Shadow, ValueType::None,
                           ycache ? ValueType::None : ValueType::Primal,
                           ValueType::None},
                          Builder2, /*lookup*/ true));
                }
              },
              dx, dy, dif);

          setDiffe(
              &call,
              Constant::getNullValue(gutils->getShadowType(call.getType())),
              Builder2);
        }

        if (Mode == DerivativeMode::ReverseModeCombined ||
            Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ForwardModeSplit) {
          if (shouldFree()) {
            if (xcache) {
              auto ci = cast<CallInst>(CallInst::CreateFree(
                  Builder2.CreatePointerCast(
                      xdata_ptr, Type::getInt8PtrTy(xdata_ptr->getContext())),
                  Builder2.GetInsertBlock()));
#if LLVM_VERSION_MAJOR >= 14
              ci->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                      Attribute::NonNull);
#else
              ci->addAttribute(AttributeList::FirstArgIndex,
                               Attribute::NonNull);
#endif
              if (ci->getParent() == nullptr) {
                Builder2.Insert(ci);
              }
            }
            if (ycache) {
              auto ci = cast<CallInst>(CallInst::CreateFree(
                  Builder2.CreatePointerCast(
                      ydata_ptr, Type::getInt8PtrTy(ydata_ptr->getContext())),
                  Builder2.GetInsertBlock()));
#if LLVM_VERSION_MAJOR >= 14
              ci->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                      Attribute::NonNull);
#else
              ci->addAttribute(AttributeList::FirstArgIndex,
                               Attribute::NonNull);
#endif
              if (ci->getParent() == nullptr) {
                Builder2.Insert(ci);
              }
            }
          }
        }

        if (gutils->knownRecomputeHeuristic.find(&call) !=
            gutils->knownRecomputeHeuristic.end()) {
          if (!gutils->knownRecomputeHeuristic[&call]) {
            gutils->cacheForReverse(BuilderZ, newCall,
                                    getIndex(&call, CacheType::Self));
          }
        }
      }

      if (Mode == DerivativeMode::ReverseModeGradient) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      } else {
        eraseIfUnused(call);
      }
      return true;
    }
    llvm::errs() << " fallback?\n";
    return false;
  }

  void handleMPI(llvm::CallInst &call, Function *called, StringRef funcName) {
    assert(called);
    assert(gutils->getWidth() == 1);

    IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&call));
    BuilderZ.setFastMathFlags(getFast());

    // MPI send / recv can only send float/integers
    if (funcName == "PMPI_Isend" || funcName == "MPI_Isend" ||
        funcName == "PMPI_Irecv" || funcName == "MPI_Irecv") {
      if (!gutils->isConstantInstruction(&call)) {
        if (Mode == DerivativeMode::ReverseModePrimal ||
            Mode == DerivativeMode::ReverseModeCombined) {
          assert(!gutils->isConstantValue(call.getOperand(0)));
          assert(!gutils->isConstantValue(call.getOperand(6)));
          Value *d_req = gutils->invertPointerM(call.getOperand(6), BuilderZ);
          if (d_req->getType()->isIntegerTy()) {
            d_req = BuilderZ.CreateIntToPtr(
                d_req,
                PointerType::getUnqual(Type::getInt8PtrTy(call.getContext())));
          }

          auto i64 = Type::getInt64Ty(call.getContext());
          auto impi = getMPIHelper(call.getContext());

          Value *impialloc = CallInst::CreateMalloc(
              gutils->getNewFromOriginal(&call), i64, impi,
              ConstantInt::get(
                  i64,
                  called->getParent()->getDataLayout().getTypeAllocSizeInBits(
                      impi) /
                      8),
              nullptr, nullptr, "");
          BuilderZ.SetInsertPoint(gutils->getNewFromOriginal(&call));

          d_req = BuilderZ.CreateBitCast(
              d_req, PointerType::getUnqual(impialloc->getType()));
#if LLVM_VERSION_MAJOR > 7
          Value *d_req_prev = BuilderZ.CreateLoad(impialloc->getType(), d_req);
#else
          Value *d_req_prev = BuilderZ.CreateLoad(d_req);
#endif
          BuilderZ.CreateStore(
              BuilderZ.CreatePointerCast(d_req_prev,
                                         Type::getInt8PtrTy(call.getContext())),
              getMPIMemberPtr<MPI_Elem::Old>(BuilderZ, impialloc));
          BuilderZ.CreateStore(impialloc, d_req);

          if (funcName == "MPI_Isend" || funcName == "PMPI_Isend") {
            Value *tysize =
                MPI_TYPE_SIZE(gutils->getNewFromOriginal(call.getOperand(2)),
                              BuilderZ, call.getType());

            auto len_arg = BuilderZ.CreateZExtOrTrunc(
                gutils->getNewFromOriginal(call.getOperand(1)),
                Type::getInt64Ty(call.getContext()));
            len_arg = BuilderZ.CreateMul(
                len_arg,
                BuilderZ.CreateZExtOrTrunc(tysize,
                                           Type::getInt64Ty(call.getContext())),
                "", true, true);

            Value *firstallocation = CallInst::CreateMalloc(
                &*BuilderZ.GetInsertPoint(), len_arg->getType(),
                Type::getInt8Ty(call.getContext()),
                ConstantInt::get(Type::getInt64Ty(len_arg->getContext()), 1),
                len_arg, nullptr, "mpirecv_malloccache");
            BuilderZ.CreateStore(
                firstallocation,
                getMPIMemberPtr<MPI_Elem::Buf>(BuilderZ, impialloc));
            BuilderZ.SetInsertPoint(gutils->getNewFromOriginal(&call));
          } else {
            Value *ibuf = gutils->invertPointerM(call.getOperand(0), BuilderZ);
            if (ibuf->getType()->isIntegerTy())
              ibuf = BuilderZ.CreateIntToPtr(
                  ibuf, Type::getInt8PtrTy(call.getContext()));
            BuilderZ.CreateStore(
                ibuf, getMPIMemberPtr<MPI_Elem::Buf>(BuilderZ, impialloc));
          }

          BuilderZ.CreateStore(
              BuilderZ.CreateZExtOrTrunc(
                  gutils->getNewFromOriginal(call.getOperand(1)), i64),
              getMPIMemberPtr<MPI_Elem::Count>(BuilderZ, impialloc));

          Value *dataType = gutils->getNewFromOriginal(call.getOperand(2));
          if (dataType->getType()->isIntegerTy())
            dataType = BuilderZ.CreateIntToPtr(
                dataType, Type::getInt8PtrTy(dataType->getContext()));
          BuilderZ.CreateStore(
              BuilderZ.CreatePointerCast(dataType,
                                         Type::getInt8PtrTy(call.getContext())),
              getMPIMemberPtr<MPI_Elem::DataType>(BuilderZ, impialloc));

          BuilderZ.CreateStore(
              BuilderZ.CreateZExtOrTrunc(
                  gutils->getNewFromOriginal(call.getOperand(3)), i64),
              getMPIMemberPtr<MPI_Elem::Src>(BuilderZ, impialloc));

          BuilderZ.CreateStore(
              BuilderZ.CreateZExtOrTrunc(
                  gutils->getNewFromOriginal(call.getOperand(4)), i64),
              getMPIMemberPtr<MPI_Elem::Tag>(BuilderZ, impialloc));

          Value *comm = gutils->getNewFromOriginal(call.getOperand(5));
          if (comm->getType()->isIntegerTy())
            comm = BuilderZ.CreateIntToPtr(
                comm, Type::getInt8PtrTy(dataType->getContext()));
          BuilderZ.CreateStore(
              BuilderZ.CreatePointerCast(comm,
                                         Type::getInt8PtrTy(call.getContext())),
              getMPIMemberPtr<MPI_Elem::Comm>(BuilderZ, impialloc));

          BuilderZ.CreateStore(
              ConstantInt::get(
                  Type::getInt8Ty(impialloc->getContext()),
                  (funcName == "MPI_Isend" || funcName == "PMPI_Isend")
                      ? (int)MPI_CallType::ISEND
                      : (int)MPI_CallType::IRECV),
              getMPIMemberPtr<MPI_Elem::Call>(BuilderZ, impialloc));
          // TODO old
        }
        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ReverseModeCombined) {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);

          Type *statusType = nullptr;

          if (Function *recvfn =
                  called->getParent()->getFunction("PMPI_Wait")) {
            auto statusArg = recvfn->arg_end();
            statusArg--;
            if (auto PT = dyn_cast<PointerType>(statusArg->getType()))
              statusType = PT->getPointerElementType();
          }
          if (Function *recvfn = called->getParent()->getFunction("MPI_Wait")) {
            auto statusArg = recvfn->arg_end();
            statusArg--;
            if (auto PT = dyn_cast<PointerType>(statusArg->getType()))
              statusType = PT->getPointerElementType();
          }
          if (statusType == nullptr) {
            statusType = ArrayType::get(Type::getInt8Ty(call.getContext()), 24);
            llvm::errs() << " warning could not automatically determine mpi "
                            "status type, assuming [24 x i8]\n";
          }
          Value *req =
              lookup(gutils->getNewFromOriginal(call.getOperand(6)), Builder2);
          Value *d_req = lookup(
              gutils->invertPointerM(call.getOperand(6), Builder2), Builder2);
          if (d_req->getType()->isIntegerTy()) {
            d_req = Builder2.CreateIntToPtr(
                d_req, Type::getInt8PtrTy(call.getContext()));
          }
          Type *helperTy =
              llvm::PointerType::getUnqual(getMPIHelper(call.getContext()));
          Value *helper = Builder2.CreatePointerCast(
              d_req, PointerType::getUnqual(helperTy));
#if LLVM_VERSION_MAJOR > 7
          helper = Builder2.CreateLoad(helperTy, helper);
#else
          helper = Builder2.CreateLoad(helper);
#endif

          auto i64 = Type::getInt64Ty(call.getContext());

          Value *firstallocation;
#if LLVM_VERSION_MAJOR > 7
          firstallocation = Builder2.CreateLoad(
              Type::getInt8PtrTy(call.getContext()),
              getMPIMemberPtr<MPI_Elem::Buf>(Builder2, helper));
#else
          firstallocation = Builder2.CreateLoad(
              getMPIMemberPtr<MPI_Elem::Buf>(Builder2, helper));
#endif
          Value *len_arg = nullptr;
          if (auto C = dyn_cast<Constant>(
                  gutils->getNewFromOriginal(call.getOperand(1)))) {
            len_arg = Builder2.CreateZExtOrTrunc(C, i64);
          } else {
#if LLVM_VERSION_MAJOR > 7
            len_arg = Builder2.CreateLoad(
                i64, getMPIMemberPtr<MPI_Elem::Count>(Builder2, helper));
#else
            len_arg = Builder2.CreateLoad(
                getMPIMemberPtr<MPI_Elem::Count>(Builder2, helper));
#endif
          }
          Value *tysize = nullptr;
          if (auto C = dyn_cast<Constant>(
                  gutils->getNewFromOriginal(call.getOperand(2)))) {
            tysize = C;
          } else {
#if LLVM_VERSION_MAJOR > 7
            tysize = Builder2.CreateLoad(
                Type::getInt8PtrTy(call.getContext()),
                getMPIMemberPtr<MPI_Elem::DataType>(Builder2, helper));
#else
            tysize = Builder2.CreateLoad(
                getMPIMemberPtr<MPI_Elem::DataType>(Builder2, helper));
#endif
          }

          Value *prev;
#if LLVM_VERSION_MAJOR > 7
          prev = Builder2.CreateLoad(
              Type::getInt8PtrTy(call.getContext()),
              getMPIMemberPtr<MPI_Elem::Old>(Builder2, helper));
#else
          prev = Builder2.CreateLoad(
              getMPIMemberPtr<MPI_Elem::Old>(Builder2, helper));
#endif
          Builder2.CreateStore(
              prev, Builder2.CreatePointerCast(
                        d_req, PointerType::getUnqual(prev->getType())));

          assert(shouldFree());

          assert(tysize);
          tysize = MPI_TYPE_SIZE(tysize, Builder2, call.getType());

          Value *args[] = {/*req*/ req,
                           /*status*/ IRBuilder<>(gutils->inversionAllocs)
                               .CreateAlloca(statusType)};
#if LLVM_VERSION_MAJOR >= 9
          FunctionCallee waitFunc = nullptr;
#else
          Constant *waitFunc = nullptr;
#endif
          for (auto name : {"PMPI_Wait", "MPI_Wait"})
            if (Function *recvfn = called->getParent()->getFunction(name)) {
              auto statusArg = recvfn->arg_end();
              statusArg--;
              if (statusArg->getType()->isIntegerTy())
                args[1] =
                    Builder2.CreatePtrToInt(args[1], statusArg->getType());
              else
                args[1] = Builder2.CreateBitCast(args[1], statusArg->getType());
              waitFunc = recvfn;
              break;
            }
          if (!waitFunc) {
            Type *types[sizeof(args) / sizeof(*args)];
            for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
              types[i] = args[i]->getType();
            FunctionType *FT = FunctionType::get(call.getType(), types, false);
            waitFunc = called->getParent()->getOrInsertFunction("MPI_Wait", FT);
          }
          assert(waitFunc);

          // Need to preserve the shadow Request (operand 6 in isend/irecv),
          // which becomes operand 0 for iwait.
          auto ReqDefs = gutils->getInvertedBundles(
              &call,
              {ValueType::None, ValueType::None, ValueType::None,
               ValueType::None, ValueType::None, ValueType::None,
               ValueType::Shadow},
              Builder2, /*lookup*/ true);

          auto BufferDefs = gutils->getInvertedBundles(
              &call,
              {ValueType::Shadow, ValueType::None, ValueType::None,
               ValueType::None, ValueType::None, ValueType::None,
               ValueType::None},
              Builder2, /*lookup*/ true);

          auto fcall = Builder2.CreateCall(waitFunc, args, ReqDefs);
          fcall->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
#if LLVM_VERSION_MAJOR >= 9
          if (auto F = dyn_cast<Function>(waitFunc.getCallee()))
            fcall->setCallingConv(F->getCallingConv());
#else
          if (auto F = dyn_cast<Function>(waitFunc))
            fcall->setCallingConv(F->getCallingConv());
#endif
          len_arg = Builder2.CreateMul(
              len_arg,
              Builder2.CreateZExtOrTrunc(
                  tysize, Type::getInt64Ty(Builder2.getContext())),
              "", true, true);
          if (funcName == "MPI_Irecv" || funcName == "PMPI_Irecv") {
            auto val_arg =
                ConstantInt::get(Type::getInt8Ty(Builder2.getContext()), 0);
            auto volatile_arg = ConstantInt::getFalse(Builder2.getContext());
            assert(!gutils->isConstantValue(call.getOperand(0)));
            auto dbuf = firstallocation;
#if LLVM_VERSION_MAJOR == 6
            auto align_arg =
                ConstantInt::get(Type::getInt32Ty(B.getContext()), 1);
            Value *nargs[] = {dbuf, val_arg, len_arg, align_arg, volatile_arg};
#else
            Value *nargs[] = {dbuf, val_arg, len_arg, volatile_arg};
#endif

            Type *tys[] = {dbuf->getType(), len_arg->getType()};

            auto memset = cast<CallInst>(Builder2.CreateCall(
                Intrinsic::getDeclaration(called->getParent(),
                                          Intrinsic::memset, tys),
                nargs, BufferDefs));
            memset->addParamAttr(0, Attribute::NonNull);
          } else if (funcName == "MPI_Isend" || funcName == "PMPI_Isend") {
            assert(!gutils->isConstantValue(call.getOperand(0)));
            Value *shadow = lookup(
                gutils->invertPointerM(call.getOperand(0), Builder2), Builder2);

            // TODO add operand bundle (unless force inlined?)
            DifferentiableMemCopyFloats(call, call.getOperand(0),
                                        firstallocation, shadow, len_arg,
                                        Builder2, BufferDefs);

            if (shouldFree()) {
              auto ci = cast<CallInst>(CallInst::CreateFree(
                  firstallocation, Builder2.GetInsertBlock()));
#if LLVM_VERSION_MAJOR >= 14
              ci->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                      Attribute::NonNull);
#else
              ci->addAttribute(AttributeList::FirstArgIndex,
                               Attribute::NonNull);
#endif
              if (ci->getParent() == nullptr) {
                Builder2.Insert(ci);
              }
            }
          } else
            assert(0 && "illegal mpi");

          CallInst *freecall = cast<CallInst>(CallInst::CreateFree(
              Builder2.CreatePointerCast(helper,
                                         Type::getInt8PtrTy(call.getContext())),
              Builder2.GetInsertBlock()));
#if LLVM_VERSION_MAJOR >= 14
          freecall->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                        Attribute::NonNull);
#else
          freecall->addAttribute(AttributeList::FirstArgIndex,
                                 Attribute::NonNull);
#endif
          if (freecall->getParent() == nullptr) {
            Builder2.Insert(freecall);
          }
        }
        if (Mode == DerivativeMode::ForwardMode) {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);

          assert(!gutils->isConstantValue(call.getOperand(0)));
          assert(!gutils->isConstantValue(call.getOperand(6)));

          Value *buf = gutils->invertPointerM(call.getOperand(0), Builder2);
          Value *count = gutils->getNewFromOriginal(call.getOperand(1));
          Value *datatype = gutils->getNewFromOriginal(call.getOperand(2));
          Value *source = gutils->getNewFromOriginal(call.getOperand(3));
          Value *tag = gutils->getNewFromOriginal(call.getOperand(4));
          Value *comm = gutils->getNewFromOriginal(call.getOperand(5));
          Value *request = gutils->invertPointerM(call.getOperand(6), Builder2);

          Value *args[] = {
              /*buf*/ buf,
              /*count*/ count,
              /*datatype*/ datatype,
              /*source*/ source,
              /*tag*/ tag,
              /*comm*/ comm,
              /*request*/ request,
          };

          auto Defs = gutils->getInvertedBundles(
              &call,
              {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Primal, ValueType::Primal, ValueType::Primal,
               ValueType::Shadow},
              Builder2, /*lookup*/ false);

#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif

#if LLVM_VERSION_MAJOR > 7
          Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
#else
          Builder2.CreateCall(callval, args, Defs);
#endif
          return;
        }
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    if (funcName == "MPI_Wait" || funcName == "PMPI_Wait") {
      Value *d_reqp = nullptr;
      auto impi = getMPIHelper(call.getContext());
      if (Mode == DerivativeMode::ReverseModePrimal ||
          Mode == DerivativeMode::ReverseModeCombined) {
        Value *req = gutils->getNewFromOriginal(call.getOperand(0));
        Value *d_req = gutils->invertPointerM(call.getOperand(0), BuilderZ);

        if (req->getType()->isIntegerTy()) {
          req = BuilderZ.CreateIntToPtr(
              req,
              PointerType::getUnqual(Type::getInt8PtrTy(call.getContext())));
        }

        Value *isNull = nullptr;
        if (auto GV = gutils->newFunc->getParent()->getNamedValue(
                "ompi_request_null")) {
          Value *reql = BuilderZ.CreatePointerCast(
              req, PointerType::getUnqual(GV->getType()));
#if LLVM_VERSION_MAJOR > 7
          reql = BuilderZ.CreateLoad(GV->getType(), reql);
#else
          reql = BuilderZ.CreateLoad(reql);
#endif
          isNull = BuilderZ.CreateICmpEQ(reql, GV);
        }

        if (d_req->getType()->isIntegerTy()) {
          d_req = BuilderZ.CreateIntToPtr(
              d_req,
              PointerType::getUnqual(Type::getInt8PtrTy(call.getContext())));
        }

#if LLVM_VERSION_MAJOR > 7
        d_reqp = BuilderZ.CreateLoad(
            PointerType::getUnqual(impi),
            BuilderZ.CreatePointerCast(
                d_req, PointerType::getUnqual(PointerType::getUnqual(impi))));
#else
        d_reqp = BuilderZ.CreateLoad(BuilderZ.CreatePointerCast(
            d_req, PointerType::getUnqual(PointerType::getUnqual(impi))));
#endif
        if (isNull)
          d_reqp = BuilderZ.CreateSelect(
              isNull, Constant::getNullValue(d_reqp->getType()), d_reqp);
        if (auto I = dyn_cast<Instruction>(d_reqp))
          gutils->TapesToPreventRecomputation.insert(I);
        d_reqp = gutils->cacheForReverse(BuilderZ, d_reqp,
                                         getIndex(&call, CacheType::Tape));
      }
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);

        assert(!gutils->isConstantValue(call.getOperand(0)));
        Value *req =
            lookup(gutils->getNewFromOriginal(call.getOperand(0)), Builder2);

        if (Mode != DerivativeMode::ReverseModeCombined) {
          d_reqp = BuilderZ.CreatePHI(PointerType::getUnqual(impi), 0);
          d_reqp = gutils->cacheForReverse(BuilderZ, d_reqp,
                                           getIndex(&call, CacheType::Tape));
        } else
          assert(d_reqp);
        d_reqp = lookup(d_reqp, Builder2);

        Value *isNull = Builder2.CreateICmpEQ(
            d_reqp, Constant::getNullValue(d_reqp->getType()));

        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *nonnullBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_nonnull");
        BasicBlock *endBlock = gutils->addReverseBlock(
            nonnullBlock, currentBlock->getName() + "_end",
            /*fork*/ true, /*push*/ false);

        Builder2.CreateCondBr(isNull, endBlock, nonnullBlock);
        Builder2.SetInsertPoint(nonnullBlock);

#if LLVM_VERSION_MAJOR > 7
        Value *cache = Builder2.CreateLoad(
            d_reqp->getType()->getPointerElementType(), d_reqp);
#else
        Value *cache = Builder2.CreateLoad(d_reqp);
#endif

        Value *args[] = {
            getMPIMemberPtr<MPI_Elem::Buf, false>(Builder2, cache),
            getMPIMemberPtr<MPI_Elem::Count, false>(Builder2, cache),
            getMPIMemberPtr<MPI_Elem::DataType, false>(Builder2, cache),
            getMPIMemberPtr<MPI_Elem::Src, false>(Builder2, cache),
            getMPIMemberPtr<MPI_Elem::Tag, false>(Builder2, cache),
            getMPIMemberPtr<MPI_Elem::Comm, false>(Builder2, cache),
            getMPIMemberPtr<MPI_Elem::Call, false>(Builder2, cache),
            req};
        Type *types[sizeof(args) / sizeof(*args) - 1];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args) - 1; i++)
          types[i] = args[i]->getType();
        Function *dwait = getOrInsertDifferentialMPI_Wait(
            *called->getParent(), types, call.getOperand(0)->getType());

        // Need to preserve the shadow Request (operand 0 in wait).
        // However, this doesn't end up preserving
        // the underlying buffers for the adjoint. To rememdy, force inline.
        auto cal =
            Builder2.CreateCall(dwait, args,
                                gutils->getInvertedBundles(
                                    &call, {ValueType::Shadow, ValueType::None},
                                    Builder2, /*lookup*/ true));
        cal->setCallingConv(dwait->getCallingConv());
        cal->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
#if LLVM_VERSION_MAJOR >= 14
        cal->addFnAttr(Attribute::AlwaysInline);
#else
        cal->addAttribute(AttributeList::FunctionIndex,
                          Attribute::AlwaysInline);
#endif
        Builder2.CreateBr(endBlock);
        {
          auto found = gutils->reverseBlockToPrimal.find(endBlock);
          assert(found != gutils->reverseBlockToPrimal.end());
          SmallVector<BasicBlock *, 4> &vec =
              gutils->reverseBlocks[found->second];
          assert(vec.size());
          vec.push_back(endBlock);
        }
        Builder2.SetInsertPoint(endBlock);
      } else if (Mode == DerivativeMode::ForwardMode) {
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);

        assert(!gutils->isConstantValue(call.getOperand(0)));

        Value *request =
            gutils->invertPointerM(call.getArgOperand(0), Builder2);
        Value *status = gutils->invertPointerM(call.getArgOperand(1), Builder2);

        if (request->getType()->isIntegerTy()) {
          request = Builder2.CreateIntToPtr(
              request,
              PointerType::getUnqual(Type::getInt8PtrTy(call.getContext())));
        }

        Value *args[] = {/*request*/ request,
                         /*status*/ status};

        auto Defs = gutils->getInvertedBundles(
            &call, {ValueType::Shadow, ValueType::Shadow}, Builder2,
            /*lookup*/ false);

#if LLVM_VERSION_MAJOR >= 11
        auto callval = call.getCalledOperand();
#else
        auto callval = call.getCalledValue();
#endif

#if LLVM_VERSION_MAJOR > 7
        Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
#else
        Builder2.CreateCall(callval, args, Defs);
#endif
        return;
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    if (funcName == "MPI_Waitall" || funcName == "PMPI_Waitall") {
      Value *d_reqp = nullptr;
      auto impi = getMPIHelper(call.getContext());
      if (Mode == DerivativeMode::ReverseModePrimal ||
          Mode == DerivativeMode::ReverseModeCombined) {
        Value *count = gutils->getNewFromOriginal(call.getOperand(0));
        Value *req = gutils->getNewFromOriginal(call.getOperand(1));
        Value *d_req = gutils->invertPointerM(call.getOperand(1), BuilderZ);

        if (req->getType()->isIntegerTy()) {
          req = BuilderZ.CreateIntToPtr(
              req,
              PointerType::getUnqual(Type::getInt8PtrTy(call.getContext())));
        }

        if (d_req->getType()->isIntegerTy()) {
          d_req = BuilderZ.CreateIntToPtr(
              d_req,
              PointerType::getUnqual(Type::getInt8PtrTy(call.getContext())));
        }

        Function *dsave = getOrInsertDifferentialWaitallSave(
            *gutils->oldFunc->getParent(),
            {count->getType(), req->getType(), d_req->getType()},
            PointerType::getUnqual(impi));

        d_reqp = BuilderZ.CreateCall(dsave, {count, req, d_req});
        cast<CallInst>(d_reqp)->setCallingConv(dsave->getCallingConv());
        cast<CallInst>(d_reqp)->setDebugLoc(
            gutils->getNewFromOriginal(call.getDebugLoc()));
        d_reqp = gutils->cacheForReverse(BuilderZ, d_reqp,
                                         getIndex(&call, CacheType::Tape));
      }
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);

        assert(!gutils->isConstantValue(call.getOperand(1)));
        Value *count =
            lookup(gutils->getNewFromOriginal(call.getOperand(0)), Builder2);
        Value *req_orig =
            lookup(gutils->getNewFromOriginal(call.getOperand(1)), Builder2);

        if (Mode != DerivativeMode::ReverseModeCombined) {
          d_reqp = BuilderZ.CreatePHI(
              PointerType::getUnqual(PointerType::getUnqual(impi)), 0);
          d_reqp = gutils->cacheForReverse(BuilderZ, d_reqp,
                                           getIndex(&call, CacheType::Tape));
        }

        d_reqp = lookup(d_reqp, Builder2);

        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *loopBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_loop");
        BasicBlock *nonnullBlock = gutils->addReverseBlock(
            loopBlock, currentBlock->getName() + "_nonnull");
        BasicBlock *eloopBlock = gutils->addReverseBlock(
            nonnullBlock, currentBlock->getName() + "_eloop");
        BasicBlock *endBlock = gutils->addReverseBlock(
            eloopBlock, currentBlock->getName() + "_end",
            /*fork*/ true, /*push*/ false);

        Builder2.CreateCondBr(
            Builder2.CreateICmpNE(count,
                                  ConstantInt::get(count->getType(), 0, false)),
            loopBlock, endBlock);

        Builder2.SetInsertPoint(loopBlock);
        auto idx = Builder2.CreatePHI(count->getType(), 2);
        idx->addIncoming(ConstantInt::get(count->getType(), 0, false),
                         currentBlock);
        Value *inc = Builder2.CreateAdd(
            idx, ConstantInt::get(count->getType(), 1, false), "", true, true);
        idx->addIncoming(inc, eloopBlock);

        Value *idxs[] = {idx};
#if LLVM_VERSION_MAJOR > 7
        Value *req = Builder2.CreateInBoundsGEP(
            req_orig->getType()->getPointerElementType(), req_orig, idxs);
        Value *d_req = Builder2.CreateInBoundsGEP(
            d_reqp->getType()->getPointerElementType(), d_reqp, idxs);
#else
        Value *req = Builder2.CreateInBoundsGEP(req_orig, idxs);
        Value *d_req = Builder2.CreateInBoundsGEP(d_reqp, idxs);
#endif

#if LLVM_VERSION_MAJOR > 7
        d_req = Builder2.CreateLoad(
            PointerType::getUnqual(impi),
            Builder2.CreatePointerCast(
                d_req, PointerType::getUnqual(PointerType::getUnqual(impi))));
#else
        d_req = Builder2.CreateLoad(Builder2.CreatePointerCast(
            d_req, PointerType::getUnqual(PointerType::getUnqual(impi))));
#endif

        Value *isNull = Builder2.CreateICmpEQ(
            d_req, Constant::getNullValue(d_req->getType()));

        Builder2.CreateCondBr(isNull, eloopBlock, nonnullBlock);
        Builder2.SetInsertPoint(nonnullBlock);

#if LLVM_VERSION_MAJOR > 7
        Value *cache = Builder2.CreateLoad(
            d_req->getType()->getPointerElementType(), d_req);
#else
        Value *cache = Builder2.CreateLoad(d_req);
#endif

        Value *args[] = {
            getMPIMemberPtr<MPI_Elem::Buf, false>(Builder2, cache),
            getMPIMemberPtr<MPI_Elem::Count, false>(Builder2, cache),
            getMPIMemberPtr<MPI_Elem::DataType, false>(Builder2, cache),
            getMPIMemberPtr<MPI_Elem::Src, false>(Builder2, cache),
            getMPIMemberPtr<MPI_Elem::Tag, false>(Builder2, cache),
            getMPIMemberPtr<MPI_Elem::Comm, false>(Builder2, cache),
            getMPIMemberPtr<MPI_Elem::Call, false>(Builder2, cache),
            req};
        Type *types[sizeof(args) / sizeof(*args) - 1];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args) - 1; i++)
          types[i] = args[i]->getType();
        Function *dwait = getOrInsertDifferentialMPI_Wait(
            *called->getParent(), types, req->getType());
        // Need to preserve the shadow Request (operand 6 in isend/irecv), which
        // becomes operand 0 for iwait. However, this doesn't end up preserving
        // the underlying buffers for the adjoint. To remedy, force inline the
        // function.
        auto cal = Builder2.CreateCall(
            dwait, args,
            gutils->getInvertedBundles(&call,
                                       {ValueType::None, ValueType::None,
                                        ValueType::None, ValueType::None,
                                        ValueType::None, ValueType::None,
                                        ValueType::Shadow},
                                       Builder2, /*lookup*/ true));
        cal->setCallingConv(dwait->getCallingConv());
        cal->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
#if LLVM_VERSION_MAJOR >= 14
        cal->addFnAttr(Attribute::AlwaysInline);
#else
        cal->addAttribute(AttributeList::FunctionIndex,
                          Attribute::AlwaysInline);
#endif
        Builder2.CreateBr(eloopBlock);

        Builder2.SetInsertPoint(eloopBlock);
        Builder2.CreateCondBr(Builder2.CreateICmpEQ(inc, count), endBlock,
                              loopBlock);
        {
          auto found = gutils->reverseBlockToPrimal.find(endBlock);
          assert(found != gutils->reverseBlockToPrimal.end());
          SmallVector<BasicBlock *, 4> &vec =
              gutils->reverseBlocks[found->second];
          assert(vec.size());
          vec.push_back(endBlock);
        }
        Builder2.SetInsertPoint(endBlock);
        if (shouldFree()) {
          auto ci = cast<CallInst>(CallInst::CreateFree(
              Builder2.CreatePointerCast(
                  d_reqp, Type::getInt8PtrTy(d_reqp->getContext())),
              Builder2.GetInsertBlock()));
#if LLVM_VERSION_MAJOR >= 14
          ci->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                  Attribute::NonNull);
#else
          ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
#endif
          if (ci->getParent() == nullptr) {
            Builder2.Insert(ci);
          }
        }
      } else if (Mode == DerivativeMode::ForwardMode) {
        IRBuilder<> Builder2(&call);

        assert(!gutils->isConstantValue(call.getOperand(1)));

        Value *count = gutils->getNewFromOriginal(call.getOperand(0));
        Value *array_of_requests = gutils->invertPointerM(
            gutils->getNewFromOriginal(call.getOperand(1)), Builder2);
        if (array_of_requests->getType()->isIntegerTy()) {
          array_of_requests = Builder2.CreateIntToPtr(
              array_of_requests,
              PointerType::getUnqual(Type::getInt8PtrTy(call.getContext())));
        }

        Value *args[] = {
            /*count*/ count,
            /*array_of_requests*/ array_of_requests,
        };

        auto Defs = gutils->getInvertedBundles(
            &call,
            {ValueType::None, ValueType::None, ValueType::None, ValueType::None,
             ValueType::None, ValueType::None, ValueType::Shadow},
            Builder2, /*lookup*/ true);

#if LLVM_VERSION_MAJOR >= 11
        auto callval = call.getCalledOperand();
#else
        auto callval = call.getCalledValue();
#endif

#if LLVM_VERSION_MAJOR > 7
        Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
#else
        Builder2.CreateCall(callval, args, Defs);
#endif
        return;
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    if (funcName == "MPI_Send" || funcName == "MPI_Ssend" ||
        funcName == "PMPI_Send" || funcName == "PMPI_Ssend") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ForwardMode) {
        bool forwardMode = Mode == DerivativeMode::ForwardMode;

        IRBuilder<> Builder2 =
            forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
        if (forwardMode) {
          getForwardBuilder(Builder2);
        } else {
          getReverseBuilder(Builder2);
        }

        Value *shadow = gutils->invertPointerM(call.getOperand(0), Builder2);
        if (!forwardMode)
          shadow = lookup(shadow, Builder2);
        if (shadow->getType()->isIntegerTy())
          shadow = Builder2.CreateIntToPtr(
              shadow, Type::getInt8PtrTy(call.getContext()));

        Type *statusType = nullptr;
        if (Function *recvfn = called->getParent()->getFunction("MPI_Recv")) {
          auto statusArg = recvfn->arg_end();
          statusArg--;
          if (auto PT = dyn_cast<PointerType>(statusArg->getType()))
            statusType = PT->getPointerElementType();
        } else if (Function *recvfn =
                       called->getParent()->getFunction("PMPI_Recv")) {
          auto statusArg = recvfn->arg_end();
          statusArg--;
          if (auto PT = dyn_cast<PointerType>(statusArg->getType()))
            statusType = PT->getPointerElementType();
        }
        if (statusType == nullptr) {
          statusType = ArrayType::get(Type::getInt8Ty(call.getContext()), 24);
          llvm::errs() << " warning could not automatically determine mpi "
                          "status type, assuming [24 x i8]\n";
        }

        Value *count = gutils->getNewFromOriginal(call.getOperand(1));
        if (!forwardMode)
          count = lookup(count, Builder2);

        Value *datatype = gutils->getNewFromOriginal(call.getOperand(2));
        if (!forwardMode)
          datatype = lookup(datatype, Builder2);

        Value *src = gutils->getNewFromOriginal(call.getOperand(3));
        if (!forwardMode)
          src = lookup(src, Builder2);

        Value *tag = gutils->getNewFromOriginal(call.getOperand(4));
        if (!forwardMode)
          tag = lookup(tag, Builder2);

        Value *comm = gutils->getNewFromOriginal(call.getOperand(5));
        if (!forwardMode)
          comm = lookup(comm, Builder2);

        if (forwardMode) {
          Value *args[] = {
              /*buf*/ shadow,
              /*count*/ count,
              /*datatype*/ datatype,
              /*dest*/ src,
              /*tag*/ tag,
              /*comm*/ comm,
          };

          auto Defs = gutils->getInvertedBundles(
              &call,
              {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Primal, ValueType::Primal, ValueType::Primal},
              Builder2, /*lookup*/ false);

#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif

#if LLVM_VERSION_MAJOR > 7
          Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
#else
          Builder2.CreateCall(callval, args, Defs);
#endif
          return;
        }

        Value *args[] = {
            /*buf*/ NULL,
            /*count*/ count,
            /*datatype*/ datatype,
            /*src*/ src,
            /*tag*/ tag,
            /*comm*/ comm,
            /*status*/
            IRBuilder<>(gutils->inversionAllocs).CreateAlloca(statusType)};

        Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType());

        auto len_arg = Builder2.CreateZExtOrTrunc(
            args[1], Type::getInt64Ty(call.getContext()));
        len_arg =
            Builder2.CreateMul(len_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);

        Value *firstallocation = CallInst::CreateMalloc(
            Builder2.GetInsertBlock(), len_arg->getType(),
            Type::getInt8Ty(call.getContext()),
            ConstantInt::get(Type::getInt64Ty(len_arg->getContext()), 1),
            len_arg, nullptr, "mpirecv_malloccache");
        if (cast<Instruction>(firstallocation)->getParent() == nullptr) {
          Builder2.Insert(cast<Instruction>(firstallocation));
        }
        args[0] = firstallocation;

        Type *types[sizeof(args) / sizeof(*args)];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
          types[i] = args[i]->getType();
        FunctionType *FT = FunctionType::get(call.getType(), types, false);

        Builder2.SetInsertPoint(Builder2.GetInsertBlock());

        auto BufferDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::None, ValueType::None,
             ValueType::None, ValueType::None, ValueType::None,
             ValueType::None},
            Builder2, /*lookup*/ true);

        auto fcall = Builder2.CreateCall(
            called->getParent()->getOrInsertFunction("MPI_Recv", FT), args);
        fcall->setCallingConv(call.getCallingConv());

        DifferentiableMemCopyFloats(call, call.getOperand(0), firstallocation,
                                    shadow, len_arg, Builder2, BufferDefs);

        if (shouldFree()) {
          auto ci = cast<CallInst>(
              CallInst::CreateFree(firstallocation, Builder2.GetInsertBlock()));
#if LLVM_VERSION_MAJOR >= 14
          ci->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                  Attribute::NonNull);
#else
          ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
#endif
          if (ci->getParent() == nullptr) {
            Builder2.Insert(ci);
          }
        }
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    if (funcName == "MPI_Recv" || funcName == "PMPI_Recv") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        bool forwardMode = Mode == DerivativeMode::ForwardMode;

        IRBuilder<> Builder2 =
            forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
        if (forwardMode) {
          getForwardBuilder(Builder2);
        } else {
          getReverseBuilder(Builder2);
        }

        Value *shadow = gutils->invertPointerM(call.getOperand(0), Builder2);
        if (!forwardMode)
          shadow = lookup(shadow, Builder2);
        if (shadow->getType()->isIntegerTy())
          shadow = Builder2.CreateIntToPtr(
              shadow, Type::getInt8PtrTy(call.getContext()));

        Value *count = gutils->getNewFromOriginal(call.getOperand(1));
        if (!forwardMode)
          count = lookup(count, Builder2);

        Value *datatype = gutils->getNewFromOriginal(call.getOperand(2));
        if (!forwardMode)
          datatype = lookup(datatype, Builder2);

        Value *source = gutils->getNewFromOriginal(call.getOperand(3));
        if (!forwardMode)
          source = lookup(source, Builder2);

        Value *tag = gutils->getNewFromOriginal(call.getOperand(4));
        if (!forwardMode)
          tag = lookup(tag, Builder2);

        Value *comm = gutils->getNewFromOriginal(call.getOperand(5));
        if (!forwardMode)
          comm = lookup(comm, Builder2);

        Value *args[] = {
            shadow, count, datatype, source, tag, comm,
        };

        auto Defs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Primal, ValueType::Primal, ValueType::Primal,
             ValueType::None},
            Builder2, /*lookup*/ !forwardMode);

        if (forwardMode) {
#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif

#if LLVM_VERSION_MAJOR > 7
          Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
#else
          Builder2.CreateCall(callval, args, Defs);
#endif
          return;
        }

        Type *types[sizeof(args) / sizeof(*args)];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
          types[i] = args[i]->getType();
        FunctionType *FT = FunctionType::get(call.getType(), types, false);

        auto fcall = Builder2.CreateCall(
            called->getParent()->getOrInsertFunction("MPI_Send", FT), args,
            Defs);
        fcall->setCallingConv(call.getCallingConv());

        auto dst_arg = Builder2.CreateBitCast(
            args[0], Type::getInt8PtrTy(call.getContext()));
        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto len_arg = Builder2.CreateZExtOrTrunc(
            args[1], Type::getInt64Ty(call.getContext()));
        auto tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType());
        len_arg =
            Builder2.CreateMul(len_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());

#if LLVM_VERSION_MAJOR == 6
        auto align_arg =
            ConstantInt::get(Type::getInt32Ty(call.getContext()), 1);
        Value *nargs[] = {dst_arg, val_arg, len_arg, align_arg, volatile_arg};
#else
        Value *nargs[] = {dst_arg, val_arg, len_arg, volatile_arg};
#endif

        Type *tys[] = {dst_arg->getType(), len_arg->getType()};

        auto MemsetDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::None, ValueType::None,
             ValueType::None, ValueType::None, ValueType::None,
             ValueType::None},
            Builder2, /*lookup*/ true);
        auto memset = cast<CallInst>(Builder2.CreateCall(
            Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                      Intrinsic::memset, tys),
            nargs));
        memset->addParamAttr(0, Attribute::NonNull);
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root,
    //           MPI_Comm comm )
    // 1. if root, malloc intermediate buffer
    // 2. reduce sum diff(buffer) into intermediate
    // 3. if root, set shadow(buffer) = intermediate [memcpy] then free
    // 3-e. else, set shadow(buffer) = 0 [memset]
    if (funcName == "MPI_Bcast") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ForwardMode) {
        bool forwardMode = Mode == DerivativeMode::ForwardMode;

        IRBuilder<> Builder2 =
            forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
        if (forwardMode) {
          getForwardBuilder(Builder2);
        } else {
          getReverseBuilder(Builder2);
        }

        Value *shadow = gutils->invertPointerM(call.getOperand(0), Builder2);
        if (!forwardMode)
          shadow = lookup(shadow, Builder2);
        if (shadow->getType()->isIntegerTy())
          shadow = Builder2.CreateIntToPtr(
              shadow, Type::getInt8PtrTy(call.getContext()));

        ConcreteType CT = TR.firstPointer(1, call.getOperand(0));
        Type *MPI_OP_Ptr_type =
            PointerType::getUnqual(Type::getInt8PtrTy(call.getContext()));

        Value *count = gutils->getNewFromOriginal(call.getOperand(1));
        if (!forwardMode)
          count = lookup(count, Builder2);
        Value *datatype = gutils->getNewFromOriginal(call.getOperand(2));
        if (!forwardMode)
          datatype = lookup(datatype, Builder2);
        Value *root = gutils->getNewFromOriginal(call.getOperand(3));
        if (!forwardMode)
          root = lookup(root, Builder2);

        Value *comm = gutils->getNewFromOriginal(call.getOperand(4));
        if (!forwardMode)
          comm = lookup(comm, Builder2);

        if (forwardMode) {
          Value *args[] = {
              /*buffer*/ shadow,
              /*count*/ count,
              /*datatype*/ datatype,
              /*root*/ root,
              /*comm*/ comm,
          };

          auto Defs = gutils->getInvertedBundles(
              &call,
              {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Primal, ValueType::Primal},
              Builder2, /*lookup*/ false);

#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif

#if LLVM_VERSION_MAJOR > 7
          Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
#else
          Builder2.CreateCall(callval, args, Defs);
#endif
          return;
        }

        Value *rank = MPI_COMM_RANK(comm, Builder2, root->getType());
        Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType());

        auto len_arg = Builder2.CreateZExtOrTrunc(
            count, Type::getInt64Ty(call.getContext()));
        len_arg =
            Builder2.CreateMul(len_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);

        // 1. if root, malloc intermediate buffer, else undef
        PHINode *buf;

        {
          BasicBlock *currentBlock = Builder2.GetInsertBlock();
          BasicBlock *rootBlock = gutils->addReverseBlock(
              currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
          BasicBlock *mergeBlock = gutils->addReverseBlock(
              rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

          Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                                mergeBlock);

          Builder2.SetInsertPoint(rootBlock);

          Value *rootbuf = CallInst::CreateMalloc(
              Builder2.GetInsertBlock(), len_arg->getType(),
              Type::getInt8Ty(call.getContext()),
              ConstantInt::get(Type::getInt64Ty(len_arg->getContext()), 1),
              len_arg, nullptr, "mpireduce_malloccache");
          if (cast<Instruction>(rootbuf)->getParent() == nullptr) {
            Builder2.Insert(cast<Instruction>(rootbuf));
          }
          Builder2.SetInsertPoint(rootBlock);
          Builder2.CreateBr(mergeBlock);

          Builder2.SetInsertPoint(mergeBlock);

          buf = Builder2.CreatePHI(rootbuf->getType(), 2);
          buf->addIncoming(rootbuf, rootBlock);
          buf->addIncoming(UndefValue::get(buf->getType()), currentBlock);
        }

        // Need to preserve the shadow buffer.
        auto BufferDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Primal, ValueType::Primal},
            Builder2, /*lookup*/ true);

        // 2. reduce sum diff(buffer) into intermediate
        {
          // int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
          // MPI_Datatype datatype,
          //     MPI_Op op, int root, MPI_Comm comm)
          Value *args[] = {
              /*sendbuf*/ shadow,
              /*recvbuf*/ buf,
              /*count*/ count,
              /*datatype*/ datatype,
              /*op (MPI_SUM)*/
              getOrInsertOpFloatSum(*gutils->newFunc->getParent(),
                                    MPI_OP_Ptr_type, CT, root->getType(),
                                    Builder2),
              /*int root*/ root,
              /*comm*/ comm,
          };
          Type *types[sizeof(args) / sizeof(*args)];
          for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
            types[i] = args[i]->getType();

          FunctionType *FT = FunctionType::get(call.getType(), types, false);

          Builder2.CreateCall(
              called->getParent()->getOrInsertFunction("MPI_Reduce", FT), args,
              BufferDefs);
        }

        // 3. if root, set shadow(buffer) = intermediate [memcpy]
        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *rootBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
        BasicBlock *nonrootBlock = gutils->addReverseBlock(
            rootBlock, currentBlock->getName() + "_nonroot", gutils->newFunc);
        BasicBlock *mergeBlock = gutils->addReverseBlock(
            nonrootBlock, currentBlock->getName() + "_post", gutils->newFunc);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                              nonrootBlock);

        Builder2.SetInsertPoint(rootBlock);

        {
          auto volatile_arg = ConstantInt::getFalse(call.getContext());
          Value *nargs[] = {shadow, buf, len_arg, volatile_arg};

          Type *tys[] = {shadow->getType(), buf->getType(), len_arg->getType()};

          auto memcpyF = Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                                   Intrinsic::memcpy, tys);

          auto mem =
              cast<CallInst>(Builder2.CreateCall(memcpyF, nargs, BufferDefs));
          mem->setCallingConv(memcpyF->getCallingConv());

          // Free up the memory of the buffer
          if (shouldFree()) {
            auto ci = cast<CallInst>(
                CallInst::CreateFree(buf, Builder2.GetInsertBlock()));
#if LLVM_VERSION_MAJOR >= 14
            ci->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                    Attribute::NonNull);
#else
            ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
#endif
            if (ci->getParent() == nullptr) {
              Builder2.Insert(ci);
            }
          }
        }

        Builder2.CreateBr(mergeBlock);

        Builder2.SetInsertPoint(nonrootBlock);

        // 3-e. else, set shadow(buffer) = 0 [memset]
        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *args[] = {shadow, val_arg, len_arg, volatile_arg};
        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                      Intrinsic::memset, tys),
            args, BufferDefs));
        memset->addParamAttr(0, Attribute::NonNull);
        Builder2.CreateBr(mergeBlock);

        Builder2.SetInsertPoint(mergeBlock);
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // Approximate algo (for sum):  -> if statement yet to be
    // 1. malloc intermediate buffer
    // 1.5 if root, set intermediate = diff(recvbuffer)
    // 2. MPI_Bcast intermediate to all
    // 3. if root, Zero diff(recvbuffer) [memset to 0]
    // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
    // 5. free intermediate buffer

    // int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
    // MPI_Datatype datatype,
    //                      MPI_Op op, int root, MPI_Comm comm)

    if (funcName == "MPI_Reduce" || funcName == "PMPI_Reduce") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ForwardMode) {
        // TODO insert a check for sum

        bool forwardMode = Mode == DerivativeMode::ForwardMode;

        IRBuilder<> Builder2 =
            forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
        if (forwardMode) {
          getForwardBuilder(Builder2);
        } else {
          getReverseBuilder(Builder2);
        }

        // Get the operations from MPI_Receive
        Value *orig_sendbuf = call.getOperand(0);
        Value *orig_recvbuf = call.getOperand(1);
        Value *orig_count = call.getOperand(2);
        Value *orig_datatype = call.getOperand(3);
        Value *orig_op = call.getOperand(4);
        Value *orig_root = call.getOperand(5);
        Value *orig_comm = call.getOperand(6);

        bool isSum = false;
        if (Constant *C = dyn_cast<Constant>(orig_op)) {
          while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
            C = CE->getOperand(0);
          }
          if (auto GV = dyn_cast<GlobalVariable>(C)) {
            if (GV->getName() == "ompi_mpi_op_sum") {
              isSum = true;
            }
          }
          // MPICH
          if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
            if (CI->getValue() == 1476395011) {
              isSum = true;
            }
          }
        }
        if (!isSum) {
          std::string s;
          llvm::raw_string_ostream ss(s);
          ss << *gutils->oldFunc << "\n";
          ss << *gutils->newFunc << "\n";
          ss << " call: " << call << "\n";
          ss << " unhandled mpi_allreduce op: " << *orig_op << "\n";
          if (CustomErrorHandler) {
            CustomErrorHandler(ss.str().c_str(), wrap(&call),
                               ErrorType::NoDerivative, nullptr);
          }
          llvm::errs() << ss.str() << "\n";
          report_fatal_error("unhandled mpi_allreduce op");
        }

        Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
        if (!forwardMode)
          shadow_recvbuf = lookup(shadow_recvbuf, Builder2);
        if (shadow_recvbuf->getType()->isIntegerTy())
          shadow_recvbuf = Builder2.CreateIntToPtr(
              shadow_recvbuf, Type::getInt8PtrTy(call.getContext()));

        Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
        if (!forwardMode)
          shadow_sendbuf = lookup(shadow_sendbuf, Builder2);
        if (shadow_sendbuf->getType()->isIntegerTy())
          shadow_sendbuf = Builder2.CreateIntToPtr(
              shadow_sendbuf, Type::getInt8PtrTy(call.getContext()));

        // Need to preserve the shadow send/recv buffers.
        auto BufferDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Shadow, ValueType::Primal,
             ValueType::Primal, ValueType::Primal, ValueType::Primal,
             ValueType::Primal},
            Builder2, /*lookup*/ true);

        Value *count = gutils->getNewFromOriginal(orig_count);
        if (!forwardMode)
          count = lookup(count, Builder2);

        Value *datatype = gutils->getNewFromOriginal(orig_datatype);
        if (!forwardMode)
          datatype = lookup(datatype, Builder2);

        Value *op = lookup(gutils->getNewFromOriginal(orig_op), Builder2);
        if (!forwardMode)
          op = lookup(op, Builder2);

        Value *root = gutils->getNewFromOriginal(orig_root);
        if (!forwardMode)
          root = lookup(root, Builder2);

        Value *comm = lookup(gutils->getNewFromOriginal(orig_comm), Builder2);
        if (!forwardMode)
          comm = lookup(comm, Builder2);

        Value *rank = MPI_COMM_RANK(comm, Builder2, root->getType());

        if (forwardMode) {
          Value *args[] = {
              /*sendbuf*/ shadow_sendbuf,
              /*recvbuf*/ shadow_recvbuf,
              /*count*/ count,
              /*datatype*/ datatype,
              /*op*/ op,
              /*root*/ root,
              /*comm*/ comm,
          };

          auto Defs = gutils->getInvertedBundles(
              &call,
              {ValueType::Shadow, ValueType::Shadow, ValueType::Primal,
               ValueType::Primal, ValueType::Primal, ValueType::Primal,
               ValueType::Primal},
              Builder2, /*lookup*/ false);

#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif

#if LLVM_VERSION_MAJOR > 7
          Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
#else
          Builder2.CreateCall(callval, args, Defs);
#endif
          return;
        }

        Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType());

        // Get the length for the allocation of the intermediate buffer
        auto len_arg = Builder2.CreateZExtOrTrunc(
            count, Type::getInt64Ty(call.getContext()));
        len_arg =
            Builder2.CreateMul(len_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);

        // 1. Alloc intermediate buffer
        Value *buf = CallInst::CreateMalloc(
            Builder2.GetInsertBlock(), len_arg->getType(),
            Type::getInt8Ty(call.getContext()),
            ConstantInt::get(Type::getInt64Ty(len_arg->getContext()), 1),
            len_arg, nullptr, "mpireduce_malloccache");
        if (cast<Instruction>(buf)->getParent() == nullptr) {
          Builder2.Insert(cast<Instruction>(buf));
        }

        // 1.5 if root, set intermediate = diff(recvbuffer)
        {

          BasicBlock *currentBlock = Builder2.GetInsertBlock();
          BasicBlock *rootBlock = gutils->addReverseBlock(
              currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
          BasicBlock *mergeBlock = gutils->addReverseBlock(
              rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

          Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                                mergeBlock);

          Builder2.SetInsertPoint(rootBlock);

          {
            auto volatile_arg = ConstantInt::getFalse(call.getContext());
            Value *nargs[] = {buf, shadow_recvbuf, len_arg, volatile_arg};

            Type *tys[] = {nargs[0]->getType(), nargs[1]->getType(),
                           len_arg->getType()};

            auto memcpyF = Intrinsic::getDeclaration(
                gutils->newFunc->getParent(), Intrinsic::memcpy, tys);

            auto mem =
                cast<CallInst>(Builder2.CreateCall(memcpyF, nargs, BufferDefs));
            mem->setCallingConv(memcpyF->getCallingConv());
          }

          Builder2.CreateBr(mergeBlock);
          Builder2.SetInsertPoint(mergeBlock);
        }

        // 2. MPI_Bcast intermediate to all
        {
          // int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int
          // root,
          //     MPI_Comm comm )
          Value *args[] = {
              /*buf*/ buf,
              /*count*/ count,
              /*datatype*/ datatype,
              /*int root*/ root,
              /*comm*/ comm,
          };
          Type *types[sizeof(args) / sizeof(*args)];
          for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
            types[i] = args[i]->getType();

          FunctionType *FT = FunctionType::get(call.getType(), types, false);
          Builder2.CreateCall(
              called->getParent()->getOrInsertFunction("MPI_Bcast", FT), args,
              BufferDefs);
        }

        // 3. if root, Zero diff(recvbuffer) [memset to 0]
        {
          BasicBlock *currentBlock = Builder2.GetInsertBlock();
          BasicBlock *rootBlock = gutils->addReverseBlock(
              currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
          BasicBlock *mergeBlock = gutils->addReverseBlock(
              rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

          Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                                mergeBlock);

          Builder2.SetInsertPoint(rootBlock);

          auto val_arg =
              ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
          auto volatile_arg = ConstantInt::getFalse(call.getContext());
          Value *args[] = {shadow_recvbuf, val_arg, len_arg, volatile_arg};
          Type *tys[] = {args[0]->getType(), args[2]->getType()};
          auto memset = cast<CallInst>(Builder2.CreateCall(
              Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                        Intrinsic::memset, tys),
              args, BufferDefs));
          memset->addParamAttr(0, Attribute::NonNull);

          Builder2.CreateBr(mergeBlock);
          Builder2.SetInsertPoint(mergeBlock);
        }

        // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
        DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                    len_arg, Builder2, BufferDefs);

        // Free up intermediate buffer
        if (shouldFree()) {
          auto ci = cast<CallInst>(
              CallInst::CreateFree(buf, Builder2.GetInsertBlock()));
#if LLVM_VERSION_MAJOR >= 14
          ci->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                  Attribute::NonNull);
#else
          ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
#endif
          if (ci->getParent() == nullptr) {
            Builder2.Insert(ci);
          }
        }
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // Approximate algo (for sum):  -> if statement yet to be
    // 1. malloc intermediate buffers
    // 2. MPI_Allreduce (sum) of diff(recvbuffer) to intermediate
    // 3. Zero diff(recvbuffer) [memset to 0]
    // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
    // 5. free intermediate buffer

    // int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
    //              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)

    if (funcName == "MPI_Allreduce") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ForwardMode) {
        // TODO insert a check for sum

        bool forwardMode = Mode == DerivativeMode::ForwardMode;

        IRBuilder<> Builder2 =
            forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
        if (forwardMode) {
          getForwardBuilder(Builder2);
        } else {
          getReverseBuilder(Builder2);
        }

        // Get the operations from MPI_Receive
        Value *orig_sendbuf = call.getOperand(0);
        Value *orig_recvbuf = call.getOperand(1);
        Value *orig_count = call.getOperand(2);
        Value *orig_datatype = call.getOperand(3);
        Value *orig_op = call.getOperand(4);
        Value *orig_comm = call.getOperand(5);

        bool isSum = false;
        if (Constant *C = dyn_cast<Constant>(orig_op)) {
          while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
            C = CE->getOperand(0);
          }
          if (auto GV = dyn_cast<GlobalVariable>(C)) {
            if (GV->getName() == "ompi_mpi_op_sum") {
              isSum = true;
            }
          }
          // MPICH
          if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
            if (CI->getValue() == 1476395011) {
              isSum = true;
            }
          }
        }
        if (!isSum) {
          std::string s;
          llvm::raw_string_ostream ss(s);
          ss << *gutils->oldFunc << "\n";
          ss << *gutils->newFunc << "\n";
          ss << " call: " << call << "\n";
          ss << " unhandled mpi_allreduce op: " << *orig_op << "\n";
          if (CustomErrorHandler) {
            CustomErrorHandler(ss.str().c_str(), wrap(&call),
                               ErrorType::NoDerivative, nullptr);
          }
          llvm::errs() << ss.str() << "\n";
          report_fatal_error("unhandled mpi_allreduce op");
        }

        Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
        if (!forwardMode)
          shadow_recvbuf = lookup(shadow_recvbuf, Builder2);
        if (shadow_recvbuf->getType()->isIntegerTy())
          shadow_recvbuf = Builder2.CreateIntToPtr(
              shadow_recvbuf, Type::getInt8PtrTy(call.getContext()));

        Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
        if (!forwardMode)
          shadow_sendbuf = lookup(shadow_sendbuf, Builder2);
        if (shadow_sendbuf->getType()->isIntegerTy())
          shadow_sendbuf = Builder2.CreateIntToPtr(
              shadow_sendbuf, Type::getInt8PtrTy(call.getContext()));

        // Need to preserve the shadow send/recv buffers.
        auto BufferDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Shadow, ValueType::Primal,
             ValueType::Primal, ValueType::Primal, ValueType::Primal},
            Builder2, /*lookup*/ !forwardMode);

        Value *count = gutils->getNewFromOriginal(orig_count);
        if (!forwardMode)
          count = lookup(count, Builder2);

        Value *datatype = gutils->getNewFromOriginal(orig_datatype);
        if (!forwardMode)
          datatype = lookup(datatype, Builder2);

        Value *comm = gutils->getNewFromOriginal(orig_comm);
        if (!forwardMode)
          comm = lookup(comm, Builder2);

        Value *op = gutils->getNewFromOriginal(orig_op);
        if (!forwardMode)
          op = lookup(op, Builder2);

        if (forwardMode) {
          Value *args[] = {
              /*sendbuf*/ shadow_sendbuf,
              /*recvbuf*/ shadow_recvbuf,
              /*count*/ count,
              /*datatype*/ datatype,
              /*op*/ op,
              /*comm*/ comm,
          };

#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif

#if LLVM_VERSION_MAJOR > 7
          Builder2.CreateCall(call.getFunctionType(), callval, args,
                              BufferDefs);
#else
          Builder2.CreateCall(callval, args, BufferDefs);
#endif

          return;
        }

        Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType());

        // Get the length for the allocation of the intermediate buffer
        auto len_arg = Builder2.CreateZExtOrTrunc(
            count, Type::getInt64Ty(call.getContext()));
        len_arg =
            Builder2.CreateMul(len_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);

        // 1. Alloc intermediate buffer
        Value *buf = CallInst::CreateMalloc(
            Builder2.GetInsertBlock(), len_arg->getType(),
            Type::getInt8Ty(call.getContext()),
            ConstantInt::get(Type::getInt64Ty(len_arg->getContext()), 1),
            len_arg, nullptr, "mpireduce_malloccache");
        if (cast<Instruction>(buf)->getParent() == nullptr) {
          Builder2.Insert(cast<Instruction>(buf));
        }

        // 2. MPI_Allreduce (sum) of diff(recvbuffer) to intermediate
        {
          // int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
          //              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
          Value *args[] = {
              /*sendbuf*/ shadow_recvbuf,
              /*recvbuf*/ buf,
              /*count*/ count,
              /*datatype*/ datatype,
              /*op*/ op,
              /*comm*/ comm,
          };
          Type *types[sizeof(args) / sizeof(*args)];
          for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
            types[i] = args[i]->getType();

          FunctionType *FT = FunctionType::get(call.getType(), types, false);
          Builder2.CreateCall(
              called->getParent()->getOrInsertFunction("MPI_Allreduce", FT),
              args, BufferDefs);
        }

        // 3. Zero diff(recvbuffer) [memset to 0]
        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *args[] = {shadow_recvbuf, val_arg, len_arg, volatile_arg};
        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                      Intrinsic::memset, tys),
            args, BufferDefs));
        memset->addParamAttr(0, Attribute::NonNull);

        // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
        DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                    len_arg, Builder2, BufferDefs);

        // Free up intermediate buffer
        if (shouldFree()) {
          auto ci = cast<CallInst>(
              CallInst::CreateFree(buf, Builder2.GetInsertBlock()));
#if LLVM_VERSION_MAJOR >= 14
          ci->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                  Attribute::NonNull);
#else
          ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
#endif
          if (ci->getParent() == nullptr) {
            Builder2.Insert(ci);
          }
        }
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // Approximate algo (for sum):  -> if statement yet to be
    // 1. malloc intermediate buffer
    // 2. Scatter diff(recvbuffer) to intermediate buffer
    // 3. if root, Zero diff(recvbuffer) [memset to 0]
    // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
    // 5. free intermediate buffer

    // int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    //           void *recvbuf, int recvcount, MPI_Datatype recvtype,
    //           int root, MPI_Comm comm)

    if (funcName == "MPI_Gather") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ForwardMode) {
        bool forwardMode = Mode == DerivativeMode::ForwardMode;

        IRBuilder<> Builder2 =
            forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
        if (forwardMode) {
          getForwardBuilder(Builder2);
        } else {
          getReverseBuilder(Builder2);
        }

        Value *orig_sendbuf = call.getOperand(0);
        Value *orig_sendcount = call.getOperand(1);
        Value *orig_sendtype = call.getOperand(2);
        Value *orig_recvbuf = call.getOperand(3);
        Value *orig_recvcount = call.getOperand(4);
        Value *orig_recvtype = call.getOperand(5);
        Value *orig_root = call.getOperand(6);
        Value *orig_comm = call.getOperand(7);

        Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
        if (!forwardMode)
          shadow_recvbuf = lookup(shadow_recvbuf, Builder2);
        if (shadow_recvbuf->getType()->isIntegerTy())
          shadow_recvbuf = Builder2.CreateIntToPtr(
              shadow_recvbuf, Type::getInt8PtrTy(call.getContext()));

        Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
        if (!forwardMode)
          shadow_sendbuf = lookup(shadow_sendbuf, Builder2);
        if (shadow_sendbuf->getType()->isIntegerTy())
          shadow_sendbuf = Builder2.CreateIntToPtr(
              shadow_sendbuf, Type::getInt8PtrTy(call.getContext()));

        Value *recvcount = gutils->getNewFromOriginal(orig_recvcount);
        if (!forwardMode)
          recvcount = lookup(recvcount, Builder2);

        Value *recvtype = gutils->getNewFromOriginal(orig_recvtype);
        if (!forwardMode)
          recvtype = lookup(recvtype, Builder2);

        Value *sendcount = gutils->getNewFromOriginal(orig_sendcount);
        if (!sendcount)
          sendcount = lookup(sendcount, Builder2);

        Value *sendtype = gutils->getNewFromOriginal(orig_sendtype);
        if (!forwardMode)
          sendtype = lookup(sendtype, Builder2);

        Value *root = gutils->getNewFromOriginal(orig_root);
        if (!forwardMode)
          root = lookup(root, Builder2);

        Value *comm = gutils->getNewFromOriginal(orig_comm);
        if (!forwardMode)
          comm = lookup(comm, Builder2);

        Value *rank = MPI_COMM_RANK(comm, Builder2, root->getType());
        Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType());

        if (forwardMode) {
          Value *args[] = {
              /*sendbuf*/ shadow_sendbuf,
              /*sendcount*/ sendcount,
              /*sendtype*/ sendtype,
              /*recvbuf*/ shadow_recvbuf,
              /*recvcount*/ recvcount,
              /*recvtype*/ recvtype,
              /*root*/ root,
              /*comm*/ comm,
          };

          auto Defs = gutils->getInvertedBundles(
              &call,
              {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Primal, ValueType::Primal},
              Builder2, /*lookup*/ false);

#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif

#if LLVM_VERSION_MAJOR > 7
          Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
#else
          Builder2.CreateCall(callval, args, Defs);
#endif

          return;
        }

        // Get the length for the allocation of the intermediate buffer
        auto sendlen_arg = Builder2.CreateZExtOrTrunc(
            sendcount, Type::getInt64Ty(call.getContext()));
        sendlen_arg =
            Builder2.CreateMul(sendlen_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);

        // Need to preserve the shadow send/recv buffers.
        auto BufferDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Primal, ValueType::Primal},
            Builder2, /*lookup*/ true);

        // 1. Alloc intermediate buffer
        Value *buf = CallInst::CreateMalloc(
            Builder2.GetInsertBlock(), sendlen_arg->getType(),
            Type::getInt8Ty(call.getContext()),
            ConstantInt::get(Type::getInt64Ty(sendlen_arg->getContext()), 1),
            sendlen_arg, nullptr, "mpireduce_malloccache");
        if (cast<Instruction>(buf)->getParent() == nullptr) {
          Builder2.Insert(cast<Instruction>(buf));
        }

        // 2. Scatter diff(recvbuffer) to intermediate buffer
        {
          // int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype
          // sendtype,
          //     void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
          //     MPI_Comm comm)
          Value *args[] = {
              /*sendbuf*/ shadow_recvbuf,
              /*sendcount*/ recvcount,
              /*sendtype*/ recvtype,
              /*recvbuf*/ buf,
              /*recvcount*/ sendcount,
              /*recvtype*/ sendtype,
              /*op*/ root,
              /*comm*/ comm,
          };
          Type *types[sizeof(args) / sizeof(*args)];
          for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
            types[i] = args[i]->getType();

          FunctionType *FT = FunctionType::get(call.getType(), types, false);
          Builder2.CreateCall(
              called->getParent()->getOrInsertFunction("MPI_Scatter", FT), args,
              BufferDefs);
        }

        // 3. if root, Zero diff(recvbuffer) [memset to 0]
        {

          BasicBlock *currentBlock = Builder2.GetInsertBlock();
          BasicBlock *rootBlock = gutils->addReverseBlock(
              currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
          BasicBlock *mergeBlock = gutils->addReverseBlock(
              rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

          Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                                mergeBlock);

          Builder2.SetInsertPoint(rootBlock);
          auto recvlen_arg = Builder2.CreateZExtOrTrunc(
              recvcount, Type::getInt64Ty(call.getContext()));
          recvlen_arg = Builder2.CreateMul(
              recvlen_arg,
              Builder2.CreateZExtOrTrunc(tysize,
                                         Type::getInt64Ty(call.getContext())),
              "", true, true);
          recvlen_arg = Builder2.CreateMul(
              recvlen_arg,
              Builder2.CreateZExtOrTrunc(
                  MPI_COMM_SIZE(comm, Builder2, root->getType()),
                  Type::getInt64Ty(call.getContext())),
              "", true, true);

          auto val_arg =
              ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
          auto volatile_arg = ConstantInt::getFalse(call.getContext());
          Value *args[] = {shadow_recvbuf, val_arg, recvlen_arg, volatile_arg};
          Type *tys[] = {args[0]->getType(), args[2]->getType()};
          auto memset = cast<CallInst>(Builder2.CreateCall(
              Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                        Intrinsic::memset, tys),
              args, BufferDefs));
          memset->addParamAttr(0, Attribute::NonNull);

          Builder2.CreateBr(mergeBlock);
          Builder2.SetInsertPoint(mergeBlock);
        }

        // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
        DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                    sendlen_arg, Builder2, BufferDefs);

        // Free up intermediate buffer
        if (shouldFree()) {
          auto ci = cast<CallInst>(
              CallInst::CreateFree(buf, Builder2.GetInsertBlock()));
#if LLVM_VERSION_MAJOR >= 14
          ci->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                  Attribute::NonNull);
#else
          ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
#endif
          if (ci->getParent() == nullptr) {
            Builder2.Insert(ci);
          }
        }
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // Approximate algo (for sum):  -> if statement yet to be
    // 1. if root, malloc intermediate buffer, else undef
    // 2. Gather diff(recvbuffer) to intermediate buffer
    // 3. Zero diff(recvbuffer) [memset to 0]
    // 4. if root, diff(sendbuffer) += intermediate buffer (diffmemcopy)
    // 5. if root, free intermediate buffer

    // int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype
    // sendtype,
    //           void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
    //           MPI_Comm comm)
    if (funcName == "MPI_Scatter") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ForwardMode) {
        bool forwardMode = Mode == DerivativeMode::ForwardMode;

        IRBuilder<> Builder2 =
            forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
        if (forwardMode) {
          getForwardBuilder(Builder2);
        } else {
          getReverseBuilder(Builder2);
        }

        Value *orig_sendbuf = call.getOperand(0);
        Value *orig_sendcount = call.getOperand(1);
        Value *orig_sendtype = call.getOperand(2);
        Value *orig_recvbuf = call.getOperand(3);
        Value *orig_recvcount = call.getOperand(4);
        Value *orig_recvtype = call.getOperand(5);
        Value *orig_root = call.getOperand(6);
        Value *orig_comm = call.getOperand(7);

        Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
        if (!forwardMode)
          shadow_recvbuf = lookup(shadow_recvbuf, Builder2);
        if (shadow_recvbuf->getType()->isIntegerTy())
          shadow_recvbuf = Builder2.CreateIntToPtr(
              shadow_recvbuf, Type::getInt8PtrTy(call.getContext()));

        Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
        if (!forwardMode)
          shadow_sendbuf = lookup(shadow_sendbuf, Builder2);
        if (shadow_sendbuf->getType()->isIntegerTy())
          shadow_sendbuf = Builder2.CreateIntToPtr(
              shadow_sendbuf, Type::getInt8PtrTy(call.getContext()));

        Value *recvcount = gutils->getNewFromOriginal(orig_recvcount);
        if (!forwardMode)
          recvcount = lookup(recvcount, Builder2);

        Value *recvtype = gutils->getNewFromOriginal(orig_recvtype);
        if (!forwardMode)
          recvtype = lookup(recvtype, Builder2);

        Value *sendcount = gutils->getNewFromOriginal(orig_sendcount);
        if (!forwardMode)
          sendcount = lookup(sendcount, Builder2);

        Value *sendtype = gutils->getNewFromOriginal(orig_sendtype);
        if (!forwardMode)
          sendtype = lookup(sendtype, Builder2);

        Value *root = gutils->getNewFromOriginal(orig_root);
        if (!forwardMode)
          root = lookup(root, Builder2);

        Value *comm = gutils->getNewFromOriginal(orig_comm);
        if (!forwardMode)
          comm = lookup(comm, Builder2);

        Value *rank = MPI_COMM_RANK(comm, Builder2, root->getType());
        Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType());

        if (forwardMode) {
          Value *args[] = {
              /*sendbuf*/ shadow_sendbuf,
              /*sendcount*/ sendcount,
              /*sendtype*/ sendtype,
              /*recvbuf*/ shadow_recvbuf,
              /*recvcount*/ recvcount,
              /*recvtype*/ recvtype,
              /*root*/ root,
              /*comm*/ comm,
          };

          auto Defs = gutils->getInvertedBundles(
              &call,
              {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Primal, ValueType::Primal},
              Builder2, /*lookup*/ false);

#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif

#if LLVM_VERSION_MAJOR > 7
          Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
#else
          Builder2.CreateCall(callval, args, Defs);
#endif

          return;
        }
        // Get the length for the allocation of the intermediate buffer
        auto recvlen_arg = Builder2.CreateZExtOrTrunc(
            recvcount, Type::getInt64Ty(call.getContext()));
        recvlen_arg =
            Builder2.CreateMul(recvlen_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);

        // Need to preserve the shadow send/recv buffers.
        auto BufferDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Primal, ValueType::Primal},
            Builder2, /*lookup*/ true);

        // 1. if root, malloc intermediate buffer, else undef
        PHINode *buf;
        PHINode *sendlen_phi;

        {
          BasicBlock *currentBlock = Builder2.GetInsertBlock();
          BasicBlock *rootBlock = gutils->addReverseBlock(
              currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
          BasicBlock *mergeBlock = gutils->addReverseBlock(
              rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

          Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                                mergeBlock);

          Builder2.SetInsertPoint(rootBlock);

          auto sendlen_arg = Builder2.CreateZExtOrTrunc(
              sendcount, Type::getInt64Ty(call.getContext()));
          sendlen_arg = Builder2.CreateMul(
              sendlen_arg,
              Builder2.CreateZExtOrTrunc(tysize,
                                         Type::getInt64Ty(call.getContext())),
              "", true, true);
          sendlen_arg = Builder2.CreateMul(
              sendlen_arg,
              Builder2.CreateZExtOrTrunc(
                  MPI_COMM_SIZE(comm, Builder2, root->getType()),
                  Type::getInt64Ty(call.getContext())),
              "", true, true);

          Value *rootbuf = CallInst::CreateMalloc(
              Builder2.GetInsertBlock(), sendlen_arg->getType(),
              Type::getInt8Ty(call.getContext()),
              ConstantInt::get(Type::getInt64Ty(sendlen_arg->getContext()), 1),
              sendlen_arg, nullptr, "mpireduce_malloccache");
          if (cast<Instruction>(rootbuf)->getParent() == nullptr) {
            Builder2.Insert(cast<Instruction>(rootbuf));
          }

          Builder2.CreateBr(mergeBlock);

          Builder2.SetInsertPoint(mergeBlock);

          buf = Builder2.CreatePHI(rootbuf->getType(), 2);
          buf->addIncoming(rootbuf, rootBlock);
          buf->addIncoming(UndefValue::get(buf->getType()), currentBlock);

          sendlen_phi = Builder2.CreatePHI(sendlen_arg->getType(), 2);
          sendlen_phi->addIncoming(sendlen_arg, rootBlock);
          sendlen_phi->addIncoming(UndefValue::get(sendlen_arg->getType()),
                                   currentBlock);
        }

        // 2. Gather diff(recvbuffer) to intermediate buffer
        {
          // int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype
          // sendtype,
          //     void *recvbuf, int recvcount, MPI_Datatype recvtype,
          //     int root, MPI_Comm comm)
          Value *args[] = {
              /*sendbuf*/ shadow_recvbuf,
              /*sendcount*/ recvcount,
              /*sendtype*/ recvtype,
              /*recvbuf*/ buf,
              /*recvcount*/ sendcount,
              /*recvtype*/ sendtype,
              /*root*/ root,
              /*comm*/ comm,
          };
          Type *types[sizeof(args) / sizeof(*args)];
          for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
            types[i] = args[i]->getType();

          FunctionType *FT = FunctionType::get(call.getType(), types, false);
          Builder2.CreateCall(
              called->getParent()->getOrInsertFunction("MPI_Gather", FT), args,
              BufferDefs);
        }

        // 3. Zero diff(recvbuffer) [memset to 0]
        {
          auto val_arg =
              ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
          auto volatile_arg = ConstantInt::getFalse(call.getContext());
          Value *args[] = {shadow_recvbuf, val_arg, recvlen_arg, volatile_arg};
          Type *tys[] = {args[0]->getType(), args[2]->getType()};
          auto memset = cast<CallInst>(Builder2.CreateCall(
              Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                        Intrinsic::memset, tys),
              args, BufferDefs));
          memset->addParamAttr(0, Attribute::NonNull);
        }

        // 4. if root, diff(sendbuffer) += intermediate buffer (diffmemcopy)
        // 5. if root, free intermediate buffer

        {
          BasicBlock *currentBlock = Builder2.GetInsertBlock();
          BasicBlock *rootBlock = gutils->addReverseBlock(
              currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
          BasicBlock *mergeBlock = gutils->addReverseBlock(
              rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

          Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                                mergeBlock);

          Builder2.SetInsertPoint(rootBlock);

          // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
          DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                      sendlen_phi, Builder2, BufferDefs);

          // Free up intermediate buffer
          if (shouldFree()) {
            auto ci = cast<CallInst>(
                CallInst::CreateFree(buf, Builder2.GetInsertBlock()));
#if LLVM_VERSION_MAJOR >= 14
            ci->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                    Attribute::NonNull);
#else
            ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
#endif
            if (ci->getParent() == nullptr) {
              Builder2.Insert(ci);
            }
          }

          Builder2.CreateBr(mergeBlock);
          Builder2.SetInsertPoint(mergeBlock);
        }
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // Approximate algo (for sum):  -> if statement yet to be
    // 1. malloc intermediate buffer
    // 2. reduce diff(recvbuffer) then scatter to corresponding input node's
    // intermediate buffer
    // 3. Zero diff(recvbuffer) [memset to 0]
    // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
    // 5. free intermediate buffer

    // int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype
    // sendtype,
    //           void *recvbuf, int recvcount, MPI_Datatype recvtype,
    //           MPI_Comm comm)

    if (funcName == "MPI_Allgather") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ForwardMode) {
        bool forwardMode = Mode == DerivativeMode::ForwardMode;

        IRBuilder<> Builder2 =
            forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
        if (forwardMode) {
          getForwardBuilder(Builder2);
        } else {
          getReverseBuilder(Builder2);
        }

        Value *orig_sendbuf = call.getOperand(0);
        Value *orig_sendcount = call.getOperand(1);
        Value *orig_sendtype = call.getOperand(2);
        Value *orig_recvbuf = call.getOperand(3);
        Value *orig_recvcount = call.getOperand(4);
        Value *orig_recvtype = call.getOperand(5);
        Value *orig_comm = call.getOperand(6);

        Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
        if (!forwardMode)
          shadow_recvbuf = lookup(shadow_recvbuf, Builder2);

        if (shadow_recvbuf->getType()->isIntegerTy())
          shadow_recvbuf = Builder2.CreateIntToPtr(
              shadow_recvbuf, Type::getInt8PtrTy(call.getContext()));

        Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
        if (!forwardMode)
          shadow_sendbuf = lookup(shadow_sendbuf, Builder2);

        if (shadow_sendbuf->getType()->isIntegerTy())
          shadow_sendbuf = Builder2.CreateIntToPtr(
              shadow_sendbuf, Type::getInt8PtrTy(call.getContext()));

        Value *recvcount = gutils->getNewFromOriginal(orig_recvcount);
        if (!forwardMode)
          recvcount = lookup(recvcount, Builder2);

        Value *recvtype = gutils->getNewFromOriginal(orig_recvtype);
        if (!forwardMode)
          recvtype = lookup(recvtype, Builder2);

        Value *sendcount = gutils->getNewFromOriginal(orig_sendcount);
        if (!forwardMode)
          sendcount = lookup(sendcount, Builder2);

        Value *sendtype = gutils->getNewFromOriginal(orig_sendtype);
        if (!forwardMode)
          sendtype = lookup(sendtype, Builder2);

        Value *comm = gutils->getNewFromOriginal(orig_comm);
        if (!forwardMode)
          comm = lookup(comm, Builder2);

        Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType());

        if (forwardMode) {
          Value *args[] = {
              /*sendbuf*/ shadow_sendbuf,
              /*sendcount*/ sendcount,
              /*sendtype*/ sendtype,
              /*recvbuf*/ shadow_recvbuf,
              /*recvcount*/ recvcount,
              /*recvtype*/ recvtype,
              /*comm*/ comm,
          };

          auto Defs = gutils->getInvertedBundles(
              &call,
              {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Primal},
              Builder2, /*lookup*/ false);

#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif

#if LLVM_VERSION_MAJOR > 7
          Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
#else
          Builder2.CreateCall(callval, args, Defs);
#endif

          return;
        }
        // Get the length for the allocation of the intermediate buffer
        auto sendlen_arg = Builder2.CreateZExtOrTrunc(
            sendcount, Type::getInt64Ty(call.getContext()));
        sendlen_arg =
            Builder2.CreateMul(sendlen_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);

        // Need to preserve the shadow send/recv buffers.
        auto BufferDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Primal},
            Builder2, /*lookup*/ true);

        // 1. Alloc intermediate buffer
        Value *buf = CallInst::CreateMalloc(
            Builder2.GetInsertBlock(), sendlen_arg->getType(),
            Type::getInt8Ty(call.getContext()),
            ConstantInt::get(Type::getInt64Ty(sendlen_arg->getContext()), 1),
            sendlen_arg, nullptr, "mpireduce_malloccache");
        if (cast<Instruction>(buf)->getParent() == nullptr) {
          Builder2.Insert(cast<Instruction>(buf));
        }

        ConcreteType CT = TR.firstPointer(1, orig_sendbuf);
        Type *MPI_OP_Ptr_type =
            PointerType::getUnqual(Type::getInt8PtrTy(call.getContext()));

        // 2. reduce diff(recvbuffer) then scatter to corresponding input node's
        // intermediate buffer
        {
          // int MPI_Reduce_scatter_block(const void* send_buffer,
          //                    void* receive_buffer,
          //                    int count,
          //                    MPI_Datatype datatype,
          //                    MPI_Op operation,
          //                    MPI_Comm communicator);
          Value *args[] = {
              /*sendbuf*/ shadow_recvbuf,
              /*recvbuf*/ buf,
              /*recvcount*/ sendcount,
              /*recvtype*/ sendtype,
              /*op (MPI_SUM)*/
              getOrInsertOpFloatSum(*gutils->newFunc->getParent(),
                                    MPI_OP_Ptr_type, CT, call.getType(),
                                    Builder2),
              /*comm*/ comm,
          };
          Type *types[sizeof(args) / sizeof(*args)];
          for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
            types[i] = args[i]->getType();

          FunctionType *FT = FunctionType::get(call.getType(), types, false);
          Builder2.CreateCall(called->getParent()->getOrInsertFunction(
                                  "MPI_Reduce_scatter_block", FT),
                              args, BufferDefs);
        }

        // 3. zero diff(recvbuffer) [memset to 0]
        {
          auto recvlen_arg = Builder2.CreateZExtOrTrunc(
              recvcount, Type::getInt64Ty(call.getContext()));
          recvlen_arg = Builder2.CreateMul(
              recvlen_arg,
              Builder2.CreateZExtOrTrunc(tysize,
                                         Type::getInt64Ty(call.getContext())),
              "", true, true);
          recvlen_arg = Builder2.CreateMul(
              recvlen_arg,
              Builder2.CreateZExtOrTrunc(
                  MPI_COMM_SIZE(comm, Builder2, call.getType()),
                  Type::getInt64Ty(call.getContext())),
              "", true, true);
          auto val_arg =
              ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
          auto volatile_arg = ConstantInt::getFalse(call.getContext());
          Value *args[] = {shadow_recvbuf, val_arg, recvlen_arg, volatile_arg};
          Type *tys[] = {args[0]->getType(), args[2]->getType()};
          auto memset = cast<CallInst>(Builder2.CreateCall(
              Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                        Intrinsic::memset, tys),
              args, BufferDefs));
          memset->addParamAttr(0, Attribute::NonNull);
        }

        // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
        DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                    sendlen_arg, Builder2, BufferDefs);

        // Free up intermediate buffer
        if (shouldFree()) {
          auto ci = cast<CallInst>(
              CallInst::CreateFree(buf, Builder2.GetInsertBlock()));
#if LLVM_VERSION_MAJOR >= 14
          ci->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                  Attribute::NonNull);
#else
          ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
#endif
          if (ci->getParent() == nullptr) {
            Builder2.Insert(ci);
          }
        }
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // Adjoint of barrier is to place a barrier at the corresponding
    // location in the reverse.
    if (funcName == "MPI_Barrier") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);
#if LLVM_VERSION_MAJOR >= 11
        auto callval = call.getCalledOperand();
#else
        auto callval = call.getCalledValue();
#endif
        Value *args[] = {
            lookup(gutils->getNewFromOriginal(call.getOperand(0)), Builder2)};
        Builder2.CreateCall(call.getFunctionType(), callval, args);
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // Remove free's in forward pass so the comm can be used in the reverse
    // pass
    if (funcName == "MPI_Comm_free" || funcName == "MPI_Comm_disconnect") {
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // Adjoint of MPI_Comm_split / MPI_Graph_create (which allocates a comm in a
    // pointer) is to free the created comm at the corresponding place in the
    // reverse pass
    auto commFound = MPIInactiveCommAllocators.find(funcName.str());
    if (commFound != MPIInactiveCommAllocators.end()) {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);

        Value *args[] = {lookup(call.getOperand(commFound->second), Builder2)};
        Type *types[] = {args[0]->getType()};

        FunctionType *FT = FunctionType::get(call.getType(), types, false);
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction("MPI_Comm_free", FT),
            args);
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    llvm::errs() << *gutils->oldFunc->getParent() << "\n";
    llvm::errs() << *gutils->oldFunc << "\n";
    llvm::errs() << call << "\n";
    llvm::errs() << called << "\n";
    llvm_unreachable("Unhandled MPI FUNCTION");
  }

  // Return
  void visitCallInst(llvm::CallInst &call) {
    CallInst *const newCall = cast<CallInst>(gutils->getNewFromOriginal(&call));
    IRBuilder<> BuilderZ(newCall);
    BuilderZ.setFastMathFlags(getFast());

    if (uncacheable_args_map.find(&call) == uncacheable_args_map.end() &&
        Mode != DerivativeMode::ForwardMode) {
      llvm::errs() << " call: " << call << "\n";
      for (auto &pair : uncacheable_args_map) {
        llvm::errs() << " + " << *pair.first << "\n";
      }
    }

    assert(uncacheable_args_map.find(&call) != uncacheable_args_map.end() ||
           Mode == DerivativeMode::ForwardMode);
    const std::map<Argument *, bool> &uncacheable_args =
        Mode == DerivativeMode::ForwardMode
            ? std::map<Argument *, bool>()
            : uncacheable_args_map.find(&call)->second;

    CallInst *orig = &call;

    Function *called = getFunctionFromCall(orig);

    StringRef funcName = "";
    if (called) {
      if (called->hasFnAttribute("enzyme_math"))
        funcName = called->getFnAttribute("enzyme_math").getValueAsString();
      else
        funcName = called->getName();
    }

    bool subretused = unnecessaryValues.find(orig) == unnecessaryValues.end();
    if (gutils->knownRecomputeHeuristic.find(orig) !=
        gutils->knownRecomputeHeuristic.end()) {
      if (!gutils->knownRecomputeHeuristic[orig]) {
        subretused = true;
      }
    }
    bool shadowReturnUsed = false;

    DIFFE_TYPE subretType;
    if (gutils->isConstantValue(orig)) {
      subretType = DIFFE_TYPE::CONSTANT;
    } else {
      if (Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeSplit) {
        subretType = DIFFE_TYPE::DUP_ARG;
        shadowReturnUsed = true;
      } else {
        if (!orig->getType()->isFPOrFPVectorTy() &&
            TR.query(orig).Inner0().isPossiblePointer()) {
          if (is_value_needed_in_reverse<ValueType::Shadow>(gutils, orig, Mode,
                                                            oldUnreachable)) {
            subretType = DIFFE_TYPE::DUP_ARG;
            shadowReturnUsed = true;
          } else
            subretType = DIFFE_TYPE::CONSTANT;
        } else {
          subretType = DIFFE_TYPE::OUT_DIFF;
        }
      }
    }

    if (Mode == DerivativeMode::ForwardMode) {
      auto found = customFwdCallHandlers.find(funcName.str());
      if (found != customFwdCallHandlers.end()) {
        Value *invertedReturn = nullptr;
        auto ifound = gutils->invertedPointers.find(orig);
        if (ifound != gutils->invertedPointers.end()) {
          invertedReturn = cast<PHINode>(&*ifound->second);
        }

        Value *normalReturn = subretused ? newCall : nullptr;

        found->second(BuilderZ, orig, *gutils, normalReturn, invertedReturn);

        if (ifound != gutils->invertedPointers.end()) {
          auto placeholder = cast<PHINode>(&*ifound->second);
          if (invertedReturn && invertedReturn != placeholder) {
            if (invertedReturn->getType() !=
                gutils->getShadowType(orig->getType())) {
              llvm::errs() << " o: " << *orig << "\n";
              llvm::errs() << " ot: " << *orig->getType() << "\n";
              llvm::errs() << " ir: " << *invertedReturn << "\n";
              llvm::errs() << " irt: " << *invertedReturn->getType() << "\n";
              llvm::errs() << " p: " << *placeholder << "\n";
              llvm::errs() << " PT: " << *placeholder->getType() << "\n";
              llvm::errs() << " newCall: " << *newCall << "\n";
              llvm::errs() << " newCallT: " << *newCall->getType() << "\n";
            }
            assert(invertedReturn->getType() ==
                   gutils->getShadowType(orig->getType()));
            placeholder->replaceAllUsesWith(invertedReturn);
            gutils->erase(placeholder);
            gutils->invertedPointers.insert(
                std::make_pair((const Value *)orig,
                               InvertedPointerVH(gutils, invertedReturn)));
          } else {
            gutils->invertedPointers.erase(orig);
            gutils->erase(placeholder);
          }
        }

        if (normalReturn && normalReturn != newCall) {
          assert(normalReturn->getType() == newCall->getType());
          gutils->replaceAWithB(newCall, normalReturn);
          gutils->erase(newCall);
        }
        return;
      }
    }

    if (Mode == DerivativeMode::ReverseModePrimal ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ReverseModeGradient) {
      auto found = customCallHandlers.find(funcName.str());
      if (found != customCallHandlers.end()) {
        IRBuilder<> Builder2(call.getParent());
        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ReverseModeCombined)
          getReverseBuilder(Builder2);

        Value *invertedReturn = nullptr;
        bool hasNonReturnUse = false;
        auto ifound = gutils->invertedPointers.find(orig);
        if (ifound != gutils->invertedPointers.end()) {
          //! We only need the shadow pointer for non-forward Mode if it is used
          //! in a non return setting
          if (!gutils->isConstantValue(orig)) {
            if (!orig->getType()->isFPOrFPVectorTy() &&
                TR.query(orig).Inner0().isPossiblePointer()) {
              if (is_value_needed_in_reverse<ValueType::Shadow>(
                      gutils, orig, DerivativeMode::ReverseModePrimal,
                      oldUnreachable)) {
                hasNonReturnUse = true;
              }
            }
          }
          if (hasNonReturnUse)
            invertedReturn = cast<PHINode>(&*ifound->second);
        }

        Value *normalReturn = subretused ? newCall : nullptr;

        Value *tape = nullptr;

        if (Mode == DerivativeMode::ReverseModePrimal ||
            Mode == DerivativeMode::ReverseModeCombined) {
          found->second.first(BuilderZ, orig, *gutils, normalReturn,
                              invertedReturn, tape);
          if (tape)
            gutils->cacheForReverse(BuilderZ, tape,
                                    getIndex(orig, CacheType::Tape));
        }

        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ReverseModeCombined) {
          if (Mode == DerivativeMode::ReverseModeGradient &&
              augmentedReturn->tapeIndices.find(
                  std::make_pair(orig, CacheType::Tape)) !=
                  augmentedReturn->tapeIndices.end()) {
            tape = BuilderZ.CreatePHI(Type::getInt32Ty(orig->getContext()), 0);
            tape = gutils->cacheForReverse(BuilderZ, tape,
                                           getIndex(orig, CacheType::Tape),
                                           /*ignoreType*/ true);
          }
          if (tape)
            tape = gutils->lookupM(tape, Builder2);
          found->second.second(Builder2, orig, *(DiffeGradientUtils *)gutils,
                               tape);
        }

        if (ifound != gutils->invertedPointers.end()) {
          auto placeholder = cast<PHINode>(&*ifound->second);
          if (!hasNonReturnUse) {
            gutils->invertedPointers.erase(ifound);
            gutils->erase(placeholder);
          } else {
            if (invertedReturn && invertedReturn != placeholder) {
              if (invertedReturn->getType() !=
                  gutils->getShadowType(orig->getType())) {
                llvm::errs() << " o: " << *orig << "\n";
                llvm::errs() << " ot: " << *orig->getType() << "\n";
                llvm::errs() << " ir: " << *invertedReturn << "\n";
                llvm::errs() << " irt: " << *invertedReturn->getType() << "\n";
                llvm::errs() << " p: " << *placeholder << "\n";
                llvm::errs() << " PT: " << *placeholder->getType() << "\n";
                llvm::errs() << " newCall: " << *newCall << "\n";
                llvm::errs() << " newCallT: " << *newCall->getType() << "\n";
              }
              assert(invertedReturn->getType() ==
                     gutils->getShadowType(orig->getType()));
              placeholder->replaceAllUsesWith(invertedReturn);
              gutils->erase(placeholder);
            } else
              invertedReturn = placeholder;

            invertedReturn = gutils->cacheForReverse(
                BuilderZ, invertedReturn, getIndex(orig, CacheType::Shadow));

            gutils->invertedPointers.insert(
                std::make_pair((const Value *)orig,
                               InvertedPointerVH(gutils, invertedReturn)));
          }
        }

        bool primalNeededInReverse;

        if (gutils->knownRecomputeHeuristic.count(orig)) {
          primalNeededInReverse = !gutils->knownRecomputeHeuristic[orig];
        } else {
          std::map<UsageKey, bool> Seen;
          for (auto pair : gutils->knownRecomputeHeuristic)
            if (!pair.second)
              Seen[UsageKey(pair.first, ValueType::Primal)] = false;
          primalNeededInReverse = is_value_needed_in_reverse<ValueType::Primal>(
              gutils, orig, Mode, Seen, oldUnreachable);
        }
        if (subretused && primalNeededInReverse) {
          if (normalReturn != newCall) {
            assert(normalReturn->getType() == newCall->getType());
            gutils->replaceAWithB(newCall, normalReturn);
            BuilderZ.SetInsertPoint(newCall->getNextNode());
            gutils->erase(newCall);
          }
          normalReturn = gutils->cacheForReverse(
              BuilderZ, normalReturn, getIndex(orig, CacheType::Self));
        } else {
          if (normalReturn && normalReturn != newCall) {
            assert(normalReturn->getType() == newCall->getType());
            assert(Mode != DerivativeMode::ReverseModeGradient);
            gutils->replaceAWithB(newCall, normalReturn);
            BuilderZ.SetInsertPoint(newCall->getNextNode());
            gutils->erase(newCall);
          } else if ((!orig->mayWriteToMemory() ||
                      Mode == DerivativeMode::ReverseModeGradient) &&
                     !orig->getType()->isTokenTy())
            eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
        }
        return;
      }
    }

    if (Mode != DerivativeMode::ReverseModePrimal && called) {
      if (funcName == "__kmpc_for_static_init_4" ||
          funcName == "__kmpc_for_static_init_4u" ||
          funcName == "__kmpc_for_static_init_8" ||
          funcName == "__kmpc_for_static_init_8u") {
        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);
        auto fini = called->getParent()->getFunction("__kmpc_for_static_fini");
        assert(fini);
        Value *args[] = {
            lookup(gutils->getNewFromOriginal(call.getArgOperand(0)), Builder2),
            lookup(gutils->getNewFromOriginal(call.getArgOperand(1)),
                   Builder2)};
        auto fcall = Builder2.CreateCall(fini->getFunctionType(), fini, args);
        fcall->setCallingConv(fini->getCallingConv());
        return;
      }
    }

    if ((funcName.startswith("MPI_") || funcName.startswith("PMPI_")) &&
        (!gutils->isConstantInstruction(&call) || funcName == "MPI_Barrier" ||
         funcName == "MPI_Comm_free" || funcName == "MPI_Comm_disconnect" ||
         MPIInactiveCommAllocators.find(funcName.str()) !=
             MPIInactiveCommAllocators.end())) {
      handleMPI(call, called, funcName);
      return;
    }

    if (!called || called->empty()) {
      std::string prefix, suffix;
      std::string found = extractBLAS(funcName, prefix, suffix);
      if (found.size()) {
        if (handleBLAS(call, called, found, prefix, suffix, uncacheable_args))
          return;
      }
    }

    if (funcName == "printf" || funcName == "puts" ||
        funcName.startswith("_ZN3std2io5stdio6_print") ||
        funcName.startswith("_ZN4core3fmt")) {
      if (Mode == DerivativeMode::ReverseModeGradient) {
        eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
      }
      return;
    }
    if (called && (called->getName().contains("__enzyme_float") ||
                   called->getName().contains("__enzyme_double") ||
                   called->getName().contains("__enzyme_integer") ||
                   called->getName().contains("__enzyme_pointer"))) {
      eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
      return;
    }

    // Handle lgamma, safe to recompute so no store/change to forward
    if (called) {
      if (funcName == "__kmpc_fork_call") {
        visitOMPCall(call);
        return;
      }

      if (funcName == "__kmpc_for_static_init_4" ||
          funcName == "__kmpc_for_static_init_4u" ||
          funcName == "__kmpc_for_static_init_8" ||
          funcName == "__kmpc_for_static_init_8u") {
        if (Mode != DerivativeMode::ReverseModePrimal) {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);
          auto fini =
              called->getParent()->getFunction("__kmpc_for_static_fini");
          assert(fini);
          Value *args[] = {
              lookup(gutils->getNewFromOriginal(call.getArgOperand(0)),
                     Builder2),
              lookup(gutils->getNewFromOriginal(call.getArgOperand(1)),
                     Builder2)};
          auto fcall = Builder2.CreateCall(fini->getFunctionType(), fini, args);
          fcall->setCallingConv(fini->getCallingConv());
        }
        return;
      }
      if (funcName == "__kmpc_for_static_fini") {
        if (Mode != DerivativeMode::ReverseModePrimal) {
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        }
        return;
      }
      // TODO check
      // Adjoint of barrier is to place a barrier at the corresponding
      // location in the reverse.
      if (funcName == "__kmpc_barrier") {
        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ReverseModeCombined) {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);
#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif
          Value *args[] = {
              lookup(gutils->getNewFromOriginal(call.getOperand(0)), Builder2),
              lookup(gutils->getNewFromOriginal(call.getOperand(1)), Builder2)};
          Builder2.CreateCall(call.getFunctionType(), callval, args);
        }
        return;
      }
      if (funcName == "__kmpc_critical") {
        if (Mode != DerivativeMode::ReverseModePrimal) {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);
          auto crit2 = called->getParent()->getFunction("__kmpc_end_critical");
          assert(crit2);
          Value *args[] = {
              lookup(gutils->getNewFromOriginal(call.getArgOperand(0)),
                     Builder2),
              lookup(gutils->getNewFromOriginal(call.getArgOperand(1)),
                     Builder2),
              lookup(gutils->getNewFromOriginal(call.getArgOperand(2)),
                     Builder2)};
          auto fcall =
              Builder2.CreateCall(crit2->getFunctionType(), crit2, args);
          fcall->setCallingConv(crit2->getCallingConv());
        }
        return;
      }
      if (funcName == "__kmpc_end_critical") {
        if (Mode != DerivativeMode::ReverseModePrimal) {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);
          auto crit2 = called->getParent()->getFunction("__kmpc_critical");
          assert(crit2);
          Value *args[] = {
              lookup(gutils->getNewFromOriginal(call.getArgOperand(0)),
                     Builder2),
              lookup(gutils->getNewFromOriginal(call.getArgOperand(1)),
                     Builder2),
              lookup(gutils->getNewFromOriginal(call.getArgOperand(2)),
                     Builder2)};
          auto fcall =
              Builder2.CreateCall(crit2->getFunctionType(), crit2, args);
          fcall->setCallingConv(crit2->getCallingConv());
        }
        return;
      }

      if (funcName.startswith("__kmpc") &&
          funcName != "__kmpc_global_thread_num") {
        llvm::errs() << *gutils->oldFunc << "\n";
        llvm::errs() << call << "\n";
        assert(0 && "unhandled openmp function");
        llvm_unreachable("unhandled openmp function");
      }

      if (funcName == "log1p" || funcName == "log1pf" || funcName == "log1pl") {
        if (gutils->knownRecomputeHeuristic.find(orig) !=
            gutils->knownRecomputeHeuristic.end()) {
          if (!gutils->knownRecomputeHeuristic[orig]) {
            gutils->cacheForReverse(BuilderZ, newCall,
                                    getIndex(orig, CacheType::Self));
          }
        }
        eraseIfUnused(*orig);
        if (gutils->isConstantInstruction(orig))
          return;

        switch (Mode) {
        case DerivativeMode::ForwardModeSplit:
        case DerivativeMode::ForwardMode: {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);
          Value *x = gutils->getNewFromOriginal(orig->getArgOperand(0));
          Value *onePx =
              Builder2.CreateFAdd(ConstantFP::get(x->getType(), 1.0), x);

          Value *op = diffe(orig->getArgOperand(0), Builder2);

          auto rule = [&](Value *op) { return Builder2.CreateFDiv(op, onePx); };
          Value *dif0 = applyChainRule(call.getType(), Builder2, rule, op);
          setDiffe(orig, dif0, Builder2);
          return;
        }
        case DerivativeMode::ReverseModeGradient:
        case DerivativeMode::ReverseModeCombined: {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);
          Value *x = lookup(gutils->getNewFromOriginal(orig->getArgOperand(0)),
                            Builder2);
          Value *onePx =
              Builder2.CreateFAdd(ConstantFP::get(x->getType(), 1.0), x);

          auto rule = [&](Value *dorig) {
            return Builder2.CreateFDiv(dorig, onePx);
          };

          Value *dorig = diffe(orig, Builder2);
          Value *dif0 = applyChainRule(orig->getArgOperand(0)->getType(),
                                       Builder2, rule, dorig);

          addToDiffe(orig->getArgOperand(0), dif0, Builder2, x->getType());
          return;
        }
        case DerivativeMode::ReverseModePrimal: {
          return;
        }
        }
      }

      if (funcName == "asin" || funcName == "asinf" || funcName == "asinl") {
        if (gutils->knownRecomputeHeuristic.find(orig) !=
            gutils->knownRecomputeHeuristic.end()) {
          if (!gutils->knownRecomputeHeuristic[orig]) {
            gutils->cacheForReverse(BuilderZ, newCall,
                                    getIndex(orig, CacheType::Self));
          }
        }
        eraseIfUnused(*orig);
        if (gutils->isConstantInstruction(orig))
          return;

        switch (Mode) {
        case DerivativeMode::ForwardModeSplit:
        case DerivativeMode::ForwardMode: {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);
          Value *x = gutils->getNewFromOriginal(orig->getArgOperand(0));
          Value *oneMx2 = Builder2.CreateFSub(
              ConstantFP::get(x->getType(), 1.0), Builder2.CreateFMul(x, x));

          SmallVector<Value *, 1> args = {oneMx2};
          Type *tys[] = {x->getType()};
          auto cal = cast<CallInst>(Builder2.CreateCall(
              Intrinsic::getDeclaration(called->getParent(), Intrinsic::sqrt,
                                        tys),
              args));

          Value *op = diffe(orig->getArgOperand(0), Builder2);

          auto rule = [&](Value *op) { return Builder2.CreateFDiv(op, cal); };

          Value *dif0 = applyChainRule(call.getType(), Builder2, rule, op);
          setDiffe(orig, dif0, Builder2);
          return;
        }
        case DerivativeMode::ReverseModeGradient:
        case DerivativeMode::ReverseModeCombined: {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);
          Value *x = lookup(gutils->getNewFromOriginal(orig->getArgOperand(0)),
                            Builder2);
          Value *oneMx2 = Builder2.CreateFSub(
              ConstantFP::get(x->getType(), 1.0), Builder2.CreateFMul(x, x));

          SmallVector<Value *, 1> args = {oneMx2};
          Type *tys[] = {x->getType()};
          auto cal = cast<CallInst>(Builder2.CreateCall(
              Intrinsic::getDeclaration(called->getParent(), Intrinsic::sqrt,
                                        tys),
              args));

          auto rule = [&](Value *dorig) {
            return Builder2.CreateFDiv(dorig, cal);
          };

          Value *dorig = diffe(orig, Builder2);
          Value *dif0 = applyChainRule(orig->getArgOperand(0)->getType(),
                                       Builder2, rule, dorig);

          addToDiffe(orig->getArgOperand(0), dif0, Builder2, x->getType());
          return;
        }
        case DerivativeMode::ReverseModePrimal: {
          return;
        }
        }
      }

#include "InstructionDerivatives.inc"

      // Functions that only modify pointers and don't allocate memory,
      // needs to be run on shadow in primal
      if (funcName == "_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_"
                      "node_baseS0_RS_") {
        if (Mode == DerivativeMode::ReverseModeGradient) {
          eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
          return;
        }
        if (gutils->isConstantValue(orig->getArgOperand(3)))
          return;
        SmallVector<Value *, 2> args;
#if LLVM_VERSION_MAJOR >= 14
        for (auto &arg : call.args())
#else
        for (auto &arg : call.arg_operands())
#endif
        {
          if (gutils->isConstantValue(arg))
            args.push_back(gutils->getNewFromOriginal(arg));
          else
            args.push_back(gutils->invertPointerM(arg, BuilderZ));
        }
        BuilderZ.CreateCall(called, args);
        return;
      }

      // Functions that initialize a shadow data structure (with no
      // other arguments) needs to be run on shadow in primal.
      if (funcName == "_ZNSt8ios_baseC2Ev" ||
          funcName == "_ZNSt8ios_baseD2Ev" || funcName == "_ZNSt6localeC1Ev" ||
          funcName == "_ZNSt6localeD1Ev" ||
          funcName == "_ZNKSt5ctypeIcE13_M_widen_initEv") {
        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ForwardModeSplit) {
          eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
          return;
        }
        if (gutils->isConstantValue(orig->getArgOperand(0)))
          return;
        Value *args[] = {
            gutils->invertPointerM(orig->getArgOperand(0), BuilderZ)};
        BuilderZ.CreateCall(called, args);
        return;
      }

      if (funcName == "_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_"
                      "streambufIcS1_E") {
        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ForwardModeSplit) {
          eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
          return;
        }
        if (gutils->isConstantValue(orig->getArgOperand(0)))
          return;
        Value *args[] = {
            gutils->invertPointerM(orig->getArgOperand(0), BuilderZ),
            gutils->invertPointerM(orig->getArgOperand(1), BuilderZ)};
        BuilderZ.CreateCall(called, args);
        return;
      }

      // if constant instruction and readonly (thus must be pointer return)
      // and shadow return recomputable from shadow arguments.
      if (funcName == "__dynamic_cast" ||
          funcName == "_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base" ||
          funcName == "_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base") {
        bool shouldCache = false;
        if (gutils->knownRecomputeHeuristic.find(orig) !=
            gutils->knownRecomputeHeuristic.end()) {
          if (!gutils->knownRecomputeHeuristic[orig]) {
            shouldCache = true;
          }
        }
        ValueToValueMapTy empty;
        bool lrc = gutils->legalRecompute(orig, empty, nullptr);

        if (!gutils->isConstantValue(orig)) {
          auto ifound = gutils->invertedPointers.find(orig);
          assert(ifound != gutils->invertedPointers.end());
          auto placeholder = cast<PHINode>(&*ifound->second);
          gutils->invertedPointers.erase(ifound);

          if (subretType == DIFFE_TYPE::DUP_ARG) {
            Value *shadow = placeholder;
            if (lrc || Mode == DerivativeMode::ReverseModePrimal ||
                Mode == DerivativeMode::ReverseModeCombined ||
                Mode == DerivativeMode::ForwardMode) {
              if (gutils->isConstantValue(orig->getArgOperand(0)))
                shadow = gutils->getNewFromOriginal(orig);
              else {
                SmallVector<Value *, 2> args;
                size_t i = 0;
#if LLVM_VERSION_MAJOR >= 14
                for (auto &arg : call.args())
#else
                for (auto &arg : call.arg_operands())
#endif
                {
                  if (gutils->isConstantValue(arg) ||
                      (funcName == "__dynamic_cast" && i > 0))
                    args.push_back(gutils->getNewFromOriginal(arg));
                  else
                    args.push_back(gutils->invertPointerM(arg, BuilderZ));
                  i++;
                }
                shadow = BuilderZ.CreateCall(called, args);
              }
            }

            bool needsReplacement = true;
            if (!lrc && (Mode == DerivativeMode::ReverseModePrimal ||
                         Mode == DerivativeMode::ReverseModeGradient)) {
              shadow = gutils->cacheForReverse(
                  BuilderZ, shadow, getIndex(orig, CacheType::Shadow));
              if (Mode == DerivativeMode::ReverseModeGradient)
                needsReplacement = false;
            }
            gutils->invertedPointers.insert(std::make_pair(
                (const Value *)orig, InvertedPointerVH(gutils, shadow)));
            if (needsReplacement) {
              assert(shadow != placeholder);
              gutils->replaceAWithB(placeholder, shadow);
              gutils->erase(placeholder);
            }
          } else {
            gutils->erase(placeholder);
          }
        }

        if (Mode == DerivativeMode::ForwardMode) {
          eraseIfUnused(*orig);
          assert(gutils->isConstantInstruction(orig));
          return;
        }

        if (!shouldCache && !lrc) {
          std::map<UsageKey, bool> Seen;
          for (auto pair : gutils->knownRecomputeHeuristic)
            Seen[UsageKey(pair.first, ValueType::Primal)] = false;
          bool primalNeededInReverse =
              is_value_needed_in_reverse<ValueType::Primal>(
                  gutils, orig, Mode, Seen, oldUnreachable);
          shouldCache = primalNeededInReverse;
        }

        if (shouldCache) {
          BuilderZ.SetInsertPoint(newCall->getNextNode());
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(orig, CacheType::Self));
        }
        eraseIfUnused(*orig);
        assert(gutils->isConstantInstruction(orig));
        return;
      }

      if (funcName == "__mulsc3" || funcName == "__muldc3" ||
          funcName == "__multc3" || funcName == "__mulxc3") {
        if (gutils->knownRecomputeHeuristic.find(orig) !=
            gutils->knownRecomputeHeuristic.end()) {
          if (!gutils->knownRecomputeHeuristic[orig]) {
            gutils->cacheForReverse(BuilderZ, newCall,
                                    getIndex(orig, CacheType::Self));
          }
        }

        eraseIfUnused(*orig);
        if (gutils->isConstantInstruction(orig))
          return;

        Value *orig_op0 = call.getOperand(0);
        Value *orig_op1 = call.getOperand(1);
        Value *orig_op2 = call.getOperand(2);
        Value *orig_op3 = call.getOperand(3);

        bool constantval0 = gutils->isConstantValue(orig_op0);
        bool constantval1 = gutils->isConstantValue(orig_op1);
        bool constantval2 = gutils->isConstantValue(orig_op2);
        bool constantval3 = gutils->isConstantValue(orig_op3);

        Value *prim[4] = {gutils->getNewFromOriginal(orig_op0),
                          gutils->getNewFromOriginal(orig_op1),
                          gutils->getNewFromOriginal(orig_op2),
                          gutils->getNewFromOriginal(orig_op3)};

        auto mul = gutils->oldFunc->getParent()->getOrInsertFunction(
            funcName, called->getFunctionType(), called->getAttributes());

        switch (Mode) {
        case DerivativeMode::ForwardMode:
        case DerivativeMode::ForwardModeSplit: {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);

          Value *diff[4] = {
              constantval0 ? Constant::getNullValue(orig_op0->getType())
                           : diffe(orig_op0, Builder2),
              constantval1 ? Constant::getNullValue(orig_op1->getType())
                           : diffe(orig_op1, Builder2),
              constantval2 ? Constant::getNullValue(orig_op2->getType())
                           : diffe(orig_op2, Builder2),
              constantval3 ? Constant::getNullValue(orig_op3->getType())
                           : diffe(orig_op3, Builder2)};

          auto cal1 =
              Builder2.CreateCall(mul, {diff[0], diff[1], prim[2], prim[3]});
          auto cal2 =
              Builder2.CreateCall(mul, {prim[0], prim[1], diff[2], diff[3]});

          Value *resReal =
              Builder2.CreateFAdd(Builder2.CreateExtractValue(cal1, {0}),
                                  Builder2.CreateExtractValue(cal2, {0}));
          Value *resImag =
              Builder2.CreateFAdd(Builder2.CreateExtractValue(cal1, {1}),
                                  Builder2.CreateExtractValue(cal2, {1}));

          Value *res = Builder2.CreateInsertValue(
              UndefValue::get(call.getType()), resReal, {0});
          res = Builder2.CreateInsertValue(res, resImag, {1});

          setDiffe(&call, res, Builder2);
          return;
        }
        case DerivativeMode::ReverseModeGradient:
        case DerivativeMode::ReverseModeCombined: {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);

          Value *idiff = diffe(&call, Builder2);
          Value *idiffReal = Builder2.CreateExtractValue(idiff, {0});
          Value *idiffImag = Builder2.CreateExtractValue(idiff, {1});

          Value *diff0 = nullptr;
          Value *diff1 = nullptr;

          if (!constantval0 || !constantval1)
            diff0 = Builder2.CreateCall(mul, {idiffReal, idiffImag,
                                              lookup(prim[2], Builder2),
                                              lookup(prim[3], Builder2)});

          if (!constantval2 || !constantval3)
            diff1 = Builder2.CreateCall(mul, {lookup(prim[0], Builder2),
                                              lookup(prim[1], Builder2),
                                              idiffReal, idiffImag});

          if (diff0 || diff1)
            setDiffe(&call, Constant::getNullValue(call.getType()), Builder2);

          if (diff0) {
            addToDiffe(orig_op0, Builder2.CreateExtractValue(diff0, {0}),
                       Builder2, orig_op0->getType());
            addToDiffe(orig_op1, Builder2.CreateExtractValue(diff0, {1}),
                       Builder2, orig_op1->getType());
          }

          if (diff1) {
            addToDiffe(orig_op2, Builder2.CreateExtractValue(diff1, {0}),
                       Builder2, orig_op2->getType());
            addToDiffe(orig_op3, Builder2.CreateExtractValue(diff1, {1}),
                       Builder2, orig_op3->getType());
          }

          return;
        }
        case DerivativeMode::ReverseModePrimal:
          return;
        }
      }

      if (funcName == "__divsc3" || funcName == "__divdc3" ||
          funcName == "__divtc3" || funcName == "__divxc3") {
        if (gutils->knownRecomputeHeuristic.find(orig) !=
            gutils->knownRecomputeHeuristic.end()) {
          if (!gutils->knownRecomputeHeuristic[orig]) {
            gutils->cacheForReverse(BuilderZ, newCall,
                                    getIndex(orig, CacheType::Self));
          }
        }

        if (gutils->isConstantInstruction(orig))
          return;

        StringMap<StringRef> map = {
            {"__divsc3", "__mulsc3"},
            {"__divdc3", "__muldc3"},
            {"__divtc3", "__multc3"},
            {"__divxc3", "__mulxc3"},
        };

        auto mul = gutils->oldFunc->getParent()->getOrInsertFunction(
            map[funcName], called->getFunctionType(), called->getAttributes());

        auto div = gutils->oldFunc->getParent()->getOrInsertFunction(
            funcName, called->getFunctionType(), called->getAttributes());

        Value *orig_op0 = call.getOperand(0);
        Value *orig_op1 = call.getOperand(1);
        Value *orig_op2 = call.getOperand(2);
        Value *orig_op3 = call.getOperand(3);

        bool constantval0 = gutils->isConstantValue(orig_op0);
        bool constantval1 = gutils->isConstantValue(orig_op1);
        bool constantval2 = gutils->isConstantValue(orig_op2);
        bool constantval3 = gutils->isConstantValue(orig_op3);

        Value *prim[4] = {gutils->getNewFromOriginal(orig_op0),
                          gutils->getNewFromOriginal(orig_op1),
                          gutils->getNewFromOriginal(orig_op2),
                          gutils->getNewFromOriginal(orig_op3)};

        switch (Mode) {
        case DerivativeMode::ForwardMode:
        case DerivativeMode::ForwardModeSplit: {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);

          Value *diff[4] = {
              constantval0 ? Constant::getNullValue(orig_op0->getType())
                           : diffe(orig_op0, Builder2),
              constantval1 ? Constant::getNullValue(orig_op1->getType())
                           : diffe(orig_op1, Builder2),
              constantval2 ? Constant::getNullValue(orig_op2->getType())
                           : diffe(orig_op2, Builder2),
              constantval3 ? Constant::getNullValue(orig_op3->getType())
                           : diffe(orig_op3, Builder2)};

          auto mul1 =
              Builder2.CreateCall(mul, {diff[0], diff[1], prim[2], prim[3]});
          auto mul2 =
              Builder2.CreateCall(mul, {prim[0], prim[1], diff[2], diff[3]});
          auto sq1 =
              Builder2.CreateCall(mul, {prim[2], prim[3], prim[2], prim[3]});

          Value *subReal =
              Builder2.CreateFSub(Builder2.CreateExtractValue(mul1, {0}),
                                  Builder2.CreateExtractValue(mul2, {0}));
          Value *subImag =
              Builder2.CreateFSub(Builder2.CreateExtractValue(mul1, {1}),
                                  Builder2.CreateExtractValue(mul2, {1}));

          auto div1 = Builder2.CreateCall(
              div, {subReal, subImag, Builder2.CreateExtractValue(sq1, {0}),
                    Builder2.CreateExtractValue(sq1, {1})});

          setDiffe(&call, div1, Builder2);

          eraseIfUnused(*orig);

          return;
        }
        case DerivativeMode::ReverseModeGradient:
        case DerivativeMode::ReverseModeCombined: {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);

          Value *idiff = diffe(&call, Builder2);
          Value *idiffReal = Builder2.CreateExtractValue(idiff, {0});
          Value *idiffImag = Builder2.CreateExtractValue(idiff, {1});

          Value *diff0 = nullptr;
          Value *diff1 = nullptr;

          if (!constantval0 || !constantval1)
            diff0 = Builder2.CreateCall(div, {idiffReal, idiffImag,
                                              lookup(prim[2], Builder2),
                                              lookup(prim[3], Builder2)});

          if (!constantval2 || !constantval3) {
            auto fdiv = Builder2.CreateCall(div, {idiffReal, idiffImag,
                                                  lookup(prim[1], Builder2),
                                                  lookup(prim[2], Builder2)});

            Value *newcall = gutils->getNewFromOriginal(&call);

            diff1 = Builder2.CreateCall(
                mul,
                {Builder2.CreateFNeg(Builder2.CreateExtractValue(newcall, {0})),
                 Builder2.CreateFNeg(Builder2.CreateExtractValue(newcall, {1})),
                 Builder2.CreateExtractValue(fdiv, {0}),
                 Builder2.CreateExtractValue(fdiv, {1})});
          }

          if (diff0 || diff1)
            setDiffe(&call, Constant::getNullValue(call.getType()), Builder2);

          if (diff0) {
            addToDiffe(orig_op0, Builder2.CreateExtractValue(diff0, {0}),
                       Builder2, orig_op0->getType());
            addToDiffe(orig_op1, Builder2.CreateExtractValue(diff0, {1}),
                       Builder2, orig_op1->getType());
          }

          if (diff1) {
            addToDiffe(orig_op2, Builder2.CreateExtractValue(diff1, {0}),
                       Builder2, orig_op2->getType());
            addToDiffe(orig_op3, Builder2.CreateExtractValue(diff1, {1}),
                       Builder2, orig_op3->getType());
          }

          if (constantval2 && constantval3)
            eraseIfUnused(*orig);

          return;
        }
        case DerivativeMode::ReverseModePrimal:;
          return;
        }
      }

      if (funcName == "scalbn" || funcName == "scalbnf" ||
          funcName == "scalbnl" || funcName == "scalbln" ||
          funcName == "scalblnf" || funcName == "scalblnl") {
        eraseIfUnused(*orig);

        Value *orig_op0 = call.getOperand(0);
        Value *orig_op1 = call.getOperand(1);

        bool constantval0 = gutils->isConstantValue(orig_op0);

        if (gutils->isConstantInstruction(orig) || constantval0)
          return;

        Value *op0 = gutils->getNewFromOriginal(orig_op0);
        Value *op1 = gutils->getNewFromOriginal(orig_op1);

        auto scal = gutils->oldFunc->getParent()->getOrInsertFunction(
            funcName, called->getFunctionType(), called->getAttributes());

        switch (Mode) {
        case DerivativeMode::ForwardMode:
        case DerivativeMode::ForwardModeSplit: {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);

          Value *diff0 = diffe(orig_op0, Builder2);

          auto cal1 = Builder2.CreateCall(scal, {op0, op1});
          auto cal2 = Builder2.CreateCall(scal, {diff0, op1});

          Value *diff = Builder2.CreateFMul(
              cal1, ConstantFP::get(call.getType(), 0.3010299957));
          diff = Builder2.CreateFAdd(diff, cal2);

          setDiffe(&call, diff, Builder2);
          return;
        }
        case DerivativeMode::ReverseModeGradient:
        case DerivativeMode::ReverseModeCombined: {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);

          Value *idiff = diffe(&call, Builder2);

          if (idiff && !constantval0) {
            op1 = lookup(op1, Builder2);

            auto cal1 = Builder2.CreateCall(scal, {op0, op1});
            auto cal2 = Builder2.CreateCall(scal, {idiff, op1});

            Value *diff = Builder2.CreateFMul(
                cal1, ConstantFP::get(call.getType(), 0.3010299957));
            diff = Builder2.CreateFAdd(diff, cal2);

            addToDiffe(orig_op0, diff, Builder2, call.getType());
          }

          return;
        }
        case DerivativeMode::ReverseModePrimal:;
          return;
        }
      }

      if (called) {
        if (funcName == "erf" || funcName == "erfi" || funcName == "erfc" ||
            funcName == "Faddeeva_erf" || funcName == "Faddeeva_erfi" ||
            funcName == "Faddeeva_erfc") {
          if (gutils->knownRecomputeHeuristic.find(orig) !=
              gutils->knownRecomputeHeuristic.end()) {
            if (!gutils->knownRecomputeHeuristic[orig]) {
              gutils->cacheForReverse(BuilderZ, newCall,
                                      getIndex(orig, CacheType::Self));
            }
          }
          eraseIfUnused(*orig);
          if (gutils->isConstantInstruction(orig))
            return;

          switch (Mode) {
          case DerivativeMode::ReverseModePrimal:
            return;
          case DerivativeMode::ForwardMode:
          case DerivativeMode::ForwardModeSplit:
          case DerivativeMode::ReverseModeGradient:
          case DerivativeMode::ReverseModeCombined: {
            IRBuilder<> Builder2(&call);
            if (Mode == DerivativeMode::ForwardMode ||
                Mode == DerivativeMode::ForwardModeSplit)
              getForwardBuilder(Builder2);
            else
              getReverseBuilder(Builder2);

            Value *x = gutils->getNewFromOriginal(orig->getArgOperand(0));
            if (Mode != DerivativeMode::ForwardMode &&
                Mode != DerivativeMode::ForwardModeSplit)
              x = lookup(x, Builder2);

            Value *sq;
            Type *tys[1];
            if (funcName.startswith("Faddeeva")) {
              Value *re = Builder2.CreateExtractValue(x, 0);
              Value *im = Builder2.CreateExtractValue(x, 1);
              sq = UndefValue::get(x->getType());
              sq = Builder2.CreateInsertValue(
                  sq,
                  Builder2.CreateFSub(Builder2.CreateFMul(re, re),
                                      Builder2.CreateFMul(im, im)),
                  0);
              Value *p = Builder2.CreateFMul(re, im);
              sq = Builder2.CreateInsertValue(sq, Builder2.CreateFAdd(p, p), 1);
              tys[0] = re->getType();
            } else {
              sq = Builder2.CreateFMul(x, x);
              tys[0] = sq->getType();
            }

            if (funcName == "erf" || funcName == "erfc") {
              sq = Builder2.CreateFNeg(sq);
            } else if (funcName == "Faddeeva_erf" ||
                       funcName == "Faddeeva_erfc") {
              Value *re = Builder2.CreateExtractValue(sq, 0);
              Value *im = Builder2.CreateExtractValue(sq, 1);
              sq = UndefValue::get(x->getType());
              sq = Builder2.CreateInsertValue(sq, Builder2.CreateFNeg(re), 0);
              sq = Builder2.CreateInsertValue(sq, Builder2.CreateFNeg(im), 1);
            }

            Function *ExpF = Intrinsic::getDeclaration(
                gutils->oldFunc->getParent(), Intrinsic::exp, tys);

            Value *cal;

            if (funcName.startswith("Faddeeva")) {
              Value *re = Builder2.CreateExtractValue(sq, 0);
              Value *im = Builder2.CreateExtractValue(sq, 1);
              Value *reexp = Builder2.CreateCall(ExpF, {re});

              Function *CosF = Intrinsic::getDeclaration(
                  gutils->oldFunc->getParent(), Intrinsic::cos, tys);
              Function *SinF = Intrinsic::getDeclaration(
                  gutils->oldFunc->getParent(), Intrinsic::sin, tys);

              cal = UndefValue::get(x->getType());
              cal = Builder2.CreateInsertValue(
                  cal,
                  Builder2.CreateFMul(reexp, Builder2.CreateCall(CosF, {im})),
                  0);
              cal = Builder2.CreateInsertValue(
                  cal,
                  Builder2.CreateFMul(reexp, Builder2.CreateCall(SinF, {im})),
                  1);
            } else {
              cal = Builder2.CreateCall(ExpF, {sq});
            }

            Value *factor = ConstantFP::get(
                tys[0],
                (funcName == "erfc" || funcName == "Faddeeva_erfc")
                    ? -1.1283791670955125738961589031215451716881012586580
                    : 1.1283791670955125738961589031215451716881012586580);

            if (funcName.startswith("Faddeeva")) {
              Value *re = Builder2.CreateExtractValue(cal, 0);
              Value *im = Builder2.CreateExtractValue(cal, 1);
              cal = UndefValue::get(x->getType());
              cal = Builder2.CreateInsertValue(
                  cal, Builder2.CreateFMul(re, factor), 0);
              cal = Builder2.CreateInsertValue(
                  cal, Builder2.CreateFMul(im, factor), 1);
            } else {
              cal = Builder2.CreateFMul(cal, factor);
            }

            Value *dfactor = (Mode == DerivativeMode::ForwardMode ||
                              Mode == DerivativeMode::ForwardModeSplit)
                                 ? diffe(orig->getArgOperand(0), Builder2)
                                 : diffe(orig, Builder2);

            auto rule1 = [&](Value *dfactor) {
              Value *res = UndefValue::get(x->getType());
              Value *re = Builder2.CreateExtractValue(cal, 0);
              Value *im = Builder2.CreateExtractValue(cal, 1);

              Value *fac_re = Builder2.CreateExtractValue(dfactor, 0);
              Value *fac_im = Builder2.CreateExtractValue(dfactor, 1);

              res = Builder2.CreateInsertValue(
                  cal,
                  Builder2.CreateFSub(Builder2.CreateFMul(re, fac_re),
                                      Builder2.CreateFMul(im, fac_im)),
                  0);
              res = Builder2.CreateInsertValue(
                  res,
                  Builder2.CreateFAdd(Builder2.CreateFMul(im, fac_re),
                                      Builder2.CreateFMul(re, fac_im)),
                  1);

              return res;
            };

            auto rule2 = [&](Value *dfactor) {
              return Builder2.CreateFMul(cal, dfactor);
            };

            if (funcName.startswith("Faddeeva")) {
              cal = applyChainRule(call.getType(), Builder2, rule1, dfactor);
            } else {
              cal = applyChainRule(call.getType(), Builder2, rule2, dfactor);
            }

            if (Mode == DerivativeMode::ForwardMode ||
                Mode == DerivativeMode::ForwardModeSplit) {
              setDiffe(orig, cal, Builder2);
            } else {
              setDiffe(orig,
                       Constant::getNullValue(
                           gutils->getShadowType(orig->getType())),
                       Builder2);
              addToDiffe(orig->getArgOperand(0), cal, Builder2, x->getType());
            }
            return;
          }
          }
        }

        if (funcName == "j0" || funcName == "y0" || funcName == "j0f" ||
            funcName == "y0f") {
          if (gutils->knownRecomputeHeuristic.find(orig) !=
              gutils->knownRecomputeHeuristic.end()) {
            if (!gutils->knownRecomputeHeuristic[orig]) {
              gutils->cacheForReverse(BuilderZ, newCall,
                                      getIndex(orig, CacheType::Self));
            }
          }
          eraseIfUnused(*orig);
          if (gutils->isConstantInstruction(orig))
            return;

          switch (Mode) {
          case DerivativeMode::ForwardModeSplit:
          case DerivativeMode::ForwardMode: {
            IRBuilder<> Builder2(&call);
            getForwardBuilder(Builder2);
            Value *x = gutils->getNewFromOriginal(orig->getArgOperand(0));

            Value *dx = Builder2.CreateCall(
                gutils->oldFunc->getParent()->getOrInsertFunction(
                    (funcName[0] == 'j') ? ((funcName == "j0") ? "j1" : "j1f")
                                         : ((funcName == "y0") ? "y1" : "y1f"),
                    called->getFunctionType()),
                {x});
            dx = Builder2.CreateFNeg(dx);
            Value *op = diffe(orig->getArgOperand(0), Builder2);

            auto rule = [&](Value *op) { return Builder2.CreateFMul(dx, op); };

            Value *diff = applyChainRule(call.getType(), Builder2, rule, op);
            setDiffe(orig, diff, Builder2);
            return;
          }
          case DerivativeMode::ReverseModeGradient:
          case DerivativeMode::ReverseModeCombined: {
            IRBuilder<> Builder2(call.getParent());
            getReverseBuilder(Builder2);
            Value *x = lookup(
                gutils->getNewFromOriginal(orig->getArgOperand(0)), Builder2);

            Value *dx = Builder2.CreateCall(
                gutils->oldFunc->getParent()->getOrInsertFunction(
                    (funcName[0] == 'j') ? ((funcName == "j0") ? "j1" : "j1f")
                                         : ((funcName == "y0") ? "y1" : "y1f"),
                    called->getFunctionType()),
                {x});
            dx = Builder2.CreateFNeg(dx);
            auto rule = [&](Value *dorig) {
              return Builder2.CreateFMul(dx, dorig);
            };
            Value *dorig = diffe(orig, Builder2);
            dx = applyChainRule(orig->getArgOperand(0)->getType(), Builder2,
                                rule, dorig);
            setDiffe(
                orig,
                Constant::getNullValue(gutils->getShadowType(orig->getType())),
                Builder2);
            addToDiffe(orig->getArgOperand(0), dx, Builder2, x->getType());
            return;
          }
          case DerivativeMode::ReverseModePrimal: {
            return;
          }
          }
        }

        if (funcName == "j1" || funcName == "y1" || funcName == "j1f" ||
            funcName == "y1f") {
          if (gutils->knownRecomputeHeuristic.find(orig) !=
              gutils->knownRecomputeHeuristic.end()) {
            if (!gutils->knownRecomputeHeuristic[orig]) {
              gutils->cacheForReverse(BuilderZ, newCall,
                                      getIndex(orig, CacheType::Self));
            }
          }
          eraseIfUnused(*orig);
          if (gutils->isConstantInstruction(orig))
            return;

          switch (Mode) {
          case DerivativeMode::ForwardModeSplit:
          case DerivativeMode::ForwardMode: {
            IRBuilder<> Builder2(&call);
            getForwardBuilder(Builder2);
            Value *x = gutils->getNewFromOriginal(orig->getArgOperand(0));

            Value *d0 = Builder2.CreateCall(
                gutils->oldFunc->getParent()->getOrInsertFunction(
                    (funcName[0] == 'j') ? ((funcName == "j1") ? "j0" : "j0f")
                                         : ((funcName == "y1") ? "y0" : "y0f"),
                    called->getFunctionType()),
                {x});

            Type *intType =
                Type::getIntNTy(called->getContext(), sizeof(int) * 8);
            Type *pargs[] = {intType, x->getType()};
            auto FT2 = FunctionType::get(x->getType(), pargs, false);
            Value *d2 = Builder2.CreateCall(
                gutils->oldFunc->getParent()->getOrInsertFunction(
                    (funcName[0] == 'j') ? ((funcName == "j1") ? "jn" : "jnf")
                                         : ((funcName == "y1") ? "yn" : "ynf"),
                    FT2),
                {ConstantInt::get(intType, 2), x});
            Value *dx = Builder2.CreateFSub(d0, d2);
            dx = Builder2.CreateFMul(dx, ConstantFP::get(x->getType(), 0.5));
            Value *op = diffe(orig->getArgOperand(0), Builder2);

            auto rule = [&](Value *op) { return Builder2.CreateFMul(dx, op); };

            Value *diff = applyChainRule(call.getType(), Builder2, rule, op);
            setDiffe(orig, diff, Builder2);
            return;
          }
          case DerivativeMode::ReverseModeGradient:
          case DerivativeMode::ReverseModeCombined: {
            IRBuilder<> Builder2(call.getParent());
            getReverseBuilder(Builder2);
            Value *x = lookup(
                gutils->getNewFromOriginal(orig->getArgOperand(0)), Builder2);

            Value *d0 = Builder2.CreateCall(
                gutils->oldFunc->getParent()->getOrInsertFunction(
                    (funcName[0] == 'j') ? ((funcName == "j1") ? "j0" : "j0f")
                                         : ((funcName == "y1") ? "y0" : "y0f"),
                    called->getFunctionType()),
                {x});

            Type *intType =
                Type::getIntNTy(called->getContext(), sizeof(int) * 8);
            Type *pargs[] = {intType, x->getType()};
            auto FT2 = FunctionType::get(x->getType(), pargs, false);
            Value *d2 = Builder2.CreateCall(
                gutils->oldFunc->getParent()->getOrInsertFunction(
                    (funcName[0] == 'j') ? ((funcName == "j1") ? "jn" : "jnf")
                                         : ((funcName == "y1") ? "yn" : "ynf"),
                    FT2),
                {ConstantInt::get(intType, 2), x});
            Value *dx = Builder2.CreateFSub(d0, d2);
            dx = Builder2.CreateFMul(dx, ConstantFP::get(x->getType(), 0.5));
            auto rule = [&](Value *dorig) {
              return Builder2.CreateFMul(dx, dorig);
            };
            Value *dorig = diffe(orig, Builder2);
            dx = applyChainRule(orig->getArgOperand(0)->getType(), Builder2,
                                rule, dorig);
            setDiffe(
                orig,
                Constant::getNullValue(gutils->getShadowType(orig->getType())),
                Builder2);
            addToDiffe(orig->getArgOperand(0), dx, Builder2, x->getType());
            return;
          }
          case DerivativeMode::ReverseModePrimal: {
            return;
          }
          }
        }

        if (funcName == "jn" || funcName == "yn" || funcName == "jnf" ||
            funcName == "ynf") {
          if (gutils->knownRecomputeHeuristic.find(orig) !=
              gutils->knownRecomputeHeuristic.end()) {
            if (!gutils->knownRecomputeHeuristic[orig]) {
              gutils->cacheForReverse(BuilderZ, newCall,
                                      getIndex(orig, CacheType::Self));
            }
          }
          eraseIfUnused(*orig);
          if (gutils->isConstantInstruction(orig))
            return;

          switch (Mode) {
          case DerivativeMode::ForwardModeSplit:
          case DerivativeMode::ForwardMode: {
            IRBuilder<> Builder2(&call);
            getForwardBuilder(Builder2);
            Value *x = gutils->getNewFromOriginal(orig->getArgOperand(1));
            Value *n = gutils->getNewFromOriginal(orig->getArgOperand(0));

            Value *d0 = Builder2.CreateCall(
                called,
                {Builder2.CreateSub(n, ConstantInt::get(n->getType(), 1)), x});

            Value *d2 = Builder2.CreateCall(
                called,
                {Builder2.CreateAdd(n, ConstantInt::get(n->getType(), 1)), x});

            Value *op = diffe(orig->getArgOperand(1), Builder2);
            Value *dx = Builder2.CreateFMul(Builder2.CreateFSub(d0, d2),
                                            ConstantFP::get(x->getType(), 0.5));

            auto rule = [&](Value *op) { return Builder2.CreateFMul(dx, op); };

            Value *dif = applyChainRule(call.getType(), Builder2, rule, op);
            setDiffe(orig, dif, Builder2);
            return;
          }
          case DerivativeMode::ReverseModeGradient:
          case DerivativeMode::ReverseModeCombined: {
            IRBuilder<> Builder2(call.getParent());
            getReverseBuilder(Builder2);
            Value *x = lookup(
                gutils->getNewFromOriginal(orig->getArgOperand(1)), Builder2);
            Value *n = lookup(
                gutils->getNewFromOriginal(orig->getArgOperand(0)), Builder2);

            Value *d0 = Builder2.CreateCall(
                called,
                {Builder2.CreateSub(n, ConstantInt::get(n->getType(), 1)), x});

            Value *d2 = Builder2.CreateCall(
                called,
                {Builder2.CreateAdd(n, ConstantInt::get(n->getType(), 1)), x});

            Value *dx = Builder2.CreateFSub(d0, d2);
            dx = Builder2.CreateFMul(dx, ConstantFP::get(x->getType(), 0.5));
            auto rule = [&](Value *dorig) {
              return Builder2.CreateFMul(dx, dorig);
            };
            Value *dorig = diffe(orig, Builder2);
            dx = applyChainRule(orig->getArgOperand(1)->getType(), Builder2,
                                rule, dorig);
            setDiffe(
                orig,
                Constant::getNullValue(gutils->getShadowType(orig->getType())),
                Builder2);
            addToDiffe(orig->getArgOperand(1), dx, Builder2, x->getType());
            return;
          }
          case DerivativeMode::ReverseModePrimal: {
            return;
          }
          }
        }

        if (funcName == "julia.write_barrier") {
          bool backwardsShadow = false;
          bool forwardsShadow = true;
          for (auto pair : gutils->backwardsOnlyShadows) {
            if (pair.second.stores.count(orig)) {
              backwardsShadow = true;
              forwardsShadow = pair.second.primalInitialize;
              if (auto inst = dyn_cast<Instruction>(pair.first))
                if (!forwardsShadow && pair.second.LI &&
                    pair.second.LI->contains(inst->getParent()))
                  backwardsShadow = false;
              break;
            }
          }

          if (Mode == DerivativeMode::ForwardMode ||
              (Mode == DerivativeMode::ReverseModeCombined &&
               (forwardsShadow || backwardsShadow)) ||
              (Mode == DerivativeMode::ReverseModePrimal && forwardsShadow) ||
              (Mode == DerivativeMode::ReverseModeGradient &&
               backwardsShadow)) {
            SmallVector<Value *, 1> iargs;
            IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&call));
#if LLVM_VERSION_MAJOR >= 14
            for (auto &arg : orig->args())
#else
            for (auto &arg : orig->arg_operands())
#endif
            {
              if (!gutils->isConstantValue(arg)) {
                Value *ptrshadow = gutils->invertPointerM(arg, BuilderZ);
                iargs.push_back(ptrshadow);
              }
            }
            if (iargs.size()) {
              BuilderZ.CreateCall(called, iargs);
            }
          }

          eraseIfUnused(*orig);
          return;
        }
        Intrinsic::ID ID = Intrinsic::not_intrinsic;
        if (isMemFreeLibMFunction(funcName, &ID)) {
          if (Mode == DerivativeMode::ReverseModePrimal ||
              gutils->isConstantInstruction(orig)) {

            if (gutils->knownRecomputeHeuristic.find(orig) !=
                gutils->knownRecomputeHeuristic.end()) {
              if (!gutils->knownRecomputeHeuristic[orig]) {
                gutils->cacheForReverse(BuilderZ, newCall,
                                        getIndex(orig, CacheType::Self));
              }
            }
            eraseIfUnused(*orig);
            return;
          }

          if (ID != Intrinsic::not_intrinsic) {
            SmallVector<Value *, 2> orig_ops(orig->getNumOperands());
            for (unsigned i = 0; i < orig->getNumOperands(); ++i) {
              orig_ops[i] = orig->getOperand(i);
            }
            handleAdjointForIntrinsic(ID, *orig, orig_ops);
            if (gutils->knownRecomputeHeuristic.find(orig) !=
                gutils->knownRecomputeHeuristic.end()) {
              if (!gutils->knownRecomputeHeuristic[orig]) {
                gutils->cacheForReverse(BuilderZ, newCall,
                                        getIndex(orig, CacheType::Self));
              }
            }
            eraseIfUnused(*orig);
            return;
          }
        }
        if (funcName == "__fd_sincos_1") {
          if (gutils->knownRecomputeHeuristic.find(orig) !=
              gutils->knownRecomputeHeuristic.end()) {
            if (!gutils->knownRecomputeHeuristic[orig]) {
              gutils->cacheForReverse(BuilderZ, newCall,
                                      getIndex(orig, CacheType::Self));
            }
          }
          eraseIfUnused(*orig);
          if (gutils->isConstantInstruction(orig)) {
            return;
          }

          switch (Mode) {
          case DerivativeMode::ForwardModeSplit:
          case DerivativeMode::ForwardMode: {
            IRBuilder<> Builder2(&call);
            getForwardBuilder(Builder2);

            Value *vdiff = diffe(orig->getArgOperand(0), Builder2);
            Value *x = gutils->getNewFromOriginal(orig->getArgOperand(0));
            Value *args[] = {x};

            Type *tys[] = {orig->getOperand(0)->getType()};
            CallInst *dsin = cast<CallInst>(Builder2.CreateCall(
                Intrinsic::getDeclaration(gutils->oldFunc->getParent(),
                                          Intrinsic::cos, tys),
                args));
            CallInst *dcos = cast<CallInst>(Builder2.CreateCall(
                Intrinsic::getDeclaration(gutils->oldFunc->getParent(),
                                          Intrinsic::sin, tys),
                args));

            auto rule = [&](Value *vdiff) {
              Value *res = UndefValue::get(orig->getType());
              res = Builder2.CreateInsertValue(
                  res, Builder2.CreateFMul(vdiff, dsin), {0});
              return Builder2.CreateInsertValue(
                  res, Builder2.CreateFNeg(Builder2.CreateFMul(vdiff, dcos)),
                  {1});
            };

            Value *dif0 = applyChainRule(call.getType(), Builder2, rule, vdiff);
            setDiffe(orig, dif0, Builder2);
            return;
          }
          case DerivativeMode::ReverseModeGradient:
          case DerivativeMode::ReverseModeCombined: {
            IRBuilder<> Builder2(call.getParent());
            getReverseBuilder(Builder2);

            Value *x = lookup(
                gutils->getNewFromOriginal(orig->getArgOperand(0)), Builder2);

            Value *args[] = {x};

            Type *tys[] = {orig->getOperand(0)->getType()};
            CallInst *dsin = cast<CallInst>(Builder2.CreateCall(
                Intrinsic::getDeclaration(gutils->oldFunc->getParent(),
                                          Intrinsic::cos, tys),
                args));
            CallInst *dcos = cast<CallInst>(Builder2.CreateCall(
                Intrinsic::getDeclaration(gutils->oldFunc->getParent(),
                                          Intrinsic::sin, tys),
                args));
            auto rule = [&](Value *vdiff) {
              return Builder2.CreateFSub(
                  Builder2.CreateFMul(Builder2.CreateExtractValue(vdiff, {0}),
                                      dsin),
                  Builder2.CreateFMul(Builder2.CreateExtractValue(vdiff, {1}),
                                      dcos));
            };
            Value *vdiff = diffe(orig, Builder2);
            Value *dif0 = applyChainRule(orig->getArgOperand(0)->getType(),
                                         Builder2, rule, vdiff);
            setDiffe(
                orig,
                Constant::getNullValue(gutils->getShadowType(orig->getType())),
                Builder2);
            addToDiffe(orig->getArgOperand(0), dif0, Builder2, x->getType());
            return;
          }
          case DerivativeMode::ReverseModePrimal: {
            return;
          }
          }
        }
        if (funcName == "cabs" || funcName == "cabsf" || funcName == "cabsl") {
          if (gutils->knownRecomputeHeuristic.find(orig) !=
              gutils->knownRecomputeHeuristic.end()) {
            if (!gutils->knownRecomputeHeuristic[orig]) {
              gutils->cacheForReverse(BuilderZ, newCall,
                                      getIndex(orig, CacheType::Self));
            }
          }
          eraseIfUnused(*orig);
          if (gutils->isConstantInstruction(orig)) {
            return;
          }

          switch (Mode) {
          case DerivativeMode::ForwardModeSplit:
          case DerivativeMode::ForwardMode: {
            IRBuilder<> Builder2(&call);
            getForwardBuilder(Builder2);

            SmallVector<Value *, 2> args;
#if LLVM_VERSION_MAJOR >= 14
            for (auto &arg : orig->args()) {
#else
            for (auto &arg : orig->arg_operands()) {
#endif
              Value *argument = gutils->getNewFromOriginal(arg);
              args.push_back(argument);
            }

            Value *d = Builder2.CreateCall(called, args);

            if (args.size() == 2) {
              Value *op0 = diffe(orig->getArgOperand(0), Builder2);

              Value *op1 = diffe(orig->getArgOperand(1), Builder2);

              auto rule = [&](Value *op0, Value *op1) {
                Value *dif1 =
                    Builder2.CreateFMul(args[0], Builder2.CreateFDiv(op0, d));
                Value *dif2 =
                    Builder2.CreateFMul(args[1], Builder2.CreateFDiv(op1, d));
                return Builder2.CreateFAdd(dif1, dif2);
              };

              Value *dif =
                  applyChainRule(call.getType(), Builder2, rule, op0, op1);
              setDiffe(orig, dif, Builder2);
              return;
            } else {
              llvm::errs() << *orig << "\n";
              llvm_unreachable("unknown calling convention found for cabs");
            }
          }
          case DerivativeMode::ReverseModeGradient:
          case DerivativeMode::ReverseModeCombined: {
            IRBuilder<> Builder2(call.getParent());
            getReverseBuilder(Builder2);

            SmallVector<Value *, 2> args;
#if LLVM_VERSION_MAJOR >= 14
            for (auto &arg : orig->args())
#else
            for (auto &arg : orig->arg_operands())
#endif
              args.push_back(lookup(gutils->getNewFromOriginal(arg), Builder2));

            CallInst *d = cast<CallInst>(Builder2.CreateCall(called, args));

            auto rule = [&](Value *vdiff) {
              return Builder2.CreateFDiv(vdiff, d);
            };
            Value *vdiff = diffe(orig, Builder2);
            Value *div = applyChainRule(orig->getType(), Builder2, rule, vdiff);
            setDiffe(
                orig,
                Constant::getNullValue(gutils->getShadowType(orig->getType())),
                Builder2);

            if (args.size() == 2) {
              for (int i = 0; i < 2; i++)
                if (!gutils->isConstantValue(orig->getArgOperand(i)))
                  addToDiffe(orig->getArgOperand(i),
                             Builder2.CreateFMul(args[i], div), Builder2,
                             orig->getType());
              return;
            } else {
              llvm::errs() << *orig << "\n";
              llvm_unreachable("unknown calling convention found for cabs");
            }
          }
          case DerivativeMode::ReverseModePrimal: {
            return;
          }
          }
        }
        if (funcName == "ldexp" || funcName == "ldexpf" ||
            funcName == "ldexpl") {
          if (gutils->knownRecomputeHeuristic.find(orig) !=
              gutils->knownRecomputeHeuristic.end()) {
            if (!gutils->knownRecomputeHeuristic[orig]) {
              gutils->cacheForReverse(BuilderZ, newCall,
                                      getIndex(orig, CacheType::Self));
            }
          }
          eraseIfUnused(*orig);
          if (gutils->isConstantInstruction(orig)) {
            return;
          }

          switch (Mode) {
          case DerivativeMode::ForwardModeSplit:
          case DerivativeMode::ForwardMode: {
            IRBuilder<> Builder2(&call);
            getForwardBuilder(Builder2);

            Value *vdiff = diffe(orig->getArgOperand(0), Builder2);

            auto rule = [&](Value *vdiff) {
              Value *exponent =
                  gutils->getNewFromOriginal(orig->getArgOperand(1));

              Value *args[] = {vdiff, exponent};

              return cast<CallInst>(Builder2.CreateCall(called, args));
            };

            Value *darg = applyChainRule(call.getType(), Builder2, rule, vdiff);
            setDiffe(orig, darg, Builder2);
            return;
          }
          case DerivativeMode::ReverseModeGradient:
          case DerivativeMode::ReverseModeCombined: {
            IRBuilder<> Builder2(call.getParent());
            getReverseBuilder(Builder2);

            Value *exponent = lookup(
                gutils->getNewFromOriginal(orig->getArgOperand(1)), Builder2);

            auto rule = [&](Value *vdiff) {
              return Builder2.CreateCall(called, {vdiff, exponent});
            };
            Value *vdiff = diffe(orig, Builder2);
            Value *darg = applyChainRule(orig->getArgOperand(0)->getType(),
                                         Builder2, rule, vdiff);
            setDiffe(
                orig,
                Constant::getNullValue(gutils->getShadowType(orig->getType())),
                Builder2);
            addToDiffe(orig->getArgOperand(0), darg, Builder2, orig->getType());
            return;
          }
          case DerivativeMode::ReverseModePrimal: {
            return;
          }
          }
        }
      }

      if (funcName == "lgamma" || funcName == "lgammaf" ||
          funcName == "lgammal" || funcName == "lgamma_r" ||
          funcName == "lgammaf_r" || funcName == "lgammal_r" ||
          funcName == "__lgamma_r_finite" || funcName == "__lgammaf_r_finite" ||
          funcName == "__lgammal_r_finite") {
        if (gutils->knownRecomputeHeuristic.find(orig) !=
            gutils->knownRecomputeHeuristic.end()) {
          if (!gutils->knownRecomputeHeuristic[orig]) {
            gutils->cacheForReverse(BuilderZ, newCall,
                                    getIndex(orig, CacheType::Self));
          }
        }
        if (Mode == DerivativeMode::ReverseModePrimal ||
            gutils->isConstantInstruction(orig)) {
          return;
        }
      }
    }
#if LLVM_VERSION_MAJOR >= 11
    if (auto assembly = dyn_cast<InlineAsm>(orig->getCalledOperand()))
#else
    if (auto assembly = dyn_cast<InlineAsm>(orig->getCalledValue()))
#endif
    {
      if (assembly->getAsmString() == "maxpd $1, $0") {
        if (Mode == DerivativeMode::ReverseModePrimal ||
            gutils->isConstantInstruction(orig)) {

          if (gutils->knownRecomputeHeuristic.find(orig) !=
              gutils->knownRecomputeHeuristic.end()) {
            if (!gutils->knownRecomputeHeuristic[orig]) {
              gutils->cacheForReverse(BuilderZ, newCall,
                                      getIndex(orig, CacheType::Self));
            }
          }
          eraseIfUnused(*orig);
          return;
        }

        SmallVector<Value *, 2> orig_ops(orig->getNumOperands());
        for (unsigned i = 0; i < orig->getNumOperands(); ++i) {
          orig_ops[i] = orig->getOperand(i);
        }
        handleAdjointForIntrinsic(Intrinsic::maxnum, *orig, orig_ops);
        if (gutils->knownRecomputeHeuristic.find(orig) !=
            gutils->knownRecomputeHeuristic.end()) {
          if (!gutils->knownRecomputeHeuristic[orig]) {
            gutils->cacheForReverse(BuilderZ, newCall,
                                    getIndex(orig, CacheType::Self));
          }
        }
        eraseIfUnused(*orig);
        return;
      }
    }

    if (called && isAllocationFunction(*called, gutils->TLI)) {

      bool constval = gutils->isConstantValue(orig);

      if (!constval) {
        auto dbgLoc = gutils->getNewFromOriginal(orig)->getDebugLoc();
        auto found = gutils->invertedPointers.find(orig);
        PHINode *placeholder = cast<PHINode>(&*found->second);
        IRBuilder<> bb(placeholder);

        SmallVector<Value *, 8> args;
#if LLVM_VERSION_MAJOR >= 14
        for (auto &arg : orig->args())
#else
        for (auto &arg : orig->arg_operands())
#endif
        {
          args.push_back(gutils->getNewFromOriginal(arg));
        }

        if (Mode == DerivativeMode::ReverseModeCombined ||
            Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ReverseModePrimal ||
            Mode == DerivativeMode::ForwardModeSplit) {

          Value *anti = placeholder;
          // If rematerializable allocations and split mode, we can
          // simply elect to build the entire piece in the reverse
          // since it should be possible to perform any shadow stores
          // of pointers (from rematerializable property) and it does
          // not escape the function scope (lest it not be
          // rematerializable) so all input derivatives remain zero.
          bool backwardsShadow = false;
          bool forwardsShadow = true;
          bool inLoop = false;
          {
            auto found = gutils->backwardsOnlyShadows.find(orig);
            if (found != gutils->backwardsOnlyShadows.end()) {
              backwardsShadow = true;
              forwardsShadow = found->second.primalInitialize;
              // If in a loop context, maintain the same free behavior.
              if (!forwardsShadow && found->second.LI &&
                  found->second.LI->contains(orig->getParent()))
                inLoop = true;
            }
          }
          {

            if (!forwardsShadow) {
              if (Mode == DerivativeMode::ReverseModePrimal) {
                // Needs a stronger replacement check/assertion.
                Value *replacement = UndefValue::get(placeholder->getType());
                gutils->replaceAWithB(placeholder, replacement);
                gutils->invertedPointers.erase(found);
                gutils->invertedPointers.insert(std::make_pair(
                    orig, InvertedPointerVH(gutils, replacement)));
                gutils->erase(placeholder);
                anti = nullptr;
                goto endAnti;
              } else if (inLoop) {
                gutils->rematerializedShadowPHIs.push_back(placeholder);
                goto endAnti;
              }
            }
            placeholder->setName("");
            if (shadowHandlers.find(called->getName().str()) !=
                shadowHandlers.end()) {
              bb.SetInsertPoint(placeholder);

              if (Mode == DerivativeMode::ReverseModeCombined ||
                  (Mode == DerivativeMode::ReverseModePrimal &&
                   forwardsShadow) ||
                  (Mode == DerivativeMode::ReverseModeGradient &&
                   backwardsShadow)) {
                anti = applyChainRule(call.getType(), bb, [&]() {
                  return shadowHandlers[called->getName().str()](bb, orig,
                                                                 args);
                });
                if (anti->getType() != placeholder->getType()) {
                  llvm::errs() << "orig: " << *orig << "\n";
                  llvm::errs() << "placeholder: " << *placeholder << "\n";
                  llvm::errs() << "anti: " << *anti << "\n";
                }
                gutils->invertedPointers.erase(found);
                bb.SetInsertPoint(placeholder);

                gutils->replaceAWithB(placeholder, anti);
                gutils->erase(placeholder);
              }

              if (auto inst = dyn_cast<Instruction>(anti))
                bb.SetInsertPoint(inst);

              if (!backwardsShadow)
                anti = gutils->cacheForReverse(
                    bb, anti, getIndex(orig, CacheType::Shadow));
            } else {
              auto rule = [&]() {
#if LLVM_VERSION_MAJOR >= 11
                Value *anti = bb.CreateCall(orig->getFunctionType(),
                                            orig->getCalledOperand(), args,
                                            orig->getName() + "'mi");
#else
                anti = bb.CreateCall(orig->getCalledValue(), args,
                                     orig->getName() + "'mi");
#endif
                cast<CallInst>(anti)->setAttributes(orig->getAttributes());
                cast<CallInst>(anti)->setCallingConv(orig->getCallingConv());
                cast<CallInst>(anti)->setTailCallKind(orig->getTailCallKind());
                cast<CallInst>(anti)->setDebugLoc(dbgLoc);

#if LLVM_VERSION_MAJOR >= 14
                cast<CallInst>(anti)->addAttributeAtIndex(
                    AttributeList::ReturnIndex, Attribute::NoAlias);
                cast<CallInst>(anti)->addAttributeAtIndex(
                    AttributeList::ReturnIndex, Attribute::NonNull);
#else
                cast<CallInst>(anti)->addAttribute(AttributeList::ReturnIndex,
                                                   Attribute::NoAlias);
                cast<CallInst>(anti)->addAttribute(AttributeList::ReturnIndex,
                                                   Attribute::NonNull);
#endif

                if (called->getName() == "malloc" ||
                    called->getName() == "_Znwm") {
                  if (auto ci = dyn_cast<ConstantInt>(args[0])) {
                    unsigned derefBytes = ci->getLimitedValue();
                    CallInst *cal =
                        cast<CallInst>(gutils->getNewFromOriginal(orig));
#if LLVM_VERSION_MAJOR >= 14
                    cast<CallInst>(anti)->addDereferenceableRetAttr(derefBytes);
                    cal->addDereferenceableRetAttr(derefBytes);
#if !defined(FLANG) && !defined(ROCM)
                    AttrBuilder B(called->getContext());
#else
                    AttrBuilder B;
#endif
                    B.addDereferenceableOrNullAttr(derefBytes);
                    cast<CallInst>(anti)->setAttributes(
                        cast<CallInst>(anti)->getAttributes().addRetAttributes(
                            orig->getContext(), B));
                    cal->setAttributes(cal->getAttributes().addRetAttributes(
                        orig->getContext(), B));
                    cal->addAttributeAtIndex(AttributeList::ReturnIndex,
                                             Attribute::NoAlias);
                    cal->addAttributeAtIndex(AttributeList::ReturnIndex,
                                             Attribute::NonNull);
#else
                    cast<CallInst>(anti)->addDereferenceableAttr(
                        llvm::AttributeList::ReturnIndex, derefBytes);
                    cal->addDereferenceableAttr(
                        llvm::AttributeList::ReturnIndex, derefBytes);
                    cast<CallInst>(anti)->addDereferenceableOrNullAttr(
                        llvm::AttributeList::ReturnIndex, derefBytes);
                    cal->addDereferenceableOrNullAttr(
                        llvm::AttributeList::ReturnIndex, derefBytes);
                    cal->addAttribute(AttributeList::ReturnIndex,
                                      Attribute::NoAlias);
                    cal->addAttribute(AttributeList::ReturnIndex,
                                      Attribute::NonNull);
#endif
                  }
                }
                return anti;
              };

              anti = applyChainRule(orig->getType(), bb, rule);

              gutils->invertedPointers.erase(found);
              if (&*bb.GetInsertPoint() == placeholder)
                bb.SetInsertPoint(placeholder->getNextNode());
              gutils->replaceAWithB(placeholder, anti);
              gutils->erase(placeholder);

              if (!backwardsShadow)
                anti = gutils->cacheForReverse(
                    bb, anti, getIndex(orig, CacheType::Shadow));
              else {
                if (auto MD = hasMetadata(orig, "enzyme_fromstack")) {
                  AllocaInst *replacement = bb.CreateAlloca(
                      Type::getInt8Ty(orig->getContext()), args[0]);
                  replacement->takeName(anti);
                  auto Alignment = cast<ConstantInt>(cast<ConstantAsMetadata>(
                                                         MD->getOperand(0))
                                                         ->getValue())
                                       ->getLimitedValue();
#if LLVM_VERSION_MAJOR >= 10
                  replacement->setAlignment(Align(Alignment));
#else
                  replacement->setAlignment(Alignment);
#endif

                  gutils->replaceAWithB(cast<Instruction>(anti), replacement);
                  gutils->erase(cast<Instruction>(anti));
                  anti = replacement;
                }
              }

              if (Mode == DerivativeMode::ReverseModeCombined ||
                  (Mode == DerivativeMode::ReverseModePrimal &&
                   forwardsShadow) ||
                  (Mode == DerivativeMode::ReverseModeGradient &&
                   backwardsShadow) ||
                  (Mode == DerivativeMode::ForwardModeSplit &&
                   backwardsShadow)) {
                if (!inLoop) {
                  applyChainRule(
                      bb,
                      [&](Value *anti) {
                        zeroKnownAllocation(bb, anti, args, *called,
                                            gutils->TLI);
                      },
                      anti);
                }
              }
            }
            gutils->invertedPointers.insert(
                std::make_pair(orig, InvertedPointerVH(gutils, anti)));
          }
        endAnti:;

          bool isAlloca = anti ? isa<AllocaInst>(anti) : false;
          if (gutils->getWidth() != 1) {
            if (auto insertion = dyn_cast_or_null<InsertElementInst>(anti)) {
              isAlloca = isa<AllocaInst>(insertion->getOperand(1));
            }
          }

          if (((Mode == DerivativeMode::ReverseModeCombined && shouldFree()) ||
               (Mode == DerivativeMode::ReverseModeGradient && shouldFree()) ||
               (Mode == DerivativeMode::ForwardModeSplit && shouldFree())) &&
              !isAlloca) {
            IRBuilder<> Builder2(call.getParent());
            getReverseBuilder(Builder2);
            assert(anti);
            Value *tofree = lookup(anti, Builder2);
            assert(tofree);
            assert(tofree->getType());
            assert(Type::getInt8Ty(tofree->getContext()));
            assert(
                PointerType::getUnqual(Type::getInt8Ty(tofree->getContext())));
            assert(Type::getInt8PtrTy(tofree->getContext()));
            auto rule = [&](Value *tofree) {
              auto CI = freeKnownAllocation(Builder2, tofree, *called, dbgLoc,
                                            gutils->TLI);
              if (CI)
#if LLVM_VERSION_MAJOR >= 14
                CI->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                        Attribute::NonNull);
#else
                CI->addAttribute(AttributeList::FirstArgIndex,
                                 Attribute::NonNull);
#endif
            };
            applyChainRule(Builder2, rule, tofree);
          }
        } else if (Mode == DerivativeMode::ForwardMode) {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);

          SmallVector<Value *, 2> args;
#if LLVM_VERSION_MAJOR >= 14
          for (unsigned i = 0; i < orig->arg_size(); ++i)
#else
          for (unsigned i = 0; i < orig->getNumArgOperands(); ++i)
#endif
          {
            auto arg = orig->getArgOperand(i);
            args.push_back(gutils->getNewFromOriginal(arg));
          }

          auto rule = [&]() {
            CallInst *CI = Builder2.CreateCall(orig->getFunctionType(),
                                               orig->getCalledFunction(), args);
            CI->setAttributes(orig->getAttributes());
            CI->setCallingConv(orig->getCallingConv());
            CI->setTailCallKind(orig->getTailCallKind());
            CI->setDebugLoc(dbgLoc);
            return CI;
          };

          Value *CI = applyChainRule(call.getType(), Builder2, rule);

          auto found = gutils->invertedPointers.find(orig);
          PHINode *placeholder = cast<PHINode>(&*found->second);

          gutils->invertedPointers.erase(found);
          gutils->replaceAWithB(placeholder, CI);
          gutils->erase(placeholder);
          gutils->invertedPointers.insert(
              std::make_pair(orig, InvertedPointerVH(gutils, CI)));
        }
      }

      // Cache and rematerialization irrelevant for forward mode.
      if (Mode == DerivativeMode::ForwardMode)
        return;

      std::map<UsageKey, bool> Seen;
      for (auto pair : gutils->knownRecomputeHeuristic)
        if (!pair.second)
          Seen[UsageKey(pair.first, ValueType::Primal)] = false;
      bool primalNeededInReverse =
          Mode == DerivativeMode::ForwardMode
              ? false
              : is_value_needed_in_reverse<ValueType::Primal>(
                    gutils, orig, Mode, Seen, oldUnreachable);

      bool cacheWholeAllocation = false;
      if (gutils->knownRecomputeHeuristic.count(orig)) {
        if (!gutils->knownRecomputeHeuristic[orig]) {
          cacheWholeAllocation = true;
          primalNeededInReverse = true;
        }
      }

      // Don't erase any store that needs to be preserved for a
      // rematerialization
      {
        auto found = gutils->rematerializableAllocations.find(orig);
        if (found != gutils->rematerializableAllocations.end()) {
          // If rematerializing (e.g. needed in reverse, but not needing
          //  the whole allocation):
          if (primalNeededInReverse && !cacheWholeAllocation) {
            // if rematerialize, don't ever cache and downgrade to stack
            // allocation where possible.
            if (auto MD = hasMetadata(orig, "enzyme_fromstack")) {
              IRBuilder<> B(newCall);
              if (auto CI = dyn_cast<ConstantInt>(orig->getArgOperand(0))) {
                B.SetInsertPoint(gutils->inversionAllocs);
              }

              auto rule = [&]() {
                auto replacement = B.CreateAlloca(
                    Type::getInt8Ty(orig->getContext()),
                    gutils->getNewFromOriginal(orig->getArgOperand(0)));
                auto Alignment =
                    cast<ConstantInt>(
                        cast<ConstantAsMetadata>(MD->getOperand(0))->getValue())
                        ->getLimitedValue();
#if LLVM_VERSION_MAJOR >= 10
                replacement->setAlignment(Align(Alignment));
#else
                replacement->setAlignment(Alignment);
#endif
                return replacement;
              };

              Value *replacement =
                  applyChainRule(Type::getInt8Ty(orig->getContext()), B, rule);

              gutils->replaceAWithB(newCall, replacement);
              gutils->erase(newCall);
              return;
            }

            // No need to free GC.
            if (funcName == "ijl_alloc_array_1d" ||
                funcName == "ijl_alloc_array_2d" ||
                funcName == "ijl_alloc_array_3d" ||
                funcName == "ijl_array_copy" ||
                funcName == "jl_alloc_array_1d" ||
                funcName == "jl_alloc_array_2d" ||
                funcName == "jl_alloc_array_3d" ||
                funcName == "jl_array_copy" || funcName == "julia.gc_alloc_obj")
              return;

            // Otherwise if in reverse pass, free the newly created allocation.
            if (Mode == DerivativeMode::ReverseModeGradient ||
                Mode == DerivativeMode::ReverseModeCombined ||
                Mode == DerivativeMode::ForwardModeSplit) {
              IRBuilder<> Builder2(call.getParent());
              getReverseBuilder(Builder2);
              auto dbgLoc = gutils->getNewFromOriginal(orig->getDebugLoc());
              freeKnownAllocation(Builder2, lookup(newCall, Builder2), *called,
                                  dbgLoc, gutils->TLI);
              return;
            }
            // If in primal, do nothing (keeping the original caching behavior)
            if (Mode == DerivativeMode::ReverseModePrimal)
              return;
          } else if (!cacheWholeAllocation) {
            // If not caching allocation and not needed in the reverse, we can
            // use the original freeing behavior for the function. If in the
            // reverse pass we should not recreate this allocation.
            if (Mode == DerivativeMode::ReverseModeGradient)
              eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
            else if (auto MD = hasMetadata(orig, "enzyme_fromstack")) {
              IRBuilder<> B(newCall);
              if (auto CI = dyn_cast<ConstantInt>(orig->getArgOperand(0))) {
                B.SetInsertPoint(gutils->inversionAllocs);
              }

              auto rule = [&]() {
                auto replacement = B.CreateAlloca(
                    Type::getInt8Ty(orig->getContext()),
                    gutils->getNewFromOriginal(orig->getArgOperand(0)));
                auto Alignment =
                    cast<ConstantInt>(
                        cast<ConstantAsMetadata>(MD->getOperand(0))->getValue())
                        ->getLimitedValue();
                // Don't set zero alignment
                if (Alignment) {
#if LLVM_VERSION_MAJOR >= 10
                  replacement->setAlignment(Align(Alignment));
#else
                  replacement->setAlignment(Alignment);
#endif
                }
                return replacement;
              };

              Value *replacement =
                  applyChainRule(Type::getInt8Ty(orig->getContext()), B, rule);

              gutils->replaceAWithB(newCall, replacement);
              gutils->erase(newCall);
            }
            return;
          }
        }
      }

      // If an allocation is not needed in the reverse, maintain the original
      // free behavior and do not rematerialize this for the reverse. However,
      // this is only safe to perform for allocations with a guaranteed free
      // as can we can only guarantee that we don't erase those frees.
      bool hasPDFree = gutils->allocationsWithGuaranteedFree.count(orig);
      if (!primalNeededInReverse && hasPDFree) {
        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ForwardModeSplit) {
          eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
        } else {
          if (auto MD = hasMetadata(orig, "enzyme_fromstack")) {
            IRBuilder<> B(newCall);
            if (auto CI = dyn_cast<ConstantInt>(orig->getArgOperand(0))) {
              B.SetInsertPoint(gutils->inversionAllocs);
            }
            auto replacement = B.CreateAlloca(
                Type::getInt8Ty(orig->getContext()),
                gutils->getNewFromOriginal(orig->getArgOperand(0)));
            auto Alignment =
                cast<ConstantInt>(
                    cast<ConstantAsMetadata>(MD->getOperand(0))->getValue())
                    ->getLimitedValue();
            // Don't set zero alignment
            if (Alignment) {
#if LLVM_VERSION_MAJOR >= 10
              replacement->setAlignment(Align(Alignment));
#else
              replacement->setAlignment(Alignment);
#endif
            }
            gutils->replaceAWithB(newCall, replacement);
            gutils->erase(newCall);
          }
        }
        return;
      }

      // If an object is managed by the GC do not preserve it for later free,
      // Thus it only needs caching if there is a need for it in the reverse.
      if (funcName == "jl_alloc_array_1d" || funcName == "jl_alloc_array_2d" ||
          funcName == "jl_alloc_array_3d" || funcName == "jl_array_copy" ||
          funcName == "ijl_alloc_array_1d" ||
          funcName == "ijl_alloc_array_2d" ||
          funcName == "ijl_alloc_array_3d" || funcName == "ijl_array_copy" ||
          funcName == "julia.gc_alloc_obj") {
        if (!primalNeededInReverse) {
          if (Mode == DerivativeMode::ReverseModeGradient ||
              Mode == DerivativeMode::ForwardModeSplit) {
            auto pn = BuilderZ.CreatePHI(
                orig->getType(), 1, (orig->getName() + "_replacementJ").str());
            gutils->fictiousPHIs[pn] = orig;
            gutils->replaceAWithB(newCall, pn);
            gutils->erase(newCall);
          }
        } else if (Mode != DerivativeMode::ReverseModeCombined) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(orig, CacheType::Self));
        }
        return;
      }

      if (EnzymeFreeInternalAllocations)
        hasPDFree = true;

      // TODO enable this if we need to free the memory
      // NOTE THAT TOPLEVEL IS THERE SIMPLY BECAUSE THAT WAS PREVIOUS ATTITUTE
      // TO FREE'ing
      if ((primalNeededInReverse &&
           !gutils->unnecessaryIntermediates.count(orig)) ||
          hasPDFree) {
        Value *nop = gutils->cacheForReverse(BuilderZ, newCall,
                                             getIndex(orig, CacheType::Self));
        if (hasPDFree &&
            ((Mode == DerivativeMode::ReverseModeGradient && shouldFree()) ||
             Mode == DerivativeMode::ReverseModeCombined ||
             (Mode == DerivativeMode::ForwardModeSplit && shouldFree()))) {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);
          auto dbgLoc = gutils->getNewFromOriginal(orig->getDebugLoc());
          freeKnownAllocation(Builder2, lookup(nop, Builder2), *called, dbgLoc,
                              gutils->TLI);
        }
      } else if (Mode == DerivativeMode::ReverseModeGradient ||
                 Mode == DerivativeMode::ReverseModeCombined ||
                 Mode == DerivativeMode::ForwardModeSplit) {
        // Note that here we cannot simply replace with null as users who
        // try to find the shadow pointer will use the shadow of null rather
        // than the true shadow of this
        auto pn = BuilderZ.CreatePHI(orig->getType(), 1,
                                     (orig->getName() + "_replacementB").str());
        gutils->fictiousPHIs[pn] = orig;
        gutils->replaceAWithB(newCall, pn);
        gutils->erase(newCall);
      }

      return;
    }

    if (funcName == "julia.pointer_from_objref") {
      if (gutils->isConstantValue(orig)) {
        eraseIfUnused(*orig);
        return;
      }

      auto ifound = gutils->invertedPointers.find(orig);
      assert(ifound != gutils->invertedPointers.end());

      auto placeholder = cast<PHINode>(&*ifound->second);

      bool needShadow = (Mode == DerivativeMode::ForwardMode ||
                         Mode == DerivativeMode::ForwardModeSplit)
                            ? true
                            : is_value_needed_in_reverse<ValueType::Shadow>(
                                  gutils, orig, Mode, oldUnreachable);
      if (!needShadow) {
        gutils->invertedPointers.erase(ifound);
        gutils->erase(placeholder);
        eraseIfUnused(*orig);
        return;
      }

      Value *ptrshadow =
          gutils->invertPointerM(call.getArgOperand(0), BuilderZ);

      Value *val = applyChainRule(
          call.getType(), BuilderZ,
          [&](Value *v) -> Value * { return BuilderZ.CreateCall(called, {v}); },
          ptrshadow);

      gutils->replaceAWithB(placeholder, val);
      gutils->erase(placeholder);
      eraseIfUnused(*orig);
      return;
    }
    if (funcName == "memcpy" || funcName == "memmove") {
      auto ID = (funcName == "memcpy") ? Intrinsic::memcpy : Intrinsic::memmove;
#if LLVM_VERSION_MAJOR >= 10
      visitMemTransferCommon(ID, /*srcAlign*/ MaybeAlign(1),
                             /*dstAlign*/ MaybeAlign(1), *orig,
                             orig->getArgOperand(0), orig->getArgOperand(1),
                             gutils->getNewFromOriginal(orig->getArgOperand(2)),
                             ConstantInt::getFalse(orig->getContext()));
#else
      visitMemTransferCommon(ID, /*srcAlign*/ 1,
                             /*dstAlign*/ 1, *orig, orig->getArgOperand(0),
                             orig->getArgOperand(1),
                             gutils->getNewFromOriginal(orig->getArgOperand(2)),
                             ConstantInt::getFalse(orig->getContext()));
#endif
      return;
    }
    if (funcName == "memset") {
      visitMemSetCommon(*orig);
      return;
    }
    if (funcName == "posix_memalign") {
      bool constval = gutils->isConstantInstruction(orig);

      if (!constval) {
        Value *val;
        if (Mode == DerivativeMode::ReverseModePrimal ||
            Mode == DerivativeMode::ReverseModeCombined ||
            Mode == DerivativeMode::ForwardMode) {
          Value *ptrshadow =
              gutils->invertPointerM(call.getArgOperand(0), BuilderZ);
          BuilderZ.CreateCall(
              called,
              {ptrshadow, gutils->getNewFromOriginal(call.getArgOperand(1)),
               gutils->getNewFromOriginal(call.getArgOperand(2))});
#if LLVM_VERSION_MAJOR > 7
          val = BuilderZ.CreateLoad(
              ptrshadow->getType()->getPointerElementType(), ptrshadow);
#else
          val = BuilderZ.CreateLoad(ptrshadow);
#endif
          val = gutils->cacheForReverse(BuilderZ, val,
                                        getIndex(orig, CacheType::Shadow));

          auto dst_arg = BuilderZ.CreateBitCast(
              val, Type::getInt8PtrTy(call.getContext()));
          auto val_arg =
              ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
          auto len_arg = BuilderZ.CreateZExtOrTrunc(
              gutils->getNewFromOriginal(call.getArgOperand(2)),
              Type::getInt64Ty(call.getContext()));
          auto volatile_arg = ConstantInt::getFalse(call.getContext());

#if LLVM_VERSION_MAJOR == 6
          auto align_arg =
              ConstantInt::get(Type::getInt32Ty(call.getContext()), 1);
          Value *nargs[] = {dst_arg, val_arg, len_arg, align_arg, volatile_arg};
#else
          Value *nargs[] = {dst_arg, val_arg, len_arg, volatile_arg};
#endif

          Type *tys[] = {dst_arg->getType(), len_arg->getType()};

          auto memset = cast<CallInst>(BuilderZ.CreateCall(
              Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                        Intrinsic::memset, tys),
              nargs));
          // memset->addParamAttr(0, Attribute::getWithAlignment(Context,
          // inst->getAlignment()));
          memset->addParamAttr(0, Attribute::NonNull);
        } else if (Mode == DerivativeMode::ReverseModeGradient) {
          PHINode *toReplace = BuilderZ.CreatePHI(
              call.getArgOperand(0)->getType()->getPointerElementType(), 1,
              orig->getName() + "_psxtmp");
          val = gutils->cacheForReverse(BuilderZ, toReplace,
                                        getIndex(orig, CacheType::Shadow));
        }

        if (Mode == DerivativeMode::ReverseModeCombined ||
            Mode == DerivativeMode::ReverseModeGradient) {
          if (shouldFree()) {
            IRBuilder<> Builder2(call.getParent());
            getReverseBuilder(Builder2);
            Value *tofree = gutils->lookupM(val, Builder2, ValueToValueMapTy(),
                                            /*tryLegalRecompute*/ false);
            auto freeCall = cast<CallInst>(
                CallInst::CreateFree(tofree, Builder2.GetInsertBlock()));
            Builder2.GetInsertBlock()->getInstList().push_back(freeCall);
          }
        }
      }

      // TODO enable this if we need to free the memory
      // NOTE THAT TOPLEVEL IS THERE SIMPLY BECAUSE THAT WAS PREVIOUS ATTITUTE
      // TO FREE'ing
      if (Mode == DerivativeMode::ReverseModeGradient) {
        eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
      } else if (Mode == DerivativeMode::ReverseModePrimal) {
        // if (is_value_needed_in_reverse<Primal>(
        //        TR, gutils, orig, /*topLevel*/ Mode ==
        //        DerivativeMode::Both))
        //        {

        //  gutils->cacheForReverse(BuilderZ, newCall,
        //                          getIndex(orig, CacheType::Self));
        //} else if (Mode != DerivativeMode::Forward) {
        // Note that here we cannot simply replace with null as users who try
        // to find the shadow pointer will use the shadow of null rather than
        // the true shadow of this
        //}
      } else if (Mode == DerivativeMode::ReverseModeCombined && shouldFree()) {
        IRBuilder<> Builder2(newCall->getNextNode());
#if LLVM_VERSION_MAJOR > 7
        auto load = Builder2.CreateLoad(
            call.getOperand(0)->getType()->getPointerElementType(),
            gutils->getNewFromOriginal(call.getOperand(0)), "posix_preread");
#else
        auto load = Builder2.CreateLoad(
            gutils->getNewFromOriginal(call.getOperand(0)), "posix_preread");
#endif
        Builder2.SetInsertPoint(&call);
        getReverseBuilder(Builder2);
        auto freeCall = cast<CallInst>(CallInst::CreateFree(
            gutils->lookupM(load, Builder2, ValueToValueMapTy(),
                            /*tryLegal*/ false),
            Builder2.GetInsertBlock()));
        Builder2.GetInsertBlock()->getInstList().push_back(freeCall);
      }

      return;
    }

    // Remove free's in forward pass so the memory can be used in the reverse
    // pass
    if (called && isDeallocationFunction(*called, gutils->TLI)) {
      assert(gutils->invertedPointers.find(orig) ==
             gutils->invertedPointers.end());

      if (Mode == DerivativeMode::ForwardMode) {
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);
          auto origfree = orig->getArgOperand(0);
          auto newfree = gutils->getNewFromOriginal(orig->getArgOperand(0));
          auto tofree = gutils->invertPointerM(origfree, Builder2);

          Function *free = getOrInsertCheckedFree(
              *orig->getModule(), orig, newfree->getType(), gutils->getWidth());

          SmallVector<Value *, 3> args;
          args.push_back(newfree);

          auto rule = [&args](Value *tofree) { args.push_back(tofree); };
          applyChainRule(Builder2, rule, tofree);

          auto frees = Builder2.CreateCall(free->getFunctionType(), free, args);
          frees->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));

          return;
        }
      }

      for (auto rmat : gutils->backwardsOnlyShadows) {
        if (rmat.second.frees.count(orig) && rmat.second.primalInitialize) {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);
          auto origfree = orig->getArgOperand(0);
          auto tofree = gutils->invertPointerM(origfree, Builder2);
          if (tofree != origfree) {
            SmallVector<Value *, 2> args = {tofree};
            CallInst *CI = Builder2.CreateCall(orig->getFunctionType(),
                                               orig->getCalledFunction(), args);
            CI->setAttributes(orig->getAttributes());
          }
          break;
        }
      }

      // If a rematerializable allocation.
      for (auto rmat : gutils->rematerializableAllocations) {
        if (rmat.second.frees.count(orig)) {

          // Leave the original free behavior since this won't be used
          // in the reverse pass in split mode
          if (Mode == DerivativeMode::ReverseModePrimal) {
            return;
          } else if (Mode == DerivativeMode::ReverseModeGradient) {
            eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
            return;
          } else {
            assert(Mode == DerivativeMode::ReverseModeCombined);
            std::map<UsageKey, bool> Seen;
            for (auto pair : gutils->knownRecomputeHeuristic)
              if (!pair.second)
                Seen[UsageKey(pair.first, ValueType::Primal)] = false;
            bool primalNeededInReverse =
                is_value_needed_in_reverse<ValueType::Primal>(
                    gutils, rmat.first, Mode, Seen, oldUnreachable);
            bool cacheWholeAllocation = false;
            if (gutils->knownRecomputeHeuristic.count(rmat.first)) {
              if (!gutils->knownRecomputeHeuristic[rmat.first]) {
                cacheWholeAllocation = true;
                primalNeededInReverse = true;
              }
            }
            // If in a loop context, maintain the same free behavior, unless
            // caching whole allocation.
            if (!cacheWholeAllocation)
              if (auto inst = dyn_cast<Instruction>(rmat.first))
                if (rmat.second.LI &&
                    rmat.second.LI->contains(inst->getParent())) {
                  return;
                }
            // In combined mode, if we don't need this allocation
            // in the reverse, we can use the original deallocation
            // behavior.
            if (!primalNeededInReverse)
              return;
          }
        }
      }

      if (gutils->forwardDeallocations.count(orig)) {
        if (Mode == DerivativeMode::ReverseModeGradient) {
          eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
        }
        return;
      }

      if (gutils->postDominatingFrees.count(orig)) {
        eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
        return;
      }

      llvm::Value *val = orig->getArgOperand(0);
      while (auto cast = dyn_cast<CastInst>(val))
        val = cast->getOperand(0);
      if (isa<ConstantPointerNull>(val)) {
        llvm::errs() << "removing free of null pointer\n";
        eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
        return;
      }

      // TODO HANDLE FREE
      llvm::errs() << "freeing without malloc " << *val << "\n";
      eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
      return;
    }

    if (gutils->isConstantInstruction(orig) && gutils->isConstantValue(orig)) {
      if (gutils->knownRecomputeHeuristic.find(orig) !=
          gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[orig]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(orig, CacheType::Self));
          eraseIfUnused(*orig);
          return;
        }
      }

      // If we need this value and it is illegal to recompute it (it writes or
      // may load uncacheable data)
      //    Store and reload it
      if (Mode != DerivativeMode::ReverseModeCombined &&
          Mode != DerivativeMode::ForwardMode && subretused &&
          (orig->mayWriteToMemory() ||
           !gutils->legalRecompute(orig, ValueToValueMapTy(), nullptr))) {
        if (!gutils->unnecessaryIntermediates.count(orig)) {

          std::map<UsageKey, bool> Seen;
          bool primalNeededInReverse = false;
          for (auto pair : gutils->knownRecomputeHeuristic)
            if (!pair.second) {
              if (pair.first == orig) {
                primalNeededInReverse = true;
                break;
              } else {
                Seen[UsageKey(pair.first, ValueType::Primal)] = false;
              }
            }
          if (!primalNeededInReverse) {

            auto minCutMode = (Mode == DerivativeMode::ReverseModePrimal)
                                  ? DerivativeMode::ReverseModeGradient
                                  : Mode;
            primalNeededInReverse =
                is_value_needed_in_reverse<ValueType::Primal>(
                    gutils, orig, minCutMode, Seen, oldUnreachable);
          }
          if (primalNeededInReverse)
            gutils->cacheForReverse(BuilderZ, newCall,
                                    getIndex(orig, CacheType::Self));
        }
        eraseIfUnused(*orig);
        return;
      }

      // If this call may write to memory and is a copy (in the just reverse
      // pass), erase it
      //  Any uses of it should be handled by the case above so it is safe to
      //  RAUW
      if (orig->mayWriteToMemory() &&
          (Mode == DerivativeMode::ReverseModeGradient ||
           Mode == DerivativeMode::ForwardModeSplit)) {
        eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
        return;
      }

      // if call does not write memory and isn't used, we can erase it
      if (!orig->mayWriteToMemory() && !subretused) {
        eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
        return;
      }

      return;
    }

    bool foreignFunction = called == nullptr;

    FnTypeInfo nextTypeInfo(called);

    if (called) {
      nextTypeInfo = TR.getCallInfo(*orig, *called);
    }

    const AugmentedReturn *subdata = nullptr;
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ForwardModeSplit) {
      assert(augmentedReturn);
      if (augmentedReturn) {
        auto fd = augmentedReturn->subaugmentations.find(&call);
        if (fd != augmentedReturn->subaugmentations.end()) {
          subdata = fd->second;
        }
      }
    }

    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeSplit) {
      IRBuilder<> Builder2(&call);
      getForwardBuilder(Builder2);

      SmallVector<Value *, 8> args;
      std::vector<DIFFE_TYPE> argsInverted;
      std::map<int, Type *> gradByVal;

#if LLVM_VERSION_MAJOR >= 14
      for (unsigned i = 0; i < orig->arg_size(); ++i)
#else
      for (unsigned i = 0; i < orig->getNumArgOperands(); ++i)
#endif
      {

        auto argi = gutils->getNewFromOriginal(orig->getArgOperand(i));

#if LLVM_VERSION_MAJOR >= 9
        if (orig->isByValArgument(i)) {
          gradByVal[args.size()] = orig->getParamByValType(i);
        }
#endif
        args.push_back(argi);

        if (gutils->isConstantValue(orig->getArgOperand(i)) &&
            !foreignFunction) {
          argsInverted.push_back(DIFFE_TYPE::CONSTANT);
          continue;
        }

        auto argType = argi->getType();

        if (!argType->isFPOrFPVectorTy() &&
            (TR.query(orig->getArgOperand(i)).Inner0().isPossiblePointer() ||
             foreignFunction)) {
          DIFFE_TYPE ty = DIFFE_TYPE::DUP_ARG;
          if (argType->isPointerTy()) {
#if LLVM_VERSION_MAJOR >= 12
            auto at = getUnderlyingObject(orig->getArgOperand(i), 100);
#else
            auto at = GetUnderlyingObject(
                orig->getArgOperand(i),
                gutils->oldFunc->getParent()->getDataLayout(), 100);
#endif
            if (auto arg = dyn_cast<Argument>(at)) {
              if (constant_args[arg->getArgNo()] == DIFFE_TYPE::DUP_NONEED) {
                ty = DIFFE_TYPE::DUP_NONEED;
              }
            }
          }
          args.push_back(
              gutils->invertPointerM(orig->getArgOperand(i), Builder2));
          argsInverted.push_back(ty);

          // Note sometimes whattype mistakenly says something should be
          // constant [because composed of integer pointers alone]
          assert(whatType(argType, Mode) == DIFFE_TYPE::DUP_ARG ||
                 whatType(argType, Mode) == DIFFE_TYPE::CONSTANT);
        } else {
          if (foreignFunction)
            assert(!argType->isIntOrIntVectorTy());

          args.push_back(diffe(orig->getArgOperand(i), Builder2));
          argsInverted.push_back(DIFFE_TYPE::DUP_ARG);
        }
      }

      Optional<int> tapeIdx;
      if (subdata) {
        auto found = subdata->returns.find(AugmentedStruct::Tape);
        if (found != subdata->returns.end()) {
          tapeIdx = found->second;
        }
      }
      Value *tape = nullptr;
      if (tapeIdx.hasValue()) {

        FunctionType *FT =
            cast<FunctionType>(subdata->fn->getType()->getPointerElementType());

        tape = BuilderZ.CreatePHI(
            (tapeIdx == -1) ? FT->getReturnType()
                            : cast<StructType>(FT->getReturnType())
                                  ->getElementType(tapeIdx.getValue()),
            1, "tapeArg");

        assert(!tape->getType()->isEmptyTy());
        gutils->TapesToPreventRecomputation.insert(cast<Instruction>(tape));
        tape = gutils->cacheForReverse(BuilderZ, tape,
                                       getIndex(orig, CacheType::Tape));
        args.push_back(tape);
      }

      Value *newcalled = nullptr;

      if (called) {
        newcalled = gutils->Logic.CreateForwardDiff(
            cast<Function>(called), subretType, argsInverted,
            TR.analyzer.interprocedural, /*returnValue*/ subretused, Mode,
            ((DiffeGradientUtils *)gutils)->FreeMemory, gutils->getWidth(),
            tape ? tape->getType() : nullptr, nextTypeInfo, uncacheable_args,
            /*augmented*/ subdata);
      } else {
#if LLVM_VERSION_MAJOR >= 11
        auto callval = orig->getCalledOperand();
#else
        auto callval = orig->getCalledValue();
#endif
        newcalled = gutils->invertPointerM(callval, BuilderZ);

        if (gutils->getWidth() > 1) {
          newcalled = BuilderZ.CreateExtractValue(newcalled, {0});
        }

        ErrorIfRuntimeInactive(BuilderZ, gutils->getNewFromOriginal(callval),
                               newcalled,
                               "Attempting to call an indirect active function "
                               "whose runtime value is inactive");

        auto ft =
            cast<FunctionType>(callval->getType()->getPointerElementType());
        bool retActive = subretType != DIFFE_TYPE::CONSTANT;

        ReturnType subretVal =
            subretused
                ? (retActive ? ReturnType::TwoReturns : ReturnType::Return)
                : (retActive ? ReturnType::Return : ReturnType::Void);

        FunctionType *FTy = getFunctionTypeForClone(
            ft, Mode, gutils->getWidth(), tape ? tape->getType() : nullptr,
            argsInverted, false, subretVal, subretType);
        PointerType *fptype = PointerType::getUnqual(FTy);
        newcalled = BuilderZ.CreatePointerCast(newcalled,
                                               PointerType::getUnqual(fptype));
#if LLVM_VERSION_MAJOR > 7
        newcalled = BuilderZ.CreateLoad(fptype, newcalled);
#else
        newcalled = BuilderZ.CreateLoad(newcalled);
#endif
      }

      assert(newcalled);
      FunctionType *FT =
          cast<FunctionType>(newcalled->getType()->getPointerElementType());

      SmallVector<ValueType, 2> BundleTypes;
      for (auto A : argsInverted)
        if (A == DIFFE_TYPE::CONSTANT)
          BundleTypes.push_back(ValueType::Primal);
        else
          BundleTypes.push_back(ValueType::Both);

      auto Defs = gutils->getInvertedBundles(orig, BundleTypes, Builder2,
                                             /*lookup*/ false);

#if LLVM_VERSION_MAJOR > 7
      CallInst *diffes = Builder2.CreateCall(FT, newcalled, args, Defs);
#else
      CallInst *diffes = Builder2.CreateCall(newcalled, args, Defs);
#endif
      diffes->setCallingConv(orig->getCallingConv());
      diffes->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
#if LLVM_VERSION_MAJOR >= 9
      for (auto pair : gradByVal) {
        diffes->addParamAttr(
            pair.first,
            Attribute::getWithByValType(diffes->getContext(), pair.second));
      }
#endif

      auto newcall = gutils->getNewFromOriginal(orig);
      auto ifound = gutils->invertedPointers.find(orig);
      Value *primal = nullptr;
      Value *diffe = nullptr;

      if (subretused && subretType != DIFFE_TYPE::CONSTANT) {
        primal = Builder2.CreateExtractValue(diffes, 0);
        diffe = Builder2.CreateExtractValue(diffes, 1);
      } else if (subretType != DIFFE_TYPE::CONSTANT) {
        diffe = diffes;
      } else if (!FT->getReturnType()->isVoidTy()) {
        primal = diffes;
      }

      if (ifound != gutils->invertedPointers.end()) {
        auto placeholder = cast<PHINode>(&*ifound->second);
        if (primal) {
          gutils->replaceAWithB(newcall, primal);
          gutils->erase(newcall);
        } else {
          eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
        }
        if (diffe) {
          gutils->replaceAWithB(placeholder, diffe);
        } else {
          gutils->invertedPointers.erase(ifound);
        }
        gutils->erase(placeholder);
      } else {
        if (primal && diffe) {
          gutils->replaceAWithB(newcall, primal);
          if (!gutils->isConstantValue(&call)) {
            setDiffe(&call, diffe, Builder2);
          }
          gutils->erase(newcall);
        } else if (diffe) {
          setDiffe(&call, diffe, Builder2);
          eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
        } else if (primal) {
          gutils->replaceAWithB(newcall, primal);
          gutils->erase(newcall);
        } else {
          eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
        }
      }

      return;
    }

    bool modifyPrimal = shouldAugmentCall(orig, gutils);

    SmallVector<Value *, 8> args;
    SmallVector<Value *, 8> pre_args;
    std::vector<DIFFE_TYPE> argsInverted;
    SmallVector<Instruction *, 4> postCreate;
    SmallVector<Instruction *, 4> userReplace;
    std::map<int, Type *> preByVal;
    std::map<int, Type *> gradByVal;

#if LLVM_VERSION_MAJOR >= 14
    for (unsigned i = 0; i < orig->arg_size(); ++i)
#else
    for (unsigned i = 0; i < orig->getNumArgOperands(); ++i)
#endif
    {

      auto argi = gutils->getNewFromOriginal(orig->getArgOperand(i));

#if LLVM_VERSION_MAJOR >= 9
      if (orig->isByValArgument(i)) {
        preByVal[pre_args.size()] = orig->getParamByValType(i);
      }
#endif

      pre_args.push_back(argi);

      if (Mode != DerivativeMode::ReverseModePrimal) {
        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);
#if LLVM_VERSION_MAJOR >= 9
        if (orig->isByValArgument(i)) {
          gradByVal[args.size()] = orig->getParamByValType(i);
        }
#endif
        args.push_back(lookup(argi, Builder2));
      }

      if (gutils->isConstantValue(orig->getArgOperand(i)) && !foreignFunction) {
        argsInverted.push_back(DIFFE_TYPE::CONSTANT);
        continue;
      }

      auto argType = argi->getType();

      if (!argType->isFPOrFPVectorTy() &&
          (TR.query(orig->getArgOperand(i)).Inner0().isPossiblePointer() ||
           foreignFunction)) {
        DIFFE_TYPE ty = DIFFE_TYPE::DUP_ARG;
        if (argType->isPointerTy()) {
#if LLVM_VERSION_MAJOR >= 12
          auto at = getUnderlyingObject(orig->getArgOperand(i), 100);
#else
          auto at = GetUnderlyingObject(
              orig->getArgOperand(i),
              gutils->oldFunc->getParent()->getDataLayout(), 100);
#endif
          if (auto arg = dyn_cast<Argument>(at)) {
            if (constant_args[arg->getArgNo()] == DIFFE_TYPE::DUP_NONEED) {
              ty = DIFFE_TYPE::DUP_NONEED;
            }
          }
        }
        argsInverted.push_back(ty);

        if (Mode != DerivativeMode::ReverseModePrimal) {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);
          args.push_back(
              lookup(gutils->invertPointerM(orig->getArgOperand(i), Builder2),
                     Builder2));
        }
        pre_args.push_back(
            gutils->invertPointerM(orig->getArgOperand(i), BuilderZ));

        // Note sometimes whattype mistakenly says something should be
        // constant [because composed of integer pointers alone]
        assert(whatType(argType, Mode) == DIFFE_TYPE::DUP_ARG ||
               whatType(argType, Mode) == DIFFE_TYPE::CONSTANT);
      } else {
        if (foreignFunction)
          assert(!argType->isIntOrIntVectorTy());
        argsInverted.push_back(DIFFE_TYPE::OUT_DIFF);
        assert(whatType(argType, Mode) == DIFFE_TYPE::OUT_DIFF ||
               whatType(argType, Mode) == DIFFE_TYPE::CONSTANT);
      }
    }
    SmallVector<ValueType, 2> BundleTypes;
    for (auto A : argsInverted)
      if (A == DIFFE_TYPE::CONSTANT)
        BundleTypes.push_back(ValueType::Primal);
      else
        BundleTypes.push_back(ValueType::Both);
    if (called) {
#if LLVM_VERSION_MAJOR >= 14
      if (orig->arg_size() !=
          cast<Function>(called)->getFunctionType()->getNumParams())
#else
      if (orig->getNumArgOperands() !=
          cast<Function>(called)->getFunctionType()->getNumParams())
#endif
      {
        llvm::errs() << *gutils->oldFunc << "\n";
        llvm::errs() << *orig << "\n";
        assert(0 && "number of arg operands != function parameters");
      }
      assert(argsInverted.size() ==
             cast<Function>(called)->getFunctionType()->getNumParams());
    }

    bool replaceFunction = false;

    if (Mode == DerivativeMode::ReverseModeCombined && !foreignFunction) {
      replaceFunction = legalCombinedForwardReverse(
          orig, *replacedReturns, postCreate, userReplace, gutils,
          unnecessaryInstructions, oldUnreachable, subretused);
      if (replaceFunction)
        modifyPrimal = false;
    }

    Value *tape = nullptr;
    CallInst *augmentcall = nullptr;
    Value *cachereplace = nullptr;

    // llvm::Optional<std::map<std::pair<Instruction*, std::string>,
    // unsigned>> sub_index_map;
    Optional<int> tapeIdx;
    Optional<int> returnIdx;
    Optional<int> differetIdx;

    if (modifyPrimal) {

      Value *newcalled = nullptr;
      const AugmentedReturn *fnandtapetype = nullptr;

      if (!called) {
#if LLVM_VERSION_MAJOR >= 11
        auto callval = orig->getCalledOperand();
#else
        auto callval = orig->getCalledValue();
#endif
        newcalled = gutils->invertPointerM(callval, BuilderZ);

        if (Mode != DerivativeMode::ReverseModeGradient)
          ErrorIfRuntimeInactive(
              BuilderZ, gutils->getNewFromOriginal(callval), newcalled,
              "Attempting to call an indirect active function "
              "whose runtime value is inactive");

        auto ft =
            cast<FunctionType>(callval->getType()->getPointerElementType());

        DIFFE_TYPE subretType = orig->getType()->isFPOrFPVectorTy()
                                    ? DIFFE_TYPE::OUT_DIFF
                                    : DIFFE_TYPE::DUP_ARG;
        if (orig->getType()->isVoidTy() || orig->getType()->isEmptyTy())
          subretType = DIFFE_TYPE::CONSTANT;
        auto res = getDefaultFunctionTypeForAugmentation(
            ft, /*returnUsed*/ true, /*subretType*/ subretType);
        auto fptype = PointerType::getUnqual(FunctionType::get(
            StructType::get(newcalled->getContext(), res.second), res.first,
            ft->isVarArg()));
        newcalled = BuilderZ.CreatePointerCast(newcalled,
                                               PointerType::getUnqual(fptype));
#if LLVM_VERSION_MAJOR > 7
        newcalled = BuilderZ.CreateLoad(fptype, newcalled);
#else
        newcalled = BuilderZ.CreateLoad(newcalled);
#endif
        tapeIdx = 0;

        if (!orig->getType()->isVoidTy()) {
          returnIdx = 1;
          if (subretType == DIFFE_TYPE::DUP_ARG ||
              subretType == DIFFE_TYPE::DUP_NONEED) {
            differetIdx = 2;
          }
        }

      } else {
        if (Mode == DerivativeMode::ReverseModePrimal ||
            Mode == DerivativeMode::ReverseModeCombined) {
          subdata = &gutils->Logic.CreateAugmentedPrimal(
              cast<Function>(called), subretType, argsInverted,
              TR.analyzer.interprocedural, /*return is used*/ subretused,
              shadowReturnUsed, nextTypeInfo, uncacheable_args, false,
              gutils->getWidth(), gutils->AtomicAdd);
          if (Mode == DerivativeMode::ReverseModePrimal) {
            assert(augmentedReturn);
            auto subaugmentations =
                (std::map<const llvm::CallInst *, AugmentedReturn *>
                     *)&augmentedReturn->subaugmentations;
            insert_or_assign2<const llvm::CallInst *, AugmentedReturn *>(
                *subaugmentations, orig, (AugmentedReturn *)subdata);
          }
        }
        if (!subdata) {
          llvm::errs() << *gutils->oldFunc->getParent() << "\n";
          llvm::errs() << *gutils->oldFunc << "\n";
          llvm::errs() << *gutils->newFunc << "\n";
          llvm::errs() << *called << "\n";
        }
        assert(subdata);
        fnandtapetype = subdata;
        newcalled = subdata->fn;

        auto found = subdata->returns.find(AugmentedStruct::DifferentialReturn);
        if (found != subdata->returns.end()) {
          differetIdx = found->second;
        }

        found = subdata->returns.find(AugmentedStruct::Return);
        if (found != subdata->returns.end()) {
          returnIdx = found->second;
        }

        found = subdata->returns.find(AugmentedStruct::Tape);
        if (found != subdata->returns.end()) {
          tapeIdx = found->second;
        }
      }
      // sub_index_map = fnandtapetype.tapeIndices;

      assert(newcalled);
      FunctionType *FT =
          cast<FunctionType>(newcalled->getType()->getPointerElementType());

      // llvm::errs() << "seeing sub_index_map of " << sub_index_map->size()
      // << " in ap " << cast<Function>(called)->getName() << "\n";
      if (Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ReverseModePrimal) {

        if (false) {
        badaugmentedfn:;
          auto NC = dyn_cast<Function>(newcalled);
          llvm::errs() << *gutils->oldFunc << "\n";
          llvm::errs() << *gutils->newFunc << "\n";
          if (NC)
            llvm::errs() << " trying to call " << NC->getName() << " " << *FT
                         << "\n";
          else
            llvm::errs() << " trying to call " << *newcalled << " " << *FT
                         << "\n";

          for (unsigned i = 0; i < pre_args.size(); ++i) {
            assert(pre_args[i]);
            assert(pre_args[i]->getType());
            llvm::errs() << "args[" << i << "] = " << *pre_args[i]
                         << " FT:" << *FT->getParamType(i) << "\n";
          }
          assert(0 && "calling with wrong number of arguments");
          exit(1);
        }

        if (pre_args.size() != FT->getNumParams())
          goto badaugmentedfn;

        for (unsigned i = 0; i < pre_args.size(); ++i) {
          if (pre_args[i]->getType() == FT->getParamType(i))
            continue;
          else if (!orig->getCalledFunction())
            pre_args[i] =
                BuilderZ.CreateBitCast(pre_args[i], FT->getParamType(i));
          else
            goto badaugmentedfn;
        }

#if LLVM_VERSION_MAJOR > 7
        augmentcall = BuilderZ.CreateCall(
            FT, newcalled, pre_args,
            gutils->getInvertedBundles(orig, BundleTypes, BuilderZ,
                                       /*lookup*/ false));
#else
        augmentcall = BuilderZ.CreateCall(
            newcalled, pre_args,
            gutils->getInvertedBundles(orig, BundleTypes, BuilderZ,
                                       /*lookup*/ false));
#endif
        augmentcall->setCallingConv(orig->getCallingConv());
        augmentcall->setDebugLoc(
            gutils->getNewFromOriginal(orig->getDebugLoc()));
#if LLVM_VERSION_MAJOR >= 9
        for (auto pair : preByVal) {
          augmentcall->addParamAttr(
              pair.first, Attribute::getWithByValType(augmentcall->getContext(),
                                                      pair.second));
        }
#endif

        if (!augmentcall->getType()->isVoidTy())
          augmentcall->setName(orig->getName() + "_augmented");

        if (tapeIdx.hasValue()) {
          tape = (tapeIdx.getValue() == -1)
                     ? augmentcall
                     : BuilderZ.CreateExtractValue(
                           augmentcall, {(unsigned)tapeIdx.getValue()},
                           "subcache");
          if (tape->getType()->isEmptyTy()) {
            auto tt = tape->getType();
            gutils->erase(cast<Instruction>(tape));
            tape = UndefValue::get(tt);
          } else {
            gutils->TapesToPreventRecomputation.insert(cast<Instruction>(tape));
          }
          tape = gutils->cacheForReverse(BuilderZ, tape,
                                         getIndex(orig, CacheType::Tape));
        }

        if (subretused) {
          Value *dcall = nullptr;
          assert(returnIdx);
          dcall = (returnIdx.getValue() < 0)
                      ? augmentcall
                      : BuilderZ.CreateExtractValue(
                            augmentcall, {(unsigned)returnIdx.getValue()});
          gutils->originalToNewFn[orig] = dcall;
          gutils->newToOriginalFn.erase(newCall);
          gutils->newToOriginalFn[dcall] = orig;

          assert(dcall->getType() == orig->getType());
          assert(dcall);

          if (!gutils->isConstantValue(orig)) {
            if (!orig->getType()->isFPOrFPVectorTy() &&
                TR.query(orig).Inner0().isPossiblePointer()) {
            } else if (Mode != DerivativeMode::ReverseModePrimal) {
              ((DiffeGradientUtils *)gutils)->differentials[dcall] =
                  ((DiffeGradientUtils *)gutils)->differentials[newCall];
              ((DiffeGradientUtils *)gutils)->differentials.erase(newCall);
            }
          }
          assert(dcall->getType() == orig->getType());
          gutils->replaceAWithB(newCall, dcall);

          if (isa<Instruction>(dcall) && !isa<PHINode>(dcall)) {
            cast<Instruction>(dcall)->takeName(newCall);
          }

          if (Mode == DerivativeMode::ReverseModePrimal &&
              !gutils->unnecessaryIntermediates.count(orig)) {

            std::map<UsageKey, bool> Seen;
            bool primalNeededInReverse = false;
            for (auto pair : gutils->knownRecomputeHeuristic)
              if (!pair.second) {
                if (pair.first == orig) {
                  primalNeededInReverse = true;
                  break;
                } else {
                  Seen[UsageKey(pair.first, ValueType::Primal)] = false;
                }
              }
            if (!primalNeededInReverse) {

              auto minCutMode = (Mode == DerivativeMode::ReverseModePrimal)
                                    ? DerivativeMode::ReverseModeGradient
                                    : Mode;
              primalNeededInReverse =
                  is_value_needed_in_reverse<ValueType::Primal>(
                      gutils, orig, minCutMode, Seen, oldUnreachable);
            }
            if (primalNeededInReverse)
              gutils->cacheForReverse(BuilderZ, dcall,
                                      getIndex(orig, CacheType::Self));
          }
          BuilderZ.SetInsertPoint(newCall->getNextNode());
          gutils->erase(newCall);
        } else {
          BuilderZ.SetInsertPoint(BuilderZ.GetInsertPoint()->getNextNode());
          eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
          gutils->originalToNewFn[orig] = augmentcall;
          gutils->newToOriginalFn[augmentcall] = orig;
        }

      } else {
        if (subdata && subdata->returns.find(AugmentedStruct::Tape) ==
                           subdata->returns.end()) {
        } else {
          // assert(!tape);
          // assert(subdata);
          if (!tape) {
            assert(tapeIdx.hasValue());
            tape = BuilderZ.CreatePHI(
                (tapeIdx == -1) ? FT->getReturnType()
                                : cast<StructType>(FT->getReturnType())
                                      ->getElementType(tapeIdx.getValue()),
                1, "tapeArg");
          }
          tape = gutils->cacheForReverse(BuilderZ, tape,
                                         getIndex(orig, CacheType::Tape));
        }

        if (subretused) {
          if (is_value_needed_in_reverse<ValueType::Primal>(gutils, orig, Mode,
                                                            oldUnreachable) &&
              !gutils->unnecessaryIntermediates.count(orig)) {
            cachereplace = BuilderZ.CreatePHI(orig->getType(), 1,
                                              orig->getName() + "_tmpcacheB");
            cachereplace = gutils->cacheForReverse(
                BuilderZ, cachereplace, getIndex(orig, CacheType::Self));
          } else {
            auto pn = BuilderZ.CreatePHI(
                orig->getType(), 1, (orig->getName() + "_replacementE").str());
            gutils->fictiousPHIs[pn] = orig;
            cachereplace = pn;
          }
        } else {
          // TODO move right after newCall for the insertion point of BuilderZ

          BuilderZ.SetInsertPoint(BuilderZ.GetInsertPoint()->getNextNode());
          eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
        }
      }

      auto ifound = gutils->invertedPointers.find(orig);
      if (ifound != gutils->invertedPointers.end()) {
        auto placeholder = cast<PHINode>(&*ifound->second);

        bool subcheck = (subretType == DIFFE_TYPE::DUP_ARG ||
                         subretType == DIFFE_TYPE::DUP_NONEED);

        //! We only need the shadow pointer for non-forward Mode if it is used
        //! in a non return setting
        bool hasNonReturnUse = false;
        for (auto use : orig->users()) {
          if (Mode == DerivativeMode::ReverseModePrimal ||
              !isa<ReturnInst>(
                  use)) { // || returnuses.find(cast<Instruction>(use)) ==
                          // returnuses.end()) {
            hasNonReturnUse = true;
          }
        }

        if (subcheck && hasNonReturnUse) {

          Value *newip = nullptr;
          if (Mode == DerivativeMode::ReverseModeCombined ||
              Mode == DerivativeMode::ReverseModePrimal) {
            newip = (differetIdx.getValue() < 0)
                        ? augmentcall
                        : BuilderZ.CreateExtractValue(
                              augmentcall, {(unsigned)differetIdx.getValue()},
                              orig->getName() + "'ac");
            assert(newip->getType() == orig->getType());
            placeholder->replaceAllUsesWith(newip);
            gutils->erase(placeholder);
          } else {
            newip = placeholder;
          }

          newip = gutils->cacheForReverse(BuilderZ, newip,
                                          getIndex(orig, CacheType::Shadow));

          gutils->invertedPointers.insert(std::make_pair(
              (const Value *)orig, InvertedPointerVH(gutils, newip)));
        } else {
          gutils->invertedPointers.erase(ifound);
          if (placeholder == &*BuilderZ.GetInsertPoint()) {
            BuilderZ.SetInsertPoint(placeholder->getNextNode());
          }
          gutils->erase(placeholder);
        }
      }

      if (fnandtapetype && fnandtapetype->tapeType &&
          (Mode == DerivativeMode::ReverseModeCombined ||
           Mode == DerivativeMode::ReverseModeGradient ||
           Mode == DerivativeMode::ForwardModeSplit) &&
          shouldFree()) {
        assert(tape);
        auto tapep = BuilderZ.CreatePointerCast(
            tape, PointerType::getUnqual(fnandtapetype->tapeType));
#if LLVM_VERSION_MAJOR > 7
        auto truetape =
            BuilderZ.CreateLoad(fnandtapetype->tapeType, tapep, "tapeld");
#else
        auto truetape = BuilderZ.CreateLoad(tapep, "tapeld");
#endif
        truetape->setMetadata("enzyme_mustcache",
                              MDNode::get(truetape->getContext(), {}));

        CallInst *ci = cast<CallInst>(
            CallInst::CreateFree(tape, &*BuilderZ.GetInsertPoint()));
#if LLVM_VERSION_MAJOR >= 14
        ci->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                Attribute::NonNull);
#else
        ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
#endif
        tape = truetape;
      }
    } else {
      auto ifound = gutils->invertedPointers.find(orig);
      if (ifound != gutils->invertedPointers.end()) {
        auto placeholder = cast<PHINode>(&*ifound->second);
        gutils->invertedPointers.erase(ifound);
        gutils->erase(placeholder);
      }
      if (/*!topLevel*/ Mode != DerivativeMode::ReverseModeCombined &&
          subretused && !orig->doesNotAccessMemory()) {
        if (is_value_needed_in_reverse<ValueType::Primal>(gutils, orig, Mode,
                                                          oldUnreachable) &&
            !gutils->unnecessaryIntermediates.count(orig)) {
          assert(!replaceFunction);
          cachereplace = BuilderZ.CreatePHI(orig->getType(), 1,
                                            orig->getName() + "_cachereplace2");
          cachereplace = gutils->cacheForReverse(
              BuilderZ, cachereplace, getIndex(orig, CacheType::Self));
        } else {
          auto pn = BuilderZ.CreatePHI(
              orig->getType(), 1, (orig->getName() + "_replacementC").str());
          gutils->fictiousPHIs[pn] = orig;
          cachereplace = pn;
        }
      }

      if (!subretused && !replaceFunction)
        eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
    }

    // Note here down only contains the reverse bits
    if (Mode == DerivativeMode::ReverseModePrimal) {
      return;
    }

    IRBuilder<> Builder2(call.getParent());
    getReverseBuilder(Builder2);

    Value *newcalled = nullptr;

    DerivativeMode subMode = (replaceFunction || !modifyPrimal)
                                 ? DerivativeMode::ReverseModeCombined
                                 : DerivativeMode::ReverseModeGradient;
    if (called) {
      newcalled = gutils->Logic.CreatePrimalAndGradient(
          (ReverseCacheKey){.todiff = cast<Function>(called),
                            .retType = subretType,
                            .constant_args = argsInverted,
                            .uncacheable_args = uncacheable_args,
                            .returnUsed = replaceFunction && subretused,
                            .shadowReturnUsed =
                                shadowReturnUsed && replaceFunction,
                            .mode = subMode,
                            .width = gutils->getWidth(),
                            .freeMemory = true,
                            .AtomicAdd = gutils->AtomicAdd,
                            .additionalType = tape ? tape->getType() : nullptr,
                            .typeInfo = nextTypeInfo},
          TR.analyzer.interprocedural, subdata);
      if (!newcalled)
        return;
    } else {

      assert(subMode != DerivativeMode::ReverseModeCombined);

#if LLVM_VERSION_MAJOR >= 11
      auto callval = orig->getCalledOperand();
#else
      auto callval = orig->getCalledValue();
#endif

      if (gutils->isConstantValue(callval)) {
        llvm::errs() << *gutils->newFunc->getParent() << "\n";
        llvm::errs() << " orig: " << *orig << " callval: " << *callval << "\n";
      }
      assert(!gutils->isConstantValue(callval));
      newcalled = lookup(gutils->invertPointerM(callval, Builder2), Builder2);

      auto ft = cast<FunctionType>(callval->getType()->getPointerElementType());

      DIFFE_TYPE subretType = orig->getType()->isFPOrFPVectorTy()
                                  ? DIFFE_TYPE::OUT_DIFF
                                  : DIFFE_TYPE::DUP_ARG;
      if (orig->getType()->isVoidTy() || orig->getType()->isEmptyTy())
        subretType = DIFFE_TYPE::CONSTANT;
      auto res =
          getDefaultFunctionTypeForGradient(ft, /*subretType*/ subretType);
      // TODO Note there is empty tape added here, replace with generic
      res.first.push_back(Type::getInt8PtrTy(newcalled->getContext()));
      auto fptype = PointerType::getUnqual(FunctionType::get(
          StructType::get(newcalled->getContext(), res.second), res.first,
          ft->isVarArg()));
      newcalled =
          Builder2.CreatePointerCast(newcalled, PointerType::getUnqual(fptype));
#if LLVM_VERSION_MAJOR > 7
      newcalled = Builder2.CreateLoad(
          fptype, Builder2.CreateConstGEP1_64(fptype, newcalled, 1));
#else
      newcalled =
          Builder2.CreateLoad(Builder2.CreateConstGEP1_64(newcalled, 1));
#endif
    }

    if (subretType == DIFFE_TYPE::OUT_DIFF) {
      args.push_back(diffe(orig, Builder2));
    }

    if (tape) {
      auto ntape = gutils->lookupM(tape, Builder2);
      assert(ntape);
      assert(ntape->getType());
      args.push_back(ntape);
    }

    assert(newcalled);
    // if (auto NC = dyn_cast<Function>(newcalled)) {
    FunctionType *FT =
        cast<FunctionType>(newcalled->getType()->getPointerElementType());

    if (false) {
    badfn:;
      auto NC = dyn_cast<Function>(newcalled);
      llvm::errs() << *gutils->oldFunc << "\n";
      llvm::errs() << *gutils->newFunc << "\n";
      if (NC)
        llvm::errs() << " trying to call " << NC->getName() << " " << *FT
                     << "\n";
      else
        llvm::errs() << " trying to call " << *newcalled << " " << *FT << "\n";

      for (unsigned i = 0; i < args.size(); ++i) {
        assert(args[i]);
        assert(args[i]->getType());
        llvm::errs() << "args[" << i << "] = " << *args[i]
                     << " FT:" << *FT->getParamType(i) << "\n";
      }
      assert(0 && "calling with wrong number of arguments");
      exit(1);
    }

    if (args.size() != FT->getNumParams())
      goto badfn;

    for (unsigned i = 0; i < args.size(); ++i) {
      if (args[i]->getType() == FT->getParamType(i))
        continue;
      else if (!orig->getCalledFunction())
        args[i] = Builder2.CreateBitCast(args[i], FT->getParamType(i));
      else
        goto badfn;
    }

#if LLVM_VERSION_MAJOR > 7
    CallInst *diffes =
        Builder2.CreateCall(FT, newcalled, args,
                            gutils->getInvertedBundles(
                                orig, BundleTypes, Builder2, /*lookup*/ true));
#else
    CallInst *diffes =
        Builder2.CreateCall(newcalled, args,
                            gutils->getInvertedBundles(
                                orig, BundleTypes, Builder2, /*lookup*/ true));
#endif
    diffes->setCallingConv(orig->getCallingConv());
    diffes->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
#if LLVM_VERSION_MAJOR >= 9
    for (auto pair : gradByVal) {
      diffes->addParamAttr(pair.first, Attribute::getWithByValType(
                                           diffes->getContext(), pair.second));
    }
#endif

    unsigned structidx = 0;
    if (replaceFunction) {
      if (subretused)
        structidx++;
      if (shadowReturnUsed)
        structidx++;
    }

#if LLVM_VERSION_MAJOR >= 14
    for (unsigned i = 0; i < orig->arg_size(); ++i)
#else
    for (unsigned i = 0; i < orig->getNumArgOperands(); ++i)
#endif
    {
      if (argsInverted[i] == DIFFE_TYPE::OUT_DIFF) {
        Value *diffeadd = Builder2.CreateExtractValue(diffes, {structidx});
        ++structidx;

        size_t size = 1;
        if (orig->getArgOperand(i)->getType()->isSized())
          size =
              (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                   orig->getArgOperand(i)->getType()) +
               7) /
              8;

        addToDiffe(orig->getArgOperand(i), diffeadd, Builder2,
                   TR.addingType(size, orig->getArgOperand(i)));
      }
    }

    if (diffes->getType()->isVoidTy()) {
      if (structidx != 0) {
        llvm::errs() << *gutils->oldFunc->getParent() << "\n";
        llvm::errs() << "diffes: " << *diffes << " structidx=" << structidx
                     << " subretused=" << subretused
                     << " shadowReturnUsed=" << shadowReturnUsed << "\n";
      }
      assert(structidx == 0);
    } else {
      assert(cast<StructType>(diffes->getType())->getNumElements() ==
             structidx);
    }

    if (subretType == DIFFE_TYPE::OUT_DIFF)
      setDiffe(orig,
               Constant::getNullValue(gutils->getShadowType(orig->getType())),
               Builder2);

    if (replaceFunction) {

      // if a function is replaced for joint forward/reverse, handle inverted
      // pointers
      auto ifound = gutils->invertedPointers.find(orig);
      if (ifound != gutils->invertedPointers.end()) {
        auto placeholder = cast<PHINode>(&*ifound->second);
        gutils->invertedPointers.erase(ifound);
        if (shadowReturnUsed) {
          dumpMap(gutils->invertedPointers);
          auto dretval = cast<Instruction>(
              Builder2.CreateExtractValue(diffes, {subretused ? 1U : 0U}));
          /* todo handle this case later */
          assert(!subretused);
          gutils->invertedPointers.insert(std::make_pair(
              (const Value *)orig, InvertedPointerVH(gutils, dretval)));
        }
        gutils->erase(placeholder);
      }

      Instruction *retval = nullptr;

      ValueToValueMapTy mapp;
      if (subretused) {
        retval = cast<Instruction>(Builder2.CreateExtractValue(diffes, {0}));
        gutils->replaceAWithB(newCall, retval, /*storeInCache*/ true);
        mapp[newCall] = retval;
      } else {
        eraseIfUnused(*orig, /*erase*/ false, /*check*/ false);
      }

      for (auto &a : *gutils
                          ->reverseBlocks[cast<BasicBlock>(
                              gutils->getNewFromOriginal(orig->getParent()))]
                          .back()) {
        mapp[&a] = &a;
      }

      std::reverse(postCreate.begin(), postCreate.end());
      for (auto a : postCreate) {

        // If is the store to return handle manually since no original inst
        // for
        bool fromStore = false;
        for (auto &pair : *replacedReturns) {
          if (pair.second == a) {
            for (unsigned i = 0; i < a->getNumOperands(); ++i) {
              a->setOperand(i, gutils->unwrapM(a->getOperand(i), Builder2, mapp,
                                               UnwrapMode::LegalFullUnwrap));
            }
            a->moveBefore(*Builder2.GetInsertBlock(),
                          Builder2.GetInsertPoint());
            fromStore = true;
            break;
          }
        }
        if (fromStore)
          continue;

        auto orig_a = gutils->isOriginal(a);
        if (orig_a) {
          for (unsigned i = 0; i < a->getNumOperands(); ++i) {
            a->setOperand(i,
                          gutils->unwrapM(
                              gutils->getNewFromOriginal(orig_a->getOperand(i)),
                              Builder2, mapp, UnwrapMode::LegalFullUnwrap));
          }
        }
        a->moveBefore(*Builder2.GetInsertBlock(), Builder2.GetInsertPoint());
        mapp[a] = a;
      }

      gutils->originalToNewFn[orig] = retval ? retval : diffes;
      gutils->newToOriginalFn.erase(newCall);
      gutils->newToOriginalFn[retval ? retval : diffes] = orig;

      // llvm::errs() << "newFunc postrep: " << *gutils->newFunc << "\n";

      erased.insert(orig);
      gutils->erase(newCall);

      return;
    }

    if (cachereplace) {
      if (subretused) {
        Value *dcall = nullptr;
        assert(cachereplace->getType() == orig->getType());
        assert(dcall == nullptr);
        dcall = cachereplace;
        assert(dcall);

        if (!gutils->isConstantValue(orig)) {
          gutils->originalToNewFn[orig] = dcall;
          gutils->newToOriginalFn.erase(newCall);
          gutils->newToOriginalFn[dcall] = orig;
          if (!orig->getType()->isFPOrFPVectorTy() &&
              TR.query(orig).Inner0().isPossiblePointer()) {
          } else {
            ((DiffeGradientUtils *)gutils)->differentials[dcall] =
                ((DiffeGradientUtils *)gutils)->differentials[newCall];
            ((DiffeGradientUtils *)gutils)->differentials.erase(newCall);
          }
        }
        assert(dcall->getType() == orig->getType());
        newCall->replaceAllUsesWith(dcall);
        if (isa<Instruction>(dcall) && !isa<PHINode>(dcall)) {
          cast<Instruction>(dcall)->takeName(orig);
        }
        gutils->erase(newCall);
      } else {
        eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
        if (augmentcall) {
          gutils->originalToNewFn[orig] = augmentcall;
          gutils->newToOriginalFn.erase(newCall);
          gutils->newToOriginalFn[augmentcall] = orig;
        }
      }
    }
    return;
  }
};
