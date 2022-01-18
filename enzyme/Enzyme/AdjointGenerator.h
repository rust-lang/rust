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
  const std::vector<DIFFE_TYPE> &constant_args;
  DIFFE_TYPE retType;
  TypeResults &TR;
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
      const std::vector<DIFFE_TYPE> &constant_args, DIFFE_TYPE retType,
      TypeResults &TR,
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
        retType(retType), TR(TR), getIndex(getIndex),
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
    if (!I.getType()->isVoidTy() && isa<Instruction>(iload)) {
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
    return B.CreateLoad(alloc);
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
    return B.CreateLoad(alloc);
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
    return B.CreateLoad(alloc);
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

      Value *idiff = diffe(&inst, Builder2);
      Value *dif1 = Builder2.CreateFreeze(idiff);
      setDiffe(&inst, Constant::getNullValue(inst.getType()), Builder2);
      size_t size = 1;
      if (inst.getType()->isSized())
        size = (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                    orig_op0->getType()) +
                7) /
               8;
      addToDiffe(orig_op0, dif1, Builder2, TR.addingType(size, orig_op0));
      return;
    }
    case DerivativeMode::ForwardMode: {
      IRBuilder<> BuilderZ(&inst);
      getForwardBuilder(BuilderZ);

      Value *idiff = diffe(orig_op0, BuilderZ);
      Value *dif1 = BuilderZ.CreateFreeze(idiff);
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

          Value *idiff = diffe(FPMO, Builder2);
          Value *dif1 = Builder2.CreateFNeg(idiff);
          setDiffe(FPMO, Constant::getNullValue(FPMO->getType()), Builder2);
          addToDiffe(orig_op1, dif1, Builder2,
                     dif1->getType()->getScalarType());
          break;
        }
        case DerivativeMode::ForwardMode: {
          IRBuilder<> Builder2(&inst);
          getForwardBuilder(Builder2);

          Value *idiff = diffe(orig_op1, Builder2);
          Value *dif1 = Builder2.CreateFNeg(idiff);
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

    llvm::errs() << *gutils->oldFunc << "\n";
    llvm::errs() << *gutils->newFunc << "\n";
    llvm::errs() << "in Mode: " << to_string(Mode) << "\n";
    llvm::errs() << "cannot handle unknown instruction\n" << inst;
    report_fatal_error("unknown value");
  }

  void visitAllocaInst(llvm::AllocaInst &I) { eraseIfUnused(I); }
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
    Type *type = I.getType();

    auto *newi = dyn_cast<Instruction>(gutils->getNewFromOriginal(&I));

    //! Store inverted pointer loads that need to be cached for use in reverse
    //! pass
    if (!type->isEmptyTy() && !type->isFPOrFPVectorTy() &&
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
        bool needShadow =
            Mode == DerivativeMode::ForwardMode
                ? false
                : is_value_needed_in_reverse<ValueType::ShadowPtr>(
                      TR, gutils, &I, Mode, oldUnreachable);

        switch (Mode) {

        case DerivativeMode::ReverseModePrimal:
        case DerivativeMode::ReverseModeCombined: {
          if (!needShadow) {
            gutils->erase(placeholder);
          } else {
            newip = gutils->invertPointerM(&I, BuilderZ);
            assert(newip->getType() == type);
            if (Mode == DerivativeMode::ReverseModePrimal && can_modref) {
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
        case DerivativeMode::ForwardMode: {
          newip = gutils->invertPointerM(&I, BuilderZ);
          assert(newip->getType() == type);
          placeholder->replaceAllUsesWith(newip);
          gutils->erase(placeholder);
          gutils->invertedPointers.insert(std::make_pair(
              (const Value *)&I, InvertedPointerVH(gutils, newip)));
          break;
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

    // Allow forcing cache reads to be on or off using flags.
    assert(!(cache_reads_always && cache_reads_never) &&
           "Both cache_reads_always and cache_reads_never are true. This "
           "doesn't make sense.");

    Value *inst = newi;

    //! Store loads that need to be cached for use in reverse pass

    // Only cache value here if caching decision isn't precomputed.
    // Otherwise caching will be done inside EnzymeLogic.cpp at
    // the end of the function jointly.
    if ((Mode != DerivativeMode::ForwardMode &&
         gutils->knownRecomputeHeuristic.count(&I) == 0 &&
         !gutils->unnecessaryIntermediates.count(&I) && can_modref &&
         !cache_reads_never) ||
        cache_reads_always) {
      // we can pre initialize all the knownRecomputeHeuristic values to false
      // (not needing) as we may assume that minCutCache already preserves
      // everything it requires.
      std::map<UsageKey, bool> Seen;
      for (auto pair : gutils->knownRecomputeHeuristic)
        Seen[UsageKey(pair.first, ValueType::Primal)] = false;
      bool primalNeededInReverse =
          is_value_needed_in_reverse<ValueType::Primal>(TR, gutils, &I, Mode,
                                                        Seen, oldUnreachable);
      if (primalNeededInReverse) {
        IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&I));
        inst = gutils->cacheForReverse(BuilderZ, newi,
                                       getIndex(&I, CacheType::Self));
        assert(inst->getType() == type);

        if (Mode == DerivativeMode::ReverseModeGradient) {
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
      Type *isfloat =
          type->isFPOrFPVectorTy() ? type->getScalarType() : nullptr;
      if (!isfloat && type->isIntOrIntVectorTy()) {
        auto LoadSize = DL.getTypeSizeInBits(type) / 8;
        ConcreteType vd = BaseType::Unknown;
        if (!OrigOffset)
          vd =
              TR.firstPointer(LoadSize, I.getOperand(0),
                              /*errifnotfound*/ false, /*pointerIntSame*/ true);
        if (vd.isKnown())
          isfloat = vd.isFloat();
        else
          isfloat =
              TR.intType(LoadSize, &I, /*errIfNotFound*/ !looseTypeAnalysis)
                  .isFloat();
      }

      if (isfloat) {

        switch (Mode) {
        case DerivativeMode::ForwardMode: {
          IRBuilder<> Builder2(&I);
          getForwardBuilder(Builder2);

          if (!gutils->isConstantValue(&I)) {
            Value *diff;
            if (!mask) {
              auto LI = Builder2.CreateLoad(
                  gutils->invertPointerM(I.getOperand(0), Builder2));
              if (alignment)
#if LLVM_VERSION_MAJOR >= 10
                LI->setAlignment(*alignment);
#else
                LI->setAlignment(alignment);
#endif
              diff = LI;
            } else {
              Type *tys[] = {I.getType(), I.getOperand(0)->getType()};
              auto F = Intrinsic::getDeclaration(gutils->oldFunc->getParent(),
                                                 Intrinsic::masked_load, tys);
#if LLVM_VERSION_MAJOR >= 10
              Value *alignv =
                  ConstantInt::get(Type::getInt32Ty(mask->getContext()),
                                   alignment ? alignment->value() : 0);
#else
              Value *alignv = ConstantInt::get(
                  Type::getInt32Ty(mask->getContext()), alignment);
#endif
              Value *args[] = {
                  gutils->invertPointerM(I.getOperand(0), Builder2), alignv,
                  mask, diffe(orig_maskInit, Builder2)};
              diff = Builder2.CreateCall(F, args);
            }
            setDiffe(&I, diff, Builder2);
          }
          break;
        }
        case DerivativeMode::ReverseModeGradient:
        case DerivativeMode::ReverseModeCombined: {
          IRBuilder<> Builder2(parent);
          getReverseBuilder(Builder2);

          auto prediff = diffe(&I, Builder2);
          setDiffe(&I, Constant::getNullValue(type), Builder2);

          if (!gutils->isConstantValue(I.getOperand(0))) {
            ((DiffeGradientUtils *)gutils)
                ->addToInvertedPtrDiffe(
                    I.getOperand(0), prediff, Builder2, alignment, OrigOffset,
                    mask ? lookup(mask, Builder2) : nullptr);
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
    if (!gutils->isConstantInstruction(&I) || !gutils->isConstantValue(&I)) {
      TR.dump();
      llvm::errs() << "oldFunc: " << *gutils->newFunc << "\n";
      llvm::errs() << "I: " << I << "\n";
    }
    assert(gutils->isConstantInstruction(&I));
    assert(gutils->isConstantValue(&I));

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
                        Value *mask = nullptr)
#else
  void visitCommonStore(llvm::Instruction &I, Value *orig_ptr, Value *orig_val,
                        unsigned align, bool isVolatile,
                        AtomicOrdering ordering, SyncScope::ID syncScope,
                        Value *mask = nullptr)
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
          gutils->setPtrDiffe(orig_ptr, Constant::getNullValue(valType),
                              Builder2, align, isVolatile, ordering, syncScope,
                              mask);
        } else {
          Value *diff;
          if (!mask) {
            auto dif1 = Builder2.CreateLoad(
                lookup(gutils->invertPointerM(orig_ptr, Builder2), Builder2),
                isVolatile);
            if (align)
#if LLVM_VERSION_MAJOR >= 10
              dif1->setAlignment(*align);
#else
              dif1->setAlignment(align);
#endif
            dif1->setOrdering(ordering);
            dif1->setSyncScopeID(syncScope);
            diff = dif1;
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
            Value *args[] = {
                lookup(gutils->invertPointerM(orig_ptr, Builder2), Builder2),
                alignv, mask, Constant::getNullValue(valType)};
            diff = Builder2.CreateCall(F, args);
          }
          gutils->setPtrDiffe(orig_ptr, Constant::getNullValue(valType),
                              Builder2, align, isVolatile, ordering, syncScope,
                              mask);
          addToDiffe(orig_val, diff, Builder2, FT, mask);
        }
        break;
      }
      case DerivativeMode::ForwardMode: {
        IRBuilder<> Builder2(&I);
        getForwardBuilder(Builder2);

        Value *diff = constantval ? Constant::getNullValue(valType)
                                  : diffe(orig_val, Builder2);
        gutils->setPtrDiffe(orig_ptr, diff, Builder2, align, isVolatile,
                            ordering, syncScope, mask);
        break;
      }
      }

      //! Storing an integer or pointer
    } else {
      //! Only need to update the forward function
      if (Mode == DerivativeMode::ReverseModePrimal ||
          Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ForwardMode) {
        IRBuilder<> storeBuilder(gutils->getNewFromOriginal(&I));

        Value *valueop = nullptr;

        if (constantval) {
          valueop = val;
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
  }

  void visitPHINode(llvm::PHINode &phi) {
    eraseIfUnused(phi);
    if (gutils->isConstantInstruction(&phi))
      return;

    switch (Mode) {
    case DerivativeMode::ReverseModePrimal:
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
      return;
    }
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      break;
    }
    }

    BasicBlock *oBB = phi.getParent();
    BasicBlock *nBB = gutils->getNewFromOriginal(oBB);

    IRBuilder<> diffeBuilder(nBB->getFirstNonPHI());
    diffeBuilder.setFastMathFlags(getFast());

    IRBuilder<> phiBuilder(&phi);
    getForwardBuilder(phiBuilder);

    auto newPhi = phiBuilder.CreatePHI(phi.getType(), 1, phi.getName() + "'");
    for (unsigned int i = 0; i < phi.getNumIncomingValues(); ++i) {
      auto val = phi.getIncomingValue(i);
      auto block = phi.getIncomingBlock(i);

      auto newBlock = gutils->getNewFromOriginal(block);
      IRBuilder<> pBuilder(newBlock->getTerminator());
      pBuilder.setFastMathFlags(getFast());

      if (gutils->isConstantValue(val)) {
        newPhi->addIncoming(Constant::getNullValue(val->getType()), newBlock);
      } else {
        auto diff = diffe(val, pBuilder);
        newPhi->addIncoming(diff, newBlock);
      }
    }

    setDiffe(&phi, newPhi, diffeBuilder);
  }

  void visitCastInst(llvm::CastInst &I) {
    eraseIfUnused(I);
    if (gutils->isConstantInstruction(&I))
      return;

    if (I.getType()->isPointerTy() ||
        I.getOpcode() == CastInst::CastOps::PtrToInt)
      return;

    switch (Mode) {
    case DerivativeMode::ReverseModePrimal: {
      return;
    }
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
      Value *orig_op0 = I.getOperand(0);
      Value *op0 = gutils->getNewFromOriginal(orig_op0);

      IRBuilder<> Builder2(I.getParent());
      getReverseBuilder(Builder2);

      if (!gutils->isConstantValue(orig_op0)) {
        Value *dif = diffe(&I, Builder2);

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
        if (I.getOpcode() == CastInst::CastOps::FPTrunc ||
            I.getOpcode() == CastInst::CastOps::FPExt) {
          addToDiffe(orig_op0, Builder2.CreateFPCast(dif, op0->getType()),
                     Builder2, FT);
        } else if (I.getOpcode() == CastInst::CastOps::BitCast) {
          addToDiffe(orig_op0, Builder2.CreateBitCast(dif, op0->getType()),
                     Builder2, FT);
        } else if (I.getOpcode() == CastInst::CastOps::Trunc) {
          // TODO CHECK THIS
          auto trunced = Builder2.CreateZExt(dif, op0->getType());
          addToDiffe(orig_op0, trunced, Builder2, FT);
        } else {
          TR.dump();
          llvm::errs() << *I.getParent()->getParent() << "\n"
                       << *I.getParent() << "\n";
          llvm::errs() << "cannot handle above cast " << I << "\n";
          report_fatal_error("unknown instruction");
        }
      }
      setDiffe(&I, Constant::getNullValue(I.getType()), Builder2);

      break;
    }
    case DerivativeMode::ForwardMode: {
      Value *orig_op0 = I.getOperand(0);

      IRBuilder<> Builder2(&I);
      getForwardBuilder(Builder2);

      if (!gutils->isConstantValue(orig_op0)) {
        Value *dif = diffe(orig_op0, Builder2);
        setDiffe(&I, Builder2.CreateCast(I.getOpcode(), dif, I.getType()),
                 Builder2);
      } else {
        setDiffe(&I, Constant::getNullValue(I.getType()), Builder2);
      }

      break;
    }
    }
  }

  void visitSelectInst(llvm::SelectInst &SI) {
    eraseIfUnused(SI);

    if (gutils->isConstantInstruction(&SI))
      return;
    if (SI.getType()->isPointerTy())
      return;

    switch (Mode) {
    case DerivativeMode::ReverseModePrimal:
      return;
    case DerivativeMode::ReverseModeCombined:
    case DerivativeMode::ReverseModeGradient: {
      createSelectInstAdjoint(SI);
      return;
    }
    case DerivativeMode::ForwardMode: {
      createSelectInstDual(SI);
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
                    Constant::getNullValue(op1->getType()));
                addToDiffe(SI.getOperand(2 - i), dif, Builder2, addingType);
              }
            }
            return;
          }
        }
      }

    if (!gutils->isConstantValue(orig_op1))
      dif1 = Builder2.CreateSelect(lookup(op0, Builder2), diffe(&SI, Builder2),
                                   Constant::getNullValue(op1->getType()),
                                   "diffe" + op1->getName());
    if (!gutils->isConstantValue(orig_op2))
      dif2 = Builder2.CreateSelect(
          lookup(op0, Builder2), Constant::getNullValue(op2->getType()),
          diffe(&SI, Builder2), "diffe" + op2->getName());

    setDiffe(&SI, Constant::getNullValue(SI.getType()), Builder2);
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

  void createSelectInstDual(llvm::SelectInst &SI) {
    Value *orig_cond = SI.getOperand(0);
    Value *cond = gutils->getNewFromOriginal(orig_cond);

    Value *op1 = SI.getOperand(1);
    Value *op2 = SI.getOperand(2);

    bool constantval0 = gutils->isConstantValue(op1);
    bool constantval1 = gutils->isConstantValue(op2);

    IRBuilder<> Builder2(&SI);
    getForwardBuilder(Builder2);

    Value *dif1;
    Value *dif2;

    if (!constantval0) {
      dif1 = diffe(op1, Builder2);
    } else {
      dif1 = Constant::getNullValue(SI.getType());
    }

    if (!constantval1) {
      dif2 = diffe(op2, Builder2);
    } else {
      dif2 = Constant::getNullValue(SI.getType());
    }

    Value *diffe = Builder2.CreateSelect(cond, dif1, dif2);
    setDiffe(&SI, diffe, Builder2);
  }

  void visitExtractElementInst(llvm::ExtractElementInst &EEI) {
    eraseIfUnused(EEI);
    if (gutils->isConstantInstruction(&EEI))
      return;
    if (Mode == DerivativeMode::ReverseModePrimal)
      return;

    switch (Mode) {
    case DerivativeMode::ForwardMode: {
      IRBuilder<> Builder2(&EEI);
      getForwardBuilder(Builder2);

      Value *orig_vec = EEI.getVectorOperand();

      auto vec_diffe = gutils->isConstantValue(orig_vec)
                           ? ConstantVector::getNullValue(orig_vec->getType())
                           : diffe(orig_vec, Builder2);
      auto diffe =
          Builder2.CreateExtractElement(vec_diffe, EEI.getIndexOperand());

      setDiffe(&EEI, diffe, Builder2);
      return;
    }
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
      IRBuilder<> Builder2(EEI.getParent());
      getReverseBuilder(Builder2);

      Value *orig_vec = EEI.getVectorOperand();

      if (!gutils->isConstantValue(orig_vec)) {
        SmallVector<Value *, 4> sv;
        sv.push_back(gutils->getNewFromOriginal(EEI.getIndexOperand()));

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
      setDiffe(&EEI, Constant::getNullValue(EEI.getType()), Builder2);
      return;
    }
    case DerivativeMode::ReverseModePrimal: {
      return;
    }
    }
  }

  void visitInsertElementInst(llvm::InsertElementInst &IEI) {
    eraseIfUnused(IEI);
    if (gutils->isConstantInstruction(&IEI))
      return;

    switch (Mode) {
    case DerivativeMode::ForwardMode: {
      IRBuilder<> Builder2(&IEI);
      getForwardBuilder(Builder2);

      Value *orig_vector = IEI.getOperand(0);
      Value *orig_inserted = IEI.getOperand(1);
      Value *orig_index = IEI.getOperand(2);

      Value *diff_inserted = gutils->isConstantValue(orig_inserted)
                                 ? ConstantFP::get(orig_inserted->getType(), 0)
                                 : diffe(orig_inserted, Builder2);

      Value *prediff =
          gutils->isConstantValue(orig_vector)
              ? ConstantVector::getNullValue(orig_vector->getType())
              : diffe(orig_vector, Builder2);

      auto dindex = Builder2.CreateInsertElement(
          prediff, diff_inserted, gutils->getNewFromOriginal(orig_index));
      setDiffe(&IEI, dindex, Builder2);

      return;
    }
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
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
        addToDiffe(orig_op0,
                   Builder2.CreateInsertElement(
                       dif1, Constant::getNullValue(op1->getType()),
                       lookup(op2, Builder2)),
                   Builder2, TR.addingType(size0, orig_op0));

      if (!gutils->isConstantValue(orig_op1))
        addToDiffe(orig_op1,
                   Builder2.CreateExtractElement(dif1, lookup(op2, Builder2)),
                   Builder2, TR.addingType(size1, orig_op1));

      setDiffe(&IEI, Constant::getNullValue(IEI.getType()), Builder2);
      return;
    }
    case DerivativeMode::ReverseModePrimal: {
      return;
    }
    }
  }

  void visitShuffleVectorInst(llvm::ShuffleVectorInst &SVI) {
    eraseIfUnused(SVI);
    if (gutils->isConstantInstruction(&SVI))
      return;

    switch (Mode) {
    case DerivativeMode::ForwardMode: {
      IRBuilder<> Builder2(&SVI);
      getForwardBuilder(Builder2);

      Value *orig_vector1 = SVI.getOperand(0);
      Value *orig_vector2 = SVI.getOperand(1);

      auto diffe_vector1 =
          gutils->isConstantValue(orig_vector1)
              ? ConstantVector::getNullValue(orig_vector1->getType())
              : diffe(orig_vector1, Builder2);
      auto diffe_vector2 =
          gutils->isConstantValue(orig_vector2)
              ? ConstantVector::getNullValue(orig_vector2->getType())
              : diffe(orig_vector2, Builder2);

#if LLVM_VERSION_MAJOR >= 11
      auto diffe = Builder2.CreateShuffleVector(diffe_vector1, diffe_vector2,
                                                SVI.getShuffleMaskForBitcode());
#else
      auto diffe = Builder2.CreateShuffleVector(diffe_vector1, diffe_vector2,
                                                SVI.getOperand(2));
#endif

      setDiffe(&SVI, diffe, Builder2);
      return;
    }
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
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
        SmallVector<Value *, 4> sv;
        sv.push_back(
            ConstantInt::get(Type::getInt32Ty(SVI.getContext()), opidx));
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
      setDiffe(&SVI, Constant::getNullValue(SVI.getType()), Builder2);
      return;
    }
    case DerivativeMode::ReverseModePrimal: {
      return;
    }
    }
  }

  void visitExtractValueInst(llvm::ExtractValueInst &EVI) {
    eraseIfUnused(EVI);
    if (gutils->isConstantInstruction(&EVI))
      return;
    if (EVI.getType()->isPointerTy())
      return;

    switch (Mode) {
    case DerivativeMode::ForwardMode: {
      IRBuilder<> Builder2(&EVI);
      getForwardBuilder(Builder2);

      Value *orig_aggregate = EVI.getAggregateOperand();

      Value *diffe_aggregate =
          gutils->isConstantValue(orig_aggregate)
              ? ConstantAggregate::getNullValue(orig_aggregate->getType())
              : diffe(orig_aggregate, Builder2);
      Value *diffe =
          Builder2.CreateExtractValue(diffe_aggregate, EVI.getIndices());

      setDiffe(&EVI, diffe, Builder2);
      return;
    }
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
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

      setDiffe(&EVI, Constant::getNullValue(EVI.getType()), Builder2);
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
    case DerivativeMode::ForwardMode: {
      IRBuilder<> Builder2(&IVI);
      getForwardBuilder(Builder2);

      Value *orig_inserted = IVI.getInsertedValueOperand();
      Value *orig_agg = IVI.getAggregateOperand();

      Value *diff_inserted = gutils->isConstantValue(orig_inserted)
                                 ? ConstantFP::get(orig_inserted->getType(), 0)
                                 : diffe(orig_inserted, Builder2);

      Value *prediff =
          gutils->isConstantValue(orig_agg)
              ? ConstantAggregate::getNullValue(orig_agg->getType())
              : diffe(orig_agg, Builder2);
      auto dindex =
          Builder2.CreateInsertValue(prediff, diff_inserted, IVI.getIndices());
      setDiffe(&IVI, dindex, Builder2);

      return;
    }
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
          auto prediff = diffe(&IVI, Builder2);
          auto dindex = Builder2.CreateExtractValue(prediff, IVI.getIndices());
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
        auto prediff = diffe(&IVI, Builder2);
        auto dindex = Builder2.CreateInsertValue(
            prediff, Constant::getNullValue(orig_inserted->getType()),
            IVI.getIndices());
        addToDiffe(orig_agg, dindex, Builder2, TR.addingType(size1, orig_agg));
      }

      setDiffe(&IVI, Constant::getNullValue(IVI.getType()), Builder2);
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

  bool shouldFree() {
    assert(Mode == DerivativeMode::ReverseModeCombined ||
           Mode == DerivativeMode::ReverseModeGradient ||
           Mode == DerivativeMode::ForwardModeSplit);
    return ((DiffeGradientUtils *)gutils)->FreeMemory;
  }

  std::vector<SelectInst *> addToDiffe(Value *val, Value *dif,
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
    if (gutils->isConstantInstruction(&BO))
      return;

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
      if (!constantval0)
        dif0 = Builder2.CreateFMul(
            idiff, lookup(gutils->getNewFromOriginal(orig_op1), Builder2),
            "m0diffe" + orig_op0->getName());
      if (!constantval1)
        dif1 = Builder2.CreateFMul(
            idiff, lookup(gutils->getNewFromOriginal(orig_op0), Builder2),
            "m1diffe" + orig_op1->getName());
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
      if (!constantval1)
        dif1 = Builder2.CreateFNeg(idiff);
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
                dif1 = Builder2.CreateFDiv(
                    Builder2.CreateFNeg(Builder2.CreateFMul(idiff, lop0)),
                    lop1);
              } else {
                auto product = gutils->getOrInsertTotalMultiplicativeProduct(
                    gutils->getNewFromOriginal(orig_op1), lc);
                IRBuilder<> EB(*lc.exitBlocks.begin());
                getReverseBuilder(EB, /*original=*/false);
                Value *s = lookup(gutils->getNewFromOriginal(Pstart), Builder2);
                Value *lop0 = lookup(product, EB);
                Value *lop1 =
                    lookup(gutils->getNewFromOriginal(orig_op1), Builder2);
                dif1 = Builder2.CreateFDiv(
                    Builder2.CreateFNeg(Builder2.CreateFMul(
                        s, Builder2.CreateFDiv(idiff, lop0))),
                    lop1);
              }
              addToDiffe(orig_op1, dif1, Builder2, addingType);
            }
            return;
          }
        }
      }
      if (!constantval0)
        dif0 = Builder2.CreateFDiv(
            idiff, lookup(gutils->getNewFromOriginal(orig_op1), Builder2),
            "d0diffe" + orig_op0->getName());
      if (!constantval1) {
        Value *lop1 = lookup(gutils->getNewFromOriginal(orig_op1), Builder2);
        Value *lastdiv = lookup(gutils->getNewFromOriginal(&BO), Builder2);
        dif1 = Builder2.CreateFNeg(
            Builder2.CreateFMul(lastdiv, Builder2.CreateFDiv(idiff, lop1)));
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
              dif0 = Builder2.CreateShl(idiff, ci);
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
              setDiffe(&BO, Constant::getNullValue(BO.getType()), Builder2);
              // Derivative is zero, no update
              return;
            }
            if (eFT->isDoubleTy() && CI->getValue() == -134217728) {
              setDiffe(&BO, Constant::getNullValue(BO.getType()), Builder2);
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
              setDiffe(&BO, Constant::getNullValue(BO.getType()), Builder2);
              auto neg = Builder2.CreateFNeg(Builder2.CreateBitCast(idiff, FT));
              auto bc = Builder2.CreateBitCast(neg, BO.getType());
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
              setDiffe(&BO, Constant::getNullValue(BO.getType()), Builder2);
              Value *V = UndefValue::get(CV->getType());
              for (size_t i = 0, end = CV->getNumOperands(); i < end; ++i) {
                auto CI = dyn_cast<ConstantInt>(CV->getOperand(i))->getValue();
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
              setDiffe(&BO, Constant::getNullValue(BO.getType()), Builder2);
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
              setDiffe(&BO, Constant::getNullValue(BO.getType()), Builder2);
              auto arg = lookup(
                  gutils->getNewFromOriginal(BO.getOperand(1 - i)), Builder2);
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
              addToDiffe(BO.getOperand(1 - i), prev, Builder2, FT);
              return;
            }
          }
        }
      goto def;
    }
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
      llvm::errs() << *gutils->oldFunc << "\n";
      for (auto &arg : gutils->oldFunc->args()) {
        llvm::errs() << " constantarg[" << arg
                     << "] = " << gutils->isConstantValue(&arg)
                     << " type: " << TR.query(&arg).str() << " - vals: {";
        for (auto v : TR.knownIntegralValues(&arg))
          llvm::errs() << v << ",";
        llvm::errs() << "}\n";
      }
      for (auto &BB : *gutils->oldFunc)
        for (auto &I : BB) {
          llvm::errs() << " constantinst[" << I
                       << "] = " << gutils->isConstantInstruction(&I)
                       << " val:" << gutils->isConstantValue(&I)
                       << " type: " << TR.query(&I).str() << "\n";
        }
      llvm::errs() << "cannot handle unknown binary operator: " << BO << "\n";
      report_fatal_error("unknown binary operator");
    }

  done:;
    if (dif0 || dif1)
      setDiffe(&BO, Constant::getNullValue(BO.getType()), Builder2);
    if (dif0)
      addToDiffe(orig_op0, dif0, Builder2, addingType);
    if (dif1)
      addToDiffe(orig_op1, dif1, Builder2, addingType);
  }

  void createBinaryOperatorDual(llvm::BinaryOperator &BO) {
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
        Value *idiff0 =
            Builder2.CreateFMul(dif[0], gutils->getNewFromOriginal(orig_op1));
        Value *idiff1 =
            Builder2.CreateFMul(dif[1], gutils->getNewFromOriginal(orig_op0));
        Value *diff = Builder2.CreateFAdd(idiff0, idiff1);
        setDiffe(&BO, diff, Builder2);
      } else if (!constantval0) {
        Value *idiff0 =
            Builder2.CreateFMul(dif[0], gutils->getNewFromOriginal(orig_op1));
        setDiffe(&BO, idiff0, Builder2);
      } else if (!constantval1) {
        Value *idiff1 =
            Builder2.CreateFMul(dif[1], gutils->getNewFromOriginal(orig_op0));
        setDiffe(&BO, idiff1, Builder2);
      }
      break;
    }
    case Instruction::FAdd: {
      if (!constantval0 && !constantval1) {
        Value *diff = Builder2.CreateFAdd(dif[0], dif[1]);
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
        Value *diff = Builder2.CreateFAdd(dif[0], Builder2.CreateFNeg(dif[1]));
        setDiffe(&BO, diff, Builder2);
      } else if (!constantval0) {
        setDiffe(&BO, dif[0], Builder2);
      } else if (!constantval1) {
        setDiffe(&BO, Builder2.CreateFNeg(dif[1]), Builder2);
      }
      break;
    }
    case Instruction::FDiv: {
      Value *idiff3 = nullptr;
      if (!constantval0 && !constantval1) {
        Value *idiff1 =
            Builder2.CreateFMul(dif[0], gutils->getNewFromOriginal(orig_op1));
        Value *idiff2 =
            Builder2.CreateFMul(gutils->getNewFromOriginal(orig_op0), dif[1]);
        idiff3 = Builder2.CreateFSub(idiff1, idiff2);
      } else if (!constantval0) {
        Value *idiff1 =
            Builder2.CreateFMul(dif[0], gutils->getNewFromOriginal(orig_op1));
        idiff3 = idiff1;
      } else if (!constantval1) {
        Value *idiff2 =
            Builder2.CreateFMul(gutils->getNewFromOriginal(orig_op0), dif[1]);
        idiff3 = Builder2.CreateFNeg(idiff2);
      }

      Value *idiff4 = Builder2.CreateFMul(gutils->getNewFromOriginal(orig_op1),
                                          gutils->getNewFromOriginal(orig_op1));
      Value *idiff5 = Builder2.CreateFDiv(idiff3, idiff4);
      setDiffe(&BO, idiff5, Builder2);

      break;
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
              setDiffe(&BO, Constant::getNullValue(BO.getType()), Builder2);
              // Derivative is zero, no update
              return;
            }
            if (eFT->isDoubleTy() && CI->getValue() == -134217728) {
              setDiffe(&BO, Constant::getNullValue(BO.getType()), Builder2);
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
              assert(dif[1 - i]);
              auto neg =
                  Builder2.CreateFNeg(Builder2.CreateBitCast(dif[1 - i], FT));
              auto bc = Builder2.CreateBitCast(neg, BO.getType());
              setDiffe(&BO, bc, Builder2);
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
              Value *V = UndefValue::get(CV->getType());
              for (size_t j = 0, end = CV->getNumOperands(); j < end; ++j) {
                auto CI = dyn_cast<ConstantInt>(CV->getOperand(j))->getValue();
                if (CI.isNullValue())
                  V = Builder2.CreateInsertElement(
                      V, Builder2.CreateExtractElement(dif[1 - i], j), j);
                if (CI.isMinSignedValue())
                  V = Builder2.CreateInsertElement(
                      V,
                      Builder2.CreateBitCast(
                          Builder2.CreateFNeg(Builder2.CreateBitCast(
                              Builder2.CreateExtractElement(dif[1 - i], j),
                              eFT)),
                          CV->getOperand(j)->getType()),
                      j);
              }
              setDiffe(&BO, V, Builder2);
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
              Value *V = UndefValue::get(CV->getType());
              for (size_t j = 0, end = CV->getNumElements(); j < end; ++j) {
                auto CI = CV->getElementAsAPInt(j);
                if (CI.isNullValue())
                  V = Builder2.CreateInsertElement(
                      V, Builder2.CreateExtractElement(dif[1 - i], j), j);
                if (CI.isMinSignedValue())
                  V = Builder2.CreateInsertElement(
                      V,
                      Builder2.CreateBitCast(
                          Builder2.CreateFNeg(Builder2.CreateBitCast(
                              Builder2.CreateExtractElement(dif[1 - i], j),
                              eFT)),
                          CV->getElementType()),
                      j);
              }
              setDiffe(&BO, V, Builder2);
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
                  Builder2.CreateFMul(Builder2.CreateBitCast(dif[1 - i], FT),
                                      Builder2.CreateBitCast(prev, FT)),
                  prev->getType());
              setDiffe(&BO, prev, Builder2);
              return;
            }
          }
        }
      goto def;
    }
    default:
    def:;
      llvm::errs() << *gutils->oldFunc << "\n";
      for (auto &arg : gutils->oldFunc->args()) {
        llvm::errs() << " constantarg[" << arg
                     << "] = " << gutils->isConstantValue(&arg)
                     << " type: " << TR.query(&arg).str() << " - vals: {";
        for (auto v : TR.knownIntegralValues(&arg))
          llvm::errs() << v << ",";
        llvm::errs() << "}\n";
      }
      for (auto &BB : *gutils->oldFunc)
        for (auto &I : BB) {
          llvm::errs() << " constantinst[" << I
                       << "] = " << gutils->isConstantInstruction(&I)
                       << " val:" << gutils->isConstantValue(&I)
                       << " type: " << TR.query(&I).str() << "\n";
        }
      llvm::errs() << "cannot handle unknown binary operator: " << BO << "\n";
      report_fatal_error("unknown binary operator");
      break;
    }
  }

  void visitMemSetInst(llvm::MemSetInst &MS) {
    // Don't duplicate set in reverse pass
    if (Mode == DerivativeMode::ReverseModeGradient) {
      erased.insert(&MS);
      gutils->erase(gutils->getNewFromOriginal(&MS));
    }

    if (gutils->isConstantInstruction(&MS))
      return;

    Value *orig_op0 = MS.getOperand(0);
    Value *orig_op1 = MS.getOperand(1);
    Value *op1 = gutils->getNewFromOriginal(orig_op1);
    Value *op2 = gutils->getNewFromOriginal(MS.getOperand(2));
    Value *op3 = gutils->getNewFromOriginal(MS.getOperand(3));

    // TODO this should 1) assert that the value being meset is constant
    //                 2) duplicate the memset for the inverted pointer

    if (!gutils->isConstantValue(orig_op1)) {
      llvm::errs() << "couldn't handle non constant inst in memset to "
                      "propagate differential to\n"
                   << MS;
      report_fatal_error("non constant in memset");
    }

    if (Mode == DerivativeMode::ReverseModePrimal ||
        Mode == DerivativeMode::ReverseModeCombined) {
      IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&MS));

      SmallVector<Value *, 4> args;
      if (!gutils->isConstantValue(orig_op0)) {
        args.push_back(gutils->invertPointerM(orig_op0, BuilderZ));
      } else {
        // If constant destination then no operation needs doing
        return;
        // args.push_back(gutils->lookupM(MS.getOperand(0), BuilderZ));
      }

      args.push_back(gutils->lookupM(op1, BuilderZ));
      args.push_back(gutils->lookupM(op2, BuilderZ));
      args.push_back(gutils->lookupM(op3, BuilderZ));

      Type *tys[] = {args[0]->getType(), args[2]->getType()};
      auto cal = BuilderZ.CreateCall(
          Intrinsic::getDeclaration(MS.getParent()->getParent()->getParent(),
                                    Intrinsic::memset, tys),
          args);
      cal->setAttributes(MS.getAttributes());
      cal->setCallingConv(MS.getCallingConv());
      cal->setTailCallKind(MS.getTailCallKind());
    }

    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined) {
      // TODO consider what reverse pass memset should be
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
      if (ddst->getType()->isIntegerTy())
        ddst = Builder2.CreateIntToPtr(ddst,
                                       Type::getInt8PtrTy(ddst->getContext()));
      auto dsrc = gutils->invertPointerM(orig_src, Builder2);
      if (dsrc->getType()->isIntegerTy())
        dsrc = Builder2.CreateIntToPtr(dsrc,
                                       Type::getInt8PtrTy(dsrc->getContext()));

      auto call =
          Builder2.CreateMemCpy(ddst, dstAlign, dsrc, srcAlign, new_size);
      call->setAttributes(MTI.getAttributes());
      call->setTailCallKind(MTI.getTailCallKind());

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
              if (PT->getElementType()->isFPOrFPVectorTy()) {
                vd = TypeTree(
                         ConcreteType(PT->getElementType()->getScalarType()))
                         .Only(0);
                goto known;
              }
              if (PT->getElementType()->isIntOrIntVectorTy()) {
                vd = TypeTree(BaseType::Integer).Only(0);
                goto known;
              }
              if (PT->getElementType()->isPointerTy()) {
                vd = TypeTree(BaseType::Pointer).Only(0);
                goto known;
              }
              auto ET = PT->getElementType();
              while (auto ST = dyn_cast<StructType>(ET)) {
                if (!ST->getNumElements())
                  break;
                ET = ST->getElementType(0);
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
      SubTransferHelper(gutils, Mode, dt.isFloat(), ID, subdstalign,
                        subsrcalign, /*offset*/ start,
                        gutils->isConstantValue(orig_dst), shadow_dst,
                        gutils->isConstantValue(orig_src), shadow_src,
                        /*length*/ length, /*volatile*/ isVolatile, &MTI);

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
      case Intrinsic::fma:
        return;
      default:
        if (gutils->isConstantInstruction(&I))
          return;
        llvm::errs() << *gutils->oldFunc << "\n";
        llvm::errs() << *gutils->newFunc << "\n";
        llvm::errs() << "cannot handle (augmented) unknown intrinsic\n" << I;
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
        setDiffe(&I, Constant::getNullValue(I.getType()), Builder2);
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
          auto vec = Builder2.CreateShuffleVector(
              Builder2.CreateInsertElement(und, vdiff, (uint64_t)0), und, mask);
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
          Type *tys[] = {orig_ops[0]->getType()};
          Function *SqrtF;
          if (ID == Intrinsic::sqrt)
            SqrtF = Intrinsic::getDeclaration(M, ID, tys);
          else
            SqrtF = Intrinsic::getDeclaration(M, ID);

          auto cal = cast<CallInst>(Builder2.CreateCall(SqrtF, args));
          cal->setCallingConv(SqrtF->getCallingConv());
          cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));

          Value *dif0 = Builder2.CreateBinOp(
              Instruction::FDiv,
              Builder2.CreateFMul(ConstantFP::get(I.getType(), 0.5), vdiff),
              cal);

          Value *cmp = Builder2.CreateFCmpOEQ(
              args[0], ConstantFP::get(orig_ops[0]->getType(), 0));
          dif0 = Builder2.CreateSelect(
              cmp, ConstantFP::get(orig_ops[0]->getType(), 0), dif0);

          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        return;
      }

      case Intrinsic::fabs: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          Value *cmp = Builder2.CreateFCmpOLT(
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2),
              ConstantFP::get(orig_ops[0]->getType(), 0));
          Value *dif0 = Builder2.CreateFMul(
              Builder2.CreateSelect(cmp,
                                    ConstantFP::get(orig_ops[0]->getType(), -1),
                                    ConstantFP::get(orig_ops[0]->getType(), 1)),
              vdiff);
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
              cmp, ConstantFP::get(orig_ops[0]->getType(), 0), vdiff);
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        if (vdiff && !gutils->isConstantValue(orig_ops[1])) {
          Value *cmp = Builder2.CreateFCmpOLT(
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2),
              lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2));
          Value *dif1 = Builder2.CreateSelect(
              cmp, vdiff, ConstantFP::get(orig_ops[0]->getType(), 0));
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
              cmp, vdiff, ConstantFP::get(orig_ops[0]->getType(), 0));
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        if (vdiff && !gutils->isConstantValue(orig_ops[1])) {
          Value *cmp = Builder2.CreateFCmpOLT(
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2),
              lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2));
          Value *dif1 = Builder2.CreateSelect(
              cmp, ConstantFP::get(orig_ops[0]->getType(), 0), vdiff);
          addToDiffe(orig_ops[1], dif1, Builder2, I.getType());
        }
        return;
      }

      case Intrinsic::fma: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          Value *dif0 = Builder2.CreateFMul(
              vdiff, lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2));
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType()->getScalarType());
        }
        if (vdiff && !gutils->isConstantValue(orig_ops[1])) {
          Value *dif1 = Builder2.CreateFMul(
              vdiff, lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2));
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
          Value *dif0 = Builder2.CreateFDiv(
              vdiff, lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2));
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        return;
      }

      case Intrinsic::log2: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          Value *dif0 = Builder2.CreateFDiv(
              vdiff,
              Builder2.CreateFMul(
                  ConstantFP::get(I.getType(), 0.6931471805599453),
                  lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2)));
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        return;
      }
      case Intrinsic::log10: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          Value *dif0 = Builder2.CreateFDiv(
              vdiff,
              Builder2.CreateFMul(
                  ConstantFP::get(I.getType(), 2.302585092994046),
                  lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2)));
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

          Value *dif0 = Builder2.CreateFMul(vdiff, cal);
          if (ID != Intrinsic::exp) {
            dif0 = Builder2.CreateFMul(
                dif0, ConstantFP::get(I.getType(), 0.6931471805599453));
          }
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
          Value *dif0 =
              Builder2.CreateFMul(Builder2.CreateFMul(xsign, ysign), vdiff);
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        return;
      }
      case Intrinsic::powi: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          Value *op0 = gutils->getNewFromOriginal(orig_ops[0]);
          Value *op1 = gutils->getNewFromOriginal(orig_ops[1]);
          SmallVector<Value *, 2> args = {
              lookup(op0, Builder2),
              Builder2.CreateSub(lookup(op1, Builder2),
                                 ConstantInt::get(op1->getType(), 1))};
          Type *tys[] = {
            orig_ops[0]->getType()
#if LLVM_VERSION_MAJOR >= 13
                ,
            orig_ops[1]->getType()
#endif
          };
          Function *PowF = Intrinsic::getDeclaration(M, Intrinsic::powi, tys);
          auto cal = cast<CallInst>(Builder2.CreateCall(PowF, args));
          cal->setCallingConv(PowF->getCallingConv());
          cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));
          Value *dif0 = Builder2.CreateFMul(
              Builder2.CreateFMul(vdiff, cal),
              Builder2.CreateSIToFP(lookup(op1, Builder2),
                                    op0->getType()->getScalarType()));
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        return;
      }
      case Intrinsic::pow: {
        Type *tys[] = {orig_ops[0]->getType()};
        Function *PowF = Intrinsic::getDeclaration(M, Intrinsic::pow, tys);
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
          auto cal = cast<CallInst>(Builder2.CreateCall(PowF, args));
          cal->setCallingConv(PowF->getCallingConv());
          cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));

          Value *dif0 = Builder2.CreateFMul(Builder2.CreateFMul(vdiff, cal),
                                            lookup(op1, Builder2));
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }

        if (vdiff && !gutils->isConstantValue(orig_ops[1])) {

          CallInst *cal;
          {
            SmallVector<Value *, 2> args = {
                lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2),
                lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2)};

            cal = cast<CallInst>(Builder2.CreateCall(PowF, args));
            cal->setCallingConv(PowF->getCallingConv());
            cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));
          }

          Value *args[] = {
              lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2)};

          Value *dif1 = Builder2.CreateFMul(
              Builder2.CreateFMul(vdiff, cal),
              Builder2.CreateCall(
                  Intrinsic::getDeclaration(M, Intrinsic::log, tys), args));
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
          Value *dif0 = Builder2.CreateFMul(vdiff, cal);
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
          Value *dif0 = Builder2.CreateFMul(vdiff, Builder2.CreateFNeg(cal));
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        return;
      }
      default:
        if (gutils->isConstantInstruction(&I))
          return;
        llvm::errs() << *gutils->oldFunc << "\n";
        llvm::errs() << *gutils->newFunc << "\n";
        if (Intrinsic::isOverloaded(ID))
#if LLVM_VERSION_MAJOR >= 13
          llvm::errs() << "cannot handle (reverse) unknown intrinsic\n"
                       << Intrinsic::getName(ID, ArrayRef<Type *>(), nullptr,
                                             nullptr)
                       << "\n"
                       << I;
#else
          llvm::errs() << "cannot handle (reverse) unknown intrinsic\n"
                       << Intrinsic::getName(ID, ArrayRef<Type *>()) << "\n"
                       << I;
#endif
        else
          llvm::errs() << "cannot handle (reverse) unknown intrinsic\n"
                       << Intrinsic::getName(ID) << "\n"
                       << I;
        report_fatal_error("(reverse) unknown intrinsic");
      }
      return;
    }
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

        auto accdif = gutils->isConstantValue(orig_ops[0])
                          ? ConstantFP::get(orig_ops[0]->getType(), 0)
                          : diffe(orig_ops[0], Builder2);

        auto vecdif = gutils->isConstantValue(orig_ops[1])
                          ? ConstantVector::getNullValue(orig_ops[1]->getType())
                          : diffe(orig_ops[1], Builder2);

#if LLVM_VERSION_MAJOR < 12
        auto vfra = Intrinsic::getDeclaration(
            M, ID, {orig_ops[0]->getType(), orig_ops[1]->getType()});
#else
        auto vfra = Intrinsic::getDeclaration(M, ID, {orig_ops[1]->getType()});
#endif
        auto cal = Builder2.CreateCall(vfra, {accdif, vecdif});
        cal->setCallingConv(vfra->getCallingConv());
        cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));

        setDiffe(&I, cal, Builder2);

        return;
      }
#endif
      case Intrinsic::nvvm_sqrt_rn_d:
      case Intrinsic::sqrt: {
        if (gutils->isConstantInstruction(&I))
          return;
        Value *op = diffe(orig_ops[0], Builder2);

        Value *args[1] = {gutils->getNewFromOriginal(orig_ops[0])};
        Type *tys[] = {orig_ops[0]->getType()};
        Function *SqrtF;
        if (ID == Intrinsic::sqrt)
          SqrtF = Intrinsic::getDeclaration(M, ID, tys);
        else
          SqrtF = Intrinsic::getDeclaration(M, ID);

        auto cal = cast<CallInst>(Builder2.CreateCall(SqrtF, args));
        cal->setCallingConv(SqrtF->getCallingConv());
        cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));

        Value *dif0 = Builder2.CreateFDiv(
            Builder2.CreateFMul(ConstantFP::get(I.getType(), 0.5), op), cal);

        Value *cmp = Builder2.CreateFCmpOEQ(
            args[0], ConstantFP::get(orig_ops[0]->getType(), 0));
        dif0 = Builder2.CreateSelect(
            cmp, ConstantFP::get(orig_ops[0]->getType(), 0), dif0);

        setDiffe(&I, dif0, Builder2);
        return;
      }

      case Intrinsic::fabs: {
        if (gutils->isConstantInstruction(&I))
          return;
        Value *op = diffe(orig_ops[0], Builder2);

        Value *cmp =
            Builder2.CreateFCmpOLT(gutils->getNewFromOriginal(orig_ops[0]),
                                   ConstantFP::get(orig_ops[0]->getType(), 0));
        Value *dif0 = Builder2.CreateFMul(
            Builder2.CreateSelect(cmp,
                                  ConstantFP::get(orig_ops[0]->getType(), -1),
                                  ConstantFP::get(orig_ops[0]->getType(), 1)),
            op);
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

        Value *diffe0 = gutils->isConstantValue(orig_ops[0])
                            ? ConstantFP::get(orig_ops[0]->getType(), 0)
                            : diffe(orig_ops[0], Builder2);
        Value *diffe1 = gutils->isConstantValue(orig_ops[1])
                            ? ConstantFP::get(orig_ops[1]->getType(), 0)
                            : diffe(orig_ops[1], Builder2);

        Value *dif = Builder2.CreateSelect(cmp, diffe0, diffe1);
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

        Value *diffe0 = gutils->isConstantValue(orig_ops[0])
                            ? ConstantFP::get(orig_ops[0]->getType(), 0)
                            : diffe(orig_ops[0], Builder2);
        Value *diffe1 = gutils->isConstantValue(orig_ops[1])
                            ? ConstantFP::get(orig_ops[1]->getType(), 0)
                            : diffe(orig_ops[1], Builder2);

        Value *dif = Builder2.CreateSelect(cmp, diffe0, diffe1);
        setDiffe(&I, dif, Builder2);

        return;
      }

      case Intrinsic::fma: {
        if (gutils->isConstantInstruction(&I))
          return;
        Value *op1 = gutils->getNewFromOriginal(orig_ops[1]);
        Value *op2 = gutils->getNewFromOriginal(orig_ops[2]);

        Value *dif0 = gutils->isConstantValue(orig_ops[0])
                          ? ConstantFP::get(orig_ops[0]->getType(), 0)
                          : diffe(orig_ops[0], Builder2);
        Value *dif1 = gutils->isConstantValue(orig_ops[1])
                          ? ConstantFP::get(orig_ops[1]->getType(), 0)
                          : diffe(orig_ops[1], Builder2);
        Value *dif2 = gutils->isConstantValue(orig_ops[2])
                          ? ConstantFP::get(orig_ops[2]->getType(), 0)
                          : diffe(orig_ops[2], Builder2);

        Value *dif = Builder2.CreateFAdd(Builder2.CreateFMul(op1, dif2),
                                         Builder2.CreateFMul(dif1, op2));

        dif = Builder2.CreateFAdd(dif, dif0);
        setDiffe(&I, dif, Builder2);

        return;
      }

      case Intrinsic::log: {
        if (gutils->isConstantInstruction(&I))
          return;
        Value *op = diffe(orig_ops[0], Builder2);

        Value *dif0 =
            Builder2.CreateFDiv(op, gutils->getNewFromOriginal(orig_ops[0]));
        setDiffe(&I, dif0, Builder2);
        return;
      }

      case Intrinsic::log2: {
        if (gutils->isConstantInstruction(&I))
          return;
        Value *op = diffe(orig_ops[0], Builder2);

        Value *dif0 = Builder2.CreateFDiv(
            op, Builder2.CreateFMul(
                    ConstantFP::get(I.getType(), 0.6931471805599453),
                    gutils->getNewFromOriginal(orig_ops[0])));
        setDiffe(&I, dif0, Builder2);
        return;
      }
      case Intrinsic::log10: {
        if (gutils->isConstantInstruction(&I))
          return;
        Value *op = diffe(orig_ops[0], Builder2);

        Value *dif0 = Builder2.CreateFDiv(
            op,
            Builder2.CreateFMul(ConstantFP::get(I.getType(), 2.302585092994046),
                                gutils->getNewFromOriginal(orig_ops[0])));
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
        auto cal = cast<CallInst>(Builder2.CreateCall(ExpF, args));
        cal->setCallingConv(ExpF->getCallingConv());
        cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));

        Value *dif0 = Builder2.CreateFMul(op, cal);
        if (ID != Intrinsic::exp) {
          dif0 = Builder2.CreateFMul(
              dif0, ConstantFP::get(I.getType(), 0.6931471805599453));
        }
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

        Value *dif0 = Builder2.CreateFMul(Builder2.CreateFMul(xsign, ysign),
                                          diffe(orig_ops[0], Builder2));
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
          Type *tys[] = {
            orig_ops[0]->getType()
#if LLVM_VERSION_MAJOR >= 13
                ,
            orig_ops[1]->getType()
#endif
          };
          Function *PowF = Intrinsic::getDeclaration(M, Intrinsic::powi, tys);
          auto cal = cast<CallInst>(Builder2.CreateCall(PowF, args));
          cal->setCallingConv(PowF->getCallingConv());
          cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));

          Value *dif0 = Builder2.CreateFMul(
              Builder2.CreateFMul(diffe(orig_ops[0], Builder2), cal),
              Builder2.CreateSIToFP(op1, op0->getType()->getScalarType()));

          setDiffe(&I, dif0, Builder2);
        }
        return;
      }
      case Intrinsic::pow: {
        if (gutils->isConstantInstruction(&I))
          return;
        Type *tys[] = {orig_ops[0]->getType()};
        Function *PowF = Intrinsic::getDeclaration(M, Intrinsic::pow, tys);

        Value *op0 = gutils->getNewFromOriginal(orig_ops[0]);
        Value *op1 = gutils->getNewFromOriginal(orig_ops[1]);

        Value *res = ConstantFP::get(orig_ops[0]->getType(), 0);

        if (!gutils->isConstantValue(orig_ops[0])) {
          Value *args[2] = {
              op0, Builder2.CreateFSub(op1, ConstantFP::get(I.getType(), 1.0))};
          CallInst *powcall1 = cast<CallInst>(Builder2.CreateCall(PowF, args));
          powcall1->setCallingConv(PowF->getCallingConv());
          powcall1->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));

          Value *dfdx = Builder2.CreateFMul(Builder2.CreateFMul(op1, powcall1),
                                            diffe(orig_ops[0], Builder2));
          res = Builder2.CreateFAdd(res, dfdx);
        }
        if (!gutils->isConstantValue(orig_ops[1])) {
          CallInst *powcall =
              cast<CallInst>(Builder2.CreateCall(PowF, {op0, op1}));
          powcall->setCallingConv(PowF->getCallingConv());
          powcall->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));

          CallInst *logcall = Builder2.CreateCall(
              Intrinsic::getDeclaration(M, Intrinsic::log, tys), {op0});

          Value *dfdy =
              Builder2.CreateFMul(Builder2.CreateFMul(powcall, logcall),
                                  diffe(orig_ops[1], Builder2));
          res = Builder2.CreateFAdd(res, dfdy);
        }
        setDiffe(&I, res, Builder2);

        return;
      }
      case Intrinsic::sin: {
        if (gutils->isConstantInstruction(&I))
          return;
        Value *op = diffe(orig_ops[0], Builder2);

        Value *args[] = {gutils->getNewFromOriginal(orig_ops[0])};
        Type *tys[] = {orig_ops[0]->getType()};
        CallInst *cal = cast<CallInst>(Builder2.CreateCall(
            Intrinsic::getDeclaration(M, Intrinsic::cos, tys), args));
        Value *dif0 = Builder2.CreateFMul(op, cal);
        setDiffe(&I, dif0, Builder2);
        return;
      }
      case Intrinsic::cos: {
        if (gutils->isConstantInstruction(&I))
          return;
        Value *op = diffe(orig_ops[0], Builder2);

        Value *args[] = {gutils->getNewFromOriginal(orig_ops[0])};
        Type *tys[] = {orig_ops[0]->getType()};
        CallInst *cal = cast<CallInst>(Builder2.CreateCall(
            Intrinsic::getDeclaration(M, Intrinsic::sin, tys), args));
        Value *dif0 = Builder2.CreateFMul(op, Builder2.CreateFNeg(cal));
        setDiffe(&I, dif0, Builder2);
        return;
      }
      default:
        if (gutils->isConstantInstruction(&I))
          return;
        llvm::errs() << *gutils->oldFunc << "\n";
        llvm::errs() << *gutils->newFunc << "\n";
        if (Intrinsic::isOverloaded(ID))
#if LLVM_VERSION_MAJOR >= 13
          llvm::errs() << "cannot handle (forward) unknown intrinsic\n"
                       << Intrinsic::getName(ID, ArrayRef<Type *>(), nullptr,
                                             nullptr)
                       << "\n"
                       << I;
#else
          llvm::errs() << "cannot handle (forward) unknown intrinsic\n"
                       << Intrinsic::getName(ID, ArrayRef<Type *>()) << "\n"
                       << I;
#endif
        else
          llvm::errs() << "cannot handle (forward) unknown intrinsic\n"
                       << Intrinsic::getName(ID) << "\n"
                       << I;
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
    const std::map<Argument *, bool> &uncacheable_argsAbove =
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

    std::map<Argument *, bool> uncacheable_args;
    {
      auto in_arg = call.getCalledFunction()->arg_begin();
      auto pp_arg = task->arg_begin();

      // Global.tid is cacheable
      uncacheable_args[pp_arg] = false;
      ++pp_arg;
      // Bound.tid is cacheable
      uncacheable_args[pp_arg] = false;
      ++pp_arg;

      // Ignore the first three args of init call
      ++in_arg;
      ++in_arg;
      ++in_arg;

      for (; pp_arg != task->arg_end();) {
        // If var-args then we may still have args even though outermost
        // has no more
        if (in_arg == call.getCalledFunction()->arg_end()) {
          uncacheable_args[pp_arg] = true;
        } else {
          assert(uncacheable_argsAbove.find(in_arg) !=
                 uncacheable_argsAbove.end());
          uncacheable_args[pp_arg] = uncacheable_argsAbove.find(in_arg)->second;
          ++in_arg;
        }
        ++pp_arg;
      }
    }

    auto called = task;
    // bool modifyPrimal = true;

    bool foreignFunction = called == nullptr;

    SmallVector<Value *, 8> args = {0, 0, 0};
    SmallVector<Value *, 8> pre_args = {0, 0, 0};
    std::vector<DIFFE_TYPE> argsInverted = {DIFFE_TYPE::CONSTANT,
                                            DIFFE_TYPE::CONSTANT};
    std::vector<Instruction *> postCreate;
    std::vector<Instruction *> userReplace;

    SmallVector<Value *, 0> OutTypes;
    SmallVector<Type *, 0> OutFPTypes;

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
            cast<Function>(called), subretType, argsInverted, gutils->TLI,
            TR.analyzer.interprocedural, /*return is used*/ false, nextTypeInfo,
            uncacheable_args, false, /*AtomicAdd*/ true, /*PostOpt*/ false,
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
          std::vector<std::pair<ssize_t, Value *>> geps;
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
          tape = UndefValue::get(
              cast<PointerType>(tapeArg->getType())->getElementType());
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
                  op->replaceAllUsesWith(ph.CreateLoad(
                      pair.first == -1 ? tapeArg
                                       : ph.CreateInBoundsGEP(tapeArg, Idxs)));
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
            op->replaceAllUsesWith(ph.CreateLoad(
                pair.first == -1 ? tapeArg
                                 : ph.CreateInBoundsGEP(tapeArg, Idxs)));
            cast<Instruction>(op)->eraseFromParent();
          }
          assert(tape);
          auto alloc =
              IRBuilder<>(gutils->inversionAllocs)
                  .CreateAlloca(
                      cast<PointerType>(tapeArg->getType())->getElementType());
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
            gutils->TLI, TR.analyzer.interprocedural, subdata,
            /*postopt*/ false, /*omp*/ true);

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
          std::vector<Value *> extracts;
          if (subdata->tapeIndices.size() == 1) {
            assert(subdata->tapeIndices.begin()->second == -1);
            extracts.push_back(tape);
          } else {
            for (auto a : tape->users()) {
              extracts.push_back(a);
            }
          }
          std::vector<LoadInst *> geps;
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
        if (OutTypes.size()) {
          auto ST = StructType::get(newcalled->getContext(), OutFPTypes);
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
            Value *ptr = B.CreateInBoundsGEP(cacheArg, Idxs);

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
                auto vptr = B.CreateGEP(ptr, Idxs);
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
                           Builder2.CreateLoad(
                               Builder2.CreateInBoundsGEP(OutAlloc, Idxs)),
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
                                   IRBuilder<> &Builder2) {
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
            cast<PointerType>(cast<CastInst>(origArg)->getSrcTy())
                ->getElementType()
                ->isFPOrFPVectorTy()) {
          vd = TypeTree(ConcreteType(cast<PointerType>(
                                         cast<CastInst>(origArg)->getSrcTy())
                                         ->getElementType()
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
        SmallVector<Value *, 4> args;
        if (dsto->getType()->isIntegerTy())
          dsto = Builder2.CreateIntToPtr(
              dsto, Type::getInt8PtrTy(dsto->getContext()));
        unsigned dstaddr =
            cast<PointerType>(dsto->getType())->getAddressSpace();
        auto secretpt = PointerType::get(secretty, dstaddr);
        if (offset != 0)
          dsto = Builder2.CreateConstInBoundsGEP1_64(dsto, offset);
        args.push_back(Builder2.CreatePointerCast(dsto, secretpt));
        if (srco->getType()->isIntegerTy())
          srco = Builder2.CreateIntToPtr(
              srco, Type::getInt8PtrTy(dsto->getContext()));
        unsigned srcaddr =
            cast<PointerType>(srco->getType())->getAddressSpace();
        secretpt = PointerType::get(secretty, srcaddr);

        if (offset != 0)
          srco = Builder2.CreateConstInBoundsGEP1_64(srco, offset);
        args.push_back(Builder2.CreatePointerCast(srco, secretpt));
        args.push_back(Builder2.CreateUDiv(
            length,

            ConstantInt::get(length->getType(),
                             Builder2.GetInsertBlock()
                                     ->getParent()
                                     ->getParent()
                                     ->getDataLayout()
                                     .getTypeAllocSizeInBits(secretty) /
                                 8)));

        auto dmemcpy = getOrInsertDifferentialFloatMemcpy(
            *Builder2.GetInsertBlock()->getParent()->getParent(), secretty,
            /*dstalign*/ 1, /*srcalign*/ 1, dstaddr, srcaddr);
        Builder2.CreateCall(dmemcpy, args);
      }

      if (nextStart == size)
        break;
      start = nextStart;
    }
  }

  bool handleBLAS(llvm::CallInst &call, Function *called, StringRef funcName,
                  const std::map<Argument *, bool> &uncacheable_args) {
    CallInst *const newCall = cast<CallInst>(gutils->getNewFromOriginal(&call));
    IRBuilder<> BuilderZ(newCall);
    BuilderZ.setFastMathFlags(getFast());

    if ((funcName == "cblas_ddot" || funcName == "cblas_sdot") &&
        called->isDeclaration()) {
      Type *innerType;
      std::string dfuncName;
      if (funcName == "cblas_ddot") {
        innerType = Type::getDoubleTy(call.getContext());
        dfuncName = "cblas_daxpy";
      } else if (funcName == "cblas_sdot") {
        innerType = Type::getFloatTy(call.getContext());
        dfuncName = "cblas_saxpy";
      } else {
        assert(false && "Unreachable");
      }
      Type *castvals[2] = {call.getArgOperand(1)->getType(),
                           call.getArgOperand(3)->getType()};
      auto *cachetype =
          StructType::get(call.getContext(), ArrayRef<Type *>(castvals));
      Value *undefinit = UndefValue::get(cachetype);
      Value *cacheval;
      auto in_arg = call.getCalledFunction()->arg_begin();
      in_arg++;
      Argument *xfuncarg = in_arg;
      in_arg++;
      in_arg++;
      Argument *yfuncarg = in_arg;
      bool xcache = !gutils->isConstantValue(call.getArgOperand(3)) &&
                    uncacheable_args.find(xfuncarg)->second;
      bool ycache = !gutils->isConstantValue(call.getArgOperand(1)) &&
                    uncacheable_args.find(yfuncarg)->second;
      if ((Mode == DerivativeMode::ReverseModeCombined ||
           Mode == DerivativeMode::ReverseModePrimal) &&
          (xcache || ycache)) {

        Value *arg1, *arg2;
        auto size = ConstantExpr::getSizeOf(innerType);
        if (xcache) {
          auto dmemcpy =
              getOrInsertMemcpyStrided(*gutils->oldFunc->getParent(),
                                       PointerType::getUnqual(innerType), 0, 0);
          auto malins = CallInst::CreateMalloc(
              gutils->getNewFromOriginal(&call), size->getType(), innerType,
              size, call.getArgOperand(0), nullptr, "");
          arg1 =
              BuilderZ.CreateBitCast(malins, call.getArgOperand(1)->getType());
          SmallVector<Value *, 4> args;
          args.push_back(arg1);
          args.push_back(gutils->getNewFromOriginal(call.getArgOperand(1)));
          args.push_back(call.getArgOperand(0));
          args.push_back(call.getArgOperand(2));
          BuilderZ.CreateCall(dmemcpy, args);
        }
        if (ycache) {
          auto dmemcpy =
              getOrInsertMemcpyStrided(*gutils->oldFunc->getParent(),
                                       PointerType::getUnqual(innerType), 0, 0);
          auto malins = CallInst::CreateMalloc(
              gutils->getNewFromOriginal(&call), size->getType(), innerType,
              size, call.getArgOperand(0), nullptr, "");
          arg2 =
              BuilderZ.CreateBitCast(malins, call.getArgOperand(3)->getType());
          SmallVector<Value *, 4> args;
          args.push_back(arg2);
          args.push_back(gutils->getNewFromOriginal(call.getArgOperand(3)));
          args.push_back(call.getArgOperand(0));
          args.push_back(call.getArgOperand(4));
          BuilderZ.CreateCall(dmemcpy, args);
        }
        if (xcache && ycache) {
          auto valins1 = BuilderZ.CreateInsertValue(undefinit, arg1, 0);
          cacheval = BuilderZ.CreateInsertValue(valins1, arg2, 1);
        } else if (xcache)
          cacheval = arg1;
        else {
          assert(ycache);
          cacheval = arg2;
        }
        gutils->cacheForReverse(BuilderZ, cacheval,
                                getIndex(&call, CacheType::Tape));
      }
      if (Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ReverseModeGradient) {
        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);
        auto derivcall = gutils->oldFunc->getParent()->getOrInsertFunction(
            dfuncName, Builder2.getVoidTy(), Builder2.getInt32Ty(), innerType,
            call.getArgOperand(1)->getType(), Builder2.getInt32Ty(),
            call.getArgOperand(3)->getType(), Builder2.getInt32Ty());
        Value *structarg1;
        Value *structarg2;
        if (xcache || ycache) {
          if (Mode == DerivativeMode::ReverseModeGradient &&
              (!gutils->isConstantValue(call.getArgOperand(1)) ||
               !gutils->isConstantValue(call.getArgOperand(3)))) {
            cacheval = BuilderZ.CreatePHI(cachetype, 0);
          }
          cacheval =
              lookup(gutils->cacheForReverse(BuilderZ, cacheval,
                                             getIndex(&call, CacheType::Tape)),
                     Builder2);
          if (xcache && ycache) {
            structarg1 = BuilderZ.CreateExtractValue(cacheval, 0);
            structarg2 = BuilderZ.CreateExtractValue(cacheval, 1);
          } else if (xcache)
            structarg1 = cacheval;
          else if (ycache)
            structarg2 = cacheval;
        }
        if (!xcache)
          structarg1 = lookup(gutils->getNewFromOriginal(call.getArgOperand(1)),
                              Builder2);
        if (!ycache)
          structarg2 = lookup(gutils->getNewFromOriginal(call.getArgOperand(3)),
                              Builder2);
        CallInst *firstdcall, *seconddcall;
        if (!gutils->isConstantValue(call.getArgOperand(3))) {
          Value *estride;
          if (xcache)
            estride = Builder2.getInt32(1);
          else
            estride = lookup(gutils->getNewFromOriginal(call.getArgOperand(2)),
                             Builder2);
          SmallVector<Value *, 6> args1 = {
              lookup(gutils->getNewFromOriginal(call.getArgOperand(0)),
                     Builder2),
              diffe(&call, Builder2),
              structarg1,
              estride,
              lookup(gutils->invertPointerM(call.getArgOperand(3), Builder2),
                     Builder2),
              lookup(gutils->getNewFromOriginal(call.getArgOperand(4)),
                     Builder2)};
          firstdcall = Builder2.CreateCall(derivcall, args1);
        }
        if (!gutils->isConstantValue(call.getArgOperand(1))) {
          Value *estride;
          if (ycache)
            estride = Builder2.getInt32(1);
          else
            estride = lookup(gutils->getNewFromOriginal(call.getArgOperand(4)),
                             Builder2);
          SmallVector<Value *, 6> args2 = {
              lookup(gutils->getNewFromOriginal(call.getArgOperand(0)),
                     Builder2),
              diffe(&call, Builder2),
              structarg2,
              estride,
              lookup(gutils->invertPointerM(call.getArgOperand(1), Builder2),
                     Builder2),
              lookup(gutils->getNewFromOriginal(call.getArgOperand(2)),
                     Builder2)};
          seconddcall = Builder2.CreateCall(derivcall, args2);
        }
        setDiffe(&call, Constant::getNullValue(call.getType()), Builder2);
        if (shouldFree()) {
          if (xcache)
            CallInst::CreateFree(structarg1, firstdcall->getNextNode());
          if (ycache)
            CallInst::CreateFree(structarg2, seconddcall->getNextNode());
        }
      }

      if (gutils->knownRecomputeHeuristic.find(&call) !=
          gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[&call]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(&call, CacheType::Self));
        }
      }

      if (Mode == DerivativeMode::ReverseModeGradient) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      } else {
        eraseIfUnused(call);
      }
      return true;
    }
    return false;
  }

  void handleMPI(llvm::CallInst &call, Function *called, StringRef funcName) {
    assert(Mode != DerivativeMode::ForwardMode);
    assert(called);

    IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&call));
    BuilderZ.setFastMathFlags(getFast());

    // MPI send / recv can only send float/integers
    if (funcName == "PMPI_Isend" || funcName == "MPI_Isend" ||
        funcName == "PMPI_Irecv" || funcName == "MPI_Irecv") {
      Value *firstallocation = nullptr;
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
          auto i32 = Type::getInt32Ty(call.getContext());
          auto c0_64 = ConstantInt::get(i64, 0);
          Type *types[] = {
              /*0 */ Type::getInt8PtrTy(call.getContext()),
              /*1 */ i64,
              /*2 */ Type::getInt8PtrTy(call.getContext()),
              /*3 */ i64,
              /*4 */ i64,
              /*5 */ Type::getInt8PtrTy(call.getContext()),
              /*6 */ Type::getInt8Ty(call.getContext()),
          };
          auto impi = StructType::get(call.getContext(), types, false);

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

            firstallocation = CallInst::CreateMalloc(
                &*BuilderZ.GetInsertPoint(), len_arg->getType(),
                Type::getInt8Ty(call.getContext()),
                ConstantInt::get(Type::getInt64Ty(len_arg->getContext()), 1),
                len_arg, nullptr, "mpirecv_malloccache");
            BuilderZ.CreateStore(
                firstallocation,
                BuilderZ.CreateInBoundsGEP(impialloc,
                                           {c0_64, ConstantInt::get(i32, 0)}));
            BuilderZ.SetInsertPoint(gutils->getNewFromOriginal(&call));

            firstallocation = gutils->cacheForReverse(
                BuilderZ, firstallocation, getIndex(&call, CacheType::Tape));

          } else {
            Value *ibuf = gutils->invertPointerM(call.getOperand(0), BuilderZ);
            if (ibuf->getType()->isIntegerTy())
              ibuf = BuilderZ.CreateIntToPtr(
                  ibuf, Type::getInt8PtrTy(call.getContext()));
            BuilderZ.CreateStore(
                ibuf, BuilderZ.CreateInBoundsGEP(
                          impialloc, {c0_64, ConstantInt::get(i32, 0)}));
          }

          BuilderZ.CreateStore(
              BuilderZ.CreateZExtOrTrunc(
                  gutils->getNewFromOriginal(call.getOperand(1)), types[1]),
              BuilderZ.CreateInBoundsGEP(impialloc,
                                         {c0_64, ConstantInt::get(i32, 1)}));

          Value *dataType = gutils->getNewFromOriginal(call.getOperand(2));
          if (dataType->getType()->isIntegerTy())
            dataType = BuilderZ.CreateIntToPtr(
                dataType, Type::getInt8PtrTy(dataType->getContext()));
          BuilderZ.CreateStore(
              BuilderZ.CreatePointerCast(dataType, types[2]),
              BuilderZ.CreateInBoundsGEP(impialloc,
                                         {c0_64, ConstantInt::get(i32, 2)}));

          BuilderZ.CreateStore(
              BuilderZ.CreateZExtOrTrunc(
                  gutils->getNewFromOriginal(call.getOperand(3)), types[3]),
              BuilderZ.CreateInBoundsGEP(impialloc,
                                         {c0_64, ConstantInt::get(i32, 3)}));

          BuilderZ.CreateStore(
              BuilderZ.CreateZExtOrTrunc(
                  gutils->getNewFromOriginal(call.getOperand(4)), types[4]),
              BuilderZ.CreateInBoundsGEP(impialloc,
                                         {c0_64, ConstantInt::get(i32, 4)}));

          Value *comm = gutils->getNewFromOriginal(call.getOperand(5));
          if (comm->getType()->isIntegerTy())
            comm = BuilderZ.CreateIntToPtr(
                comm, Type::getInt8PtrTy(dataType->getContext()));
          BuilderZ.CreateStore(
              BuilderZ.CreatePointerCast(comm, types[5]),
              BuilderZ.CreateInBoundsGEP(impialloc,
                                         {c0_64, ConstantInt::get(i32, 5)}));

          BuilderZ.CreateStore(
              ConstantInt::get(
                  Type::getInt8Ty(impialloc->getContext()),
                  (funcName == "MPI_Isend" || funcName == "PMPI_Isend")
                      ? (int)MPI_CallType::ISEND
                      : (int)MPI_CallType::IRECV),
              BuilderZ.CreateInBoundsGEP(impialloc,
                                         {c0_64, ConstantInt::get(i32, 6)}));
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
              statusType = PT->getElementType();
          }
          if (Function *recvfn = called->getParent()->getFunction("MPI_Wait")) {
            auto statusArg = recvfn->arg_end();
            statusArg--;
            if (auto PT = dyn_cast<PointerType>(statusArg->getType()))
              statusType = PT->getElementType();
          }
          if (statusType == nullptr) {
            statusType = ArrayType::get(Type::getInt8Ty(call.getContext()), 24);
            llvm::errs() << " warning could not automatically determine mpi "
                            "status type, assuming [24 x i8]\n";
          }
          Value *d_req = lookup(
              gutils->invertPointerM(call.getOperand(6), Builder2), Builder2);
          Value *args[] = {/*req*/ d_req,
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
          auto fcall = Builder2.CreateCall(waitFunc, args);
          fcall->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
#if LLVM_VERSION_MAJOR >= 9
          if (auto F = dyn_cast<Function>(waitFunc.getCallee()))
            fcall->setCallingConv(F->getCallingConv());
#else
          if (auto F = dyn_cast<Function>(waitFunc))
            fcall->setCallingConv(F->getCallingConv());
#endif
          auto len_arg = Builder2.CreateZExtOrTrunc(
              lookup(gutils->getNewFromOriginal(call.getOperand(1)), Builder2),
              Type::getInt64Ty(Builder2.getContext()));
          auto tysize = MPI_TYPE_SIZE(
              lookup(gutils->getNewFromOriginal(call.getOperand(2)), Builder2),
              Builder2, call.getType());
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
            auto dbuf = lookup(
                gutils->invertPointerM(call.getOperand(0), Builder2), Builder2);
            if (dbuf->getType()->isIntegerTy())
              dbuf = Builder2.CreateIntToPtr(
                  dbuf, Type::getInt8PtrTy(call.getContext()));
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
                nargs));
            memset->addParamAttr(0, Attribute::NonNull);
          } else if (funcName == "MPI_Isend" || funcName == "PMPI_Isend") {
            assert(!gutils->isConstantValue(call.getOperand(0)));
            Value *shadow = lookup(
                gutils->invertPointerM(call.getOperand(0), Builder2), Builder2);
            if (Mode == DerivativeMode::ReverseModeCombined) {
              assert(firstallocation);
              firstallocation = lookup(firstallocation, Builder2);
            } else {
              firstallocation =
                  BuilderZ.CreatePHI(Type::getInt8PtrTy(call.getContext()), 0);
              firstallocation = gutils->lookupM(
                  gutils->cacheForReverse(BuilderZ, firstallocation,
                                          getIndex(&call, CacheType::Tape)),
                  Builder2);
            }

            DifferentiableMemCopyFloats(call, call.getOperand(0),
                                        firstallocation, shadow, len_arg,
                                        Builder2);

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
        }
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    if (funcName == "MPI_Wait" || funcName == "PMPI_Wait") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);

        assert(!gutils->isConstantValue(call.getOperand(0)));
        Value *d_req = lookup(
            gutils->invertPointerM(call.getOperand(0), Builder2), Builder2);
        if (d_req->getType()->isIntegerTy()) {
          d_req = Builder2.CreateIntToPtr(
              d_req,
              PointerType::getUnqual(Type::getInt8PtrTy(call.getContext())));
        }
        auto i64 = Type::getInt64Ty(call.getContext());
        Type *types[] = {
            /*0 */ Type::getInt8PtrTy(call.getContext()),
            /*1 */ i64,
            /*2 */ Type::getInt8PtrTy(call.getContext()),
            /*3 */ i64,
            /*4 */ i64,
            /*5 */ Type::getInt8PtrTy(call.getContext()),
            /*6 */ Type::getInt8Ty(call.getContext()),
        };
        auto impi = StructType::get(call.getContext(), types, false);

        Value *d_reqp = Builder2.CreateLoad(Builder2.CreatePointerCast(
            d_req, PointerType::getUnqual(PointerType::getUnqual(impi))));

        Value *isNull = Builder2.CreateICmpEQ(
            d_reqp, Constant::getNullValue(d_reqp->getType()));
        if (auto GV = gutils->newFunc->getParent()->getNamedValue(
                "ompi_request_null")) {
          isNull = Builder2.CreateICmpEQ(
              d_reqp, Builder2.CreateBitCast(GV, d_reqp->getType()));
        }
        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *nonnullBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_nonnull",
            gutils->newFunc);
        BasicBlock *endBlock = gutils->addReverseBlock(
            nonnullBlock, currentBlock->getName() + "_end", gutils->newFunc);

        Builder2.CreateCondBr(isNull, endBlock, nonnullBlock);
        Builder2.SetInsertPoint(nonnullBlock);

        Value *cache = Builder2.CreateLoad(d_reqp);
        if (shouldFree()) {
          CallInst *freecall = cast<CallInst>(
              CallInst::CreateFree(d_reqp, Builder2.GetInsertBlock()));
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

        Function *dwait = getOrInsertDifferentialMPI_Wait(
            *called->getParent(), types, d_req->getType());
        Value *args[] = {Builder2.CreateExtractValue(cache, 0),
                         Builder2.CreateExtractValue(cache, 1),
                         Builder2.CreateExtractValue(cache, 2),
                         Builder2.CreateExtractValue(cache, 3),
                         Builder2.CreateExtractValue(cache, 4),
                         Builder2.CreateExtractValue(cache, 5),
                         Builder2.CreateExtractValue(cache, 6),
                         d_req};
        auto cal = Builder2.CreateCall(dwait, args);
        cal->setCallingConv(dwait->getCallingConv());
        cal->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
        Builder2.CreateBr(endBlock);

        Builder2.SetInsertPoint(endBlock);
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    if (funcName == "MPI_Waitall" || funcName == "PMPI_Waitall") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);

        assert(!gutils->isConstantValue(call.getOperand(1)));
        Value *count =
            lookup(gutils->getNewFromOriginal(call.getOperand(0)), Builder2);
        Value *d_req_orig = lookup(
            gutils->invertPointerM(call.getOperand(1), Builder2), Builder2);
        if (d_req_orig->getType()->isIntegerTy()) {
          d_req_orig = Builder2.CreateIntToPtr(
              d_req_orig,
              PointerType::getUnqual(Type::getInt8PtrTy(call.getContext())));
        }

        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *loopBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_loop", gutils->newFunc);
        BasicBlock *nonnullBlock = gutils->addReverseBlock(
            loopBlock, currentBlock->getName() + "_nonnull", gutils->newFunc);
        BasicBlock *eloopBlock = gutils->addReverseBlock(
            nonnullBlock, currentBlock->getName() + "_eloop", gutils->newFunc);
        BasicBlock *endBlock = gutils->addReverseBlock(
            eloopBlock, currentBlock->getName() + "_end", gutils->newFunc);

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
        Value *d_req = Builder2.CreateGEP(d_req_orig, idxs);

        auto i64 = Type::getInt64Ty(call.getContext());
        Type *types[] = {
            /*0 */ Type::getInt8PtrTy(call.getContext()),
            /*1 */ i64,
            /*2 */ Type::getInt8PtrTy(call.getContext()),
            /*3 */ i64,
            /*4 */ i64,
            /*5 */ Type::getInt8PtrTy(call.getContext()),
            /*6 */ Type::getInt8Ty(call.getContext()),
        };
        auto impi = StructType::get(call.getContext(), types, false);

        Value *d_reqp = Builder2.CreateLoad(Builder2.CreatePointerCast(
            d_req, PointerType::getUnqual(PointerType::getUnqual(impi))));

        Value *isNull = Builder2.CreateICmpEQ(
            d_reqp, Constant::getNullValue(d_reqp->getType()));
        if (auto GV = gutils->newFunc->getParent()->getNamedValue(
                "ompi_request_null")) {
          isNull = Builder2.CreateICmpEQ(
              d_reqp, Builder2.CreateBitCast(GV, d_reqp->getType()));
        }

        Builder2.CreateCondBr(isNull, eloopBlock, nonnullBlock);
        Builder2.SetInsertPoint(nonnullBlock);

        Value *cache = Builder2.CreateLoad(d_reqp);
        if (shouldFree()) {
          CallInst *freecall = cast<CallInst>(
              CallInst::CreateFree(d_reqp, Builder2.GetInsertBlock()));
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

        Function *dwait = getOrInsertDifferentialMPI_Wait(
            *called->getParent(), types, d_req->getType());
        Value *args[] = {Builder2.CreateExtractValue(cache, 0),
                         Builder2.CreateExtractValue(cache, 1),
                         Builder2.CreateExtractValue(cache, 2),
                         Builder2.CreateExtractValue(cache, 3),
                         Builder2.CreateExtractValue(cache, 4),
                         Builder2.CreateExtractValue(cache, 5),
                         Builder2.CreateExtractValue(cache, 6),
                         d_req};
        auto cal = Builder2.CreateCall(dwait, args);
        cal->setCallingConv(dwait->getCallingConv());
        cal->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
        Builder2.CreateBr(eloopBlock);

        Builder2.SetInsertPoint(eloopBlock);
        Builder2.CreateCondBr(Builder2.CreateICmpEQ(inc, count), endBlock,
                              loopBlock);
        Builder2.SetInsertPoint(endBlock);
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    if (funcName == "MPI_Send" || funcName == "MPI_Ssend") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);
        Value *shadow = lookup(
            gutils->invertPointerM(call.getOperand(0), Builder2), Builder2);

        if (shadow->getType()->isIntegerTy())
          shadow = Builder2.CreateIntToPtr(
              shadow, Type::getInt8PtrTy(call.getContext()));

        Type *statusType = nullptr;

        if (Function *recvfn = called->getParent()->getFunction("MPI_Recv")) {
          auto statusArg = recvfn->arg_end();
          statusArg--;
          if (auto PT = dyn_cast<PointerType>(statusArg->getType()))
            statusType = PT->getElementType();
        }
        if (statusType == nullptr) {
          statusType = ArrayType::get(Type::getInt8Ty(call.getContext()), 24);
          llvm::errs() << " warning could not automatically determine mpi "
                          "status type, assuming [24 x i8]\n";
        }

        Value *datatype =
            lookup(gutils->getNewFromOriginal(call.getOperand(2)), Builder2);

        Value *args[] = {
            /*buf*/ NULL,
            /*count*/
            lookup(gutils->getNewFromOriginal(call.getOperand(1)), Builder2),
            /*datatype*/ datatype,
            /*src*/
            lookup(gutils->getNewFromOriginal(call.getOperand(3)), Builder2),
            /*tag*/
            lookup(gutils->getNewFromOriginal(call.getOperand(4)), Builder2),
            /*comm*/
            lookup(gutils->getNewFromOriginal(call.getOperand(5)), Builder2),
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
        auto fcall = Builder2.CreateCall(
            called->getParent()->getOrInsertFunction("MPI_Recv", FT), args);
        fcall->setCallingConv(call.getCallingConv());

        DifferentiableMemCopyFloats(call, call.getOperand(0), firstallocation,
                                    shadow, len_arg, Builder2);

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
        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);

        Value *shadow = lookup(
            gutils->invertPointerM(call.getOperand(0), Builder2), Builder2);
        if (shadow->getType()->isIntegerTy())
          shadow = Builder2.CreateIntToPtr(
              shadow, Type::getInt8PtrTy(call.getContext()));
        Value *datatype =
            lookup(gutils->getNewFromOriginal(call.getOperand(2)), Builder2);

        Value *args[] = {
            shadow,
            lookup(gutils->getNewFromOriginal(call.getOperand(1)), Builder2),
            datatype,
            lookup(gutils->getNewFromOriginal(call.getOperand(3)), Builder2),
            lookup(gutils->getNewFromOriginal(call.getOperand(4)), Builder2),
            lookup(gutils->getNewFromOriginal(call.getOperand(5)), Builder2),
        };
        Type *types[sizeof(args) / sizeof(*args)];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
          types[i] = args[i]->getType();
        FunctionType *FT = FunctionType::get(call.getType(), types, false);

        auto fcall = Builder2.CreateCall(
            called->getParent()->getOrInsertFunction("MPI_Send", FT), args);
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
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);

        Value *shadow = lookup(
            gutils->invertPointerM(call.getOperand(0), Builder2), Builder2);
        if (shadow->getType()->isIntegerTy())
          shadow = Builder2.CreateIntToPtr(
              shadow, Type::getInt8PtrTy(call.getContext()));

        ConcreteType CT = TR.firstPointer(1, call.getOperand(0));
        Type *MPI_OP_Ptr_type =
            PointerType::getUnqual(Type::getInt8PtrTy(call.getContext()));

        Value *count =
            lookup(gutils->getNewFromOriginal(call.getOperand(1)), Builder2);
        Value *datatype =
            lookup(gutils->getNewFromOriginal(call.getOperand(2)), Builder2);
        Value *root =
            lookup(gutils->getNewFromOriginal(call.getOperand(3)), Builder2);
        Value *comm =
            lookup(gutils->getNewFromOriginal(call.getOperand(4)), Builder2);

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
              called->getParent()->getOrInsertFunction("MPI_Reduce", FT), args);
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

          auto mem = cast<CallInst>(Builder2.CreateCall(memcpyF, nargs));
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
            args));
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
          Mode == DerivativeMode::ReverseModeCombined) {
        // TODO insert a check for sum

        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);

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
          llvm::errs() << *gutils->oldFunc << "\n";
          llvm::errs() << *gutils->newFunc << "\n";
          llvm::errs() << " call: " << call << "\n";
          llvm::errs() << " unhandled mpi_allreduce op: " << *orig_op << "\n";
          report_fatal_error("unhandled mpi_allreduce op");
        }

        Value *shadow_recvbuf =
            lookup(gutils->invertPointerM(orig_recvbuf, Builder2), Builder2);
        if (shadow_recvbuf->getType()->isIntegerTy())
          shadow_recvbuf = Builder2.CreateIntToPtr(
              shadow_recvbuf, Type::getInt8PtrTy(call.getContext()));
        Value *shadow_sendbuf =
            lookup(gutils->invertPointerM(orig_sendbuf, Builder2), Builder2);
        if (shadow_sendbuf->getType()->isIntegerTy())
          shadow_sendbuf = Builder2.CreateIntToPtr(
              shadow_sendbuf, Type::getInt8PtrTy(call.getContext()));

        Value *count = lookup(gutils->getNewFromOriginal(orig_count), Builder2);
        Value *datatype =
            lookup(gutils->getNewFromOriginal(orig_datatype), Builder2);
        Value *root = lookup(gutils->getNewFromOriginal(orig_root), Builder2);
        Value *comm = lookup(gutils->getNewFromOriginal(orig_comm), Builder2);

        Value *rank = MPI_COMM_RANK(comm, Builder2, root->getType());

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

            auto mem = cast<CallInst>(Builder2.CreateCall(memcpyF, nargs));
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
              called->getParent()->getOrInsertFunction("MPI_Bcast", FT), args);
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
              args));
          memset->addParamAttr(0, Attribute::NonNull);

          Builder2.CreateBr(mergeBlock);
          Builder2.SetInsertPoint(mergeBlock);
        }

        // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
        DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                    len_arg, Builder2);

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
          Mode == DerivativeMode::ReverseModeCombined) {
        // TODO insert a check for sum

        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);

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
          llvm::errs() << *gutils->oldFunc << "\n";
          llvm::errs() << *gutils->newFunc << "\n";
          llvm::errs() << " call: " << call << "\n";
          llvm::errs() << " unhandled mpi_allreduce op: " << *orig_op << "\n";
          report_fatal_error("unhandled mpi_allreduce op");
        }

        Value *shadow_recvbuf =
            lookup(gutils->invertPointerM(orig_recvbuf, Builder2), Builder2);
        if (shadow_recvbuf->getType()->isIntegerTy())
          shadow_recvbuf = Builder2.CreateIntToPtr(
              shadow_recvbuf, Type::getInt8PtrTy(call.getContext()));
        Value *shadow_sendbuf =
            lookup(gutils->invertPointerM(orig_sendbuf, Builder2), Builder2);
        if (shadow_sendbuf->getType()->isIntegerTy())
          shadow_sendbuf = Builder2.CreateIntToPtr(
              shadow_sendbuf, Type::getInt8PtrTy(call.getContext()));

        Value *count = lookup(gutils->getNewFromOriginal(orig_count), Builder2);
        Value *datatype =
            lookup(gutils->getNewFromOriginal(orig_datatype), Builder2);
        Value *comm = lookup(gutils->getNewFromOriginal(orig_comm), Builder2);

        Value *op = lookup(gutils->getNewFromOriginal(orig_op), Builder2);

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
              args);
        }

        // 3. Zero diff(recvbuffer) [memset to 0]
        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *args[] = {shadow_recvbuf, val_arg, len_arg, volatile_arg};
        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                      Intrinsic::memset, tys),
            args));
        memset->addParamAttr(0, Attribute::NonNull);

        // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
        DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                    len_arg, Builder2);

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
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);

        Value *orig_sendbuf = call.getOperand(0);
        Value *orig_sendcount = call.getOperand(1);
        Value *orig_sendtype = call.getOperand(2);
        Value *orig_recvbuf = call.getOperand(3);
        Value *orig_recvcount = call.getOperand(4);
        Value *orig_recvtype = call.getOperand(5);
        Value *orig_root = call.getOperand(6);
        Value *orig_comm = call.getOperand(7);

        Value *shadow_recvbuf =
            lookup(gutils->invertPointerM(orig_recvbuf, Builder2), Builder2);
        if (shadow_recvbuf->getType()->isIntegerTy())
          shadow_recvbuf = Builder2.CreateIntToPtr(
              shadow_recvbuf, Type::getInt8PtrTy(call.getContext()));
        Value *shadow_sendbuf =
            lookup(gutils->invertPointerM(orig_sendbuf, Builder2), Builder2);
        if (shadow_sendbuf->getType()->isIntegerTy())
          shadow_sendbuf = Builder2.CreateIntToPtr(
              shadow_sendbuf, Type::getInt8PtrTy(call.getContext()));

        Value *recvcount =
            lookup(gutils->getNewFromOriginal(orig_recvcount), Builder2);
        Value *recvtype =
            lookup(gutils->getNewFromOriginal(orig_recvtype), Builder2);

        Value *sendcount =
            lookup(gutils->getNewFromOriginal(orig_sendcount), Builder2);
        Value *sendtype =
            lookup(gutils->getNewFromOriginal(orig_sendtype), Builder2);

        Value *root = lookup(gutils->getNewFromOriginal(orig_root), Builder2);
        Value *comm = lookup(gutils->getNewFromOriginal(orig_comm), Builder2);

        Value *rank = MPI_COMM_RANK(comm, Builder2, root->getType());
        Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType());

        // Get the length for the allocation of the intermediate buffer
        auto sendlen_arg = Builder2.CreateZExtOrTrunc(
            sendcount, Type::getInt64Ty(call.getContext()));
        sendlen_arg =
            Builder2.CreateMul(sendlen_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);

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
              called->getParent()->getOrInsertFunction("MPI_Scatter", FT),
              args);
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
              args));
          memset->addParamAttr(0, Attribute::NonNull);

          Builder2.CreateBr(mergeBlock);
          Builder2.SetInsertPoint(mergeBlock);
        }

        // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
        DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                    sendlen_arg, Builder2);

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
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);

        Value *orig_sendbuf = call.getOperand(0);
        Value *orig_sendcount = call.getOperand(1);
        Value *orig_sendtype = call.getOperand(2);
        Value *orig_recvbuf = call.getOperand(3);
        Value *orig_recvcount = call.getOperand(4);
        Value *orig_recvtype = call.getOperand(5);
        Value *orig_root = call.getOperand(6);
        Value *orig_comm = call.getOperand(7);

        Value *shadow_recvbuf =
            lookup(gutils->invertPointerM(orig_recvbuf, Builder2), Builder2);
        if (shadow_recvbuf->getType()->isIntegerTy())
          shadow_recvbuf = Builder2.CreateIntToPtr(
              shadow_recvbuf, Type::getInt8PtrTy(call.getContext()));
        Value *shadow_sendbuf =
            lookup(gutils->invertPointerM(orig_sendbuf, Builder2), Builder2);
        if (shadow_sendbuf->getType()->isIntegerTy())
          shadow_sendbuf = Builder2.CreateIntToPtr(
              shadow_sendbuf, Type::getInt8PtrTy(call.getContext()));

        Value *recvcount =
            lookup(gutils->getNewFromOriginal(orig_recvcount), Builder2);
        Value *recvtype =
            lookup(gutils->getNewFromOriginal(orig_recvtype), Builder2);

        Value *sendcount =
            lookup(gutils->getNewFromOriginal(orig_sendcount), Builder2);
        Value *sendtype =
            lookup(gutils->getNewFromOriginal(orig_sendtype), Builder2);

        Value *root = lookup(gutils->getNewFromOriginal(orig_root), Builder2);
        Value *comm = lookup(gutils->getNewFromOriginal(orig_comm), Builder2);

        Value *rank = MPI_COMM_RANK(comm, Builder2, root->getType());
        Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType());

        // Get the length for the allocation of the intermediate buffer
        auto recvlen_arg = Builder2.CreateZExtOrTrunc(
            recvcount, Type::getInt64Ty(call.getContext()));
        recvlen_arg =
            Builder2.CreateMul(recvlen_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);

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
              called->getParent()->getOrInsertFunction("MPI_Gather", FT), args);
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
              args));
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
                                      sendlen_phi, Builder2);

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
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);

        Value *orig_sendbuf = call.getOperand(0);
        Value *orig_sendcount = call.getOperand(1);
        Value *orig_sendtype = call.getOperand(2);
        Value *orig_recvbuf = call.getOperand(3);
        Value *orig_recvcount = call.getOperand(4);
        Value *orig_comm = call.getOperand(6);

        Value *shadow_recvbuf =
            lookup(gutils->invertPointerM(orig_recvbuf, Builder2), Builder2);
        if (shadow_recvbuf->getType()->isIntegerTy())
          shadow_recvbuf = Builder2.CreateIntToPtr(
              shadow_recvbuf, Type::getInt8PtrTy(call.getContext()));
        Value *shadow_sendbuf =
            lookup(gutils->invertPointerM(orig_sendbuf, Builder2), Builder2);
        if (shadow_sendbuf->getType()->isIntegerTy())
          shadow_sendbuf = Builder2.CreateIntToPtr(
              shadow_sendbuf, Type::getInt8PtrTy(call.getContext()));

        Value *recvcount =
            lookup(gutils->getNewFromOriginal(orig_recvcount), Builder2);

        Value *sendcount =
            lookup(gutils->getNewFromOriginal(orig_sendcount), Builder2);
        Value *sendtype =
            lookup(gutils->getNewFromOriginal(orig_sendtype), Builder2);

        Value *comm = lookup(gutils->getNewFromOriginal(orig_comm), Builder2);

        Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType());

        // Get the length for the allocation of the intermediate buffer
        auto sendlen_arg = Builder2.CreateZExtOrTrunc(
            sendcount, Type::getInt64Ty(call.getContext()));
        sendlen_arg =
            Builder2.CreateMul(sendlen_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);

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
                              args);
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
              args));
          memset->addParamAttr(0, Attribute::NonNull);
        }

        // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
        DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                    sendlen_arg, Builder2);

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
        uncacheable_args_map.find(&call)->second;

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

    DIFFE_TYPE subretType;
    if (gutils->isConstantValue(orig)) {
      subretType = DIFFE_TYPE::CONSTANT;
    } else {
      if (Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeSplit) {
        subretType = DIFFE_TYPE::DUP_ARG;
      } else {
        if (!orig->getType()->isFPOrFPVectorTy() &&
            TR.query(orig).Inner0().isPossiblePointer()) {
          if (is_value_needed_in_reverse<ValueType::ShadowPtr>(
                  TR, gutils, orig, Mode, oldUnreachable))
            subretType = DIFFE_TYPE::DUP_ARG;
          else
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
            if (invertedReturn->getType() != orig->getType()) {
              llvm::errs() << " o: " << *orig << "\n";
              llvm::errs() << " ot: " << *orig->getType() << "\n";
              llvm::errs() << " ir: " << *invertedReturn << "\n";
              llvm::errs() << " irt: " << *invertedReturn->getType() << "\n";
              llvm::errs() << " p: " << *placeholder << "\n";
              llvm::errs() << " PT: " << *placeholder->getType() << "\n";
              llvm::errs() << " newCall: " << *newCall << "\n";
              llvm::errs() << " newCallT: " << *newCall->getType() << "\n";
            }
            assert(invertedReturn->getType() == orig->getType());
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
          assert(Mode != DerivativeMode::ReverseModeGradient);
          gutils->replaceAWithB(newCall, normalReturn);
          gutils->erase(newCall);
        }
        eraseIfUnused(*orig);
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
          hasNonReturnUse = subretType == DIFFE_TYPE::DUP_ARG;
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
              if (invertedReturn->getType() != orig->getType()) {
                llvm::errs() << " o: " << *orig << "\n";
                llvm::errs() << " ot: " << *orig->getType() << "\n";
                llvm::errs() << " ir: " << *invertedReturn << "\n";
                llvm::errs() << " irt: " << *invertedReturn->getType() << "\n";
                llvm::errs() << " p: " << *placeholder << "\n";
                llvm::errs() << " PT: " << *placeholder->getType() << "\n";
                llvm::errs() << " newCall: " << *newCall << "\n";
                llvm::errs() << " newCallT: " << *newCall->getType() << "\n";
              }
              assert(invertedReturn->getType() == orig->getType());
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
              TR, gutils, orig, Mode, Seen, oldUnreachable);
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
          } else if (!orig->mayWriteToMemory() ||
                     Mode == DerivativeMode::ReverseModeGradient)
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

    if ((funcName == "cblas_ddot" || funcName == "cblas_sdot")) {
      if (handleBLAS(call, called, funcName, uncacheable_args))
        return;
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

          Value *dif0 =
              Builder2.CreateFDiv(diffe(orig->getArgOperand(0), Builder2), cal);
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

          Value *dif0 = Builder2.CreateFDiv(diffe(orig, Builder2), cal);
          addToDiffe(orig->getArgOperand(0), dif0, Builder2, x->getType());
          return;
        }
        case DerivativeMode::ReverseModePrimal: {
          return;
        }
        }
      }

      if (funcName == "atan" || funcName == "atanf" || funcName == "atanl" ||
          funcName == "__fd_atan_1") {
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
        case DerivativeMode::ForwardMode: {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);
          Value *x = gutils->getNewFromOriginal(orig->getArgOperand(0));
          Value *onePx2 = Builder2.CreateFAdd(
              ConstantFP::get(x->getType(), 1.0), Builder2.CreateFMul(x, x));
          Value *dif0 = Builder2.CreateFDiv(
              diffe(orig->getArgOperand(0), Builder2), onePx2);
          setDiffe(orig, dif0, Builder2);
          return;
        }
        case DerivativeMode::ReverseModeGradient:
        case DerivativeMode::ReverseModeCombined: {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);
          Value *x = lookup(gutils->getNewFromOriginal(orig->getArgOperand(0)),
                            Builder2);
          Value *onePx2 = Builder2.CreateFAdd(
              ConstantFP::get(x->getType(), 1.0), Builder2.CreateFMul(x, x));
          Value *dif0 = Builder2.CreateFDiv(diffe(orig, Builder2), onePx2);
          addToDiffe(orig->getArgOperand(0), dif0, Builder2, x->getType());
          return;
        }
        case DerivativeMode::ReverseModePrimal: {
          return;
        }
        }
      }

      if (funcName == "cbrt") {
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
        case DerivativeMode::ForwardMode: {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);

          Value *x = gutils->getNewFromOriginal(orig->getArgOperand(0));
          Value *args[] = {x};
#if LLVM_VERSION_MAJOR >= 11
          auto callval = orig->getCalledOperand();
#else
          auto callval = orig->getCalledValue();
#endif
          CallInst *cubcall = cast<CallInst>(
              Builder2.CreateCall(orig->getFunctionType(), callval, args));
          cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
          cubcall->setCallingConv(orig->getCallingConv());
          Value *dif0 = Builder2.CreateFDiv(
              Builder2.CreateFMul(diffe(orig->getArgOperand(0), Builder2),
                                  cubcall),
              Builder2.CreateFMul(ConstantFP::get(x->getType(), 3), x));
          setDiffe(orig, dif0, Builder2);
          return;
        }
        case DerivativeMode::ReverseModeGradient:
        case DerivativeMode::ReverseModeCombined: {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);

          Value *x = lookup(gutils->getNewFromOriginal(orig->getArgOperand(0)),
                            Builder2);
          Value *args[] = {x};
#if LLVM_VERSION_MAJOR >= 11
          auto callval = orig->getCalledOperand();
#else
          auto callval = orig->getCalledValue();
#endif
          CallInst *cubcall = cast<CallInst>(
              Builder2.CreateCall(orig->getFunctionType(), callval, args));
          cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
          cubcall->setCallingConv(orig->getCallingConv());
          Value *dif0 = Builder2.CreateFDiv(
              Builder2.CreateFMul(diffe(orig, Builder2), cubcall),
              Builder2.CreateFMul(ConstantFP::get(x->getType(), 3), x));
          addToDiffe(orig->getArgOperand(0), dif0, Builder2, x->getType());
          return;
        }
        case DerivativeMode::ReverseModePrimal: {
          return;
        }
        }
      }

      if (funcName == "tanhf" || funcName == "tanh") {
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
        case DerivativeMode::ForwardMode: {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);
          Value *x = gutils->getNewFromOriginal(orig->getArgOperand(0));

          SmallVector<Value *, 1> args = {x};
          auto coshf = gutils->oldFunc->getParent()->getOrInsertFunction(
              (funcName == "tanh") ? "cosh" : "coshf",
              called->getFunctionType(), called->getAttributes());
          auto cal = cast<CallInst>(Builder2.CreateCall(coshf, args));
          Value *dif0 =
              Builder2.CreateFDiv(diffe(orig->getArgOperand(0), Builder2),
                                  Builder2.CreateFMul(cal, cal));
          setDiffe(orig, dif0, Builder2);
          return;
        }
        case DerivativeMode::ReverseModeGradient:
        case DerivativeMode::ReverseModeCombined: {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);
          Value *x = lookup(gutils->getNewFromOriginal(orig->getArgOperand(0)),
                            Builder2);

          SmallVector<Value *, 1> args = {x};
          auto coshf = gutils->oldFunc->getParent()->getOrInsertFunction(
              (funcName == "tanh") ? "cosh" : "coshf",
              called->getFunctionType(), called->getAttributes());
          auto cal = cast<CallInst>(Builder2.CreateCall(coshf, args));
          Value *dif0 = Builder2.CreateFDiv(diffe(orig, Builder2),
                                            Builder2.CreateFMul(cal, cal));
          setDiffe(orig, Constant::getNullValue(orig->getType()), Builder2);
          addToDiffe(orig->getArgOperand(0), dif0, Builder2, x->getType());
          return;
        }
        case DerivativeMode::ReverseModePrimal: {
          return;
        }
        }
      }

      if (funcName == "coshf" || funcName == "cosh") {
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
        case DerivativeMode::ForwardMode: {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);
          Value *x = gutils->getNewFromOriginal(orig->getArgOperand(0));

          SmallVector<Value *, 1> args = {x};
          auto sinhf = gutils->oldFunc->getParent()->getOrInsertFunction(
              (funcName == "cosh") ? "sinh" : "sinhf",
              called->getFunctionType(), called->getAttributes());
          auto cal = cast<CallInst>(Builder2.CreateCall(sinhf, args));
          Value *dif0 =
              Builder2.CreateFMul(diffe(orig->getArgOperand(0), Builder2), cal);
          setDiffe(orig, dif0, Builder2);
          return;
        }
        case DerivativeMode::ReverseModeGradient:
        case DerivativeMode::ReverseModeCombined: {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);
          Value *x = lookup(gutils->getNewFromOriginal(orig->getArgOperand(0)),
                            Builder2);

          SmallVector<Value *, 1> args = {x};
          auto sinhf = gutils->oldFunc->getParent()->getOrInsertFunction(
              (funcName == "cosh") ? "sinh" : "sinhf",
              called->getFunctionType(), called->getAttributes());
          auto cal = cast<CallInst>(Builder2.CreateCall(sinhf, args));
          Value *dif0 = Builder2.CreateFMul(diffe(orig, Builder2), cal);
          setDiffe(orig, Constant::getNullValue(orig->getType()), Builder2);
          addToDiffe(orig->getArgOperand(0), dif0, Builder2, x->getType());
          return;
        }
        case DerivativeMode::ReverseModePrimal: {
          return;
        }
        }
      }
      if (funcName == "sinhf" || funcName == "sinh") {
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
        case DerivativeMode::ForwardMode: {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);
          Value *x = gutils->getNewFromOriginal(orig->getArgOperand(0));

          SmallVector<Value *, 1> args = {x};
          auto sinhf = gutils->oldFunc->getParent()->getOrInsertFunction(
              (funcName == "sinh") ? "cosh" : "coshf",
              called->getFunctionType(), called->getAttributes());
          auto cal = cast<CallInst>(Builder2.CreateCall(sinhf, args));
          Value *dif0 =
              Builder2.CreateFMul(diffe(orig->getArgOperand(0), Builder2), cal);
          setDiffe(orig, dif0, Builder2);
          return;
        }
        case DerivativeMode::ReverseModeGradient:
        case DerivativeMode::ReverseModeCombined: {
          IRBuilder<> Builder2(call.getParent());
          getReverseBuilder(Builder2);
          Value *x = lookup(gutils->getNewFromOriginal(orig->getArgOperand(0)),
                            Builder2);

          SmallVector<Value *, 1> args = {x};
          auto sinhf = gutils->oldFunc->getParent()->getOrInsertFunction(
              (funcName == "sinh") ? "cosh" : "coshf",
              called->getFunctionType(), called->getAttributes());
          auto cal = cast<CallInst>(Builder2.CreateCall(sinhf, args));
          Value *dif0 = Builder2.CreateFMul(diffe(orig, Builder2), cal);
          setDiffe(orig, Constant::getNullValue(orig->getType()), Builder2);
          addToDiffe(orig->getArgOperand(0), dif0, Builder2, x->getType());
          return;
        }
        case DerivativeMode::ReverseModePrimal: {
          return;
        }
        }
      }

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
                  TR, gutils, orig, Mode, Seen, oldUnreachable);
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
          default:
            llvm_unreachable("unhandled mode");
          case DerivativeMode::ReverseModePrimal:
            return;
          case DerivativeMode::ForwardMode:
          case DerivativeMode::ReverseModeGradient:
          case DerivativeMode::ReverseModeCombined: {
            IRBuilder<> Builder2(&call);
            if (Mode == DerivativeMode::ForwardMode)
              getForwardBuilder(Builder2);
            else
              getReverseBuilder(Builder2);

            Value *x = gutils->getNewFromOriginal(orig->getArgOperand(0));
            if (Mode != DerivativeMode::ForwardMode)
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
              Value *reexp =
                  Builder2.CreateCall(ExpF, std::vector<Value *>({re}));

              Function *CosF = Intrinsic::getDeclaration(
                  gutils->oldFunc->getParent(), Intrinsic::cos, tys);
              Function *SinF = Intrinsic::getDeclaration(
                  gutils->oldFunc->getParent(), Intrinsic::sin, tys);

              cal = UndefValue::get(x->getType());
              cal = Builder2.CreateInsertValue(
                  cal,
                  Builder2.CreateFMul(
                      reexp,
                      Builder2.CreateCall(CosF, std::vector<Value *>({im}))),
                  0);
              cal = Builder2.CreateInsertValue(
                  cal,
                  Builder2.CreateFMul(
                      reexp,
                      Builder2.CreateCall(SinF, std::vector<Value *>({im}))),
                  1);
            } else {
              cal = Builder2.CreateCall(ExpF, std::vector<Value *>({sq}));
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

            Value *dfactor = (Mode == DerivativeMode::ForwardMode)
                                 ? diffe(orig->getArgOperand(0), Builder2)
                                 : diffe(orig, Builder2);

            if (funcName.startswith("Faddeeva")) {
              Value *re = Builder2.CreateExtractValue(cal, 0);
              Value *im = Builder2.CreateExtractValue(cal, 1);

              Value *fac_re = Builder2.CreateExtractValue(dfactor, 0);
              Value *fac_im = Builder2.CreateExtractValue(dfactor, 1);

              cal = UndefValue::get(x->getType());
              cal = Builder2.CreateInsertValue(
                  cal,
                  Builder2.CreateFSub(Builder2.CreateFMul(re, fac_re),
                                      Builder2.CreateFMul(im, fac_im)),
                  0);
              cal = Builder2.CreateInsertValue(
                  cal,
                  Builder2.CreateFAdd(Builder2.CreateFMul(im, fac_re),
                                      Builder2.CreateFMul(re, fac_im)),
                  1);
            } else {
              cal = Builder2.CreateFMul(cal, dfactor);
            }

            if (Mode == DerivativeMode::ForwardMode) {
              setDiffe(orig, cal, Builder2);
            } else {
              setDiffe(orig, Constant::getNullValue(orig->getType()), Builder2);
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
          case DerivativeMode::ForwardMode: {
            IRBuilder<> Builder2(&call);
            getForwardBuilder(Builder2);
            Value *x = gutils->getNewFromOriginal(orig->getArgOperand(0));

            Value *dx = Builder2.CreateCall(
                gutils->oldFunc->getParent()->getOrInsertFunction(
                    (funcName[0] == 'j') ? ((funcName == "j0") ? "j1" : "j1f")
                                         : ((funcName == "y0") ? "y1" : "y1f"),
                    called->getFunctionType()),
                std::vector<Value *>({x}));
            dx = Builder2.CreateFNeg(dx);
            dx = Builder2.CreateFMul(dx,
                                     diffe(orig->getArgOperand(0), Builder2));
            setDiffe(orig, dx, Builder2);
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
                std::vector<Value *>({x}));
            dx = Builder2.CreateFNeg(dx);
            dx = Builder2.CreateFMul(dx, diffe(orig, Builder2));
            setDiffe(orig, Constant::getNullValue(orig->getType()), Builder2);
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
          case DerivativeMode::ForwardMode: {
            IRBuilder<> Builder2(&call);
            getForwardBuilder(Builder2);
            Value *x = gutils->getNewFromOriginal(orig->getArgOperand(0));

            Value *d0 = Builder2.CreateCall(
                gutils->oldFunc->getParent()->getOrInsertFunction(
                    (funcName[0] == 'j') ? ((funcName == "j1") ? "j0" : "j0f")
                                         : ((funcName == "y1") ? "y0" : "y0f"),
                    called->getFunctionType()),
                std::vector<Value *>({x}));

            Type *intType =
                Type::getIntNTy(called->getContext(), sizeof(int) * 8);
            Type *pargs[] = {intType, x->getType()};
            auto FT2 = FunctionType::get(x->getType(), pargs, false);
            Value *d2 = Builder2.CreateCall(
                gutils->oldFunc->getParent()->getOrInsertFunction(
                    (funcName[0] == 'j') ? ((funcName == "j1") ? "jn" : "jnf")
                                         : ((funcName == "y1") ? "yn" : "ynf"),
                    FT2),
                std::vector<Value *>({ConstantInt::get(intType, 2), x}));
            Value *dx = Builder2.CreateFSub(d0, d2);
            dx = Builder2.CreateFMul(dx, ConstantFP::get(x->getType(), 0.5));
            dx = Builder2.CreateFMul(dx,
                                     diffe(orig->getArgOperand(0), Builder2));
            setDiffe(orig, dx, Builder2);
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
                std::vector<Value *>({x}));

            Type *intType =
                Type::getIntNTy(called->getContext(), sizeof(int) * 8);
            Type *pargs[] = {intType, x->getType()};
            auto FT2 = FunctionType::get(x->getType(), pargs, false);
            Value *d2 = Builder2.CreateCall(
                gutils->oldFunc->getParent()->getOrInsertFunction(
                    (funcName[0] == 'j') ? ((funcName == "j1") ? "jn" : "jnf")
                                         : ((funcName == "y1") ? "yn" : "ynf"),
                    FT2),
                std::vector<Value *>({ConstantInt::get(intType, 2), x}));
            Value *dx = Builder2.CreateFSub(d0, d2);
            dx = Builder2.CreateFMul(dx, ConstantFP::get(x->getType(), 0.5));
            dx = Builder2.CreateFMul(dx, diffe(orig, Builder2));
            setDiffe(orig, Constant::getNullValue(orig->getType()), Builder2);
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
          case DerivativeMode::ForwardMode: {
            IRBuilder<> Builder2(&call);
            getForwardBuilder(Builder2);
            Value *x = gutils->getNewFromOriginal(orig->getArgOperand(1));
            Value *n = gutils->getNewFromOriginal(orig->getArgOperand(0));

            Value *d0 = Builder2.CreateCall(
                called,
                std::vector<Value *>(
                    {Builder2.CreateSub(n, ConstantInt::get(n->getType(), 1)),
                     x}));

            Value *d2 = Builder2.CreateCall(
                called,
                std::vector<Value *>(
                    {Builder2.CreateAdd(n, ConstantInt::get(n->getType(), 1)),
                     x}));

            Value *dx = Builder2.CreateFSub(d0, d2);
            dx = Builder2.CreateFMul(dx, ConstantFP::get(x->getType(), 0.5));
            dx = Builder2.CreateFMul(dx,
                                     diffe(orig->getArgOperand(1), Builder2));
            setDiffe(orig, dx, Builder2);
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
                std::vector<Value *>(
                    {Builder2.CreateSub(n, ConstantInt::get(n->getType(), 1)),
                     x}));

            Value *d2 = Builder2.CreateCall(
                called,
                std::vector<Value *>(
                    {Builder2.CreateAdd(n, ConstantInt::get(n->getType(), 1)),
                     x}));

            Value *dx = Builder2.CreateFSub(d0, d2);
            dx = Builder2.CreateFMul(dx, ConstantFP::get(x->getType(), 0.5));
            dx = Builder2.CreateFMul(dx, diffe(orig, Builder2));
            setDiffe(orig, Constant::getNullValue(orig->getType()), Builder2);
            addToDiffe(orig->getArgOperand(1), dx, Builder2, x->getType());
            return;
          }
          case DerivativeMode::ReverseModePrimal: {
            return;
          }
          }
        }

        if (funcName == "julia.write_barrier") {
          if (Mode == DerivativeMode::ReverseModeGradient) {
            eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
            return;
          }
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
          return;
        }
        Intrinsic::ID ID = Intrinsic::not_intrinsic;
        if (isMemFreeLibMFunction(funcName, &ID)) {
          if (Mode == DerivativeMode::ReverseModePrimal ||
              gutils->isConstantInstruction(orig)) {
            eraseIfUnused(*orig);
            return;
          }

          if (ID != Intrinsic::not_intrinsic) {
            SmallVector<Value *, 2> orig_ops(orig->getNumOperands());
            for (unsigned i = 0; i < orig->getNumOperands(); ++i) {
              orig_ops[i] = orig->getOperand(i);
            }
            handleAdjointForIntrinsic(ID, *orig, orig_ops);
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
            Value *dif0 = Builder2.CreateFSub(
                Builder2.CreateFMul(Builder2.CreateExtractValue(vdiff, {0}),
                                    dsin),
                Builder2.CreateFMul(Builder2.CreateExtractValue(vdiff, {1}),
                                    dcos));

            setDiffe(orig, dif0, Builder2);
            return;
          }
          case DerivativeMode::ReverseModeGradient:
          case DerivativeMode::ReverseModeCombined: {
            IRBuilder<> Builder2(call.getParent());
            getReverseBuilder(Builder2);

            Value *vdiff = diffe(orig, Builder2);
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
            Value *dif0 = Builder2.CreateFSub(
                Builder2.CreateFMul(Builder2.CreateExtractValue(vdiff, {0}),
                                    dsin),
                Builder2.CreateFMul(Builder2.CreateExtractValue(vdiff, {1}),
                                    dcos));

            setDiffe(orig, Constant::getNullValue(orig->getType()), Builder2);
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
          case DerivativeMode::ForwardMode: {
            IRBuilder<> Builder2(&call);
            getForwardBuilder(Builder2);

            SmallVector<Value *, 2> args;
#if LLVM_VERSION_MAJOR >= 14
            for (auto &arg : orig->args())
#else
            for (auto &arg : orig->arg_operands())
#endif
              args.push_back(gutils->getNewFromOriginal(arg));

            CallInst *d = cast<CallInst>(Builder2.CreateCall(called, args));

            if (args.size() == 2) {
              Value *dif1 = Builder2.CreateFMul(
                  args[0], Builder2.CreateFDiv(
                               diffe(orig->getArgOperand(0), Builder2), d));

              Value *dif2 = Builder2.CreateFMul(
                  args[1], Builder2.CreateFDiv(
                               diffe(orig->getArgOperand(1), Builder2), d));

              setDiffe(orig, Builder2.CreateFAdd(dif1, dif2), Builder2);
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

            Value *vdiff = diffe(orig, Builder2);

            SmallVector<Value *, 2> args;
#if LLVM_VERSION_MAJOR >= 14
            for (auto &arg : orig->args())
#else
            for (auto &arg : orig->arg_operands())
#endif
              args.push_back(lookup(gutils->getNewFromOriginal(arg), Builder2));

            CallInst *d = cast<CallInst>(Builder2.CreateCall(called, args));

            Value *div = Builder2.CreateFDiv(vdiff, d);

            setDiffe(orig, Constant::getNullValue(orig->getType()), Builder2);

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
          case DerivativeMode::ForwardMode: {
            IRBuilder<> Builder2(&call);
            getForwardBuilder(Builder2);

            Value *vdiff = diffe(orig->getArgOperand(0), Builder2);
            Value *exponent =
                gutils->getNewFromOriginal(orig->getArgOperand(1));

            Value *args[] = {vdiff, exponent};

            CallInst *darg = cast<CallInst>(Builder2.CreateCall(called, args));
            setDiffe(orig, darg, Builder2);
            return;
          }
          case DerivativeMode::ReverseModeGradient:
          case DerivativeMode::ReverseModeCombined: {
            IRBuilder<> Builder2(call.getParent());
            getReverseBuilder(Builder2);

            Value *vdiff = diffe(orig, Builder2);
            Value *exponent = lookup(
                gutils->getNewFromOriginal(orig->getArgOperand(1)), Builder2);

            Value *args[] = {vdiff, exponent};

            CallInst *darg = cast<CallInst>(Builder2.CreateCall(called, args));
            setDiffe(orig, Constant::getNullValue(orig->getType()), Builder2);
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

    if (called && isAllocationFunction(*called, gutils->TLI)) {

      bool constval = gutils->isConstantValue(orig);

      if (!constval) {
        if (Mode == DerivativeMode::ReverseModeCombined ||
            Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ReverseModePrimal) {
          auto anti =
              gutils->createAntiMalloc(orig, getIndex(orig, CacheType::Shadow));
          if ((Mode == DerivativeMode::ReverseModeCombined ||
               Mode == DerivativeMode::ReverseModeGradient ||
               Mode == DerivativeMode::ForwardModeSplit) &&
              shouldFree()) {
            IRBuilder<> Builder2(call.getParent());
            getReverseBuilder(Builder2);
            Value *tofree = lookup(anti, Builder2);
            assert(tofree);
            assert(tofree->getType());
            assert(Type::getInt8Ty(tofree->getContext()));
            assert(
                PointerType::getUnqual(Type::getInt8Ty(tofree->getContext())));
            assert(Type::getInt8PtrTy(tofree->getContext()));
            auto dbgLoc = gutils->getNewFromOriginal(orig)->getDebugLoc();
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
          }
        } else if (Mode == DerivativeMode::ForwardMode) {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);

          SmallVector<Value *, 2> args;
          for (unsigned i = 0; i < orig->getNumArgOperands(); ++i) {
            auto arg = orig->getArgOperand(i);
            args.push_back(gutils->getNewFromOriginal(arg));
          }
          CallInst *CI = Builder2.CreateCall(orig->getFunctionType(),
                                             orig->getCalledFunction(), args);
          CI->setAttributes(orig->getAttributes());

          auto found = gutils->invertedPointers.find(orig);
          PHINode *placeholder = cast<PHINode>(&*found->second);

          gutils->invertedPointers.erase(found);
          gutils->replaceAWithB(placeholder, CI);
          gutils->erase(placeholder);
          gutils->invertedPointers.insert(
              std::make_pair(orig, InvertedPointerVH(gutils, CI)));
          return;
        }
      }

      std::map<UsageKey, bool> Seen;
      for (auto pair : gutils->knownRecomputeHeuristic)
        if (!pair.second)
          Seen[UsageKey(pair.first, ValueType::Primal)] = false;
      bool primalNeededInReverse =
          Mode == DerivativeMode::ForwardMode
              ? false
              : is_value_needed_in_reverse<ValueType::Primal>(
                    TR, gutils, orig, Mode, Seen, oldUnreachable);
      bool hasPDFree = gutils->allocationsWithGuaranteedFree.count(orig);
      if (!primalNeededInReverse && hasPDFree) {
        if (Mode == DerivativeMode::ReverseModeGradient) {
          eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
        } else {
          if (hasMetadata(orig, "enzyme_fromstack")) {
            IRBuilder<> B(newCall);
            if (auto CI = dyn_cast<ConstantInt>(orig->getArgOperand(0))) {
              B.SetInsertPoint(gutils->inversionAllocs);
            }
            auto replacement = B.CreateAlloca(
                Type::getInt8Ty(orig->getContext()),
                gutils->getNewFromOriginal(orig->getArgOperand(0)));
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
          funcName == "julia.gc_alloc_obj") {
        if (!primalNeededInReverse) {
          if (Mode == DerivativeMode::ReverseModeGradient) {
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
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModePrimal) {
        if ((primalNeededInReverse &&
             !gutils->unnecessaryIntermediates.count(orig)) ||
            hasPDFree) {
          Value *nop = gutils->cacheForReverse(BuilderZ, newCall,
                                               getIndex(orig, CacheType::Self));
          if (Mode == DerivativeMode::ReverseModeGradient && hasPDFree &&
              shouldFree()) {
            IRBuilder<> Builder2(call.getParent());
            getReverseBuilder(Builder2);
            auto dbgLoc = gutils->getNewFromOriginal(orig)->getDebugLoc();
            freeKnownAllocation(Builder2, lookup(nop, Builder2), *called,
                                dbgLoc, gutils->TLI);
          }
        } else if (Mode == DerivativeMode::ReverseModeGradient ||
                   Mode == DerivativeMode::ReverseModeCombined) {
          // Note that here we cannot simply replace with null as users who
          // try to find the shadow pointer will use the shadow of null rather
          // than the true shadow of this
          auto pn = BuilderZ.CreatePHI(
              orig->getType(), 1, (orig->getName() + "_replacementB").str());
          gutils->fictiousPHIs[pn] = orig;
          gutils->replaceAWithB(newCall, pn);
          gutils->erase(newCall);
        }
      } else if (Mode == DerivativeMode::ReverseModeCombined && shouldFree()) {
        IRBuilder<> Builder2(call.getParent());
        getReverseBuilder(Builder2);
        auto dbgLoc = gutils->getNewFromOriginal(orig)->getDebugLoc();
        freeKnownAllocation(Builder2, lookup(newCall, Builder2), *called,
                            dbgLoc, gutils->TLI);
      }

      return;
    }

    if (funcName == "julia.pointer_from_objref") {
      if (gutils->isConstantValue(orig)) {
        eraseIfUnused(*orig);
        return;
      }

      Value *ptrshadow =
          gutils->invertPointerM(call.getArgOperand(0), BuilderZ);
      Value *val =
          BuilderZ.CreateCall(called, std::vector<Value *>({ptrshadow}));

      auto ifound = gutils->invertedPointers.find(orig);
      assert(ifound != gutils->invertedPointers.end());

      auto placeholder = cast<PHINode>(&*ifound->second);
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
              std::vector<Value *>(
                  {ptrshadow, gutils->getNewFromOriginal(call.getArgOperand(1)),
                   gutils->getNewFromOriginal(call.getArgOperand(2))}));
          val = BuilderZ.CreateLoad(ptrshadow);
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
              cast<PointerType>(call.getArgOperand(0)->getType())
                  ->getElementType(),
              1, orig->getName() + "_psxtmp");
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
        auto load = Builder2.CreateLoad(
            gutils->getNewFromOriginal(call.getOperand(0)), "posix_preread");
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
          auto tofree = gutils->invertPointerM(origfree, Builder2);
          if (tofree != origfree) {
            SmallVector<Value *, 2> args = {tofree};
            CallInst *CI = Builder2.CreateCall(orig->getFunctionType(),
                                               orig->getCalledFunction(), args);
            CI->setAttributes(orig->getAttributes());
          }
        }
        return;
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
          Mode == DerivativeMode::ReverseModeGradient) {
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

    if (Mode == DerivativeMode::ForwardMode) {
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

      Value *newcalled = nullptr;

      if (called) {
        newcalled = gutils->Logic.CreateForwardDiff(
            cast<Function>(called), subretType, argsInverted, gutils->TLI,
            TR.analyzer.interprocedural, /*returnValue*/ subretused, Mode,
            gutils->getWidth(), nullptr, nextTypeInfo, {});
      } else {
#if LLVM_VERSION_MAJOR >= 11
        auto callval = orig->getCalledOperand();
#else
        auto callval = orig->getCalledValue();
#endif
        newcalled = gutils->invertPointerM(callval, BuilderZ);

        auto ft = cast<FunctionType>(
            cast<PointerType>(callval->getType())->getElementType());
        bool retActive = subretType != DIFFE_TYPE::CONSTANT;

        ReturnType subretVal =
            subretused
                ? (retActive ? ReturnType::TwoReturns : ReturnType::Return)
                : (retActive ? ReturnType::Return : ReturnType::Void);

        FunctionType *FTy =
            getFunctionTypeForClone(ft, Mode, gutils->getWidth(), nullptr,
                                    argsInverted, false, subretVal, subretType);
        PointerType *fptype = PointerType::getUnqual(FTy);
        newcalled = BuilderZ.CreatePointerCast(newcalled,
                                               PointerType::getUnqual(fptype));
        newcalled = BuilderZ.CreateLoad(newcalled);
      }

      assert(newcalled);
      FunctionType *FT = cast<FunctionType>(
          cast<PointerType>(newcalled->getType())->getElementType());

      CallInst *diffes = Builder2.CreateCall(FT, newcalled, args);
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
      } else if (!FT->getReturnType()->isVoidTy()) {
        diffe = diffes;
      }

      if (ifound != gutils->invertedPointers.end()) {
        auto placeholder = cast<PHINode>(&*ifound->second);
        if (primal) {
          gutils->replaceAWithB(newcall, primal);
          gutils->erase(newcall);
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
          gutils->replaceAWithB(newcall, diffe);
          if (!gutils->isConstantValue(&call)) {
            setDiffe(&call, diffe, Builder2);
          }
          gutils->erase(newcall);
        } else {
          eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);
        }
      }

      return;
    }

    bool modifyPrimal = shouldAugmentCall(orig, gutils, TR);

    SmallVector<Value *, 8> args;
    SmallVector<Value *, 8> pre_args;
    std::vector<DIFFE_TYPE> argsInverted;
    std::vector<Instruction *> postCreate;
    std::vector<Instruction *> userReplace;
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
          orig, *replacedReturns, postCreate, userReplace, gutils, TR,
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

        auto ft = cast<FunctionType>(
            cast<PointerType>(callval->getType())->getElementType());

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
        newcalled = BuilderZ.CreateLoad(newcalled);
        tapeIdx = 0;

        if (subretType == DIFFE_TYPE::DUP_ARG ||
            subretType == DIFFE_TYPE::DUP_NONEED) {
          returnIdx = 1;
          differetIdx = 2;
        }

      } else {
        if (Mode == DerivativeMode::ReverseModePrimal ||
            Mode == DerivativeMode::ReverseModeCombined) {
          subdata = &gutils->Logic.CreateAugmentedPrimal(
              cast<Function>(called), subretType, argsInverted, gutils->TLI,
              TR.analyzer.interprocedural, /*return is used*/ subretused,
              nextTypeInfo, uncacheable_args, false, gutils->AtomicAdd,
              /*PostOpt*/ false);
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
      FunctionType *FT = cast<FunctionType>(
          cast<PointerType>(newcalled->getType())->getElementType());

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

        augmentcall = BuilderZ.CreateCall(FT, newcalled, pre_args);
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
              is_value_needed_in_reverse<ValueType::Primal>(
                  TR, gutils, orig, Mode, oldUnreachable) &&
              !gutils->unnecessaryIntermediates.count(orig)) {
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
          if (is_value_needed_in_reverse<ValueType::Primal>(
                  TR, gutils, orig, Mode, oldUnreachable) &&
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
        auto truetape = BuilderZ.CreateLoad(tapep, "tapeld");
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
        if (is_value_needed_in_reverse<ValueType::Primal>(
                TR, gutils, orig, Mode, oldUnreachable) &&
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

    bool retUsed = replaceFunction && subretused;
    Value *newcalled = nullptr;

    bool subdretptr = (subretType == DIFFE_TYPE::DUP_ARG ||
                       subretType == DIFFE_TYPE::DUP_NONEED) &&
                      replaceFunction; // && (call.getNumUses() != 0);
    DerivativeMode subMode = (replaceFunction || !modifyPrimal)
                                 ? DerivativeMode::ReverseModeCombined
                                 : DerivativeMode::ReverseModeGradient;
    if (called) {
      newcalled = gutils->Logic.CreatePrimalAndGradient(
          (ReverseCacheKey){.todiff = cast<Function>(called),
                            .retType = subretType,
                            .constant_args = argsInverted,
                            .uncacheable_args = uncacheable_args,
                            .returnUsed = retUsed,
                            .shadowReturnUsed = subdretptr,
                            .mode = subMode,
                            .width = gutils->getWidth(),
                            .freeMemory = true,
                            .AtomicAdd = gutils->AtomicAdd,
                            .additionalType = tape ? tape->getType() : nullptr,
                            .typeInfo = nextTypeInfo},
          gutils->TLI, TR.analyzer.interprocedural, subdata);
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

      auto ft = cast<FunctionType>(
          cast<PointerType>(callval->getType())->getElementType());

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
      newcalled =
          Builder2.CreateLoad(Builder2.CreateConstGEP1_64(newcalled, 1));
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
    FunctionType *FT = cast<FunctionType>(
        cast<PointerType>(newcalled->getType())->getElementType());

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

    CallInst *diffes = Builder2.CreateCall(FT, newcalled, args);
    diffes->setCallingConv(orig->getCallingConv());
    diffes->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
#if LLVM_VERSION_MAJOR >= 9
    for (auto pair : gradByVal) {
      diffes->addParamAttr(pair.first, Attribute::getWithByValType(
                                           diffes->getContext(), pair.second));
    }
#endif

    unsigned structidx = retUsed ? 1 : 0;
    if (subdretptr)
      ++structidx;

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
                     << " retUsed=" << retUsed << " subretptr=" << subdretptr
                     << "\n";
      }
      assert(structidx == 0);
    } else {
      assert(cast<StructType>(diffes->getType())->getNumElements() ==
             structidx);
    }

    if (subretType == DIFFE_TYPE::OUT_DIFF)
      setDiffe(orig, Constant::getNullValue(orig->getType()), Builder2);

    if (replaceFunction) {

      // if a function is replaced for joint forward/reverse, handle inverted
      // pointers
      auto ifound = gutils->invertedPointers.find(orig);
      if (ifound != gutils->invertedPointers.end()) {
        auto placeholder = cast<PHINode>(&*ifound->second);
        gutils->invertedPointers.erase(ifound);
        if (subdretptr) {
          dumpMap(gutils->invertedPointers);
          auto dretval =
              cast<Instruction>(Builder2.CreateExtractValue(diffes, {1}));
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
