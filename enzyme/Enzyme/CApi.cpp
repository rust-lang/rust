//===- CApi.cpp - Enzyme API exported to C for external use -----------===//
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
// This file defines various utility functions of Enzyme for access via C
//
//===----------------------------------------------------------------------===//
#include "CApi.h"
#include "EnzymeLogic.h"
#include "LibraryFuncs.h"
#include "SCEV/TargetLibraryInfo.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/GlobalsModRef.h"

using namespace llvm;

TargetLibraryInfo eunwrap(LLVMTargetLibraryInfoRef P) {
  return TargetLibraryInfo(*reinterpret_cast<TargetLibraryInfoImpl *>(P));
}

EnzymeLogic &eunwrap(EnzymeLogicRef LR) { return *(EnzymeLogic *)LR; }

TypeAnalysis &eunwrap(EnzymeTypeAnalysisRef TAR) {
  return *(TypeAnalysis *)TAR;
}
AugmentedReturn *eunwrap(EnzymeAugmentedReturnPtr ARP) {
  return (AugmentedReturn *)ARP;
}
EnzymeAugmentedReturnPtr ewrap(const AugmentedReturn &AR) {
  return (EnzymeAugmentedReturnPtr)(&AR);
}

ConcreteType eunwrap(CConcreteType CDT, llvm::LLVMContext &ctx) {
  switch (CDT) {
  case DT_Anything:
    return BaseType::Anything;
  case DT_Integer:
    return BaseType::Integer;
  case DT_Pointer:
    return BaseType::Pointer;
  case DT_Half:
    return ConcreteType(llvm::Type::getHalfTy(ctx));
  case DT_Float:
    return ConcreteType(llvm::Type::getFloatTy(ctx));
  case DT_Double:
    return ConcreteType(llvm::Type::getDoubleTy(ctx));
  case DT_Unknown:
    return BaseType::Unknown;
  }
  llvm_unreachable("Unknown concrete type to unwrap");
}

std::vector<int> eunwrap(IntList IL) {
  std::vector<int> v;
  for (size_t i = 0; i < IL.size; i++) {
    v.push_back((int)IL.data[i]);
  }
  return v;
}
std::set<int64_t> eunwrap64(IntList IL) {
  std::set<int64_t> v;
  for (size_t i = 0; i < IL.size; i++) {
    v.insert((int64_t)IL.data[i]);
  }
  return v;
}
TypeTree eunwrap(CTypeTreeRef CTT) { return *(TypeTree *)CTT; }

CConcreteType ewrap(const ConcreteType &CT) {
  if (auto flt = CT.isFloat()) {
    if (flt->isHalfTy())
      return DT_Half;
    if (flt->isFloatTy())
      return DT_Float;
    if (flt->isDoubleTy())
      return DT_Double;
  } else {
    switch (CT.SubTypeEnum) {
    case BaseType::Integer:
      return DT_Integer;
    case BaseType::Pointer:
      return DT_Pointer;
    case BaseType::Anything:
      return DT_Anything;
    case BaseType::Unknown:
      return DT_Unknown;
    case BaseType::Float:
      llvm_unreachable("Illegal conversion of concretetype");
    }
  }
  llvm_unreachable("Illegal conversion of concretetype");
}

IntList ewrap(const std::vector<int> &offsets) {
  IntList IL;
  IL.size = offsets.size();
  IL.data = (int64_t *)malloc(IL.size * sizeof(*IL.data));
  for (size_t i = 0; i < offsets.size(); i++) {
    IL.data[i] = offsets[i];
  }
  return IL;
}

CTypeTreeRef ewrap(const TypeTree &TT) {
  return (CTypeTreeRef)(new TypeTree(TT));
}

FnTypeInfo eunwrap(CFnTypeInfo CTI, llvm::Function *F) {
  FnTypeInfo FTI(F);
  // auto &ctx = F->getContext();
  FTI.Return = eunwrap(CTI.Return);

  size_t argnum = 0;
  for (auto &arg : F->args()) {
    FTI.Arguments[&arg] = eunwrap(CTI.Arguments[argnum]);
    FTI.KnownValues[&arg] = eunwrap64(CTI.KnownValues[argnum]);
    argnum++;
  }
  return FTI;
}

extern "C" {

EnzymeLogicRef CreateEnzymeLogic() {
  return (EnzymeLogicRef)(new EnzymeLogic());
}

void ClearEnzymeLogic(EnzymeLogicRef Ref) { eunwrap(Ref).clear(); }

void FreeEnzymeLogic(EnzymeLogicRef Ref) { delete (EnzymeLogic *)Ref; }

EnzymeTypeAnalysisRef CreateTypeAnalysis(char *TripleStr,
                                         char **customRuleNames,
                                         CustomRuleType *customRules,
                                         size_t numRules) {
  TypeAnalysis *TA = new TypeAnalysis(*(
      new TargetLibraryInfo(*(new TargetLibraryInfoImpl(Triple(TripleStr))))));
  for (size_t i = 0; i < numRules; i++) {
    CustomRuleType rule = customRules[i];
    TA->CustomRules[customRuleNames[i]] =
        [=](int direction, TypeTree &returnTree,
            std::vector<TypeTree> &argTrees,
            std::vector<std::set<int64_t>> &knownValues,
            CallInst *call) -> uint8_t {
      CTypeTreeRef creturnTree = (CTypeTreeRef)(&returnTree);
      CTypeTreeRef *cargs = new CTypeTreeRef[argTrees.size()];
      IntList *kvs = new IntList[argTrees.size()];
      for (size_t i = 0; i < argTrees.size(); ++i) {
        cargs[i] = (CTypeTreeRef)(&(argTrees[i]));
        kvs[i].size = knownValues[i].size();
        kvs[i].data = (int64_t *)malloc(kvs[i].size * sizeof(*kvs[i].data));
        size_t j = 0;
        for (auto val : knownValues[i]) {
          kvs[i].data[j] = val;
          j++;
        }
      }
      uint8_t result =
          rule(direction, creturnTree, cargs, kvs, argTrees.size(), wrap(call));
      delete[] cargs;
      for (size_t i = 0; i < argTrees.size(); ++i) {
        free(kvs[i].data);
      }
      delete[] kvs;
      return result;
    };
  }
  return (EnzymeTypeAnalysisRef)TA;
}

void ClearTypeAnalysis(EnzymeTypeAnalysisRef TAR) { eunwrap(TAR).clear(); }

void FreeTypeAnalysis(EnzymeTypeAnalysisRef TAR) {
  TypeAnalysis *TA = (TypeAnalysis *)TAR;
  delete &TA->TLI.Impl;
  delete &TA->TLI;
  delete TA;
}

void EnzymeRegisterAllocationHandler(char *Name, CustomShadowAlloc AHandle,
                                     CustomShadowFree FHandle) {
  shadowHandlers[std::string(Name)] =
      [=](IRBuilder<> &B, CallInst *CI,
          ArrayRef<Value *> Args) -> llvm::Value * {
    SmallVector<LLVMValueRef, 3> refs;
    for (auto a : Args)
      refs.push_back(wrap(a));
    return unwrap(AHandle(wrap(&B), wrap(CI), Args.size(), refs.data()));
  };
  shadowErasers[std::string(Name)] = [=](IRBuilder<> &B, Value *ToFree,
                                         Function *AllocF) -> llvm::CallInst * {
    return cast_or_null<CallInst>(
        unwrap(FHandle(wrap(&B), wrap(ToFree), wrap(AllocF))));
  };
}

LLVMValueRef EnzymeCreatePrimalAndGradient(
    EnzymeLogicRef Logic, LLVMValueRef todiff, CDIFFE_TYPE retType,
    CDIFFE_TYPE *constant_args, size_t constant_args_size,
    EnzymeTypeAnalysisRef TA, uint8_t returnValue, uint8_t dretUsed,
    uint8_t topLevel, LLVMTypeRef additionalArg, CFnTypeInfo typeInfo,
    uint8_t *_uncacheable_args, size_t uncacheable_args_size,
    EnzymeAugmentedReturnPtr augmented, uint8_t AtomicAdd, uint8_t PostOpt) {
  std::vector<DIFFE_TYPE> nconstant_args((DIFFE_TYPE *)constant_args,
                                         (DIFFE_TYPE *)constant_args +
                                             constant_args_size);
  std::map<llvm::Argument *, bool> uncacheable_args;
  size_t argnum = 0;
  for (auto &arg : cast<Function>(unwrap(todiff))->args()) {
    assert(argnum < uncacheable_args_size);
    uncacheable_args[&arg] = _uncacheable_args[argnum];
    argnum++;
  }
  return wrap(eunwrap(Logic).CreatePrimalAndGradient(
      cast<Function>(unwrap(todiff)), (DIFFE_TYPE)retType, nconstant_args,
      eunwrap(TA).TLI, eunwrap(TA), returnValue, dretUsed, topLevel,
      unwrap(additionalArg), eunwrap(typeInfo, cast<Function>(unwrap(todiff))),
      uncacheable_args, eunwrap(augmented), AtomicAdd, PostOpt));
}
EnzymeAugmentedReturnPtr EnzymeCreateAugmentedPrimal(
    EnzymeLogicRef Logic, LLVMValueRef todiff, CDIFFE_TYPE retType,
    CDIFFE_TYPE *constant_args, size_t constant_args_size,
    EnzymeTypeAnalysisRef TA, uint8_t returnUsed, CFnTypeInfo typeInfo,
    uint8_t *_uncacheable_args, size_t uncacheable_args_size,
    uint8_t forceAnonymousTape, uint8_t AtomicAdd, uint8_t PostOpt) {

  std::vector<DIFFE_TYPE> nconstant_args((DIFFE_TYPE *)constant_args,
                                         (DIFFE_TYPE *)constant_args +
                                             constant_args_size);
  std::map<llvm::Argument *, bool> uncacheable_args;
  size_t argnum = 0;
  for (auto &arg : cast<Function>(unwrap(todiff))->args()) {
    assert(argnum < uncacheable_args_size);
    uncacheable_args[&arg] = _uncacheable_args[argnum];
    argnum++;
  }
  return ewrap(eunwrap(Logic).CreateAugmentedPrimal(
      cast<Function>(unwrap(todiff)), (DIFFE_TYPE)retType, nconstant_args,
      eunwrap(TA).TLI, eunwrap(TA), returnUsed,
      eunwrap(typeInfo, cast<Function>(unwrap(todiff))), uncacheable_args,
      forceAnonymousTape, AtomicAdd, PostOpt));
}

LLVMValueRef
EnzymeExtractFunctionFromAugmentation(EnzymeAugmentedReturnPtr ret) {
  auto AR = (AugmentedReturn *)ret;
  return wrap(AR->fn);
}

LLVMTypeRef
EnzymeExtractTapeTypeFromAugmentation(EnzymeAugmentedReturnPtr ret) {
  auto AR = (AugmentedReturn *)ret;
  auto found = AR->returns.find(AugmentedStruct::Tape);
  if (found == AR->returns.end()) {
    return wrap((Type *)nullptr);
  }
  if (found->second == -1) {
    return wrap(AR->fn->getReturnType());
  }
  return wrap(
      cast<StructType>(AR->fn->getReturnType())->getTypeAtIndex(found->second));
}

void EnzymeExtractReturnInfo(EnzymeAugmentedReturnPtr ret, int64_t *data,
                             uint8_t *existed, size_t len) {
  assert(len == 3);
  auto AR = (AugmentedReturn *)ret;
  AugmentedStruct todo[] = {AugmentedStruct::Tape, AugmentedStruct::Return,
                            AugmentedStruct::DifferentialReturn};
  for (size_t i = 0; i < len; i++) {
    auto found = AR->returns.find(todo[i]);
    if (found != AR->returns.end()) {
      existed[i] = true;
      data[i] = (int64_t)found->second;
    } else {
      existed[i] = false;
    }
  }
}

CTypeTreeRef EnzymeNewTypeTree() { return (CTypeTreeRef)(new TypeTree()); }
CTypeTreeRef EnzymeNewTypeTreeCT(CConcreteType CT, LLVMContextRef ctx) {
  return (CTypeTreeRef)(new TypeTree(eunwrap(CT, *unwrap(ctx))));
}
CTypeTreeRef EnzymeNewTypeTreeTR(CTypeTreeRef CTR) {
  return (CTypeTreeRef)(new TypeTree(*(TypeTree *)(CTR)));
}
void EnzymeFreeTypeTree(CTypeTreeRef CTT) { delete (TypeTree *)CTT; }
uint8_t EnzymeSetTypeTree(CTypeTreeRef dst, CTypeTreeRef src) {
  return *(TypeTree *)dst = *(TypeTree *)src;
}
uint8_t EnzymeMergeTypeTree(CTypeTreeRef dst, CTypeTreeRef src) {
  return ((TypeTree *)dst)->orIn(*(TypeTree *)src, /*PointerIntSame*/ false);
}

void EnzymeTypeTreeOnlyEq(CTypeTreeRef CTT, int64_t x) {
  *(TypeTree *)CTT = ((TypeTree *)CTT)->Only(x);
}
void EnzymeTypeTreeData0Eq(CTypeTreeRef CTT) {
  *(TypeTree *)CTT = ((TypeTree *)CTT)->Data0();
}
void EnzymeTypeTreeShiftIndiciesEq(CTypeTreeRef CTT, const char *datalayout,
                                   int64_t offset, int64_t maxSize,
                                   uint64_t addOffset) {
  DataLayout DL(datalayout);
  *(TypeTree *)CTT =
      ((TypeTree *)CTT)->ShiftIndices(DL, offset, maxSize, addOffset);
}
const char *EnzymeTypeTreeToString(CTypeTreeRef src) {
  std::string tmp = ((TypeTree *)src)->str();
  char *cstr = new char[tmp.length() + 1];
  std::strcpy(cstr, tmp.c_str());

  return cstr;
}
void EnzymeTypeTreeToStringFree(const char *cstr) { delete[] cstr; }
}
