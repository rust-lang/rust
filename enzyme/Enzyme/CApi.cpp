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
#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"

#include "EnzymeLogic.h"
#include "GradientUtils.h"
#include "LibraryFuncs.h"
#include "SCEV/TargetLibraryInfo.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/GlobalsModRef.h"

#if LLVM_VERSION_MAJOR >= 9
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/Attributor.h"
#endif

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
  IL.data = new int64_t[IL.size];
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

void EnzymeSetCLBool(void *ptr, uint8_t val) {
  auto cl = (llvm::cl::opt<bool> *)ptr;
  cl->setValue((bool)val);
}

uint8_t EnzymeGetCLBool(void *ptr) {
  auto cl = (llvm::cl::opt<bool> *)ptr;
  return (uint8_t)(bool)cl->getValue();
}

void EnzymeSetCLInteger(void *ptr, int64_t val) {
  auto cl = (llvm::cl::opt<int> *)ptr;
  cl->setValue((int)val);
}

int64_t EnzymeGetCLInteger(void *ptr) {
  auto cl = (llvm::cl::opt<int> *)ptr;
  return (int64_t)cl->getValue();
}

EnzymeLogicRef CreateEnzymeLogic(uint8_t PostOpt) {
  return (EnzymeLogicRef)(new EnzymeLogic((bool)PostOpt));
}

void ClearEnzymeLogic(EnzymeLogicRef Ref) { eunwrap(Ref).clear(); }

void EnzymeLogicErasePreprocessedFunctions(EnzymeLogicRef Ref) {
  auto &Logic = eunwrap(Ref);
  for (const auto &pair : Logic.PPC.cache)
    pair.second->eraseFromParent();
}

void FreeEnzymeLogic(EnzymeLogicRef Ref) { delete (EnzymeLogic *)Ref; }

EnzymeTypeAnalysisRef CreateTypeAnalysis(EnzymeLogicRef Log,
                                         char **customRuleNames,
                                         CustomRuleType *customRules,
                                         size_t numRules) {
  TypeAnalysis *TA = new TypeAnalysis(((EnzymeLogic *)Log)->PPC.FAM);
  for (size_t i = 0; i < numRules; i++) {
    CustomRuleType rule = customRules[i];
    TA->CustomRules[customRuleNames[i]] =
        [=](int direction, TypeTree &returnTree, ArrayRef<TypeTree> argTrees,
            ArrayRef<std::set<int64_t>> knownValues,
            CallInst *call) -> uint8_t {
      CTypeTreeRef creturnTree = (CTypeTreeRef)(&returnTree);
      CTypeTreeRef *cargs = new CTypeTreeRef[argTrees.size()];
      IntList *kvs = new IntList[argTrees.size()];
      for (size_t i = 0; i < argTrees.size(); ++i) {
        cargs[i] = (CTypeTreeRef)(&(argTrees[i]));
        kvs[i].size = knownValues[i].size();
        kvs[i].data = new int64_t[kvs[i].size];
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
        delete[] kvs[i].data;
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
  delete TA;
}

void *EnzymeAnalyzeTypes(EnzymeTypeAnalysisRef TAR, CFnTypeInfo CTI,
                         LLVMValueRef F) {
  FnTypeInfo FTI(eunwrap(CTI, cast<Function>(unwrap(F))));
  return (void *)&((TypeAnalysis *)TAR)->analyzeFunction(FTI).analyzer;
}

void *EnzymeGradientUtilsTypeAnalyzer(GradientUtils *G) {
  return (void *)&G->TR.analyzer;
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
  shadowErasers[std::string(Name)] = [=](IRBuilder<> &B,
                                         Value *ToFree) -> llvm::CallInst * {
    return cast_or_null<CallInst>(unwrap(FHandle(wrap(&B), wrap(ToFree))));
  };
}

void EnzymeRegisterCallHandler(char *Name,
                               CustomAugmentedFunctionForward FwdHandle,
                               CustomFunctionReverse RevHandle) {
  auto &pair = customCallHandlers[std::string(Name)];
  pair.first = [=](IRBuilder<> &B, CallInst *CI, GradientUtils &gutils,
                   Value *&normalReturn, Value *&shadowReturn, Value *&tape) {
    LLVMValueRef normalR = wrap(normalReturn);
    LLVMValueRef shadowR = wrap(shadowReturn);
    LLVMValueRef tapeR = wrap(tape);
    FwdHandle(wrap(&B), wrap(CI), &gutils, &normalR, &shadowR, &tapeR);
    normalReturn = unwrap(normalR);
    shadowReturn = unwrap(shadowR);
    tape = unwrap(tapeR);
  };
  pair.second = [=](IRBuilder<> &B, CallInst *CI, DiffeGradientUtils &gutils,
                    Value *tape) {
    RevHandle(wrap(&B), wrap(CI), &gutils, wrap(tape));
  };
}

void EnzymeRegisterFwdCallHandler(char *Name, CustomFunctionForward FwdHandle) {
  auto &pair = customFwdCallHandlers[std::string(Name)];
  pair = [=](IRBuilder<> &B, CallInst *CI, GradientUtils &gutils,
             Value *&normalReturn, Value *&shadowReturn) {
    LLVMValueRef normalR = wrap(normalReturn);
    LLVMValueRef shadowR = wrap(shadowReturn);
    FwdHandle(wrap(&B), wrap(CI), &gutils, &normalR, &shadowR);
    normalReturn = unwrap(normalR);
    shadowReturn = unwrap(shadowR);
  };
}

uint64_t EnzymeGradientUtilsGetWidth(GradientUtils *gutils) {
  return gutils->getWidth();
}

LLVMTypeRef EnzymeGradientUtilsGetShadowType(GradientUtils *gutils,
                                             LLVMTypeRef T) {
  return wrap(gutils->getShadowType(unwrap(T)));
}

LLVMTypeRef EnzymeGetShadowType(uint64_t width, LLVMTypeRef T) {
  return wrap(GradientUtils::getShadowType(unwrap(T), width));
}

LLVMValueRef EnzymeGradientUtilsNewFromOriginal(GradientUtils *gutils,
                                                LLVMValueRef val) {
  return wrap(gutils->getNewFromOriginal(unwrap(val)));
}

CDerivativeMode EnzymeGradientUtilsGetMode(GradientUtils *gutils) {
  return (CDerivativeMode)gutils->mode;
}

CDIFFE_TYPE
EnzymeGradientUtilsGetDiffeType(GradientUtils *G, LLVMValueRef oval,
                                uint8_t foreignFunction) {
  return (CDIFFE_TYPE)(G->getDiffeType(unwrap(oval), foreignFunction != 0));
}

CDIFFE_TYPE
EnzymeGradientUtilsGetReturnDiffeType(GradientUtils *G, LLVMValueRef oval,
                                      uint8_t *needsPrimal,
                                      uint8_t *needsShadow) {
  bool needsPrimalB;
  bool needsShadowB;
  auto res = (CDIFFE_TYPE)(G->getReturnDiffeType(cast<CallInst>(unwrap(oval)),
                                                 &needsPrimalB, &needsShadowB));
  if (needsPrimal)
    *needsPrimal = needsPrimalB;
  if (needsShadow)
    *needsShadow = needsShadowB;
  return res;
}

void EnzymeGradientUtilsSetDebugLocFromOriginal(GradientUtils *gutils,
                                                LLVMValueRef val,
                                                LLVMValueRef orig) {
  return cast<Instruction>(unwrap(val))
      ->setDebugLoc(gutils->getNewFromOriginal(
          cast<Instruction>(unwrap(orig))->getDebugLoc()));
}

LLVMValueRef EnzymeGradientUtilsLookup(GradientUtils *gutils, LLVMValueRef val,
                                       LLVMBuilderRef B) {
  return wrap(gutils->lookupM(unwrap(val), *unwrap(B)));
}

LLVMValueRef EnzymeGradientUtilsInvertPointer(GradientUtils *gutils,
                                              LLVMValueRef val,
                                              LLVMBuilderRef B) {
  return wrap(gutils->invertPointerM(unwrap(val), *unwrap(B)));
}

LLVMValueRef EnzymeGradientUtilsDiffe(DiffeGradientUtils *gutils,
                                      LLVMValueRef val, LLVMBuilderRef B) {
  return wrap(gutils->diffe(unwrap(val), *unwrap(B)));
}

void EnzymeGradientUtilsAddToDiffe(DiffeGradientUtils *gutils, LLVMValueRef val,
                                   LLVMValueRef diffe, LLVMBuilderRef B,
                                   LLVMTypeRef T) {
  gutils->addToDiffe(unwrap(val), unwrap(diffe), *unwrap(B), unwrap(T));
}

void EnzymeGradientUtilsSetDiffe(DiffeGradientUtils *gutils, LLVMValueRef val,
                                 LLVMValueRef diffe, LLVMBuilderRef B) {
  gutils->setDiffe(unwrap(val), unwrap(diffe), *unwrap(B));
}

uint8_t EnzymeGradientUtilsIsConstantValue(GradientUtils *gutils,
                                           LLVMValueRef val) {
  return gutils->isConstantValue(unwrap(val));
}

uint8_t EnzymeGradientUtilsIsConstantInstruction(GradientUtils *gutils,
                                                 LLVMValueRef val) {
  return gutils->isConstantInstruction(cast<Instruction>(unwrap(val)));
}

LLVMBasicBlockRef EnzymeGradientUtilsAllocationBlock(GradientUtils *gutils) {
  return wrap(gutils->inversionAllocs);
}

void EnzymeGradientUtilsGetUncacheableArgs(GradientUtils *gutils,
                                           LLVMValueRef orig, uint8_t *data,
                                           uint64_t size) {
  if (gutils->mode == DerivativeMode::ForwardMode)
    return;

  CallInst *call = cast<CallInst>(unwrap(orig));

  auto found = gutils->uncacheable_args_map_ptr->find(call);
  assert(found != gutils->uncacheable_args_map_ptr->end());

  const std::map<Argument *, bool> &uncacheable_args = found->second;

  auto Fn = getFunctionFromCall(call);
  assert(Fn);
  size_t cur = 0;
  for (auto &arg : Fn->args()) {
    assert(cur < size);
    auto found2 = uncacheable_args.find(&arg);
    assert(found2 != uncacheable_args.end());
    data[cur] = found2->second;
    cur++;
  }
}

CTypeTreeRef EnzymeGradientUtilsAllocAndGetTypeTree(GradientUtils *gutils,
                                                    LLVMValueRef val) {
  auto v = unwrap(val);
  TypeTree TT = gutils->TR.query(v);
  TypeTree *pTT = new TypeTree(TT);
  return (CTypeTreeRef)pTT;
}

void EnzymeGradientUtilsDumpTypeResults(GradientUtils *gutils) {
  gutils->TR.dump();
}

void EnzymeGradientUtilsSubTransferHelper(
    GradientUtils *gutils, CDerivativeMode mode, LLVMTypeRef secretty,
    uint64_t intrinsic, uint64_t dstAlign, uint64_t srcAlign, uint64_t offset,
    uint8_t dstConstant, LLVMValueRef shadow_dst, uint8_t srcConstant,
    LLVMValueRef shadow_src, LLVMValueRef length, LLVMValueRef isVolatile,
    LLVMValueRef MTI, uint8_t allowForward, uint8_t shadowsLookedUp) {
  auto orig = unwrap(MTI);
  assert(orig);
  SubTransferHelper(gutils, (DerivativeMode)mode, unwrap(secretty),
                    (Intrinsic::ID)intrinsic, (unsigned)dstAlign,
                    (unsigned)srcAlign, (unsigned)offset, (bool)dstConstant,
                    unwrap(shadow_dst), (bool)srcConstant, unwrap(shadow_src),
                    unwrap(length), unwrap(isVolatile), cast<CallInst>(orig),
                    (bool)allowForward, (bool)shadowsLookedUp);
}

LLVMValueRef EnzymeCreateForwardDiff(
    EnzymeLogicRef Logic, LLVMValueRef todiff, CDIFFE_TYPE retType,
    CDIFFE_TYPE *constant_args, size_t constant_args_size,
    EnzymeTypeAnalysisRef TA, uint8_t returnValue, CDerivativeMode mode,
    uint8_t freeMemory, unsigned width, LLVMTypeRef additionalArg,
    CFnTypeInfo typeInfo, uint8_t *_uncacheable_args,
    size_t uncacheable_args_size, EnzymeAugmentedReturnPtr augmented) {
  SmallVector<DIFFE_TYPE, 4> nconstant_args((DIFFE_TYPE *)constant_args,
                                            (DIFFE_TYPE *)constant_args +
                                                constant_args_size);
  std::map<llvm::Argument *, bool> uncacheable_args;
  size_t argnum = 0;
  for (auto &arg : cast<Function>(unwrap(todiff))->args()) {
    assert(argnum < uncacheable_args_size);
    uncacheable_args[&arg] = _uncacheable_args[argnum];
    argnum++;
  }
  return wrap(eunwrap(Logic).CreateForwardDiff(
      cast<Function>(unwrap(todiff)), (DIFFE_TYPE)retType, nconstant_args,
      eunwrap(TA), returnValue, (DerivativeMode)mode, freeMemory, width,
      unwrap(additionalArg), eunwrap(typeInfo, cast<Function>(unwrap(todiff))),
      uncacheable_args, eunwrap(augmented)));
}
LLVMValueRef EnzymeCreatePrimalAndGradient(
    EnzymeLogicRef Logic, LLVMValueRef todiff, CDIFFE_TYPE retType,
    CDIFFE_TYPE *constant_args, size_t constant_args_size,
    EnzymeTypeAnalysisRef TA, uint8_t returnValue, uint8_t dretUsed,
    CDerivativeMode mode, unsigned width, uint8_t freeMemory,
    LLVMTypeRef additionalArg, CFnTypeInfo typeInfo, uint8_t *_uncacheable_args,
    size_t uncacheable_args_size, EnzymeAugmentedReturnPtr augmented,
    uint8_t AtomicAdd) {
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
      (ReverseCacheKey){
          .todiff = cast<Function>(unwrap(todiff)),
          .retType = (DIFFE_TYPE)retType,
          .constant_args = nconstant_args,
          .uncacheable_args = uncacheable_args,
          .returnUsed = (bool)returnValue,
          .shadowReturnUsed = (bool)dretUsed,
          .mode = (DerivativeMode)mode,
          .width = width,
          .freeMemory = (bool)freeMemory,
          .AtomicAdd = (bool)AtomicAdd,
          .additionalType = unwrap(additionalArg),
          .typeInfo = eunwrap(typeInfo, cast<Function>(unwrap(todiff))),
      },
      eunwrap(TA), eunwrap(augmented)));
}
EnzymeAugmentedReturnPtr EnzymeCreateAugmentedPrimal(
    EnzymeLogicRef Logic, LLVMValueRef todiff, CDIFFE_TYPE retType,
    CDIFFE_TYPE *constant_args, size_t constant_args_size,
    EnzymeTypeAnalysisRef TA, uint8_t returnUsed, uint8_t shadowReturnUsed,
    CFnTypeInfo typeInfo, uint8_t *_uncacheable_args,
    size_t uncacheable_args_size, uint8_t forceAnonymousTape, unsigned width,
    uint8_t AtomicAdd) {

  SmallVector<DIFFE_TYPE, 4> nconstant_args((DIFFE_TYPE *)constant_args,
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
      eunwrap(TA), returnUsed, shadowReturnUsed,
      eunwrap(typeInfo, cast<Function>(unwrap(todiff))), uncacheable_args,
      forceAnonymousTape, width, AtomicAdd));
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

void EnzymeTypeTreeLookupEq(CTypeTreeRef CTT, int64_t size, const char *dl) {
  *(TypeTree *)CTT = ((TypeTree *)CTT)->Lookup(size, DataLayout(dl));
}

CConcreteType EnzymeTypeTreeInner0(CTypeTreeRef CTT) {
  return ewrap(((TypeTree *)CTT)->Inner0());
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

// TODO deprecated
void EnzymeTypeTreeToStringFree(const char *cstr) { delete[] cstr; }

const char *EnzymeTypeAnalyzerToString(void *src) {
  auto TA = (TypeAnalyzer *)src;
  std::string str;
  raw_string_ostream ss(str);
  TA->dump(ss);
  ss.str();
  char *cstr = new char[str.length() + 1];
  std::strcpy(cstr, str.c_str());
  return cstr;
}

const char *EnzymeGradientUtilsInvertedPointersToString(GradientUtils *gutils,
                                                        void *src) {
  std::string str;
  raw_string_ostream ss(str);
  for (auto z : gutils->invertedPointers) {
    ss << "available inversion for " << *z.first << " of " << *z.second << "\n";
  }
  ss.str();
  char *cstr = new char[str.length() + 1];
  std::strcpy(cstr, str.c_str());
  return cstr;
}

LLVMValueRef EnzymeGradientUtilsCallWithInvertedBundles(
    GradientUtils *gutils, LLVMValueRef func, LLVMValueRef *args_vr,
    uint64_t args_size, LLVMValueRef orig_vr, CValueType *valTys,
    uint64_t valTys_size, LLVMBuilderRef B, uint8_t lookup) {
  auto orig = cast<CallInst>(unwrap(orig_vr));

  ArrayRef<ValueType> ar((ValueType *)valTys, valTys_size);

  IRBuilder<> &BR = *unwrap(B);

  auto Defs = gutils->getInvertedBundles(orig, ar, BR, lookup != 0);

  SmallVector<Value *, 1> args;
  for (size_t i = 0; i < args_size; i++) {
    args.push_back(unwrap(args_vr[i]));
  }

  auto callval = unwrap(func);

#if LLVM_VERSION_MAJOR > 7
  auto res = BR.CreateCall(
      cast<FunctionType>(callval->getType()->getPointerElementType()), callval,
      args, Defs);
#else
  auto res = BR.CreateCall(callval, args, Defs);
#endif
  return wrap(res);
}

void EnzymeStringFree(const char *cstr) { delete[] cstr; }

void EnzymeMoveBefore(LLVMValueRef inst1, LLVMValueRef inst2,
                      LLVMBuilderRef B) {
  Instruction *I1 = cast<Instruction>(unwrap(inst1));
  Instruction *I2 = cast<Instruction>(unwrap(inst2));
  if (I1 != I2) {
    if (B != nullptr) {
      IRBuilder<> &BR = *unwrap(B);
      if (I1->getIterator() == BR.GetInsertPoint()) {
        if (I2->getNextNode() == nullptr)
          BR.SetInsertPoint(I1->getParent());
        else
          BR.SetInsertPoint(I1->getNextNode());
      }
    }
    I1->moveBefore(I2);
  }
}

void EnzymeSetMustCache(LLVMValueRef inst1) {
  Instruction *I1 = cast<Instruction>(unwrap(inst1));
  I1->setMetadata("enzyme_mustcache", MDNode::get(I1->getContext(), {}));
}

uint8_t EnzymeHasFromStack(LLVMValueRef inst1) {
  Instruction *I1 = cast<Instruction>(unwrap(inst1));
  return hasMetadata(I1, "enzyme_fromstack") != 0;
}

void EnzymeReplaceFunctionImplementation(LLVMModuleRef M) {
  ReplaceFunctionImplementation(*unwrap(M));
}

#if LLVM_VERSION_MAJOR >= 9
void EnzymeAddAttributorLegacyPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createAttributorLegacyPass());
}
#endif
LLVMMetadataRef EnzymeMakeNonConstTBAA(LLVMMetadataRef MD) {
  auto M = cast<MDNode>(unwrap(MD));
  if (M->getNumOperands() != 4)
    return MD;
  auto CAM = dyn_cast<ConstantAsMetadata>(M->getOperand(3));
  if (!CAM)
    return MD;
  if (!CAM->getValue()->isOneValue())
    return MD;
  SmallVector<Metadata *, 4> MDs;
  for (auto &M : M->operands())
    MDs.push_back(M);
  MDs[3] =
      ConstantAsMetadata::get(ConstantInt::get(CAM->getValue()->getType(), 0));
  return wrap(MDNode::get(M->getContext(), MDs));
}
}
