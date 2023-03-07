#include "TraceInterface.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

using namespace llvm;

IntegerType *TraceInterface::sizeType(LLVMContext &C) {
  return IntegerType::getInt64Ty(C);
}

Type *TraceInterface::stringType(LLVMContext &C) {
  return IntegerType::getInt8PtrTy(C);
}

FunctionType *TraceInterface::getTraceTy() { return getTraceTy(C); }
FunctionType *TraceInterface::getChoiceTy() { return getChoiceTy(C); }
FunctionType *TraceInterface::insertCallTy() { return insertCallTy(C); }
FunctionType *TraceInterface::insertChoiceTy() { return insertChoiceTy(C); }
FunctionType *TraceInterface::newTraceTy() { return newTraceTy(C); }
FunctionType *TraceInterface::freeTraceTy() { return freeTraceTy(C); }
FunctionType *TraceInterface::hasCallTy() { return hasCallTy(C); }
FunctionType *TraceInterface::hasChoiceTy() { return hasChoiceTy(C); }

FunctionType *TraceInterface::getTraceTy(LLVMContext &C) {
  return FunctionType::get(PointerType::getInt8PtrTy(C),
                           {PointerType::getInt8PtrTy(C), stringType(C)},
                           false);
}

FunctionType *TraceInterface::getChoiceTy(LLVMContext &C) {
  return FunctionType::get(sizeType(C),
                           {PointerType::getInt8PtrTy(C), stringType(C),
                            PointerType::getInt8PtrTy(C), sizeType(C)},
                           false);
}

FunctionType *TraceInterface::insertCallTy(LLVMContext &C) {
  return FunctionType::get(Type::getVoidTy(C),
                           {PointerType::getInt8PtrTy(C), stringType(C),
                            PointerType::getInt8PtrTy(C)},
                           false);
}

FunctionType *TraceInterface::insertChoiceTy(LLVMContext &C) {
  return FunctionType::get(Type::getVoidTy(C),
                           {PointerType::getInt8PtrTy(C), stringType(C),
                            Type::getDoubleTy(C), PointerType::getInt8PtrTy(C),
                            sizeType(C)},
                           false);
}

FunctionType *TraceInterface::newTraceTy(LLVMContext &C) {
  return FunctionType::get(PointerType::getInt8PtrTy(C), {}, false);
}

FunctionType *TraceInterface::freeTraceTy(LLVMContext &C) {
  return FunctionType::get(Type::getVoidTy(C), {PointerType::getInt8PtrTy(C)},
                           false);
}

FunctionType *TraceInterface::hasCallTy(LLVMContext &C) {
  return FunctionType::get(
      Type::getInt1Ty(C), {PointerType::getInt8PtrTy(C), stringType(C)}, false);
}

FunctionType *TraceInterface::hasChoiceTy(LLVMContext &C) {
  return FunctionType::get(
      Type::getInt1Ty(C), {PointerType::getInt8PtrTy(C), stringType(C)}, false);
}

StaticTraceInterface::StaticTraceInterface(Module *M)
    : TraceInterface(M->getContext()) {
  for (auto &&F : M->functions()) {
    if (F.getName().contains("__enzyme_newtrace")) {
      assert(F.getFunctionType() == newTraceTy());
      newTraceFunction = &F;
    } else if (F.getName().contains("__enzyme_freetrace")) {
      assert(F.getFunctionType() == freeTraceTy());
      freeTraceFunction = &F;
    } else if (F.getName().contains("__enzyme_get_trace")) {
      assert(F.getFunctionType() == getTraceTy());
      getTraceFunction = &F;
    } else if (F.getName().contains("__enzyme_get_choice")) {
      assert(F.getFunctionType() == getChoiceTy());
      getChoiceFunction = &F;
    } else if (F.getName().contains("__enzyme_insert_call")) {
      assert(F.getFunctionType() == insertCallTy());
      insertCallFunction = &F;
    } else if (F.getName().contains("__enzyme_insert_choice")) {
      assert(F.getFunctionType() == insertChoiceTy());
      insertChoiceFunction = &F;
    } else if (F.getName().contains("__enzyme_has_call")) {
      assert(F.getFunctionType() == hasCallTy());
      hasCallFunction = &F;
    } else if (F.getName().contains("__enzyme_has_choice")) {
      assert(F.getFunctionType() == hasChoiceTy());
      hasChoiceFunction = &F;
    } else if (F.getName().contains(sampleFunctionName)) {
      assert(F.getFunctionType()->getNumParams() >= 3);
      sampleFunction = &F;
    }
  }

  assert(newTraceFunction != nullptr && freeTraceFunction != nullptr &&
         getTraceFunction != nullptr && getChoiceFunction != nullptr &&
         insertCallFunction != nullptr && insertChoiceFunction != nullptr &&
         hasCallFunction != nullptr && hasChoiceFunction != nullptr &&
         sampleFunction != nullptr);
}

// implemented by enzyme
Function *StaticTraceInterface::getSampleFunction() { return sampleFunction; }

// user implemented
Value *StaticTraceInterface::getTrace() { return getTraceFunction; }
Value *StaticTraceInterface::getChoice() { return getChoiceFunction; }
Value *StaticTraceInterface::insertCall() { return insertCallFunction; }
Value *StaticTraceInterface::insertChoice() { return insertChoiceFunction; }
Value *StaticTraceInterface::newTrace() { return newTraceFunction; }
Value *StaticTraceInterface::freeTrace() { return freeTraceFunction; }
Value *StaticTraceInterface::hasCall() { return hasCallFunction; }
Value *StaticTraceInterface::hasChoice() { return hasChoiceFunction; }

DynamicTraceInterface::DynamicTraceInterface(Value *dynamicInterface,
                                             Function *F)
    : TraceInterface(F->getContext()), dynamicInterface(dynamicInterface),
      F(F) {

  for (auto &&interface_func : F->getParent()->functions()) {
    if (interface_func.getName().contains(TraceInterface::sampleFunctionName)) {
      assert(interface_func.getFunctionType()->getNumParams() >= 3);
      sampleFunction = &interface_func;
    }
  }

  assert(sampleFunction);
}

// implemented by enzyme
Function *DynamicTraceInterface::getSampleFunction() { return sampleFunction; }

// user implemented
Value *DynamicTraceInterface::getTrace() {
  if (getTraceFunction)
    return getTraceFunction;

  IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());

  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(0));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  return getTraceFunction = Builder.CreatePointerCast(
             load, PointerType::getUnqual(getTraceTy()), "get_trace");
}

Value *DynamicTraceInterface::getChoice() {
  if (getChoiceFunction)
    return getChoiceFunction;

  IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(1));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  return getChoiceFunction = Builder.CreatePointerCast(
             load, PointerType::getUnqual(getChoiceTy()), "get_choice");
}

Value *DynamicTraceInterface::insertCall() {
  if (insertCallFunction)
    return insertCallFunction;

  IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(2));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  return insertCallFunction = Builder.CreatePointerCast(
             load, PointerType::getUnqual(insertCallTy()), "insert_call");
}

Value *DynamicTraceInterface::insertChoice() {
  if (insertChoiceFunction)
    return insertChoiceFunction;

  IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(3));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  return insertChoiceFunction = Builder.CreatePointerCast(
             load, PointerType::getUnqual(insertChoiceTy()), "insert_choice");
}

Value *DynamicTraceInterface::newTrace() {
  if (newTraceFunction)
    return newTraceFunction;

  IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(4));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  return newTraceFunction = Builder.CreatePointerCast(
             load, PointerType::getUnqual(newTraceTy()), "new_trace");
}

Value *DynamicTraceInterface::freeTrace() {
  if (freeTraceFunction)
    return freeTraceFunction;

  IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(5));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  return freeTraceFunction = Builder.CreatePointerCast(
             load, PointerType::getUnqual(freeTraceTy()), "free_trace");
}

Value *DynamicTraceInterface::hasCall() {
  if (hasCallFunction)
    return hasCallFunction;

  IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(6));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  return hasCallFunction = Builder.CreatePointerCast(
             load, PointerType::getUnqual(hasCallTy()), "has_call");
}

Value *DynamicTraceInterface::hasChoice() {
  if (hasChoiceFunction)
    return hasChoiceFunction;

  IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(7));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  return hasChoiceFunction = Builder.CreatePointerCast(
             load, PointerType::getUnqual(hasChoiceTy()), "has_choice");
}
