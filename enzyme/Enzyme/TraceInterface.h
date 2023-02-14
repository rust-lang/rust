#ifndef TraceInterface_h
#define TraceInterface_h

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

using namespace llvm;

class TraceInterface {
private:
  LLVMContext &C;

public:
  TraceInterface(LLVMContext &C) : C(C) {}

  virtual ~TraceInterface() = default;

public:
  // implemented by enzyme
  virtual Function *getSampleFunction() = 0;
  static constexpr const char sampleFunctionName[] = "__enzyme_sample";

  // user implemented
  virtual Value *getTrace() = 0;
  virtual Value *getChoice() = 0;
  virtual Value *insertCall() = 0;
  virtual Value *insertChoice() = 0;
  virtual Value *newTrace() = 0;
  virtual Value *freeTrace() = 0;
  virtual Value *hasCall() = 0;
  virtual Value *hasChoice() = 0;

public:
  static IntegerType *sizeType(LLVMContext &C) {
    return IntegerType::getInt64Ty(C);
  }
  static Type *stringType(LLVMContext &C) {
    return IntegerType::getInt8PtrTy(C);
  }

public:
  FunctionType *getTraceTy() { return getTraceTy(C); }
  FunctionType *getChoiceTy() { return getChoiceTy(C); }
  FunctionType *insertCallTy() { return insertCallTy(C); }
  FunctionType *insertChoiceTy() { return insertChoiceTy(C); }
  FunctionType *newTraceTy() { return newTraceTy(C); }
  FunctionType *freeTraceTy() { return freeTraceTy(C); }
  FunctionType *hasCallTy() { return hasCallTy(C); }
  FunctionType *hasChoiceTy() { return hasChoiceTy(C); }

  static FunctionType *getTraceTy(LLVMContext &C) {
    return FunctionType::get(PointerType::getInt8PtrTy(C),
                             {PointerType::getInt8PtrTy(C), stringType(C)},
                             false);
  }

  static FunctionType *getChoiceTy(LLVMContext &C) {
    return FunctionType::get(sizeType(C),
                             {PointerType::getInt8PtrTy(C), stringType(C),
                              PointerType::getInt8PtrTy(C), sizeType(C)},
                             false);
  }

  static FunctionType *insertCallTy(LLVMContext &C) {
    return FunctionType::get(Type::getVoidTy(C),
                             {PointerType::getInt8PtrTy(C), stringType(C),
                              PointerType::getInt8PtrTy(C)},
                             false);
  }

  static FunctionType *insertChoiceTy(LLVMContext &C) {
    return FunctionType::get(Type::getVoidTy(C),
                             {PointerType::getInt8PtrTy(C), stringType(C),
                              Type::getDoubleTy(C),
                              PointerType::getInt8PtrTy(C), sizeType(C)},
                             false);
  }

  static FunctionType *newTraceTy(LLVMContext &C) {
    return FunctionType::get(PointerType::getInt8PtrTy(C), {}, false);
  }

  static FunctionType *freeTraceTy(LLVMContext &C) {
    return FunctionType::get(Type::getVoidTy(C), {PointerType::getInt8PtrTy(C)},
                             false);
  }

  static FunctionType *hasCallTy(LLVMContext &C) {
    return FunctionType::get(Type::getInt1Ty(C),
                             {PointerType::getInt8PtrTy(C), stringType(C)},
                             false);
  }

  static FunctionType *hasChoiceTy(LLVMContext &C) {
    return FunctionType::get(Type::getInt1Ty(C),
                             {PointerType::getInt8PtrTy(C), stringType(C)},
                             false);
  }
};

class StaticTraceInterface final : public TraceInterface {
private:
  Function *sampleFunction = nullptr;
  // user implemented
  Function *getTraceFunction = nullptr;
  Function *getChoiceFunction = nullptr;
  Function *insertCallFunction = nullptr;
  Function *insertChoiceFunction = nullptr;
  Function *newTraceFunction = nullptr;
  Function *freeTraceFunction = nullptr;
  Function *hasCallFunction = nullptr;
  Function *hasChoiceFunction = nullptr;

public:
  StaticTraceInterface(Module *M) : TraceInterface(M->getContext()) {
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

  ~StaticTraceInterface() = default;

public:
  // implemented by enzyme
  Function *getSampleFunction() { return sampleFunction; }

  // user implemented
  Value *getTrace() { return getTraceFunction; }
  Value *getChoice() { return getChoiceFunction; }
  Value *insertCall() { return insertCallFunction; }
  Value *insertChoice() { return insertChoiceFunction; }
  Value *newTrace() { return newTraceFunction; }
  Value *freeTrace() { return freeTraceFunction; }
  Value *hasCall() { return hasCallFunction; }
  Value *hasChoice() { return hasChoiceFunction; }
};

class DynamicTraceInterface final : public TraceInterface {
private:
  Function *sampleFunction = nullptr;
  Value *dynamicInterface;
  Function *F;

private:
  Value *getTraceFunction = nullptr;
  Value *getChoiceFunction = nullptr;
  Value *insertCallFunction = nullptr;
  Value *insertChoiceFunction = nullptr;
  Value *newTraceFunction = nullptr;
  Value *freeTraceFunction = nullptr;
  Value *hasCallFunction = nullptr;
  Value *hasChoiceFunction = nullptr;

public:
  DynamicTraceInterface(Value *dynamicInterface, Function *F)
      : TraceInterface(F->getContext()), dynamicInterface(dynamicInterface),
        F(F) {

    for (auto &&interface_func : F->getParent()->functions()) {
      if (interface_func.getName().contains(
              TraceInterface::sampleFunctionName)) {
        assert(interface_func.getFunctionType()->getNumParams() >= 3);
        sampleFunction = &interface_func;
      }
    }

    assert(sampleFunction);
  }

  ~DynamicTraceInterface() = default;

public:
  // implemented by enzyme
  Function *getSampleFunction() { return sampleFunction; }

  // user implemented
  Value *getTrace() {
    if (getTraceFunction)
      return getTraceFunction;

    IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());

    auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(),
                                         dynamicInterface, Builder.getInt32(0));
    auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
    return getTraceFunction = Builder.CreatePointerCast(
               load, PointerType::getUnqual(getTraceTy()), "get_trace");
  }

  Value *getChoice() {
    if (getChoiceFunction)
      return getChoiceFunction;

    IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(),
                                         dynamicInterface, Builder.getInt32(1));
    auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
    return getChoiceFunction = Builder.CreatePointerCast(
               load, PointerType::getUnqual(getChoiceTy()), "get_choice");
  }

  Value *insertCall() {
    if (insertCallFunction)
      return insertCallFunction;

    IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(),
                                         dynamicInterface, Builder.getInt32(2));
    auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
    return insertCallFunction = Builder.CreatePointerCast(
               load, PointerType::getUnqual(insertCallTy()), "insert_call");
  }

  Value *insertChoice() {
    if (insertChoiceFunction)
      return insertChoiceFunction;

    IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(),
                                         dynamicInterface, Builder.getInt32(3));
    auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
    return insertChoiceFunction = Builder.CreatePointerCast(
               load, PointerType::getUnqual(insertChoiceTy()), "insert_choice");
  }

  Value *newTrace() {
    if (newTraceFunction)
      return newTraceFunction;

    IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(),
                                         dynamicInterface, Builder.getInt32(4));
    auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
    return newTraceFunction = Builder.CreatePointerCast(
               load, PointerType::getUnqual(newTraceTy()), "new_trace");
  }

  Value *freeTrace() {
    if (freeTraceFunction)
      return freeTraceFunction;

    IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(),
                                         dynamicInterface, Builder.getInt32(5));
    auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
    return freeTraceFunction = Builder.CreatePointerCast(
               load, PointerType::getUnqual(freeTraceTy()), "free_trace");
  }

  Value *hasCall() {
    if (hasCallFunction)
      return hasCallFunction;

    IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(),
                                         dynamicInterface, Builder.getInt32(6));
    auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
    return hasCallFunction = Builder.CreatePointerCast(
               load, PointerType::getUnqual(hasCallTy()), "has_call");
  }

  Value *hasChoice() {
    if (hasChoiceFunction)
      return hasChoiceFunction;

    IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(),
                                         dynamicInterface, Builder.getInt32(7));
    auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
    return hasChoiceFunction = Builder.CreatePointerCast(
               load, PointerType::getUnqual(hasChoiceTy()), "has_choice");
  }
};

#endif
