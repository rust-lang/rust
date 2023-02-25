#ifndef TraceInterface_h
#define TraceInterface_h

#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

class TraceInterface {
private:
  llvm::LLVMContext &C;

public:
  TraceInterface(llvm::LLVMContext &C) : C(C) {}

  virtual ~TraceInterface() = default;

public:
  // implemented by enzyme
  virtual llvm::Function *getSampleFunction() = 0;
  static constexpr const char sampleFunctionName[] = "__enzyme_sample";

  // user implemented
  virtual llvm::Value *getTrace() = 0;
  virtual llvm::Value *getChoice() = 0;
  virtual llvm::Value *insertCall() = 0;
  virtual llvm::Value *insertChoice() = 0;
  virtual llvm::Value *newTrace() = 0;
  virtual llvm::Value *freeTrace() = 0;
  virtual llvm::Value *hasCall() = 0;
  virtual llvm::Value *hasChoice() = 0;

public:
  static llvm::IntegerType *sizeType(llvm::LLVMContext &C);
  static llvm::Type *stringType(llvm::LLVMContext &C);

public:
  llvm::FunctionType *getTraceTy();
  llvm::FunctionType *getChoiceTy();
  llvm::FunctionType *insertCallTy();
  llvm::FunctionType *insertChoiceTy();
  llvm::FunctionType *newTraceTy();
  llvm::FunctionType *freeTraceTy();
  llvm::FunctionType *hasCallTy();
  llvm::FunctionType *hasChoiceTy();

  static llvm::FunctionType *getTraceTy(llvm::LLVMContext &C);

  static llvm::FunctionType *getChoiceTy(llvm::LLVMContext &C);

  static llvm::FunctionType *insertCallTy(llvm::LLVMContext &C);

  static llvm::FunctionType *insertChoiceTy(llvm::LLVMContext &C);

  static llvm::FunctionType *newTraceTy(llvm::LLVMContext &C);

  static llvm::FunctionType *freeTraceTy(llvm::LLVMContext &C);

  static llvm::FunctionType *hasCallTy(llvm::LLVMContext &C);

  static llvm::FunctionType *hasChoiceTy(llvm::LLVMContext &C);
};

class StaticTraceInterface final : public TraceInterface {
private:
  llvm::Function *sampleFunction = nullptr;
  // user implemented
  llvm::Function *getTraceFunction = nullptr;
  llvm::Function *getChoiceFunction = nullptr;
  llvm::Function *insertCallFunction = nullptr;
  llvm::Function *insertChoiceFunction = nullptr;
  llvm::Function *newTraceFunction = nullptr;
  llvm::Function *freeTraceFunction = nullptr;
  llvm::Function *hasCallFunction = nullptr;
  llvm::Function *hasChoiceFunction = nullptr;

public:
  StaticTraceInterface(llvm::Module *M);

  ~StaticTraceInterface() = default;

public:
  // implemented by enzyme
  llvm::Function *getSampleFunction();

  // user implemented
  llvm::Value *getTrace();
  llvm::Value *getChoice();
  llvm::Value *insertCall();
  llvm::Value *insertChoice();
  llvm::Value *newTrace();
  llvm::Value *freeTrace();
  llvm::Value *hasCall();
  llvm::Value *hasChoice();
};

class DynamicTraceInterface final : public TraceInterface {
private:
  llvm::Function *sampleFunction = nullptr;
  llvm::Value *dynamicInterface;
  llvm::Function *F;

private:
  llvm::Value *getTraceFunction = nullptr;
  llvm::Value *getChoiceFunction = nullptr;
  llvm::Value *insertCallFunction = nullptr;
  llvm::Value *insertChoiceFunction = nullptr;
  llvm::Value *newTraceFunction = nullptr;
  llvm::Value *freeTraceFunction = nullptr;
  llvm::Value *hasCallFunction = nullptr;
  llvm::Value *hasChoiceFunction = nullptr;

public:
  DynamicTraceInterface(llvm::Value *dynamicInterface, llvm::Function *F);

  ~DynamicTraceInterface() = default;

public:
  // implemented by enzyme
  llvm::Function *getSampleFunction();

  // user implemented
  llvm::Value *getTrace();

  llvm::Value *getChoice();

  llvm::Value *insertCall();

  llvm::Value *insertChoice();

  llvm::Value *newTrace();

  llvm::Value *freeTrace();

  llvm::Value *hasCall();

  llvm::Value *hasChoice();
};

#endif
