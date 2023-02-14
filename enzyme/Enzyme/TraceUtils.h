#ifndef TraceUtils_h
#define TraceUtils_h

#include <deque>

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueMap.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "TraceInterface.h"

using namespace llvm;

class TraceUtils {

private:
  TraceInterface *interface;
  Value *dynamic_interface = nullptr;
  Instruction *trace;
  Value *observations = nullptr;

public:
  ProbProgMode mode;
  Function *newFunc;
  Function *oldFunc;

public:
  ValueToValueMapTy originalToNewFn;
  SmallPtrSetImpl<Function *> &generativeFunctions;

public:
  TraceUtils(ProbProgMode mode, bool has_dynamic_interface, Function *newFunc,
             Function *oldFunc, ValueToValueMapTy vmap,
             SmallPtrSetImpl<Function *> &generativeFunctions)
      : mode(mode), newFunc(newFunc), oldFunc(oldFunc),
        generativeFunctions(generativeFunctions) {
    originalToNewFn.insert(vmap.begin(), vmap.end());
    originalToNewFn.getMDMap() = vmap.getMDMap();
  }

  TraceUtils(ProbProgMode mode, bool has_dynamic_interface, Function *F,
             SmallPtrSetImpl<Function *> &generativeFunctions)
      : mode(mode), oldFunc(F), generativeFunctions(generativeFunctions) {
    auto &Context = oldFunc->getContext();

    FunctionType *orig_FTy = oldFunc->getFunctionType();
    Type *traceType =
        TraceInterface::getTraceTy(F->getContext())->getReturnType();

    SmallVector<Type *, 4> params;

    for (unsigned i = 0; i < orig_FTy->getNumParams(); ++i) {
      params.push_back(orig_FTy->getParamType(i));
    }

    if (has_dynamic_interface)
      params.push_back(
          PointerType::getUnqual(PointerType::getInt8PtrTy(Context)));

    if (mode == ProbProgMode::Condition)
      params.push_back(traceType);

    Type *RetTy = traceType;
    if (!oldFunc->getReturnType()->isVoidTy())
      RetTy = StructType::get(Context, {oldFunc->getReturnType(), traceType});

    FunctionType *FTy = FunctionType::get(RetTy, params, oldFunc->isVarArg());

    Twine Name = (mode == ProbProgMode::Condition ? "condition_" : "trace_") +
                 Twine(oldFunc->getName());

    newFunc = Function::Create(FTy, Function::LinkageTypes::InternalLinkage,
                               Name, oldFunc->getParent());

    auto DestArg = newFunc->arg_begin();
    auto SrcArg = oldFunc->arg_begin();

    for (unsigned i = 0; i < orig_FTy->getNumParams(); ++i) {
      Argument *arg = SrcArg;
      originalToNewFn[arg] = DestArg;
      DestArg->setName(arg->getName());
      DestArg++;
      SrcArg++;
    }

    if (has_dynamic_interface) {
      auto arg = newFunc->arg_end() - (1 + (mode == ProbProgMode::Condition));
      dynamic_interface = arg;
      arg->setName("interface");
      arg->addAttr(Attribute::ReadOnly);
      arg->addAttr(Attribute::NoCapture);
    }

    if (mode == ProbProgMode::Condition) {
      auto arg = newFunc->arg_end() - 1;
      observations = arg;
      arg->setName("observations");
      if (oldFunc->getReturnType()->isVoidTy())
        arg->addAttr(Attribute::Returned);
    }

    SmallVector<ReturnInst *, 4> Returns;
#if LLVM_VERSION_MAJOR >= 13
    CloneFunctionInto(newFunc, oldFunc, originalToNewFn,
                      CloneFunctionChangeType::LocalChangesOnly, Returns, "",
                      nullptr);
#else
    CloneFunctionInto(newFunc, oldFunc, originalToNewFn, true, Returns, "",
                      nullptr);
#endif

    newFunc->setLinkage(Function::LinkageTypes::InternalLinkage);

    if (has_dynamic_interface) {
      interface = new DynamicTraceInterface(dynamic_interface, newFunc);
    } else {
      interface = new StaticTraceInterface(F->getParent());
    }

    // Create trace for current function

    IRBuilder<> Builder(
        newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    Builder.SetCurrentDebugLocation(oldFunc->getEntryBlock()
                                        .getFirstNonPHIOrDbgOrLifetime()
                                        ->getDebugLoc());

    trace = CreateTrace(Builder);

    // Replace returns with ret trace

    SmallVector<ReturnInst *, 3> toReplace;
    for (auto &&BB : *newFunc) {
      for (auto &&Inst : BB) {
        if (auto Ret = dyn_cast<ReturnInst>(&Inst)) {
          toReplace.push_back(Ret);
        }
      }
    }

    for (auto Ret : toReplace) {
      IRBuilder<> Builder(Ret);
      if (Ret->getReturnValue()) {
        Value *retvals[2] = {Ret->getReturnValue(), trace};
        Builder.CreateAggregateRet(retvals, 2);
      } else {
        Builder.CreateRet(trace);
      }
      Ret->eraseFromParent();
    }
  };

  ~TraceUtils() { delete interface; }

public:
  TraceInterface *getTraceInterface() { return interface; }

  Value *getDynamicTraceInterface() { return dynamic_interface; }

  bool hasDynamicTraceInterface() { return dynamic_interface != nullptr; }

  Value *getTrace() { return trace; }

  CallInst *CreateTrace(IRBuilder<> &Builder, const Twine &Name = "trace") {
    return Builder.CreateCall(interface->newTraceTy(), interface->newTrace(),
                              {}, Name);
  }

  CallInst *InsertChoice(IRBuilder<> &Builder, Value *address, Value *score,
                         Value *choice) {
    Type *size_type = interface->getChoiceTy()->getParamType(3);
    auto choicesize = choice->getType()->getPrimitiveSizeInBits();

    Value *retval;
    if (choice->getType()->isPointerTy()) {
      retval = Builder.CreatePointerCast(choice, Builder.getInt8PtrTy());
    } else {
      auto M = interface->getSampleFunction()->getParent();
      auto &DL = M->getDataLayout();
      auto pointersize = DL.getPointerSizeInBits();
      if (choicesize <= pointersize) {
        auto cast = Builder.CreateBitCast(
            choice, IntegerType::get(M->getContext(), choicesize));
        cast = choicesize == pointersize
                   ? cast
                   : Builder.CreateZExt(cast, Builder.getIntPtrTy(DL));
        retval = Builder.CreateIntToPtr(cast, Builder.getInt8PtrTy());
      } else {
        IRBuilder<> AllocaBuilder(
            newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
        auto alloca = AllocaBuilder.CreateAlloca(choice->getType(), nullptr,
                                                 choice->getName() + ".ptr");
        Builder.CreateStore(choice, alloca);
        retval = alloca;
      }
    }

    Value *args[] = {trace, address, score, retval,
                     ConstantInt::get(size_type, choicesize / 8)};

    auto call = Builder.CreateCall(interface->insertChoiceTy(),
                                   interface->insertChoice(), args);
    call->addParamAttr(1, Attribute::ReadOnly);
    call->addParamAttr(1, Attribute::NoCapture);
    return call;
  }

  CallInst *InsertCall(IRBuilder<> &Builder, Value *address, Value *subtrace) {
    Value *args[] = {trace, address, subtrace};

    auto call = Builder.CreateCall(interface->insertCallTy(),
                                   interface->insertCall(), args);
    call->addParamAttr(1, Attribute::ReadOnly);
    call->addParamAttr(1, Attribute::NoCapture);
    return call;
  }

  CallInst *GetTrace(IRBuilder<> &Builder, Value *address,
                     const Twine &Name = "") {
    assert(address->getType()->isPointerTy());

    Value *args[] = {observations, address};

    auto call = Builder.CreateCall(interface->getTraceTy(),
                                   interface->getTrace(), args, Name);
    call->addParamAttr(1, Attribute::ReadOnly);
    call->addParamAttr(1, Attribute::NoCapture);
    return call;
  }

  Instruction *GetChoice(IRBuilder<> &Builder, Value *address, Type *choiceType,
                         const Twine &Name = "") {
    IRBuilder<> AllocaBuilder(
        newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    AllocaInst *store_dest =
        AllocaBuilder.CreateAlloca(choiceType, nullptr, Name + ".ptr");
    auto preallocated_size = choiceType->getPrimitiveSizeInBits() / 8;
    Type *size_type = interface->getChoiceTy()->getParamType(3);

    Value *args[] = {
        observations, address,
        Builder.CreatePointerCast(store_dest, Builder.getInt8PtrTy()),
        ConstantInt::get(size_type, preallocated_size)};

    auto call = Builder.CreateCall(
        interface->getChoiceTy(), interface->getChoice(), args, Name + ".size");
    call->addParamAttr(1, Attribute::ReadOnly);
    call->addParamAttr(1, Attribute::NoCapture);
    return Builder.CreateLoad(choiceType, store_dest, "from.trace." + Name);
  }

  Instruction *HasChoice(IRBuilder<> &Builder, Value *address,
                         const Twine &Name = "") {
    Value *args[]{observations, address};

    auto call = Builder.CreateCall(interface->hasChoiceTy(),
                                   interface->hasChoice(), args, Name);
    call->addParamAttr(1, Attribute::ReadOnly);
    call->addParamAttr(1, Attribute::NoCapture);
    return call;
  }

  Instruction *HasCall(IRBuilder<> &Builder, Value *address,
                       const Twine &Name = "") {
    Value *args[]{observations, address};

    auto call = Builder.CreateCall(interface->hasCallTy(), interface->hasCall(),
                                   args, Name);
    call->addParamAttr(1, Attribute::ReadOnly);
    call->addParamAttr(1, Attribute::NoCapture);
    return call;
  }
};

#endif /* TraceUtils_h */
