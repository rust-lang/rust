#ifndef TraceGenerator_h
#define TraceGenerator_h

#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#if LLVM_VERSION_MAJOR >= 8
#include "llvm/IR/InstrTypes.h"
#endif
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "FunctionUtils.h"
#include "TraceInterface.h"
#include "TraceUtils.h"
#include "Utils.h"

using namespace llvm;

class TraceGenerator final : public llvm::InstVisitor<TraceGenerator> {
private:
  EnzymeLogic &Logic;
  TraceUtils *const tutils;
  ProbProgMode mode = tutils->mode;

public:
  TraceGenerator(EnzymeLogic &Logic, TraceUtils *const tutils)
      : Logic(Logic), tutils(tutils){};

  void visitCallInst(llvm::CallInst &call) {

    if (!tutils->generativeFunctions.count(call.getCalledFunction()))
      return;

    CallInst *new_call = dyn_cast<CallInst>(tutils->originalToNewFn[&call]);
    IRBuilder<> Builder(new_call);

    if (call.getCalledFunction() ==
        tutils->getTraceInterface()->getSampleFunction()) {
      Function *samplefn = GetFunctionFromValue(new_call->getArgOperand(0));
      Function *loglikelihoodfn =
          GetFunctionFromValue(new_call->getArgOperand(1));
      Value *address = new_call->getArgOperand(2);

      SmallVector<Value *, 2> sample_args;
      for (auto it = new_call->arg_begin() + 3; it != new_call->arg_end();
           it++) {
        sample_args.push_back(*it);
      }

      Value *choice;
      switch (mode) {
      case ProbProgMode::Trace: {
        choice = Builder.CreateCall(samplefn->getFunctionType(), samplefn,
                                    sample_args);
        break;
      }
      case ProbProgMode::Condition: {
        Instruction *hasChoice =
            tutils->HasChoice(Builder, address, "has.choice." + call.getName());
#if LLVM_VERSION_MAJOR >= 8
        Instruction *ThenTerm, *ElseTerm;
#else
        TerminatorInst *ThenTerm, *ElseTerm;
#endif
        Value *ThenChoice, *ElseChoice;
        SplitBlockAndInsertIfThenElse(hasChoice, new_call, &ThenTerm,
                                      &ElseTerm);

        new_call->getParent()->setName(hasChoice->getParent()->getName() +
                                       ".cntd");

        Builder.SetInsertPoint(ThenTerm);
        {
          ThenTerm->getParent()->setName("condition." + call.getName() +
                                         ".with.trace");
          ThenChoice = tutils->GetChoice(
              Builder, address, samplefn->getFunctionType()->getReturnType(),
              call.getName());
        }

        Builder.SetInsertPoint(ElseTerm);
        {
          ElseTerm->getParent()->setName("condition." + call.getName() +
                                         ".without.trace");

          auto choice =
              Builder.CreateCall(samplefn->getFunctionType(), samplefn,
                                 sample_args, "sample." + call.getName());
          ElseChoice = choice;
        }

        Builder.SetInsertPoint(new_call);
        auto phi = Builder.CreatePHI(new_call->getType(), 2);
        phi->addIncoming(ThenChoice, ThenTerm->getParent());
        phi->addIncoming(ElseChoice, ElseTerm->getParent());
        choice = phi;
        break;
      }
      }

      SmallVector<Value *, 3> loglikelihood_args = SmallVector(sample_args);
      loglikelihood_args.push_back(choice);
      auto score = Builder.CreateCall(loglikelihoodfn->getFunctionType(),
                                      loglikelihoodfn, loglikelihood_args,
                                      "likelihood." + call.getName());

      choice->takeName(new_call);
      tutils->InsertChoice(Builder, address, score, choice);

      new_call->replaceAllUsesWith(choice);
      new_call->eraseFromParent();
    } else {
      auto str = call.getName() + "." + call.getCalledFunction()->getName();
      auto address = Builder.CreateGlobalStringPtr(str.str());

      SmallVector<Value *, 2> args;
      for (auto it = new_call->arg_begin(); it != new_call->arg_end(); it++) {
        args.push_back(*it);
      }

      if (tutils->hasDynamicTraceInterface())
        args.push_back(tutils->getDynamicTraceInterface());

      Function *called = getFunctionFromCall(&call);
      assert(called);

      Function *samplefn =
          Logic.CreateTrace(called, tutils->generativeFunctions, tutils->mode,
                            tutils->hasDynamicTraceInterface());

      auto trace = tutils->CreateTrace(Builder);

      Instruction *tracecall;
      switch (mode) {
      case ProbProgMode::Trace: {
        SmallVector<Value *, 2> args_and_trace = SmallVector(args);
        args_and_trace.push_back(trace);
        tracecall =
            Builder.CreateCall(samplefn->getFunctionType(), samplefn,
                               args_and_trace, "trace." + called->getName());
        break;
      }
      case ProbProgMode::Condition: {
        Instruction *hasCall =
            tutils->HasCall(Builder, address, "has.call." + call.getName());
#if LLVM_VERSION_MAJOR >= 8
        Instruction *ThenTerm, *ElseTerm;
#else
        TerminatorInst *ThenTerm, *ElseTerm;
#endif
        Value *ElseTracecall, *ThenTracecall;
        SplitBlockAndInsertIfThenElse(hasCall, new_call, &ThenTerm, &ElseTerm);

        new_call->getParent()->setName(hasCall->getParent()->getName() +
                                       ".cntd");

        Builder.SetInsertPoint(ThenTerm);
        {
          ThenTerm->getParent()->setName("condition." + call.getName() +
                                         ".with.trace");
          SmallVector<Value *, 2> args_and_cond = SmallVector(args);
          auto observations = tutils->GetTrace(Builder, address,
                                               called->getName() + ".subtrace");
          args_and_cond.push_back(observations);
          args_and_cond.push_back(trace);
          ThenTracecall = Builder.CreateCall(samplefn->getFunctionType(),
                                             samplefn, args_and_cond,
                                             "condition." + called->getName());
        }

        Builder.SetInsertPoint(ElseTerm);
        {
          ElseTerm->getParent()->setName("condition." + call.getName() +
                                         ".without.trace");
          SmallVector<Value *, 2> args_and_null = SmallVector(args);
          auto observations = ConstantPointerNull::get(cast<PointerType>(
              tutils->getTraceInterface()->newTraceTy()->getReturnType()));
          args_and_null.push_back(observations);
          args_and_null.push_back(trace);
          ElseTracecall =
              Builder.CreateCall(samplefn->getFunctionType(), samplefn,
                                 args_and_null, "trace." + called->getName());
        }

        Builder.SetInsertPoint(new_call);
        auto phi = Builder.CreatePHI(
            samplefn->getFunctionType()->getReturnType(), 2, call.getName());
        phi->addIncoming(ThenTracecall, ThenTerm->getParent());
        phi->addIncoming(ElseTracecall, ElseTerm->getParent());
        tracecall = phi;
      }
      }

      tutils->InsertCall(Builder, address, trace);

      tracecall->takeName(new_call);
      new_call->replaceAllUsesWith(tracecall);
      new_call->eraseFromParent();
    }
  }
};

#endif /* TraceGenerator_h */
