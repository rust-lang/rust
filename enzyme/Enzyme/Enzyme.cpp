/*
 * Enzyme.cpp - Lower autodiff intrinsic
 *
 * Copyright (C) 2019 William S. Moses (enzyme@wsmoses.com) - All Rights Reserved
 *
 * For commercial use of this code please contact the author(s) above.
 *
 * For research use of the code please use the following citation.
 *
 * \misc{mosesenzyme,
    author = {William S. Moses, Tim Kaler},
    title = {Enzyme: LLVM Automatic Differentiation},
    year = {2019},
    howpublished = {\url{https://github.com/wsmoses/Enzyme/}},
    note = {commit xxxxxxx}
 */

#include "SCEV/ScalarEvolutionExpander.h"

#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"

#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"

#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"

#include "ActiveVariable.h"
#include "EnzymeLogic.h"
#include "GradientUtils.h"
#include "Utils.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "lower-autodiff-intrinsic"

void HandleAutoDiff(CallInst *CI, TargetLibraryInfo &TLI, AAResults &AA) {//, LoopInfo& LI, DominatorTree& DT) {

  Value* fn = CI->getArgOperand(0);

  while (auto ci = dyn_cast<CastInst>(fn)) {
    fn = ci->getOperand(0);
  }
  while (auto ci = dyn_cast<BlockAddress>(fn)) {
    fn = ci->getFunction();
  }
  while (auto ci = dyn_cast<ConstantExpr>(fn)) {
    fn = ci->getOperand(0);
  }
  auto FT = cast<Function>(fn)->getFunctionType();
  assert(fn);

  if (enzyme_print)
      llvm::errs() << "prefn:\n" << *fn << "\n";

  std::set<unsigned> constants;
  SmallVector<Value*,2> args;

  unsigned truei = 0;
  IRBuilder<> Builder(CI);

  for(unsigned i=1; i<CI->getNumArgOperands(); i++) {
    Value* res = CI->getArgOperand(i);

    assert(truei < FT->getNumParams());
    auto PTy = FT->getParamType(truei);
    DIFFE_TYPE ty = DIFFE_TYPE::CONSTANT;

    if (auto av = dyn_cast<MetadataAsValue>(res)) {
        auto MS = cast<MDString>(av->getMetadata())->getString();
        if (MS == "diffe_dup") {
            ty = DIFFE_TYPE::DUP_ARG;
        } else if(MS == "diffe_out") {
            llvm::errs() << "saw metadata for diffe_out\n";
            ty = DIFFE_TYPE::OUT_DIFF;
        } else if (MS == "diffe_const") {
            ty = DIFFE_TYPE::CONSTANT;
        } else {
            assert(0 && "illegal diffe metadata string");
        }
        i++;
        res = CI->getArgOperand(i);
    } else
      ty = whatType(PTy);

    //llvm::errs() << "considering arg " << *res << " argnum " << truei << "\n";

    if (ty == DIFFE_TYPE::CONSTANT)
      constants.insert(truei);

    assert(truei < FT->getNumParams());
    if (PTy != res->getType()) {
        if (auto ptr = dyn_cast<PointerType>(res->getType())) {
            if (auto PT = dyn_cast<PointerType>(PTy)) {
                if (ptr->getAddressSpace() != PT->getAddressSpace()) {
                    res = Builder.CreateAddrSpaceCast(res, PointerType::get(ptr->getElementType(), PT->getAddressSpace()));
                    assert(res);
                    assert(PTy);
                    assert(FT);
                    llvm::errs() << "Warning cast(1) __builtin_autodiff argument " << i << " " << *res <<"|" << *res->getType()<< " to argument " << truei << " " << *PTy << "\n" << "orig: " << *FT << "\n";
                }
            }
        }
      if (!res->getType()->canLosslesslyBitCastTo(PTy)) {
        llvm::errs() << "Cannot cast(1) __builtin_autodiff argument " << i << " " << *res << "|"<< *res->getType() << " to argument " << truei << " " << *PTy << "\n" << "orig: " << *FT << "\n";
        report_fatal_error("Illegal cast(1)");
      }
      res = Builder.CreateBitCast(res, PTy);
    }

    args.push_back(res);
    if (ty == DIFFE_TYPE::DUP_ARG) {
      i++;

      Value* res = CI->getArgOperand(i);
      if (PTy != res->getType()) {
        if (auto ptr = dyn_cast<PointerType>(res->getType())) {
            if (auto PT = dyn_cast<PointerType>(PTy)) {
                if (ptr->getAddressSpace() != PT->getAddressSpace()) {
                    res = Builder.CreateAddrSpaceCast(res, PointerType::get(ptr->getElementType(), PT->getAddressSpace()));
                    assert(res);
                    assert(PTy);
                    assert(FT);
                    llvm::errs() << "Warning cast(2) __builtin_autodiff argument " << i << " " << *res <<"|" << *res->getType()<< " to argument " << truei << " " << *PTy << "\n" << "orig: " << *FT << "\n";
                }
            }
        }
        if (!res->getType()->canLosslesslyBitCastTo(PTy)) {
          assert(res);
          assert(res->getType());
          assert(PTy);
          assert(FT);
          llvm::errs() << "Cannot cast(2) __builtin_autodiff argument " << i << " " << *res <<"|" << *res->getType()<< " to argument " << truei << " " << *PTy << "\n" << "orig: " << *FT << "\n";
          report_fatal_error("Illegal cast(2)");
        }
        res = Builder.CreateBitCast(res, PTy);
      }
      args.push_back(res);
    }

    truei++;
  }

  //for(auto a : constants) {
  //    llvm::errs() << "constant argnum: " << a << "\n";
  //}

  bool differentialReturn = cast<Function>(fn)->getReturnType()->isFPOrFPVectorTy();

  std::map<Argument*, bool> volatile_args;
  std::map<Argument*, ValueData> type_args;
  for(auto &a : cast<Function>(fn)->args()) {
    volatile_args[&a] = false;
    ValueData dt;
    if (a.getType()->isFPOrFPVectorTy()) {
        dt = DataType(a.getType()->getScalarType());
    } else if (a.getType()->isPointerTy()) {
        auto et = cast<PointerType>(a.getType())->getElementType();
        if (et->isFPOrFPVectorTy()) {
            dt = ValueData(DataType(et->getScalarType())).Only({-1});
        } else if (et->isPointerTy()) {
            dt = ValueData(DataType(IntType::Pointer)).Only({-1});
        }
    }
    type_args.insert(std::pair<Argument*, ValueData>(&a, dt));
  }

  TypeAnalysis TA;
  type_args = TA.analyzeFunction(type_args, cast<Function>(fn)).getAnalyzedTypeInfo();

  auto newFunc = CreatePrimalAndGradient(cast<Function>(fn), constants, TLI, TA, AA, /*should return*/false, differentialReturn, /*dretPtr*/false, /*topLevel*/true, /*addedType*/nullptr, type_args, volatile_args, /*index mapping*/nullptr); //llvm::Optional<std::map<std::pair<Instruction*, std::string>, unsigned>>({}));

  if (differentialReturn)
    args.push_back(ConstantFP::get(cast<Function>(fn)->getReturnType(), 1.0));
  assert(newFunc);

  if (enzyme_print)
    llvm::errs() << "postfn:\n" << *newFunc << "\n";
  Builder.setFastMathFlags(getFast());

  CallInst* diffret = cast<CallInst>(Builder.CreateCall(newFunc, args));
  diffret->setCallingConv(CI->getCallingConv());
  diffret->setDebugLoc(CI->getDebugLoc());
  if (!diffret->getType()->isEmptyTy()) {
    unsigned idxs[] = {0};
    auto diffreti = Builder.CreateExtractValue(diffret, idxs);
    CI->replaceAllUsesWith(diffreti);
  } else {
    CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
  }
  CI->eraseFromParent();
}

static bool lowerEnzymeCalls(Function &F, TargetLibraryInfo &TLI, AAResults &AA) {//, LoopInfo& LI, DominatorTree& DT) {

  bool Changed = false;

reset:
  for (BasicBlock &BB : F) {

    for (auto BI = BB.rbegin(), BE = BB.rend(); BI != BE; BI++) {
      CallInst *CI = dyn_cast<CallInst>(&*BI);
      if (!CI) continue;

      Function *Fn = CI->getCalledFunction();

      if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledValue())) {
        if (castinst->isCast())
            if (auto fn = dyn_cast<Function>(castinst->getOperand(0)))
                Fn = fn;
      }

      if (Fn && ( Fn->getName() == "__enzyme_autodiff" || Fn->getName().startswith("__enzyme_autodiff")) ) {
        HandleAutoDiff(CI, TLI, AA);//, LI, DT);
        Changed = true;
        goto reset;
      }
    }
  }

  return Changed;
}

namespace {

class Enzyme : public ModulePass {
public:
  static char ID;
  Enzyme() : ModulePass(ID) {
    //initializeLowerAutodiffIntrinsicPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<GlobalsAAWrapperPass>();
    AU.addRequired<BasicAAWrapperPass>();
    //AU.addRequiredID(LCSSAID);

    //LoopInfo is required to ensure that all loops have preheaders
    //AU.addRequired<LoopInfoWrapperPass>();
    
    //AU.addRequiredID(llvm::LoopSimplifyID);//<LoopSimplifyWrapperPass>();
  }

  bool runOnModule(Module &M) override {
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
    //auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto &G_AA = getAnalysis<GlobalsAAWrapperPass>().getResult();

    //llvm::errs() << "G_AA: " << &G_AA << "\n";
    //AAResults AA(TLI);
    //AA.addAAResult(B_AA);
    //AA.addAAResult(G_AA);

    /*
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    */

    bool changed = false;
    for(Function& F: M) {
        if (F.empty()) continue;
        
        AAResults AA(TLI);
        //auto &B_AA = getAnalysis<BasicAAWrapperPass>().getResult();
        //AA.addAAResult(B_AA);
        AA.addAAResult(G_AA);

        //auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
        //auto &LI = getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
        changed |= lowerEnzymeCalls(F, TLI, AA);
    }
    return changed;
  }
};

}

char Enzyme::ID = 0;

static RegisterPass<Enzyme> X("enzyme", "Enzyme Pass");

ModulePass *createEnzymePass() {
  return new Enzyme();
}

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include "llvm/IR/LegacyPassManager.h"

extern "C" void AddEnzymePass(LLVMPassManagerRef PM) {
    unwrap(PM)->add(createEnzymePass());
}
