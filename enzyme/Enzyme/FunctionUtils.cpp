/*
 * FunctionUtils.cpp
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

#include "FunctionUtils.h"

#include "EnzymeLogic.h"
#include "GradientUtils.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#if LLVM_VERSION_MAJOR > 6
#include "llvm/Analysis/PhiValues.h"
#endif
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#if LLVM_VERSION_MAJOR > 6
#include "llvm/Transforms/Scalar/InstSimplifyPass.h"
#endif
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/CorrelatedValuePropagation.h"
#include "llvm/Transforms/Scalar/LoopIdiomRecognize.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"

#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"

using namespace llvm;

static cl::opt<bool> enzyme_preopt(
            "enzyme_preopt", cl::init(true), cl::Hidden,
            cl::desc("Run enzyme preprocessing optimizations"));

static cl::opt<bool> autodiff_inline(
            "enzyme_inline", cl::init(false), cl::Hidden,
                cl::desc("Force inlining of autodiff"));

static cl::opt<int> autodiff_inline_count(
            "enzyme_inline_count", cl::init(10000), cl::Hidden,
                cl::desc("Limit of number of functions to inline"));

static bool promoteMemoryToRegister(Function &F, DominatorTree &DT,
                                     AssumptionCache &AC) {
   std::vector<AllocaInst *> Allocas;
   BasicBlock &BB = F.getEntryBlock(); // Get the entry node for the function
   bool Changed = false;

   while (true) {
     Allocas.clear();

     // Find allocas that are safe to promote, by looking at all instructions in
     // the entry node
     for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
       if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) // Is it an alloca?
         if (isAllocaPromotable(AI))
           Allocas.push_back(AI);

     if (Allocas.empty())
       break;

     PromoteMemToReg(Allocas, DT, &AC);
     Changed = true;
   }
   return Changed;
}

void forceRecursiveInlining(Function *NewF, const Function* F) {
   static int count = 0;
   remover:
     SmallPtrSet<Instruction*, 10> originalInstructions;
     for (inst_iterator I = inst_begin(NewF), E = inst_end(NewF); I != E; ++I) {
         originalInstructions.insert(&*I);
     }
        if (count >= autodiff_inline_count)
            return;
   for (inst_iterator I = inst_begin(NewF), E = inst_end(NewF); I != E; ++I)
     if (auto call = dyn_cast<CallInst>(&*I)) {
        //if (isconstantM(call, constants, nonconstant, returnvals, originalInstructions)) continue;
        if (call->getCalledFunction() == nullptr) continue;
        if (call->getCalledFunction()->empty()) continue;
        /*
        if (call->getCalledFunction()->hasFnAttribute(Attribute::NoInline)) {
            llvm::errs() << "can't inline noinline " << call->getCalledFunction()->getName() << "\n";
            continue;
        }
        */
        if (call->getCalledFunction()->hasFnAttribute(Attribute::ReturnsTwice)) continue;
        if (call->getCalledFunction() == F || call->getCalledFunction() == NewF) {
            llvm::errs() << "can't inline recursive " << call->getCalledFunction()->getName() << "\n";
            continue;
        }
        //llvm::errs() << "inlining " << call->getCalledFunction()->getName() << "\n";
        InlineFunctionInfo IFI;
        InlineFunction(call, IFI);
        count++;
        if (count >= autodiff_inline_count)
            break;
        else
          goto remover;
     }
}

PHINode* canonicalizeIVs(fake::SCEVExpander &e, Type *Ty, Loop *L, DominatorTree &DT, GradientUtils* gutils) {
    PHINode *CanonicalIV = e.getOrInsertCanonicalInductionVariable(L, Ty);
    assert (CanonicalIV && "canonicalizing IV");

    // This ensures that SE knows that the canonical IV doesn't wrap around
    // This is permissible as Enzyme may assume that your program doesn't have an infinite loop (and thus will never end)
    for(auto& a : CanonicalIV->incoming_values()) {
        if (auto add = dyn_cast<BinaryOperator>(a.getUser())) {
            add->setHasNoUnsignedWrap(true);
            add->setHasNoSignedWrap(true);
        }
    }

    SmallVector<WeakTrackingVH, 16> DeadInst0;
    e.replaceCongruentIVs(L, &DT, DeadInst0);

    for (WeakTrackingVH V : DeadInst0) {
        gutils->erase(cast<Instruction>(V)); //->eraseFromParent();
    }

    return CanonicalIV;
}

Function* preprocessForClone(Function *F, AAResults &AA, TargetLibraryInfo &TLI) {
 static std::map<Function*,Function*> cache;
 static std::map<Function*, BasicAAResult*> cache_AA;
 if (cache.find(F) != cache.end()) {
   AA.addAAResult(*(cache_AA[F]));
   return cache[F];
 }
 Function *NewF = Function::Create(F->getFunctionType(), F->getLinkage(), "preprocess_" + F->getName(), F->getParent());

 ValueToValueMapTy VMap;
 for (auto i=F->arg_begin(), j=NewF->arg_begin(); i != F->arg_end(); ) {
     VMap[i] = j;
     j->setName(i->getName());
     i++;
     j++;
 }

 SmallVector <ReturnInst*,4> Returns;
 CloneFunctionInto(NewF, F, VMap, F->getSubprogram() != nullptr, Returns, "",
                   nullptr);
 NewF->setAttributes(F->getAttributes());

 {
    FunctionAnalysisManager AM;
 AM.registerPass([] { return LoopAnalysis(); });
 AM.registerPass([] { return DominatorTreeAnalysis(); });
 AM.registerPass([] { return ScalarEvolutionAnalysis(); });
 AM.registerPass([] { return AssumptionAnalysis(); });

#if LLVM_VERSION_MAJOR >= 8
 AM.registerPass([] { return PassInstrumentationAnalysis(); });
#endif
    LoopSimplifyPass().run(*NewF, AM);

 }

 if (enzyme_preopt) {

  if(autodiff_inline) {
      //llvm::errs() << "running inlining process\n";
      forceRecursiveInlining(NewF, F);

      {
         DominatorTree DT(*NewF);
         AssumptionCache AC(*NewF);
         promoteMemoryToRegister(*NewF, DT, AC);
      }

      {
         FunctionAnalysisManager AM;
         AM.registerPass([] { return AAManager(); });
         AM.registerPass([] { return ScalarEvolutionAnalysis(); });
         AM.registerPass([] { return AssumptionAnalysis(); });
         AM.registerPass([] { return TargetLibraryAnalysis(); });
         AM.registerPass([] { return TargetIRAnalysis(); });
         AM.registerPass([] { return MemorySSAAnalysis(); });
         AM.registerPass([] { return DominatorTreeAnalysis(); });
         AM.registerPass([] { return MemoryDependenceAnalysis(); });
         AM.registerPass([] { return LoopAnalysis(); });
         AM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
        AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
         AM.registerPass([] { return LazyValueAnalysis(); });

        GVN().run(*NewF, AM);

        SROA().run(*NewF, AM);
      }
 }

 bool repeat = false;
 do {
     repeat = false;
 for(auto& BB: *NewF) {
 for(Instruction &I : BB) {
    if (auto bc = dyn_cast<BitCastInst>(&I)) {
        if (auto bc2 = dyn_cast<BitCastInst>(bc->getOperand(0))) {
            if (bc2->getNumUses() == 1) {
                IRBuilder<> b(bc2);
                auto c = b.CreateBitCast(bc2->getOperand(0), I.getType());
                bc->replaceAllUsesWith(c);
                bc->eraseFromParent();
                bc2->eraseFromParent();
                repeat = true;
                break;
            }
        } else if (auto pt = dyn_cast<PointerType>(bc->getOperand(0)->getType())) {
          if (auto st = dyn_cast<StructType>(pt->getElementType())) {

          if (auto pt2 = dyn_cast<PointerType>(bc->getType())) {
            if (st->getNumElements() && st->getElementType(0) == pt2->getElementType()) {
                IRBuilder<> b(bc);
                auto c = b.CreateGEP(bc->getOperand(0), {
                        ConstantInt::get(Type::getInt64Ty(I.getContext()), 0),
                        ConstantInt::get(Type::getInt32Ty(I.getContext()), 0),
                        });
                bc->replaceAllUsesWith(c);
                bc->eraseFromParent();
                repeat = true;
                break;
            }}
          }
        }
    }
 }
 if (repeat) break;
 }
 } while(repeat);

 {
     FunctionAnalysisManager AM;
     AM.registerPass([] { return AAManager(); });
     AM.registerPass([] { return ScalarEvolutionAnalysis(); });
     AM.registerPass([] { return AssumptionAnalysis(); });
     AM.registerPass([] { return TargetLibraryAnalysis(); });
     AM.registerPass([] { return TargetIRAnalysis(); });
     AM.registerPass([] { return MemorySSAAnalysis(); });
     AM.registerPass([] { return DominatorTreeAnalysis(); });
     AM.registerPass([] { return MemoryDependenceAnalysis(); });
     AM.registerPass([] { return LoopAnalysis(); });
     AM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
     AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
#if LLVM_VERSION_MAJOR >= 8
 AM.registerPass([] { return PassInstrumentationAnalysis(); });
#endif
     AM.registerPass([] { return LazyValueAnalysis(); });
     InstCombinePass().run(*NewF, AM);
#if LLVM_VERSION_MAJOR > 6
     InstSimplifyPass().run(*NewF, AM);
#endif
     InstCombinePass().run(*NewF, AM);

     EarlyCSEPass(/*memoryssa*/true).run(*NewF, AM);

     GVN().run(*NewF, AM);
     SROA().run(*NewF, AM);

     CorrelatedValuePropagationPass().run(*NewF, AM);

     DCEPass().run(*NewF, AM);
 }

 do {
     repeat = false;
 for(Instruction &I : NewF->getEntryBlock()) {
    if (auto ci = dyn_cast<ICmpInst>(&I)) {
        for(Instruction &J : *ci->getParent()) {
            if (&J == &I) break;
            if (auto ci2 = dyn_cast<ICmpInst>(&J)) {
                if (  (ci->getPredicate() == ci2->getInversePredicate()) &&
                        (
                         ( ci->getOperand(0) == ci2->getOperand(0) && ci->getOperand(1) == ci2->getOperand(1) )
                         ||
                         ( (ci->isEquality() || ci2->isEquality()) && ci->getOperand(0) == ci2->getOperand(1) && ci->getOperand(1) == ci2->getOperand(0) )
                            ) ) {
                    IRBuilder<> b(ci);
                    Value* tonot = ci2;
                    for(User* a : ci2->users()) {
                        if (auto ii = dyn_cast<IntrinsicInst>(a)) {
                            if (ii->getIntrinsicID() == Intrinsic::assume) {
                                tonot = ConstantInt::getTrue(ii->getContext());
                                break;
                            }
                        }
                    }
                    auto c = b.CreateNot(tonot);

                    ci->replaceAllUsesWith(c);
                    ci->eraseFromParent();
                    repeat = true;
                    break;
                }
            }
        }
        if (repeat) break;
    }
 }
 } while(repeat);


 {
     FunctionAnalysisManager AM;
     AM.registerPass([] { return AAManager(); });
     AM.registerPass([] { return ScalarEvolutionAnalysis(); });
     AM.registerPass([] { return AssumptionAnalysis(); });
     AM.registerPass([] { return TargetLibraryAnalysis(); });
     AM.registerPass([] { return TargetIRAnalysis(); });
     AM.registerPass([] { return MemorySSAAnalysis(); });
     AM.registerPass([] { return DominatorTreeAnalysis(); });
     AM.registerPass([] { return MemoryDependenceAnalysis(); });
     AM.registerPass([] { return LoopAnalysis(); });
     AM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
     AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
#if LLVM_VERSION_MAJOR >= 8
 AM.registerPass([] { return PassInstrumentationAnalysis(); });
#endif
     AM.registerPass([] { return LazyValueAnalysis(); });

     DSEPass().run(*NewF, AM);
 }

 {
    FunctionAnalysisManager AM;
     AM.registerPass([] { return AAManager(); });
     AM.registerPass([] { return ScalarEvolutionAnalysis(); });
     AM.registerPass([] { return AssumptionAnalysis(); });
     AM.registerPass([] { return TargetLibraryAnalysis(); });
     AM.registerPass([] { return TargetIRAnalysis(); });
     AM.registerPass([] { return MemorySSAAnalysis(); });
     AM.registerPass([] { return DominatorTreeAnalysis(); });
     AM.registerPass([] { return MemoryDependenceAnalysis(); });
     AM.registerPass([] { return LoopAnalysis(); });
     AM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
     AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
#if LLVM_VERSION_MAJOR >= 8
 AM.registerPass([] { return PassInstrumentationAnalysis(); });
#endif
     AM.registerPass([] { return LazyValueAnalysis(); });
    LoopAnalysisManager LAM;
     AM.registerPass([&] { return LoopAnalysisManagerFunctionProxy(LAM); });
     LAM.registerPass([&] { return FunctionAnalysisManagerLoopProxy(AM); });

 SimplifyCFGOptions scfgo(/*unsigned BonusThreshold=*/1, /*bool ForwardSwitchCond=*/false, /*bool SwitchToLookup=*/false, /*bool CanonicalLoops=*/true, /*bool SinkCommon=*/true, /*AssumptionCache *AssumpCache=*/nullptr);
 SimplifyCFGPass(scfgo).run(*NewF, AM);
 LoopSimplifyPass().run(*NewF, AM);

 if (autodiff_inline) {
     createFunctionToLoopPassAdaptor(LoopIdiomRecognizePass()).run(*NewF, AM);
 }
 DSEPass().run(*NewF, AM);
 LoopSimplifyPass().run(*NewF, AM);

 }
 }

 /*
 //! Loop rotation now necessary to ensure that the condition of a loop is at the end
 {
    FunctionAnalysisManager AM;
     AM.registerPass([] { return AAManager(); });
     AM.registerPass([] { return ScalarEvolutionAnalysis(); });
     AM.registerPass([] { return AssumptionAnalysis(); });
     AM.registerPass([] { return TargetLibraryAnalysis(); });
     AM.registerPass([] { return TargetIRAnalysis(); });
     AM.registerPass([] { return LoopAnalysis(); });
     AM.registerPass([] { return MemorySSAAnalysis(); });
     AM.registerPass([] { return DominatorTreeAnalysis(); });
     AM.registerPass([] { return MemoryDependenceAnalysis(); });
     #if LLVM_VERSION_MAJOR > 6
     AM.registerPass([] { return PhiValuesAnalysis(); });
     #endif
     #if LLVM_VERSION_MAJOR >= 8
     AM.registerPass([] { return PassInstrumentationAnalysis(); });
     #endif

     {

     LoopAnalysisManager LAM;
     AM.registerPass([&] { return LoopAnalysisManagerFunctionProxy(LAM); });
     LAM.registerPass([&] { return FunctionAnalysisManagerLoopProxy(AM); });
 
        {
        //Loop rotation is necessary to ensure we are of the form body then conditional
        createFunctionToLoopPassAdaptor(LoopRotatePass()).run(*NewF, AM); 
        }
    LAM.clear();
    }
    AM.clear();
  }
 */
 

 {
    FunctionAnalysisManager AM;
     AM.registerPass([] { return AAManager(); });
     AM.registerPass([] { return ScalarEvolutionAnalysis(); });
     //AM.registerPass([] { return AssumptionAnalysis(); });
     AM.registerPass([] { return TargetLibraryAnalysis(); });
     AM.registerPass([] { return TargetIRAnalysis(); });
     AM.registerPass([] { return LoopAnalysis(); });
     AM.registerPass([] { return MemorySSAAnalysis(); });
     AM.registerPass([] { return DominatorTreeAnalysis(); });
     AM.registerPass([] { return MemoryDependenceAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
     AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
#if LLVM_VERSION_MAJOR >= 8
 AM.registerPass([] { return PassInstrumentationAnalysis(); });
#endif

     ModuleAnalysisManager MAM;
     AM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });
     MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(AM); });

 //Alias analysis is necessary to ensure can query whether we can move a forward pass function
 //BasicAA ba;
 //auto baa = new BasicAAResult(ba.run(*NewF, AM));
 AssumptionCache* AC = new AssumptionCache(*NewF);
 TargetLibraryInfo* TLI = new TargetLibraryInfo(AM.getResult<TargetLibraryAnalysis>(*NewF));
 DominatorTree* DTL = new DominatorTree(*NewF);
 LoopInfo* LI = new LoopInfo(*DTL);
 #if LLVM_VERSION_MAJOR > 6
 PhiValues* PV = new PhiValues(*F);
 #endif
 auto baa = new BasicAAResult(NewF->getParent()->getDataLayout(),
#if LLVM_VERSION_MAJOR > 6
                        *NewF,
#endif
                        *TLI,
                        *AC,
                        DTL/*&AM.getResult<DominatorTreeAnalysis>(*NewF)*/,
                        LI
#if LLVM_VERSION_MAJOR > 6
                        ,PV
#endif
                        );
 cache_AA[F] = baa;
 AA.addAAResult(*baa);
 //ScopedNoAliasAA sa;
 //auto saa = new ScopedNoAliasAAResult(sa.run(*NewF, AM));
 //AA.addAAResult(*saa);

 }

  if (enzyme_print)
      llvm::errs() << "after simplification :\n" << *NewF << "\n";

  if (llvm::verifyFunction(*NewF, &llvm::errs())) {
      llvm::errs() << *NewF << "\n";
      report_fatal_error("function failed verification (1)");
  }
  cache[F] = NewF;
  return NewF;
}

Function *CloneFunctionWithReturns(Function *&F, AAResults &AA, TargetLibraryInfo &TLI, ValueToValueMapTy& ptrInputs, const std::set<unsigned>& constant_args, SmallPtrSetImpl<Value*> &constants, SmallPtrSetImpl<Value*> &nonconstant, SmallPtrSetImpl<Value*> &returnvals, ReturnType returnValue, bool differentialReturn, Twine name, ValueToValueMapTy *VMapO, bool diffeReturnArg, llvm::Type* additionalArg) {
 assert(!F->empty());
 F = preprocessForClone(F, AA, TLI);
 diffeReturnArg &= differentialReturn;
 std::vector<Type*> RetTypes;
 if (returnValue == ReturnType::ArgsWithReturn || returnValue == ReturnType::ArgsWithTwoReturns)
   RetTypes.push_back(F->getReturnType());
 if (returnValue == ReturnType::ArgsWithTwoReturns)
   RetTypes.push_back(F->getReturnType());
 std::vector<Type*> ArgTypes;

 ValueToValueMapTy VMap;

 // The user might be deleting arguments to the function by specifying them in
 // the VMap.  If so, we need to not add the arguments to the arg ty vector
 //
 unsigned argno = 0;
 for (const Argument &I : F->args()) {
     ArgTypes.push_back(I.getType());
     if (constant_args.count(argno)) {
        argno++;
        continue;
     }
     if (I.getType()->isPointerTy() || I.getType()->isIntegerTy()) {
        ArgTypes.push_back(I.getType());
        /*
        if (I.getType()->isPointerTy() && !(I.hasAttribute(Attribute::ReadOnly) || I.hasAttribute(Attribute::ReadNone) ) ) {
          llvm::errs() << "Cannot take derivative of function " <<F->getName()<< " input argument to function " << I.getName() << " is not marked read-only\n";
          exit(1);
        }
        */
     } else {
       RetTypes.push_back(I.getType());
     }
     argno++;
 }

 if (diffeReturnArg && !F->getReturnType()->isPointerTy() && !F->getReturnType()->isIntegerTy()) {
    assert(!F->getReturnType()->isVoidTy());
    ArgTypes.push_back(F->getReturnType());
 }
 if (additionalArg) {
    ArgTypes.push_back(additionalArg);
 }
 Type* RetType = StructType::get(F->getContext(), RetTypes);
 if (returnValue == ReturnType::TapeAndReturns || returnValue == ReturnType::Tape) {
     RetTypes.clear();
     RetTypes.push_back(Type::getInt8PtrTy(F->getContext()));
     if (!F->getReturnType()->isVoidTy() && returnValue == ReturnType::TapeAndReturns) {
        RetTypes.push_back(F->getReturnType());
        if ( (F->getReturnType()->isPointerTy() || F->getReturnType()->isIntegerTy()) && differentialReturn)
          RetTypes.push_back(F->getReturnType());
    }
    RetType = StructType::get(F->getContext(), RetTypes);
 }

 // Create a new function type...
 FunctionType *FTy = FunctionType::get(RetType,
                                   ArgTypes, F->getFunctionType()->isVarArg());

 // Create the new function...
 Function *NewF = Function::Create(FTy, F->getLinkage(), name, F->getParent());
 if (diffeReturnArg && !F->getReturnType()->isPointerTy() && !F->getReturnType()->isIntegerTy()) {
    auto I = NewF->arg_end();
    I--;
    if(additionalArg)
        I--;
    I->setName("differeturn");
 }
 if (additionalArg) {
    auto I = NewF->arg_end();
    I--;
    I->setName("tapeArg");
 }

 bool hasPtrInput = false;

 unsigned ii = 0, jj = 0;
 for (auto i=F->arg_begin(), j=NewF->arg_begin(); i != F->arg_end(); ) {
   bool isconstant = (constant_args.count(ii) > 0);

   if (isconstant) {
      constants.insert(j);
      if (printconst)
        llvm::errs() << "in new function " << NewF->getName() << " constant arg " << *j << "\n";
   } else {
      nonconstant.insert(j);
      if (printconst)
        llvm::errs() << "in new function " << NewF->getName() << " nonconstant arg " << *j << "\n";
   }

   if (!isconstant && ( i->getType()->isPointerTy() || i->getType()->isIntegerTy()) ) {
     VMap[i] = j;
     hasPtrInput = true;
     ptrInputs[j] = (j+1);
     if (F->hasParamAttribute(ii, Attribute::NoCapture)) {
       NewF->addParamAttr(jj, Attribute::NoCapture);
       NewF->addParamAttr(jj+1, Attribute::NoCapture);
     }
     if (F->hasParamAttribute(ii, Attribute::NoAlias)) {
       NewF->addParamAttr(jj, Attribute::NoAlias);
       NewF->addParamAttr(jj+1, Attribute::NoAlias);
     }

     j->setName(i->getName());
     j++;
     j->setName(i->getName()+"'");
     nonconstant.insert(j);
     j++;
     jj+=2;

     i++;
     ii++;

   } else {
     VMap[i] = j;
     j->setName(i->getName());

     j++;
     jj++;
     i++;
     ii++;
   }
 }

 // Loop over the arguments, copying the names of the mapped arguments over...
 Function::arg_iterator DestI = NewF->arg_begin();


 for (const Argument & I : F->args())
   if (VMap.count(&I) == 0) {     // Is this argument preserved?
     DestI->setName(I.getName()); // Copy the name over...
     VMap[&I] = &*DestI++;        // Add mapping to VMap
   }
 SmallVector <ReturnInst*,4> Returns;
 CloneFunctionInto(NewF, F, VMap, F->getSubprogram() != nullptr, Returns, "",
                   nullptr);
 if (VMapO) VMapO->insert(VMap.begin(), VMap.end());

 if (hasPtrInput) {
    if (NewF->hasFnAttribute(Attribute::ReadNone)) {
    NewF->removeFnAttr(Attribute::ReadNone);
    }
    if (NewF->hasFnAttribute(Attribute::ReadOnly)) {
    NewF->removeFnAttr(Attribute::ReadOnly);
    }
 }
 NewF->setLinkage(Function::LinkageTypes::InternalLinkage);
 assert(NewF->hasLocalLinkage());

 if (differentialReturn) {
   for(auto& r : Returns) {
     if (auto a = r->getReturnValue()) {
       returnvals.insert(a);
     }
   }
 }

 //SmallPtrSet<Value*,4> constants2;
 //for (auto a :constants){
 //   constants2.insert(a);
// }
 //for (auto a :nonconstant){
 //   nonconstant2.insert(a);
 //}

 return NewF;
}



static cl::opt<bool> autodiff_optimize(
            "enzyme_optimize", cl::init(false), cl::Hidden,
                cl::desc("Force inlining of autodiff"));

void optimizeIntermediate(GradientUtils* gutils, bool topLevel, Function *F) {
    if (!autodiff_optimize) return;

    {
        DominatorTree DT(*F);
        AssumptionCache AC(*F);
        promoteMemoryToRegister(*F, DT, AC);
    }

    FunctionAnalysisManager AM;
     AM.registerPass([] { return AAManager(); });
     AM.registerPass([] { return ScalarEvolutionAnalysis(); });
     AM.registerPass([] { return AssumptionAnalysis(); });
     AM.registerPass([] { return TargetLibraryAnalysis(); });
     AM.registerPass([] { return TargetIRAnalysis(); });
     AM.registerPass([] { return MemorySSAAnalysis(); });
     AM.registerPass([] { return DominatorTreeAnalysis(); });
     AM.registerPass([] { return MemoryDependenceAnalysis(); });
     AM.registerPass([] { return LoopAnalysis(); });
     AM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
     AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
     AM.registerPass([] { return LazyValueAnalysis(); });
     LoopAnalysisManager LAM;
     AM.registerPass([&] { return LoopAnalysisManagerFunctionProxy(LAM); });
     LAM.registerPass([&] { return FunctionAnalysisManagerLoopProxy(AM); });
    //LoopSimplifyPass().run(*F, AM);

 //TODO function attributes
 //PostOrderFunctionAttrsPass().run(*F, AM);
 GVN().run(*F, AM);
 SROA().run(*F, AM);
 EarlyCSEPass(/*memoryssa*/true).run(*F, AM);
#if LLVM_VERSION_MAJOR > 6
 InstSimplifyPass().run(*F, AM);
#endif
 CorrelatedValuePropagationPass().run(*F, AM);

 DCEPass().run(*F, AM);
 DSEPass().run(*F, AM);

 createFunctionToLoopPassAdaptor(LoopDeletionPass()).run(*F, AM);

 SimplifyCFGOptions scfgo(/*unsigned BonusThreshold=*/1, /*bool ForwardSwitchCond=*/false, /*bool SwitchToLookup=*/false, /*bool CanonicalLoops=*/true, /*bool SinkCommon=*/true, /*AssumptionCache *AssumpCache=*/nullptr);
 SimplifyCFGPass(scfgo).run(*F, AM);

 if (!topLevel) {
 for(BasicBlock& BB: *F) {

        for (auto I = BB.begin(), E = BB.end(); I != E;) {
          Instruction* inst = &*I;
          assert(inst);
          I++;

          if (gutils->originalInstructions.find(inst) == gutils->originalInstructions.end()) continue;

          if (gutils->replaceableCalls.find(inst) != gutils->replaceableCalls.end()) {
            if (inst->getNumUses() != 0 && !cast<CallInst>(inst)->getCalledFunction()->hasFnAttribute(Attribute::ReadNone) ) {
                llvm::errs() << "found call ripe for replacement " << *inst;
            } else {
                    gutils->erase(inst);
                    continue;
            }
          }
        }
      }
 }
 //LCSSAPass().run(*NewF, AM);
}
