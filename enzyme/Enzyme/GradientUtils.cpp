/*
 * GradientUtils.cpp - Gradient Utility data structures and functions
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

#include "GradientUtils.h"

#include <llvm/Config/llvm-config.h>

#include "EnzymeLogic.h"

#include "FunctionUtils.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"

#include <algorithm>

  //! Given an edge from BB to branchingBlock get the corresponding block to branch to in the reverse pass
  BasicBlock* GradientUtils::getReverseOrLatchMerge(BasicBlock* BB, BasicBlock* branchingBlock) {
    assert(BB);
    // BB should be a forward pass block, assert that
    if (reverseBlocks.find(BB) == reverseBlocks.end()) {
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << "BB: " << *BB << "\n";
        llvm::errs() << "branchingBlock: " << *branchingBlock << "\n";
    }
    assert(reverseBlocks.find(BB) != reverseBlocks.end());
    LoopContext lc;
    bool inLoop = getContext(BB, lc);
    
    LoopContext branchingContext;
    bool inLoopContext = getContext(branchingBlock, branchingContext);
    
    if (!inLoop) return reverseBlocks[BB];

    auto tup = std::make_tuple(BB, branchingBlock);
    if (newBlocksForLoop_cache.find(tup) != newBlocksForLoop_cache.end()) return newBlocksForLoop_cache[tup];

    if (inLoop && inLoopContext && branchingBlock == lc.header && lc.header == branchingContext.header) {
        BasicBlock* incB = BasicBlock::Create(BB->getContext(), "inc" + reverseBlocks[lc.header]->getName(), BB->getParent());
        incB->moveAfter(reverseBlocks[lc.header]);
        
        IRBuilder<> tbuild(incB);
		
        Value* av = tbuild.CreateLoad(lc.antivaralloc);
        Value* sub = tbuild.CreateSub(av, ConstantInt::get(av->getType(), 1), "", true, true);
        tbuild.CreateStore(sub, lc.antivaralloc);
        tbuild.CreateBr(reverseBlocks[BB]);
        return newBlocksForLoop_cache[tup] = incB;
    }
    
    if (inLoop) {
        auto latches = fake::SCEVExpander::getLatches(LI.getLoopFor(BB), lc.exitBlocks);

        if (std::find(latches.begin(), latches.end(), BB) != latches.end() && std::find(lc.exitBlocks.begin(), lc.exitBlocks.end(), branchingBlock) != lc.exitBlocks.end()) {
            BasicBlock* incB = BasicBlock::Create(BB->getContext(), "merge" + reverseBlocks[lc.header]->getName()+"_" + branchingBlock->getName(), BB->getParent());
            incB->moveAfter(reverseBlocks[branchingBlock]);
            
            IRBuilder<> tbuild(reverseBlocks[branchingBlock]);
            
            Value* lim = nullptr;
            if (lc.dynamic) {
                lim = lookupValueFromCache(tbuild, lc.preheader, cast<AllocaInst>(lc.limit));
            } else {
                lim = lookupM(lc.limit, tbuild);
            }
            
            tbuild.SetInsertPoint(incB);
            tbuild.CreateStore(lim, lc.antivaralloc);
            tbuild.CreateBr(reverseBlocks[BB]);

            return newBlocksForLoop_cache[tup] = incB;
        }
    }
        
    return newBlocksForLoop_cache[tup] = reverseBlocks[BB];
  }

  void GradientUtils::forceContexts() {
    for(auto BB : originalBlocks) {
        LoopContext lc;
        getContext(BB, lc);
    }
  }

//! Given the option to recompute a value or re-use an old one, return true if we should recompute this value from scratch
bool GradientUtils::shouldRecompute(Value* val, const ValueToValueMapTy& available) {
  if (available.count(val)) return true;
  if (isa<Argument>(val) || isa<Constant>(val)) {
    return true;
  } else if (auto op = dyn_cast<CastInst>(val)) {
    return shouldRecompute(op->getOperand(0), available);
  } else if (isa<AllocaInst>(val)) {
    //don't recompute an alloca inst (and thereby create a new allocation)
    return false;
  } else if (auto op = dyn_cast<BinaryOperator>(val)) {
    bool a0 = shouldRecompute(op->getOperand(0), available);
    if (a0) {
        //llvm::errs() << "need recompute: " << *op->getOperand(0) << "\n";
    }
    bool a1 = shouldRecompute(op->getOperand(1), available);
    if (a1) {
        //llvm::errs() << "need recompute: " << *op->getOperand(1) << "\n";
    }
    return a0 && a1;
  } else if (auto op = dyn_cast<CmpInst>(val)) {
    return shouldRecompute(op->getOperand(0), available) && shouldRecompute(op->getOperand(1), available);
  } else if (auto op = dyn_cast<SelectInst>(val)) {
    return shouldRecompute(op->getOperand(0), available) && shouldRecompute(op->getOperand(1), available) && shouldRecompute(op->getOperand(2), available);
  } else if (auto load = dyn_cast<LoadInst>(val)) {
    Value* idx = load->getOperand(0);

    while (!isa<Argument>(idx)) {

      if (auto gep = dyn_cast<GetElementPtrInst>(idx)) {
        for(auto &a : gep->indices()) {
          if (!shouldRecompute(a, available)) {
            //llvm::errs() << "not recomputable load " << *load << " as arg " << *gep << " has bad idx " << *a << "\n";
            return false;
          }
        }
        idx = gep->getPointerOperand();

      } else if(auto cast = dyn_cast<CastInst>(idx)) {
        idx = cast->getOperand(0);

      } else if(auto ci = dyn_cast<CallInst>(idx)) {
        if (!shouldRecompute(idx, available)) {
            //llvm::errs() << "not recomputable load " << *load << " as arg " << *idx << "\n";
            return false;
        }
        
        if (ci->hasRetAttr(Attribute::ReadOnly) || ci->hasRetAttr(Attribute::ReadNone)) {
          //llvm::errs() << "recomputable load " << *load << " from call readonly ret " << *ci << "\n";
          return true;
        }

        //llvm::errs() << "not recomputable load " << *load << " as arg " << *idx << "\n";
        return false;

      } else {
              //llvm::errs() << "not a gep " << *idx << "\n";
        //llvm::errs() << "not recomputable load " << *load << " unknown as arg " << *idx << "\n";
        return false;
      }
    }

    Argument* arg = cast<Argument>(idx);
    if (arg->hasAttribute(Attribute::ReadOnly) || arg->hasAttribute(Attribute::ReadNone)) {
            //llvm::errs() << "argument " << *arg << " not marked read only\n";
      //llvm::errs() << "recomputable load " << *load << " from as argument " << *arg << "\n";
      return true;
    }

    //llvm::errs() << "not recomputable load " << *load << " unknown as argument " << *arg << "\n";
    return false;

  } else if (auto phi = dyn_cast<PHINode>(val)) {
    if (phi->getNumIncomingValues () == 1) {
      bool b = shouldRecompute(phi->getIncomingValue(0) , available);
      if (b) {
            //llvm::errs() << "phi need recompute: " <<*phi->getIncomingValue(0) << "\n";
      }
      return b;
    }

    return false;
  } else if (auto op = dyn_cast<IntrinsicInst>(val)) {
    switch(op->getIntrinsicID()) {
      case Intrinsic::sin:
      case Intrinsic::cos:
      return true;
      return shouldRecompute(op->getOperand(0), available);
      default:
      return false;
    }
  }
  //llvm::errs() << "unknown inst " << *val << " unable to recompute\n";
  return false;
}

GradientUtils* GradientUtils::CreateFromClone(Function *todiff, AAResults &AA, TargetLibraryInfo &TLI, const std::set<unsigned> & constant_args, bool returnUsed, bool differentialReturn, std::map<AugmentedStruct, unsigned> &returnMapping ) {
    assert(!todiff->empty());

    // Since this is forward pass this should always return the tape (at index 0)
    returnMapping[AugmentedStruct::Tape] = 0;

    int returnCount = 0;

    if (returnUsed) {
        assert(!todiff->getReturnType()->isEmptyTy());
        returnMapping[AugmentedStruct::Return] = returnCount+1;
        returnCount++;
    } 
   
    // We don't need to differentially return something that we know is not a pointer (or somehow needed for shadow analysis)
    if (differentialReturn && !todiff->getReturnType()->isFPOrFPVectorTy()) { 
        assert(!todiff->getReturnType()->isEmptyTy());
        assert(!todiff->getReturnType()->isFPOrFPVectorTy());
        returnMapping[AugmentedStruct::DifferentialReturn] = returnCount+1;
        returnCount++;
    } 

    ReturnType returnValue;
    if (returnCount == 0) returnValue = ReturnType::Tape;
    else if (returnCount == 1) returnValue = ReturnType::TapeAndReturn;
    else if (returnCount == 2) returnValue = ReturnType::TapeAndTwoReturns;
    else llvm_unreachable("illegal number of elements in augmented return struct");

    ValueToValueMapTy invertedPointers;
    SmallPtrSet<Value*,4> constants;
    SmallPtrSet<Value*,20> nonconstant;
    SmallPtrSet<Value*,2> returnvals;
    ValueToValueMapTy originalToNew;
    auto newFunc = CloneFunctionWithReturns(todiff, AA, TLI, invertedPointers, constant_args, constants, nonconstant, returnvals, /*returnValue*/returnValue, /*differentialReturn*/differentialReturn, "fakeaugmented_"+todiff->getName(), &originalToNew, /*diffeReturnArg*/false, /*additionalArg*/nullptr);
    //llvm::errs() <<  "returnvals:" << todiff->getName() << " \n";
    //for (auto a : returnvals ) {
    //    llvm::errs() <<"   + " << *a << "\n";
    //}
    //llvm::errs() <<  "end returnvals:\n";
    auto res = new GradientUtils(newFunc, todiff, AA, TLI, invertedPointers, constants, nonconstant, returnvals, originalToNew);
    return res;
}

DiffeGradientUtils* DiffeGradientUtils::CreateFromClone(Function *todiff, AAResults &AA, TargetLibraryInfo &TLI, const std::set<unsigned> & constant_args, ReturnType returnValue, bool differentialReturn, Type* additionalArg) {
  assert(!todiff->empty());
  ValueToValueMapTy invertedPointers;
  SmallPtrSet<Value*,4> constants;
  SmallPtrSet<Value*,20> nonconstant;
  SmallPtrSet<Value*,2> returnvals;
  ValueToValueMapTy originalToNew;
  auto newFunc = CloneFunctionWithReturns(todiff, AA, TLI, invertedPointers, constant_args, constants, nonconstant, returnvals, returnValue, differentialReturn, "diffe"+todiff->getName(), &originalToNew, /*diffeReturnArg*/true, additionalArg);
  auto res = new DiffeGradientUtils(newFunc, todiff, AA, TLI, invertedPointers, constants, nonconstant, returnvals, originalToNew);
  return res;
}

Value* GradientUtils::invertPointerM(Value* val, IRBuilder<>& BuilderM) {
    if (isa<ConstantPointerNull>(val)) {
        return val;
    } else if (isa<UndefValue>(val)) {
        return val;
    } else if (auto cint = dyn_cast<ConstantInt>(val)) {
        if (cint->isZero()) return cint;
        if (cint->isOne()) return cint;
    }

    if(isConstantValue(val)) {
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        dumpSet(this->originalInstructions);
        if (auto arg = dyn_cast<Instruction>(val)) {
            llvm::errs() << *arg->getParent()->getParent() << "\n";
        }
        llvm::errs() << *val << "\n";
    }
    assert(!isConstantValue(val));

    auto M = BuilderM.GetInsertBlock()->getParent()->getParent();
    assert(val);

    if (invertedPointers.find(val) != invertedPointers.end()) {
        return lookupM(invertedPointers[val], BuilderM);
    }

    if (auto arg = dyn_cast<GlobalVariable>(val)) {
      if (!hasMetadata(arg, "enzyme_shadow")) {
          llvm::errs() << *arg << "\n";
          assert(0 && "cannot compute with global variable that doesn't have marked shadow global");
          report_fatal_error("cannot compute with global variable that doesn't have marked shadow global");
      }
      auto md = arg->getMetadata("enzyme_shadow");
      if (!isa<MDTuple>(md)) {
          llvm::errs() << *arg << "\n";
          llvm::errs() << *md << "\n";
          assert(0 && "cannot compute with global variable that doesn't have marked shadow global");
          report_fatal_error("cannot compute with global variable that doesn't have marked shadow global (metadata incorrect type)");
      }
      auto md2 = cast<MDTuple>(md);
      assert(md2->getNumOperands() == 1);
      auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
      auto cs = gvemd->getValue();
      return invertedPointers[val] = cs;
    } else if (auto fn = dyn_cast<Function>(val)) {
      //! Todo allow tape propagation
      //  Note that specifically this should _not_ be called with topLevel=true (since it may not be valid to always assume we can recompute the augmented primal)
      //  However, in the absence of a way to pass tape data from an indirect augmented (and also since we dont presently allow indirect augmented calls), topLevel MUST be true
      //  otherwise subcalls will not be able to lookup the augmenteddata/subdata (triggering an assertion failure, among much worse)
      std::map<Argument*, bool> uncacheable_args;
      auto newf = CreatePrimalAndGradient(fn, /*constant_args*/{}, TLI, AA, /*returnValue*/false, /*differentialReturn*/fn->getReturnType()->isFPOrFPVectorTy(), /*dretPtr*/false, /*topLevel*/true, /*additionalArg*/nullptr, uncacheable_args, /*map*/nullptr); //llvm::Optional<std::map<std::pair<llvm::Instruction*, std::string>, unsigned int> >({}));
      return BuilderM.CreatePointerCast(newf, fn->getType());
    } else if (auto arg = dyn_cast<CastInst>(val)) {
      auto result = BuilderM.CreateCast(arg->getOpcode(), invertPointerM(arg->getOperand(0), BuilderM), arg->getDestTy(), arg->getName()+"'ipc");
      return result;
    } else if (auto arg = dyn_cast<ConstantExpr>(val)) {
      if (arg->isCast()) {
          auto result = ConstantExpr::getCast(arg->getOpcode(), cast<Constant>(invertPointerM(arg->getOperand(0), BuilderM)), arg->getType());
          return result;
      } else if (arg->isGEPWithNoNotionalOverIndexing()) {
          auto result = arg->getWithOperandReplaced(0, cast<Constant>(invertPointerM(arg->getOperand(0), BuilderM)));
          return result;
      }
      goto end;
    } else if (auto arg = dyn_cast<ExtractValueInst>(val)) {
      IRBuilder<> bb(arg);
      auto result = bb.CreateExtractValue(invertPointerM(arg->getOperand(0), bb), arg->getIndices(), arg->getName()+"'ipev");
      invertedPointers[arg] = result;
      return lookupM(invertedPointers[arg], BuilderM);
    } else if (auto arg = dyn_cast<InsertValueInst>(val)) {
      IRBuilder<> bb(arg);
      auto result = bb.CreateInsertValue(invertPointerM(arg->getOperand(0), bb), invertPointerM(arg->getOperand(1), bb), arg->getIndices(), arg->getName()+"'ipiv");
      invertedPointers[arg] = result;
      return lookupM(invertedPointers[arg], BuilderM);
    } else if (auto arg = dyn_cast<SelectInst>(val)) {
      IRBuilder<> bb(arg);
      auto result = bb.CreateSelect(arg->getCondition(), invertPointerM(arg->getTrueValue(), bb), invertPointerM(arg->getFalseValue(), bb), arg->getName()+"'ipse");
      invertedPointers[arg] = result;
      return lookupM(invertedPointers[arg], BuilderM);
    } else if (auto arg = dyn_cast<LoadInst>(val)) {
      IRBuilder <> bb(arg);
        if(isConstantValue(arg->getOperand(0))) {
            llvm::errs() << *oldFunc << "\n";
            llvm::errs() << *newFunc << "\n";
            dumpSet(this->originalInstructions);
            if (auto arg = dyn_cast<Instruction>(val)) {
                llvm::errs() << *arg->getParent()->getParent() << "\n";
            }
            llvm::errs() << *val << "\n";
        }
      assert(!isConstantValue(arg->getOperand(0)));
      auto li = bb.CreateLoad(invertPointerM(arg->getOperand(0), bb), arg->getName()+"'ipl");
      li->setAlignment(arg->getAlignment());
      li->setVolatile(arg->isVolatile());
      li->setOrdering(arg->getOrdering());
      li->setSyncScopeID(arg->getSyncScopeID ());
      invertedPointers[arg] = li;
      return lookupM(invertedPointers[arg], BuilderM);
    } else if (auto arg = dyn_cast<BinaryOperator>(val)) {
 	  assert(arg->getType()->isIntOrIntVectorTy());
      IRBuilder <> bb(arg);
      Value* val0 = nullptr;
      Value* val1 = nullptr;

      if (isConstantValue(arg->getOperand(0)) && isConstantValue(arg->getOperand(1))) {
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        dumpSet(this->originalInstructions);
        llvm::errs() << *arg->getParent() << "\n";
        llvm::errs() << " binary operator for ip has both operands as constant values " << *arg << "\n";
      }

      //if (isa<ConstantInt>(arg->getOperand(0))) {
      if (isConstantValue(arg->getOperand(0))) {
        val0 = arg->getOperand(0);
        val1 = invertPointerM(arg->getOperand(1), bb);
      } else if (isConstantValue(arg->getOperand(1))) {
      //} else if (isa<ConstantInt>(arg->getOperand(1))) {
        val0 = invertPointerM(arg->getOperand(0), bb);
        val1 = arg->getOperand(1);
      } else {
        val0 = invertPointerM(arg->getOperand(0), bb);
        val1 = invertPointerM(arg->getOperand(1), bb);
      }
      
      auto li = bb.CreateBinOp(arg->getOpcode(), val0, val1, arg->getName());
      invertedPointers[arg] = li;
      return lookupM(invertedPointers[arg], BuilderM);
    } else if (auto arg = dyn_cast<GetElementPtrInst>(val)) {
      if (arg->getParent() == &arg->getParent()->getParent()->getEntryBlock()) {
        IRBuilder<> bb(arg);
        SmallVector<Value*,4> invertargs;
        for(auto &a: arg->indices()) {
            auto b = lookupM(a, bb);
            invertargs.push_back(b);
        }
        auto result = bb.CreateGEP(invertPointerM(arg->getPointerOperand(), bb), invertargs, arg->getName()+"'ipge");
        if (auto gep = dyn_cast<GetElementPtrInst>(result))
            gep->setIsInBounds(arg->isInBounds());
        invertedPointers[arg] = result;
        return lookupM(invertedPointers[arg], BuilderM);
      }

      SmallVector<Value*,4> invertargs;
      for(auto &a: arg->indices()) {
          auto b = lookupM(a, BuilderM);
          invertargs.push_back(b);
      }
      auto result = BuilderM.CreateGEP(invertPointerM(arg->getPointerOperand(), BuilderM), invertargs, arg->getName()+"'ipg");
      return result;
    } else if (auto inst = dyn_cast<AllocaInst>(val)) {
        IRBuilder<> bb(inst);
        AllocaInst* antialloca = bb.CreateAlloca(inst->getAllocatedType(), inst->getType()->getPointerAddressSpace(), inst->getArraySize(), inst->getName()+"'ipa");
        invertedPointers[val] = antialloca;
        antialloca->setAlignment(inst->getAlignment());

        auto dst_arg = bb.CreateBitCast(antialloca,Type::getInt8PtrTy(val->getContext()));
        auto val_arg = ConstantInt::get(Type::getInt8Ty(val->getContext()), 0);
        auto len_arg = bb.CreateNUWMul(bb.CreateZExtOrTrunc(inst->getArraySize(),Type::getInt64Ty(val->getContext())), ConstantInt::get(Type::getInt64Ty(val->getContext()), M->getDataLayout().getTypeAllocSizeInBits(inst->getAllocatedType())/8) );
        auto volatile_arg = ConstantInt::getFalse(val->getContext());

#if LLVM_VERSION_MAJOR == 6
        auto align_arg = ConstantInt::get(Type::getInt32Ty(val->getContext()), antialloca->getAlignment());
        Value *args[] = { dst_arg, val_arg, len_arg, align_arg, volatile_arg };
#else
        Value *args[] = { dst_arg, val_arg, len_arg, volatile_arg };
#endif
        Type *tys[] = {dst_arg->getType(), len_arg->getType()};
        auto memset = cast<CallInst>(bb.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args));
        if (inst->getAlignment() != 0) {
            memset->addParamAttr(0, Attribute::getWithAlignment(inst->getContext(), inst->getAlignment()));
        }
        memset->addParamAttr(0, Attribute::NonNull);
        return lookupM(invertedPointers[inst], BuilderM);
    } else if (auto phi = dyn_cast<PHINode>(val)) {
     std::map<Value*,std::set<BasicBlock*>> mapped;
     for(unsigned int i=0; i<phi->getNumIncomingValues(); i++) {
        mapped[phi->getIncomingValue(i)].insert(phi->getIncomingBlock(i));
     }

     if (false && mapped.size() == 1) {
        return invertPointerM(phi->getIncomingValue(0), BuilderM);
     }
#if 0
     else if (false && mapped.size() == 2) {
         IRBuilder <> bb(phi);
         auto which = bb.CreatePHI(Type::getInt1Ty(phi->getContext()), phi->getNumIncomingValues());
         //TODO this is not recursive

         int cnt = 0;
         Value* vals[2];
         for(auto v : mapped) {
            assert( cnt <= 1 );
            vals[cnt] = v.first;
            for (auto b : v.second) {
                which->addIncoming(ConstantInt::get(which->getType(), cnt), b);
            }
            cnt++;
         }

         auto which2 = lookupM(which, BuilderM);
         auto result = BuilderM.CreateSelect(which2, invertPointerM(vals[1], BuilderM), invertPointerM(vals[0], BuilderM));
         return result;
     }
#endif

     else {
         IRBuilder <> bb(phi);
         auto which = bb.CreatePHI(phi->getType(), phi->getNumIncomingValues());
         invertedPointers[val] = which;

         for(unsigned int i=0; i<phi->getNumIncomingValues(); i++) {
            IRBuilder <>pre(phi->getIncomingBlock(i)->getTerminator());
            which->addIncoming(invertPointerM(phi->getIncomingValue(i), pre), phi->getIncomingBlock(i));
         }

         return lookupM(which, BuilderM);
     }
    }

  end:;
    assert(BuilderM.GetInsertBlock());
    assert(BuilderM.GetInsertBlock()->getParent());
    assert(val);
    llvm::errs() << "fn:" << *BuilderM.GetInsertBlock()->getParent() << "\nval=" << *val << "\n";
    for(auto z : invertedPointers) {
      llvm::errs() << "available inversion for " << *z.first << " of " << *z.second << "\n";
    }
    assert(0 && "cannot find deal with ptr that isnt arg");
    report_fatal_error("cannot find deal with ptr that isnt arg");
}

std::pair<PHINode*,Instruction*> insertNewCanonicalIV(Loop* L, Type* Ty) {
    assert(L);
    assert(Ty);

    BasicBlock* Header = L->getHeader();
    assert(Header);
    IRBuilder <>B(&Header->front());
    PHINode *CanonicalIV = B.CreatePHI(Ty, 1, "iv");

    B.SetInsertPoint(Header->getFirstNonPHIOrDbg());
    Instruction* inc = cast<Instruction>(B.CreateNUWAdd(CanonicalIV, ConstantInt::get(CanonicalIV->getType(), 1), "iv.next"));

    for (BasicBlock *Pred : predecessors(Header)) {
        assert(Pred);
        if (L->contains(Pred)) {
            CanonicalIV->addIncoming(inc, Pred);
        } else {
            CanonicalIV->addIncoming(ConstantInt::get(CanonicalIV->getType(), 0), Pred);
        }
    }
    return std::pair<PHINode*,Instruction*>(CanonicalIV,inc);
}

void removeRedundantIVs(const Loop* L, BasicBlock* Header, BasicBlock* Preheader, PHINode* CanonicalIV, ScalarEvolution &SE, GradientUtils &gutils, Instruction* increment, const SmallVectorImpl<BasicBlock*>&& latches) {
    assert(Header);
    assert(CanonicalIV);

    SmallVector<Instruction*, 8> IVsToRemove;

    //This scope is necessary to ensure scevexpander cleans up before we erase things
    {
    fake::SCEVExpander Exp(SE, Header->getParent()->getParent()->getDataLayout(), "enzyme");

    for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
        PHINode *PN = cast<PHINode>(II);
        if (PN == CanonicalIV) continue;
        if (PN->getType()->isPointerTy()) continue;
        if (!SE.isSCEVable(PN->getType())) continue;
        const SCEV *S = SE.getSCEV(PN);
        if (SE.getCouldNotCompute() == S) continue;
        Value *NewIV = Exp.expandCodeFor(S, S->getType(), CanonicalIV);
        if (NewIV == PN) {
          continue;
        }

        PN->replaceAllUsesWith(NewIV);
        IVsToRemove.push_back(PN);
    }
    }
    
    for (Instruction *PN : IVsToRemove) {
      gutils.erase(PN);
    }

    if (latches.size() == 1 && isa<BranchInst>(latches[0]->getTerminator()) && cast<BranchInst>(latches[0]->getTerminator())->isConditional())
    for (auto use : CanonicalIV->users()) {
      if (auto cmp = dyn_cast<ICmpInst>(use)) {
        if (cast<BranchInst>(latches[0]->getTerminator())->getCondition() != cmp) continue;
        // Force i to be on LHS
        if (cmp->getOperand(0) != CanonicalIV) {
          //Below also swaps predicate correctly
          cmp->swapOperands();
        }
        assert(cmp->getOperand(0) == CanonicalIV);

        auto scv = SE.getSCEVAtScope(cmp->getOperand(1), L);
        if (cmp->isUnsigned() || (scv != SE.getCouldNotCompute() && SE.isKnownNonNegative(scv)) ) {

          // valid replacements (since unsigned comparison and i starts at 0 counting up)

          // * i < n => i != n, valid since first time i >= n occurs at i == n
          if (cmp->getPredicate() == ICmpInst::ICMP_ULT || cmp->getPredicate() == ICmpInst::ICMP_SLT) {
            cmp->setPredicate(ICmpInst::ICMP_NE);
            goto cend;
          }

          // * i <= n => i != n+1, valid since first time i > n occurs at i == n+1 [ which we assert is in bitrange as not infinite loop ]
          if (cmp->getPredicate() == ICmpInst::ICMP_ULE || cmp->getPredicate() == ICmpInst::ICMP_SLE) {
            IRBuilder <>builder (Preheader->getTerminator());
            if (auto inst = dyn_cast<Instruction>(cmp->getOperand(1))) {
              builder.SetInsertPoint(inst->getNextNode());
            }
            cmp->setOperand(1, builder.CreateNUWAdd(cmp->getOperand(1), ConstantInt::get(cmp->getOperand(1)->getType(), 1, false)));
            cmp->setPredicate(ICmpInst::ICMP_NE);
            goto cend;
          }

          // * i >= n => i == n, valid since first time i >= n occurs at i == n
          if (cmp->getPredicate() == ICmpInst::ICMP_UGE || cmp->getPredicate() == ICmpInst::ICMP_SGE) {
            cmp->setPredicate(ICmpInst::ICMP_EQ);
            goto cend;
          }

          // * i > n => i == n+1, valid since first time i > n occurs at i == n+1 [ which we assert is in bitrange as not infinite loop ]
          if (cmp->getPredicate() == ICmpInst::ICMP_UGT || cmp->getPredicate() == ICmpInst::ICMP_SGT) {
            IRBuilder <>builder (Preheader->getTerminator());
            if (auto inst = dyn_cast<Instruction>(cmp->getOperand(1))) {
              builder.SetInsertPoint(inst->getNextNode());
            }
            cmp->setOperand(1, builder.CreateNUWAdd(cmp->getOperand(1), ConstantInt::get(cmp->getOperand(1)->getType(), 1, false)));
            cmp->setPredicate(ICmpInst::ICMP_EQ);
            goto cend;
          }
        }
        cend:;
        if (cmp->getPredicate() == ICmpInst::ICMP_NE) {

        }
      }
    }


    // Replace previous increment usage with new increment value
    if (increment) {
      increment->moveAfter(CanonicalIV->getParent()->getFirstNonPHI());
      std::vector<Instruction*> toerase;
      for(auto use : CanonicalIV->users()) {
        auto bo = dyn_cast<BinaryOperator>(use);
        
        if (bo == nullptr) continue;
        if (bo->getOpcode() != BinaryOperator::Add) continue;
        if (use == increment) continue;

        Value* toadd = nullptr;
        if (bo->getOperand(0) == CanonicalIV) {
          toadd = bo->getOperand(1);
        } else {
          assert(bo->getOperand(1) == CanonicalIV);
          toadd = bo->getOperand(0);
        }
        if (auto ci = dyn_cast<ConstantInt>(toadd)) {
          if (!ci->isOne()) continue;
          bo->replaceAllUsesWith(increment);
          toerase.push_back(bo);
        } else {
          continue;
        }
      }
      for(auto inst: toerase) {
        gutils.erase(inst);
      }

      if (latches.size() == 1 && isa<BranchInst>(latches[0]->getTerminator()) && cast<BranchInst>(latches[0]->getTerminator())->isConditional())
      for (auto use : increment->users()) {
        if (auto cmp = dyn_cast<ICmpInst>(use)) {
          if (cast<BranchInst>(latches[0]->getTerminator())->getCondition() != cmp) continue;

          // Force i+1 to be on LHS
          if (cmp->getOperand(0) != increment) {
            //Below also swaps predicate correctly
            cmp->swapOperands();
          }
          assert(cmp->getOperand(0) == increment);

          auto scv = SE.getSCEVAtScope(cmp->getOperand(1), L);
          if (cmp->isUnsigned() || (scv != SE.getCouldNotCompute() && SE.isKnownNonNegative(scv)) ) {

            // valid replacements (since unsigned comparison and i starts at 0 counting up)

            // * i+1 < n => i+1 != n, valid since first time i+1 >= n occurs at i+1 == n
            if (cmp->getPredicate() == ICmpInst::ICMP_ULT || cmp->getPredicate() == ICmpInst::ICMP_SLT) {
              cmp->setPredicate(ICmpInst::ICMP_NE);
              continue;
            }

            // * i+1 <= n => i != n, valid since first time i+1 > n occurs at i+1 == n+1 => i == n
            if (cmp->getPredicate() == ICmpInst::ICMP_ULE || cmp->getPredicate() == ICmpInst::ICMP_SLE) {
              cmp->setOperand(0, CanonicalIV);
              cmp->setPredicate(ICmpInst::ICMP_NE);
              continue;
            }

            // * i+1 >= n => i+1 == n, valid since first time i+1 >= n occurs at i+1 == n
            if (cmp->getPredicate() == ICmpInst::ICMP_UGE || cmp->getPredicate() == ICmpInst::ICMP_SGE) {
              cmp->setPredicate(ICmpInst::ICMP_EQ);
              continue;
            }

            // * i+1 > n => i == n, valid since first time i+1 > n occurs at i+1 == n+1 => i == n
            if (cmp->getPredicate() == ICmpInst::ICMP_UGT || cmp->getPredicate() == ICmpInst::ICMP_SGT) {
              cmp->setOperand(0, CanonicalIV);
              cmp->setPredicate(ICmpInst::ICMP_EQ);
              continue;
            }
          }
        }
      }

    }
}

bool getContextM(BasicBlock *BB, LoopContext &loopContext, std::map<Loop*,LoopContext> &loopContexts, LoopInfo &LI,ScalarEvolution &SE,DominatorTree &DT, GradientUtils &gutils) {
    Loop* L = LI.getLoopFor(BB);

    //Not inside a loop
    if (L == nullptr) return false;

    //Already canonicalized
    if (loopContexts.find(L) == loopContexts.end()) {
        
        loopContexts[L].parent = L->getParentLoop();

        loopContexts[L].header = L->getHeader();
        assert(loopContexts[L].header && "loop must have header");

        loopContexts[L].preheader = L->getLoopPreheader();
        assert(loopContexts[L].preheader && "loop must have preheader");
        
        loopContexts[L].latchMerge = nullptr;
    
        fake::SCEVExpander::getExitBlocks(L, loopContexts[L].exitBlocks);
        if (loopContexts[L].exitBlocks.size() == 0) {
            llvm::errs() << "newFunc: " << *BB->getParent() << "\n";
            llvm::errs() << "L: " << *L << "\n";
        }
        //assert(loopContexts[L].exitBlocks.size() > 0);

        auto pair = insertNewCanonicalIV(L, Type::getInt64Ty(BB->getContext()));
        PHINode* CanonicalIV = pair.first;
        assert(CanonicalIV);
        loopContexts[L].var = CanonicalIV;
        loopContexts[L].incvar = pair.second;
        removeRedundantIVs(L, loopContexts[L].header, loopContexts[L].preheader, CanonicalIV, SE, gutils, pair.second, fake::SCEVExpander::getLatches(L, loopContexts[L].exitBlocks));
        loopContexts[L].antivaralloc = IRBuilder<>(gutils.inversionAllocs).CreateAlloca(CanonicalIV->getType(), nullptr, CanonicalIV->getName()+"'ac");
      
        PredicatedScalarEvolution PSE(SE, *L);
        //predicate.addPredicate(SE.getWrapPredicate(SE.getSCEV(CanonicalIV), SCEVWrapPredicate::IncrementNoWrapMask));
        // Note exitcount needs the true latch (e.g. the one that branches back to header)
        // tather than the latch that contains the branch (as we define latch)
        const SCEV *Limit = PSE.getBackedgeTakenCount(); //getExitCount(L, ExitckedgeTakenCountBlock); //L->getLoopLatch());

            Value *LimitVar = nullptr;

            if (SE.getCouldNotCompute() != Limit) {
            // rerun canonicalization to ensure we have canonical variable equal to limit type
            //CanonicalIV = canonicalizeIVs(Exp, Limit->getType(), L, DT, &gutils);

            if (CanonicalIV == nullptr) {
                report_fatal_error("Couldn't get canonical IV.");
            }

                if (Limit->getType() != CanonicalIV->getType())
                    Limit = SE.getZeroExtendExpr(Limit, CanonicalIV->getType());

                fake::SCEVExpander Exp(SE, BB->getParent()->getParent()->getDataLayout(), "enzyme");
                LimitVar = Exp.expandCodeFor(Limit, CanonicalIV->getType(), loopContexts[L].preheader->getTerminator());
                loopContexts[L].dynamic = false;
            } else {
            //llvm::errs() << "Se has any info: " << SE.getBackedgeTakenInfo(L).hasAnyInfo() << "\n";
            llvm::errs() << "SE could not compute loop limit.\n";
        
            //TODO should eventually ensure this is freed
            LimitVar = gutils.createCacheForScope(loopContexts[L].preheader, CanonicalIV->getType(), "loopLimit", /*shouldfree*/false);

              for(auto ExitBlock: loopContexts[L].exitBlocks) {
                  IRBuilder <> B(&ExitBlock->front());
                  auto herephi = B.CreatePHI(CanonicalIV->getType(), 1);

                  for (BasicBlock *Pred : predecessors(ExitBlock)) {
                    if (LI.getLoopFor(Pred) == L) {
                        herephi->addIncoming(CanonicalIV, Pred);
                    } else {
                        herephi->addIncoming(ConstantInt::get(CanonicalIV->getType(), 0), Pred);
                    }
                  }

                  gutils.storeInstructionInCache(loopContexts[L].preheader, herephi, cast<AllocaInst>(LimitVar));
              }
              loopContexts[L].dynamic = true;
            }
            loopContexts[L].limit = LimitVar;
    }

    loopContext = loopContexts.find(L)->second;
    return true;
}

Value* GradientUtils::lookupM(Value* val, IRBuilder<>& BuilderM) {
    if (isa<Constant>(val)) {
        return val;
    }
    if (isa<BasicBlock>(val)) {
        return val;
    }
    if (isa<Function>(val)) {
        return val;
    }
    if (isa<UndefValue>(val)) {
        return val;
    }
    if (isa<Argument>(val)) {
        return val;
    }
    if (isa<MetadataAsValue>(val)) {
        return val;
    }
    if (!isa<Instruction>(val)) {
        llvm::errs() << *val << "\n";
    }

    auto inst = cast<Instruction>(val);
    if (inversionAllocs && inst->getParent() == inversionAllocs) {
        return val;
    }

    if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
        if (BuilderM.GetInsertBlock()->size() && BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
            Instruction* use = &*BuilderM.GetInsertPoint();
            while (isa<PHINode>(use)) use = use->getNextNode();
            if (DT.dominates(inst, use)) {
                //llvm::errs() << "allowed " << *inst << "from domination\n";
                return inst;
            } else {
                llvm::errs() << *BuilderM.GetInsertBlock()->getParent() << "\n";
                llvm::errs() << "didnt dominate inst: " << *inst << "  point: " << *BuilderM.GetInsertPoint() << "\nbb: " << *BuilderM.GetInsertBlock() << "\n";
            }
        } else {
            if (inst->getParent() == BuilderM.GetInsertBlock() || DT.dominates(inst, BuilderM.GetInsertBlock())) {
                //llvm::errs() << "allowed " << *inst << "from block domination\n";
                return inst;
            } else {
                llvm::errs() << *BuilderM.GetInsertBlock()->getParent() << "\n";
                llvm::errs() << "didnt dominate inst: " << *inst << "\nbb: " << *BuilderM.GetInsertBlock() << "\n";
            }
        }
    }
    val = inst = fixLCSSA(inst, BuilderM);

    assert(!this->isOriginalBlock(*BuilderM.GetInsertBlock()));

    auto idx = std::make_pair(val, BuilderM.GetInsertBlock());
    if (lookup_cache.find(idx) != lookup_cache.end()) {
        return lookup_cache[idx];
    }

    LoopContext lc;
    bool inLoop = getContext(inst->getParent(), lc);

    ValueToValueMapTy available;
    if (inLoop) {
        for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx)) {
          if (!isOriginalBlock(*BuilderM.GetInsertBlock())) {
            available[idx.var] = BuilderM.CreateLoad(idx.antivaralloc);
          } else {
            available[idx.var] = idx.var;
          }
          if (idx.parent == nullptr) break;
        }
    }

    bool legalRecompute = true;
    if (isa<LoadInst>(inst) && originalInstructions.find(inst) != originalInstructions.end()) {
        auto found = can_modref_map->find(getOriginal(inst));
        if(found == can_modref_map->end()) {
            llvm::errs() << "can_modref_map:\n"; 
            for(auto& pair : *can_modref_map) {
                llvm::errs() << " + " << *pair.first << ": " << pair.second << " of func " << pair.first->getParent()->getParent()->getName() << "\n";
            }
            llvm::errs() << "couldn't find in can_modref_map: " << *getOriginal(inst) << " in fn: " << getOriginal(inst)->getParent()->getParent()->getName();
        }
        assert(found != can_modref_map->end());
        legalRecompute = !found->second;
    }

    if (legalRecompute) {
      if (shouldRecompute(inst, available)) {
          auto op = unwrapM(inst, BuilderM, available, /*lookupIfAble*/true);
          assert(op);
          return op;
      }
    }

    ensureLookupCached(inst);
    assert(scopeMap[inst]);
    Value* result = lookupValueFromCache(BuilderM, inst->getParent(), scopeMap[inst]);
    assert(result->getType() == inst->getType());
    lookup_cache[idx] = result;
    return result;
}

bool GradientUtils::getContext(BasicBlock* BB, LoopContext& loopContext) {
    return getContextM(BB, loopContext, this->loopContexts, this->LI, this->SE, this->DT, *this);
}

//! Given a map of edges we could have taken to desired target, compute a value that determines which target should be branched to
//  This function attempts to determine an equivalent condition from earlier in the code and use that if possible, falling back to creating a phi node of which edge was taken if necessary
//  This function can be used in two ways:
//   * If replacePHIs is null (usual case), this function does the branch
//   * If replacePHIs isn't null, do not perform the branch and instead replace the PHI's with the derived condition as to whether we should branch to a particular target
void GradientUtils::branchToCorrespondingTarget(BasicBlock* ctx, IRBuilder <>& BuilderM, const std::map<BasicBlock*, std::vector<std::pair</*pred*/BasicBlock*,/*successor*/BasicBlock*>>> &targetToPreds, const std::map<BasicBlock*,PHINode*>* replacePHIs) {
  assert(targetToPreds.size() > 0);
  if (replacePHIs) {
      if (replacePHIs->size() == 0) return;

      for(auto x: *replacePHIs) {
          assert(targetToPreds.find(x.first) != targetToPreds.end());
      }
  }

  if (targetToPreds.size() == 1) {
      if (replacePHIs == nullptr) {
          assert(BuilderM.GetInsertBlock()->size() == 0 || !isa<BranchInst>(BuilderM.GetInsertBlock()->back()));
          BuilderM.CreateBr( targetToPreds.begin()->first );
      } else {
          for (auto pair : *replacePHIs) {
              pair.second->replaceAllUsesWith(ConstantInt::getTrue(pair.second->getContext()));
              pair.second->eraseFromParent();
          }
      }
      return;
  }
        
  // Map of function edges to list of targets this can branch to we have 
  std::map<std::pair</*pred*/BasicBlock*,/*successor*/BasicBlock*>,std::set<BasicBlock*>> done;
  {
        std::deque<std::tuple<std::pair</*pred*/BasicBlock*,/*successor*/BasicBlock*>,BasicBlock*>> Q; // newblock, target

        for (auto pair: targetToPreds) {
          for (auto pred_edge : pair.second) {
              Q.push_back(std::make_pair(pred_edge, pair.first));
          }
        }

        for(std::tuple<std::pair</*pred*/BasicBlock*,/*successor*/BasicBlock*>,BasicBlock*> trace; Q.size() > 0;) {
              trace = Q.front();
              Q.pop_front();
              auto edge = std::get<0>(trace);
              auto block = edge.first;
              //auto blocksuc = edge.second;
              auto target = std::get<1>(trace);
              //llvm::errs() << " seeing Q edge [" << block->getName() << "," << blocksuc->getName() << "] " << " to target " << target->getName() << "\n";

              if (done[edge].count(target)) continue;
              done[edge].insert(target);

              Loop* blockLoop = LI.getLoopFor(block);

              for (BasicBlock *Pred : predecessors(block)) {
                //llvm::errs() << " seeing in pred [" << Pred->getName() << "," << block->getName() << "] to target " << target->getName() << "\n";
                // Don't go up the backedge as we can use the last value if desired via lcssa
                if (blockLoop && blockLoop->getHeader() == block && blockLoop == LI.getLoopFor(Pred)) continue;    

                Q.push_back(std::tuple<std::pair<BasicBlock*,BasicBlock*>,BasicBlock*>(std::make_pair(Pred, block), target ));
                //llvm::errs() << " adding to Q pred [" << Pred->getName() << "," << block->getName() << "] to target " << target->getName() << "\n";
              }
        }
  }

  IntegerType* T = (targetToPreds.size() == 2) ? Type::getInt1Ty(BuilderM.getContext()) : Type::getInt8Ty(BuilderM.getContext());

  Instruction* equivalentTerminator = nullptr;
  
  std::set<BasicBlock*> blocks;
  for(auto pair : done) {
      //const auto& targets = pair.second;
      const auto& edge = pair.first;
      //llvm::errs() << " edge: (" << edge.first->getName() << "," << edge.second->getName() << ")\n";
      //llvm::errs() << "   targets: [";
      //for(auto t : targets) llvm::errs() << t->getName() << ", ";
      //llvm::errs() << "]\n";
      blocks.insert(edge.first);
  }

  for(auto block : blocks) {
      std::set<BasicBlock*> foundtargets;
      for (BasicBlock* succ : successors(block)) {
          auto edge = std::make_pair(block, succ);
          if (done[edge].size() != 1) {
              //llvm::errs() << " | failed to use multisuccessor edge [" << block->getName() << "," << succ->getName() << "\n";
              goto nextpair;
          }
          BasicBlock* target = *done[edge].begin();
          if (foundtargets.find(target) != foundtargets.end()) {
              //llvm::errs() << " | double target for block edge [" << block->getName() << "," << succ->getName() << "\n";
              goto nextpair;
          }
          foundtargets.insert(target);
      }
      if (foundtargets.size() != targetToPreds.size()) {
          //llvm::errs() << " | failed to use " << block->getName() << " since noneq targets\n";
          goto nextpair;
      }
      equivalentTerminator = block->getTerminator();
      goto fast;
      
      nextpair:; 
  }
  goto nofast;


  fast:;
  assert(equivalentTerminator);

  if (auto branch = dyn_cast<BranchInst>(equivalentTerminator)) {
      BasicBlock* block = equivalentTerminator->getParent();
      assert(branch->getCondition());

      assert(branch->getCondition()->getType() == T);

      AllocaInst* cache = createCacheForScope(ctx, T, "", /*shouldFree*/true);
      IRBuilder<> pbuilder(equivalentTerminator);
      pbuilder.setFastMathFlags(getFast());
      storeInstructionInCache(ctx, pbuilder, branch->getCondition(), cache);

      Value* phi = lookupValueFromCache(BuilderM, ctx, cache);

      if (replacePHIs == nullptr) {
          assert(BuilderM.GetInsertBlock()->size() == 0 || !isa<BranchInst>(BuilderM.GetInsertBlock()->back()));
          BuilderM.CreateCondBr(phi, *done[std::make_pair(block, branch->getSuccessor(0))].begin(), *done[std::make_pair(block, branch->getSuccessor(1))].begin());
      } else {
          for (auto pair : *replacePHIs) {
              Value* val = nullptr;
              if (pair.first == *done[std::make_pair(block, branch->getSuccessor(0))].begin()) {
                  val = phi;
              } else if (pair.first == *done[std::make_pair(block, branch->getSuccessor(1))].begin()) {
                  val = BuilderM.CreateNot(phi);
              } else {
                  llvm::errs() << *pair.first->getParent() << "\n";
                  llvm::errs() << *pair.first << "\n";
                  llvm::errs() << *branch << "\n";
                  llvm_unreachable("unknown successor for replacephi");
              }
              if (&*BuilderM.GetInsertPoint() == pair.second) {
                if (pair.second->getNextNode())
                  BuilderM.SetInsertPoint(pair.second->getNextNode());
                else
                  BuilderM.SetInsertPoint(pair.second->getParent());
              }
              pair.second->replaceAllUsesWith(val);
              pair.second->eraseFromParent();
          }
      }
  } else if (auto si = dyn_cast<SwitchInst>(equivalentTerminator)) {
      BasicBlock* block = equivalentTerminator->getParent();

      IRBuilder<> pbuilder(equivalentTerminator);
      pbuilder.setFastMathFlags(getFast());

      AllocaInst* cache = createCacheForScope(ctx, si->getCondition()->getType(), "", /*shouldFree*/true);
      Value* condition = si->getCondition();
      storeInstructionInCache(ctx, pbuilder, condition, cache);

      Value* phi = lookupValueFromCache(BuilderM, ctx, cache);


      if (replacePHIs == nullptr) {
          SwitchInst* swtch = BuilderM.CreateSwitch(phi, *done[std::make_pair(block, si->getDefaultDest())].begin());
          for (auto switchcase : si->cases()) {
              swtch->addCase(switchcase.getCaseValue(), *done[std::make_pair(block, switchcase.getCaseSuccessor())].begin());
          }
      } else {
          for (auto pair : *replacePHIs) {
              Value* cas = si->findCaseDest(pair.first);
              Value* val = nullptr;
              if (cas) {
                  val = BuilderM.CreateICmpEQ(cas, phi);
              } else {
                  //default case
                  val = ConstantInt::getTrue(pair.second->getContext());
                  for(auto switchcase : si->cases()) {
                      val = BuilderM.CreateOr(val, BuilderM.CreateICmpEQ(switchcase.getCaseValue(), phi));
                  }
                  val = BuilderM.CreateNot(val);
              }
              if (&*BuilderM.GetInsertPoint() == pair.second) {
                if (pair.second->getNextNode())
                  BuilderM.SetInsertPoint(pair.second->getNextNode());
                else
                  BuilderM.SetInsertPoint(pair.second->getParent());
              }
              pair.second->replaceAllUsesWith(val);
              pair.second->eraseFromParent();
          }
      }
  } else {
      llvm::errs() << "unknown equivalent terminator\n";
      llvm::errs() << *equivalentTerminator << "\n";
      llvm_unreachable("unknown equivalent terminator");
  }
  return;


  nofast:;

  AllocaInst* cache = createCacheForScope(ctx, T, "", /*shouldFree*/true);
  std::vector<BasicBlock*> targets;
  {
  size_t idx = 0;
  std::map<BasicBlock* /*storingblock*/, std::map<ConstantInt* /*target*/, std::vector<BasicBlock*> /*predecessors*/ > >
      storing;
  for(const auto &pair: targetToPreds) {
      for(auto pred : pair.second) {
          storing[pred.first][ConstantInt::get(T, idx)].push_back(pred.second);
      }
      targets.push_back(pair.first);
      idx++;
  }
  assert(targets.size() > 0);

  for(const auto &pair: storing) {
      IRBuilder<> pbuilder(pair.first);

      if (pair.first->getTerminator())
          pbuilder.SetInsertPoint(pair.first->getTerminator());

      pbuilder.setFastMathFlags(getFast());

      Value* tostore = ConstantInt::get(T, 0);

      if (pair.second.size() == 1) {
          tostore = pair.second.begin()->first;
      } else {
          assert(0 && "multi exit edges not supported");
          exit(1);
         //for(auto targpair : pair.second) {
         //     tostore = pbuilder.CreateOr(tostore, pred);
         //}
      }
      storeInstructionInCache(ctx, pbuilder, tostore, cache);
  }
  }
  
  Value* which = lookupValueFromCache(BuilderM, ctx, cache);
  assert(which);
  assert(which->getType() == T);
  
  if (replacePHIs == nullptr) {
      if (targetToPreds.size() == 2) {
          assert(BuilderM.GetInsertBlock()->size() == 0 || !isa<BranchInst>(BuilderM.GetInsertBlock()->back()));
          BuilderM.CreateCondBr(which, /*true*/targets[1], /*false*/targets[0]);
      } else {
          assert(targets.size() > 0);
          //llvm::errs() << "which: " << *which << "\n";
          //llvm::errs() << "targets.back(): " << *targets.back() << "\n";
          auto swit = BuilderM.CreateSwitch(which, targets.back(), targets.size()-1);
          for(unsigned i=0; i<targets.size()-1; i++) {
            swit->addCase(ConstantInt::get(T, i), targets[i]);
          }
      }
  } else {
      for(unsigned i=0; i<targets.size(); i++) {
          auto found = replacePHIs->find(targets[i]);
          if (found == replacePHIs->end()) continue;

          Value* val = nullptr;
          if (targets.size() == 2 && i == 0) {
              val = BuilderM.CreateNot(which);
          } else if (targets.size() == 2 && i == 1) {
              val = which;
          } else {
              val = BuilderM.CreateICmpEQ(ConstantInt::get(T, i), which);
          }
          if (&*BuilderM.GetInsertPoint() == found->second) {
            if (found->second->getNextNode())
              BuilderM.SetInsertPoint(found->second->getNextNode());
            else
              BuilderM.SetInsertPoint(found->second->getParent());
          }
          found->second->replaceAllUsesWith(val);
          found->second->eraseFromParent();
      }
  }
  return;

}
