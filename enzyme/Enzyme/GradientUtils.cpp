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

#include "llvm/IR/InstrTypes.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"

#include <algorithm>

//! Is possibleChild a child loop or the same loop as possibleParent
static bool isParentOrSameContext(LoopContext & possibleChild, LoopContext & possibleParent) {
    if (possibleChild.header == possibleParent.header) return true;
    
    for(Loop* lp = possibleChild.parent; lp != nullptr; lp = lp->getParentLoop()) {
        if (lp->getHeader() == possibleParent.header) return true;
    }
    return false;
}

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
    if (!inLoop) return reverseBlocks[BB];
    
    auto latches = fake::SCEVExpander::getLatches(LI.getLoopFor(BB), lc.exitBlocks);
    if (std::find(latches.begin(), latches.end(), BB) == latches.end()) return reverseBlocks[BB];

    LoopContext branchingContext;
    bool inLoopContext = getContext(branchingBlock, branchingContext);

    assert(lc.latchMerge);

    // if this is a branch into the loop, this certainly should go to the merged
    //  block as this represents starting the loop
    if (!inLoopContext || !isParentOrSameContext(branchingContext, lc) ) {
        //llvm::errs() << "LC BB:" << BB->getName() << " branchingBlock:" << branchingBlock->getName() << "\n";
        return lc.latchMerge;
    }

    // if we branch from inside the loop, we only need to go to the merged loop
    //   if the original branch is to the header (otherwise its an internal branch in the loop)
    if (branchingBlock == lc.header) {
        //llvm::errs() << "LH BB:" << BB->getName() << " branchingBlock:" << branchingBlock->getName() << "\n";
        return lc.latchMerge;
    }

    //llvm::errs() << " BB:" << BB->getName() << " branchingBlock:" << branchingBlock->getName() << "\n";
    return reverseBlocks[BB];
  }

  void GradientUtils::forceContexts(bool setupMerge) {
    for(auto BB : originalBlocks) {
        LoopContext lc;
        getContext(BB, lc);
    }

	if (setupMerge) {
        for(auto pair : loopContexts) {
			auto &lc = pair.second;

            lc.latchMerge = BasicBlock::Create(newFunc->getContext(), "loopMerge", newFunc);
            loopContexts[pair.first].latchMerge = lc.latchMerge;
            {
                LoopContext bar;
                getContext(lc.header, bar);
                assert(bar.latchMerge == lc.latchMerge);
            }
            lc.latchMerge->getInstList().push_front(lc.antivar);

			IRBuilder<> mergeBuilder(lc.latchMerge);
            PHINode* firstiter = mergeBuilder.CreatePHI(Type::getInt1Ty(mergeBuilder.getContext()), 1);
			Instruction* sub = cast<Instruction>(mergeBuilder.CreateSub(lc.antivar, ConstantInt::get(lc.antivar->getType(), 1)));

            auto latches = fake::SCEVExpander::getLatches(LI.getLoopFor(lc.header), lc.exitBlocks);
            
            for(BasicBlock* exit : lc.exitBlocks) {
                IRBuilder<> tbuild(reverseBlocks[exit]);
                Value* lim = nullptr;
                if (lc.dynamic) {
                    lim = lookupValueFromCache(tbuild, lc.preheader, cast<AllocaInst>(lc.limit));
                } else {
                    lim = lookupM(lc.limit, tbuild);
                }
                lc.antivar->addIncoming(lim, reverseBlocks[exit]);
                firstiter->addIncoming(ConstantInt::getTrue(mergeBuilder.getContext()), reverseBlocks[exit]);
            }            

			lc.antivar->addIncoming(sub, reverseBlocks[lc.header]);
            firstiter->addIncoming(ConstantInt::getFalse(mergeBuilder.getContext()), reverseBlocks[lc.header]);

			if (latches.size() == 1) {
                lc.latchMerge->takeName(reverseBlocks[latches[0]]);
                reverseBlocks[latches[0]]->setName(lc.latchMerge->getName()+"_exit");
                lc.latchMerge->moveBefore(reverseBlocks[latches[0]]);
            }

            std::map<BasicBlock*,std::vector<std::pair</*pred*/BasicBlock*,/*succ*/BasicBlock*>>> targetToPreds;

            for(BasicBlock* exit : lc.exitBlocks) {
                for(auto pred : predecessors(exit)) {
                    auto fd = std::find(latches.begin(), latches.end(), pred);
                    if ( fd != latches.end()) {
                        auto latch = *fd;
                        targetToPreds[reverseBlocks[latch]].push_back(std::make_pair(pred, exit));
                    }
                }
            }

            BasicBlock* backlatch = nullptr;
            for(auto blk : predecessors(lc.header)) {
               if (blk == lc.preheader) continue;
               assert(backlatch == nullptr);
               backlatch = blk;
            }
            assert(backlatch != nullptr);
 
            this->branchToCorrespondingTarget(lc.preheader, mergeBuilder, targetToPreds);
            Instruction* termInst = lc.latchMerge->getTerminator();
            SmallVector<BasicBlock*, 4> succ;
            for(BasicBlock* suc : successors(lc.latchMerge)) {
              succ.push_back(suc);
            }
            assert(termInst);
            mergeBuilder.SetInsertPoint(termInst);

            // ensure we only start at the correct exiting block on the first reverse iteration, otherwise
            //  we want to branch to the backlatch edge

            // Case 1: The correct exiting block terminator unconditionally branches to the backlatch we need to do for all other iterations, no modification
            if(succ.size() == 1 && (reverseBlocks[backlatch] == succ[0]) ) {
                //Do nothing here, remove helper firstiter
                firstiter->eraseFromParent();
                
            // Case 2: The correct exiting block terminator unconditionally branches a different block, change to a conditional branch depending on if we are the first iteration
            } else if (succ.size() == 1) {

                // If first iteration, branch to the exiting block, otherwise the backlatch
                mergeBuilder.CreateCondBr(firstiter, succ[0], reverseBlocks[backlatch]);
                
            // Case 3: The correct exiting block terminator conditionally branches to the backlatch different block, change to a conditional branch depending on if we are the first iteration
            } else if(succ.size() == 2 && (reverseBlocks[backlatch] == succ[0] || reverseBlocks[backlatch] == succ[1]) ) {
                auto branch = cast<BranchInst>(termInst);
                if (reverseBlocks[backlatch] == succ[0]) {
                    // if we branch to backlatch on true, modify condition to branch if usual condition or not the first iteration
                    branch->setCondition(mergeBuilder.CreateOr(branch->getCondition(), mergeBuilder.CreateNot(firstiter)));
                } else {
                    assert(reverseBlocks[backlatch] == succ[1]);
                    // if we branch to backlatch on false (ie go to special exit on true), modify condition to only go to special exit if usual condition and first iteration
                    branch->setCondition(mergeBuilder.CreateAnd(branch->getCondition(), firstiter));
                }
            
            // Case 4 (default fallback): First branch depending on first iteration or not, then branch on the special exit
            } else {
                BasicBlock* splitBlock = lc.latchMerge->splitBasicBlock(sub->getNextNode());
                assert(cast<BranchInst>(lc.latchMerge->getTerminator())->getNumSuccessors() == 1);

                lc.latchMerge->getTerminator()->eraseFromParent();
                mergeBuilder.SetInsertPoint(lc.latchMerge);
                mergeBuilder.CreateCondBr(firstiter, splitBlock, reverseBlocks[backlatch]);

            }
        }
	}
  }

bool shouldRecompute(Value* val, const ValueToValueMapTy& available) {
  if (available.count(val)) return false;
  if (isa<Argument>(val) || isa<Constant>(val)) {
    return false;
  } else if (auto op = dyn_cast<CastInst>(val)) {
    return shouldRecompute(op->getOperand(0), available);
  } else if (isa<AllocaInst>(val)) {
    return true;
  } else if (auto op = dyn_cast<BinaryOperator>(val)) {
    bool a0 = shouldRecompute(op->getOperand(0), available);
    if (a0) {
        //llvm::errs() << "need recompute: " << *op->getOperand(0) << "\n";
    }
    bool a1 = shouldRecompute(op->getOperand(1), available);
    if (a1) {
        //llvm::errs() << "need recompute: " << *op->getOperand(1) << "\n";
    }
    return a0 || a1;
  } else if (auto op = dyn_cast<CmpInst>(val)) {
    return shouldRecompute(op->getOperand(0), available) || shouldRecompute(op->getOperand(1), available);
  } else if (auto op = dyn_cast<SelectInst>(val)) {
    return shouldRecompute(op->getOperand(0), available) || shouldRecompute(op->getOperand(1), available) || shouldRecompute(op->getOperand(2), available);
  } else if (auto load = dyn_cast<LoadInst>(val)) {
    Value* idx = load->getOperand(0);
    while (!isa<Argument>(idx)) {
      if (auto gep = dyn_cast<GetElementPtrInst>(idx)) {
        for(auto &a : gep->indices()) {
          if (shouldRecompute(a, available)) {
                        //llvm::errs() << "not recomputable: " << *a << "\n";
            return true;
          }
        }
        idx = gep->getPointerOperand();
      } else if(auto cast = dyn_cast<CastInst>(idx)) {
        idx = cast->getOperand(0);
      } else if(isa<CallInst>(idx)) {
            //} else if(auto call = dyn_cast<CallInst>(idx)) {
                //if (call->getCalledFunction()->getName() == "malloc")
                //    return false;
                //else
        {
                    //llvm::errs() << "unknown call " << *call << "\n";
          return true;
        }
      } else {
              //llvm::errs() << "not a gep " << *idx << "\n";
        return true;
      }
    }
    Argument* arg = cast<Argument>(idx);
    if (! ( arg->hasAttribute(Attribute::ReadOnly) || arg->hasAttribute(Attribute::ReadNone)) ) {
            //llvm::errs() << "argument " << *arg << " not marked read only\n";
      return true;
    }
    return false;
  } else if (auto phi = dyn_cast<PHINode>(val)) {
    if (phi->getNumIncomingValues () == 1) {
      bool b = shouldRecompute(phi->getIncomingValue(0) , available);
      if (b) {
            //llvm::errs() << "phi need recompute: " <<*phi->getIncomingValue(0) << "\n";
      }
      return b;
    }

    return true;
  } else if (auto op = dyn_cast<IntrinsicInst>(val)) {
    switch(op->getIntrinsicID()) {
      case Intrinsic::sin:
      case Intrinsic::cos:
      return false;
      return shouldRecompute(op->getOperand(0), available);
      default:
      return true;
    }
  }
  //llvm::errs() << "unknown inst " << *val << " unable to recompute\n";
  return true;
}

GradientUtils* GradientUtils::CreateFromClone(Function *todiff, AAResults &AA, TargetLibraryInfo &TLI, const std::set<unsigned> & constant_args, ReturnType returnValue, bool differentialReturn, llvm::Type* additionalArg) {
    assert(!todiff->empty());
    ValueToValueMapTy invertedPointers;
    SmallPtrSet<Value*,4> constants;
    SmallPtrSet<Value*,20> nonconstant;
    SmallPtrSet<Value*,2> returnvals;
    ValueToValueMapTy originalToNew;
    auto newFunc = CloneFunctionWithReturns(todiff, AA, TLI, invertedPointers, constant_args, constants, nonconstant, returnvals, /*returnValue*/returnValue, /*differentialReturn*/differentialReturn, "fakeaugmented_"+todiff->getName(), &originalToNew, /*diffeReturnArg*/false, additionalArg);
    auto res = new GradientUtils(newFunc, AA, TLI, invertedPointers, constants, nonconstant, returnvals, originalToNew);
    res->oldFunc = todiff;
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
  auto res = new DiffeGradientUtils(newFunc, AA, TLI, invertedPointers, constants, nonconstant, returnvals, originalToNew);
  res->oldFunc = todiff;
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
          report_fatal_error("cannot compute with global variable that doesn't have marked shadow global");
      }
      auto md = arg->getMetadata("enzyme_shadow");
      if (!isa<MDTuple>(md)) {
          llvm::errs() << *arg << "\n";
          llvm::errs() << *md << "\n";
          report_fatal_error("cannot compute with global variable that doesn't have marked shadow global (metadata incorrect type)");
      }
      auto md2 = cast<MDTuple>(md);
      assert(md2->getNumOperands() == 1);
      auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
      auto cs = gvemd->getValue();
      return invertedPointers[val] = cs;
    } else if (auto fn = dyn_cast<Function>(val)) {
      //! Todo allow tape propagation
      auto newf = CreatePrimalAndGradient(fn, /*constant_args*/{}, TLI, AA, /*returnValue*/false, /*differentialReturn*/fn->getReturnType()->isFPOrFPVectorTy(), /*topLevel*/false, /*additionalArg*/nullptr);
      return BuilderM.CreatePointerCast(newf, fn->getType());
    } else if (auto arg = dyn_cast<CastInst>(val)) {
      auto result = BuilderM.CreateCast(arg->getOpcode(), invertPointerM(arg->getOperand(0), BuilderM), arg->getDestTy(), arg->getName()+"'ipc");
      return result;
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
      auto li = bb.CreateBinOp(arg->getOpcode(), invertPointerM(arg->getOperand(0), bb), invertPointerM(arg->getOperand(1), bb), arg->getName());
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
        memset->addParamAttr(0, Attribute::getWithAlignment(inst->getContext(), inst->getAlignment()));
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
        if (!SE.isSCEVable(PN->getType())) continue;
        const SCEV *S = SE.getSCEV(PN);
        if (SE.getCouldNotCompute() == S) continue;
        Value *NewIV = Exp.expandCodeFor(S, S->getType(), CanonicalIV);
        if (NewIV == PN) {
          llvm::errs() << "TODO: odd case need to ensure replacement\n";
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
          llvm::errs() << "coing to think about " << *cmp << "\n";
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

        auto pair = insertNewCanonicalIV(L, Type::getInt64Ty(BB->getContext()));
        PHINode* CanonicalIV = pair.first;
        assert(CanonicalIV);
        removeRedundantIVs(L, loopContexts[L].header, loopContexts[L].preheader, CanonicalIV, SE, gutils, pair.second, fake::SCEVExpander::getLatches(L, loopContexts[L].exitBlocks));
        loopContexts[L].var = CanonicalIV;
        loopContexts[L].antivar = PHINode::Create(CanonicalIV->getType(), CanonicalIV->getNumIncomingValues(), CanonicalIV->getName()+"'phi");
      
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
        

              LimitVar = gutils.createCacheForScope(loopContexts[L].preheader, CanonicalIV->getType(), "loopLimit", nullptr, nullptr);

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
            if (DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
                //llvm::errs() << "allowed " << *inst << "from domination\n";
                return inst;
            }
        } else {
            if (DT.dominates(inst, BuilderM.GetInsertBlock())) {
                //llvm::errs() << "allowed " << *inst << "from block domination\n";
                return inst;
            }
        }
    }
    val = inst = fixLCSSA(inst, BuilderM);

    assert(!this->isOriginalBlock(*BuilderM.GetInsertBlock()));

    static std::map<std::pair<Value*, BasicBlock*>, Value*> cache;
    auto idx = std::make_pair(val, BuilderM.GetInsertBlock());
    if (cache.find(idx) != cache.end()) {
        return cache[idx];
    }

    LoopContext lc;
    bool inLoop = getContext(inst->getParent(), lc);

    ValueToValueMapTy available;
    if (inLoop) {
        for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx)) {
          if (!isOriginalBlock(*BuilderM.GetInsertBlock())) {
            available[idx.var] = idx.antivar;
          } else {
            available[idx.var] = idx.var;
          }
          if (idx.parent == nullptr) break;
        }
    }

    if (!shouldRecompute(inst, available)) {
        auto op = unwrapM(inst, BuilderM, available, /*lookupIfAble*/true);
        assert(op);
        return op;
    }
    /*
    if (!inLoop) {
        if (!isOriginalBlock(*BuilderM.GetInsertBlock()) && inst->getParent() == BuilderM.GetInsertBlock());
        todo here/re
    }
    */

    ensureLookupCached(inst);
    assert(scopeMap[inst]);
    Value* result = lookupValueFromCache(BuilderM, inst->getParent(), scopeMap[inst]);
    assert(result->getType() == inst->getType());
    cache[idx] = result;
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
  if (replacePHIs) {
      if (replacePHIs->size() == 0) return;

      for(auto x: *replacePHIs) {
          assert(targetToPreds.find(x.first) != targetToPreds.end());
      }
  }

  if (targetToPreds.size() == 1) {
      if (replacePHIs == nullptr) {
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
  CallInst* freeLocation;
  AllocaInst* cache = createCacheForScope(ctx, T, "", /*shouldFree*/&freeLocation, /*lastAlloca*/nullptr);

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

      IRBuilder<> pbuilder(equivalentTerminator);
      pbuilder.setFastMathFlags(getFast());
      storeInstructionInCache(ctx, pbuilder, branch->getCondition(), cache);

      Value* phi = lookupValueFromCache(BuilderM, ctx, cache);

      if (replacePHIs == nullptr) {
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
      assert(branch->getCondition());
      BasicBlock* block = equivalentTerminator->getParent();

      IRBuilder<> pbuilder(equivalentTerminator);
      pbuilder.setFastMathFlags(getFast());
      storeInstructionInCache(ctx, pbuilder, branch->getCondition(), cache);

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

  for(const auto &pair: storing) {
      assert(pair.first->getTerminator());
      assert(cast<Instruction>(pair.first->getTerminator()));
      IRBuilder<> pbuilder(pair.first->getTerminator());
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
          BuilderM.CreateCondBr(which, /*true*/targets[1], /*false*/targets[0]);
      } else {
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
