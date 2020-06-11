/*
 * GradientUtils.cpp - Gradient Utility data structures and functions
 *
 * Copyright (C) 2020 William S. Moses (enzyme@wsmoses.com) - All Rights Reserved
 *
 * For commercial use of this code please contact the author(s) above.
 */

#include "GradientUtils.h"

#include <llvm/Config/llvm-config.h>

#include "EnzymeLogic.h"

#include "FunctionUtils.h"

#include "LibraryFuncs.h"

#include "llvm/IR/GlobalValue.h"

#include "llvm/IR/Constants.h"

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"

#include <algorithm>

cl::opt<bool> efficientBoolCache(
            "enzyme_smallbool", cl::init(false), cl::Hidden,
            cl::desc("Place 8 bools together in a single byte"));


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
        Value* sub = tbuild.CreateAdd(av, ConstantInt::get(av->getType(), -1), "", /*NUW*/false, /*NSW*/true);
        tbuild.CreateStore(sub, lc.antivaralloc);
        tbuild.CreateBr(reverseBlocks[BB]);
        return newBlocksForLoop_cache[tup] = incB;
    }

    if (inLoop) {
        auto latches = getLatches(LI.getLoopFor(BB), lc.exitBlocks);

        if (std::find(latches.begin(), latches.end(), BB) != latches.end() && std::find(lc.exitBlocks.begin(), lc.exitBlocks.end(), branchingBlock) != lc.exitBlocks.end()) {
            BasicBlock* incB = BasicBlock::Create(BB->getContext(), "merge" + reverseBlocks[lc.header]->getName()+"_" + branchingBlock->getName(), BB->getParent());
            incB->moveAfter(reverseBlocks[branchingBlock]);

            IRBuilder<> tbuild(reverseBlocks[branchingBlock]);

            Value* lim = nullptr;
            if (lc.dynamic) {
                lim = lookupValueFromCache(tbuild, lc.preheader, cast<AllocaInst>(lc.limit), /*isi1*/false);
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

bool GradientUtils::legalRecompute(const Value* val, const ValueToValueMapTy& available) const {
  if (available.count(val)) {
    return true;
  }

  if (isa<PHINode>(val)) {
    if (auto dli = dyn_cast_or_null<LoadInst>(hasUninverted(val))) {
      return legalRecompute(dli, available); // TODO ADD && !TR.intType(getOriginal(dli), /*mustfind*/false).isPossibleFloat();
    }
    //if (SE.isSCEVable(phi->getType())) {
      //auto scev = const_cast<GradientUtils*>(this)->SE.getSCEV(const_cast<Value*>(val));
      //llvm::errs() << "phi: " << *val << " scev: " << *scev << "\n";
    //}
    //llvm::errs() << "illegal recompute: " << *val << "\n";
    return false;
  }

  if (isa<Instruction>(val) && cast<Instruction>(val)->getMetadata("enzyme_mustcache")) {
    //llvm::errs() << "illegal recompute: " << *val << "\n";
    return false;
  }

  // If this is a load from cache already, dont force a cache of this
  if (isa<LoadInst>(val) && cast<LoadInst>(val)->getMetadata("enzyme_fromcache")) return true;

  //TODO consider callinst here

  if (auto li = dyn_cast<LoadInst>(val)) {

    // If this is an already unwrapped value, legal to recompute again.
    if (li->getMetadata("enzyme_unwrapped"))
      return true;

    const Instruction* orig = nullptr;
    if (li->getParent()->getParent() == oldFunc) {
      orig = li;
    } else {
      orig = isOriginal(li);
    }

    if (orig) {
      auto found = can_modref_map->find(const_cast<Instruction*>(orig));
      //llvm::errs() << "legality of recomputing: " << *li << " is " << !found->second << "\n";
      if(found == can_modref_map->end()) {
          llvm::errs() << "can_modref_map:\n";
          for(auto& pair : *can_modref_map) {
              llvm::errs() << " + " << *pair.first << ": " << pair.second << " of func " << pair.first->getParent()->getParent()->getName() << "\n";
          }
          llvm::errs() << "couldn't find in can_modref_map: " << *getOriginal(li) << " in fn: " << orig->getParent()->getParent()->getName();
      }
      assert(found != can_modref_map->end());
      //llvm::errs() << " legal [ " << legalRecompute << " ] recompute of " << *inst << "\n";
      return !found->second;
    } else {
      if (auto dli = dyn_cast_or_null<LoadInst>(hasUninverted(li))) {
        return legalRecompute(dli, available);
      }

      // TODO mark all the explicitly legal nodes (caches, etc)
      return true;
      llvm::errs() << *li << " parent: " << li->getParent()->getParent()->getName() << "\n";
      llvm_unreachable("unknown load to redo!");
    }
  }

  if (auto ci = dyn_cast<CallInst>(val)) {
    if (auto called = ci->getCalledFunction()) {
      auto n = called->getName();
      if (n == "lgamma" || n == "lgammaf" || n == "lgammal" || n == "lgamma_r" || n == "lgammaf_r" || n == "lgammal_r"
        || n == "__lgamma_r_finite" || n == "__lgammaf_r_finite" || n == "__lgammal_r_finite"
        || n == "tanh" || n == "tanhf") {
        return true;
      }
    }
  }

  if (auto inst = dyn_cast<Instruction>(val)) {
    if (inst->mayReadOrWriteMemory()) {
      //llvm::errs() << "illegal recompute: " << *val << "\n";
      return false;
    }
  }

  return true;
}

//! Given the option to recompute a value or re-use an old one, return true if it is faster to recompute this value from scratch
bool GradientUtils::shouldRecompute(const Value* val, const ValueToValueMapTy& available) const {
  if (available.count(val)) return true;
  //TODO: remake such that this returns whether a load to a cache is more expensive than redoing the computation.

  // If this is a load from cache already, just reload this
  if (isa<LoadInst>(val) && cast<LoadInst>(val)->getMetadata("enzyme_fromcache")) return true;

  if (isa<CastInst>(val) || isa<GetElementPtrInst>(val)) return true;

  if (!isa<Instruction>(val)) return true;

  //llvm::errs() << " considering recompute of " << *val << "\n";
  const Instruction* inst = cast<Instruction>(val);

  // if this has operands that need to be loaded and haven't already been loaded (TODO), just cache this
  for(auto &op : inst->operands()) {
    //llvm::errs() << "   + " << *op << " legalRecompute:" << legalRecompute(op, available) << "\n";
    if (!legalRecompute(op, available)) {

      // If this is a load from cache already, dont force a cache of this
      if (isa<LoadInst>(op) && cast<LoadInst>(op)->getMetadata("enzyme_fromcache")) continue;

      // If a plcaeholder phi for inversion (and we know from above not recomputable)
      if (!isa<PHINode>(op) && dyn_cast_or_null<LoadInst>(hasUninverted(op))) {
        goto forceCache;
      }

      // Even if cannot recompute (say a phi node), don't force a reload if it is possible to just use this instruction from forward pass without issue
      if (auto i2 = dyn_cast<Instruction>(op)) {
        if (!i2->mayReadOrWriteMemory()) {
          LoopContext lc;
          bool inLoop = const_cast<GradientUtils*>(this)->getContext(i2->getParent(), lc);
          if (!inLoop) {
            if (i2->getParent() == &newFunc->getEntryBlock()) {
              continue;
            }
            // TODO upgrade this to be all returns that this could enter from
            bool legal = true;
            for(auto &BB: *oldFunc) {
              if (isa<ReturnInst>(BB.getTerminator())) {
                BasicBlock* returningBlock = cast<BasicBlock>(getNewFromOriginal(&BB));
                if (i2->getParent() == returningBlock) continue;
                if (!DT.dominates(i2, returningBlock)) {
                  legal = false;
                  break;
                }
              }
            }
            if (legal) {
              continue;
            }
          }
        }
      }

      forceCache:;
      //llvm::errs() << "choosing to cache " << *val << " because of " << *op << "\n";
      return false;
    }
  }

  if (auto op = dyn_cast<IntrinsicInst>(val)) {
    if (!op->mayReadOrWriteMemory()) return true;
    switch(op->getIntrinsicID()) {
      case Intrinsic::sin:
      case Intrinsic::cos:
      case Intrinsic::exp:
      case Intrinsic::log:
      return true;
      default:
      return false;
    }
  }

  if (auto ci = dyn_cast<CallInst>(val)) {
    if (auto called = ci->getCalledFunction()) {
      auto n = called->getName();
      if (n == "lgamma" || n == "lgammaf" || n == "lgammal" || n == "lgamma_r" || n == "lgammaf_r" || n == "lgammal_r"
        || n == "__lgamma_r_finite" || n == "__lgammaf_r_finite" || n == "__lgammal_r_finite"
        || n == "tanh" || n == "tanhf") {
        return true;
      }
    }
  }

  //cache a call, assuming its longer to run that
  if (isa<CallInst>(val)) {
    llvm::errs() << " caching call: " << *val << "\n";
    return false;
  }

  //llvm::errs() << "unknown inst " << *val << " unable to recompute\n";
  return true;
}

GradientUtils* GradientUtils::CreateFromClone(Function *todiff, TargetLibraryInfo &TLI, TypeAnalysis &TA, AAResults &AA, DIFFE_TYPE retType, const std::vector<DIFFE_TYPE> & constant_args, bool returnUsed, std::map<AugmentedStruct, int> &returnMapping ) {
    assert(!todiff->empty());

    // Since this is forward pass this should always return the tape (at index 0)
    returnMapping[AugmentedStruct::Tape] = 0;

    int returnCount = 0;

    if (returnUsed) {
        assert(!todiff->getReturnType()->isEmptyTy());
        assert(!todiff->getReturnType()->isVoidTy());
        returnMapping[AugmentedStruct::Return] = returnCount+1;
        returnCount++;
    }

    // We don't need to differentially return something that we know is not a pointer (or somehow needed for shadow analysis)
    if (retType == DIFFE_TYPE::DUP_ARG || retType == DIFFE_TYPE::DUP_NONEED) {
        assert(!todiff->getReturnType()->isEmptyTy());
        assert(!todiff->getReturnType()->isVoidTy());
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

    auto newFunc = CloneFunctionWithReturns(/*topLevel*/false, todiff, AA, TLI, invertedPointers, constant_args, constants, nonconstant, returnvals, /*returnValue*/returnValue, "fakeaugmented_"+todiff->getName(), &originalToNew, /*diffeReturnArg*/false, /*additionalArg*/nullptr);
    //llvm::errs() <<  "returnvals:" << todiff->getName() << " \n";
    //for (auto a : returnvals ) {
    //    llvm::errs() <<"   + " << *a << "\n";
    //}
    //llvm::errs() <<  "end returnvals:\n";
    SmallPtrSet<Value*,4> constant_values;
    SmallPtrSet<Value*,4> nonconstant_values;

    if (retType != DIFFE_TYPE::CONSTANT) {
      for(auto a : returnvals) {
          nonconstant_values.insert(a);
      }
    }

    auto res = new GradientUtils(newFunc, todiff, TLI, TA, AA, invertedPointers, constants, nonconstant, constant_values, nonconstant_values, originalToNew);
    return res;
}

DiffeGradientUtils* DiffeGradientUtils::CreateFromClone(bool topLevel, Function *todiff, TargetLibraryInfo &TLI, TypeAnalysis &TA, AAResults &AA, DIFFE_TYPE retType, const std::vector<DIFFE_TYPE>& constant_args, ReturnType returnValue, Type* additionalArg) {
  assert(!todiff->empty());
  ValueToValueMapTy invertedPointers;
  SmallPtrSet<Value*,4> constants;
  SmallPtrSet<Value*,20> nonconstant;
  SmallPtrSet<Value*,2> returnvals;
  ValueToValueMapTy originalToNew;

  bool diffeReturnArg = retType == DIFFE_TYPE::OUT_DIFF;
  auto newFunc = CloneFunctionWithReturns(topLevel, todiff, AA, TLI, invertedPointers, constant_args, constants, nonconstant, returnvals, returnValue, "diffe"+todiff->getName(), &originalToNew, /*diffeReturnArg*/diffeReturnArg, additionalArg);
  SmallPtrSet<Value*,4> constant_values;
  SmallPtrSet<Value*,4> nonconstant_values;
  if (retType != DIFFE_TYPE::CONSTANT) {
    for(auto a : returnvals) {
      nonconstant_values.insert(a);
    }
  }

  //llvm::errs() << "creating from clone: " << todiff->getName() << " differentialReturn: " << differentialReturn << " returnvals: " << returnvals.size() << " nonconstant_values: " << nonconstant_values.size() << "\n";
  auto res = new DiffeGradientUtils(newFunc, todiff, TLI, TA, AA, invertedPointers, constants, nonconstant, constant_values, nonconstant_values, originalToNew);
  return res;
}

Value* GradientUtils::invertPointerM(Value* oval, IRBuilder<>& BuilderM) {
    if (auto inst = dyn_cast<Instruction>(oval)) {
      assert(inst->getParent()->getParent() == oldFunc);
    }
    if (auto arg = dyn_cast<Argument>(oval)) {
      assert(arg->getParent() == oldFunc);
    }


    if (isa<ConstantPointerNull>(oval)) {
        return oval;
    } else if (isa<UndefValue>(oval)) {
        return oval;
    } else if (auto cint = dyn_cast<ConstantInt>(oval)) {
        if (cint->isZero()) return cint;
        if (cint->isOne()) return cint;
    }

    if(isConstantValue(oval)) {
        //NOTE, this is legal and the correct resolution, however, our activity analysis honeypot no longer exists
        return lookupM(getNewFromOriginal(oval), BuilderM);
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *oval << "\n";
    }
    assert(!isConstantValue(oval));

    auto M = oldFunc->getParent();
    assert(oval);

    if (invertedPointers.find(oval) != invertedPointers.end()) {
        return lookupM(invertedPointers[oval], BuilderM);
    }

    if (auto arg = dyn_cast<GlobalVariable>(oval)) {
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
      return invertedPointers[oval] = cs;
    } else if (auto fn = dyn_cast<Function>(oval)) {
      //! Todo allow tape propagation
      //  Note that specifically this should _not_ be called with topLevel=true (since it may not be valid to always assume we can recompute the augmented primal)
      //  However, in the absence of a way to pass tape data from an indirect augmented (and also since we dont presently allow indirect augmented calls), topLevel MUST be true
      //  otherwise subcalls will not be able to lookup the augmenteddata/subdata (triggering an assertion failure, among much worse)
      std::map<Argument*, bool> uncacheable_args;
      NewFnTypeInfo type_args(fn);

      //conservatively assume that we can only cache existing floating types (i.e. that all args are uncacheable)
      std::vector<DIFFE_TYPE> types;
      for(auto &a : fn->args()) {
          uncacheable_args[&a] = !a.getType()->isFPOrFPVectorTy();
          type_args.first.insert(std::pair<Argument*, ValueData>(&a, {}));
          type_args.knownValues.insert(std::pair<Argument*, std::set<int64_t>>(&a, {}));
          DIFFE_TYPE typ;
          if (a.getType()->isFPOrFPVectorTy()) {
            typ = DIFFE_TYPE::OUT_DIFF;
          } else {
            typ = DIFFE_TYPE::DUP_ARG;
          }
          types.push_back(typ);
      }

      DIFFE_TYPE retType = fn->getReturnType()->isFPOrFPVectorTy() ? DIFFE_TYPE::OUT_DIFF : DIFFE_TYPE::DUP_ARG;
      if (fn->getReturnType()->isVoidTy() || fn->getReturnType()->isEmptyTy()) retType = DIFFE_TYPE::CONSTANT;

      auto& augdata = CreateAugmentedPrimal(fn, retType, /*constant_args*/types, TLI, TA, AA, /*returnUsed*/!fn->getReturnType()->isEmptyTy() && !fn->getReturnType()->isVoidTy(), type_args, uncacheable_args, /*forceAnonymousTape*/true);
      auto newf = CreatePrimalAndGradient(fn, retType, /*constant_args*/types, TLI, TA, AA, /*returnValue*/false, /*dretPtr*/false, /*topLevel*/false, /*additionalArg*/Type::getInt8PtrTy(fn->getContext()), type_args, uncacheable_args, /*map*/&augdata); //llvm::Optional<std::map<std::pair<llvm::Instruction*, std::string>, unsigned int> >({}));
      auto cdata = ConstantStruct::get(StructType::get(newf->getContext(), {augdata.fn->getType(), newf->getType()}), {augdata.fn, newf});
      std::string globalname = ("_enzyme_" + fn->getName() + "'").str();
      auto GV = newf->getParent()->getNamedValue(globalname);

      if (GV == nullptr) {
        GV = new GlobalVariable(*newf->getParent(), cdata->getType(), true, GlobalValue::LinkageTypes::InternalLinkage, cdata, globalname);
      }

      return BuilderM.CreatePointerCast(GV, fn->getType());
    } else if (auto arg = dyn_cast<CastInst>(oval)) {
      IRBuilder<> bb(getNewFromOriginal(arg));
      invertedPointers[arg] = bb.CreateCast(arg->getOpcode(), invertPointerM(arg->getOperand(0), bb), arg->getDestTy(), arg->getName()+"'ipc");
      return lookupM(invertedPointers[arg], BuilderM);
    } else if (auto arg = dyn_cast<ConstantExpr>(oval)) {
      if (arg->isCast()) {
          auto result = ConstantExpr::getCast(arg->getOpcode(), cast<Constant>(invertPointerM(arg->getOperand(0), BuilderM)), arg->getType());
          return result;
      } else if (arg->isGEPWithNoNotionalOverIndexing()) {
          auto result = arg->getWithOperandReplaced(0, cast<Constant>(invertPointerM(arg->getOperand(0), BuilderM)));
          return result;
      }
      goto end;
    } else if (auto arg = dyn_cast<ExtractValueInst>(oval)) {
      IRBuilder<> bb(getNewFromOriginal(arg));
      auto result = bb.CreateExtractValue(invertPointerM(arg->getOperand(0), bb), arg->getIndices(), arg->getName()+"'ipev");
      invertedPointers[arg] = result;
      return lookupM(invertedPointers[arg], BuilderM);
    } else if (auto arg = dyn_cast<InsertValueInst>(oval)) {
      IRBuilder<> bb(getNewFromOriginal(arg));
      auto result = bb.CreateInsertValue(invertPointerM(arg->getOperand(0), bb), invertPointerM(arg->getOperand(1), bb), arg->getIndices(), arg->getName()+"'ipiv");
      invertedPointers[arg] = result;
      return lookupM(invertedPointers[arg], BuilderM);
    } else if (auto arg = dyn_cast<ExtractElementInst>(oval)) {
      IRBuilder<> bb(getNewFromOriginal(arg));
      auto result = bb.CreateExtractElement(invertPointerM(arg->getVectorOperand(), bb), getNewFromOriginal(arg->getIndexOperand()), arg->getName()+"'ipee");
      invertedPointers[arg] = result;
      return lookupM(invertedPointers[arg], BuilderM);
    } else if (auto arg = dyn_cast<InsertElementInst>(oval)) {
      IRBuilder<> bb(getNewFromOriginal(arg));
      Value* op0 = arg->getOperand(0);
      Value* op1 = arg->getOperand(1);
      Value* op2 = arg->getOperand(2);
      auto result = bb.CreateInsertElement(invertPointerM(op0, bb), invertPointerM(op1,bb), getNewFromOriginal(op2), arg->getName()+"'ipie");
      invertedPointers[arg] = result;
      return lookupM(invertedPointers[arg], BuilderM);
    } else if (auto arg = dyn_cast<SelectInst>(oval)) {
      IRBuilder<> bb(getNewFromOriginal(arg));
      auto result = bb.CreateSelect(getNewFromOriginal(arg->getCondition()), invertPointerM(arg->getTrueValue(), bb), invertPointerM(arg->getFalseValue(), bb), arg->getName()+"'ipse");
      invertedPointers[arg] = result;
      return lookupM(invertedPointers[arg], BuilderM);
    } else if (auto arg = dyn_cast<LoadInst>(oval)) {
      IRBuilder <> bb(getNewFromOriginal(arg));
      Value* op0 = arg->getOperand(0);
      auto li = bb.CreateLoad(invertPointerM(op0, bb), arg->getName()+"'ipl");
      li->setAlignment(arg->getAlignment());
      li->setVolatile(arg->isVolatile());
      li->setOrdering(arg->getOrdering());
      li->setSyncScopeID(arg->getSyncScopeID ());
      invertedPointers[arg] = li;
      return lookupM(invertedPointers[arg], BuilderM);
    } else if (auto arg = dyn_cast<BinaryOperator>(oval)) {
 	  assert(arg->getType()->isIntOrIntVectorTy());
      IRBuilder <> bb(getNewFromOriginal(arg));
      Value* val0 = nullptr;
      Value* val1 = nullptr;

      val0 = invertPointerM(arg->getOperand(0), bb);
      val1 = invertPointerM(arg->getOperand(1), bb);

      auto li = bb.CreateBinOp(arg->getOpcode(), val0, val1, arg->getName());
      cast<BinaryOperator>(li)->copyIRFlags(arg);
      invertedPointers[arg] = li;
      return lookupM(invertedPointers[arg], BuilderM);
    } else if (auto arg = dyn_cast<GetElementPtrInst>(oval)) {
      IRBuilder<> bb(getNewFromOriginal(arg));
      SmallVector<Value*,4> invertargs;
      for(unsigned i=0; i<arg->getNumIndices(); i++) {
          Value* b = getNewFromOriginal(arg->getOperand(1+i));
          invertargs.push_back(b);
      }
      auto result = bb.CreateGEP(invertPointerM(arg->getPointerOperand(), bb), invertargs, arg->getName()+"'ipg");
      if (auto gep = dyn_cast<GetElementPtrInst>(result))
          gep->setIsInBounds(arg->isInBounds());
      invertedPointers[arg] = result;
      return lookupM(invertedPointers[arg], BuilderM);
    } else if (auto inst = dyn_cast<AllocaInst>(oval)) {
      IRBuilder<> bb(getNewFromOriginal(inst));
      Value* asize = getNewFromOriginal(inst->getArraySize());
      AllocaInst* antialloca = bb.CreateAlloca(inst->getAllocatedType(), inst->getType()->getPointerAddressSpace(), asize, inst->getName()+"'ipa");
      invertedPointers[inst] = antialloca;
      antialloca->setAlignment(inst->getAlignment());

      if (auto ci = dyn_cast<ConstantInt>(asize)) {
        if (ci->isOne()) {
          auto st = bb.CreateStore(Constant::getNullValue(inst->getAllocatedType()), antialloca);
          st->setAlignment(inst->getAlignment());
          return lookupM(invertedPointers[inst], BuilderM);
        }
      }

      auto dst_arg = bb.CreateBitCast(antialloca,Type::getInt8PtrTy(oval->getContext()));
      auto val_arg = ConstantInt::get(Type::getInt8Ty(oval->getContext()), 0);
      auto len_arg = bb.CreateMul(bb.CreateZExtOrTrunc(asize,Type::getInt64Ty(oval->getContext())), ConstantInt::get(Type::getInt64Ty(oval->getContext()), M->getDataLayout().getTypeAllocSizeInBits(inst->getAllocatedType())/8), "", true, true);
      auto volatile_arg = ConstantInt::getFalse(oval->getContext());

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
    } else if (auto phi = dyn_cast<PHINode>(oval)) {

     if (phi->getNumIncomingValues() == 0) {
      dumpMap(invertedPointers);
      assert(0 && "illegal iv of phi");
     }
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
         IRBuilder <> bb(getNewFromOriginal(phi));
         auto which = bb.CreatePHI(phi->getType(), phi->getNumIncomingValues());
         invertedPointers[phi] = which;

         for(unsigned int i=0; i<phi->getNumIncomingValues(); i++) {
            IRBuilder <>pre(cast<BasicBlock>(getNewFromOriginal(phi->getIncomingBlock(i)))->getTerminator());
            Value* val = invertPointerM(phi->getIncomingValue(i), pre);
            which->addIncoming(val, cast<BasicBlock>(getNewFromOriginal(phi->getIncomingBlock(i))));
         }

         return lookupM(which, BuilderM);
     }
    }

  end:;
    assert(BuilderM.GetInsertBlock());
    assert(BuilderM.GetInsertBlock()->getParent());
    assert(oval);
    llvm::errs() << "fn:" << *BuilderM.GetInsertBlock()->getParent() << "\noval=" << *oval << "\n";
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
    Instruction* inc = cast<Instruction>(B.CreateAdd(CanonicalIV, ConstantInt::get(CanonicalIV->getType(), 1), "iv.next", /*NUW*/true, /*NSW*/true));

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

void removeRedundantIVs(const Loop* L, BasicBlock* Header, BasicBlock* Preheader, PHINode* CanonicalIV, MyScalarEvolution &SE, GradientUtils &gutils, Instruction* increment, const SmallVectorImpl<BasicBlock*>&& latches) {
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
        //llvm::errs() << " pn: " << *PN << " scev: " << *S << " for NewIV: " << *NewIV << "\n";
        if (NewIV == PN) {
          continue;
        }
        if (auto BO = dyn_cast<BinaryOperator>(NewIV)) {
          if (BO->getOpcode() == BinaryOperator::Add || BO->getOpcode() == BinaryOperator::Mul) {
            BO->setHasNoSignedWrap(true);
            BO->setHasNoUnsignedWrap(true);
          }
          for(int i=0; i<2; i++) {
            if (auto BO2 = dyn_cast<BinaryOperator>(BO->getOperand(i))) {
              if (BO2->getOpcode() == BinaryOperator::Add || BO2->getOpcode() == BinaryOperator::Mul) {
                BO2->setHasNoSignedWrap(true);
                BO2->setHasNoUnsignedWrap(true);
              }
            }
          }
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

ScalarEvolution::ExitLimit MyScalarEvolution::computeExitLimitFromCond(
    const Loop *L, Value *ExitCond, bool ExitIfTrue,
    bool ControlsExit, bool AllowPredicates) {
  ScalarEvolution::ExitLimitCacheTy Cache(L, ExitIfTrue, AllowPredicates);
  return computeExitLimitFromCondCached(Cache, L, ExitCond, ExitIfTrue,
                                        ControlsExit, AllowPredicates);
}

ScalarEvolution::ExitLimit
MyScalarEvolution::computeExitLimit(const Loop *L, BasicBlock *ExitingBlock,
                                      bool AllowPredicates) {
  assert(L->contains(ExitingBlock) && "Exit count for non-loop block?");
  // If our exiting block does not dominate the latch, then its connection with
  // loop's exit limit may be far from trivial.
  const BasicBlock *Latch = L->getLoopLatch();
  if (!Latch || !DT.dominates(ExitingBlock, Latch))
    return getCouldNotCompute();

  bool IsOnlyExit = (L->getExitingBlock() != nullptr);
  auto *Term = ExitingBlock->getTerminator();
  if (BranchInst *BI = dyn_cast<BranchInst>(Term)) {
    assert(BI->isConditional() && "If unconditional, it can't be in loop!");
    bool ExitIfTrue = !L->contains(BI->getSuccessor(0));
    assert(ExitIfTrue == L->contains(BI->getSuccessor(1)) &&
           "It should have one successor in loop and one exit block!");
    // Proceed to the next level to examine the exit condition expression.
    return computeExitLimitFromCond(
        L, BI->getCondition(), ExitIfTrue,
        /*ControlsExit=*/IsOnlyExit, AllowPredicates);
  }

  if (SwitchInst *SI = dyn_cast<SwitchInst>(Term)) {
    // For switch, make sure that there is a single exit from the loop.
    BasicBlock *Exit = nullptr;
    for (auto *SBB : successors(ExitingBlock))
      if (!L->contains(SBB)) {
        if (Exit) // Multiple exit successors.
          return getCouldNotCompute();
        Exit = SBB;
      }
    assert(Exit && "Exiting block must have at least one exit");
    return computeExitLimitFromSingleExitSwitch(L, SI, Exit,
                                                /*ControlsExit=*/IsOnlyExit);
  }

  return getCouldNotCompute();
}

ScalarEvolution::ExitLimit MyScalarEvolution::computeExitLimitFromCondCached(
    ExitLimitCacheTy &Cache, const Loop *L, Value *ExitCond, bool ExitIfTrue,
    bool ControlsExit, bool AllowPredicates) {

  if (auto MaybeEL =
          Cache.find(L, ExitCond, ExitIfTrue, ControlsExit, AllowPredicates))
    return *MaybeEL;

  ExitLimit EL = computeExitLimitFromCondImpl(Cache, L, ExitCond, ExitIfTrue,
                                              ControlsExit, AllowPredicates);
  Cache.insert(L, ExitCond, ExitIfTrue, ControlsExit, AllowPredicates, EL);
  return EL;
}

ScalarEvolution::ExitLimit MyScalarEvolution::computeExitLimitFromCondImpl(
    ExitLimitCacheTy &Cache, const Loop *L, Value *ExitCond, bool ExitIfTrue,
    bool ControlsExit, bool AllowPredicates) {
  // Check if the controlling expression for this loop is an And or Or.
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(ExitCond)) {
    if (BO->getOpcode() == Instruction::And) {
      // Recurse on the operands of the and.
      bool EitherMayExit = !ExitIfTrue;
      ExitLimit EL0 = computeExitLimitFromCondCached(
          Cache, L, BO->getOperand(0), ExitIfTrue,
          ControlsExit && !EitherMayExit, AllowPredicates);
      ExitLimit EL1 = computeExitLimitFromCondCached(
          Cache, L, BO->getOperand(1), ExitIfTrue,
          ControlsExit && !EitherMayExit, AllowPredicates);
      const SCEV *BECount = getCouldNotCompute();
      const SCEV *MaxBECount = getCouldNotCompute();
      if (EitherMayExit) {
        // Both conditions must be true for the loop to continue executing.
        // Choose the less conservative count.
        if (EL0.ExactNotTaken == getCouldNotCompute() ||
            EL1.ExactNotTaken == getCouldNotCompute())
          BECount = getCouldNotCompute();
        else
          BECount =
              getUMinFromMismatchedTypes(EL0.ExactNotTaken, EL1.ExactNotTaken);
        if (EL0.MaxNotTaken == getCouldNotCompute())
          MaxBECount = EL1.MaxNotTaken;
        else if (EL1.MaxNotTaken == getCouldNotCompute())
          MaxBECount = EL0.MaxNotTaken;
        else
          MaxBECount =
              getUMinFromMismatchedTypes(EL0.MaxNotTaken, EL1.MaxNotTaken);
      } else {
        // Both conditions must be true at the same time for the loop to exit.
        // For now, be conservative.
        if (EL0.MaxNotTaken == EL1.MaxNotTaken)
          MaxBECount = EL0.MaxNotTaken;
        if (EL0.ExactNotTaken == EL1.ExactNotTaken)
          BECount = EL0.ExactNotTaken;
      }

      // There are cases (e.g. PR26207) where computeExitLimitFromCond is able
      // to be more aggressive when computing BECount than when computing
      // MaxBECount.  In these cases it is possible for EL0.ExactNotTaken and
      // EL1.ExactNotTaken to match, but for EL0.MaxNotTaken and EL1.MaxNotTaken
      // to not.
      if (isa<SCEVCouldNotCompute>(MaxBECount) &&
          !isa<SCEVCouldNotCompute>(BECount))
        MaxBECount = getConstant(getUnsignedRangeMax(BECount));

      return ExitLimit(BECount, MaxBECount, false,
                       {&EL0.Predicates, &EL1.Predicates});
    }
    if (BO->getOpcode() == Instruction::Or) {
      // Recurse on the operands of the or.
      bool EitherMayExit = ExitIfTrue;
      ExitLimit EL0 = computeExitLimitFromCondCached(
          Cache, L, BO->getOperand(0), ExitIfTrue,
          ControlsExit && !EitherMayExit, AllowPredicates);
      ExitLimit EL1 = computeExitLimitFromCondCached(
          Cache, L, BO->getOperand(1), ExitIfTrue,
          ControlsExit && !EitherMayExit, AllowPredicates);
      const SCEV *BECount = getCouldNotCompute();
      const SCEV *MaxBECount = getCouldNotCompute();
      if (EitherMayExit) {
        // Both conditions must be false for the loop to continue executing.
        // Choose the less conservative count.
        if (EL0.ExactNotTaken == getCouldNotCompute() ||
            EL1.ExactNotTaken == getCouldNotCompute())
          BECount = getCouldNotCompute();
        else
          BECount =
              getUMinFromMismatchedTypes(EL0.ExactNotTaken, EL1.ExactNotTaken);
        if (EL0.MaxNotTaken == getCouldNotCompute())
          MaxBECount = EL1.MaxNotTaken;
        else if (EL1.MaxNotTaken == getCouldNotCompute())
          MaxBECount = EL0.MaxNotTaken;
        else
          MaxBECount =
              getUMinFromMismatchedTypes(EL0.MaxNotTaken, EL1.MaxNotTaken);
      } else {
        // Both conditions must be false at the same time for the loop to exit.
        // For now, be conservative.
        if (EL0.MaxNotTaken == EL1.MaxNotTaken)
          MaxBECount = EL0.MaxNotTaken;
        if (EL0.ExactNotTaken == EL1.ExactNotTaken)
          BECount = EL0.ExactNotTaken;
      }

      return ExitLimit(BECount, MaxBECount, false,
                       {&EL0.Predicates, &EL1.Predicates});
    }
  }

  // With an icmp, it may be feasible to compute an exact backedge-taken count.
  // Proceed to the next level to examine the icmp.
  if (ICmpInst *ExitCondICmp = dyn_cast<ICmpInst>(ExitCond)) {
    ExitLimit EL =
        computeExitLimitFromICmp(L, ExitCondICmp, ExitIfTrue, ControlsExit);
    if (EL.hasFullInfo() || !AllowPredicates)
      return EL;

    // Try again, but use SCEV predicates this time.
    return computeExitLimitFromICmp(L, ExitCondICmp, ExitIfTrue, ControlsExit,
                                    /*AllowPredicates=*/true);
  }

  // Check for a constant condition. These are normally stripped out by
  // SimplifyCFG, but ScalarEvolution may be used by a pass which wishes to
  // preserve the CFG and is temporarily leaving constant conditions
  // in place.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(ExitCond)) {
    if (ExitIfTrue == !CI->getZExtValue())
      // The backedge is always taken.
      return getCouldNotCompute();
    else
      // The backedge is never taken.
      return getZero(CI->getType());
  }

  // If it's not an integer or pointer comparison then compute it the hard way.
  return computeExitCountExhaustively(L, ExitCond, ExitIfTrue);
}

ScalarEvolution::ExitLimit
MyScalarEvolution::computeExitLimitFromICmp(const Loop *L,
                                          ICmpInst *ExitCond,
                                          bool ExitIfTrue,
                                          bool ControlsExit,
                                          bool AllowPredicates) {
  // If the condition was exit on true, convert the condition to exit on false
  ICmpInst::Predicate Pred;
  if (!ExitIfTrue)
    Pred = ExitCond->getPredicate();
  else
    Pred = ExitCond->getInversePredicate();
  const ICmpInst::Predicate OriginalPred = Pred;

  // Handle common loops like: for (X = "string"; *X; ++X)
  if (LoadInst *LI = dyn_cast<LoadInst>(ExitCond->getOperand(0)))
    if (Constant *RHS = dyn_cast<Constant>(ExitCond->getOperand(1))) {
      ExitLimit ItCnt =
        computeLoadConstantCompareExitLimit(LI, RHS, L, Pred);
      if (ItCnt.hasAnyInfo())
        return ItCnt;
    }

  const SCEV *LHS = getSCEV(ExitCond->getOperand(0));
  const SCEV *RHS = getSCEV(ExitCond->getOperand(1));

  #define PROP_PHI(LHS)\
    if (auto un = dyn_cast<SCEVUnknown>(LHS)) {\
      if (auto pn = dyn_cast_or_null<PHINode>(un->getValue())) {\
        const SCEV *sc = nullptr;\
        bool failed = false;\
        for(auto &a : pn->incoming_values()) {\
          auto subsc = getSCEV(a);\
          if (sc == nullptr) {\
            sc = subsc;\
            continue;\
          }\
          if (subsc != sc) {\
            failed = true;\
            break;\
          }\
        }\
        if (!failed) {\
          LHS = sc;\
        }\
      }\
    }
  PROP_PHI(LHS)
  PROP_PHI(RHS)
  //llvm::errs() << "pLHS: " << *LHS << "\n";
  //llvm::errs() << "pRHS: " << *RHS << "\n";

  // Try to evaluate any dependencies out of the loop.
  LHS = getSCEVAtScope(LHS, L);
  RHS = getSCEVAtScope(RHS, L);

  //llvm::errs() << "LHS: " << *LHS << "\n";
  //llvm::errs() << "RHS: " << *RHS << "\n";

  // At this point, we would like to compute how many iterations of the
  // loop the predicate will return true for these inputs.
  if (isLoopInvariant(LHS, L) && !isLoopInvariant(RHS, L)) {
    // If there is a loop-invariant, force it into the RHS.
    std::swap(LHS, RHS);
    Pred = ICmpInst::getSwappedPredicate(Pred);
  }

  // Simplify the operands before analyzing them.
  (void)SimplifyICmpOperands(Pred, LHS, RHS);

  // If we have a comparison of a chrec against a constant, try to use value
  // ranges to answer this query.
  if (const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(RHS))
    if (const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(LHS))
      if (AddRec->getLoop() == L) {
        // Form the constant range.
        ConstantRange CompRange =
            ConstantRange::makeExactICmpRegion(Pred, RHSC->getAPInt());

        const SCEV *Ret = AddRec->getNumIterationsInRange(CompRange, *this);
        if (!isa<SCEVCouldNotCompute>(Ret)) return Ret;
      }

  switch (Pred) {
  case ICmpInst::ICMP_NE: {                     // while (X != Y)
    // Convert to: while (X-Y != 0)
    ExitLimit EL = howFarToZero(getMinusSCEV(LHS, RHS), L, ControlsExit,
                                AllowPredicates);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  case ICmpInst::ICMP_EQ: {                     // while (X == Y)
    // Convert to: while (X-Y == 0)
    ExitLimit EL = howFarToNonZero(getMinusSCEV(LHS, RHS), L);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  case ICmpInst::ICMP_SLT:
  case ICmpInst::ICMP_ULT: {                    // while (X < Y)
    bool IsSigned = Pred == ICmpInst::ICMP_SLT;
    ExitLimit EL = howManyLessThans(LHS, RHS, L, IsSigned, ControlsExit,
                                    AllowPredicates);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  case ICmpInst::ICMP_SGT:
  case ICmpInst::ICMP_UGT: {                    // while (X > Y)
    bool IsSigned = Pred == ICmpInst::ICMP_SGT;
    ExitLimit EL =
        howManyGreaterThans(LHS, RHS, L, IsSigned, ControlsExit,
                            AllowPredicates);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  default:
    break;
  }

  auto *ExhaustiveCount =
      computeExitCountExhaustively(L, ExitCond, ExitIfTrue);

  if (!isa<SCEVCouldNotCompute>(ExhaustiveCount))
    return ExhaustiveCount;

  return computeShiftCompareExitLimit(ExitCond->getOperand(0),
                                      ExitCond->getOperand(1), L, OriginalPred);
}

ScalarEvolution::ExitLimit
MyScalarEvolution::howManyLessThans(const SCEV *LHS, const SCEV *RHS,
                                  const Loop *L, bool IsSigned,
                                  bool ControlsExit, bool AllowPredicates) {
  SmallPtrSet<const SCEVPredicate *, 4> Predicates;

  const SCEVAddRecExpr *IV = dyn_cast<SCEVAddRecExpr>(LHS);

  if (!IV && AllowPredicates) {
    // Try to make this an AddRec using runtime tests, in the first X
    // iterations of this loop, where X is the SCEV expression found by the
    // algorithm below.
    IV = convertSCEVToAddRecWithPredicates(LHS, L, Predicates);
  }

  // Avoid weird loops
  if (!IV || IV->getLoop() != L || !IV->isAffine())
    return getCouldNotCompute();


  bool NoWrap = ControlsExit &&
                true; // changed this to assume no wrap for inc
  //              IV->getNoWrapFlags(IsSigned ? SCEV::FlagNSW : SCEV::FlagNUW);

  const SCEV *Stride = IV->getStepRecurrence(*this);

  bool PositiveStride = isKnownPositive(Stride);

  // Avoid negative or zero stride values.
  if (!PositiveStride) {
    // We can compute the correct backedge taken count for loops with unknown
    // strides if we can prove that the loop is not an infinite loop with side
    // effects. Here's the loop structure we are trying to handle -
    //
    // i = start
    // do {
    //   A[i] = i;
    //   i += s;
    // } while (i < end);
    //
    // The backedge taken count for such loops is evaluated as -
    // (max(end, start + stride) - start - 1) /u stride
    //
    // The additional preconditions that we need to check to prove correctness
    // of the above formula is as follows -
    //
    // a) IV is either nuw or nsw depending upon signedness (indicated by the
    //    NoWrap flag).
    // b) loop is single exit with no side effects. // dont need this
    //
    //
    // Precondition a) implies that if the stride is negative, this is a single
    // trip loop. The backedge taken count formula reduces to zero in this case.
    //
    // Precondition b) implies that the unknown stride cannot be zero otherwise
    // we have UB.
    //
    // The positive stride case is the same as isKnownPositive(Stride) returning
    // true (original behavior of the function).
    //
    // We want to make sure that the stride is truly unknown as there are edge
    // cases where ScalarEvolution propagates no wrap flags to the
    // post-increment/decrement IV even though the increment/decrement operation
    // itself is wrapping. The computed backedge taken count may be wrong in
    // such cases. This is prevented by checking that the stride is not known to
    // be either positive or non-positive. For example, no wrap flags are
    // propagated to the post-increment IV of this loop with a trip count of 2 -
    //
    // unsigned char i;
    // for(i=127; i<128; i+=129)
    //   A[i] = i;
    //
    if (!NoWrap) // THIS LINE CHANGED
      return getCouldNotCompute();
  } else if (!Stride->isOne() &&
             doesIVOverflowOnLT(RHS, Stride, IsSigned, NoWrap))
    // Avoid proven overflow cases: this will ensure that the backedge taken
    // count will not generate any unsigned overflow. Relaxed no-overflow
    // conditions exploit NoWrapFlags, allowing to optimize in presence of
    // undefined behaviors like the case of C language.
    return getCouldNotCompute();

  ICmpInst::Predicate Cond = IsSigned ? ICmpInst::ICMP_SLT
                                      : ICmpInst::ICMP_ULT;
  const SCEV *Start = IV->getStart();
  const SCEV *End = RHS;
  // When the RHS is not invariant, we do not know the end bound of the loop and
  // cannot calculate the ExactBECount needed by ExitLimit. However, we can
  // calculate the MaxBECount, given the start, stride and max value for the end
  // bound of the loop (RHS), and the fact that IV does not overflow (which is
  // checked above).
  if (!isLoopInvariant(RHS, L)) {
    const SCEV *MaxBECount = computeMaxBECountForLT(
        Start, Stride, RHS, getTypeSizeInBits(LHS->getType()), IsSigned);
    return ExitLimit(getCouldNotCompute() /* ExactNotTaken */, MaxBECount,
                     false /*MaxOrZero*/, Predicates);
  }
  // If the backedge is taken at least once, then it will be taken
  // (End-Start)/Stride times (rounded up to a multiple of Stride), where Start
  // is the LHS value of the less-than comparison the first time it is evaluated
  // and End is the RHS.
  const SCEV *BECountIfBackedgeTaken =
    computeBECount(getMinusSCEV(End, Start), Stride, false);
  // If the loop entry is guarded by the result of the backedge test of the
  // first loop iteration, then we know the backedge will be taken at least
  // once and so the backedge taken count is as above. If not then we use the
  // expression (max(End,Start)-Start)/Stride to describe the backedge count,
  // as if the backedge is taken at least once max(End,Start) is End and so the
  // result is as above, and if not max(End,Start) is Start so we get a backedge
  // count of zero.
  const SCEV *BECount;
  if (isLoopEntryGuardedByCond(L, Cond, getMinusSCEV(Start, Stride), RHS))
    BECount = BECountIfBackedgeTaken;
  else {
    End = IsSigned ? getSMaxExpr(RHS, Start) : getUMaxExpr(RHS, Start);
    BECount = computeBECount(getMinusSCEV(End, Start), Stride, false);
  }

  const SCEV *MaxBECount;
  bool MaxOrZero = false;
  if (isa<SCEVConstant>(BECount))
    MaxBECount = BECount;
  else if (isa<SCEVConstant>(BECountIfBackedgeTaken)) {
    // If we know exactly how many times the backedge will be taken if it's
    // taken at least once, then the backedge count will either be that or
    // zero.
    MaxBECount = BECountIfBackedgeTaken;
    MaxOrZero = true;
  } else {
    MaxBECount = computeMaxBECountForLT(
        Start, Stride, RHS, getTypeSizeInBits(LHS->getType()), IsSigned);
  }

  if (isa<SCEVCouldNotCompute>(MaxBECount) &&
      !isa<SCEVCouldNotCompute>(BECount))
    MaxBECount = getConstant(getUnsignedRangeMax(BECount));

  return ExitLimit(BECount, MaxBECount, MaxOrZero, Predicates);
}

bool getContextM(BasicBlock *BB, LoopContext &loopContext, std::map<Loop*,LoopContext> &loopContexts, LoopInfo &LI,MyScalarEvolution &SE,DominatorTree &DT, GradientUtils &gutils) {
    Loop* L = LI.getLoopFor(BB);

    //Not inside a loop
    if (L == nullptr) return false;

    //Already canonicalized
    if (loopContexts.find(L) == loopContexts.end()) {

        loopContexts[L].parent = L->getParentLoop();

        loopContexts[L].header = L->getHeader();
        assert(loopContexts[L].header && "loop must have header");

        loopContexts[L].preheader = L->getLoopPreheader();
        if (!L->getLoopPreheader()) {
            llvm::errs() << "fn: " << *L->getHeader()->getParent() << "\n";
            llvm::errs() << "L: " << *L << "\n";
        }
        assert(loopContexts[L].preheader && "loop must have preheader");

        loopContexts[L].latchMerge = nullptr;

        getExitBlocks(L, loopContexts[L].exitBlocks);
        //if (loopContexts[L].exitBlocks.size() == 0) {
        //    llvm::errs() << "newFunc: " << *BB->getParent() << "\n";
        //    llvm::errs() << "L: " << *L << "\n";
        //}
        //assert(loopContexts[L].exitBlocks.size() > 0);

        auto pair = insertNewCanonicalIV(L, Type::getInt64Ty(BB->getContext()));
        PHINode* CanonicalIV = pair.first;
        assert(CanonicalIV);
        loopContexts[L].var = CanonicalIV;
        loopContexts[L].incvar = pair.second;
        removeRedundantIVs(L, loopContexts[L].header, loopContexts[L].preheader, CanonicalIV, SE, gutils, pair.second, getLatches(L, loopContexts[L].exitBlocks));
        loopContexts[L].antivaralloc = IRBuilder<>(gutils.inversionAllocs).CreateAlloca(CanonicalIV->getType(), nullptr, CanonicalIV->getName()+"'ac");
        loopContexts[L].antivaralloc->setAlignment(cast<IntegerType>(CanonicalIV->getType())->getBitWidth() / 8);

        SCEVUnionPredicate BackedgePred;

      const SCEV *Limit = nullptr;
        {


            const SCEV *MayExitMaxBECount = nullptr;

            SmallVector<BasicBlock *, 8> ExitingBlocks;
            L->getExitingBlocks(ExitingBlocks);

            for (BasicBlock* ExitBB : ExitingBlocks) {
              assert(L->contains(ExitBB));
              auto EL = SE.computeExitLimit(L, ExitBB, /*AllowPredicates*/true);

              //llvm::errs() << "MaxNotTaken:" << *EL.MaxNotTaken << "\n";
              //llvm::errs() << "ExactNotTaken:" << *EL.ExactNotTaken << "\n";

              if (MayExitMaxBECount != SE.getCouldNotCompute()) {
                if (!MayExitMaxBECount || EL.ExactNotTaken == SE.getCouldNotCompute())
                  MayExitMaxBECount = EL.ExactNotTaken;
                else {
                  if (MayExitMaxBECount != EL.ExactNotTaken) {
                    llvm::errs() << "Optimization opportunity! could allocate max!\n";
                    MayExitMaxBECount = SE.getCouldNotCompute();
                    break;
                  }

                  MayExitMaxBECount =
                      SE.getUMaxFromMismatchedTypes(MayExitMaxBECount, EL.ExactNotTaken);
                }
              } else {
                MayExitMaxBECount = SE.getCouldNotCompute();
              }
            }
            if (ExitingBlocks.size() == 0) {
              MayExitMaxBECount = SE.getCouldNotCompute();
            }
            Limit = MayExitMaxBECount;
        }
        assert(Limit);
        //const SCEV *Limit = SE.computeBackedgeTakenCount(L, /*allowpred*/true).getExact(L, &SE, &BackedgePred);// SE.getPredicatedBackedgeTakenCount(L, BackedgePred);

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
             //for(auto B : L->blocks()) {
             //   llvm::errs() << *B << "\n";
             //}
            llvm::errs() << "SE could not compute loop limit of " << L->getHeader()->getName() << " " << L->getHeader()->getParent()->getName() << "\n";

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

Value* GradientUtils::lookupM(Value* val, IRBuilder<>& BuilderM, const ValueToValueMapTy& incoming_available) {
    assert(val->getName() != "<badref>");
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
    assert(inst->getName() != "<badref>");
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
        // This is a reverse block
    } else if (BuilderM.GetInsertBlock() != inversionAllocs) {
      // Something in the entry (or anything that dominates all returns, doesn't need caching)

      BasicBlock* forwardBlock = originalForReverseBlock(*BuilderM.GetInsertBlock());

      // Don't allow this if we're not definitely using the last iteration of this value
      //   + either because the value isn't in a loop
      //   + or because the forward of the block usage location isn't in a loop (thus last iteration)
      //   + or because the loop nests share no ancestry

      bool loopLegal = true;
      for(Loop* idx = LI.getLoopFor(inst->getParent()); idx != nullptr; idx = idx->getParentLoop()) {
        for(Loop* fdx = LI.getLoopFor(forwardBlock); fdx != nullptr; fdx = fdx->getParentLoop()) {
          if (idx == fdx) {
            loopLegal = false;
            break;
          }
        }
      }

      if (loopLegal) {
        if (inst->getParent() == &newFunc->getEntryBlock()) {
          return inst;
        }
        // TODO upgrade this to be all returns that this could enter from
        bool legal = true;
        for(auto &BB: *oldFunc) {
          if (isa<ReturnInst>(BB.getTerminator())) {
            BasicBlock* returningBlock = cast<BasicBlock>(getNewFromOriginal(&BB));
            if (inst->getParent() == returningBlock) continue;
            if (!DT.dominates(inst, returningBlock)) {
              legal = false;
              break;
            }
          }
        }
        if (legal) {
          return inst;
        }
      }
    }

    Instruction* prelcssaInst = inst;

    assert(inst->getName() != "<badref>");
    val = inst = fixLCSSA(inst, BuilderM);

    assert(!this->isOriginalBlock(*BuilderM.GetInsertBlock()));

    auto idx = std::make_pair(val, BuilderM.GetInsertBlock());
    if (lookup_cache.find(idx) != lookup_cache.end()) {
        auto result = lookup_cache[idx];
        assert(result);
        assert(result->getType());
        return result;
    }

    ValueToValueMapTy available;
    for(auto pair : incoming_available) {
      available[pair.first] = pair.second;
    }

    LoopContext lc;
    bool inLoop = getContext(inst->getParent(), lc);

    if (inLoop) {
        bool first = true;
        for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx)) {
          if (!isOriginalBlock(*BuilderM.GetInsertBlock())) {
            available[idx.var] = BuilderM.CreateLoad(idx.antivaralloc);
          } else {
            available[idx.var] = idx.var;
          }
          if (!first && idx.var == inst) return available[idx.var];
          if (first) {
            first = false;
          }
          if (idx.parent == nullptr) break;
        }
    }

    if (available.count(inst)) return available[inst];

    //TODO consider call as part of
    //llvm::errs() << " considering " << *inst << " legal: " << legalRecompute(inst, available) << " should: " << shouldRecompute(inst, available) << "\n";
    if (legalRecompute(prelcssaInst, available)) {
      if (shouldRecompute(prelcssaInst, available)) {
          auto op = unwrapM(prelcssaInst, BuilderM, available, UnwrapMode::AttemptSingleUnwrap);
          //llvm::errs() << "for op " << *inst << " choosing to unwrap and found: " << op << "\n";
          if (op) {
            assert(op);
            assert(op->getType());
            if (auto load_op = dyn_cast<LoadInst>(prelcssaInst)) {
              if (auto new_op = dyn_cast<LoadInst>(op)) {
                  MDNode* invgroup = load_op->getMetadata(LLVMContext::MD_invariant_group);
                  if (invgroup == nullptr) {
                    invgroup = MDNode::getDistinct(load_op->getContext(), {});
                    load_op->setMetadata(LLVMContext::MD_invariant_group, invgroup);
                  }
                  new_op->setMetadata(LLVMContext::MD_invariant_group, invgroup);
              }
            }
            lookup_cache[idx] = op;
            return op;
          }
      } else {
        if (isa<LoadInst>(prelcssaInst)) {
          //llvm::errs() << " + loading " << *inst << "\n";
        }
      }
    }
    //llvm::errs() << "looking from cache: " << *inst << "\n";

    if (auto origInst = isOriginal(inst))
    if (auto li = dyn_cast<LoadInst>(inst)) {

      auto liobj = GetUnderlyingObject(li->getPointerOperand(), oldFunc->getParent()->getDataLayout(), 100);

      for(auto pair : scopeMap) {
        if (auto li2 = dyn_cast<LoadInst>(const_cast<Value*>(pair.first))) {
          if (!isOriginal(li2)) continue;

          auto li2obj = GetUnderlyingObject(li2->getPointerOperand(), oldFunc->getParent()->getDataLayout(), 100);

          if (liobj == li2obj && DT.dominates(li2, li)) {
            bool failed = false;

            llvm::errs() << "found potential candidate loads: oli:" << *origInst << " oli2: " << *getOriginal(li2) << "\n";
            auto scev1 = SE.getSCEV(li->getPointerOperand());
            auto scev2 = SE.getSCEV(li2->getPointerOperand());
            llvm::errs() << " scev1: " << *scev1 << " scev2: " << *scev2 << "\n";

            allInstructionsBetween(OrigLI, getOriginal(li2), origInst, [&](Instruction* I) -> bool {
              //llvm::errs() << "examining instruction: " << *I << " between: " << *li2 << " and " << *li << "\n";
              if ( I->mayWriteToMemory() && writesToMemoryReadBy(AA, /*maybeReader*/origInst, /*maybeWriter*/I) ) {
                failed = true;
                llvm::errs() << "FAILED: " << *I << "\n";
                return /*earlyBreak*/true;
              }
              return /*earlyBreak*/false;
            });
            if (failed) continue;


            if (auto ar1 = dyn_cast<SCEVAddRecExpr>(scev1)) {
              if (auto ar2 = dyn_cast<SCEVAddRecExpr>(scev2)) {
                if (ar1->getStart() != SE.getCouldNotCompute() && ar1->getStart() == ar2->getStart() &&
                    ar1->getStart() != SE.getCouldNotCompute() && ar1->getStepRecurrence(SE) == ar2->getStepRecurrence(SE)) {

                  LoopContext l1;
                  getContext(ar1->getLoop()->getHeader(), l1);
                  LoopContext l2;
                  getContext(ar2->getLoop()->getHeader(), l2);

                  //TODO IF len(ar2) >= len(ar1) then we can replace li with li2
                  if (SE.getSCEV(l1.limit) != SE.getCouldNotCompute() && SE.getSCEV(l1.limit) == SE.getSCEV(l2.limit)) {
                    llvm::errs() << " step1: " << *ar1->getStepRecurrence(SE) << " step2: " << *ar2->getStepRecurrence(SE) << "\n";

                    inst = li2;
                    idx = std::make_pair(val, BuilderM.GetInsertBlock());
                    break;
                  }

                }
              }
            }

          }
        }
      }
    }

    //llvm::errs() << "looking from cache: " << *inst << "\n";

    ensureLookupCached(inst);
    assert(scopeMap[inst]);
    bool isi1 = inst->getType()->isIntegerTy() && cast<IntegerType>(inst->getType())->getBitWidth() == 1;
    Value* result = lookupValueFromCache(BuilderM, inst->getParent(), scopeMap[inst], isi1);
    assert(result->getType() == inst->getType());
    lookup_cache[idx] = result;
    assert(result);
    assert(result->getType());
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
      const auto& edge = pair.first;

      /*
      const auto& targets = pair.second;
      llvm::errs() << " edge: (" << edge.first->getName() << "," << edge.second->getName() << ")\n";
      llvm::errs() << "   targets: [";
      for(auto t : targets) llvm::errs() << t->getName() << ", ";
      llvm::errs() << "]\n";
      */

      blocks.insert(edge.first);
  }

  if (targetToPreds.size() == 3) {
    //llvm::errs() << "trying special 3pair\n";
    for(auto block : blocks) {
      std::set<BasicBlock*> foundtargets;
      std::set<BasicBlock*> uniqueTargets;
      for (BasicBlock* succ : successors(block)) {
          auto edge = std::make_pair(block, succ);
          //llvm::errs() << " edge: " << edge.first->getName() << " | " << edge.second->getName() << "\n";
          for(BasicBlock* target : done[edge]) {
            if (foundtargets.find(target) != foundtargets.end()) {
              goto rnextpair;
            }
            foundtargets.insert(target);
            if (done[edge].size() == 1) uniqueTargets.insert(target);
          }
      }
      if (foundtargets.size() != 3) goto rnextpair;
      if (uniqueTargets.size() != 1) goto rnextpair;

      /*
      llvm::errs() << " valid block1: " << block->getName() << " : targets:[";
      for(auto a : foundtargets) llvm::errs() << a->getName() << ",";
      llvm::errs() << "] uniqueTargets:[";
      for(auto a : uniqueTargets) llvm::errs() << a->getName() << ",";
      llvm::errs() << "]\n";
      */

      {
      BasicBlock* subblock = nullptr;
      for(auto block2 : blocks) {
        std::set<BasicBlock*> seen2;
        //llvm::errs() << " + trying block: " << block2->getName() << "\n";
        for (BasicBlock* succ : successors(block2)) {
          //llvm::errs() << " + + trying succ: " << succ->getName() << "\n";
          auto edge = std::make_pair(block2, succ);
          if (done[edge].size() != 1) {
            //llvm::errs() << " -- failed from noonesize\n";
            goto nextblock;
          }
          for(BasicBlock* target : done[edge]) {
            if (seen2.find(target) != seen2.end()) {
              //llvm::errs() << " -- failed from not uniqueTargets\n";
              goto nextblock;
            }
            seen2.insert(target);
            if (foundtargets.find(target) == foundtargets.end()) {
              //llvm::errs() << " -- failed from not unknown target\n";
              goto nextblock;
            }
            if (uniqueTargets.find(target) != uniqueTargets.end()) {
              //llvm::errs() << " -- failed from not same target\n";
              goto nextblock;
            }
          }
        }
        if (seen2.size() != 2) {
          //llvm::errs() << " -- failed from not 2 seen\n";
          goto nextblock;
        }
        subblock = block2;
        break;
        nextblock:;
      }

      if (subblock == nullptr) goto rnextpair;

      //llvm::errs() << " valid block2: " << subblock->getName() << "\n";

      {
      auto bi1 = cast<BranchInst>(block->getTerminator());

      auto cond1 = lookupM(bi1->getCondition(), BuilderM);
      auto bi2 = cast<BranchInst>(subblock->getTerminator());
      auto cond2 = lookupM(bi2->getCondition(), BuilderM);

      if (replacePHIs == nullptr) {
        BasicBlock* staging = BasicBlock::Create(oldFunc->getContext(), "staging", newFunc);
        auto stagingIfNeeded = [&](BasicBlock* B) {
          auto edge = std::make_pair(block, B);
          if (done[edge].size() == 1) {
            return *done[edge].begin();
          } else {
            return staging;
          }
        };
        BuilderM.CreateCondBr(cond1, stagingIfNeeded(bi1->getSuccessor(0)), stagingIfNeeded(bi1->getSuccessor(1)));
        BuilderM.SetInsertPoint(staging);
        BuilderM.CreateCondBr(cond2, *done[std::make_pair(subblock, bi2->getSuccessor(0))].begin(), *done[std::make_pair(subblock, bi2->getSuccessor(1))].begin());
      } else {
        Value* otherBranch = nullptr;
        for(unsigned i=0; i<2; i++) {
          Value* val = cond1;
          if (i == 1) val = BuilderM.CreateNot(val, "anot1_");
          auto edge = std::make_pair(block, bi1->getSuccessor(i));
          if (done[edge].size() == 1) {
            auto found = replacePHIs->find(*done[edge].begin());
            if(found == replacePHIs->end()) continue;
            if (&*BuilderM.GetInsertPoint() == found->second) {
              if (found->second->getNextNode())
                BuilderM.SetInsertPoint(found->second->getNextNode());
              else
                BuilderM.SetInsertPoint(found->second->getParent());
            }
            found->second->replaceAllUsesWith(val);
            found->second->eraseFromParent();
          } else {
            otherBranch = val;
          }
        }

        for(unsigned i=0; i<2; i++) {
          auto edge = std::make_pair(subblock, bi2->getSuccessor(i));
          auto found = replacePHIs->find(*done[edge].begin());
          if(found == replacePHIs->end()) continue;

          Value* val = cond2;
          if (i == 1) val = BuilderM.CreateNot(val, "bnot1_");
          val = BuilderM.CreateAnd(val, otherBranch, "andVal" + i);
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
      }
      rnextpair:;
    }
  }

  BasicBlock* forwardBlock = BuilderM.GetInsertBlock();

  if (!isOriginalBlock(*forwardBlock)) {
      forwardBlock = originalForReverseBlock(*forwardBlock);
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

      if (forwardBlock == block || DT.dominates(block, forwardBlock)) {
        equivalentTerminator = block->getTerminator();
        goto fast;
      }

      nextpair:;
  }
  goto nofast;


  fast:;
  assert(equivalentTerminator);

  if (auto branch = dyn_cast<BranchInst>(equivalentTerminator)) {
      BasicBlock* block = equivalentTerminator->getParent();
      assert(branch->getCondition());

      assert(branch->getCondition()->getType() == T);

      if (replacePHIs == nullptr) {
          assert(BuilderM.GetInsertBlock()->size() == 0 || !isa<BranchInst>(BuilderM.GetInsertBlock()->back()));
          BuilderM.CreateCondBr(lookupM(branch->getCondition(), BuilderM), *done[std::make_pair(block, branch->getSuccessor(0))].begin(), *done[std::make_pair(block, branch->getSuccessor(1))].begin());
      } else {
          for (auto pair : *replacePHIs) {
              Value* phi = lookupM(branch->getCondition(), BuilderM);
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


      if (replacePHIs == nullptr) {
          SwitchInst* swtch = BuilderM.CreateSwitch(lookupM(si->getCondition(), BuilderM), *done[std::make_pair(block, si->getDefaultDest())].begin());
          for (auto switchcase : si->cases()) {
              swtch->addCase(switchcase.getCaseValue(), *done[std::make_pair(block, switchcase.getCaseSuccessor())].begin());
          }
      } else {
          for (auto pair : *replacePHIs) {
              Value* cas = si->findCaseDest(pair.first);
              Value* val = nullptr;
              Value* phi = lookupM(si->getCondition(), BuilderM);
              if (cas) {
                  val = BuilderM.CreateICmpEQ(cas, phi);
              } else {
                  //default case
                  val = ConstantInt::getFalse(pair.second->getContext());
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

  bool isi1 = T->isIntegerTy() && cast<IntegerType>(T)->getBitWidth() == 1;
  Value* which = lookupValueFromCache(BuilderM, ctx, cache, isi1);
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
