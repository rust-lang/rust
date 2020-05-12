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

#include "llvm/IR/GlobalValue.h"

#include "llvm/IR/Constants.h"

#include "llvm/Analysis/ValueTracking.h"
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

bool GradientUtils::legalRecompute(Value* val, const ValueToValueMapTy& available) {
  if (available.count(val)) {
    return true;
  }

  if (isa<PHINode>(val)) {
    if (auto dli = dyn_cast_or_null<LoadInst>(hasUninverted(val))) {
      return legalRecompute(dli, available); // TODO ADD && !TR.intType(getOriginal(dli), /*mustfind*/false).isPossibleFloat();
    }
    return false;
  }

  if (isa<Instruction>(val) && cast<Instruction>(val)->getMetadata("enzyme_mustcache")) return false;

  // If this is a load from cache already, dont force a cache of this
  if (isa<LoadInst>(val) && cast<LoadInst>(val)->getMetadata("enzyme_fromcache")) return true;

  //TODO consider callinst here

  if (auto li = dyn_cast<LoadInst>(val)) {

    // If this is an already unwrapped value, legal to recompute again.
    if (li->getMetadata("enzyme_unwrapped"))
      return true;

    Instruction* orig = nullptr;
    if (li->getParent()->getParent() == oldFunc) {
      orig = li;
    } else {
      orig = isOriginal(li);
    }

    if (orig) {
      auto found = can_modref_map->find(orig);
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

  if (auto inst = dyn_cast<Instruction>(val)) {
    return !inst->mayReadOrWriteMemory();
  }

  return true;
}

//! Given the option to recompute a value or re-use an old one, return true if it is faster to recompute this value from scratch
bool GradientUtils::shouldRecompute(Value* val, const ValueToValueMapTy& available) {
  if (available.count(val)) return true;
  //TODO: remake such that this returns whether a load to a cache is more expensive than redoing the computation.

  // If this is a load from cache already, just reload this
  if (isa<LoadInst>(val) && cast<LoadInst>(val)->getMetadata("enzyme_fromcache")) return true;

  if (isa<CastInst>(val) || isa<GetElementPtrInst>(val)) return true;

  //llvm::errs() << " considering recompute of " << *val << "\n";

  // if this has operands that need to be loaded and haven't already been loaded (TODO), just cache this
  if (auto inst = dyn_cast<Instruction>(val)) {
    for(auto &op : inst->operands()) {
      //llvm::errs() << "   + " << *op << " legalRecompute:" << legalRecompute(op, available) << "\n";
      if (!legalRecompute(op, available)) {

        // If this is a load from cache already, dont force a cache of this
        if (isa<LoadInst>(op) && cast<LoadInst>(op)->getMetadata("enzyme_fromcache")) continue;

        //llvm::errs() << "choosing to cache " << *val << " because of " << *op << "\n";
        return false;
      }
    }
  }

  if (auto op = dyn_cast<IntrinsicInst>(val)) {
    switch(op->getIntrinsicID()) {
      case Intrinsic::sin:
      case Intrinsic::cos:
      return true;
      return shouldRecompute(op->getOperand(0), available);
      default:
      return false;
    }
  }

  //cache a call, assuming its longer to run that
  if (isa<CallInst>(val)) return false;

  //llvm::errs() << "unknown inst " << *val << " unable to recompute\n";
  return true;
}

GradientUtils* GradientUtils::CreateFromClone(Function *todiff, TargetLibraryInfo &TLI, TypeAnalysis &TA, AAResults &AA, DIFFE_TYPE retType, const std::vector<DIFFE_TYPE> & constant_args, bool returnUsed, std::map<AugmentedStruct, unsigned> &returnMapping ) {
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
          type_args.first.insert(std::pair<Argument*, ValueData>(&a, DataType(IntType::Unknown)));
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
    }
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

    LoopContext lc;
    bool inLoop = getContext(inst->getParent(), lc);

    ValueToValueMapTy available;
    for(auto pair : incoming_available) {
      available[pair.first] = pair.second;
    }

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
    if (legalRecompute(inst, available)) {
      if (shouldRecompute(inst, available)) {
          auto op = unwrapM(inst, BuilderM, available, UnwrapMode::AttemptSingleUnwrap);
          //llvm::errs() << "for op " << *inst << " choosing to unwrap and found: " << op << "\n";
          if (op) {
            assert(op);
            assert(op->getType());
            if (auto load_op = dyn_cast<LoadInst>(inst)) {
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
        if (isa<LoadInst>(inst)) {
          //llvm::errs() << " + loading " << *inst << "\n";
        }
      }
    }
    //llvm::errs() << "looking from cache: " << *inst << "\n";

    ensureLookupCached(inst);
    assert(scopeMap[inst]);
    Value* result = lookupValueFromCache(BuilderM, inst->getParent(), scopeMap[inst]);
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
