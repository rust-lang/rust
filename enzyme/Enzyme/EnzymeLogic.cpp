/*
 * EnzymeLogic.cpp
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

#include "EnzymeLogic.h"

#include "SCEV/ScalarEvolutionExpander.h"

#include <deque>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Verifier.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "Utils.h"
#include "GradientUtils.h"
#include "FunctionUtils.h"
#include "LibraryFuncs.h"

using namespace llvm;

llvm::cl::opt<bool> enzyme_print("enzyme_print", cl::init(false), cl::Hidden,
                cl::desc("Print before and after fns for autodiff"));

cl::opt<bool> cache_reads_always(
            "enzyme_always_cache_reads", cl::init(false), cl::Hidden,
            cl::desc("Force always caching of all reads"));

cl::opt<bool> cache_reads_never(
            "enzyme_never_cache_reads", cl::init(false), cl::Hidden,
            cl::desc("Force never caching of all reads"));



// Computes a map of LoadInst -> boolean for a function indicating whether that load is "uncacheable".
//   A load is considered "uncacheable" if the data at the loaded memory location can be modified after
//   the load instruction.
std::map<Instruction*, bool> compute_uncacheable_load_map(GradientUtils* gutils, AAResults& AA, TargetLibraryInfo& TLI,
    const std::set<unsigned> uncacheable_args) {
  std::map<Instruction*, bool> can_modref_map;
  for(BasicBlock* BB: gutils->originalBlocks) {
    for (auto I = BB->begin(), E = BB->end(); I != E; I++) {
      Instruction* inst = &*I;
      // For each load instruction, determine if it is uncacheable.
      if (auto op = dyn_cast<LoadInst>(inst)) {
        // NOTE(TFK): The reasoning behind skipping ConstantValues and ConstantInstructions needs to be fleshed out.
        //if (gutils->isConstantValue(inst) || gutils->isConstantInstruction(inst)) {
        //  continue;
        //}

        bool can_modref = false;
        // Find the underlying object for the pointer operand of the load instruction.
        auto obj = GetUnderlyingObject(op->getPointerOperand(), BB->getModule()->getDataLayout(), 100);
        // If the pointer operand is from an argument to the function, we need to check if the argument
        //   received from the caller is uncacheable.
        if (auto arg = dyn_cast<Argument>(obj)) {
          if (uncacheable_args.find(arg->getArgNo()) != uncacheable_args.end()) {
            can_modref = true;
          }
        } else {
          // NOTE(TFK): In the case where the underlying object for the pointer operand is from a Load or Call we need
          //  to check if we need to cache. Likely, we need to play it safe in this case and cache.
          // NOTE(TFK): The logic below is an attempt at a conservative handling of the case mentioned above, but it
          //   needs to be verified.

          // Pointer operands originating from call instructions that are not malloc/free are conservatively considered uncacheable.
          if (auto obj_op = dyn_cast<CallInst>(obj)) {
            Function* called = obj_op->getCalledFunction();
            if (auto castinst = dyn_cast<ConstantExpr>(obj_op->getCalledValue())) {
              if (castinst->isCast()) {
                if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                  if (isAllocationFunction(*fn, TLI) || isDeallocationFunction(*fn, TLI)) {
                    called = fn;
                  }
                }
              }
            }
            if (isCertainMallocOrFree(called)) {
              //llvm::errs() << "OP is certain malloc or free: " << *op << "\n";
            } else {
              //llvm::errs() << "OP is a non malloc/free call so we need to cache " << *op << "\n";
              can_modref = true;
            }
          } else if (isa<LoadInst>(obj)) {
            // If obj is from a load instruction conservatively consider it uncacheable.
            can_modref = true;
          } else {
            // In absence of more information, assume that the underlying object for pointer operand is uncacheable in caller.
            can_modref = true;
          }
        }

        for (BasicBlock* BB2 : gutils->originalBlocks) {
          for (auto I2 = BB2->begin(), E2 = BB2->end(); I2 != E2; I2++) {
            Instruction* inst2 = &*I2;
            if (inst == inst2) continue;
            if (!gutils->DT.dominates(inst2, inst)) {
              if (llvm::isModSet(AA.getModRefInfo(inst2, MemoryLocation::get(op)))) {
                can_modref = true;
                //llvm::errs() << *inst << " needs to be cached due to: " << *inst2 << "\n";
                break;
              }
            }
          }
        }
        can_modref_map[inst] = can_modref;
      }
    }
  }
  return can_modref_map;
}

std::set<unsigned> compute_uncacheable_args_for_one_callsite(Instruction* callsite_inst, DominatorTree &DT,
    TargetLibraryInfo &TLI, AAResults& AA, GradientUtils* gutils, const std::set<unsigned> parent_uncacheable_args) {
  CallInst* callsite_op = dyn_cast<CallInst>(callsite_inst);
  assert(callsite_op != nullptr);

  std::set<unsigned> uncacheable_args;
  std::vector<Value*> args;
  std::vector<bool> args_safe;

  // First, we need to propagate the uncacheable status from the parent function to the callee.
  //   because memory location x modified after parent returns => x modified after callee returns.
  for (unsigned i = 0; i < callsite_op->getNumArgOperands(); i++) {
      args.push_back(callsite_op->getArgOperand(i));
      bool init_safe = true;

      // If the UnderlyingObject is from one of this function's arguments, then we need to propagate the volatility.
      Value* obj = GetUnderlyingObject(callsite_op->getArgOperand(i),
                                       callsite_inst->getParent()->getModule()->getDataLayout(),
                                       100);
      // If underlying object is an Argument, check parent volatility status.
      if (auto arg = dyn_cast<Argument>(obj)) {
        if (parent_uncacheable_args.find(arg->getArgNo()) != parent_uncacheable_args.end()) {
          init_safe = false;
        }
      } else {
        // Pointer operands originating from call instructions that are not malloc/free are conservatively considered uncacheable.
        if (auto obj_op = dyn_cast<CallInst>(obj)) {
          Function* called = obj_op->getCalledFunction();
          if (auto castinst = dyn_cast<ConstantExpr>(obj_op->getCalledValue())) {
            if (castinst->isCast()) {
              if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                if (isAllocationFunction(*fn, TLI) || isDeallocationFunction(*fn, TLI)) {
                  called = fn;
                }
              }
            }
          }
          if (isCertainMallocOrFree(called)) {
            //llvm::errs() << "OP is certain malloc or free: " << *op << "\n";
          } else {
            //llvm::errs() << "OP is a non malloc/free call so we need to cache " << *op << "\n";
            init_safe = false;
          }
        } else if (isa<LoadInst>(obj)) {
          // If obj is from a load instruction conservatively consider it uncacheable.
          init_safe = false;
        } else {
          // In absence of more information, assume that the underlying object for pointer operand is uncacheable in caller.
          init_safe = false;
        }
      }
      // TODO(TFK): Also need to check whether underlying object is traced to load / non-allocating-call instruction.
      args_safe.push_back(init_safe);
  }

  // Second, we check for memory modifications that can occur in the continuation of the
  //   callee inside the parent function.
  for(BasicBlock* BB: gutils->originalBlocks) {
    for (auto I = BB->begin(), E = BB->end(); I != E; I++) {
      Instruction* inst = &*I;
     
      // If the "inst" does not dominate "callsite_inst" then we cannot prove that
      //   "inst" happens before "callsite_inst". If "inst" modifies an argument of the call,
      //   then that call needs to consider the argument uncacheable.
      // To correctly handle case where inst == callsite_inst, we need to look at next instruction after callsite_inst.
      if (!gutils->DT.dominates(inst, callsite_inst->getNextNonDebugInstruction())) {
        //llvm::errs() << "Instruction " << *inst << " DOES NOT dominates " << *callsite_inst << "\n";
        // Consider Store Instructions.
        if (auto op = dyn_cast<StoreInst>(inst)) {
          for (unsigned i = 0; i < args.size(); i++) {
            // If the modification flag is set, then this instruction may modify the $i$th argument of the call.
            if (!llvm::isModSet(AA.getModRefInfo(op, MemoryLocation::getForArgument(callsite_op, i, TLI)))) {
              //llvm::errs() << "Instruction " << *op << " is NoModRef with call argument " << *args[i] << "\n";
            } else {
              //llvm::errs() << "Instruction " << *op << " is maybe ModRef with call argument " << *args[i] << "\n";
              args_safe[i] = false;
            }
          }
        }

        // Consider Call Instructions.
        if (auto op = dyn_cast<CallInst>(inst)) {
          //llvm::errs() << "OP is call inst: " << *op << "\n";
          // Ignore memory allocation functions.
          Function* called = op->getCalledFunction();
          if (auto castinst = dyn_cast<ConstantExpr>(op->getCalledValue())) {
            if (castinst->isCast()) {
              if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                if (isAllocationFunction(*fn, TLI) || isDeallocationFunction(*fn, TLI)) {
                  called = fn;
                }
              }
            }
          }
          if (isCertainMallocOrFree(called)) {
            //llvm::errs() << "OP is certain malloc or free: " << *op << "\n";
            continue;
          }

          // For all the arguments, perform same check as for Stores, but ignore non-pointer arguments.
          for (unsigned i = 0; i < args.size(); i++) {
            if (!args[i]->getType()->isPointerTy()) continue;  // Ignore non-pointer arguments.
            if (!llvm::isModSet(AA.getModRefInfo(op, MemoryLocation::getForArgument(callsite_op, i, TLI)))) {
              //llvm::errs() << "Instruction " << *op << " is NoModRef with call argument " << *args[i] << "\n";
            } else {
              //llvm::errs() << "Instruction " << *op << " is maybe ModRef with call argument " << *args[i] << "\n";
              args_safe[i] = false;
            }
          }
        }
      } else {
        //llvm::errs() << "Instruction " << *inst << " DOES dominates " << *callsite_inst << "\n";
      } 
    }
  }

  //llvm::errs() << "CallInst: " << *callsite_op<< "CALL ARGUMENT INFO: \n";
  for (unsigned i = 0; i < args.size(); i++) {
    if (!args_safe[i]) {
      uncacheable_args.insert(i);
    }
    //llvm::errs() << "Arg: " << *args[i] << " STATUS: " << args_safe[i] << "\n";
  }
  return uncacheable_args;
}

// Given a function and the arguments passed to it by its caller that are uncacheable (_uncacheable_args) compute
//   the set of uncacheable arguments for each callsite inside the function. A pointer argument is uncacheable at
//   a callsite if the memory pointed to might be modified after that callsite.
std::map<CallInst*, std::set<unsigned> > compute_uncacheable_args_for_callsites(
    Function* F, DominatorTree &DT, TargetLibraryInfo &TLI, AAResults& AA, GradientUtils* gutils,
    const std::set<unsigned> uncacheable_args) {
  std::map<CallInst*, std::set<unsigned> > uncacheable_args_map;
  for(BasicBlock* BB: gutils->originalBlocks) {
    for (auto I = BB->begin(), E = BB->end(); I != E; I++) {
      Instruction* inst = &*I;
      if (auto op = dyn_cast<CallInst>(inst)) {

        // We do not need uncacheable args for intrinsic functions. So skip such callsites.
        if(isa<IntrinsicInst>(inst)) { 
          continue;
        }

        // We do not need uncacheable args for memory allocation functions. So skip such callsites. 
        Function* called = op->getCalledFunction();
        if (auto castinst = dyn_cast<ConstantExpr>(op->getCalledValue())) {
          if (castinst->isCast()) {
            if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
              if (isAllocationFunction(*fn, TLI) || isDeallocationFunction(*fn, TLI)) {
                called = fn;
              }
            }
          }
        }
        if (isCertainMallocOrFree(called)) {
          continue;
        }

        // For all other calls, we compute the uncacheable args for this callsite.
        uncacheable_args_map[op] = compute_uncacheable_args_for_one_callsite(inst,
            DT, TLI, AA, gutils, uncacheable_args);
      }
    }
  }
  return uncacheable_args_map;
}

// Determine if a load is needed in the reverse pass. We only use this logic in the top level function right now.
bool is_load_needed_in_reverse(GradientUtils* gutils, AAResults& AA, Instruction* inst) {

  std::vector<Value*> uses_list;
  std::set<Value*> uses_set;
  uses_list.push_back(inst);
  uses_set.insert(inst);

  while (true) {
    bool new_user_added = false;
    for (unsigned i = 0; i < uses_list.size(); i++) {
      for (auto use = uses_list[i]->user_begin(), end = uses_list[i]->user_end(); use != end; ++use) {
        Value* v = (*use);
        //llvm::errs() << "Use list: " << *v << "\n";
        if (uses_set.find(v) == uses_set.end()) {
          uses_set.insert(v);
          uses_list.push_back(v);
          new_user_added = true;
        }
      }
    }
    if (!new_user_added) break;
  }
  //llvm::errs() << "Analysis for load " << *inst << " which has nuses: " << inst->getNumUses() << "\n"; 
  for (unsigned i = 0; i < uses_list.size(); i++) {
    //llvm::errs() << "Considering use " << *uses_list[i] << "\n";
    if (uses_list[i] == dyn_cast<Value>(inst)) continue;

    if (isa<CmpInst>(uses_list[i]) || isa<BranchInst>(uses_list[i]) || isa<BitCastInst>(uses_list[i]) || isa<PHINode>(uses_list[i]) || isa<ReturnInst>(uses_list[i]) || isa<FPExtInst>(uses_list[i]) ||
        isa<LoadInst>(uses_list[i]) /*|| isa<StoreInst>(uses_list[i])*/){
      continue;
    }

    if (auto op = dyn_cast<BinaryOperator>(uses_list[i])) {
      if (op->getOpcode() == Instruction::FAdd || op->getOpcode() == Instruction::FSub) {
        continue;
      } else {
        //llvm::errs() << "Need value of " << *inst << "\n" << "\t Due to " << *op << "\n";
        return true;
      }
    }

    //if (auto op = dyn_cast<CallInst>(uses_list[i])) {
    //  llvm::errs() << "Need value of " << *inst << "\n" << "\t Due to " << *op << "\n";
    //  return true;
    //}

    //llvm::errs() << "Need value of " << *inst << "\n" << "\t Due to " << *uses_list[i] << "\n";
    //return true;
  }
  return false;
}


//! return structtype if recursive function
std::pair<Function*,StructType*> CreateAugmentedPrimal(Function* todiff, AAResults &global_AA, const std::set<unsigned>& constant_args, TargetLibraryInfo &TLI, bool differentialReturn, bool returnUsed, const std::set<unsigned> _uncacheable_args) {
  static std::map<std::tuple<Function*,std::set<unsigned>/*constant_args*/, std::set<unsigned>/*uncacheable_args*/, bool/*differentialReturn*/, bool/*returnUsed*/>, std::pair<Function*,StructType*>> cachedfunctions;
  static std::map<std::tuple<Function*,std::set<unsigned>/*constant_args*/, std::set<unsigned>/*uncacheable_args*/, bool/*differentialReturn*/, bool/*returnUsed*/>, bool> cachedfinished;
  auto tup = std::make_tuple(todiff, std::set<unsigned>(constant_args.begin(), constant_args.end()), std::set<unsigned>(_uncacheable_args.begin(), _uncacheable_args.end()), differentialReturn, returnUsed);
  if (cachedfunctions.find(tup) != cachedfunctions.end()) {
    return cachedfunctions[tup];
  }
  if (differentialReturn) assert(returnUsed);

    if (constant_args.size() == 0 && hasMetadata(todiff, "enzyme_augment")) {
      auto md = todiff->getMetadata("enzyme_augment");
      if (!isa<MDTuple>(md)) {
          llvm::errs() << *todiff << "\n";
          llvm::errs() << *md << "\n";
          report_fatal_error("unknown augment for noninvertible function -- metadata incorrect");
      }
      auto md2 = cast<MDTuple>(md);
      assert(md2->getNumOperands() == 1);
      auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
      auto foundcalled = cast<Function>(gvemd->getValue());

      if (foundcalled->getReturnType() == todiff->getReturnType()) {
        FunctionType *FTy = FunctionType::get(StructType::get(todiff->getContext(), {StructType::get(todiff->getContext(), {}), foundcalled->getReturnType()}),
                                   foundcalled->getFunctionType()->params(), foundcalled->getFunctionType()->isVarArg());
        Function *NewF = Function::Create(FTy, Function::LinkageTypes::InternalLinkage, "fixaugmented_"+todiff->getName(), todiff->getParent());
        NewF->setAttributes(foundcalled->getAttributes());
        if (NewF->hasFnAttribute(Attribute::NoInline)) {
            NewF->removeFnAttr(Attribute::NoInline);
        }
        for (auto i=foundcalled->arg_begin(), j=NewF->arg_begin(); i != foundcalled->arg_end(); ) {
             j->setName(i->getName());
             if (j->hasAttribute(Attribute::Returned))
                 j->removeAttr(Attribute::Returned);
             if (j->hasAttribute(Attribute::StructRet))
                 j->removeAttr(Attribute::StructRet);
             i++;
             j++;
         }
        BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
        IRBuilder <>bb(BB);
        SmallVector<Value*,4> args;
        for(auto &a : NewF->args()) args.push_back(&a);
        auto cal = bb.CreateCall(foundcalled, args);
        cal->setCallingConv(foundcalled->getCallingConv());
        auto ut = UndefValue::get(NewF->getReturnType());
        auto val = bb.CreateInsertValue(ut, cal, {1u});
        bb.CreateRet(val);
        return cachedfunctions[tup] = std::pair<Function*,StructType*>(NewF, nullptr);
      }

      //assert(st->getNumElements() > 0);
      return cachedfunctions[tup] = std::pair<Function*,StructType*>(foundcalled, nullptr); //dyn_cast<StructType>(st->getElementType(0)));
    }





  if (todiff->empty()) {
    llvm::errs() << *todiff << "\n";
  }
  assert(!todiff->empty());
  AAResults AA(TLI);
  GradientUtils *gutils = GradientUtils::CreateFromClone(todiff, AA, TLI, constant_args, /*returnValue*/returnUsed ? ReturnType::TapeAndReturns : ReturnType::Tape, /*differentialReturn*/differentialReturn);
  cachedfunctions[tup] = std::pair<Function*,StructType*>(gutils->newFunc, nullptr);
  cachedfinished[tup] = false;

  std::map<CallInst*, std::set<unsigned> > uncacheable_args_map =
      compute_uncacheable_args_for_callsites(gutils->oldFunc, gutils->DT, TLI, AA, gutils, _uncacheable_args);

  std::map<Instruction*, bool> can_modref_map = compute_uncacheable_load_map(gutils, AA, TLI, _uncacheable_args);
  gutils->can_modref_map = &can_modref_map;

  // Allow forcing cache reads to be on or off using flags.
  assert(!(cache_reads_always && cache_reads_never) && "Both cache_reads_always and cache_reads_never are true. This doesn't make sense.");
  if (cache_reads_always || cache_reads_never) {
    bool is_needed = cache_reads_always ? true : false;
    for (auto iter = can_modref_map.begin(); iter != can_modref_map.end(); iter++) {
      can_modref_map[iter->first] = is_needed;
    }
  } 


    //for (auto iter = can_modref_map.begin(); iter != can_modref_map.end(); iter++) {
    //  if (iter->second) {
    //    bool is_needed = is_load_needed_in_reverse(gutils, AA, iter->first);
    //    can_modref_map[iter->first] = is_needed;
    //  }
    //}


  gutils->forceContexts();
  gutils->forceAugmentedReturns();
  
  //! Explicitly handle all returns first to ensure that all instructions know whether or not they are used
  SmallPtrSet<Instruction*, 4> returnuses;

  for(BasicBlock* BB: gutils->originalBlocks) {
    if(auto ri = dyn_cast<ReturnInst>(BB->getTerminator())) {
        auto oldval = ri->getReturnValue();
        Value* rt = UndefValue::get(gutils->newFunc->getReturnType());
        IRBuilder <>ib(ri);
        if (oldval && returnUsed) {
            rt = ib.CreateInsertValue(rt, oldval, {1});
            if (Instruction* inst = dyn_cast<Instruction>(rt)) {
                returnuses.insert(inst);
            }
        }
        ib.CreateRet(rt);
        gutils->erase(ri);
        /*
         TODO should call DCE now ideally
        if (auto inst = dyn_cast<Instruction>(rt)) {
            SmallSetVector<Instruction *, 16> WorkList;
            DCEInstruction(inst, WorkList, TLI);
            while (!WorkList.empty()) {
                Instruction *I = WorkList.pop_back_val();
                MadeChange |= DCEInstruction(I, WorkList, TLI);
            }
        }
        */
    }
  }

  for(BasicBlock* BB: gutils->originalBlocks) {
      auto term = BB->getTerminator();
      assert(term);
      if (isa<ReturnInst>(term) || isa<BranchInst>(term) || isa<SwitchInst>(term)) {

      } else if (isa<UnreachableInst>(term)) {

      } else {
        assert(BB);
        assert(BB->getParent());
        assert(term);
        llvm::errs() << *BB->getParent() << "\n";
        llvm::errs() << "unknown terminator instance " << *term << "\n";
        assert(0 && "unknown terminator inst");
      }

      if (!isa<UnreachableInst>(term))
      for (BasicBlock::reverse_iterator I = BB->rbegin(), E = BB->rend(); I != E;) {
        Instruction* inst = &*I;
        assert(inst);
        I++;
        if (gutils->originalInstructions.find(inst) == gutils->originalInstructions.end()) continue;
    
        if(auto op = dyn_cast_or_null<IntrinsicInst>(inst)) {
          switch(op->getIntrinsicID()) {
            case Intrinsic::memcpy: {
                if (gutils->isConstantInstruction(inst)) continue;

                if (!isIntPointerASecretFloat(op->getOperand(0)) ) {
                    SmallVector<Value*, 4> args;
                    IRBuilder <>BuilderZ(op);
                    args.push_back(gutils->invertPointerM(op->getOperand(0), BuilderZ));
                    args.push_back(gutils->invertPointerM(op->getOperand(1), BuilderZ));
                    args.push_back(op->getOperand(2));
                    args.push_back(op->getOperand(3));

                    Type *tys[] = {args[0]->getType(), args[1]->getType(), args[2]->getType()};
                    auto cal = BuilderZ.CreateCall(Intrinsic::getDeclaration(gutils->newFunc->getParent(), Intrinsic::memcpy, tys), args);
                    cal->setAttributes(op->getAttributes());
                    cal->setCallingConv(op->getCallingConv());
                    cal->setTailCallKind(op->getTailCallKind());
                } else {
                    //no change to forward pass if represents floats
                }
                break;
            }
            case Intrinsic::memset: {
                if (gutils->isConstantInstruction(inst)) continue;
                /*
                if (!gutils->isConstantValue(op->getOperand(1))) {
                    assert(inst);
                    llvm::errs() << "couldn't handle non constant inst in memset to propagate differential to\n" << *inst;
                    report_fatal_error("non constant in memset");
                }
                auto ptx = invertPointer(op->getOperand(0));
                SmallVector<Value*, 4> args;
                args.push_back(ptx);
                args.push_back(lookup(op->getOperand(1)));
                args.push_back(lookup(op->getOperand(2)));
                args.push_back(lookup(op->getOperand(3)));

                Type *tys[] = {args[0]->getType(), args[2]->getType()};
                auto cal = Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args);
                cal->setAttributes(op->getAttributes());
                */
                break;
            }
            case Intrinsic::stacksave:
            case Intrinsic::stackrestore:
            case Intrinsic::dbg_declare:
            case Intrinsic::dbg_value:
            #if LLVM_VERSION_MAJOR > 6
            case Intrinsic::dbg_label:
            #endif
            case Intrinsic::dbg_addr:
            case Intrinsic::lifetime_start:
            case Intrinsic::lifetime_end:
            case Intrinsic::assume:
            case Intrinsic::fabs:
            case Intrinsic::x86_sse_max_ss:
            case Intrinsic::x86_sse_max_ps:
            case Intrinsic::maxnum:
            case Intrinsic::x86_sse_min_ss:
            case Intrinsic::x86_sse_min_ps:
            case Intrinsic::minnum:
            case Intrinsic::log:
            case Intrinsic::log2:
            case Intrinsic::log10:
            case Intrinsic::exp:
            case Intrinsic::exp2:
            case Intrinsic::pow:
            case Intrinsic::sin:
            case Intrinsic::cos:
            case Intrinsic::floor:
            case Intrinsic::ceil:
            case Intrinsic::trunc:
            case Intrinsic::rint:
            case Intrinsic::nearbyint:
            case Intrinsic::round:
                break;
            default:
              if (gutils->isConstantInstruction(inst)) continue;
              assert(inst);
              llvm::errs() << *gutils->oldFunc << "\n";
              llvm::errs() << *gutils->newFunc << "\n";
              llvm::errs() << "cannot handle (augmented) unknown intrinsic\n" << *inst;
              report_fatal_error("(augmented) unknown intrinsic");
          }

        } else if(auto op = dyn_cast_or_null<CallInst>(inst)) {

            Function *called = op->getCalledFunction();

            if (auto castinst = dyn_cast<ConstantExpr>(op->getCalledValue())) {
                if (castinst->isCast())
                if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                    if (fn->getName() == "malloc" || fn->getName() == "free" || fn->getName() == "_Znwm" || fn->getName() == "_ZdlPv" || fn->getName() == "_ZdlPvm") {
                        called = fn;
                    }
                }
            }

            if (called && (called->getName() == "printf" || called->getName() == "puts"))
                continue;

            if (called && (called->getName()=="malloc" || called->getName()=="_Znwm")) {
                //TODO enable this if we need to free the memory
                if (false && op->getNumUses() != 0) {
                    IRBuilder<> BuilderZ(op);
                    gutils->addMalloc(BuilderZ, op);
                }
                if (!gutils->isConstantValue(op)) {
                    auto placeholder = cast<PHINode>(gutils->invertedPointers[op]);
                    gutils->createAntiMalloc(op);
                    if (I != E && placeholder == &*I) I++;
                }
                continue;
            }

            if (called && (called->getName()=="free" ||
                called->getName()=="_ZdlPv" || called->getName()=="_ZdlPvm"))
                continue;

            if (gutils->isConstantInstruction(op)) {
                if (op->getNumUses() != 0 && !op->doesNotAccessMemory()) {
                    IRBuilder<> BuilderZ(op);
                    gutils->addMalloc(BuilderZ, op);
                }
                continue;
            }

            if (called == nullptr) {
              assert(op);
              llvm::errs() << "cannot handle augment non constant function\n" << *op << "\n";
              report_fatal_error("unknown augment non constant function");
            }
 
            std::set<unsigned> subconstant_args;

              SmallVector<Value*, 8> args;
              SmallVector<DIFFE_TYPE, 8> argsInverted;
              bool modifyPrimal = !called->hasFnAttribute(Attribute::ReadNone);
              IRBuilder<> BuilderZ(op);
              BuilderZ.setFastMathFlags(getFast());

              if ( (op->getType()->isPointerTy() || op->getType()->isIntegerTy()) && !gutils->isConstantValue(op) ) {
                 modifyPrimal = true;
                 //llvm::errs() << "primal modified " << called->getName() << " modified via return" << "\n";
              }

              if (called->empty()) modifyPrimal = true;

              for(unsigned i=0;i<op->getNumArgOperands(); i++) {
                args.push_back(op->getArgOperand(i));

                if (gutils->isConstantValue(op->getArgOperand(i)) && !called->empty()) {
                    subconstant_args.insert(i);
                    argsInverted.push_back(DIFFE_TYPE::CONSTANT);
                    continue;
                }

                auto argType = op->getArgOperand(i)->getType();

                if (argType->isPointerTy() || argType->isIntegerTy()) {
                    argsInverted.push_back(DIFFE_TYPE::DUP_ARG);
                    args.push_back(gutils->invertPointerM(op->getArgOperand(i), BuilderZ));

                    if (! ( called->hasParamAttribute(i, Attribute::ReadOnly) || called->hasParamAttribute(i, Attribute::ReadNone)) ) {
                        modifyPrimal = true;
                        //llvm::errs() << "primal modified " << called->getName() << " modified via arg " << i << "\n";
                    }
                    //Note sometimes whattype mistakenly says something should be constant [because composed of integer pointers alone]
                    assert(whatType(argType) == DIFFE_TYPE::DUP_ARG || whatType(argType) == DIFFE_TYPE::CONSTANT);
                } else {
                    argsInverted.push_back(DIFFE_TYPE::OUT_DIFF);
                    assert(whatType(argType) == DIFFE_TYPE::OUT_DIFF || whatType(argType) == DIFFE_TYPE::CONSTANT);
                }
              }

              if (!modifyPrimal) {
                if (op->getNumUses() != 0 && !op->doesNotAccessMemory()) {
                  gutils->addMalloc(BuilderZ, op);
                }
                continue;
              }

                bool subretused = op->getNumUses() != 0;
                bool subdifferentialreturn = (!gutils->isConstantValue(op)) && subretused;
                
                //! We only need to cache something if it is used in a non return setting (since the backard pass doesnt need to use it if just returned)
                bool shouldCache = false;//outermostAugmentation;
                for(auto use : op->users()) {
                    if (!isa<Instruction>(use) || returnuses.find(cast<Instruction>(use)) == returnuses.end()) {
                        llvm::errs() << "shouldCache for " << *op << " use " << *use << "\n";
                        shouldCache = true;
                    }
                }

                auto newcalled = CreateAugmentedPrimal(dyn_cast<Function>(called), global_AA, subconstant_args, TLI, /*differentialReturn*/subdifferentialreturn, /*return is used*/subretused, uncacheable_args_map[op]).first;
                auto augmentcall = BuilderZ.CreateCall(newcalled, args);
                assert(augmentcall->getType()->isStructTy());
                augmentcall->setCallingConv(op->getCallingConv());
                augmentcall->setDebugLoc(inst->getDebugLoc());
                
                gutils->originalInstructions.insert(augmentcall);
                gutils->nonconstant.insert(augmentcall);

                Value* tp = BuilderZ.CreateExtractValue(augmentcall, {0}, "subcache");
                if (tp->getType()->isEmptyTy()) {
                    auto tpt = tp->getType();
                    gutils->erase(cast<Instruction>(tp));
                    tp = UndefValue::get(tpt);
                }
                
                gutils->addMalloc(BuilderZ, tp);

                if (subretused) {
                  auto rv = cast<Instruction>(BuilderZ.CreateExtractValue(augmentcall, {1}));
                  gutils->originalInstructions.insert(rv);
                  gutils->nonconstant.insert(rv);
                  if (!gutils->isConstantValue(op)) {
                    gutils->nonconstant_values.insert(rv);
                  }
                  assert(op->getType() == rv->getType());
                  
                  if (shouldCache) {
                    gutils->addMalloc(BuilderZ, rv);
                  }

                  if ((op->getType()->isPointerTy() || op->getType()->isIntegerTy()) && gutils->invertedPointers.count(op) != 0) {
                    auto placeholder = cast<PHINode>(gutils->invertedPointers[op]);
                    if (I != E && placeholder == &*I) I++;
                    gutils->invertedPointers.erase(op);
                    if (subdifferentialreturn) {
                      assert(cast<StructType>(augmentcall->getType())->getNumElements() == 3);
                      auto antiptr = cast<Instruction>(BuilderZ.CreateExtractValue(augmentcall, {2}, "antiptr_" + op->getName() ));
                      gutils->invertedPointers[rv] = antiptr;
                      placeholder->replaceAllUsesWith(antiptr);

                      if (shouldCache) {
                          gutils->addMalloc(BuilderZ, antiptr);
                      }
                    }
                    gutils->erase(placeholder);
                  } else {
                    if (cast<StructType>(augmentcall->getType())->getNumElements() != 2) {
                        llvm::errs() << "old called: " << *called << "\n";
                        llvm::errs() << "augmented called: " << *augmentcall << "\n";
                        llvm::errs() << "op type: " << *op->getType() << "\n";
                        llvm::errs() << "op subdifferentialreturn: " << subdifferentialreturn << "\n";
                    }
                    assert(cast<StructType>(augmentcall->getType())->getNumElements() == 2);
                    
                  }

                  gutils->replaceAWithB(op,rv);
                } else {
                  if ((op->getType()->isPointerTy() || op->getType()->isIntegerTy()) && gutils->invertedPointers.count(op) != 0) {
                    auto placeholder = cast<PHINode>(gutils->invertedPointers[op]);
                    if (I != E && placeholder == &*I) I++;
                    gutils->invertedPointers.erase(op);
                    gutils->erase(placeholder);
                  }

                }

                gutils->erase(op);
        } else if(LoadInst* li = dyn_cast<LoadInst>(inst)) {
          if (gutils->isConstantInstruction(inst) || gutils->isConstantValue(inst)) continue;
          if (can_modref_map[inst]) {
            llvm::errs() << "Forcibly caching reads " << *li << "\n"; 
            IRBuilder<> BuilderZ(li);
            gutils->addMalloc(BuilderZ, li);
          }

           //TODO IF OP IS POINTER
        } else if(auto op = dyn_cast<StoreInst>(inst)) {
          if (gutils->isConstantInstruction(inst)) continue;

          if ( op->getValueOperand()->getType()->isPointerTy() || (op->getValueOperand()->getType()->isIntegerTy() && !isIntASecretFloat(op->getValueOperand()) ) ) {
            IRBuilder <> storeBuilder(op);
            //llvm::errs() << "a op value: " << *op->getValueOperand() << "\n";
            Value* valueop = gutils->invertPointerM(op->getValueOperand(), storeBuilder);
            //llvm::errs() << "a op pointer: " << *op->getPointerOperand() << "\n";
            Value* pointerop = gutils->invertPointerM(op->getPointerOperand(), storeBuilder);
            storeBuilder.CreateStore(valueop, pointerop);
          }
        }
     }
  }

  auto nf = gutils->newFunc;

  ValueToValueMapTy invertedRetPs;
  if ((gutils->oldFunc->getReturnType()->isPointerTy() || gutils->oldFunc->getReturnType()->isIntegerTy()) && differentialReturn) {
    for (inst_iterator I = inst_begin(nf), E = inst_end(nf); I != E; ++I) {
      if (ReturnInst* ri = dyn_cast<ReturnInst>(&*I)) {
        assert(ri->getReturnValue());
        IRBuilder <>builder(ri);
        Value* toinvert = nullptr;
        if (auto iv = dyn_cast<InsertValueInst>(ri->getReturnValue())) {
            if (iv->getNumIndices() == 1 && iv->getIndices()[0] == 1) {
                toinvert = iv->getInsertedValueOperand();
            }
        }
        if (toinvert == nullptr) {
            toinvert = builder.CreateExtractValue(ri->getReturnValue(), {1});
        }
        if (!gutils->isConstantValue(toinvert)) {
            invertedRetPs[ri] = gutils->invertPointerM(toinvert, builder);
        } else {
            invertedRetPs[ri] = UndefValue::get(toinvert->getType());
        } 
        assert(invertedRetPs[ri]);
      }
    }
  }

  while(gutils->inversionAllocs->size() > 0) {
    gutils->inversionAllocs->back().moveBefore(gutils->newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  }

  (IRBuilder <>(gutils->inversionAllocs)).CreateUnreachable();
  DeleteDeadBlock(gutils->inversionAllocs);

  for (Argument &Arg : gutils->newFunc->args()) {
      if (Arg.hasAttribute(Attribute::Returned))
          Arg.removeAttr(Attribute::Returned);
      if (Arg.hasAttribute(Attribute::StructRet))
          Arg.removeAttr(Attribute::StructRet);
  }

  if (gutils->newFunc->hasFnAttribute(Attribute::OptimizeNone))
    gutils->newFunc->removeFnAttr(Attribute::OptimizeNone);

  if (auto bytes = gutils->newFunc->getDereferenceableBytes(llvm::AttributeList::ReturnIndex)) {
    AttrBuilder ab;
    ab.addDereferenceableAttr(bytes);
    gutils->newFunc->removeAttributes(llvm::AttributeList::ReturnIndex, ab);
  }

  if (gutils->newFunc->hasAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias)) {
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias);
  }
  if (gutils->newFunc->hasAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::ZExt)) {
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::ZExt);
  }

  if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
      llvm::errs() << *gutils->oldFunc << "\n";
      llvm::errs() << *gutils->newFunc << "\n";
      report_fatal_error("function failed verification (2)");
  }

  std::vector<Type*> RetTypes;

  std::vector<Type*> MallocTypes;

  for(auto a:gutils->getMallocs()) {
      MallocTypes.push_back(a->getType());
  }

  StructType* tapeType;

  //if (MallocTypes.size() > 1) {
  //  tapeType = StructType::create(MallocTypes, (todiff->getName()+"_tapeType").str(), false);
  //} else {
    tapeType = StructType::get(nf->getContext(), MallocTypes);
  //}

  bool recursive = cachedfunctions[tup].first->getNumUses() > 0;

  if (recursive) {
    RetTypes.push_back(Type::getInt8PtrTy(nf->getContext()));
  } else {
    RetTypes.push_back(tapeType);
  }

  if (returnUsed) {
    RetTypes.push_back(gutils->oldFunc->getReturnType());
    if ( (gutils->oldFunc->getReturnType()->isPointerTy() || gutils->oldFunc->getReturnType()->isIntegerTy()) && differentialReturn )
      RetTypes.push_back(gutils->oldFunc->getReturnType());
  }

  Type* RetType = StructType::get(nf->getContext(), RetTypes);

 ValueToValueMapTy VMap;
 std::vector<Type*> ArgTypes;
 for (const Argument &I : nf->args()) {
     ArgTypes.push_back(I.getType());
 }

 // Create a new function type...
 FunctionType *FTy = FunctionType::get(RetType,
                                   ArgTypes, nf->getFunctionType()->isVarArg());

 // Create the new function...
 Function *NewF = Function::Create(FTy, nf->getLinkage(), "augmented_"+todiff->getName(), nf->getParent());

 unsigned ii = 0, jj = 0;
 for (auto i=nf->arg_begin(), j=NewF->arg_begin(); i != nf->arg_end(); ) {
    VMap[i] = j;
    if (nf->hasParamAttribute(ii, Attribute::NoCapture)) {
      NewF->addParamAttr(jj, Attribute::NoCapture);
    }
    if (nf->hasParamAttribute(ii, Attribute::NoAlias)) {
       NewF->addParamAttr(jj, Attribute::NoAlias);
    }

     j->setName(i->getName());
     j++;
     jj++;

     i++;
     ii++;
 }

  SmallVector <ReturnInst*,4> Returns;
  CloneFunctionInto(NewF, nf, VMap, nf->getSubprogram() != nullptr, Returns, "",
                   nullptr);

  IRBuilder<> ib(NewF->getEntryBlock().getFirstNonPHI());

  Value* ret = ib.CreateAlloca(RetType);

  Value* tapeMemory;
  if (recursive) {
      auto i64 = Type::getInt64Ty(NewF->getContext());
      tapeMemory = CallInst::CreateMalloc(NewF->getEntryBlock().getFirstNonPHI(),
                    i64,
                    tapeType,
                    ConstantInt::get(i64, NewF->getParent()->getDataLayout().getTypeAllocSizeInBits(tapeType)/8),
                    nullptr,
                    nullptr,
                    "tapemem"
                    );
            CallInst* malloccall = dyn_cast<CallInst>(tapeMemory);
            if (malloccall == nullptr) {
                malloccall = cast<CallInst>(cast<Instruction>(tapeMemory)->getOperand(0));
            }
            malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
            malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
    Value *Idxs[] = {
        ib.getInt32(0),
        ib.getInt32(0),
    };
    ib.CreateStore(malloccall, ib.CreateGEP(ret, Idxs, ""));
  } else {
    Value *Idxs[] = {
        ib.getInt32(0),
        ib.getInt32(0),
    };
    tapeMemory = ib.CreateGEP(ret, Idxs, "");
  }

  unsigned i=0;
  for (auto v: gutils->getMallocs()) {
      if (!isa<UndefValue>(v)) {
          IRBuilder <>ib(cast<Instruction>(VMap[v])->getNextNode());
          Value *Idxs[] = {
            ib.getInt32(0),
            ib.getInt32(i)
          };
          auto gep = ib.CreateGEP(tapeMemory, Idxs, "");
          ib.CreateStore(VMap[v], gep);
      }
      i++;
  }
  if (tapeMemory->hasNUses(0)) gutils->erase(cast<Instruction>(tapeMemory));

  for (inst_iterator I = inst_begin(nf), E = inst_end(nf); I != E; ++I) {
      if (ReturnInst* ri = dyn_cast<ReturnInst>(&*I)) {
          ReturnInst* rim = cast<ReturnInst>(VMap[ri]);
          Type* oldretTy = gutils->oldFunc->getReturnType();
          IRBuilder <>ib(rim);
          if (returnUsed) {
            Value* rv = rim->getReturnValue();
            assert(rv);
            Value* actualrv = nullptr;
            if (auto iv = dyn_cast<InsertValueInst>(rv)) {
              if (iv->getNumIndices() == 1 && iv->getIndices()[0] == 1) {
                actualrv = iv->getInsertedValueOperand();
              }
            }
            if (actualrv == nullptr) {
              actualrv = ib.CreateExtractValue(rv, {1});
            }

            ib.CreateStore(actualrv, ib.CreateConstGEP2_32(RetType, ret, 0, 1, ""));

            if ((oldretTy->isPointerTy() || oldretTy->isIntegerTy()) && differentialReturn) {
              assert(invertedRetPs[ri]);
              if (!isa<UndefValue>(invertedRetPs[ri])) {
                assert(VMap[invertedRetPs[ri]]);
                ib.CreateStore( VMap[invertedRetPs[ri]], ib.CreateConstGEP2_32(RetType, ret, 0, 2, ""));
              }
            }
            i++;
          }
          ib.CreateRet(ib.CreateLoad(ret));
          gutils->erase(cast<Instruction>(VMap[ri]));
      }
  }

  for (Argument &Arg : NewF->args()) {
      if (Arg.hasAttribute(Attribute::Returned))
          Arg.removeAttr(Attribute::Returned);
      if (Arg.hasAttribute(Attribute::StructRet))
          Arg.removeAttr(Attribute::StructRet);
  }
  if (NewF->hasFnAttribute(Attribute::OptimizeNone))
    NewF->removeFnAttr(Attribute::OptimizeNone);

  if (auto bytes = NewF->getDereferenceableBytes(llvm::AttributeList::ReturnIndex)) {
    AttrBuilder ab;
    ab.addDereferenceableAttr(bytes);
    NewF->removeAttributes(llvm::AttributeList::ReturnIndex, ab);
  }
  if (NewF->hasAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias)) {
    NewF->removeAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias);
  }
  if (NewF->hasAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::ZExt)) {
    NewF->removeAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::ZExt);
  }

  if (llvm::verifyFunction(*NewF, &llvm::errs())) {
      llvm::errs() << *gutils->oldFunc << "\n";
      llvm::errs() << *NewF << "\n";
      report_fatal_error("augmented function failed verification (3)");
  }

  SmallVector<User*,4> fnusers;
  for(auto user : cachedfunctions[tup].first->users()) {
    fnusers.push_back(user);
  }
  for(auto user : fnusers) {
    cast<CallInst>(user)->setCalledFunction(NewF);
  }
  cachedfunctions[tup].first = NewF;
  if (recursive)
      cachedfunctions[tup].second = tapeType;
  cachedfinished[tup] = true;

  gutils->newFunc->eraseFromParent();

  delete gutils;
  if (enzyme_print)
    llvm::errs() << *NewF << "\n";
  return std::pair<Function*,StructType*>(NewF, recursive ? tapeType : nullptr);
}

void createInvertedTerminator(DiffeGradientUtils* gutils, BasicBlock *BB, AllocaInst* retAlloca, unsigned extraArgs) {
    LoopContext loopContext;
    bool inLoop = gutils->getContext(BB, loopContext);
    BasicBlock* BB2 = gutils->reverseBlocks[BB];
    assert(BB2);
    IRBuilder<> Builder(BB2);
    Builder.setFastMathFlags(getFast());

    std::map<BasicBlock*, std::vector<BasicBlock*>> targetToPreds;
    for(auto pred : predecessors(BB)) {
        targetToPreds[gutils->getReverseOrLatchMerge(pred, BB)].emplace_back(pred);
    }

    if (targetToPreds.size() == 0) {
        SmallVector<Value *,4> retargs;

        if (retAlloca) {
          auto result = Builder.CreateLoad(retAlloca, "retreload");
          result->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(retAlloca->getContext(), {}));
          assert(gutils->isConstantInstruction(result));
          retargs.push_back(result);
        }

        auto endidx = gutils->newFunc->arg_end();
        for(unsigned i=0; i<extraArgs; i++)
            endidx--;

        for (auto& I: gutils->newFunc->args()) {
          if (&I == endidx) {
              break;
          }
          if (!gutils->isConstantValue(&I) && whatType(I.getType()) == DIFFE_TYPE::OUT_DIFF ) {
            retargs.push_back(gutils->diffe((Value*)&I, Builder));
          }
        }

        Value* toret = UndefValue::get(gutils->newFunc->getReturnType());
        for(unsigned i=0; i<retargs.size(); i++) {
          unsigned idx[] = { i };
          toret = Builder.CreateInsertValue(toret, retargs[i], idx);
        }
        Builder.CreateRet(toret);
        return;
    }

    //PHINodes to replace that will contain true iff the predecessor was given basicblock
    std::map<BasicBlock*, PHINode*> replacePHIs;
    std::vector<SelectInst*> selects;

    IRBuilder <>phibuilder(BB2);
    bool setphi = false;

    // Ensure phi values have their derivatives propagated
    for (auto I = BB->begin(), E = BB->end(); I != E; I++) {
        if(PHINode* PN = dyn_cast<PHINode>(&*I)) {
            if (gutils->isConstantValue(PN)) continue;
            if (PN->getType()->isPointerTy() || PN->getType()->isIntegerTy()) continue;

            auto prediff = gutils->diffe(PN, Builder);
            gutils->setDiffe(PN, Constant::getNullValue(PN->getType()), Builder);
            
            for (BasicBlock* pred : predecessors(BB)) {
                if (gutils->isConstantValue(PN->getIncomingValueForBlock(pred))) continue;

                if (PN->getNumIncomingValues() == 1) {
                    gutils->addToDiffe(PN->getIncomingValueForBlock(pred), prediff, Builder);
                } else {
                    if (replacePHIs.find(pred) == replacePHIs.end()) {
                        replacePHIs[pred] = Builder.CreatePHI(Type::getInt1Ty(pred->getContext()), 1);   
                        if (!setphi) {
                            phibuilder.SetInsertPoint(replacePHIs[pred]);
                            setphi = true;
                        }
                    } 
                    SelectInst* dif = cast<SelectInst>(Builder.CreateSelect(replacePHIs[pred], prediff, Constant::getNullValue(prediff->getType())));
                    auto addedSelects = gutils->addToDiffe(PN->getIncomingValueForBlock(pred), dif, Builder);
                    if (dif->getNumUses() != 0) {
                      llvm::errs() << "oldFunc: " << *gutils->oldFunc << "\n";
                      llvm::errs() << "newFunc: " << *gutils->newFunc << "\n";
                      for (auto use : dif->users()) {
                        llvm::errs() << "user: " << *use << "\n";
                      }
                      llvm::errs() << "dif: " << *dif << "\n";
                    }
                    assert(dif->getNumUses() == 0);
                    dif->eraseFromParent();
                    for (auto select : addedSelects)
                        selects.emplace_back(select);
                }
            }
        } else break;
    }
    if (!setphi) {
        phibuilder.SetInsertPoint(Builder.GetInsertBlock(), Builder.GetInsertPoint());
    }
    
    if (inLoop && BB == loopContext.header) {
        std::map<BasicBlock*, std::vector<BasicBlock*>> targetToPreds;
        for(auto pred : predecessors(BB)) {
            if (pred == loopContext.preheader) continue;
            targetToPreds[gutils->getReverseOrLatchMerge(pred, BB)].emplace_back(pred);
        }

        assert(targetToPreds.size() && "only loops with one backedge are presently supported");

        Value* phi = phibuilder.CreateICmpEQ(loopContext.antivar, Constant::getNullValue(loopContext.antivar->getType()));

        for (auto pair : replacePHIs) {
            Value* replaceWith = nullptr;

            if (pair.first == loopContext.preheader) {
                replaceWith = phi;
            } else {
                replaceWith = phibuilder.CreateNot(phi);
            }

            pair.second->replaceAllUsesWith(replaceWith);
            pair.second->eraseFromParent();
        }

        Builder.SetInsertPoint(BB2);
        
        Builder.CreateCondBr(phi, gutils->getReverseOrLatchMerge(loopContext.preheader, BB), targetToPreds.begin()->first);

    } else {
        std::map<BasicBlock*, std::vector< std::pair<BasicBlock*, BasicBlock*> >> phiTargetToPreds;
        for (auto pair : replacePHIs) {
            phiTargetToPreds[pair.first].emplace_back(std::make_pair(pair.first, BB));
        }
        BasicBlock* fakeTarget = nullptr;
        for (auto pred : predecessors(BB)) {
            if (phiTargetToPreds.find(pred) != phiTargetToPreds.end()) continue;
            if (fakeTarget == nullptr) fakeTarget = pred;
            phiTargetToPreds[fakeTarget].emplace_back(std::make_pair(pred, BB));
        }
        gutils->branchToCorrespondingTarget(BB, phibuilder, phiTargetToPreds, &replacePHIs);

        std::map<BasicBlock*, std::vector< std::pair<BasicBlock*, BasicBlock*> >> targetToPreds;
        for(auto pred : predecessors(BB)) {
            targetToPreds[gutils->getReverseOrLatchMerge(pred, BB)].emplace_back(std::make_pair(pred, BB));
        }
        Builder.SetInsertPoint(BB2);
        gutils->branchToCorrespondingTarget(BB, Builder, targetToPreds);
    }

    // Optimize select of not to just be a select with operands switched
    for (SelectInst* select : selects) {
        if (BinaryOperator* bo = dyn_cast<BinaryOperator>(select->getCondition())) {
            if (bo->getOpcode() == BinaryOperator::Xor) {
                llvm::errs() << " considering " << *select << " " << *bo << "\n";
                if (isa<ConstantInt>(bo->getOperand(0)) && cast<ConstantInt>(bo->getOperand(0))->isOne()) {
                    select->setCondition(bo->getOperand(1));
                    auto tmp = select->getTrueValue();
                    select->setTrueValue(select->getFalseValue());
                    select->setFalseValue(tmp);
                    if (bo->getNumUses() == 0) bo->eraseFromParent();
                } else if (isa<ConstantInt>(bo->getOperand(1)) && cast<ConstantInt>(bo->getOperand(1))->isOne()) {
                    select->setCondition(bo->getOperand(0));
                    auto tmp = select->getTrueValue();
                    select->setTrueValue(select->getFalseValue());
                    select->setFalseValue(tmp);
                    if (bo->getNumUses() == 0) bo->eraseFromParent();
                }
            }
        }
    }
}

//! assuming not top level
std::pair<SmallVector<Type*,4>,SmallVector<Type*,4>> getDefaultFunctionTypeForGradient(FunctionType* called, bool returnUsed, bool differentialReturn) {
    SmallVector<Type*, 4> args;
    SmallVector<Type*, 4> outs;
    for(auto &argType : called->params()) {
        args.push_back(argType);

        if ( argType->isPointerTy() || argType->isIntegerTy() ) {
            args.push_back(argType);
        } else {
            outs.push_back(argType);
        }
    }

    auto ret = called->getReturnType();
    if (returnUsed) {
        if (differentialReturn) {
            args.push_back(ret);
        }
    }

    return std::pair<SmallVector<Type*,4>,SmallVector<Type*,4>>(args, outs);
}

void handleGradientCallInst(BasicBlock::reverse_iterator &I, const BasicBlock::reverse_iterator &E, IRBuilder <>& Builder2, CallInst* op, DiffeGradientUtils* const gutils, TargetLibraryInfo &TLI, AAResults &AA, AAResults & global_AA, const bool topLevel, const std::map<ReturnInst*,StoreInst*> &replacedReturns, std::set<unsigned> uncacheable_args) {
  llvm::errs() << "HandleGradientCall " << *op << "\n";
  Function *called = op->getCalledFunction();

  if (auto castinst = dyn_cast<ConstantExpr>(op->getCalledValue())) {
    if (castinst->isCast()) {
      if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
        if (isAllocationFunction(*fn, TLI) || isDeallocationFunction(*fn, TLI)) {
          called = fn;
        }
      }
    }
  }

  if (called && (called->getName() == "printf" || called->getName() == "puts")) {
    SmallVector<Value*, 4> args;
    for(unsigned i=0; i<op->getNumArgOperands(); i++) {
      args.push_back(gutils->lookupM(op->getArgOperand(i), Builder2));
    }
    CallInst* cal = Builder2.CreateCall(called, args);
    cal->setAttributes(op->getAttributes());
    cal->setCallingConv(op->getCallingConv());
    cal->setTailCallKind(op->getTailCallKind());
    return;
  }

  if (called && isAllocationFunction(*called, TLI)) {
    if (!gutils->isConstantValue(op)) {
      PHINode* placeholder = cast<PHINode>(gutils->invertedPointers[op]);
      auto anti = gutils->createAntiMalloc(op);
      if (I != E && placeholder == &*I) I++;
      freeKnownAllocation(Builder2, gutils->lookupM(anti, Builder2), *called, TLI)->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
    }

    //TODO enable this if we need to free the memory
    // NOTE THAT TOPLEVEL IS THERE SIMPLY BECAUSE THAT WAS PREVIOUS ATTITUTE TO FREE'ing
    if (topLevel) {
      Value* inst = op;
      if (!topLevel && op->getNumUses() != 0) {
        IRBuilder<> BuilderZ(op);
        inst = gutils->addMalloc(BuilderZ, op);
      }
      freeKnownAllocation(Builder2, gutils->lookupM(inst, Builder2), *called, TLI);
    }
    return;
  }

  if (called && called->getName()=="free") {
    llvm::Value* val = op->getArgOperand(0);

    while(auto cast = dyn_cast<CastInst>(val)) val = cast->getOperand(0);
    
    if (auto dc = dyn_cast<CallInst>(val)) {
      if (dc->getCalledFunction()->getName() == "malloc") {
        gutils->erase(op);
        return;
      }
    }

    if (isa<ConstantPointerNull>(val)) {
      gutils->erase(op);
      llvm::errs() << "removing free of null pointer\n";
      return;
    }

    //TODO HANDLE FREE
    llvm::errs() << "freeing without malloc " << *val << "\n";
    gutils->erase(op);
    return;
  }

  if (called && (called->getName()=="_ZdlPv" || called->getName()=="_ZdlPvm")) {
    llvm::Value* val = op->getArgOperand(0);
    while(auto cast = dyn_cast<CastInst>(val)) val = cast->getOperand(0);
    if (auto dc = dyn_cast<CallInst>(val)) {
      if (dc->getCalledFunction()->getName() == "_Znwm") {
        gutils->erase(op);
        return;
      }
    }
    //TODO HANDLE DELETE
    llvm::errs() << "deleting without new " << *val << "\n";
    gutils->erase(op);
    return;
  }

  if (gutils->isConstantInstruction(op)) {
    if (!topLevel && op->getNumUses() != 0 && !op->doesNotAccessMemory()) {
      IRBuilder<> BuilderZ(op);
      gutils->addMalloc(BuilderZ, op);
    }
    return;
  }

  bool modifyPrimal = false;
  bool foreignFunction = false;

  if (called && !called->hasFnAttribute(Attribute::ReadNone)) {
    modifyPrimal = true;
  }

  if (called == nullptr || called->empty()) {
    foreignFunction = true;
    modifyPrimal = true;
  }

  std::set<unsigned> subconstant_args;

  SmallVector<Value*, 8> args;
  SmallVector<Value*, 8> pre_args;
  SmallVector<DIFFE_TYPE, 8> argsInverted;
  IRBuilder<> BuilderZ(op);
  std::vector<Instruction*> postCreate;
  BuilderZ.setFastMathFlags(getFast());

  if ( (op->getType()->isPointerTy() || op->getType()->isIntegerTy()) && !gutils->isConstantValue(op)) {
    //llvm::errs() << "augmented modified " << called->getName() << " modified via return" << "\n";
    modifyPrimal = true;
  }

  for(unsigned i=0;i<op->getNumArgOperands(); i++) {
    args.push_back(gutils->lookupM(op->getArgOperand(i), Builder2));
    pre_args.push_back(op->getArgOperand(i));

    if (gutils->isConstantValue(op->getArgOperand(i)) && !foreignFunction) {
      subconstant_args.insert(i);
      argsInverted.push_back(DIFFE_TYPE::CONSTANT);
      continue;
    }

    auto argType = op->getArgOperand(i)->getType();

    if ( (argType->isPointerTy() || argType->isIntegerTy()) && !gutils->isConstantValue(op->getArgOperand(i)) ) {
      argsInverted.push_back(DIFFE_TYPE::DUP_ARG);
      args.push_back(gutils->invertPointerM(op->getArgOperand(i), Builder2));
      pre_args.push_back(gutils->invertPointerM(op->getArgOperand(i), BuilderZ));

                //TODO this check should consider whether this pointer has allocation/etc modifications and so on
      if (called && ! ( called->hasParamAttribute(i, Attribute::ReadOnly) || called->hasParamAttribute(i, Attribute::ReadNone)) ) {
                    //llvm::errs() << "augmented modified " << called->getName() << " modified via arg " << i << "\n";
        modifyPrimal = true;
      }

                //Note sometimes whattype mistakenly says something should be constant [because composed of integer pointers alone]
      assert(whatType(argType) == DIFFE_TYPE::DUP_ARG || whatType(argType) == DIFFE_TYPE::CONSTANT);
    } else {
      argsInverted.push_back(DIFFE_TYPE::OUT_DIFF);
      assert(whatType(argType) == DIFFE_TYPE::OUT_DIFF || whatType(argType) == DIFFE_TYPE::CONSTANT);
    }
  }

  bool replaceFunction = false;

  if (topLevel && op->getParent()->getSingleSuccessor() == gutils->reverseBlocks[op->getParent()] && !foreignFunction) {
      auto origop = cast<CallInst>(gutils->getOriginal(op));
      auto OBB = cast<BasicBlock>(gutils->getOriginal(op->getParent()));
                //TODO fix this to be more accurate
      auto iter = OBB->rbegin();
      SmallPtrSet<Instruction*,4> usetree;
      usetree.insert(origop);
      for(auto uinst = origop->getNextNode(); uinst != nullptr; uinst = uinst->getNextNode()) {
        bool usesInst = false;
        for(auto &operand : uinst->operands()) {
          if (auto usedinst = dyn_cast<Instruction>(operand.get())) {
            if (usetree.find(usedinst) != usetree.end()) {
              usesInst = true;
              break;
            }
          }
        }
        if (usesInst) {
          usetree.insert(uinst);
        }

      }

      while(iter != OBB->rend() && &*iter != origop) {
        if (auto call = dyn_cast<CallInst>(&*iter)) {
          if (isCertainMallocOrFree(call->getCalledFunction())) {
            iter++;
            continue;
          }
        }

        if (auto ri = dyn_cast<ReturnInst>(&*iter)) {
          auto fd = replacedReturns.find(ri);
          if (fd != replacedReturns.end()) {
            auto si = fd->second;
            if (usetree.find(dyn_cast<Instruction>(gutils->getOriginal(si->getValueOperand()))) != usetree.end()) {
              postCreate.push_back(si);
            }
          }
          iter++;
          continue;
        }

                  //TODO usesInst has created a bug here
                  // if it is used by the reverse pass before this call creates it
                  // and thus it loads an undef from cache
        bool usesInst = usetree.find(&*iter) != usetree.end();

                  //TODO remove this upon start of more accurate)
                  //if (usesInst) break;

        if (!usesInst && (!iter->mayReadOrWriteMemory() || isa<BinaryOperator>(&*iter))) {
          iter++;
          continue;
        }

        ModRefInfo mri = ModRefInfo::NoModRef;
        if (iter->mayReadOrWriteMemory()) {
          llvm::errs() << "Iter is at " << *iter << "\n";
          llvm::errs() << "origop is at " << *origop << "\n";
          mri = AA.getModRefInfo(&*iter, origop);
        }

        if (mri == ModRefInfo::NoModRef && !usesInst) {
          iter++;
          continue;
        }

                  //load that follows the original
        if (auto li = dyn_cast<LoadInst>(&*iter)) {
          bool modref = false;
          for(Instruction* it = li; it != nullptr; it = it->getNextNode()) {
            if (auto call = dyn_cast<CallInst>(it)) {
             if (isCertainMallocOrFree(call->getCalledFunction())) {
               continue;
             }
           }
           if (AA.canInstructionRangeModRef(*it, *it, MemoryLocation::get(li), ModRefInfo::Mod)) {
            modref = true;
            llvm::errs() << " inst  found mod " << *iter << " " << *it << "\n";
          }
        }

        if (modref)
          break;
        postCreate.push_back(cast<Instruction>(gutils->getNewFromOriginal(&*iter)));
        iter++;
        continue;
      }

                  //call that follows the original
      if (auto li = dyn_cast<IntrinsicInst>(&*iter)) {
        if (li->getIntrinsicID() == Intrinsic::memcpy) {
         auto mem0 = AA.getModRefInfo(&*iter, li->getOperand(0), MemoryLocation::UnknownSize);
         auto mem1 = AA.getModRefInfo(&*iter, li->getOperand(1), MemoryLocation::UnknownSize);

         llvm::errs() << "modrefinfo for mem0 " << *li->getOperand(0) << " " << (unsigned int)mem0 << "\n";
         llvm::errs() << "modrefinfo for mem1 " << *li->getOperand(1) << " " << (unsigned int)mem1 << "\n";
                      //TODO TEMPORARILY REMOVED THIS TO TEST SOMETHING
                      // PUT BACK THIS CONDITION!!
                      //if ( !isModSet(mem0) )
         {
          bool modref = false;
          for(Instruction* it = li; it != nullptr; it = it->getNextNode()) {
            if (auto call = dyn_cast<CallInst>(it)) {
             if (isCertainMallocOrFree(call->getCalledFunction())) {
               continue;
             }
           }
           if (AA.canInstructionRangeModRef(*it, *it, li->getOperand(1), MemoryLocation::UnknownSize, ModRefInfo::Mod)) {
            modref = true;
            llvm::errs() << " inst  found mod " << *iter << " " << *it << "\n";
          }
        }

        if (modref)
          break;
        postCreate.push_back(cast<Instruction>(gutils->getNewFromOriginal(&*iter)));
        iter++;
        continue;
      }
    }
  }

  if (usesInst) {
    bool modref = false;
                      //auto start = &*iter;
    if (mri != ModRefInfo::NoModRef) {
      modref = true;
                          /*
                          for(Instruction* it = start; it != nullptr; it = it->getNextNode()) {
                              if (auto call = dyn_cast<CallInst>(it)) {
                                   if (isCertainMallocOrFree(call->getCalledFunction())) {
                                       continue;
                                   }
                              }
                              if ( (isa<StoreInst>(start) || isa<CallInst>(start)) && AA.canInstructionRangeModRef(*it, *it, MemoryLocation::get(start), ModRefInfo::Mod)) {
                                  modref = true;
                          llvm::errs() << " inst  found mod " << *iter << " " << *it << "\n";
                              }
                          }
                          */
    }

    if (modref)
      break;
    postCreate.push_back(cast<Instruction>(gutils->getNewFromOriginal(&*iter)));
    iter++;
    continue;
  }

  break;
  }
  if (&*iter == gutils->getOriginal(op)) {
    bool outsideuse = false;
    for(auto user : op->users()) {
      if (gutils->originalInstructions.find(cast<Instruction>(user)) == gutils->originalInstructions.end()) {
        outsideuse = true;
      }
    }

    if (!outsideuse) {
      if (called)
        llvm::errs() << " choosing to replace function " << (called->getName()) << " and do both forward/reverse\n";
      else
        llvm::errs() << " choosing to replace function " << (*op->getCalledValue()) << " and do both forward/reverse\n";
      replaceFunction = true;
      modifyPrimal = false;
    } else {
      if (called)
        llvm::errs() << " failed to replace function (cacheuse)" << (called->getName()) << " due to " << *iter << "\n";
      else
        llvm::errs() << " failed to replace function (cacheuse)" << (*op->getCalledValue()) << " due to " << *iter << "\n";

    }
  } else {
    if (called)
      llvm::errs() << " failed to replace function " << (called->getName()) << " due to " << *iter << "\n";
    else
      llvm::errs() << " failed to replace function " << (*op->getCalledValue()) << " due to " << *iter << "\n";
  }
  }

  Value* tape = nullptr;
  CallInst* augmentcall = nullptr;
  Value* cachereplace = nullptr;

  //TODO consider what to do if called == nullptr for augmentation
  if (modifyPrimal && called) {
    bool subretused = op->getNumUses() != 0;
    bool subdifferentialreturn = (!gutils->isConstantValue(op)) && subretused;
    auto fnandtapetype = CreateAugmentedPrimal(cast<Function>(called), global_AA, subconstant_args, TLI, /*differentialReturns*/subdifferentialreturn, /*return is used*/subretused, uncacheable_args);
    if (topLevel) {
      Function* newcalled = fnandtapetype.first;
      augmentcall = BuilderZ.CreateCall(newcalled, pre_args);
      augmentcall->setCallingConv(op->getCallingConv());
      augmentcall->setDebugLoc(op->getDebugLoc());

      gutils->originalInstructions.insert(augmentcall);
      gutils->nonconstant.insert(augmentcall);

      tape = BuilderZ.CreateExtractValue(augmentcall, {0});
      if (tape->getType()->isEmptyTy()) {
        auto tt = tape->getType();
        gutils->erase(cast<Instruction>(tape));
        tape = UndefValue::get(tt);
      }

      if( (op->getType()->isPointerTy() || op->getType()->isIntegerTy()) && !gutils->isConstantValue(op) ) {
        auto newip = cast<Instruction>(BuilderZ.CreateExtractValue(augmentcall, {2}));
        auto placeholder = cast<PHINode>(gutils->invertedPointers[op]);
        if (I != E && placeholder == &*I) I++;
        placeholder->replaceAllUsesWith(newip);
        gutils->erase(placeholder);
        gutils->invertedPointers[op] = newip;
      }
    } else {
      tape = gutils->addMalloc(BuilderZ, tape);

      if (!topLevel && op->getNumUses() != 0) {
        cachereplace = BuilderZ.CreatePHI(op->getType(), 1);
        cachereplace = gutils->addMalloc(BuilderZ, cachereplace);
      }

      if( op->getNumUses() != 0 && (op->getType()->isPointerTy() || op->getType()->isIntegerTy()) && subdifferentialreturn ) {
        auto placeholder = cast<PHINode>(gutils->invertedPointers[op]);
        if (I != E && placeholder == &*I) I++;
        auto newip = gutils->addMalloc(BuilderZ, placeholder);
        gutils->invertedPointers[op] = newip;
      }
    }


    if (fnandtapetype.second) {
      auto tapep = BuilderZ.CreatePointerCast(tape, PointerType::getUnqual(fnandtapetype.second));
      auto truetape = BuilderZ.CreateLoad(tapep);

      CallInst* ci = cast<CallInst>(CallInst::CreateFree(tape, &*BuilderZ.GetInsertPoint()));
      ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
      tape = truetape;
    }

    if (!tape->getType()->isStructTy()) {
      llvm::errs() << "gutils->oldFunc: " << *gutils->oldFunc << "\n";
      llvm::errs() << "gutils->newFunc: " << *gutils->newFunc << "\n";
      llvm::errs() << "tape: " << *tape << "\n";
    }
    assert(tape->getType()->isStructTy());

  } else {
    if (!topLevel && op->getNumUses() != 0 && !op->doesNotAccessMemory()) {
      assert(!replaceFunction);
      cachereplace = IRBuilder<>(op).CreatePHI(op->getType(), 1);
      cachereplace = gutils->addMalloc(BuilderZ, cachereplace);
    }
  }

  bool retUsed = replaceFunction && (op->getNumUses() > 0);
  Value* newcalled = nullptr;

  bool subdiffereturn = (!gutils->isConstantValue(op)) && !( op->getType()->isPointerTy() || op->getType()->isIntegerTy() || op->getType()->isEmptyTy() );
  llvm::errs() << "subdifferet:" << subdiffereturn << " " << *op << "\n";
  if (called) {
    newcalled = CreatePrimalAndGradient(cast<Function>(called), subconstant_args, TLI, global_AA, /*returnValue*/retUsed, /*subdiffereturn*/subdiffereturn, /*topLevel*/replaceFunction, tape ? tape->getType() : nullptr, uncacheable_args);//, LI, DT);
  } else {
    newcalled = gutils->invertPointerM(op->getCalledValue(), Builder2);
    auto ft = cast<FunctionType>(cast<PointerType>(op->getCalledValue()->getType())->getElementType());
    auto res = getDefaultFunctionTypeForGradient(ft, /*returnUsed*/!ft->getReturnType()->isVoidTy(), subdiffereturn);
    //TODO Note there is empty tape added here, replace with generic
    //res.first.push_back(StructType::get(newcalled->getContext(), {}));
    newcalled = Builder2.CreatePointerCast(newcalled, PointerType::getUnqual(FunctionType::get(StructType::get(newcalled->getContext(), res.second), res.first, ft->isVarArg())));
  }

  if (subdiffereturn) {
    args.push_back(gutils->diffe(op, Builder2));
  }

  if (tape) {
    args.push_back(gutils->lookupM(tape, Builder2));
  }

  CallInst* diffes = Builder2.CreateCall(newcalled, args);
  diffes->setCallingConv(op->getCallingConv());
  diffes->setDebugLoc(op->getDebugLoc());

  unsigned structidx = retUsed ? 1 : 0;

  for(unsigned i=0;i<op->getNumArgOperands(); i++) {
    if (argsInverted[i] == DIFFE_TYPE::OUT_DIFF) {
      Value* diffeadd = Builder2.CreateExtractValue(diffes, {structidx});
      structidx++;
      gutils->addToDiffe(op->getArgOperand(i), diffeadd, Builder2);
    }
  }

  //TODO this shouldn't matter because this can't use itself, but setting null should be done before other sets but after load of diffe
  if (op->getNumUses() != 0 && !gutils->isConstantValue(op))
    gutils->setDiffe(op, Constant::getNullValue(op->getType()), Builder2);

  gutils->originalInstructions.insert(diffes);
  gutils->nonconstant.insert(diffes);
  if (!gutils->isConstantValue(op))
    gutils->nonconstant_values.insert(diffes);

  if (replaceFunction) {
    ValueToValueMapTy mapp;
    if (op->getNumUses() != 0) {
      auto retval = cast<Instruction>(Builder2.CreateExtractValue(diffes, {0}));
      gutils->originalInstructions.insert(retval);
      gutils->nonconstant.insert(retval);
      if (!gutils->isConstantValue(op))
        gutils->nonconstant_values.insert(retval);
      op->replaceAllUsesWith(retval);
      mapp[op] = retval;
    }
    for (auto &a : *op->getParent()) {
      if (&a != op) {
        mapp[&a] = &a;
      }
    }
    for (auto &a : *gutils->reverseBlocks[op->getParent()]) {
      mapp[&a] = &a;
    }
    std::reverse(postCreate.begin(), postCreate.end());
    for(auto a : postCreate) {
      for(unsigned i=0; i<a->getNumOperands(); i++) {
        a->setOperand(i, gutils->unwrapM(a->getOperand(i), Builder2, mapp, true));
      }
      a->moveBefore(*Builder2.GetInsertBlock(), Builder2.GetInsertPoint());
    }
    gutils->erase(op);
    return;
  }

  if (augmentcall || cachereplace) {

    if (op->getNumUses() > 0) {
      Value* dcall = nullptr;
      if (augmentcall) {
        dcall = BuilderZ.CreateExtractValue(augmentcall, {1});
      }
      if (cachereplace) {
        assert(dcall == nullptr);
        dcall = cachereplace;
      }

      if (Instruction* inst = dyn_cast<Instruction>(dcall))
          gutils->originalInstructions.insert(inst);
      gutils->nonconstant.insert(dcall);
      if (!gutils->isConstantValue(op))
        gutils->nonconstant_values.insert(dcall);

    if (!gutils->isConstantValue(op)) {
      if (op->getType()->isPointerTy() || op->getType()->isIntegerTy()) {
        gutils->invertedPointers[dcall] = gutils->invertedPointers[op];
        gutils->invertedPointers.erase(op);
      } else {
        gutils->differentials[dcall] = gutils->differentials[op];
        gutils->differentials.erase(op);
      }
    }
    op->replaceAllUsesWith(dcall);
  }

  gutils->erase(op);

  if (augmentcall)
    gutils->replaceableCalls.insert(augmentcall);
  } else {
    gutils->replaceableCalls.insert(op);
  }
}

Function* CreatePrimalAndGradient(Function* todiff, const std::set<unsigned>& constant_args, TargetLibraryInfo &TLI, AAResults &global_AA, bool returnValue, bool differentialReturn, bool topLevel, llvm::Type* additionalArg, std::set<unsigned> _uncacheable_args) {
  if (differentialReturn) {
      if(!todiff->getReturnType()->isFPOrFPVectorTy()) {
         llvm::errs() << *todiff << "\n";
      }
      assert(todiff->getReturnType()->isFPOrFPVectorTy());
  }
  if (additionalArg && !additionalArg->isStructTy()) {
      llvm::errs() << *todiff << "\n";
      llvm::errs() << "addl arg: " << *additionalArg << "\n";
  }
  if (additionalArg) assert(additionalArg->isStructTy());
  static std::map<std::tuple<Function*,std::set<unsigned>/*constant_args*/, std::set<unsigned>/*uncacheable_args*/, bool/*retval*/, bool/*differentialReturn*/, bool/*topLevel*/, llvm::Type*>, Function*> cachedfunctions;
  auto tup = std::make_tuple(todiff, std::set<unsigned>(constant_args.begin(), constant_args.end()), std::set<unsigned>(_uncacheable_args.begin(), _uncacheable_args.end()), returnValue, differentialReturn, topLevel, additionalArg);
  if (cachedfunctions.find(tup) != cachedfunctions.end()) {
    return cachedfunctions[tup];
  }




  bool hasTape = false;

  if (constant_args.size() == 0 && !topLevel && !returnValue && hasMetadata(todiff, "enzyme_gradient")) {

      auto md = todiff->getMetadata("enzyme_gradient");
      if (!isa<MDTuple>(md)) {
          llvm::errs() << *todiff << "\n";
          llvm::errs() << *md << "\n";
          report_fatal_error("unknown gradient for noninvertible function -- metadata incorrect");
      }
      auto md2 = cast<MDTuple>(md);
      assert(md2->getNumOperands() == 1);
      auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
      auto foundcalled = cast<Function>(gvemd->getValue());

      auto res = getDefaultFunctionTypeForGradient(todiff->getFunctionType(), /*has return value*/!todiff->getReturnType()->isVoidTy(), differentialReturn);


      if (foundcalled->arg_size() == res.first.size() + 1 /*tape*/) {
        auto lastarg = foundcalled->arg_end();
        lastarg--;
        res.first.push_back(lastarg->getType());
        hasTape = true;
      } else if (foundcalled->arg_size() == res.first.size()) {
        res.first.push_back(StructType::get(todiff->getContext(), {}));
      } else {
          llvm::errs() << "expected args: [";
          for(auto a : res.first) {
                llvm::errs() << *a << " ";
          }
          llvm::errs() << "]\n";
          llvm::errs() << *foundcalled << "\n";
          assert(0 && "bad type for custom gradient");
      }

      auto st = dyn_cast<StructType>(foundcalled->getReturnType());
      bool wrongRet = st == nullptr;
      if (wrongRet || !hasTape) {
        FunctionType *FTy = FunctionType::get(StructType::get(todiff->getContext(), {res.second}), res.first, todiff->getFunctionType()->isVarArg());
        Function *NewF = Function::Create(FTy, Function::LinkageTypes::InternalLinkage, "fixgradient_"+todiff->getName(), todiff->getParent());
        NewF->setAttributes(foundcalled->getAttributes());
        if (NewF->hasFnAttribute(Attribute::NoInline)) {
            NewF->removeFnAttr(Attribute::NoInline);
        }
          for (Argument &Arg : NewF->args()) {
              if (Arg.hasAttribute(Attribute::Returned))
                  Arg.removeAttr(Attribute::Returned);
              if (Arg.hasAttribute(Attribute::StructRet))
                  Arg.removeAttr(Attribute::StructRet);
          }

        BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
        IRBuilder <>bb(BB);
        SmallVector<Value*,4> args;
        for(auto &a : NewF->args()) args.push_back(&a);
        if (!hasTape) {
            args.pop_back();
        }
        llvm::errs() << *NewF << "\n";
        llvm::errs() << *foundcalled << "\n";
        auto cal = bb.CreateCall(foundcalled, args);
        cal->setCallingConv(foundcalled->getCallingConv());
        Value* val = cal;
        if (wrongRet) {
            auto ut = UndefValue::get(NewF->getReturnType());
            if (val->getType()->isEmptyTy() && res.second.size() == 0) {
                val = ut;
            } else if (res.second.size() == 1 && res.second[0] == val->getType()) {
                val = bb.CreateInsertValue(ut, cal, {0u});
            } else {
                llvm::errs() << *foundcalled << "\n";
                assert(0 && "illegal type for reverse");
            }
        }
        bb.CreateRet(val);
        foundcalled = NewF;
      }
      return cachedfunctions[tup] = foundcalled;
  }

  assert(!todiff->empty());
  auto M = todiff->getParent();

  auto& Context = M->getContext();
  AAResults AA(TLI);
  DiffeGradientUtils *gutils = DiffeGradientUtils::CreateFromClone(todiff, AA, TLI, constant_args, returnValue ? ReturnType::ArgsWithReturn : ReturnType::Args, differentialReturn, additionalArg);
  cachedfunctions[tup] = gutils->newFunc;

  std::map<CallInst*, std::set<unsigned> > uncacheable_args_map =
      compute_uncacheable_args_for_callsites(gutils->oldFunc, gutils->DT, TLI, AA, gutils, _uncacheable_args);

  std::map<Instruction*, bool> can_modref_map;
  // NOTE(TFK): Sanity check this decision.
  //   Is it always possibly to recompute the result of loads at top level?
    can_modref_map = compute_uncacheable_load_map(gutils, AA, TLI, _uncacheable_args);
  if (topLevel) {
    for (auto iter = can_modref_map.begin(); iter != can_modref_map.end(); iter++) {
      if (iter->second) {
        bool is_needed = is_load_needed_in_reverse(gutils, AA, iter->first);
        can_modref_map[iter->first] = is_needed;
      }
    }
  }

  // Allow forcing cache reads to be on or off using flags.
  assert(!(cache_reads_always && cache_reads_never) && "Both cache_reads_always and cache_reads_never are true. This doesn't make sense.");
  if (cache_reads_always || cache_reads_never) {
    bool is_needed = cache_reads_always ? true : false;
    for (auto iter = can_modref_map.begin(); iter != can_modref_map.end(); iter++) {
      can_modref_map[iter->first] = is_needed;
    }
  } 

  gutils->can_modref_map = &can_modref_map;

  gutils->forceContexts(true);
  gutils->forceAugmentedReturns();

  Argument* additionalValue = nullptr;
  if (additionalArg) {
    auto v = gutils->newFunc->arg_end();
    v--;
    additionalValue = v;
    if (!additionalValue->getType()->isStructTy()) {
        llvm::errs() << *gutils->oldFunc << "\n";
        llvm::errs() << *gutils->newFunc << "\n";
        llvm::errs() << "el incorrect tape type: " << *additionalValue << "\n";
    }
    assert(additionalValue->getType()->isStructTy());
    gutils->setTape(additionalValue);
  }

  Argument* differetval = nullptr;
  if (differentialReturn) {
    auto endarg = gutils->newFunc->arg_end();
    endarg--;
    if (additionalArg) endarg--;
    differetval = endarg;
    if (differetval->getType() != todiff->getReturnType()) {
        llvm::errs() << *gutils->oldFunc << "\n";
        llvm::errs() << *gutils->newFunc << "\n";
    }
    assert(differetval->getType() == todiff->getReturnType());
  }

  // Explicitly handle all returns first to ensure that return instructions know if they are used or not before
  //   processessing instructions
  std::map<ReturnInst*,StoreInst*> replacedReturns;
  llvm::AllocaInst* retAlloca = nullptr;
  if (returnValue) {
    retAlloca = IRBuilder<>(&gutils->newFunc->getEntryBlock().front()).CreateAlloca(todiff->getReturnType(), nullptr, "toreturn");
  }

  for(BasicBlock* BB: gutils->originalBlocks) {
    if(ReturnInst* op = dyn_cast<ReturnInst>(BB->getTerminator())) {
      Value* retval = op->getReturnValue();
      IRBuilder<> rb(op);
      rb.setFastMathFlags(getFast());

      if (retAlloca) {
        StoreInst* si = rb.CreateStore(retval, retAlloca);
        replacedReturns[cast<ReturnInst>(gutils->getOriginal(op))] = si;
      } else {
        /*
         TODO should do DCE ideally
        if (auto inst = dyn_cast<Instruction>(retval)) {
            SmallSetVector<Instruction *, 16> WorkList;
            DCEInstruction(inst, WorkList, TLI);
            while (!WorkList.empty()) {
                Instruction *I = WorkList.pop_back_val();
                MadeChange |= DCEInstruction(I, WorkList, TLI);
            }
        }
        */
      }

      if (differentialReturn && !gutils->isConstantValue(retval)) {
        IRBuilder <>reverseB(gutils->reverseBlocks[BB]);
        gutils->setDiffe(retval, differetval, reverseB);
      } else {
        assert (retAlloca == nullptr);
      }
      rb.CreateBr(gutils->reverseBlocks[BB]);
      gutils->erase(op);
    }
  }

  for(BasicBlock* BB: gutils->originalBlocks) {
    auto BB2 = gutils->reverseBlocks[BB];
    assert(BB2);

    IRBuilder<> Builder2(BB2);
    //if (BB2->size() > 0) {
    //    Builder2.SetInsertPoint(BB2->getFirstNonPHI());
    //}
    Builder2.setFastMathFlags(getFast());

    std::function<Value*(Value*)> lookup = [&](Value* val) -> Value* {
      return gutils->lookupM(val, Builder2);
    };

    auto diffe = [&Builder2,&gutils](Value* val) -> Value* {
        return gutils->diffe(val, Builder2);
    };

    auto addToDiffe = [&Builder2,&gutils](Value* val, Value* dif) -> void {
      gutils->addToDiffe(val, dif, Builder2);
    };

    auto setDiffe = [&](Value* val, Value* toset) -> void {
      gutils->setDiffe(val, toset, Builder2);
    };

    auto invertPointer = [&](Value* val) -> Value* {
        return gutils->invertPointerM(val, Builder2);
    };

  auto term = BB->getTerminator();
  assert(term);
  if (isa<ReturnInst>(term) || isa<BranchInst>(term) || isa<SwitchInst>(term)) {

  } else if (isa<UnreachableInst>(term)) {
    continue;
  } else {
    assert(BB);
    assert(BB->getParent());
    assert(term);
    llvm::errs() << *BB->getParent() << "\n";
    llvm::errs() << "unknown terminator instance " << *term << "\n";
    assert(0 && "unknown terminator inst");
  }



  for (BasicBlock::reverse_iterator I = BB->rbegin(), E = BB->rend(); I != E;) {
    Instruction* inst = &*I;
    assert(inst);
    I++;
    if (gutils->originalInstructions.find(inst) == gutils->originalInstructions.end()) continue;
    
    if (auto op = dyn_cast<BinaryOperator>(inst)) {
      if (gutils->isConstantInstruction(inst)) continue;
      Value* dif0 = nullptr;
      Value* dif1 = nullptr;
      switch(op->getOpcode()) {
        case Instruction::FMul:{
          auto idiff = diffe(inst);
          if (!gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFMul(idiff, lookup(op->getOperand(1)), "m0diffe"+op->getOperand(0)->getName());
          if (!gutils->isConstantValue(op->getOperand(1)))
            dif1 = Builder2.CreateFMul(idiff, lookup(op->getOperand(0)), "m1diffe"+op->getOperand(1)->getName());
          break;
        }
        case Instruction::FAdd:{
          auto idiff = diffe(inst);
          if (!gutils->isConstantValue(op->getOperand(0)))
            dif0 = idiff;
          if (!gutils->isConstantValue(op->getOperand(1)))
            dif1 = idiff;
          break;
        }
        case Instruction::FSub:{
          auto idiff = diffe(inst);
          if (!gutils->isConstantValue(op->getOperand(0)))
            dif0 = idiff;
          if (!gutils->isConstantValue(op->getOperand(1)))
            dif1 = Builder2.CreateFNeg(idiff);
          break;
        }
        case Instruction::FDiv:{
          auto idiff = diffe(inst);
          if (!gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFDiv(idiff, lookup(op->getOperand(1)), "d0diffe"+op->getOperand(0)->getName());
          if (!gutils->isConstantValue(op->getOperand(1)))
            dif1 = Builder2.CreateFNeg(
              Builder2.CreateFDiv(
                Builder2.CreateFMul(idiff, lookup(op)),
                lookup(op->getOperand(1)))
            );
          break;
        }
        default:
          //continue; // NOTE(TFK) added this.
          assert(op);
          llvm::errs() << *gutils->newFunc << "\n";
          llvm::errs() << "cannot handle unknown binary operator: " << *op << "\n";
          report_fatal_error("unknown binary operator");
      }

      if (dif0 || dif1) setDiffe(inst, Constant::getNullValue(inst->getType()));
      if (dif0) addToDiffe(op->getOperand(0), dif0);
      if (dif1) addToDiffe(op->getOperand(1), dif1);
    } else if(auto op = dyn_cast_or_null<IntrinsicInst>(inst)) {
      Value* dif0 = nullptr;
      Value* dif1 = nullptr;
      switch(op->getIntrinsicID()) {
        case Intrinsic::memcpy: {
            if (gutils->isConstantInstruction(inst)) continue;
                if (Type* secretty = isIntPointerASecretFloat(op->getOperand(0)) ) {
                    SmallVector<Value*, 4> args;
                    auto secretpt = PointerType::getUnqual(secretty);

                    args.push_back(Builder2.CreatePointerCast(invertPointer(op->getOperand(0)), secretpt));
                    args.push_back(Builder2.CreatePointerCast(invertPointer(op->getOperand(1)), secretpt));
                    args.push_back(Builder2.CreateUDiv(lookup(op->getOperand(2)),

                        ConstantInt::get(op->getOperand(2)->getType(), Builder2.GetInsertBlock()->getParent()->getParent()->getDataLayout().getTypeAllocSizeInBits(secretty)/8)
                    ));
                    auto dmemcpy = getOrInsertDifferentialFloatMemcpy(*M, secretpt);
                    Builder2.CreateCall(dmemcpy, args);
                } else {
                    if (topLevel) {
                        SmallVector<Value*, 4> args;
                        IRBuilder <>BuilderZ(op);
                        args.push_back(gutils->invertPointerM(op->getOperand(0), BuilderZ));
                        args.push_back(gutils->invertPointerM(op->getOperand(1), BuilderZ));
                        args.push_back(op->getOperand(2));
                        args.push_back(op->getOperand(3));

                        Type *tys[] = {args[0]->getType(), args[1]->getType(), args[2]->getType()};
                        auto cal = BuilderZ.CreateCall(Intrinsic::getDeclaration(gutils->newFunc->getParent(), Intrinsic::memcpy, tys), args);
                        cal->setAttributes(op->getAttributes());
                        cal->setCallingConv(op->getCallingConv());
                        cal->setTailCallKind(op->getTailCallKind());
                    }
                }
            break;
        }
        case Intrinsic::memset: {
            if (gutils->isConstantInstruction(inst)) continue;
            if (!gutils->isConstantValue(op->getOperand(1))) {
                assert(inst);
                llvm::errs() << "couldn't handle non constant inst in memset to propagate differential to\n" << *inst;
                report_fatal_error("non constant in memset");
            }
            auto ptx = invertPointer(op->getOperand(0));
            SmallVector<Value*, 4> args;
            args.push_back(ptx);
            for(unsigned i=1; i<op->getNumArgOperands(); i++) {
                args.push_back(lookup(op->getOperand(i)));
            }

            Type *tys[] = {args[0]->getType(), args[2]->getType()};
            auto cal = Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args);
            cal->setAttributes(op->getAttributes());
            cal->setCallingConv(op->getCallingConv());
            cal->setTailCallKind(op->getTailCallKind());

            break;
        }
        case Intrinsic::assume:
        case Intrinsic::stacksave:
        case Intrinsic::stackrestore:
        case Intrinsic::dbg_declare:
        case Intrinsic::dbg_value:
        #if LLVM_VERSION_MAJOR > 6
        case Intrinsic::dbg_label:
        #endif
        case Intrinsic::dbg_addr:
            break;
        case Intrinsic::lifetime_start:{
            if (gutils->isConstantInstruction(inst)) continue;
            SmallVector<Value*, 2> args = {lookup(op->getOperand(0)), lookup(op->getOperand(1))};
            Type *tys[] = {args[1]->getType()};
            auto cal = Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::lifetime_end, tys), args);
            cal->setAttributes(op->getAttributes());
            cal->setCallingConv(op->getCallingConv());
            cal->setTailCallKind(op->getTailCallKind());
            break;
        }
        case Intrinsic::lifetime_end:
            gutils->erase(op);
            break;
        case Intrinsic::sqrt: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateBinOp(Instruction::FDiv,
              Builder2.CreateFMul(ConstantFP::get(op->getType(), 0.5), diffe(inst)),
              lookup(op)
            );
          break;
        }
        case Intrinsic::fabs: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0))) {
            auto cmp = Builder2.CreateFCmpOLT(lookup(op->getOperand(0)), ConstantFP::get(op->getOperand(0)->getType(), 0));
            dif0 = Builder2.CreateFMul(Builder2.CreateSelect(cmp, ConstantFP::get(op->getOperand(0)->getType(), -1), ConstantFP::get(op->getOperand(0)->getType(), 1)), diffe(inst));
          }
          break;
        }
        case Intrinsic::x86_sse_max_ss:
        case Intrinsic::x86_sse_max_ps:
        case Intrinsic::maxnum: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0))) {
            auto cmp = Builder2.CreateFCmpOLT(lookup(op->getOperand(0)), lookup(op->getOperand(1)));
            dif0 = Builder2.CreateSelect(cmp, ConstantFP::get(op->getOperand(0)->getType(), 0), diffe(inst));
          }
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(1))) {
            auto cmp = Builder2.CreateFCmpOLT(lookup(op->getOperand(0)), lookup(op->getOperand(1)));
            dif1 = Builder2.CreateSelect(cmp, diffe(inst), ConstantFP::get(op->getOperand(0)->getType(), 0));
          }
          break;
        }
        case Intrinsic::x86_sse_min_ss:
        case Intrinsic::x86_sse_min_ps:
        case Intrinsic::minnum: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0))) {
            auto cmp = Builder2.CreateFCmpOLT(lookup(op->getOperand(0)), lookup(op->getOperand(1)));
            dif0 = Builder2.CreateSelect(cmp, diffe(inst), ConstantFP::get(op->getOperand(0)->getType(), 0));
          }
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(1))) {
            auto cmp = Builder2.CreateFCmpOLT(lookup(op->getOperand(0)), lookup(op->getOperand(1)));
            dif1 = Builder2.CreateSelect(cmp, ConstantFP::get(op->getOperand(0)->getType(), 0), diffe(inst));
          }
          break;
        }

        case Intrinsic::log: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFDiv(diffe(inst), lookup(op->getOperand(0)));
          break;
        }
        case Intrinsic::log2: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFDiv(diffe(inst),
              Builder2.CreateFMul(ConstantFP::get(op->getType(), 0.6931471805599453), lookup(op->getOperand(0)))
            );
          break;
        }
        case Intrinsic::log10: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFDiv(diffe(inst),
              Builder2.CreateFMul(ConstantFP::get(op->getType(), 2.302585092994046), lookup(op->getOperand(0)))
            );
          break;
        }
        case Intrinsic::exp: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFMul(diffe(inst), lookup(op));
          break;
        }
        case Intrinsic::exp2: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFMul(
              Builder2.CreateFMul(diffe(inst), lookup(op)), ConstantFP::get(op->getType(), 0.6931471805599453)
            );
          break;
        }
        case Intrinsic::pow: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0))) {

            /*
            dif0 = Builder2.CreateFMul(
              Builder2.CreateFMul(diffe(inst),
                Builder2.CreateFDiv(lookup(op), lookup(op->getOperand(0)))), lookup(op->getOperand(1))
            );
            */
            SmallVector<Value*, 2> args = {lookup(op->getOperand(0)), Builder2.CreateFSub(lookup(op->getOperand(1)), ConstantFP::get(op->getType(), 1.0))};
            Type *tys[] = {args[1]->getType()};
            auto cal = Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::pow, tys), args);
            cal->setAttributes(op->getAttributes());
            cal->setCallingConv(op->getCallingConv());
            cal->setTailCallKind(op->getTailCallKind());
            dif0 = Builder2.CreateFMul(
              Builder2.CreateFMul(diffe(inst), cal)
              , lookup(op->getOperand(1))
            );
          }

          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(1))) {
            Value *args[] = {lookup(op->getOperand(1))};
            Type *tys[] = {op->getOperand(1)->getType()};

            dif1 = Builder2.CreateFMul(
              Builder2.CreateFMul(diffe(inst), lookup(op)),
              Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::log, tys), args)
            );
          }

          break;
        }
        case Intrinsic::sin: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0))) {
            Value *args[] = {lookup(op->getOperand(0))};
            Type *tys[] = {op->getOperand(0)->getType()};
            dif0 = Builder2.CreateFMul(diffe(inst),
              Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::cos, tys), args) );
          }
          break;
        }
        case Intrinsic::cos: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0))) {
            Value *args[] = {lookup(op->getOperand(0))};
            Type *tys[] = {op->getOperand(0)->getType()};
            dif0 = Builder2.CreateFMul(diffe(inst),
              Builder2.CreateFNeg(
                Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::sin, tys), args) )
            );
          }
          break;
        }
        case Intrinsic::floor:
        case Intrinsic::ceil:
        case Intrinsic::trunc:
        case Intrinsic::rint:
        case Intrinsic::nearbyint:
        case Intrinsic::round: {
            //Derivative of these is zero
            break;
        }
        default:
          if (gutils->isConstantInstruction(inst)) continue;
          assert(inst);
          llvm::errs() << "cannot handle unknown intrinsic\n" << *inst;
          report_fatal_error("unknown intrinsic");
      }

      if (dif0 || dif1) setDiffe(inst, Constant::getNullValue(inst->getType()));
      if (dif0) addToDiffe(op->getOperand(0), dif0);
      if (dif1) addToDiffe(op->getOperand(1), dif1);
    } else if(auto op = dyn_cast_or_null<CallInst>(inst)) {
      handleGradientCallInst(I, E, Builder2, op, gutils, TLI, global_AA, global_AA, topLevel, replacedReturns, uncacheable_args_map[op]);
    } else if(auto op = dyn_cast_or_null<SelectInst>(inst)) {
      if (gutils->isConstantValue(inst)) continue;
      if (op->getType()->isPointerTy()) continue;

      Value* dif1 = nullptr;
      Value* dif2 = nullptr;

      if (!gutils->isConstantValue(op->getOperand(1)))
        dif1 = Builder2.CreateSelect(lookup(op->getOperand(0)), diffe(inst), Constant::getNullValue(op->getOperand(1)->getType()), "diffe"+op->getOperand(1)->getName());
      if (!gutils->isConstantValue(op->getOperand(2)))
        dif2 = Builder2.CreateSelect(lookup(op->getOperand(0)), Constant::getNullValue(op->getOperand(2)->getType()), diffe(inst), "diffe"+op->getOperand(2)->getName());

      setDiffe(inst, Constant::getNullValue(inst->getType()));
      if (dif1) addToDiffe(op->getOperand(1), dif1);
      if (dif2) addToDiffe(op->getOperand(2), dif2);
    } else if(auto op = dyn_cast<LoadInst>(inst)) {
      if (gutils->isConstantValue(inst) || gutils->isConstantInstruction(inst)) continue;

      auto op_operand = op->getPointerOperand();
      auto op_type = op->getType();

      if (can_modref_map[inst]) {
        IRBuilder<> BuilderZ(op->getNextNode());
        inst = cast<Instruction>(gutils->addMalloc(BuilderZ, inst));
        if (inst != op) {
          // Set to nullptr since op should never be used after invalidated through addMalloc.
          op = nullptr;
          gutils->nonconstant_values.insert(inst);
          gutils->nonconstant.insert(inst);
          gutils->originalInstructions.insert(inst);
          assert(inst->getType() == op_type);
        }
      }

      // TODO IF OP IS POINTER
      if (!op_type->isPointerTy()) {
        auto prediff = diffe(inst);
        setDiffe(inst, Constant::getNullValue(op_type));
        gutils->addToPtrDiffe(op_operand, prediff, Builder2);
      } else {
        //Builder2.CreateStore(diffe(inst), invertPointer(op->getOperand(0)));//, op->getName()+"'psweird");
        //addToNPtrDiffe(op->getOperand(0), diffe(inst));
        //assert(0 && "cannot handle non const pointer load inversion");
        //assert(op);
        //llvm::errs() << "ignoring load bc pointer of " << *op << "\n";
      }

    } else if(auto op = dyn_cast<StoreInst>(inst)) {
      if (gutils->isConstantInstruction(inst)) continue;

      //TODO const
       //TODO IF OP IS POINTER
      if (! ( op->getValueOperand()->getType()->isPointerTy() || (op->getValueOperand()->getType()->isIntegerTy() && !isIntASecretFloat(op->getValueOperand()) ) ) ) {
          StoreInst* ts;
          if (!gutils->isConstantValue(op->getValueOperand())) {
            auto dif1 = Builder2.CreateLoad(invertPointer(op->getPointerOperand()));
            ts = gutils->setPtrDiffe(op->getPointerOperand(), Constant::getNullValue(op->getValueOperand()->getType()), Builder2);
            addToDiffe(op->getValueOperand(), dif1);
          } else {
            ts = gutils->setPtrDiffe(op->getPointerOperand(), Constant::getNullValue(op->getValueOperand()->getType()), Builder2);
          }
          ts->setAlignment(op->getAlignment());
          ts->setVolatile(op->isVolatile());
          ts->setOrdering(op->getOrdering());
          ts->setSyncScopeID(op->getSyncScopeID());
      } else if (topLevel) {
        IRBuilder <> storeBuilder(op);
        //llvm::errs() << "op value: " << *op->getValueOperand() << "\n";
        Value* valueop = gutils->invertPointerM(op->getValueOperand(), storeBuilder);
        //llvm::errs() << "op pointer: " << *op->getPointerOperand() << "\n";
        Value* pointerop = gutils->invertPointerM(op->getPointerOperand(), storeBuilder);
        storeBuilder.CreateStore(valueop, pointerop);
        //llvm::errs() << "ignoring store bc pointer of " << *op << "\n";
      }
    } else if(auto op = dyn_cast<ExtractValueInst>(inst)) {
      if (gutils->isConstantValue(inst)) continue;
      if (op->getType()->isPointerTy()) continue;

      auto prediff = diffe(inst);
      //todo const
      if (!gutils->isConstantValue(op->getOperand(0))) {
        SmallVector<Value*,4> sv;
        for(auto i : op->getIndices())
            sv.push_back(ConstantInt::get(Type::getInt32Ty(Context), i));
        gutils->addToDiffeIndexed(op->getOperand(0), prediff, sv, Builder2);
      }
      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(auto op = dyn_cast<InsertValueInst>(inst)) {
      if (gutils->isConstantValue(inst)) continue;
      auto st = cast<StructType>(op->getType());
      bool hasNonPointer = false;
      for(unsigned i=0; i<st->getNumElements(); i++) {
        if (!st->getElementType(i)->isPointerTy()) {
           hasNonPointer = true;
        }
      }
      if (!hasNonPointer) continue;

      if (!gutils->isConstantValue(op->getInsertedValueOperand()) && !op->getInsertedValueOperand()->getType()->isPointerTy()) {
        auto prediff = gutils->diffe(inst, Builder2);
        auto dindex = Builder2.CreateExtractValue(prediff, op->getIndices());
        gutils->addToDiffe(op->getOperand(1), dindex, Builder2);
      }

      if (!gutils->isConstantValue(op->getAggregateOperand()) && !op->getAggregateOperand()->getType()->isPointerTy()) {
        auto prediff = gutils->diffe(inst, Builder2);
        auto dindex = Builder2.CreateInsertValue(prediff, Constant::getNullValue(op->getInsertedValueOperand()->getType()), op->getIndices());
        gutils->addToDiffe(op->getAggregateOperand(), dindex, Builder2);
      }

      gutils->setDiffe(inst, Constant::getNullValue(inst->getType()), Builder2);
    } else if (auto op = dyn_cast<ShuffleVectorInst>(inst)) {
      if (gutils->isConstantValue(inst)) continue;

      auto loaded = diffe(inst);
      size_t l1 = cast<VectorType>(op->getOperand(0)->getType())->getNumElements();
      uint64_t instidx = 0;
      for( size_t idx : op->getShuffleMask()) {
        auto opnum = (idx < l1) ? 0 : 1;
        auto opidx = (idx < l1) ? idx : (idx-l1);
        SmallVector<Value*,4> sv;
        sv.push_back(ConstantInt::get(Type::getInt32Ty(Context), opidx));
        if (!gutils->isConstantValue(op->getOperand(opnum)))
          gutils->addToDiffeIndexed(op->getOperand(opnum), Builder2.CreateExtractElement(loaded, instidx), sv, Builder2);
        instidx++;
      }
      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(auto op = dyn_cast<ExtractElementInst>(inst)) {
      if (gutils->isConstantValue(inst)) continue;

      if (!gutils->isConstantValue(op->getVectorOperand())) {
        SmallVector<Value*,4> sv;
        sv.push_back(op->getIndexOperand());
        gutils->addToDiffeIndexed(op->getVectorOperand(), diffe(inst), sv, Builder2);
      }
      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(auto op = dyn_cast<InsertElementInst>(inst)) {
      if (gutils->isConstantValue(inst)) continue;

      auto dif1 = diffe(inst);

      if (!gutils->isConstantValue(op->getOperand(0)))
        addToDiffe(op->getOperand(0), Builder2.CreateInsertElement(dif1, Constant::getNullValue(op->getOperand(1)->getType()), lookup(op->getOperand(2)) ));

      if (!gutils->isConstantValue(op->getOperand(1)))
        addToDiffe(op->getOperand(1), Builder2.CreateExtractElement(dif1, lookup(op->getOperand(2))));

      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(auto op = dyn_cast<CastInst>(inst)) {
      if (gutils->isConstantValue(inst)) continue;
      if (op->getType()->isPointerTy()) continue;

      if (!gutils->isConstantValue(op->getOperand(0))) {
        if (op->getOpcode()==CastInst::CastOps::FPTrunc || op->getOpcode()==CastInst::CastOps::FPExt) {
          addToDiffe(op->getOperand(0), Builder2.CreateFPCast(diffe(inst), op->getOperand(0)->getType()));
        } else if (op->getOpcode()==CastInst::CastOps::BitCast) {
          addToDiffe(op->getOperand(0), Builder2.CreateBitCast(diffe(inst), op->getOperand(0)->getType()));
        } else {
            llvm::errs() << *inst->getParent()->getParent() << "\n" << *inst->getParent() << "\n";
            llvm::errs() << "cannot handle above cast " << *inst << "\n";
            report_fatal_error("unknown instruction");
        }
      }
      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(isa<CmpInst>(inst) || isa<PHINode>(inst) || isa<BranchInst>(inst) || isa<SwitchInst>(inst) || isa<AllocaInst>(inst) || isa<CastInst>(inst) || isa<GetElementPtrInst>(inst)) {
        continue;
    } else {
      assert(inst);
      assert(inst->getParent());
      assert(inst->getParent()->getParent());
      llvm::errs() << *inst->getParent()->getParent() << "\n" << *inst->getParent() << "\n";
      llvm::errs() << "cannot handle above inst " << *inst << "\n";
      report_fatal_error("unknown instruction");
    }
  }

    createInvertedTerminator(gutils, BB, retAlloca, 0 + (additionalArg ? 1 : 0) + (differentialReturn ? 1 : 0));

  }

  if (!topLevel)
    gutils->eraseStructuralStoresAndCalls();

  while(gutils->inversionAllocs->size() > 0) {
    gutils->inversionAllocs->back().moveBefore(gutils->newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  }

  (IRBuilder <>(gutils->inversionAllocs)).CreateUnreachable();
  DeleteDeadBlock(gutils->inversionAllocs);
  for(auto BBs : gutils->reverseBlocks) {
    if (pred_begin(BBs.second) == pred_end(BBs.second)) {
        (IRBuilder <>(BBs.second)).CreateUnreachable();
        DeleteDeadBlock(BBs.second);
    }
  }

  for (Argument &Arg : gutils->newFunc->args()) {
      if (Arg.hasAttribute(Attribute::Returned))
          Arg.removeAttr(Attribute::Returned);
      if (Arg.hasAttribute(Attribute::StructRet))
          Arg.removeAttr(Attribute::StructRet);
  }
  if (gutils->newFunc->hasFnAttribute(Attribute::OptimizeNone))
    gutils->newFunc->removeFnAttr(Attribute::OptimizeNone);

  if (auto bytes = gutils->newFunc->getDereferenceableBytes(llvm::AttributeList::ReturnIndex)) {
    AttrBuilder ab;
    ab.addDereferenceableAttr(bytes);
    gutils->newFunc->removeAttributes(llvm::AttributeList::ReturnIndex, ab);
  }
  if (gutils->newFunc->hasAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias)) {
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias);
  }
  if (gutils->newFunc->hasAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::ZExt)) {
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::ZExt);
  }

  if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
      llvm::errs() << *gutils->oldFunc << "\n";
      llvm::errs() << *gutils->newFunc << "\n";
      report_fatal_error("function failed verification (4)");
  }

  optimizeIntermediate(gutils, topLevel, gutils->newFunc);

  auto nf = gutils->newFunc;
  delete gutils;

  return nf;
}
