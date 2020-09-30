//===- ActivityAnalysis.cpp - Implementation of Activity Analysis  -----------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of Activity Analysis -- an AD-specific
// analysis that deduces if a given instruction or value can impact the
// calculation of a derivative. This file consists of two mutually recurive
// functions that compute this for values and instructions, respectively.
//
//===----------------------------------------------------------------------===//
#include <cstdint>
#include <deque>

#include <llvm/Config/llvm-config.h>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/IR/InstIterator.h"

#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/InlineAsm.h"

#include "ActivityAnalysis.h"
#include "Utils.h"

#include "TypeAnalysis/TBAA.h"
#include "LibraryFuncs.h"

#include "llvm/Analysis/ValueTracking.h"

using namespace llvm;

cl::opt<bool> printconst("enzyme_printconst", cl::init(false), cl::Hidden,
                         cl::desc("Print constant detection algorithm"));

cl::opt<bool> nonmarkedglobals_inactive(
    "enzyme_nonmarkedglobals_inactive", cl::init(false), cl::Hidden,
    cl::desc("Consider all nonmarked globals to be inactive"));

cl::opt<bool> emptyfnconst("enzyme_emptyfnconst", cl::init(false), cl::Hidden,
                           cl::desc("Empty functions are considered constant"));

#include "llvm/IR/InstIterator.h"
#include <map>
#include <set>
#include <unordered_map>


constexpr uint8_t UP = 1;
constexpr uint8_t DOWN = 2;

bool couldFunctionArgumentCapture(CallInst *CI, Value *val) {
  Function *F = CI->getCalledFunction();
  if (F == nullptr)
    return true;

  if (F->getIntrinsicID() == Intrinsic::memset)
    return false;
  if (F->getIntrinsicID() == Intrinsic::memcpy)
    return false;
  if (F->getIntrinsicID() == Intrinsic::memmove)
    return false;

  if (F->empty())
    return false;

  auto arg = F->arg_begin();
  for(size_t i=0, size = CI->getNumArgOperands(); i < size; i++) {
    if (val == CI->getArgOperand(i)) {
      // This is a vararg, assume captured
      if (arg == F->arg_end()) {
        return true;
      } else {
        if (!arg->hasNoCaptureAttr()) {
          return true;
        }
      }
    }
    if (arg != F->arg_end())
      arg++;
  }
  // No argument captured
  return false;
}

bool ActivityAnalyzer::isFunctionArgumentConstant(CallInst *CI, Value *val) {
  Function *F = CI->getCalledFunction();
  if (F == nullptr)
    return false;

  auto fn = F->getName();
  // todo realloc consider?
  // For known library functions, special case how derivatives flow to allow for
  // more aggressive active variable detection
  if (fn == "malloc" || fn == "free" || fn == "_Znwm" ||
      fn == "__cxa_guard_acquire" || fn == "__cxa_guard_release" ||
      fn == "__cxa_guard_abort")
    return true;
  if (F->getIntrinsicID() == Intrinsic::memset && CI->getArgOperand(0) != val &&
      CI->getArgOperand(1) != val)
    return true;
  if (F->getIntrinsicID() == Intrinsic::memcpy && CI->getArgOperand(0) != val &&
      CI->getArgOperand(1) != val)
    return true;
  if (F->getIntrinsicID() == Intrinsic::memmove &&
      CI->getArgOperand(0) != val && CI->getArgOperand(1) != val)
    return true;

  if (F->empty())
    return false;

  // return false;
  if (fn.startswith("augmented"))
    return false;
  if (fn.startswith("fakeaugmented"))
    return false;
  if (fn.startswith("diffe"))
    return false;
  // if (val->getType()->isPointerTy()) return false;
  if (!val->getType()->isIntOrIntVectorTy())
    return false;

  //assert(retvals.find(val) == retvals.end());

  // TODO need to fixup the below, it currently is incorrect, but didn't have
  // time to fix rn
  return false;

  #if 0
  // static std::unordered_map<std::tuple<Function*, Value*,
  // SmallPtrSet<Value*,20>, std::set<Value*> >, bool> metacache;
  static std::map<
      std::tuple<CallInst *, Value *, std::set<Value *>, std::set<Value *>,
                 std::set<Value *>, std::set<Value *>>,
      bool>
      metacache;
  // auto metatuple = std::make_tuple(F, val,
  // SmallPtrSet<Value*,20>(constants.begin(), constants.end()),
  // std::set<Value*>(nonconstant.begin(), nonconstant.end()));
  auto metatuple = std::make_tuple(
      CI, val, std::set<Value *>(constants.begin(), constants.end()),
      std::set<Value *>(nonconstant.begin(), nonconstant.end()),
      std::set<Value *>(constantvals.begin(), constantvals.end()),
      std::set<Value *>(retvals.begin(), retvals.end()));
  if (metacache.find(metatuple) != metacache.end()) {
    if (printconst)
      llvm::errs() << " < SUBFN metacache const " << F->getName()
                   << "> arg: " << *val << " ci:" << *CI << "\n";
    return metacache[metatuple];
  }
  if (printconst)
    llvm::errs() << " < METAINDUCTIVE SUBFN const " << F->getName()
                 << "> arg: " << *val << " ci:" << *CI << "\n";

  metacache[metatuple] = true;
  // Note that the base case of true broke the up/down variant so have to be
  // very conservative
  //  as a consequence we cannot detect const of recursive functions :'( [in
  //  that will be too conservative]
  // metacache[metatuple] = false;

  SmallPtrSet<Value *, 20> constants2;
  constants2.insert(constants.begin(), constants.end());
  SmallPtrSet<Value *, 20> nonconstant2;
  nonconstant2.insert(nonconstant.begin(), nonconstant.end());
  SmallPtrSet<Value *, 20> constantvals2;
  constantvals2.insert(constantvals.begin(), constantvals.end());
  SmallPtrSet<Value *, 20> retvals2;
  retvals2.insert(retvals.begin(), retvals.end());

  // Ask the question, even if is this is active, are all its uses inactive
  // (meaning this use does not impact its activity)
  nonconstant2.insert(val);
  // retvals2.insert(val);

  // constants2.insert(val);

  if (printconst) {
    llvm::errs() << " < SUBFN " << F->getName() << "> arg: " << *val
                 << " ci:" << *CI << "\n";
  }

  auto a = F->arg_begin();

  std::set<int> arg_constants;
  std::set<int> idx_findifactive;
  SmallPtrSet<Value *, 20> arg_findifactive;

  SmallPtrSet<Value *, 20> newconstants;
  SmallPtrSet<Value *, 20> newnonconstant;

  FnTypeInfo nextTypeInfo(F);
  int argnum = 0;
  for (auto &arg : F->args()) {
    nextTypeInfo.Arguments.insert(std::pair<Argument *, TypeTree>(
        &arg, TR.query(CI->getArgOperand(argnum))));
    ++argnum;
  }
  nextTypeInfo.Return = TR.query(CI);
  TypeResults TR2 = TR.analysis.analyzeFunction(nextTypeInfo);

  for (unsigned i = 0; i < CI->getNumArgOperands(); ++i) {
    if (CI->getArgOperand(i) == val) {
      arg_findifactive.insert(a);
      idx_findifactive.insert(i);
      newnonconstant.insert(a);
      ++a;
      continue;
    }

    if (isconstantValueM(TR, CI->getArgOperand(i), constants2, nonconstant2,
                         constantvals2, retvals2, AA),
        directions) {
      newconstants.insert(a);
      arg_constants.insert(i);
    } else {
      newnonconstant.insert(a);
    }
    ++a;
  }

  bool constret;

  // allow return index as valid entry as well
  if (CI != val) {
    constret = isconstantValueM(TR, CI, constants2, nonconstant2, constantvals2,
                                retvals2, AA, directions);
    if (constret)
      arg_constants.insert(-1);
  } else {
    constret = false;
    arg_findifactive.insert(a);
    idx_findifactive.insert(-1);
  }

  static std::map<std::tuple<std::set<int>, Function *, std::set<int>>, bool>
      cache;

  auto tuple = std::make_tuple(arg_constants, F, idx_findifactive);
  if (cache.find(tuple) != cache.end()) {
    if (printconst)
      llvm::errs() << " < SUBFN cache const " << F->getName()
                   << "> arg: " << *val << " ci:" << *CI << "\n";
    return cache[tuple];
  }

  //! inductively assume that it is constant, it should be deduced nonconstant
  //! elsewhere if this is not the case
  if (printconst)
    llvm::errs() << " < INDUCTIVE SUBFN const " << F->getName()
                 << "> arg: " << *val << " ci:" << *CI << "\n";

  cache[tuple] = true;
  // Note that the base case of true broke the up/down variant so have to be
  // very conservative
  //  as a consequence we cannot detect const of recursive functions :'( [in
  //  that will be too conservative]
  // cache[tuple] = false;

  SmallPtrSet<Value *, 4> newconstantvals;
  newconstantvals.insert(constantvals2.begin(), constantvals2.end());

  SmallPtrSet<Value *, 4> newretvals;
  newretvals.insert(retvals2.begin(), retvals2.end());

  for (llvm::inst_iterator I = llvm::inst_begin(F), E = llvm::inst_end(F);
       I != E; ++I) {
    if (auto ri = dyn_cast<ReturnInst>(&*I)) {
      if (!constret) {
        newretvals.insert(ri->getReturnValue());
        if (CI == val)
          arg_findifactive.insert(ri->getReturnValue());
      } else {
        // newconstantvals.insert(ri->getReturnValue());
      }
    }
  }

  for (auto specialarg : arg_findifactive) {
    for (auto user : specialarg->users()) {
      if (printconst)
        llvm::errs() << " going to consider user " << *user << "\n";
      if (!isconstantValueM(TR2, user, newconstants, newnonconstant,
                            newconstantvals, newretvals, AA, 3)) {
        if (printconst)
          llvm::errs() << " < SUBFN nonconst " << F->getName()
                       << "> arg: " << *val << " ci:" << *CI
                       << "  from sf: " << *user << "\n";
        metacache.erase(metatuple);
        return cache[tuple] = false;
      }
    }
  }

  constants.insert(constants2.begin(), constants2.end());
  nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
  if (printconst) {
    llvm::errs() << " < SUBFN const " << F->getName() << "> arg: " << *val
                 << " ci:" << *CI << "\n";
  }
  metacache.erase(metatuple);
  return cache[tuple] = true;
  #endif
}

static bool
isGuaranteedConstantValue(TypeResults &TR, Value *val,
                          const SmallPtrSetImpl<Value *> *constantvals) {
  // This result of this instruction is certainly an integer (and only and
  // integer, not a pointer or float). Therefore its value is inactive
  // Note that this is correct, but not aggressive as it should be (we should
  // call isConstantValue(inst) here, but we need to be careful to not have an
  // infinite recursion)

  //  TODO: make this more aggressive
  if (TR.intType(val, /*errIfNotFound=*/false).isIntegral()) {
    if (printconst)
      llvm::errs() << " -> known integer " << *val << "\n";
    return true;
    // if we happen to have already deduced this instruction constant, we might
    // as well use the information
  } else if (constantvals && constantvals->find(val) != constantvals->end()) {
    if (printconst)
      llvm::errs() << " -> previous constant value " << *val << "\n";
    return true;
    // if we know the subtype contains no derivative information, we can assure
    // that this is a constant value
  } else if (val->getType()->isVoidTy() || val->getType()->isEmptyTy()) {
    return true;
  }
  return false;
}

// propagateFromOperand should consider the value that would make this call
// active and return true if it does make it active and thus
//   we don't need to consider any further operands
void propagateArgumentInformation(
    CallInst &CI, std::function<bool(Value *)> propagateFromOperand) {

  if (auto called = CI.getCalledFunction()) {
    auto n = called->getName();
    if (n == "lgamma" || n == "lgammaf" || n == "lgammal" || n == "lgamma_r" ||
        n == "lgammaf_r" || n == "lgammal_r" || n == "__lgamma_r_finite" ||
        n == "__lgammaf_r_finite" || n == "__lgammal_r_finite" || n == "tanh" ||
        n == "tanhf") {

      propagateFromOperand(CI.getArgOperand(0));
      return;
    }
  }

  for (auto &a : CI.arg_operands()) {
    if (propagateFromOperand(a))
      break;
  }
}

bool ActivityAnalyzer::isconstantM(TypeResults &TR, Instruction *inst) {
  assert(inst);
  assert(TR.info.Function == inst->getParent()->getParent());
  if (isa<ReturnInst>(inst))
    return true;

  if (isa<UnreachableInst>(inst) || isa<BranchInst>(inst) ||
      (constants.find(inst) != constants.end())) {
    return true;
  }

  if ((nonconstant.find(inst) != nonconstant.end())) {
    return false;
  }

  //if (isa<SIToFPInst>(inst) || isa<UIToFPInst>(inst) || isa<FPToSIInst>(inst) ||
  //    isa<FPToUIInst>(inst)) {
  //  constants.insert(inst);
  //  return true;
  //}

  if (auto storeinst = dyn_cast<StoreInst>(inst)) {
    auto storeSize =
        storeinst->getParent()
            ->getParent()
            ->getParent()
            ->getDataLayout()
            .getTypeSizeInBits(storeinst->getValueOperand()->getType()) /
        8;

    bool allIntegral = true;
    bool anIntegral = false;
    auto q = TR.query(storeinst->getPointerOperand()).Data0();
    for (int i = -1; i < (int)storeSize; ++i) {
      auto dt = q[{i}];
      if (dt.isIntegral() || dt == BaseType::Anything) {
        anIntegral = true;
      } else if (dt.isKnown()) {
        allIntegral = false;
        break;
      }
    }

    if (allIntegral && anIntegral) {
      if (printconst)
        llvm::errs() << " constant instruction from TA " << *inst << "\n";
      constants.insert(inst);
      return true;
    }
  }


  if (printconst)
    llvm::errs() << "checking if is constant[" << (int)directions << "] "
                 << *inst << "\n";

  std::shared_ptr<ActivityAnalyzer> DownHypothesis;
  
  // If this instruction does not write memory to memory that outlives itself
  // (therefore propagating derivative information), and the return value of
  // this instruction is known to be inactive this instruction is inactive as it
  // cannot propagate derivative information
  if (!inst->mayWriteToMemory() ||
      (isa<CallInst>(inst) && AA.onlyReadsMemory(cast<CallInst>(inst)))) {

    // Even if returning a pointer, this instruction is considered inactive
    // since the instruction doesn't prop gradients
    if (!TR.intType(inst, /*errifNotFound*/false).isPossibleFloat()) {
      if (printconst)
        llvm::errs() << " constant instruction from known non-float non-writing "
                        "instruction "
                     << *inst << "\n";
      constants.insert(inst);
      return true;
    }
    if (isconstantValueM(TR, inst)) {
      if (printconst)
        llvm::errs() << " constant instruction from known constant non-writing "
                        "instruction "
                     << *inst << "\n";
      constants.insert(inst);
      return true;
    }
    // Additionally worth checking explicitly since
    // we don't care about isConstantValue's explicit
    // ptr checks
    if ( (directions & DOWN) ) {
      // The standard optimization wherein if we are nonphi and already set as such
      // we don't need an inductive hypothesis and thus can simply use this object
      if (directions == DOWN && !isa<PHINode>(inst)) {
        if (isValueInactiveFromUsers(TR, inst)) {
          if (printconst)
            llvm::errs() << " constant instruction from users "
                            "instruction "
                        << *inst << "\n";
          constants.insert(inst);
          return true;
        }
      } else {
        DownHypothesis = std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, DOWN));
        DownHypothesis->constants.insert(inst);
        if (DownHypothesis->isValueInactiveFromUsers(TR, inst)) {
          if (printconst)
            llvm::errs() << " constant instruction from users "
                            "instruction "
                        << *inst << "\n";
          constants.insert(inst);
          insertConstantsFrom(*DownHypothesis);
          return true;
        }
      }
    }
  }

  std::shared_ptr<ActivityAnalyzer> UpHypothesis;
  if ( (directions & UP) ) {
    UpHypothesis = std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, UP));
    UpHypothesis->constants.insert(inst);
    assert(directions & UP);
    if (UpHypothesis->isInstructionInactiveFromOrigin(TR, inst)) {
      if (printconst)
        llvm::errs() << " constant instruction from origin "
                        "instruction "
                     << *inst << "\n";
      constants.insert(inst);
      insertConstantsFrom(*UpHypothesis);
      if (DownHypothesis) insertConstantsFrom(*DownHypothesis);
      return true;
    }
  }

  nonconstant.insert(inst);
  if (printconst)
    llvm::errs() << "couldnt decide fallback as nonconstant instruction(" << (int)directions
                 << "):" << *inst << "\n";
  return false;
}

bool ActivityAnalyzer::isconstantValueM(TypeResults &TR, Value *val) {
  assert(val);
  if (auto inst = dyn_cast<Instruction>(val)) {
    assert(TR.info.Function == inst->getParent()->getParent());
  }
  if (auto arg = dyn_cast<Argument>(val)) {
    assert(TR.info.Function == arg->getParent());
  }
  // assert(directions >= 0);
  assert(directions <= 3);

  // llvm::errs() << "looking for: " << *val << "\n";
  if (val->getType()->isVoidTy())
    return true;

  //! False so we can replace function with augmentation
  if (isa<Function>(val)) {
    return false;
  }

  if (isa<UndefValue>(val) || isa<MetadataAsValue>(val))
    return true;

  if (isa<ConstantData>(val) || isa<ConstantAggregate>(val)) {
    if (printconst)
      llvm::errs() << " VALUE const as constdata: " << *val << "\n";
    return true;
  }
  if (isa<BasicBlock>(val))
    return true;
  assert(!isa<InlineAsm>(val));

  if (auto op = dyn_cast<IntrinsicInst>(val)) {
      switch (op->getIntrinsicID()) {
      case Intrinsic::assume:
      case Intrinsic::stacksave:
      case Intrinsic::stackrestore:
      case Intrinsic::lifetime_start:
      case Intrinsic::lifetime_end:
      case Intrinsic::dbg_addr:
      case Intrinsic::dbg_declare:
      case Intrinsic::dbg_value:
      case Intrinsic::invariant_start:
      case Intrinsic::invariant_end:
      case Intrinsic::var_annotation:
      case Intrinsic::ptr_annotation:
      case Intrinsic::annotation:
      case Intrinsic::codeview_annotation:
      case Intrinsic::expect:
      case Intrinsic::type_test:
      case Intrinsic::donothing:
        // case Intrinsic::is_constant:
        return true;
      default:
        break;
      }
    }

  if (constantvals.find(val) != constantvals.end()) {
     if (printconst)
        llvm::errs() << " VALUE const from precomputation " << *val << "\n";
    return true;
  }
  if (retvals.find(val) != retvals.end()) {
      if (printconst)
        llvm::errs() << " VALUE nonconst from arg nonconst " << *val << "\n";
    return false;
  }

  // All arguments should be marked constant/nonconstant ahead of time
  if (isa<Argument>(val)) {
    llvm::errs() << *(cast<Argument>(val)->getParent()) << "\n";
    llvm::errs() << *val << "\n";
    assert(0 && "must've put arguments in constant/nonconstant");
  }

  // This value is certainly an integer (and only and integer, not a pointer or
  // float). Therefore its value is constant
  //llvm::errs() << "VC: " << *val << " TR: " << TR.intType(val, false).str() << "\n";
  if (TR.intType(val, /*errIfNotFound*/ false).isIntegral()) {
    if (printconst)
      llvm::errs() << " Value const as integral " << (int)directions << " "
                   << *val << " "
                   << TR.intType(val, /*errIfNotFound*/ false).str() << "\n";
    constantvals.insert(val);
    return true;
  }

  // This value is certainly a pointer to an integer (and only and integer, not
  // a pointer or float). Therefore its value is constant
  // TODO use typeInfo for more aggressive activity analysis
  if (val->getType()->isPointerTy() &&
      cast<PointerType>(val->getType())->isIntOrIntVectorTy() &&
      TR.firstPointer(1, val, /*errifnotfound*/ false).isIntegral()) {
    if (printconst)
      llvm::errs() << " Value const as integral pointer" << (int)directions
                   << " " << *val << "\n";
    constantvals.insert(val);
    return true;
  }

  if (auto gi = dyn_cast<GlobalVariable>(val)) {
    if (!hasMetadata(gi, "enzyme_shadow") && nonmarkedglobals_inactive) {
      constantvals.insert(val);
      gi->setMetadata("enzyme_activity_value",
                      MDNode::get(gi->getContext(),
                                  MDString::get(gi->getContext(), "const")));
      return true;
    }
    // TODO consider this more
    if (gi->isConstant() &&
        isconstantValueM(TR, gi->getInitializer())) {
      constantvals.insert(val);
      gi->setMetadata("enzyme_activity_value",
                      MDNode::get(gi->getContext(),
                                  MDString::get(gi->getContext(), "const")));
      if (printconst)
        llvm::errs() << " VALUE const global " << *val << "\n";
      return true;
    }
    auto res = TR.query(gi).Data0();
    auto dt = res[{-1}];
    dt |= res[{0}];
    if (dt.isIntegral()) {
      if (printconst)
        llvm::errs() << " VALUE const as global int pointer " << *val
                     << " type - " << res.str() << "\n";
      return true;
    }
    if (printconst)
      llvm::errs() << " VALUE nonconst unknown global " << *val << " type - "
                   << res.str() << "\n";
    return false;
  }

  if (auto ce = dyn_cast<ConstantExpr>(val)) {
    if (ce->isCast()) {
      if (isconstantValueM(TR, ce->getOperand(0))) {
        if (printconst)
          llvm::errs() << " VALUE const cast from from operand " << *val
                       << "\n";
        constantvals.insert(val);
        return true;
      }
    }
    if (ce->isGEPWithNoNotionalOverIndexing()) {
      if (isconstantValueM(TR, ce->getOperand(0))) {
        if (printconst)
          llvm::errs() << " VALUE const cast from gep operand " << *val << "\n";
        constantvals.insert(val);
        return true;
      }
    }
    if (printconst)
      llvm::errs() << " VALUE nonconst unknown expr " << *val << "\n";
    return false;
  }


  std::shared_ptr<ActivityAnalyzer> UpHypothesis;
  
  // Handle types that could contain pointers
  //  Consider all types except
  //   * floating point types (since those are assumed not pointers)
  //   * integers that we know are not pointers
  bool containsPointer = true;
  if (val->getType()->isFPOrFPVectorTy())
    containsPointer = false;
  if (!TR.intType(val, /*errIfNotFound*/ false).isPossiblePointer())
    containsPointer = false;

  if (containsPointer) {


    auto TmpOrig =
    #if LLVM_VERSION_MAJOR >= 12
        getUnderlyingObject(val, 100);
    #else
        GetUnderlyingObject(val,
                            TR.info.Function->getParent()->getDataLayout(), 100);
    #endif 

    // If we know that our originator is constant from up,
    // we are definitionally constant
    if (directions & UP) {

      UpHypothesis = std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, UP));
      UpHypothesis->constantvals.insert(val);

      // If our origin is a load of a known inactive (say inactive argument), we are
      // also inactive
      if (auto LI = dyn_cast<LoadInst>(TmpOrig)) {

        if (directions == UP) {
          if (isconstantValueM(TR, LI->getPointerOperand())) {
            constantvals.insert(val);
            return true;
          }
        } else {
          if (UpHypothesis->isconstantValueM(TR, LI->getPointerOperand())) {
            constantvals.insert(val);
            insertConstantsFrom(*UpHypothesis);
            return true;
          }
        }
      }

      // otherwise if the origin is a previously derived known constant value
      // assess
      if (directions == UP && !isa<PHINode>(TmpOrig)) {
        if (TmpOrig != val && isconstantValueM(TR, TmpOrig)) {
          constantvals.insert(val);
          return true;
        }
      } else {
        if (TmpOrig != val && UpHypothesis->isconstantValueM(TR, TmpOrig)) {
          constantvals.insert(val);
          insertConstantsFrom(*UpHypothesis);
          return true;
        }
      }
    }

    // If not capable of looking at both users and uses, all the ways a pointer can be
    // loaded/stored cannot be assesed and therefore we default to assume it to be active
    if (directions != 3) {
      if (printconst)
        llvm::errs() << " <Potential Pointer assumed active at " << (int)directions << ">" << *val << "\n";
      retvals.insert(val);
      return false;
    }

    if (printconst)
      llvm::errs() << " < MEMSEARCH" << (int)directions << ">" << *val << "\n";
    // A pointer value is active if two things hold:
    //   an potentially active value is stored into the memory
    //   memory loaded from the value is used in an active way
    bool potentialStore = false;
    bool potentiallyActiveLoad = false;

    // Assume the value (not instruction) is itself active
    // In spite of that can we show that there are either no active stores
    // or no active loads
    std::shared_ptr<ActivityAnalyzer> Hypothesis = std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, directions));
    Hypothesis->retvals.insert(val);

    llvm::Function* ParentF = nullptr;
    if(auto inst = dyn_cast<Instruction>(val))
      ParentF = inst->getParent()->getParent();
    else if(auto arg = dyn_cast<Argument>(val))
      ParentF = arg->getParent();
    else {
      llvm::errs() << "unknown pointer value type: " << *val << "\n";
      assert(0 && "unknown pointer value type");
      llvm_unreachable("unknown pointer value type");
    }

    for(BasicBlock& BB : *ParentF) {
      if (potentialStore && potentiallyActiveLoad) break;
      for(Instruction& I : BB) {
        if (potentialStore && potentiallyActiveLoad) break;

        // If this is a malloc or free, this doesn't impact the activity
        if (auto CI = dyn_cast<CallInst>(&I)) {
          if (auto F = CI->getCalledFunction()) {
            if (isAllocationFunction(*F, TLI) || isDeallocationFunction(*F, TLI)) {
              continue;
            }
          }
        }

        Value* memval = val;

        // BasicAA stupidy assumes that non-pointer's don't alias
        // if this is a nonpointer, use something else to force alias
        // consideration
        if (!memval->getType()->isPointerTy()) {
          if (auto ci = dyn_cast<CastInst>(val)) {
            if (ci->getOperand(0)->getType()->isPointerTy()) {
              memval = ci->getOperand(0);
            }
          }
          for(auto user : val->users()) {
            if (isa<CastInst>(user) && user->getType()->isPointerTy()) {
              memval = user;
              break;
            }
          }
        }

        #if LLVM_VERSION_MAJOR >= 9
        auto AARes = AA.getModRefInfo(&I, MemoryLocation(memval, LocationSize::unknown()));
        #else
        auto AARes = AA.getModRefInfo(&I, MemoryLocation(memval, MemoryLocation::UnknownSize));
        #endif

        // Still having failed to replace, fall back to getModref against any location.
        if (!memval->getType()->isPointerTy()) {
          if (auto CB = dyn_cast<CallInst>(&I)) {
            AARes = createModRefInfo(AA.getModRefBehavior(CB));
          } else {
            bool mayRead = I.mayReadFromMemory();
            bool mayWrite = I.mayWriteToMemory();
            AARes = mayRead ? ( mayWrite ?  ModRefInfo::ModRef : ModRefInfo::Ref ) : ( mayWrite ? ModRefInfo::Mod : ModRefInfo::NoModRef);
          }
        }

        // TODO this aliasing information is too conservative, the question isn't merely aliasing
        // but whether there is a path for THIS value to eventually be loaded by it
        // not simply because there isnt aliasing

        // If we haven't already shown a potentially active load
        // check if this loads the given value and is active
        if (!potentiallyActiveLoad && isRefSet(AARes)) {
          if (printconst)
          llvm::errs() << "potential active load: " << I << "\n";
          if (auto LI = dyn_cast<LoadInst>(&I)) {
            // If the ref'ing value is a load check if the loaded value is active
            potentiallyActiveLoad = !Hypothesis->isconstantValueM(TR, LI);
          } else {
            // Otherwise fallback and check any part of the instruction is active
            // TODO: note that this can be optimized (especially for function calls)
            potentiallyActiveLoad = !Hypothesis->isconstantM(TR, &I);
          }
        }
        if (!potentialStore && isModSet(AARes)) {
          if (printconst)
          llvm::errs() << "potential active store: " << I << "\n";
          if (isa<StoreInst>(&I)) {
            // Stores don't need to be active to cause activity if an active load exists
            potentialStore = true;
          } else {
            // Otherwise fallback and check if the instruction is active
            // TODO: note that this can be optimized (especially for function calls)
            potentialStore = !Hypothesis->isconstantM(TR, &I);
          }
        }
      }
    }

    if (printconst)
      llvm::errs() << " </MEMSEARCH" << (int)directions << ">" << *val << " potentiallyActiveLoad=" << potentiallyActiveLoad << " potentialStore=" << potentialStore << "\n";
    if (potentiallyActiveLoad && potentialStore) {
      insertAllFrom(*Hypothesis);
      return false;
    } else {
      // We now know that there isn't a matching active load/store pair in this function
      // Now the only way that this memory can facilitate a transfer of active information
      // is if it is done outside of the function

      // This can happen if either:
      // a) the memory had an active load or store before this function was called
      // b) the memory had an active load or store after this function was called

      // Case a) can occur if:
      //    1) this memory came from an active global
      //    2) this memory came from an active argument
      //    3) this memory came from a load from active memory
      // In other words, assuming this value is inactive, going up this location's argument must be inactive

      assert(UpHypothesis);
      //UpHypothesis.constantvals.insert(val);
      UpHypothesis->insertConstantsFrom(*Hypothesis);
      assert(directions & UP);
      bool ActiveUp = !UpHypothesis->isInstructionInactiveFromOrigin(TR, val);

      //if (isa<AllocaInst>(TmpOrig)) {
      //  assert(!ActiveUp);
      //}
      //if (isCalledFunction(TmpOrig) && isAllocationFunction(*isCalledFunction(TmpOrig), TLI)) {
      //  assert(!ActiveUp);
      //}

      // Case b) can occur if:
      //    1) this memory is used as part of an active return
      //    2) this memory is stored somewhere

      // TODO we never verify that an origin wasn't stored somewhere or returned.
      // to remedy correctness for now let's do something extremely simple
      std::shared_ptr<ActivityAnalyzer> DownHypothesis = std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, DOWN));
      DownHypothesis->constantvals.insert(val);
      DownHypothesis->insertConstantsFrom(*Hypothesis);
      // TODO this is too conservative and will say stored when stored into, rahter than we store this
      bool ActiveDown = DownHypothesis->isValueActivelyStoredOrReturned(TR, val);
      // BEGIN TEMPORARY

      if (!ActiveDown && TmpOrig != val) {

        if (isa<Argument>(TmpOrig) || isa<GlobalVariable>(TmpOrig) || isa<AllocaInst>(TmpOrig) || (isCalledFunction(TmpOrig) && isAllocationFunction(*isCalledFunction(TmpOrig), TLI))) {
          std::shared_ptr<ActivityAnalyzer> DownHypothesis2 = std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*DownHypothesis, DOWN));
          DownHypothesis2->constantvals.insert(TmpOrig);
          if (DownHypothesis2->isValueActivelyStoredOrReturned(TR, TmpOrig)) {
            if (printconst)
            llvm::errs() << " active from ivasor: " << *TmpOrig << "\n";
            ActiveDown = true;
          }
        } else {
          // unknown origin that could've been stored/returned/etc
          if (printconst)
          llvm::errs() << " active from unknown origin: " << *TmpOrig << "\n";
          ActiveDown = true;
        }
      }

      //END TEMPORARY
      
      // We can now consider the three places derivative information can be transferred
      //   Case A) From the origin
      //   Case B) Though the return
      //   Case C) Within the function (via either load or store)

      bool ActiveMemory = false;

      // If it is transferred via active origin and return, clearly this is active
      ActiveMemory |= (ActiveUp && ActiveDown);

      // If we come from an active origin and load, memory is clearly active
      ActiveMemory |= (ActiveUp && potentiallyActiveLoad);

      // If we come from an active origin and only store into it, it changes future state
      ActiveMemory |= (ActiveUp && potentialStore);

      // If we go to an active return and store active memory, this is active
      ActiveMemory |= (ActiveDown && potentialStore);
      // Actually more generally, if we are ActiveDown (returning memory that is used)
      // in active return, we must be active. This is necessary to ensure mallocs
      // have their differential shadows created when returned [TODO investigate more]
      ActiveMemory |= ActiveDown;

      // If we go to an active return and only load it, however, that doesnt
      // transfer derivatives and we can say this memory is inactive

      if (printconst)
        llvm::errs() << " @@MEMSEARCH" << (int)directions << ">" << *val << " potentiallyActiveLoad=" << potentiallyActiveLoad << " potentialStore=" << potentialStore << " ActiveUp=" << ActiveUp << " ActiveDown=" << ActiveDown << " ActiveMemory=" << ActiveMemory << "\n";

      if (ActiveMemory) {
        retvals.insert(val);
        assert(Hypothesis->directions == directions);
        assert(Hypothesis->retvals.count(val));
        insertAllFrom(*Hypothesis);
        return false;
      } else {
        constantvals.insert(val);
        insertConstantsFrom(*Hypothesis);
        insertConstantsFrom(*UpHypothesis);
        insertConstantsFrom(*DownHypothesis);
        return true;
      }
    }
  }

  // For all non-pointers, it is now sufficient to simply prove that
  // either activity does not flow in, or activity does not flow out
  // This alone cuts off the flow (being unable to flow through memory)

  // Not looking at uses to prove inactive (definition of up), if the creator of this value
  // is inactive, we are inactive
  // Since we won't look at uses to prove, we can inductively assume this is inactive
  if (directions & UP) {

    if (directions == UP && !isa<PHINode>(val)) {
      if (isInstructionInactiveFromOrigin(TR, val)) {
        constantvals.insert(val);
        return true;
      }
    } else {
      UpHypothesis = std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, UP));
      UpHypothesis->constantvals.insert(val);
      if (UpHypothesis->isInstructionInactiveFromOrigin(TR, val)) {
        insertConstantsFrom(*UpHypothesis);
        constantvals.insert(val);
        return true;
      }
    }


  }

  if (directions & DOWN) {
    // Not looking at users to prove inactive (definition of down)
    // If all users are inactive, this is therefore inactive. 
    // Since we won't look at origins to prove, we can inductively assume this is inactive
    
    // As an optimization if we are going down already
    // and we won't use ourselves (done by PHI's), we
    // dont need to inductively assume we're true
    // and can instead use this object!
    if (directions == DOWN && !isa<PHINode>(val)) {
      if (isValueInactiveFromUsers(TR, val)) {
        if (UpHypothesis)
          insertConstantsFrom(*UpHypothesis);
        constantvals.insert(val);
        return true;
      }
    } else {
      auto DownHypothesis = std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, DOWN));
      DownHypothesis->constantvals.insert(val);
      if (DownHypothesis->isValueInactiveFromUsers(TR, val)) {
        insertConstantsFrom(*DownHypothesis);
        if (UpHypothesis)
          insertConstantsFrom(*UpHypothesis);
        constantvals.insert(val);
        return true;
      }
    }


  }

  if (printconst)
    llvm::errs() << " Value nonconstant (couldn't disprove)[" << (int)directions
                  << "]" << *val << "\n";
  retvals.insert(val);
  return false;
}

bool ActivityAnalyzer::isInstructionInactiveFromOrigin(TypeResults &TR, llvm::Value* val) {
  // Must be an analyzer only searching up
  assert(directions == UP);
  assert(!isa<Argument>(val));
  assert(!isa<GlobalVariable>(val));

  if (auto inst = dyn_cast<Instruction>(val)) {
    if (printconst)
      llvm::errs() << " < UPSEARCH" << (int)directions << ">" << *inst
                    << "\n";

    if (auto call = dyn_cast<CallInst>(inst)) {
      #if LLVM_VERSION_MAJOR >= 11
      if (auto iasm = dyn_cast<InlineAsm>(call->getCalledOperand())) {
      #else
      if (auto iasm = dyn_cast<InlineAsm>(call->getCalledValue())) {
      #endif
        if (iasm->getAsmString() == "cpuid") {
          if (printconst)
            llvm::errs() << " constant instruction from known cpuid instruction "
                        << *inst << "\n";
          return true;
        }
      }
    }
      if (auto op = dyn_cast<CallInst>(inst)) {
      if (auto called = op->getCalledFunction()) {
        if (called->getName() == "printf" || called->getName() == "puts" ||
            called->getName() == "__assert_fail" || called->getName() == "free" ||
            called->getName() == "_ZdlPv" || called->getName() == "_ZdlPvm" ||
            called->getName() == "__cxa_guard_acquire" ||
            called->getName() == "__cxa_guard_release" ||
            called->getName() == "__cxa_guard_abort") {
          return true;
        }
        if (!isCertainPrintMallocOrFree(called) && called->empty() &&
            !hasMetadata(called, "enzyme_gradient") && !isa<IntrinsicInst>(op) &&
            emptyfnconst) {
          return true;
        }
      }
    }

    if (auto op = dyn_cast<IntrinsicInst>(inst)) {
      switch (op->getIntrinsicID()) {
      case Intrinsic::assume:
      case Intrinsic::stacksave:
      case Intrinsic::stackrestore:
      case Intrinsic::lifetime_start:
      case Intrinsic::lifetime_end:
      case Intrinsic::dbg_addr:
      case Intrinsic::dbg_declare:
      case Intrinsic::dbg_value:
      case Intrinsic::invariant_start:
      case Intrinsic::invariant_end:
      case Intrinsic::var_annotation:
      case Intrinsic::ptr_annotation:
      case Intrinsic::annotation:
      case Intrinsic::codeview_annotation:
      case Intrinsic::expect:
      case Intrinsic::type_test:
      case Intrinsic::donothing:
        // case Intrinsic::is_constant:
        return true;
      default:
        break;
      }
    }


    if (auto gep = dyn_cast<GetElementPtrInst>(inst)) {
      if (isconstantValueM(TR, gep->getPointerOperand())) {
        if (printconst)
          llvm::errs() << "constant(" << (int)directions << ") up-gep "
                        << *inst << "\n";
        return true;
      }
      return false;
    } else if (auto ci = dyn_cast<CallInst>(inst)) {
      bool seenuse = false;

      propagateArgumentInformation(*ci, [&](Value *a) {
        if (!isconstantValueM(TR, a)) {
          seenuse = true;
          if (printconst)
            llvm::errs() << "nonconstant(" << (int)directions << ")  up-call "
                          << *inst << " op " << *a << "\n";
          return true;
        }
        return false;
      });

      //! TODO consider calling interprocedural here
      //! TODO: Really need an attribute that determines whether a function
      //! can access a global (not even necessarily read)
      // if (ci->hasFnAttr(Attribute::ReadNone) ||
      // ci->hasFnAttr(Attribute::ArgMemOnly))
      if (!seenuse) {
        if (printconst)
          llvm::errs() << "constant(" << (int)directions
                        << ")  up-call:" << *inst << "\n";
        return true;
      }
      return !seenuse;
    } else if (auto si = dyn_cast<SelectInst>(inst)) {

      if ( isconstantValueM(TR, si->getTrueValue()) &&
           isconstantValueM(TR, si->getFalseValue())) {

        if (printconst)
          llvm::errs() << "constant(" << (int)directions
                        << ") up-sel:" << *inst << "\n";
        return true;
      }

    } else {
      bool seenuse = false;
      //! TODO does not consider reading from global memory that is active and not an argument
      for (auto &a : inst->operands()) {
        bool hypval = isconstantValueM(TR, a);
        if (!hypval) {
          if (printconst)
            llvm::errs() << "nonconstant(" << (int)directions << ")  up-inst "
                          << *inst << " op " << *a << "\n";
          seenuse = true;
          break;
          // return false;
        }
      }

      if (!seenuse) {
        if (printconst)
          llvm::errs() << "constant(" << (int)directions
                        << ")  up-inst:" << *inst << "\n";
        return true;
      }
      return false;
    }
  } else {
    llvm::errs() << "unknown pointer source: " << *val << "\n";
    assert(0 && "unknown pointer source");
    llvm_unreachable("unknown pointer source");
  }

  return false;
}

bool ActivityAnalyzer::isValueInactiveFromUsers(TypeResults &TR, llvm::Value* val) {
  // Must be an analyzer only searching down
  assert(directions == DOWN);
  // To ensure we can call down

  if (printconst)
    llvm::errs() << " <Value USESEARCH" << (int)directions << ">" << *val
                  << "\n";

  bool seenuse = false;

  for (const auto a : val->users()) {

    if (printconst)
      llvm::errs() << "      considering use of " << *val << " - " << *a
                    << "\n";

    if (isa<AllocaInst>(a)) {
      if (printconst)
        llvm::errs() << "found constant(" << (int)directions
                      << ")  allocainst use:" << *val << " user " << *a
                      << "\n";
      continue;
    }
    if (isa<ReturnInst>(a)) {
      return !ActiveReturns;
    }

    if (auto call = dyn_cast<CallInst>(a)) {
      bool ConstantArg = isFunctionArgumentConstant(call, val);
      if (ConstantArg) {
        if (printconst) {
          llvm::errs() << "Value found constant callinst use:" << *val
                        << " user " << *call << "\n";
        }
        continue;
      }
    }

    // is constant instruction is insufficient since while the instr itself
    // may not propagate gradients the return of the instruction may be used
    // for additional gradient propagation
    bool ConstantInst = isconstantM(TR, cast<Instruction>(a)) && isconstantValueM(TR, a);

    if (!ConstantInst) {
      if (printconst)
        llvm::errs() << "Value nonconstant inst (uses):" << *val << " user "
                      << *a << "\n";
      seenuse = true;
      break;
    } else {
      if (printconst)
        llvm::errs() << "Value found constant inst use:" << *val << " user "
                      << *a << "\n";
    }
  }

  if (printconst)
    llvm::errs() << " </Value USESEARCH" << (int)directions << " const=" << (!seenuse) << ">" << *val << "\n";
  return !seenuse;
}

bool ActivityAnalyzer::isValueActivelyStoredOrReturned(TypeResults &TR, llvm::Value* val) {
  // Must be an analyzer only searching down
  assert(directions == DOWN);

  if(StoredOrReturnedCache.find(val) != StoredOrReturnedCache.end()) {
    return StoredOrReturnedCache[val];
  }

  if (printconst)
    llvm::errs() << " <ASOR" << (int)directions << ">" << *val
                  << "\n";


  StoredOrReturnedCache[val] = false;

  for (const auto a : val->users()) {
    if (isa<AllocaInst>(a)) {
      continue;
    }
    // Loading a value prevents its pointer from being captured
    if (isa<LoadInst>(a)) {
      continue;
    }

    if (isa<ReturnInst>(a)) {
      if (!ActiveReturns) continue;

      if (printconst)
      llvm::errs() << " </ASOR" << (int)directions << " active from-ret>" << *val
                    << "\n";
      StoredOrReturnedCache[val] = true;                    
      return true;
    }

    if (auto call = dyn_cast<CallInst>(a)) {
      if (!couldFunctionArgumentCapture(call, val)) {
        continue;
      }
      bool ConstantArg = isFunctionArgumentConstant(call, val);
      if (ConstantArg) {
        continue;
      }
    }

    if (auto SI = dyn_cast<StoreInst>(a)) {
      // If we are being stored into, not storing this value
      // this case can be skipped
      if (SI->getValueOperand() != val) {
        continue;
      }
      // Storing into active memory, return true
      if (!isconstantValueM(TR, SI->getPointerOperand())) {
        StoredOrReturnedCache[val] = true;
        if (printconst)
        llvm::errs() << " </ASOR" << (int)directions << " active from-store>" << *val
                      << " store=" << *SI << "\n";
        return true;
      }
    }

    if (auto inst = dyn_cast<Instruction>(a)) {
      if (!inst->mayWriteToMemory() ||
        (isa<CallInst>(inst) && AA.onlyReadsMemory(cast<CallInst>(inst)))) {
          // if not written to memory and returning a known constant, this
          // cannot be actively returned/stored
          if (isconstantValueM(TR, a)) {
            continue;
          }
          // if not written to memory and returning a value itself
          // not actively stored or returned, this is not actively
          // stored or returned
          if (!isValueActivelyStoredOrReturned(TR, a)) {
            continue;
          }
      }
    }

    if (auto F = isCalledFunction(a)) {
      if (isAllocationFunction(*F, TLI)) {
        // if not written to memory and returning a known constant, this
        // cannot be actively returned/stored
        if (isconstantValueM(TR, a)) {
          continue;
        }
        // if not written to memory and returning a value itself
        // not actively stored or returned, this is not actively
        // stored or returned
        if (!isValueActivelyStoredOrReturned(TR, a)) {
          continue;
        }
      } else if(isDeallocationFunction(*F, TLI)) {
        //freeing memory never counts
        continue;
      }
    }
    // fallback and conservatively assume that if the value is written to
    // it is written to active memory
    // TODO handle more memory instructions above to be less conservative

    if (printconst)
    llvm::errs() << " </ASOR" << (int)directions << " active from-unknown>" << *val
                  << " - use=" << *a << "\n";
    return StoredOrReturnedCache[val] = true;
  }

  if (printconst)
  llvm::errs() << " </ASOR" << (int)directions << " inactive>" << *val
               << "\n";
  return false;
}

