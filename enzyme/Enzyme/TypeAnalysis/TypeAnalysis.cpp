//===- TypeAnalysis.cpp - Implementation of Type Analysis   ------------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of Type Analysis, a utility for
// computing the underlying data type of LLVM values.
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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/InlineAsm.h"

#include "../Utils.h"
#include "TypeAnalysis.h"

#include "../LibraryFuncs.h"

#include "TBAA.h"

llvm::cl::opt<bool> PrintType("enzyme-print-type", cl::init(false), cl::Hidden,
                              cl::desc("Print type analysis algorithm"));

TypeAnalyzer::TypeAnalyzer(const FnTypeInfo &fn, TypeAnalysis &TA,
                           uint8_t direction)
    : intseen(), fntypeinfo(fn), interprocedural(TA), direction(direction),
      Invalid(false), DT(*fn.Function) {

  assert(fntypeinfo.KnownValues.size() ==
         fntypeinfo.Function->getFunctionType()->getNumParams());

  // Add all instructions in the function
  for (BasicBlock &BB : *fntypeinfo.Function) {
    for (Instruction &I : BB) {
      workList.push_back(&I);
    }
  }
  // Add all operands referenced in the function
  // This is done to investigate any referenced globals/etc
  for (BasicBlock &BB : *fntypeinfo.Function) {
    for (Instruction &I : BB) {
      for (auto &Op : I.operands()) {
        addToWorkList(Op);
      }
    }
  }
}

/// Given a constant value, deduce any type information applicable
TypeTree getConstantAnalysis(Constant *Val, const FnTypeInfo &nfti,
                             TypeAnalysis &TA) {
  auto &DL = nfti.Function->getParent()->getDataLayout();
  // Undefined value is an anything everywhere
  if (isa<UndefValue>(Val) || isa<ConstantAggregateZero>(Val)) {
    return TypeTree(BaseType::Anything).Only(-1);
  }

  // Null pointer is a pointer to anything, everywhere
  if (isa<ConstantPointerNull>(Val)) {
    TypeTree Result(BaseType::Pointer);
    Result |= TypeTree(BaseType::Anything).Only(-1);
    return Result.Only(-1);
  }

  // Known pointers are pointers at offset 0
  if (isa<Function>(Val) || isa<BlockAddress>(Val)) {
    return TypeTree(BaseType::Pointer).Only(-1);
  }

  // Type of an aggregate is the aggregation of
  // the subtypes
  if (auto CA = dyn_cast<ConstantAggregate>(Val)) {
    TypeTree Result;
    int Off = 0;
    for (unsigned i = 0, size = CA->getNumOperands(); i < size; ++i) {
      assert(nfti.Function);
      auto Op = CA->getOperand(i);
      // TODO check this for i1 constant aggregates packing/etc
      auto ObjSize =
          (nfti.Function->getParent()->getDataLayout().getTypeSizeInBits(
               Op->getType()) +
           7) /
          8;
      Result |= getConstantAnalysis(Op, nfti, TA)
                    .ShiftIndices(DL, /*init offset*/ 0, /*maxSize*/ ObjSize,
                                  /*addOffset*/ Off);
      Off += ObjSize;
    }
    return Result;
  }

  // Type of an sequence is the aggregation of
  // the subtypes
  if (auto CD = dyn_cast<ConstantDataSequential>(Val)) {
    TypeTree Result;
    int Off = 0;
    for (unsigned i = 0, size = CD->getNumElements(); i < size; ++i) {
      assert(nfti.Function);
      auto Op = CD->getElementAsConstant(i);
      // TODO check this for i1 constant aggregates packing/etc
      auto ObjSize =
          (nfti.Function->getParent()->getDataLayout().getTypeSizeInBits(
               Op->getType()) +
           7) /
          8;
      Result |= getConstantAnalysis(Op, nfti, TA)
                    .ShiftIndices(DL, /*init offset*/ 0, /*maxSize*/ ObjSize,
                                  /*addOffset*/ Off);
      Off += size;
    }
    return Result;
  }

  if (isa<ConstantData>(Val)) {
    // Any constants == 0 are considered Anything
    // other floats are assumed to be that type
    if (auto FP = dyn_cast<ConstantFP>(Val)) {
      if (FP->isExactlyValue(0.0))
        return TypeTree(BaseType::Anything).Only(-1);
      return TypeTree(FP->getType()).Only(-1);
    }

    if (auto ci = dyn_cast<ConstantInt>(Val)) {
      // Constants in range [1, 4096] are assumed to be integral since
      // any float or pointers they may represent are ill-formed
      if (!ci->isNegative() && ci->getLimitedValue() >= 1 &&
          ci->getLimitedValue() <= 4096) {
        return TypeTree(ConcreteType(BaseType::Integer)).Only(-1);
      }

      // Constants explicitly marked as negative that aren't -1 are considered
      // integral
      if (ci->isNegative() && ci->getSExtValue() < -1) {
        return TypeTree(ConcreteType(BaseType::Integer)).Only(-1);
      }

      // Values of size < 16 (half size) are considered integral
      // since they cannot possibly represent a float or pointer
      if (ci->getType()->getBitWidth() < 16) {
        return TypeTree(ConcreteType(BaseType::Integer)).Only(-1);
      }
      // All other constant-ints could be any type
      return TypeTree(BaseType::Anything).Only(-1);
    }
  }

  // ConstantExprs are handled by considering the
  // equivalent instruction
  if (auto CE = dyn_cast<ConstantExpr>(Val)) {
    TypeTree Result;

    auto I = CE->getAsInstruction();
    I->insertBefore(nfti.Function->getEntryBlock().getTerminator());

    // Just analyze this new "instruction" and none of the others
    {
      TypeAnalyzer tmpAnalysis(nfti, TA);
      tmpAnalysis.workList.clear();
      tmpAnalysis.visit(*I);
      Result = tmpAnalysis.getAnalysis(I);
    }

    I->eraseFromParent();
    return Result;
  }

  if (auto GV = dyn_cast<GlobalVariable>(Val)) {
    // A fixed constant global is a pointer to its initializer
    if (GV->isConstant() && GV->hasInitializer()) {
      TypeTree Result = ConcreteType(BaseType::Pointer);
      Result |= getConstantAnalysis(GV->getInitializer(), nfti, TA);
      return Result.Only(-1);
    }
    auto globalSize = DL.getTypeSizeInBits(GV->getValueType()) / 8;
    // Since halfs are 16bit (2 byte) and pointers are >=32bit (4 byte) any
    // Single byte object must be integral
    if (globalSize == 1) {
      TypeTree Result = ConcreteType(BaseType::Pointer);
      Result |= TypeTree(ConcreteType(BaseType::Integer)).Only(-1);
      return Result.Only(-1);
    }
    // Otherwise, we simply know that this is a pointer, and
    // not what it is a pointer to
    return TypeTree(BaseType::Pointer).Only(-1);
  }

  // No other information can be ascertained
  return TypeTree();
}

TypeTree TypeAnalyzer::getAnalysis(Value *Val) {
  // Integers with fewer than 16 bits (size of half)
  // must be integral, since it cannot possibly represent a float or pointer
  if (!isa<UndefValue>(Val) && Val->getType()->isIntegerTy() &&
      cast<IntegerType>(Val->getType())->getBitWidth() < 16)
    return TypeTree(ConcreteType(BaseType::Integer)).Only(-1);
  if (auto C = dyn_cast<Constant>(Val)) {
    TypeTree result = getConstantAnalysis(C, fntypeinfo, interprocedural);
    if (auto found = findInMap(analysis, Val)) {
      result |= *found;
      *found = result;
    }
    return result;
  }

  // Check that this value is from the function being analyzed
  if (auto I = dyn_cast<Instruction>(Val)) {
    if (I->getParent()->getParent() != fntypeinfo.Function) {
      llvm::errs() << " function: " << *fntypeinfo.Function << "\n";
      llvm::errs() << " instParent: " << *I->getParent()->getParent() << "\n";
      llvm::errs() << " inst: " << *I << "\n";
    }
    assert(I->getParent()->getParent() == fntypeinfo.Function);
  }
  if (auto Arg = dyn_cast<Argument>(Val)) {
    if (Arg->getParent() != fntypeinfo.Function) {
      llvm::errs() << " function: " << *fntypeinfo.Function << "\n";
      llvm::errs() << " argParent: " << *Arg->getParent() << "\n";
      llvm::errs() << " arg: " << *Arg << "\n";
    }
    assert(Arg->getParent() == fntypeinfo.Function);
  }

  // Return current results
  if (isa<Argument>(Val) || isa<Instruction>(Val))
    return analysis[Val];

  // Unhandled/unknown Value
  llvm::errs() << "Error Unknown Value: " << *Val << "\n";
  assert(0 && "Error Unknown Value: ");
  llvm_unreachable("Error Unknown Value: ");
  // return TypeTree();
}

void TypeAnalyzer::updateAnalysis(Value *Val, ConcreteType Data,
                                  Value *Origin) {
  updateAnalysis(Val, TypeTree(Data), Origin);
}

void TypeAnalyzer::updateAnalysis(Value *Val, BaseType Data, Value *Origin) {
  updateAnalysis(Val, TypeTree(ConcreteType(Data)), Origin);
}

void TypeAnalyzer::addToWorkList(Value *Val) {
  // Only consider instructions/arguments
  if (!isa<Instruction>(Val) && !isa<Argument>(Val) &&
      !isa<ConstantExpr>(Val) && !isa<GlobalVariable>(Val))
    return;

  // Don't add this value to list twice
  if (std::find(workList.begin(), workList.end(), Val) != workList.end())
    return;

  // Verify this value comes from the function being analyzed
  if (auto I = dyn_cast<Instruction>(Val)) {
    if (fntypeinfo.Function != I->getParent()->getParent())
      return;
    if (fntypeinfo.Function != I->getParent()->getParent()) {
      llvm::errs() << "function: " << *fntypeinfo.Function << "\n";
      llvm::errs() << "instf: " << *I->getParent()->getParent() << "\n";
      llvm::errs() << "inst: " << *I << "\n";
    }
    assert(fntypeinfo.Function == I->getParent()->getParent());
  } else if (auto Arg = dyn_cast<Argument>(Val))
    assert(fntypeinfo.Function == Arg->getParent());

  // Add to workList
  workList.push_back(Val);
}

void TypeAnalyzer::updateAnalysis(Value *Val, TypeTree Data, Value *Origin) {
  // ConstantData's and Functions don't have analysis updated
  // We don't do "Constant" as globals are "Constant" types
  if (isa<ConstantData>(Val) || isa<Function>(Val)) {
    return;
  }

  // Print the update being made, if requested
  if (PrintType) {
    llvm::errs() << "updating analysis of val: " << *Val
                 << " current: " << analysis[Val].str() << " new "
                 << Data.str();
    if (Origin)
      llvm::errs() << " from " << *Origin;
    llvm::errs() << "\n";
  }

  if (auto I = dyn_cast<Instruction>(Val)) {
    if (fntypeinfo.Function != I->getParent()->getParent()) {
      llvm::errs() << "function: " << *fntypeinfo.Function << "\n";
      llvm::errs() << "instf: " << *I->getParent()->getParent() << "\n";
      llvm::errs() << "inst: " << *I << "\n";
    }
    assert(fntypeinfo.Function == I->getParent()->getParent());
  } else if (auto Arg = dyn_cast<Argument>(Val))
    assert(fntypeinfo.Function == Arg->getParent());

  bool pointerUse = false;
  if (Instruction *I = dyn_cast<Instruction>(Val)) {
    for (auto user : Val->users()) {
      if (isa<LoadInst>(user)) {
        pointerUse = true;
        break;
      }
      if (auto SI = dyn_cast<StoreInst>(user)) {
        if (SI->getPointerOperand() == Val) {
          pointerUse = true;
          break;
        }
      }
      if (auto GEP = dyn_cast<GetElementPtrInst>(user)) {
        if (GEP->getPointerOperand() == Val) {
          pointerUse = true;
          break;
        }
      }
    }
    if (isa<GetElementPtrInst>(I))
      pointerUse = true;
  }

  // If this is a deinite ptr type and we find instead this is an integer, error
  // early This is unlikely to occur in real codes and a very good way to
  // identify TypeAnalysis bugs
  if (pointerUse && Data.Inner0() == BaseType::Integer) {
    if (direction != BOTH) {
      Invalid = true;
      return;
    }
    llvm::errs() << *fntypeinfo.Function << "\n";
    dump();
    llvm::errs() << "illegal ptr update for val: " << *Val << "\n";
    if (Origin)
      llvm::errs() << " + " << *Origin << "\n";
    assert(0 && "illegal ptr update");
  }

  // Attempt to update the underlying analysis
  bool LegalOr = true;
  bool Changed =
      analysis[Val].checkedOrIn(Data, /*PointerIntSame*/ false, LegalOr);

  if (!LegalOr) {
    if (direction != BOTH) {
      Invalid = true;
      return;
    }
    llvm::errs() << *fntypeinfo.Function << "\n";
    dump();
    llvm::errs() << "Illegal updateAnalysis prev:" << analysis[Val].str()
                 << " new: " << Data.str() << "\n";
    llvm::errs() << "val: " << *Val;
    if (Origin)
      llvm::errs() << " origin=" << *Origin;
    llvm::errs() << "\n";
    assert(0 && "Performed illegal updateAnalysis");
    llvm_unreachable("Performed illegal updateAnalysis");
  }

  if (Changed) {
    // Add val so it can explicitly propagate this new info, if able to
    if (Val != Origin)
      addToWorkList(Val);

    // Add users and operands of the value so they can update from the new
    // operand/use
    for (User *U : Val->users()) {
      if (U != Origin) {

        if (auto I = dyn_cast<Instruction>(U)) {
          if (fntypeinfo.Function != I->getParent()->getParent()) {
            continue;
          }
        }

        addToWorkList(U);

        // per the handling of phi's
        if (auto BO = dyn_cast<BinaryOperator>(U)) {
          for (User *U2 : BO->users()) {
            if (isa<PHINode>(U2) && U2 != Origin) {
              addToWorkList(U2);
            }
          }
        }
      }
    }

    if (User *US = dyn_cast<User>(Val)) {
      for (Value *Op : US->operands()) {
        if (Op != Origin) {
          addToWorkList(Op);
        }
      }
    }
  }
}

/// Analyze type info given by the arguments, possibly adding to work queue
void TypeAnalyzer::prepareArgs() {
  // Propagate input type information for arguments
  for (auto &pair : fntypeinfo.Arguments) {
    assert(pair.first->getParent() == fntypeinfo.Function);
    updateAnalysis(pair.first, pair.second, nullptr);
  }

  // Get type and other information about argument
  // getAnalysis may add more information so this
  // is necessary/useful
  for (Argument &Arg : fntypeinfo.Function->args()) {
    updateAnalysis(&Arg, getAnalysis(&Arg), &Arg);
  }

  // Propagate return value type information
  for (BasicBlock &BB : *fntypeinfo.Function) {
    for (Instruction &I : BB) {
      if (ReturnInst *RI = dyn_cast<ReturnInst>(&I)) {
        if (Value *RV = RI->getReturnValue()) {
          updateAnalysis(RV, fntypeinfo.Return, nullptr);
          updateAnalysis(RV, getAnalysis(RV), RV);
        }
      }
    }
  }
}

/// Analyze type info given by the TBAA, possibly adding to work queue
void TypeAnalyzer::considerTBAA() {
  auto &DL = fntypeinfo.Function->getParent()->getDataLayout();

  for (BasicBlock &BB : *fntypeinfo.Function) {
    for (Instruction &I : BB) {

      if (CallInst *call = dyn_cast<CallInst>(&I)) {
        if (call->getCalledFunction() &&
            call->getCalledFunction()->getName() == "__enzyme_float") {
          assert(call->getNumArgOperands() == 2);
          assert(call->getArgOperand(0)->getType()->isPointerTy());
          assert(isa<ConstantInt>(call->getArgOperand(1)));
          TypeTree TT;
          for (size_t i = 0;
               i < cast<ConstantInt>(call->getArgOperand(1))->getLimitedValue();
               i += 4)
            TT.insert({(int)i}, Type::getFloatTy(call->getContext()));
          TT.insert({}, BaseType::Pointer);
          updateAnalysis(call->getOperand(0), TT.Only(-1), call);
        }
        if (call->getCalledFunction() &&
            call->getCalledFunction()->getName() == "__enzyme_double") {
          assert(call->getNumArgOperands() == 2);
          assert(call->getArgOperand(0)->getType()->isPointerTy());
          assert(isa<ConstantInt>(call->getArgOperand(1)));
          TypeTree TT;
          for (size_t i = 0;
               i < cast<ConstantInt>(call->getArgOperand(1))->getLimitedValue();
               i += 8)
            TT.insert({(int)i}, Type::getDoubleTy(call->getContext()));
          TT.insert({}, BaseType::Pointer);
          updateAnalysis(call->getOperand(0), TT.Only(-1), call);
        }
        if (call->getCalledFunction() &&
            call->getCalledFunction()->getName() == "__enzyme_integer") {
          assert(call->getNumArgOperands() == 2);
          assert(call->getArgOperand(0)->getType()->isPointerTy());
          assert(isa<ConstantInt>(call->getArgOperand(1)));
          TypeTree TT;
          for (size_t i = 0;
               i < cast<ConstantInt>(call->getArgOperand(1))->getLimitedValue();
               i++)
            TT.insert({(int)i}, BaseType::Integer);
          TT.insert({}, BaseType::Pointer);
          updateAnalysis(call->getOperand(0), TT.Only(-1), call);
        }
        if (call->getCalledFunction() &&
            call->getCalledFunction()->getName() == "__enzyme_pointer") {
          assert(call->getNumArgOperands() == 2);
          assert(call->getArgOperand(0)->getType()->isPointerTy());
          assert(isa<ConstantInt>(call->getArgOperand(1)));
          TypeTree TT;
          for (size_t i = 0;
               i < cast<ConstantInt>(call->getArgOperand(1))->getLimitedValue();
               i += ((DL.getPointerSizeInBits() + 7) / 8))
            TT.insert({(int)i}, BaseType::Pointer);
          TT.insert({}, BaseType::Pointer);
          updateAnalysis(call->getOperand(0), TT.Only(-1), call);
        }
      }

      TypeTree vdptr = parseTBAA(I, DL);

      // If we don't have any useful information,
      // don't bother updating
      if (!vdptr.isKnownPastPointer())
        continue;

      if (CallInst *call = dyn_cast<CallInst>(&I)) {
        if (call->getCalledFunction() &&
            (call->getCalledFunction()->getIntrinsicID() == Intrinsic::memcpy ||
             call->getCalledFunction()->getIntrinsicID() ==
                 Intrinsic::memmove)) {
          int64_t copySize = 1;
          for (auto val : fntypeinfo.knownIntegralValues(call->getOperand(2),
                                                         DT, intseen)) {
            copySize = max(copySize, val);
          }
          TypeTree update =
              vdptr.ShiftIndices(DL, /*init offset*/ 0,
                                 /*max size*/ copySize, /*new offset*/ 0);

          updateAnalysis(call->getOperand(0), update.Only(-1), call);
          updateAnalysis(call->getOperand(1), update.Only(-1), call);
          continue;
        } else if (call->getType()->isPointerTy()) {
          updateAnalysis(call, vdptr.Only(-1), call);
        } else {
          llvm::errs() << " inst: " << I << " vdptr: " << vdptr.str() << "\n";
          assert(0 && "unknown tbaa call instruction user");
        }
      } else if (auto SI = dyn_cast<StoreInst>(&I)) {
        auto StoreSize =
            (DL.getTypeSizeInBits(SI->getValueOperand()->getType()) + 7) / 8;
        updateAnalysis(SI->getPointerOperand(),
                       vdptr
                           // Cut off any values outside of store
                           .ShiftIndices(DL, /*init offset*/ 0,
                                         /*max size*/ StoreSize,
                                         /*new offset*/ 0)
                           // Don't propagate "Anything" into ptr
                           .PurgeAnything()
                           .Only(-1),
                       SI);
        TypeTree req = vdptr.Only(-1);
        updateAnalysis(SI->getValueOperand(), req.Lookup(StoreSize, DL), SI);
      } else if (auto LI = dyn_cast<LoadInst>(&I)) {
        auto LoadSize = (DL.getTypeSizeInBits(LI->getType()) + 7) / 8;
        updateAnalysis(LI->getPointerOperand(),
                       vdptr
                           // Cut off any values outside of load
                           .ShiftIndices(DL, /*init offset*/ 0,
                                         /*max size*/ LoadSize,
                                         /*new offset*/ 0)
                           // Don't propagate "Anything" into ptr
                           .PurgeAnything()
                           .Only(-1),
                       LI);
        TypeTree req = vdptr.Only(-1);
        updateAnalysis(LI, req.Lookup(LoadSize, DL), LI);
      } else {
        llvm::errs() << " inst: " << I << " vdptr: " << vdptr.str() << "\n";
        assert(0 && "unknown tbaa instruction user");
        llvm_unreachable("unknown tbaa instruction user");
      }
    }
  }
}

void TypeAnalyzer::runPHIHypotheses() {
  bool Changed;
  do {
    Changed = false;
    for (BasicBlock &BB : *fntypeinfo.Function) {
      for (Instruction &inst : BB) {
        if (PHINode *phi = dyn_cast<PHINode>(&inst)) {
          if (direction & DOWN && phi->getType()->isIntOrIntVectorTy() &&
              !getAnalysis(phi).isKnown()) {
            // Assume that this is an integer, does that mean we can prove that
            // the incoming operands are integral

            TypeAnalyzer tmpAnalysis(fntypeinfo, interprocedural, DOWN);
            tmpAnalysis.workList.clear();
            tmpAnalysis.intseen = intseen;
            tmpAnalysis.analysis = analysis;
            tmpAnalysis.analysis[phi] = TypeTree(BaseType::Integer).Only(-1);
            for (auto U : phi->users()) {
              if (auto I = dyn_cast<Instruction>(U)) {
                tmpAnalysis.visit(*I);
              }
            }
            tmpAnalysis.run();
            if (!tmpAnalysis.Invalid) {
              TypeTree Result = tmpAnalysis.getAnalysis(phi);
              for (auto &op : phi->incoming_values()) {
                Result &= tmpAnalysis.getAnalysis(op);
              }
              if (Result == TypeTree(BaseType::Integer).Only(-1) ||
                  Result == TypeTree(BaseType::Anything).Only(-1)) {
                updateAnalysis(phi, Result, phi);
                for (auto &pair : tmpAnalysis.analysis) {
                  updateAnalysis(pair.first, pair.second, phi);
                }
                Changed = true;
              }
            }
          }

          if (direction & DOWN && phi->getType()->isFPOrFPVectorTy() &&
              !getAnalysis(phi).isKnown()) {
            // Assume that this is an integer, does that mean we can prove that
            // the incoming operands are integral
            TypeAnalyzer tmpAnalysis(fntypeinfo, interprocedural, DOWN);
            tmpAnalysis.workList.clear();
            tmpAnalysis.intseen = intseen;
            tmpAnalysis.analysis = analysis;
            tmpAnalysis.analysis[phi] =
                TypeTree(phi->getType()->getScalarType()).Only(-1);
            for (auto U : phi->users()) {
              if (auto I = dyn_cast<Instruction>(U)) {
                tmpAnalysis.visit(*I);
              }
            }
            tmpAnalysis.run();
            if (!tmpAnalysis.Invalid) {
              TypeTree Result = tmpAnalysis.getAnalysis(phi);
              for (auto &op : phi->incoming_values()) {
                Result &= tmpAnalysis.getAnalysis(op);
              }
              if (Result ==
                      TypeTree(phi->getType()->getScalarType()).Only(-1) ||
                  Result == TypeTree(BaseType::Anything).Only(-1)) {
                updateAnalysis(phi, Result, phi);
                for (auto &pair : tmpAnalysis.analysis) {
                  updateAnalysis(pair.first, pair.second, phi);
                }
                Changed = true;
              }
            }
          }
        }
      }
    }
  } while (Changed);
  return;
}

void TypeAnalyzer::run() {
  // This function runs a full round of type analysis.
  // This works by doing two stages of analysis,
  // with a "deduced integer types for unused" values
  // sandwiched in-between. This is done because we only
  // perform that check for values without types.
  //
  // For performance reasons in each round of type analysis
  // only analyze any call instances after all other potential
  // updates have been done. This is to minimize the number
  // of expensive interprocedural analyses
  std::deque<CallInst *> pendingCalls;

  do {

    while (!Invalid && workList.size()) {
      auto todo = workList.front();
      workList.pop_front();
      if (auto ci = dyn_cast<CallInst>(todo)) {
        pendingCalls.push_back(ci);
        continue;
      }
      visitValue(*todo);
    }

    if (pendingCalls.size() > 0) {
      auto todo = pendingCalls.front();
      pendingCalls.pop_front();
      visitValue(*todo);
      continue;
    } else
      break;

  } while (1);

  runPHIHypotheses();

  do {

    while (!Invalid && workList.size()) {
      auto todo = workList.front();
      workList.pop_front();
      if (auto ci = dyn_cast<CallInst>(todo)) {
        pendingCalls.push_back(ci);
        continue;
      }
      visitValue(*todo);
    }

    if (pendingCalls.size() > 0) {
      auto todo = pendingCalls.front();
      pendingCalls.pop_front();
      visitValue(*todo);
      continue;
    } else
      break;

  } while (1);
}

void TypeAnalyzer::visitValue(Value &val) {
  if (auto CE = dyn_cast<ConstantExpr>(&val)) {
    visitConstantExpr(*CE);
  }

  if (isa<Constant>(&val)) {
    return;
  }

  if (!isa<Argument>(&val) && !isa<Instruction>(&val))
    return;

  if (auto inst = dyn_cast<Instruction>(&val)) {
    visit(*inst);
  }
}

void TypeAnalyzer::visitConstantExpr(ConstantExpr &CE) {
  auto I = CE.getAsInstruction();
  I->insertBefore(fntypeinfo.Function->getEntryBlock().getTerminator());
  analysis[I] = analysis[&CE];
  visit(*I);
  updateAnalysis(&CE, analysis[I], &CE);
  analysis.erase(I);
  I->eraseFromParent();
}

void TypeAnalyzer::visitCmpInst(CmpInst &cmp) {
  // No directionality check needed as always true
  updateAnalysis(&cmp, TypeTree(BaseType::Integer).Only(-1), &cmp);
  if (direction & UP) {
    updateAnalysis(
        cmp.getOperand(0),
        TypeTree(getAnalysis(cmp.getOperand(1)).Data0().PurgeAnything()[{}])
            .Only(-1),
        &cmp);
    updateAnalysis(
        cmp.getOperand(1),
        TypeTree(getAnalysis(cmp.getOperand(0)).Data0().PurgeAnything()[{}])
            .Only(-1),
        &cmp);
  }
}

void TypeAnalyzer::visitAllocaInst(AllocaInst &I) {
  // No directionality check needed as always true
  updateAnalysis(I.getArraySize(), TypeTree(BaseType::Integer).Only(-1), &I);
  updateAnalysis(&I, TypeTree(BaseType::Pointer).Only(-1), &I);
}

void TypeAnalyzer::visitLoadInst(LoadInst &I) {
  auto &DL = I.getParent()->getParent()->getParent()->getDataLayout();
  auto LoadSize = (DL.getTypeSizeInBits(I.getType()) + 7) / 8;

  // Only propagate mappings in range that aren't "Anything" into the pointer
  auto ptr = getAnalysis(&I)
                 .ShiftIndices(DL, /*start*/ 0, LoadSize, /*addOffset*/ 0)
                 .PurgeAnything();
  ptr |= TypeTree(BaseType::Pointer);
  if (direction & UP)
    updateAnalysis(I.getOperand(0), ptr.Only(-1), &I);
  if (direction & DOWN)
    updateAnalysis(&I, getAnalysis(I.getOperand(0)).Lookup(LoadSize, DL), &I);
}

void TypeAnalyzer::visitStoreInst(StoreInst &I) {
  auto &DL = I.getParent()->getParent()->getParent()->getDataLayout();
  auto StoreSize =
      (DL.getTypeSizeInBits(I.getValueOperand()->getType()) + 7) / 8;

  // Only propagate mappings in range that aren't "Anything" into the pointer
  auto ptr = TypeTree(BaseType::Pointer);
  auto purged = getAnalysis(I.getValueOperand())
                    .ShiftIndices(DL, /*start*/ 0, StoreSize, /*addOffset*/ 0)
                    .PurgeAnything();
  ptr |= purged;

  if (direction & UP) {
    updateAnalysis(I.getPointerOperand(), ptr.Only(-1), &I);
    // Note that we also must purge anything from ptr => value in case we store
    // to a nullptr which has type [-1, -1]: Anything. While storing to a
    // nullptr is obviously bad, this doesn't mean the value we're storing is an
    // Anything
    updateAnalysis(I.getValueOperand(),
                   getAnalysis(I.getPointerOperand())
                       .PurgeAnything()
                       .Lookup(StoreSize, DL),
                   &I);
  }
}

// Give a list of sets representing the legal set of values at a given index
// return a set of all possible combinations of those values
template <typename T>
std::set<std::vector<T>> getSet(const std::vector<std::set<T>> &todo,
                                size_t idx) {
  std::set<std::vector<T>> out;
  if (idx == 0) {
    for (auto val : todo[0]) {
      out.insert({val});
    }
    return out;
  }

  auto old = getSet(todo, idx - 1);
  for (const auto &oldv : old) {
    for (auto val : todo[idx]) {
      auto nex = oldv;
      nex.push_back(val);
      out.insert(nex);
    }
  }
  return out;
}

void TypeAnalyzer::visitGetElementPtrInst(GetElementPtrInst &gep) {
  auto &DL = fntypeinfo.Function->getParent()->getDataLayout();

  auto pointerAnalysis = getAnalysis(gep.getPointerOperand());
  if (direction & DOWN)
    updateAnalysis(&gep, pointerAnalysis.KeepMinusOne(), &gep);

  // If one of these is known to be a pointer, propagate it
  if (direction & DOWN)
    updateAnalysis(&gep, TypeTree(pointerAnalysis.Inner0()).Only(-1), &gep);
  if (direction & UP)
    updateAnalysis(gep.getPointerOperand(),
                   TypeTree(getAnalysis(&gep).Inner0()).Only(-1), &gep);

  if (isa<UndefValue>(gep.getPointerOperand())) {
    return;
  }

  std::vector<std::set<Value *>> idnext;

  // If we know that the pointer operand is indeed a pointer, then the indicies
  // must be integers Note that we can't do this if we don't know the pointer
  // operand is a pointer since doing 1[pointer] is legal
  //  sadly this still may not work since (nullptr)[fn] => fn where fn is
  //  pointer and not int (whereas nullptr is a pointer) However if we are
  //  inbounds you are only allowed to have nullptr[0] or nullptr[nullptr],
  //  making this valid
  // Assuming nullptr[nullptr] doesn't occur in practice, the following
  // is valid. We could make it always valid by checking the pointer
  // operand explicitly is a pointer.
  if (direction & UP) {
    if (gep.isInBounds()) {
      for (auto &ind : gep.indices()) {
        updateAnalysis(ind, TypeTree(BaseType::Integer).Only(-1), &gep);
      }
    }
  }

  for (auto &a : gep.indices()) {
    auto iset = fntypeinfo.knownIntegralValues(a, DT, intseen);
    std::set<Value *> vset;
    for (auto i : iset) {
      // Don't consider negative indices of gep
      if (i < 0)
        continue;
      vset.insert(ConstantInt::get(a->getType(), i));
    }
    idnext.push_back(vset);
    if (idnext.back().size() == 0)
      return;
  }

  for (auto vec : getSet(idnext, idnext.size() - 1)) {
    auto g2 = GetElementPtrInst::Create(nullptr, gep.getOperand(0), vec);
#if LLVM_VERSION_MAJOR > 6
    APInt ai(DL.getIndexSizeInBits(gep.getPointerAddressSpace()), 0);
#else
    APInt ai(DL.getPointerSize(gep.getPointerAddressSpace()) * 8, 0);
#endif
    g2->accumulateConstantOffset(DL, ai);
    // Using destructor rather than eraseFromParent
    //   as g2 has no parent
    delete g2;

    int off = (int)ai.getLimitedValue();

    // TODO also allow negative offsets
    if (off < 0)
      continue;

    int maxSize = -1;
    if (cast<ConstantInt>(vec[0])->getLimitedValue() == 0) {
      maxSize = DL.getTypeAllocSizeInBits(
                    cast<PointerType>(gep.getType())->getElementType()) /
                8;
    }

    auto unmerged = pointerAnalysis.Data0()
                        .ShiftIndices(DL, /*init offset*/ off,
                                      /*max size*/ maxSize, /*newoffset*/ 0)
                        .Only(-1);

    if (direction & UP)
      updateAnalysis(&gep, unmerged, &gep);

    auto merged = getAnalysis(&gep)
                      .Data0()
                      .ShiftIndices(DL, /*init offset*/ 0, /*max size*/ -1,
                                    /*new offset*/ off)
                      .Only(-1);

    if (direction & DOWN)
      updateAnalysis(gep.getPointerOperand(), merged, &gep);
  }
}

void TypeAnalyzer::visitPHINode(PHINode &phi) {
  if (direction & UP) {
    for (auto &op : phi.incoming_values()) {
      updateAnalysis(op, getAnalysis(&phi), &phi);
    }
  }

  assert(phi.getNumIncomingValues() > 0);

  // TODO generalize this (and for recursive, etc)
  std::deque<Value *> vals;
  std::set<Value *> seen{&phi};
  for (auto &op : phi.incoming_values()) {
    vals.push_back(op);
  }

  std::vector<BinaryOperator *> bos;

  // Unique values that propagate into this phi
  std::vector<Value *> UniqueValues;

  while (vals.size()) {
    Value *todo = vals.front();
    vals.pop_front();

    if (auto bo = dyn_cast<BinaryOperator>(todo)) {
      if (bo->getOpcode() == BinaryOperator::Add) {
        if (isa<ConstantInt>(bo->getOperand(0))) {
          bos.push_back(bo);
          todo = bo->getOperand(1);
        }
        if (isa<ConstantInt>(bo->getOperand(1))) {
          bos.push_back(bo);
          todo = bo->getOperand(0);
        }
      }
    }

    if (seen.count(todo))
      continue;
    seen.insert(todo);

    if (auto nphi = dyn_cast<PHINode>(todo)) {
      for (auto &op : nphi->incoming_values()) {
        vals.push_back(op);
      }
      continue;
    }
    if (auto sel = dyn_cast<SelectInst>(todo)) {
      vals.push_back(sel->getOperand(1));
      vals.push_back(sel->getOperand(2));
      continue;
    }
    UniqueValues.push_back(todo);
  }

  TypeTree PhiTypes;
  bool set = false;

  for (size_t i = 0, size = UniqueValues.size(); i < size; ++i) {
    TypeTree newData = getAnalysis(UniqueValues[i]);
    if (UniqueValues.size() == 2) {
      if (auto BO = dyn_cast<BinaryOperator>(UniqueValues[i])) {
        if (BO->getOpcode() == BinaryOperator::Add ||
            BO->getOpcode() == BinaryOperator::Mul) {
          TypeTree otherData = getAnalysis(UniqueValues[1 - i]);
          // If we are adding/muling to a constant to derive this, we can assume
          // it to be an integer rather than Anything
          if (isa<ConstantInt>(UniqueValues[1 - i])) {
            otherData = TypeTree(BaseType::Integer).Only(-1);
          }
          if (BO->getOperand(0) == &phi) {
            set = true;
            PhiTypes = otherData;
            PhiTypes.binopIn(getAnalysis(BO->getOperand(1)), BO->getOpcode());
            break;
          } else if (BO->getOperand(1) == &phi) {
            set = true;
            PhiTypes = getAnalysis(BO->getOperand(0));
            PhiTypes.binopIn(otherData, BO->getOpcode());
            break;
          }
        } else if (BO->getOpcode() == BinaryOperator::Sub) {
          // Repeated subtraction from a type X yields the type X back
          TypeTree otherData = getAnalysis(UniqueValues[1 - i]);
          // If we are subtracting from a constant to derive this, we can assume
          // it to be an integer rather than Anything
          if (isa<ConstantInt>(UniqueValues[1 - i])) {
            otherData = TypeTree(BaseType::Integer).Only(-1);
          }
          if (BO->getOperand(0) == &phi) {
            set = true;
            PhiTypes = otherData;
            break;
          }
        }
      }
    }
    if (set) {
      PhiTypes &= newData;
    } else {
      set = true;
      PhiTypes = newData;
    }
  }

  assert(set);
  // If we are only add / sub / etc to derive a value based off 0
  // we can start by assuming the type of 0 is integer rather
  // than assuming it could be anything (per null)
  if (bos.size() > 0 && UniqueValues.size() == 1 &&
      isa<ConstantInt>(UniqueValues[0]) &&
      cast<ConstantInt>(UniqueValues[0])->isZero()) {
    PhiTypes = TypeTree(BaseType::Integer).Only(-1);
  }
  for (BinaryOperator *bo : bos) {
    TypeTree vd1 = isa<ConstantInt>(bo->getOperand(0))
                       ? getAnalysis(bo->getOperand(0)).Data0()
                       : PhiTypes.Data0();
    TypeTree vd2 = isa<ConstantInt>(bo->getOperand(1))
                       ? getAnalysis(bo->getOperand(1)).Data0()
                       : PhiTypes.Data0();
    vd1.binopIn(vd2, bo->getOpcode());
    PhiTypes &= vd1.Only(bo->getType()->isIntegerTy() ? -1 : 0);
  }

  if (direction & DOWN)
    updateAnalysis(&phi, PhiTypes, &phi);
}

void TypeAnalyzer::visitTruncInst(TruncInst &I) {
  if (direction & DOWN)
    updateAnalysis(&I, getAnalysis(I.getOperand(0)), &I);
  if (direction & UP)
    updateAnalysis(I.getOperand(0), getAnalysis(&I), &I);
}

void TypeAnalyzer::visitZExtInst(ZExtInst &I) {
  if (direction & DOWN)
    updateAnalysis(&I, getAnalysis(I.getOperand(0)), &I);
  if (direction & UP)
    updateAnalysis(I.getOperand(0), getAnalysis(&I), &I);
}

void TypeAnalyzer::visitSExtInst(SExtInst &I) {
  // This is only legal on integer types [not pointers per sign]
  // nor floatings points. Likewise, there's no direction check
  // necessary since this is always valid.
  updateAnalysis(&I, TypeTree(BaseType::Integer).Only(-1), &I);
  updateAnalysis(I.getOperand(0), TypeTree(BaseType::Integer).Only(-1), &I);
}

void TypeAnalyzer::visitAddrSpaceCastInst(AddrSpaceCastInst &I) {
  if (direction & DOWN)
    updateAnalysis(&I, getAnalysis(I.getOperand(0)), &I);
  if (direction & UP)
    updateAnalysis(I.getOperand(0), getAnalysis(&I), &I);
}

void TypeAnalyzer::visitFPExtInst(FPExtInst &I) {
  // No direction check as always true
  updateAnalysis(&I, TypeTree(ConcreteType(I.getType())).Only(-1), &I);
  updateAnalysis(I.getOperand(0),
                 TypeTree(ConcreteType(I.getOperand(0)->getType())).Only(-1),
                 &I);
}

void TypeAnalyzer::visitFPTruncInst(FPTruncInst &I) {
  // No direction check as always true
  updateAnalysis(&I, TypeTree(ConcreteType(I.getType())).Only(-1), &I);
  updateAnalysis(I.getOperand(0),
                 TypeTree(ConcreteType(I.getOperand(0)->getType())).Only(-1),
                 &I);
}

void TypeAnalyzer::visitFPToUIInst(FPToUIInst &I) {
  // No direction check as always true
  updateAnalysis(&I, TypeTree(BaseType::Integer).Only(-1), &I);
  updateAnalysis(I.getOperand(0),
                 TypeTree(ConcreteType(I.getOperand(0)->getType())).Only(-1),
                 &I);
}

void TypeAnalyzer::visitFPToSIInst(FPToSIInst &I) {
  // No direction check as always true
  updateAnalysis(&I, TypeTree(BaseType::Integer).Only(-1), &I);
  updateAnalysis(I.getOperand(0),
                 TypeTree(ConcreteType(I.getOperand(0)->getType())).Only(-1),
                 &I);
}

void TypeAnalyzer::visitUIToFPInst(UIToFPInst &I) {
  // No direction check as always true
  updateAnalysis(I.getOperand(0), TypeTree(BaseType::Integer).Only(-1), &I);
  updateAnalysis(&I, TypeTree(ConcreteType(I.getType())).Only(-1), &I);
}

void TypeAnalyzer::visitSIToFPInst(SIToFPInst &I) {
  // No direction check as always true
  updateAnalysis(I.getOperand(0), TypeTree(BaseType::Integer).Only(-1), &I);
  updateAnalysis(&I, TypeTree(ConcreteType(I.getType())).Only(-1), &I);
}

void TypeAnalyzer::visitPtrToIntInst(PtrToIntInst &I) {
  // Note it is illegal to assume here that either is a pointer or an int
  if (direction & DOWN)
    updateAnalysis(&I, getAnalysis(I.getOperand(0)), &I);
  if (direction & UP)
    updateAnalysis(I.getOperand(0), getAnalysis(&I), &I);
}

void TypeAnalyzer::visitIntToPtrInst(IntToPtrInst &I) {
  // Note it is illegal to assume here that either is a pointer or an int
  if (direction & DOWN)
    updateAnalysis(&I, getAnalysis(I.getOperand(0)), &I);
  if (direction & UP)
    updateAnalysis(I.getOperand(0), getAnalysis(&I), &I);
}

void TypeAnalyzer::visitBitCastInst(BitCastInst &I) {
  if (I.getType()->isIntOrIntVectorTy() || I.getType()->isFPOrFPVectorTy()) {
    if (direction & DOWN)
      updateAnalysis(&I, getAnalysis(I.getOperand(0)), &I);
    if (direction & UP)
      updateAnalysis(I.getOperand(0), getAnalysis(&I), &I);
    return;
  }

  if (I.getType()->isPointerTy() && I.getOperand(0)->getType()->isPointerTy()) {
    Type *et1 = cast<PointerType>(I.getType())->getElementType();
    Type *et2 = cast<PointerType>(I.getOperand(0)->getType())->getElementType();

    if (direction & DOWN)
      updateAnalysis(
          &I,
          getAnalysis(I.getOperand(0))
              .Data0()
              .KeepForCast(fntypeinfo.Function->getParent()->getDataLayout(),
                           et2, et1)
              .Only(-1),
          &I);

    if (direction & UP)
      updateAnalysis(
          I.getOperand(0),
          getAnalysis(&I)
              .Data0()
              .KeepForCast(fntypeinfo.Function->getParent()->getDataLayout(),
                           et1, et2)
              .Only(-1),
          &I);
  }
}

void TypeAnalyzer::visitSelectInst(SelectInst &I) {
  if (direction & UP)
    updateAnalysis(I.getTrueValue(), getAnalysis(&I), &I);
  if (direction & UP)
    updateAnalysis(I.getFalseValue(), getAnalysis(&I), &I);

  if (direction & DOWN) {
    updateAnalysis(&I, getAnalysis(I.getTrueValue()).PurgeAnything(), &I);
    updateAnalysis(&I, getAnalysis(I.getFalseValue()).PurgeAnything(), &I);

    TypeTree vd = getAnalysis(I.getTrueValue());
    vd &= getAnalysis(I.getFalseValue());
    updateAnalysis(&I, vd, &I);
  }
}

void TypeAnalyzer::visitExtractElementInst(ExtractElementInst &I) {
  if (direction & UP)
    updateAnalysis(I.getIndexOperand(), BaseType::Integer, &I);
  if (direction & UP)
    updateAnalysis(I.getVectorOperand(), getAnalysis(&I), &I);
  if (direction & DOWN)
    updateAnalysis(&I, getAnalysis(I.getVectorOperand()), &I);
}

void TypeAnalyzer::visitInsertElementInst(InsertElementInst &I) {
  updateAnalysis(I.getOperand(2), BaseType::Integer, &I);

  // if we are inserting into undef/etc the anything should not be propagated
  auto res = getAnalysis(I.getOperand(0)).PurgeAnything();

  res |= getAnalysis(I.getOperand(1));
  // res |= getAnalysis(I.getOperand(1)).Only(idx);
  res |= getAnalysis(&I);

  if (direction & UP)
    updateAnalysis(I.getOperand(0), res, &I);
  if (direction & DOWN)
    updateAnalysis(&I, res, &I);
  if (direction & UP)
    updateAnalysis(I.getOperand(1), res, &I);
}

void TypeAnalyzer::visitShuffleVectorInst(ShuffleVectorInst &I) {
  if (direction & UP)
    updateAnalysis(I.getOperand(0), getAnalysis(&I), &I);
  if (direction & UP)
    updateAnalysis(I.getOperand(1), getAnalysis(&I), &I);

  TypeTree vd = getAnalysis(I.getOperand(0));
  vd &= getAnalysis(I.getOperand(1));

  if (direction & DOWN)
    updateAnalysis(&I, vd, &I);
}

void TypeAnalyzer::visitExtractValueInst(ExtractValueInst &I) {
  auto &dl = fntypeinfo.Function->getParent()->getDataLayout();
  std::vector<Value *> vec;
  vec.push_back(ConstantInt::get(Type::getInt64Ty(I.getContext()), 0));
  for (auto ind : I.indices()) {
    vec.push_back(ConstantInt::get(Type::getInt32Ty(I.getContext()), ind));
  }
  auto ud = UndefValue::get(PointerType::getUnqual(I.getOperand(0)->getType()));
  auto g2 = GetElementPtrInst::Create(nullptr, ud, vec);
#if LLVM_VERSION_MAJOR > 6
  APInt ai(dl.getIndexSizeInBits(g2->getPointerAddressSpace()), 0);
#else
  APInt ai(dl.getPointerSize(g2->getPointerAddressSpace()) * 8, 0);
#endif
  g2->accumulateConstantOffset(dl, ai);
  // Using destructor rather than eraseFromParent
  //   as g2 has no parent
  delete g2;

  int off = (int)ai.getLimitedValue();

  int size = dl.getTypeSizeInBits(I.getType()) / 8;

  if (direction & DOWN)
    updateAnalysis(&I,
                   getAnalysis(I.getOperand(0))
                       .ShiftIndices(dl, off, size, /*addOffset*/ 0)
                       .CanonicalizeValue(size, dl),
                   &I);

  if (direction & UP)
    updateAnalysis(I.getOperand(0),
                   getAnalysis(&I).ShiftIndices(dl, 0, size, off), &I);
}

void TypeAnalyzer::visitInsertValueInst(InsertValueInst &I) {
  auto &dl = fntypeinfo.Function->getParent()->getDataLayout();
  std::vector<Value *> vec;
  vec.push_back(ConstantInt::get(Type::getInt64Ty(I.getContext()), 0));
  for (auto ind : I.indices()) {
    vec.push_back(ConstantInt::get(Type::getInt32Ty(I.getContext()), ind));
  }
  auto ud = UndefValue::get(PointerType::getUnqual(I.getOperand(0)->getType()));
  auto g2 = GetElementPtrInst::Create(nullptr, ud, vec);
#if LLVM_VERSION_MAJOR > 6
  APInt ai(dl.getIndexSizeInBits(g2->getPointerAddressSpace()), 0);
#else
  APInt ai(dl.getPointerSize(g2->getPointerAddressSpace()) * 8, 0);
#endif
  g2->accumulateConstantOffset(dl, ai);
  // Using destructor rather than eraseFromParent
  //   as g2 has no parent
  delete g2;

  int off = (int)ai.getLimitedValue();

  int agg_size = dl.getTypeSizeInBits(I.getType()) / 8;
  int ins_size =
      dl.getTypeSizeInBits(I.getInsertedValueOperand()->getType()) / 8;

  if (direction & UP)
    updateAnalysis(I.getAggregateOperand(),
                   getAnalysis(&I).Clear(off, off + ins_size, agg_size), &I);
  if (direction & UP)
    updateAnalysis(I.getInsertedValueOperand(),
                   getAnalysis(&I)
                       .ShiftIndices(dl, off, ins_size, 0)
                       .CanonicalizeValue(ins_size, dl),
                   &I);

  auto new_res =
      getAnalysis(I.getAggregateOperand()).Clear(off, off + ins_size, agg_size);
  auto shifted = getAnalysis(I.getInsertedValueOperand())
                     .ShiftIndices(dl, 0, ins_size, off);
  new_res |= shifted;
  if (direction & DOWN)
    updateAnalysis(&I, new_res.CanonicalizeValue(agg_size, dl), &I);
}

void TypeAnalyzer::dump() {
  llvm::errs() << "<analysis>\n";
  for (auto &pair : analysis) {
    llvm::errs() << *pair.first << ": " << pair.second.str()
                 << ", intvals: " << to_string(knownIntegralValues(pair.first))
                 << "\n";
  }
  llvm::errs() << "</analysis>\n";
}

void TypeAnalyzer::visitBinaryOperator(BinaryOperator &I) {
  if (I.getOpcode() == BinaryOperator::FAdd ||
      I.getOpcode() == BinaryOperator::FSub ||
      I.getOpcode() == BinaryOperator::FMul ||
      I.getOpcode() == BinaryOperator::FDiv ||
      I.getOpcode() == BinaryOperator::FRem) {
    auto ty = I.getType()->getScalarType();
    assert(ty->isFloatingPointTy());
    ConcreteType dt(ty);
    if (direction & UP)
      updateAnalysis(I.getOperand(0), TypeTree(dt).Only(-1), &I);
    if (direction & UP)
      updateAnalysis(I.getOperand(1), TypeTree(dt).Only(-1), &I);
    if (direction & DOWN)
      updateAnalysis(&I, TypeTree(dt).Only(-1), &I);
  } else {
    auto AnalysisLHS = getAnalysis(I.getOperand(0)).Data0();
    auto AnalysisRHS = getAnalysis(I.getOperand(1)).Data0();
    auto AnalysisRet = getAnalysis(&I).Data0();
    TypeTree Result;

    switch (I.getOpcode()) {
    case BinaryOperator::Sub:
      // ptr - ptr => int and int - int => int; thus int = a - b says only that
      // these are equal ptr - int => ptr and int - ptr => ptr; thus
      // howerver we do not want to propagate underlying ptr types since it's
      // legal to subtract unrelated pointer
      if (AnalysisRet[{}] == BaseType::Integer) {
        if (direction & UP)
          updateAnalysis(I.getOperand(0), TypeTree(AnalysisRHS[{}]).Only(-1),
                         &I);
        if (direction & UP)
          updateAnalysis(I.getOperand(1), TypeTree(AnalysisLHS[{}]).Only(-1),
                         &I);
      }
      break;

    case BinaryOperator::Add:
    case BinaryOperator::Mul:
      // if a + b or a * b == int, then a and b must be ints
      if (direction & UP)
        updateAnalysis(I.getOperand(0),
                       TypeTree(AnalysisRet.JustInt()[{}]).Only(-1), &I);
      if (direction & UP)
        updateAnalysis(I.getOperand(1),
                       TypeTree(AnalysisRet.JustInt()[{}]).Only(-1), &I);
      break;

    default:
      break;
    }
    Result = AnalysisLHS;
    Result.binopIn(AnalysisRHS, I.getOpcode());

    if (I.getOpcode() == BinaryOperator::And) {
      for (int i = 0; i < 2; ++i) {
        for (auto andval :
             fntypeinfo.knownIntegralValues(I.getOperand(i), DT, intseen)) {
          if (andval <= 16 && andval >= 0) {
            Result = TypeTree(BaseType::Integer);
          } else if (andval < 0 && andval >= -64) {
            // If a small negative number, this just masks off the lower bits
            // in this case we can say that this is the same as the other
            // operand
            Result = getAnalysis(I.getOperand(1 - i)).Data0();
          }
        }
        // If we and a constant against an integer, the result remains an
        // integer
        if (isa<ConstantInt>(I.getOperand(i)) &&
            getAnalysis(I.getOperand(1 - i)).Inner0() == BaseType::Integer) {
          Result = TypeTree(BaseType::Integer);
        }
      }
    } else if (I.getOpcode() == BinaryOperator::Add ||
               I.getOpcode() == BinaryOperator::Sub) {
      for (int i = 0; i < 2; ++i) {
        if (auto CI = dyn_cast<ConstantInt>(I.getOperand(i))) {
          if (CI->isNegative()) {
            // If add/sub with a negative number, the result is equal to the
            // type of the other operand (and we don't need to assume this was
            // an "anything")
            Result = getAnalysis(I.getOperand(1 - i)).Data0();
          }
        }
      }
    }
    if (direction & DOWN)
      updateAnalysis(&I, Result.Only(-1), &I);
  }
}

void TypeAnalyzer::visitMemTransferInst(llvm::MemTransferInst &MTI) {
  // If memcpy / memmove of pointer, we can propagate type information from src
  // to dst up to the length and vice versa
  size_t sz = 1;
  for (auto val :
       fntypeinfo.knownIntegralValues(MTI.getArgOperand(2), DT, intseen)) {
    assert(val >= 0);
    sz = max(sz, (size_t)val);
  }

  updateAnalysis(MTI.getArgOperand(0), TypeTree(BaseType::Pointer).Only(-1),
                 &MTI);
  updateAnalysis(MTI.getArgOperand(1), TypeTree(BaseType::Pointer).Only(-1),
                 &MTI);

  TypeTree res = getAnalysis(MTI.getArgOperand(0)).AtMost(sz).PurgeAnything();
  TypeTree res2 = getAnalysis(MTI.getArgOperand(1)).AtMost(sz).PurgeAnything();
  res |= res2;

  if (direction & UP) {
    updateAnalysis(MTI.getArgOperand(0), res, &MTI);
    updateAnalysis(MTI.getArgOperand(1), res, &MTI);
    for (unsigned i = 2; i < MTI.getNumArgOperands(); ++i) {
      updateAnalysis(MTI.getArgOperand(i), TypeTree(BaseType::Integer).Only(-1),
                     &MTI);
    }
  }
}

void TypeAnalyzer::visitIntrinsicInst(llvm::IntrinsicInst &I) {
  switch (I.getIntrinsicID()) {
  case Intrinsic::ctpop:
  case Intrinsic::ctlz:
  case Intrinsic::cttz:
  case Intrinsic::nvvm_read_ptx_sreg_tid_x:
  case Intrinsic::nvvm_read_ptx_sreg_tid_y:
  case Intrinsic::nvvm_read_ptx_sreg_tid_z:
  case Intrinsic::nvvm_read_ptx_sreg_ntid_x:
  case Intrinsic::nvvm_read_ptx_sreg_ntid_y:
  case Intrinsic::nvvm_read_ptx_sreg_ntid_z:
  case Intrinsic::nvvm_read_ptx_sreg_ctaid_x:
  case Intrinsic::nvvm_read_ptx_sreg_ctaid_y:
  case Intrinsic::nvvm_read_ptx_sreg_ctaid_z:
  case Intrinsic::nvvm_read_ptx_sreg_nctaid_x:
  case Intrinsic::nvvm_read_ptx_sreg_nctaid_y:
  case Intrinsic::nvvm_read_ptx_sreg_nctaid_z:
  case Intrinsic::nvvm_read_ptx_sreg_warpsize:
    // No direction check as always valid
    updateAnalysis(&I, TypeTree(BaseType::Integer).Only(-1), &I);
    return;

  case Intrinsic::log:
  case Intrinsic::log2:
  case Intrinsic::log10:
  case Intrinsic::exp:
  case Intrinsic::exp2:
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::round:
  case Intrinsic::sqrt:
  case Intrinsic::fabs:
    // No direction check as always valid
    updateAnalysis(
        &I, TypeTree(ConcreteType(I.getType()->getScalarType())).Only(-1), &I);
    // No direction check as always valid
    updateAnalysis(
        I.getOperand(0),
        TypeTree(ConcreteType(I.getOperand(0)->getType()->getScalarType()))
            .Only(-1),
        &I);
    return;

  case Intrinsic::powi:
    // No direction check as always valid
    updateAnalysis(
        &I, TypeTree(ConcreteType(I.getType()->getScalarType())).Only(-1), &I);
    // No direction check as always valid
    updateAnalysis(
        I.getOperand(0),
        TypeTree(ConcreteType(I.getOperand(0)->getType()->getScalarType()))
            .Only(-1),
        &I);
    // No direction check as always valid
    updateAnalysis(I.getOperand(1), TypeTree(BaseType::Integer).Only(-1), &I);
    return;

#if LLVM_VERSION_MAJOR < 10
  case Intrinsic::x86_sse_max_ss:
  case Intrinsic::x86_sse_max_ps:
  case Intrinsic::x86_sse_min_ss:
  case Intrinsic::x86_sse_min_ps:
#endif
#if LLVM_VERSION_MAJOR >= 9
  case Intrinsic::experimental_vector_reduce_v2_fadd:
#endif
  case Intrinsic::maxnum:
  case Intrinsic::minnum:
  case Intrinsic::pow:
    // No direction check as always valid
    updateAnalysis(
        &I, TypeTree(ConcreteType(I.getType()->getScalarType())).Only(-1), &I);
    // No direction check as always valid
    updateAnalysis(
        I.getOperand(0),
        TypeTree(ConcreteType(I.getOperand(0)->getType()->getScalarType()))
            .Only(-1),
        &I);
    // No direction check as always valid
    updateAnalysis(
        I.getOperand(1),
        TypeTree(ConcreteType(I.getOperand(1)->getType()->getScalarType()))
            .Only(-1),
        &I);
    return;
  case Intrinsic::umul_with_overflow:
  case Intrinsic::smul_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::usub_with_overflow:
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::uadd_with_overflow: {
    // val, bool
    auto analysis = getAnalysis(&I).Data0();

    BinaryOperator::BinaryOps opcode;
    // TODO update to use better rules in regular binop
    switch (I.getIntrinsicID()) {
    case Intrinsic::ssub_with_overflow:
    case Intrinsic::usub_with_overflow: {
      // TODO propagate this info
      // ptr - ptr => int and int - int => int; thus int = a - b says only that
      // these are equal ptr - int => ptr and int - ptr => ptr; thus
      analysis = ConcreteType(BaseType::Unknown);
      opcode = BinaryOperator::Sub;
      break;
    }

    case Intrinsic::smul_with_overflow:
    case Intrinsic::umul_with_overflow: {
      opcode = BinaryOperator::Mul;
      // if a + b or a * b == int, then a and b must be ints
      analysis = analysis.JustInt();
      break;
    }
    case Intrinsic::sadd_with_overflow:
    case Intrinsic::uadd_with_overflow: {
      opcode = BinaryOperator::Add;
      // if a + b or a * b == int, then a and b must be ints
      analysis = analysis.JustInt();
      break;
    }
    default:
      llvm_unreachable("unknown binary operator");
    }

    // TODO update with newer binop protocol (see binop)
    if (direction & UP)
      updateAnalysis(I.getOperand(0), analysis.Only(-1), &I);
    if (direction & UP)
      updateAnalysis(I.getOperand(1), analysis.Only(-1), &I);

    TypeTree vd = getAnalysis(I.getOperand(0)).Data0();
    vd.binopIn(getAnalysis(I.getOperand(1)).Data0(), opcode);

    TypeTree overall = vd.Only(0);

    auto &dl = I.getParent()->getParent()->getParent()->getDataLayout();
    overall |=
        TypeTree(BaseType::Integer)
            .Only((dl.getTypeSizeInBits(I.getOperand(0)->getType()) + 7) / 8);

    if (direction & DOWN)
      updateAnalysis(&I, overall, &I);
    return;
  }
  default:
    return;
  }
}

/// This template class is defined to take the templated type T
/// update the analysis of the first argument (val) to be type T
/// As such, below we have several template specializations
/// to convert various c/c++ to TypeAnalysis types
template <typename T> struct TypeHandler {};

template <> struct TypeHandler<double> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TA.updateAnalysis(
        val,
        TypeTree(ConcreteType(Type::getDoubleTy(call.getContext()))).Only(-1),
        &call);
  }
};

template <> struct TypeHandler<float> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TA.updateAnalysis(
        val,
        TypeTree(ConcreteType(Type::getFloatTy(call.getContext()))).Only(-1),
        &call);
  }
};

template <> struct TypeHandler<long double> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TA.updateAnalysis(
        val,
        TypeTree(ConcreteType(Type::getX86_FP80Ty(call.getContext()))).Only(-1),
        &call);
  }
};

#if defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)
template <> struct TypeHandler<__float128> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TA.updateAnalysis(
        val,
        TypeTree(ConcreteType(Type::getFP128Ty(call.getContext()))).Only(-1),
        &call);
  }
};
#endif

template <> struct TypeHandler<double *> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(Type::getDoubleTy(call.getContext())).Only(0);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1), &call);
  }
};

template <> struct TypeHandler<float *> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(Type::getFloatTy(call.getContext())).Only(0);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1), &call);
  }
};

template <> struct TypeHandler<long double *> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(Type::getX86_FP80Ty(call.getContext())).Only(0);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1), &call);
  }
};

#if defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)
template <> struct TypeHandler<__float128 *> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(Type::getFP128Ty(call.getContext())).Only(0);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1), &call);
  }
};
#endif

template <> struct TypeHandler<void> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {}
};

template <> struct TypeHandler<void *> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1), &call);
  }
};

template <> struct TypeHandler<int> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer);
    TA.updateAnalysis(val, vd.Only(-1), &call);
  }
};

template <> struct TypeHandler<int *> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer).Only(0);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1), &call);
  }
};

template <> struct TypeHandler<unsigned int> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer);
    TA.updateAnalysis(val, vd.Only(-1), &call);
  }
};

template <> struct TypeHandler<unsigned int *> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer).Only(0);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1), &call);
  }
};

template <> struct TypeHandler<long int> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer);
    TA.updateAnalysis(val, vd.Only(-1), &call);
  }
};

template <> struct TypeHandler<long int *> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer).Only(0);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1), &call);
  }
};

template <> struct TypeHandler<long unsigned int> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer);
    TA.updateAnalysis(val, vd.Only(-1), &call);
  }
};

template <> struct TypeHandler<long unsigned int *> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer).Only(0);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1), &call);
  }
};

template <> struct TypeHandler<long long int> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer);
    TA.updateAnalysis(val, vd.Only(-1), &call);
  }
};

template <> struct TypeHandler<long long int *> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer).Only(0);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1), &call);
  }
};

template <> struct TypeHandler<long long unsigned int> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer);
    TA.updateAnalysis(val, vd.Only(-1), &call);
  }
};

template <> struct TypeHandler<long long unsigned int *> {
  static void analyzeType(Value *val, CallInst &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer).Only(0);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1), &call);
  }
};

template <typename... Arg0> struct FunctionArgumentIterator {
  static void analyzeFuncTypesHelper(unsigned idx, CallInst &call,
                                     TypeAnalyzer &TA) {}
};

template <typename Arg0, typename... Args>
struct FunctionArgumentIterator<Arg0, Args...> {
  static void analyzeFuncTypesHelper(unsigned idx, CallInst &call,
                                     TypeAnalyzer &TA) {
    TypeHandler<Arg0>::analyzeType(call.getOperand(idx), call, TA);
    FunctionArgumentIterator<Args...>::analyzeFuncTypesHelper(idx + 1, call,
                                                              TA);
  }
};

template <typename RT, typename... Args>
void analyzeFuncTypes(RT (*fn)(Args...), CallInst &call, TypeAnalyzer &TA) {
  TypeHandler<RT>::analyzeType(&call, call, TA);
  FunctionArgumentIterator<Args...>::analyzeFuncTypesHelper(0, call, TA);
}

void TypeAnalyzer::visitCallInst(CallInst &call) {
  assert(fntypeinfo.KnownValues.size() ==
         fntypeinfo.Function->getFunctionType()->getNumParams());

#if LLVM_VERSION_MAJOR >= 11
  if (auto iasm = dyn_cast<InlineAsm>(call.getCalledOperand())) {
#else
  if (auto iasm = dyn_cast<InlineAsm>(call.getCalledValue())) {
#endif
    // NO direction check as always valid
    if (iasm->getAsmString() == "cpuid") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1), &call);
      for (unsigned i = 0; i < call.getNumArgOperands(); ++i) {
        updateAnalysis(call.getArgOperand(i),
                       TypeTree(BaseType::Integer).Only(-1), &call);
      }
    }
  }

  if (Function *ci = call.getCalledFunction()) {

#define CONSIDER(fn)                                                           \
  if (ci->getName() == #fn) {                                                  \
    analyzeFuncTypes(::fn, call, *this);                                       \
    return;                                                                    \
  }

#define CONSIDER2(fn, ...)                                                     \
  if (ci->getName() == #fn) {                                                  \
    analyzeFuncTypes<__VA_ARGS__>(::fn, call, *this);                          \
    return;                                                                    \
  }
    // All these are always valid => no direction check
    // CONSIDER(malloc)
    // TODO consider handling other allocation functions integer inputs
    if (isAllocationFunction(*ci, interprocedural.TLI)) {
      size_t Idx = 0;
      for (auto &Arg : ci->args()) {
        if (Arg.getType()->isIntegerTy()) {
          updateAnalysis(call.getOperand(Idx),
                         TypeTree(BaseType::Integer).Only(-1), &call);
        }
        Idx++;
      }
      assert(ci->getReturnType()->isPointerTy());
      updateAnalysis(&call, TypeTree(BaseType::Pointer).Only(-1), &call);
    }
    if (isDeallocationFunction(*ci, interprocedural.TLI)) {
      size_t Idx = 0;
      for (auto &Arg : ci->args()) {
        if (Arg.getType()->isIntegerTy()) {
          updateAnalysis(call.getOperand(Idx),
                         TypeTree(BaseType::Integer).Only(-1), &call);
        }
        if (Arg.getType()->isPointerTy()) {
          updateAnalysis(call.getOperand(Idx),
                         TypeTree(BaseType::Pointer).Only(-1), &call);
        }
        Idx++;
      }
      assert(ci->getReturnType()->isVoidTy());
    }

    // CONSIDER(__lgamma_r_finite)
    CONSIDER2(frexp, double, double, int *)

    CONSIDER(frexpf)
    CONSIDER(frexpl)
    CONSIDER2(ldexp, double, double, int)
    CONSIDER2(modf, double, double, double *)

    CONSIDER2(cos, double, double)
    CONSIDER2(sin, double, double)
    CONSIDER2(tan, double, double)
    CONSIDER2(acos, double, double)
    CONSIDER2(asin, double, double)
    CONSIDER2(atan, double, double)
    CONSIDER2(atan2, double, double, double)
    CONSIDER2(cosh, double, double)
    CONSIDER2(sinh, double, double)
    CONSIDER2(tanh, double, double)
    CONSIDER(tanhf)
    CONSIDER2(acosh, double, double)
    CONSIDER(acoshf)
    CONSIDER(acoshl)
    CONSIDER2(asinh, double, double)
    CONSIDER(asinhf)
    CONSIDER(asinhl)
    CONSIDER2(atanh, double, double)
    CONSIDER(atanhl)
    CONSIDER(atanhf)
    CONSIDER2(exp, double, double)
    CONSIDER2(log, double, double)
    CONSIDER2(log10, double, double)
    CONSIDER2(exp2, double, double)
    CONSIDER(exp2f)
    CONSIDER(exp2l)
    CONSIDER2(expm1, double, double)
    CONSIDER(expm1f)
    CONSIDER(expm1l)
    CONSIDER2(ilogb, int, double)
    CONSIDER(ilogbf)
    CONSIDER(ilogbl)
    CONSIDER2(log1p, double, double)
    CONSIDER(log1pf)
    CONSIDER(log1pl)
    CONSIDER2(log2, double, double)
    CONSIDER(log2f)
    CONSIDER(log2l)
    CONSIDER2(logb, double, double)
    CONSIDER(logbf)
    CONSIDER(logbl)
    CONSIDER2(scalbn, double, double, int)
    CONSIDER(scalbnf)
    CONSIDER(scalbnl)
    CONSIDER2(scalbln, double, double, long)
    CONSIDER(scalblnf)
    CONSIDER(scalblnl)
    CONSIDER2(pow, double, double, double)
    CONSIDER2(sqrt, double, double)
    CONSIDER2(cbrt, double, double)
    CONSIDER(cbrtf)
    CONSIDER(cbrtl)
    CONSIDER2(hypot, double, double, double)
    CONSIDER2(erf, double, double)
    CONSIDER(erff)
    CONSIDER(erfl)
    CONSIDER2(erfc, double, double)
    CONSIDER(erfcf)
    CONSIDER(erfcl)
    CONSIDER2(tgamma, double, double)
    CONSIDER(tgammaf)
    CONSIDER(tgammal)
    CONSIDER2(lgamma, double, double)
    CONSIDER(lgammaf)
    CONSIDER(lgammal)
    CONSIDER2(ceil, double, double)
    CONSIDER2(floor, double, double)
    CONSIDER2(fmod, double, double, double)
    CONSIDER2(trunc, double, double)
    CONSIDER(truncf)
    CONSIDER(truncl)
    CONSIDER2(round, double, double)
    CONSIDER(roundf)
    CONSIDER(roundl)
    CONSIDER2(lround, long, double)
    CONSIDER(lroundf)
    CONSIDER(lroundl)
    CONSIDER2(llround, long long, double)
    CONSIDER(llroundf)
    CONSIDER(llroundl)
    CONSIDER2(rint, double, double)
    CONSIDER(rintf)
    CONSIDER(rintl)
    CONSIDER2(lrint, long, double)
    CONSIDER(lrintf)
    CONSIDER(lrintl)
    CONSIDER2(llrint, long long, double)
    CONSIDER(llrintf)
    CONSIDER(llrintl)
    CONSIDER2(remainder, double, double, double)
    CONSIDER(remainderf)
    CONSIDER(remainderl)
    CONSIDER2(remquo, double, double, double, int *)
    CONSIDER(remquof)
    CONSIDER(remquol)
    CONSIDER2(copysign, double, double, double)
    CONSIDER(copysignf)
    CONSIDER(copysignl)
    CONSIDER2(nextafter, double, double, double)
    CONSIDER(nextafterf)
    CONSIDER(nextafterl)
    CONSIDER2(nexttoward, double, double, long double)
    CONSIDER(nexttowardf)
    CONSIDER(nexttowardl)
    CONSIDER2(fdim, double, double, double)
    CONSIDER(fdimf)
    CONSIDER(fdiml)
    CONSIDER2(fmax, double, double, double)
    CONSIDER(fmaxf)
    CONSIDER(fmaxl)
    CONSIDER2(fmin, double, double, double)
    CONSIDER(fminf)
    CONSIDER(fminl)
    CONSIDER2(fabs, double, double)
    CONSIDER2(fma, double, double, double, double)
    CONSIDER(fmaf)
    CONSIDER(fmal)

    if (ci->getName() == "__lgamma_r_finite") {
      updateAnalysis(
          call.getArgOperand(0),
          TypeTree(ConcreteType(Type::getDoubleTy(call.getContext()))).Only(-1),
          &call);
      updateAnalysis(call.getArgOperand(1),
                     TypeTree(BaseType::Integer).Only(0).Only(-1), &call);
      updateAnalysis(
          &call,
          TypeTree(ConcreteType(Type::getDoubleTy(call.getContext()))).Only(-1),
          &call);
    }

    if (ci->getName() == "__cxa_guard_acquire" || ci->getName() == "printf" ||
        ci->getName() == "puts") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1), &call);
    }

    if (!ci->empty()) {
      visitIPOCall(call, *ci);
    }
  }
}

TypeTree TypeAnalyzer::getReturnAnalysis() {
  bool set = false;
  TypeTree vd;
  for (BasicBlock &BB : *fntypeinfo.Function) {
    for (auto &inst : BB) {
      if (auto ri = dyn_cast<ReturnInst>(&inst)) {
        if (auto rv = ri->getReturnValue()) {
          if (set == false) {
            set = true;
            vd = getAnalysis(rv);
            continue;
          }
          vd &= getAnalysis(rv);
        }
      }
    }
  }
  return vd;
}

std::set<int64_t> FnTypeInfo::knownIntegralValues(
    llvm::Value *val, const DominatorTree &DT,
    std::map<Value *, std::set<int64_t>> &intseen) const {
  if (auto constant = dyn_cast<ConstantInt>(val)) {
    return {constant->getSExtValue()};
  }

  assert(KnownValues.size() == Function->getFunctionType()->getNumParams());

  if (auto arg = dyn_cast<llvm::Argument>(val)) {
    auto found = KnownValues.find(arg);
    if (found == KnownValues.end()) {
      for (const auto &pair : KnownValues) {
        llvm::errs() << " KnownValues[" << *pair.first << "] - "
                     << pair.first->getParent()->getName() << "\n";
      }
      llvm::errs() << " arg: " << *arg << " - " << arg->getParent()->getName()
                   << "\n";
    }
    assert(found != KnownValues.end());
    return found->second;
  }

  if (intseen.find(val) != intseen.end())
    return intseen[val];
  intseen[val] = {};

  if (auto ci = dyn_cast<CastInst>(val)) {
    intseen[val] = knownIntegralValues(ci->getOperand(0), DT, intseen);
  }

  auto insert = [&](int64_t v) {
    if (v > -100 && v < 100) {
      intseen[val].insert(v);
    }
  };
  if (auto LI = dyn_cast<LoadInst>(val)) {
    if (auto AI = dyn_cast<AllocaInst>(LI->getPointerOperand())) {
      StoreInst *SI = nullptr;
      bool failed = false;
      for (auto u : AI->users()) {
        if (auto SIu = dyn_cast<StoreInst>(u)) {
          if (SI) {
            failed = true;
            break;
          }
          SI = SIu;
        } else if (!isa<LoadInst>(u)) {
          failed = true;
          break;
        }
      }
      if (SI && !failed && DT.dominates(SI, LI)) {
        for (auto val :
             knownIntegralValues(SI->getValueOperand(), DT, intseen)) {
          insert(val);
        }
      }
    }
  }
  if (auto pn = dyn_cast<PHINode>(val)) {
    for (unsigned i = 0; i < pn->getNumIncomingValues(); ++i) {
      auto a = pn->getIncomingValue(i);
      auto b = pn->getIncomingBlock(i);

      // do not consider loop incoming edges
      if (pn->getParent() == b || DT.dominates(pn, b)) {
        continue;
      }

      auto inset = knownIntegralValues(a, DT, intseen);

      // TODO this here is not fully justified yet
      for (auto pval : inset) {
        if (pval < 20 && pval > -20) {
          insert(pval);
        }
      }

      // if we are an iteration variable, suppose that it could be zero in that
      // range
      // TODO: could actually check the range intercepts 0
      if (auto bo = dyn_cast<BinaryOperator>(a)) {
        if (bo->getOperand(0) == pn || bo->getOperand(1) == pn) {
          if (bo->getOpcode() == BinaryOperator::Add ||
              bo->getOpcode() == BinaryOperator::Sub) {
            insert(0);
          }
        }
      }
    }
    return intseen[val];
  }

  if (auto bo = dyn_cast<BinaryOperator>(val)) {
    auto inset0 = knownIntegralValues(bo->getOperand(0), DT, intseen);
    auto inset1 = knownIntegralValues(bo->getOperand(1), DT, intseen);
    if (bo->getOpcode() == BinaryOperator::Mul) {

      if (inset0.size() == 1 || inset1.size() == 1) {
        for (auto val0 : inset0) {
          for (auto val1 : inset1) {

            insert(val0 * val1);
          }
        }
      }
      if (inset0.count(0) || inset1.count(0)) {
        intseen[val].insert(0);
      }
    }

    if (bo->getOpcode() == BinaryOperator::Add) {
      if (inset0.size() == 1 || inset1.size() == 1) {
        for (auto val0 : inset0) {
          for (auto val1 : inset1) {
            insert(val0 + val1);
          }
        }
      }
    }
    if (bo->getOpcode() == BinaryOperator::Sub) {
      if (inset0.size() == 1 || inset1.size() == 1) {
        for (auto val0 : inset0) {
          for (auto val1 : inset1) {
            insert(val0 - val1);
          }
        }
      }
    }

    if (bo->getOpcode() == BinaryOperator::Shl) {
      if (inset0.size() == 1 || inset1.size() == 1) {
        for (auto val0 : inset0) {
          for (auto val1 : inset1) {
            insert(val0 << val1);
          }
        }
      }
    }

    // TODO note C++ doesnt guarantee behavior of >> being arithmetic or logical
    //     and should replace with llvm apint internal
    if (bo->getOpcode() == BinaryOperator::AShr ||
        bo->getOpcode() == BinaryOperator::LShr) {
      if (inset0.size() == 1 || inset1.size() == 1) {
        for (auto val0 : inset0) {
          for (auto val1 : inset1) {
            insert(val0 >> val1);
          }
        }
      }
    }
  }

  return intseen[val];
}

void TypeAnalyzer::visitIPOCall(CallInst &call, Function &fn) {
  assert(fntypeinfo.KnownValues.size() ==
         fntypeinfo.Function->getFunctionType()->getNumParams());

  FnTypeInfo typeInfo(&fn);

  int argnum = 0;
  for (auto &arg : fn.args()) {
    auto dt = getAnalysis(call.getArgOperand(argnum));
    typeInfo.Arguments.insert(std::pair<Argument *, TypeTree>(&arg, dt));
    typeInfo.KnownValues.insert(std::pair<Argument *, std::set<int64_t>>(
        &arg, fntypeinfo.knownIntegralValues(call.getArgOperand(argnum), DT,
                                             intseen)));
    ++argnum;
  }

  typeInfo.Return = getAnalysis(&call);

  if (PrintType)
    llvm::errs() << " starting IPO of " << call << "\n";

  if (direction & UP) {
    auto a = fn.arg_begin();
    for (size_t i = 0; i < call.getNumArgOperands(); ++i) {
      auto dt = interprocedural.query(a, typeInfo);
      updateAnalysis(call.getArgOperand(i), dt, &call);
      ++a;
    }
  }

  if (direction & DOWN) {
    TypeTree vd = interprocedural.getReturnAnalysis(typeInfo);
    updateAnalysis(&call, vd, &call);
  }
}

TypeResults TypeAnalysis::analyzeFunction(const FnTypeInfo &fn) {
  assert(fn.KnownValues.size() ==
         fn.Function->getFunctionType()->getNumParams());

  auto found = analyzedFunctions.find(fn);
  if (found != analyzedFunctions.end()) {
    auto &analysis = found->second;
    if (analysis.fntypeinfo.Function != fn.Function) {
      llvm::errs() << " queryFunc: " << *fn.Function << "\n";
      llvm::errs() << " analysisFunc: " << *analysis.fntypeinfo.Function
                   << "\n";
    }
    assert(analysis.fntypeinfo.Function == fn.Function);

    return TypeResults(*this, fn);
  }

  auto res = analyzedFunctions.emplace(fn, TypeAnalyzer(fn, *this));
  auto &analysis = res.first->second;

  if (PrintType) {
    llvm::errs() << "analyzing function " << fn.Function->getName() << "\n";
    for (auto &pair : fn.Arguments) {
      llvm::errs() << " + knowndata: " << *pair.first << " : "
                   << pair.second.str();
      auto found = fn.KnownValues.find(pair.first);
      if (found != fn.KnownValues.end()) {
        llvm::errs() << " - " << to_string(found->second);
      }
      llvm::errs() << "\n";
    }
    llvm::errs() << " + retdata: " << fn.Return.str() << "\n";
  }

  analysis.prepareArgs();
  analysis.considerTBAA();
  analysis.run();

  if (analysis.fntypeinfo.Function != fn.Function) {
    llvm::errs() << " queryFunc: " << *fn.Function << "\n";
    llvm::errs() << " analysisFunc: " << *analysis.fntypeinfo.Function << "\n";
  }
  assert(analysis.fntypeinfo.Function == fn.Function);

  {
    auto &analysis = analyzedFunctions.find(fn)->second;
    if (analysis.fntypeinfo.Function != fn.Function) {
      llvm::errs() << " queryFunc: " << *fn.Function << "\n";
      llvm::errs() << " analysisFunc: " << *analysis.fntypeinfo.Function
                   << "\n";
    }
    assert(analysis.fntypeinfo.Function == fn.Function);
  }

  return TypeResults(*this, fn);
}

TypeTree TypeAnalysis::query(Value *val, const FnTypeInfo &fn) {
  assert(val);
  assert(val->getType());

  Function *func = nullptr;
  if (auto arg = dyn_cast<Argument>(val))
    func = arg->getParent();
  else if (auto inst = dyn_cast<Instruction>(val))
    func = inst->getParent()->getParent();
  else if (!isa<Constant>(val)) {
    llvm::errs() << "unknown value: " << *val << "\n";
    assert(0 && "could not handle unknown value type");
  }

  analyzeFunction(fn);
  auto &found = analyzedFunctions.find(fn)->second;
  if (func && found.fntypeinfo.Function != func) {
    llvm::errs() << " queryFunc: " << *func << "\n";
    llvm::errs() << " foundFunc: " << *found.fntypeinfo.Function << "\n";
  }
  assert(!func || found.fntypeinfo.Function == func);
  return found.getAnalysis(val);
}

ConcreteType TypeAnalysis::intType(Value *val, const FnTypeInfo &fn,
                                   bool errIfNotFound) {
  assert(val);
  assert(val->getType());
  auto q = query(val, fn).Data0();
  auto dt = q[{}];
  if (errIfNotFound && (!dt.isKnown() || dt == BaseType::Anything)) {
    if (auto inst = dyn_cast<Instruction>(val)) {
      llvm::errs() << *inst->getParent()->getParent()->getParent() << "\n";
      llvm::errs() << *inst->getParent()->getParent() << "\n";
      for (auto &pair : analyzedFunctions.find(fn)->second.analysis) {
        llvm::errs() << "val: " << *pair.first << " - " << pair.second.str()
                     << "\n";
      }
    }
    llvm::errs() << "could not deduce type of integer " << *val << "\n";
    assert(0 && "could not deduce type of integer");
  }
  return dt;
}

ConcreteType TypeAnalysis::firstPointer(size_t num, Value *val,
                                        const FnTypeInfo &fn,
                                        bool errIfNotFound,
                                        bool pointerIntSame) {
  assert(val);
  assert(val->getType());
  assert(val->getType()->isPointerTy());
  auto q = query(val, fn).Data0();
  auto dt = q[{0}];
  dt.orIn(q[{-1}], pointerIntSame);
  for (size_t i = 1; i < num; ++i) {
    dt.orIn(q[{(int)i}], pointerIntSame);
  }

  if (errIfNotFound && (!dt.isKnown() || dt == BaseType::Anything)) {
    auto &res = analyzedFunctions.find(fn)->second;
    if (auto inst = dyn_cast<Instruction>(val)) {
      llvm::errs() << *inst->getParent()->getParent() << "\n";
      for (auto &pair : res.analysis) {
        if (auto in = dyn_cast<Instruction>(pair.first)) {
          if (in->getParent()->getParent() != inst->getParent()->getParent()) {
            llvm::errs() << "inf: " << *in->getParent()->getParent() << "\n";
            llvm::errs() << "instf: " << *inst->getParent()->getParent()
                         << "\n";
            llvm::errs() << "in: " << *in << "\n";
            llvm::errs() << "inst: " << *inst << "\n";
          }
          assert(in->getParent()->getParent() ==
                 inst->getParent()->getParent());
        }
        llvm::errs() << "val: " << *pair.first << " - " << pair.second.str()
                     << " int: " +
                            to_string(res.knownIntegralValues(pair.first))
                     << "\n";
      }
    }
    if (auto arg = dyn_cast<Argument>(val)) {
      llvm::errs() << *arg->getParent() << "\n";
      for (auto &pair : res.analysis) {
        if (auto in = dyn_cast<Instruction>(pair.first))
          assert(in->getParent()->getParent() == arg->getParent());
        llvm::errs() << "val: " << *pair.first << " - " << pair.second.str()
                     << " int: " +
                            to_string(res.knownIntegralValues(pair.first))
                     << "\n";
      }
    }
    llvm::errs() << "fn: " << *fn.Function << "\n";
    analyzeFunction(fn).dump();
    llvm::errs() << "could not deduce type of integer " << *val
                 << " num:" << num << " q:" << q.str() << " \n";

    llvm::DiagnosticLocation loc = fn.Function->getSubprogram();
    Instruction *codeLoc = &*fn.Function->getEntryBlock().begin();
    if (auto inst = dyn_cast<Instruction>(val)) {
      loc = inst->getDebugLoc();
      codeLoc = inst;
    }
    EmitFailure("CannotDeduceType", loc, codeLoc,
                "failed to deduce type of value ", *val);

    assert(0 && "could not deduce type of integer");
  }
  return dt;
}

TypeResults::TypeResults(TypeAnalysis &analysis, const FnTypeInfo &fn)
    : analysis(analysis), info(fn) {
  assert(fn.KnownValues.size() ==
         fn.Function->getFunctionType()->getNumParams());
}

FnTypeInfo TypeResults::getAnalyzedTypeInfo() {
  FnTypeInfo res(info.Function);
  for (auto &arg : info.Function->args()) {
    res.Arguments.insert(
        std::pair<Argument *, TypeTree>(&arg, analysis.query(&arg, info)));
  }
  res.Return = getReturnAnalysis();
  res.KnownValues = info.KnownValues;
  return res;
}

TypeTree TypeResults::query(Value *val) {
  if (auto inst = dyn_cast<Instruction>(val)) {
    assert(inst->getParent()->getParent() == info.Function);
  }
  if (auto arg = dyn_cast<Argument>(val)) {
    assert(arg->getParent() == info.Function);
  }
  for (auto &pair : info.Arguments) {
    assert(pair.first->getParent() == info.Function);
  }
  return analysis.query(val, info);
}

void TypeResults::dump() {
  assert(analysis.analyzedFunctions.find(info) !=
         analysis.analyzedFunctions.end());
  analysis.analyzedFunctions.find(info)->second.dump();
}

ConcreteType TypeResults::intType(Value *val, bool errIfNotFound) {
  return analysis.intType(val, info, errIfNotFound);
}

ConcreteType TypeResults::firstPointer(size_t num, Value *val,
                                       bool errIfNotFound,
                                       bool pointerIntSame) {
  return analysis.firstPointer(num, val, info, errIfNotFound, pointerIntSame);
}

TypeTree TypeResults::getReturnAnalysis() {
  return analysis.getReturnAnalysis(info);
}

std::set<int64_t> TypeResults::knownIntegralValues(Value *val) const {
  auto found = analysis.analyzedFunctions.find(info);
  assert(found != analysis.analyzedFunctions.end());
  auto &sub = found->second;
  return sub.knownIntegralValues(val);
}

std::set<int64_t> TypeAnalyzer::knownIntegralValues(Value *val) {
  return fntypeinfo.knownIntegralValues(val, DT, intseen);
}
