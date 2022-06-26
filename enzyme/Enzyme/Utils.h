//===- Utils.h - Declaration of miscellaneous utilities -------------------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @misc{enzymeGithub,
//  author = {William S. Moses and Valentin Churavy},
//  title = {Enzyme: High Performance Automatic Differentiation of LLVM},
//  year = {2020},
//  howpublished = {\url{https://github.com/wsmoses/Enzyme}},
//  note = {commit xxxxxxx}
// }
//
//===----------------------------------------------------------------------===//
//
// This file declares miscellaneous utilities that are used as part of the
// AD process.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_UTILS_H
#define ENZYME_UTILS_H

#include "llvm/ADT/SmallPtrSet.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"

#include "llvm/IR/ValueMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/CommandLine.h"

#include "llvm/ADT/SetVector.h"

#include "llvm/IR/Dominators.h"

#if LLVM_VERSION_MAJOR >= 10
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#endif

#include <map>
#include <set>

#include "llvm/IR/DiagnosticInfo.h"

#include "llvm/Analysis/OptimizationRemarkEmitter.h"

#include "TypeAnalysis/ConcreteType.h"

namespace llvm {
class ScalarEvolution;
}

enum class ErrorType {
  NoDerivative = 0,
  NoShadow = 1,
  IllegalTypeAnalysis = 2,
  NoType = 3
};

extern "C" {
/// Print additional debug info relevant to performance
extern llvm::cl::opt<bool> EnzymePrintPerf;
extern void (*CustomErrorHandler)(const char *, LLVMValueRef, ErrorType,
                                  void *);
}

extern std::map<std::string, std::function<llvm::Value *(
                                 llvm::IRBuilder<> &, llvm::CallInst *,
                                 llvm::ArrayRef<llvm::Value *>)>>
    shadowHandlers;

template <typename... Args>
void EmitFailure(llvm::StringRef RemarkName,
                 const llvm::DiagnosticLocation &Loc,
                 const llvm::Instruction *CodeRegion, Args &...args) {

  llvm::OptimizationRemarkEmitter ORE(CodeRegion->getParent()->getParent());
  std::string str;
  llvm::raw_string_ostream ss(str);
  (ss << ... << args);
  ORE.emit(llvm::DiagnosticInfoOptimizationFailure("enzyme", RemarkName, Loc,
                                                   CodeRegion->getParent())
           << ss.str());
}

template <typename... Args>
void EmitWarning(llvm::StringRef RemarkName,
                 const llvm::DiagnosticLocation &Loc, const llvm::Function *F,
                 const llvm::BasicBlock *BB, const Args &...args) {

  llvm::OptimizationRemarkEmitter ORE(F);
  ORE.emit([&]() {
    std::string str;
    llvm::raw_string_ostream ss(str);
    (ss << ... << args);
    return llvm::OptimizationRemark("enzyme", RemarkName, Loc, BB) << ss.str();
  });
  if (EnzymePrintPerf)
    (llvm::errs() << ... << args) << "\n";
}

template <typename... Args>
void EmitWarning(llvm::StringRef RemarkName, const llvm::Function *F,
                 const Args &...args) {

  llvm::OptimizationRemarkEmitter ORE(F);
  ORE.emit([&]() {
    std::string str;
    llvm::raw_string_ostream ss(str);
    (ss << ... << args);
    return llvm::OptimizationRemark("enzyme", RemarkName, F) << ss.str();
  });
  if (EnzymePrintPerf)
    (llvm::errs() << ... << args) << "\n";
}

class EnzymeFailure : public llvm::DiagnosticInfoIROptimization {
public:
  EnzymeFailure(llvm::StringRef RemarkName, const llvm::DiagnosticLocation &Loc,
                const llvm::Instruction *CodeRegion);

  static llvm::DiagnosticKind ID();
  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == ID();
  }

  /// \see DiagnosticInfoOptimizationBase::isEnabled.
  bool isEnabled() const override;
};

static inline llvm::Function *isCalledFunction(llvm::Value *val) {
  if (llvm::CallInst *CI = llvm::dyn_cast<llvm::CallInst>(val)) {
    return CI->getCalledFunction();
  }
  return nullptr;
}

/// Get LLVM fast math flags
static inline llvm::FastMathFlags getFast() {
  llvm::FastMathFlags f;
  f.set();
  return f;
}

/// Pick the maximum value
template <typename T> static inline T max(T a, T b) {
  if (a > b)
    return a;
  return b;
}
/// Pick the maximum value
template <typename T> static inline T min(T a, T b) {
  if (a < b)
    return a;
  return b;
}

/// Output a set as a string
template <typename T>
static inline std::string to_string(const std::set<T> &us) {
  std::string s = "{";
  for (const auto &y : us)
    s += std::to_string(y) + ",";
  return s + "}";
}

/// Print a map, optionally with a shouldPrint function
/// to decide to print a given value
template <typename T, typename N>
static inline void dumpMap(
    const llvm::ValueMap<T, N> &o,
    std::function<bool(const llvm::Value *)> shouldPrint = [](T) {
      return true;
    }) {
  llvm::errs() << "<begin dump>\n";
  for (auto a : o) {
    if (shouldPrint(a.first))
      llvm::errs() << "key=" << *a.first << " val=" << *a.second << "\n";
  }
  llvm::errs() << "</end dump>\n";
}

/// Print a set
template <typename T>
static inline void dumpSet(const llvm::SmallPtrSetImpl<T *> &o) {
  llvm::errs() << "<begin dump>\n";
  for (auto a : o)
    llvm::errs() << *a << "\n";
  llvm::errs() << "</end dump>\n";
}

template <typename T>
static inline void dumpSet(const llvm::SetVector<T *> &o) {
  llvm::errs() << "<begin dump>\n";
  for (auto a : o)
    llvm::errs() << *a << "\n";
  llvm::errs() << "</end dump>\n";
}

/// Get the next non-debug instruction, if one exists
static inline llvm::Instruction *
getNextNonDebugInstructionOrNull(llvm::Instruction *Z) {
  for (llvm::Instruction *I = Z->getNextNode(); I; I = I->getNextNode())
    if (!llvm::isa<llvm::DbgInfoIntrinsic>(I))
      return I;
  return nullptr;
}

/// Get the next non-debug instruction, erring if none exists
static inline llvm::Instruction *
getNextNonDebugInstruction(llvm::Instruction *Z) {
  auto z = getNextNonDebugInstructionOrNull(Z);
  if (z)
    return z;
  llvm::errs() << *Z->getParent() << "\n";
  llvm::errs() << *Z << "\n";
  llvm_unreachable("No valid subsequent non debug instruction");
  exit(1);
  return nullptr;
}

/// Check if a global has metadata
static inline llvm::MDNode *hasMetadata(const llvm::GlobalObject *O,
                                        llvm::StringRef kind) {
  return O->getMetadata(kind);
}

/// Check if an instruction has metadata
static inline llvm::MDNode *hasMetadata(const llvm::Instruction *O,
                                        llvm::StringRef kind) {
  return O->getMetadata(kind);
}

/// Potential return type of generated functions
enum class ReturnType {
  /// Return is a struct of all args and the original return
  ArgsWithReturn,
  /// Return is a struct of all args and two of the original return
  ArgsWithTwoReturns,
  /// Return is a struct of all args
  Args,
  /// Return is a tape type and the original return
  TapeAndReturn,
  /// Return is a tape type and the two of the original return
  TapeAndTwoReturns,
  /// Return is a tape type
  Tape,
  TwoReturns,
  Return,
  Void,
};

/// Potential differentiable argument classifications
enum class DIFFE_TYPE {
  OUT_DIFF = 0,  // add differential to an output struct
  DUP_ARG = 1,   // duplicate the argument and store differential inside
  CONSTANT = 2,  // no differential
  DUP_NONEED = 3 // duplicate this argument and store differential inside, but
                 // don't need the forward
};

enum class DerivativeMode {
  ForwardMode = 0,
  ReverseModePrimal = 1,
  ReverseModeGradient = 2,
  ReverseModeCombined = 3,
  ForwardModeSplit = 4,
};

/// Classification of value as an original program
/// variable, a derivative variable, neither, or both.
/// This type is used both in differential use analysis
/// and to describe argument bundles.
enum class ValueType {
  // A value that is neither a value in the original
  // program, nor the derivative.
  None = 0,
  // The original program value
  Primal = 1,
  // The derivative value
  Shadow = 2,
  // Both the original program value and the shadow.
  Both = Primal | Shadow,
};

static inline std::string to_string(DerivativeMode mode) {
  switch (mode) {
  case DerivativeMode::ForwardMode:
    return "ForwardMode";
  case DerivativeMode::ForwardModeSplit:
    return "ForwardModeSplit";
  case DerivativeMode::ReverseModePrimal:
    return "ReverseModePrimal";
  case DerivativeMode::ReverseModeGradient:
    return "ReverseModeGradient";
  case DerivativeMode::ReverseModeCombined:
    return "ReverseModeCombined";
  }
  llvm_unreachable("illegal derivative mode");
}

/// Convert DIFFE_TYPE to a string
static inline std::string to_string(DIFFE_TYPE t) {
  switch (t) {
  case DIFFE_TYPE::OUT_DIFF:
    return "OUT_DIFF";
  case DIFFE_TYPE::CONSTANT:
    return "CONSTANT";
  case DIFFE_TYPE::DUP_ARG:
    return "DUP_ARG";
  case DIFFE_TYPE::DUP_NONEED:
    return "DUP_NONEED";
  default:
    assert(0 && "illegal diffetype");
    return "";
  }
}

/// Convert ReturnType to a string
static inline std::string to_string(ReturnType t) {
  switch (t) {
  case ReturnType::ArgsWithReturn:
    return "ArgsWithReturn";
  case ReturnType::ArgsWithTwoReturns:
    return "ArgsWithTwoReturns";
  case ReturnType::Args:
    return "Args";
  case ReturnType::TapeAndReturn:
    return "TapeAndReturn";
  case ReturnType::TapeAndTwoReturns:
    return "TapeAndTwoReturns";
  case ReturnType::Tape:
    return "Tape";
  case ReturnType::TwoReturns:
    return "TwoReturns";
  case ReturnType::Return:
    return "Return";
  case ReturnType::Void:
    return "Void";
  }
  llvm_unreachable("illegal ReturnType");
}

#include <set>

/// Attempt to automatically detect the differentiable
/// classification based off of a given type
static inline DIFFE_TYPE whatType(llvm::Type *arg, DerivativeMode mode,
                                  std::set<llvm::Type *> seen = {}) {
  assert(arg);
  if (seen.find(arg) != seen.end())
    return DIFFE_TYPE::CONSTANT;
  seen.insert(arg);

  if (arg->isVoidTy() || arg->isEmptyTy()) {
    return DIFFE_TYPE::CONSTANT;
  }

  if (arg->isPointerTy()) {
    switch (whatType(arg->getPointerElementType(), mode, seen)) {
    case DIFFE_TYPE::OUT_DIFF:
      return DIFFE_TYPE::DUP_ARG;
    case DIFFE_TYPE::CONSTANT:
      return DIFFE_TYPE::CONSTANT;
    case DIFFE_TYPE::DUP_ARG:
      return DIFFE_TYPE::DUP_ARG;
    case DIFFE_TYPE::DUP_NONEED:
      llvm_unreachable("impossible case");
    }
    assert(arg);
    llvm::errs() << "arg: " << *arg << "\n";
    assert(0 && "Cannot handle type0");
    return DIFFE_TYPE::CONSTANT;
  } else if (arg->isArrayTy()) {
    return whatType(llvm::cast<llvm::ArrayType>(arg)->getElementType(), mode,
                    seen);
  } else if (arg->isStructTy()) {
    auto st = llvm::cast<llvm::StructType>(arg);
    if (st->getNumElements() == 0)
      return DIFFE_TYPE::CONSTANT;

    auto ty = DIFFE_TYPE::CONSTANT;
    for (unsigned i = 0; i < st->getNumElements(); ++i) {
      auto midTy = whatType(st->getElementType(i), mode, seen);
      switch (midTy) {
      case DIFFE_TYPE::OUT_DIFF:
        switch (ty) {
        case DIFFE_TYPE::OUT_DIFF:
        case DIFFE_TYPE::CONSTANT:
          ty = DIFFE_TYPE::OUT_DIFF;
          break;
        case DIFFE_TYPE::DUP_ARG:
          ty = DIFFE_TYPE::DUP_ARG;
          return ty;
        case DIFFE_TYPE::DUP_NONEED:
          llvm_unreachable("impossible case");
        }
        break;
      case DIFFE_TYPE::CONSTANT:
        switch (ty) {
        case DIFFE_TYPE::OUT_DIFF:
          ty = DIFFE_TYPE::OUT_DIFF;
          break;
        case DIFFE_TYPE::CONSTANT:
          break;
        case DIFFE_TYPE::DUP_ARG:
          ty = DIFFE_TYPE::DUP_ARG;
          return ty;
        case DIFFE_TYPE::DUP_NONEED:
          llvm_unreachable("impossible case");
        }
        break;
      case DIFFE_TYPE::DUP_ARG:
        return DIFFE_TYPE::DUP_ARG;
      case DIFFE_TYPE::DUP_NONEED:
        llvm_unreachable("impossible case");
      }
    }
    return ty;
  } else if (arg->isIntOrIntVectorTy() || arg->isFunctionTy()) {
    return DIFFE_TYPE::CONSTANT;
  } else if (arg->isFPOrFPVectorTy()) {
    return (mode == DerivativeMode::ForwardMode ||
            mode == DerivativeMode::ForwardModeSplit)
               ? DIFFE_TYPE::DUP_ARG
               : DIFFE_TYPE::OUT_DIFF;
  } else {
    assert(arg);
    llvm::errs() << "arg: " << *arg << "\n";
    assert(0 && "Cannot handle type");
    return DIFFE_TYPE::CONSTANT;
  }
}

/// Check whether this instruction is returned
static inline bool isReturned(llvm::Instruction *inst) {
  for (const auto a : inst->users()) {
    if (llvm::isa<llvm::ReturnInst>(a))
      return true;
  }
  return false;
}

/// Convert a floating point type to an integer type
/// of the same size
static inline llvm::Type *FloatToIntTy(llvm::Type *T) {
  assert(T->isFPOrFPVectorTy());
  if (auto ty = llvm::dyn_cast<llvm::VectorType>(T)) {
    return llvm::VectorType::get(FloatToIntTy(ty->getElementType()),
#if LLVM_VERSION_MAJOR >= 11
                                 ty->getElementCount());
#else
                                 ty->getNumElements());
#endif
  }
  if (T->isHalfTy())
    return llvm::IntegerType::get(T->getContext(), 16);
  if (T->isFloatTy())
    return llvm::IntegerType::get(T->getContext(), 32);
  if (T->isDoubleTy())
    return llvm::IntegerType::get(T->getContext(), 64);
  assert(0 && "unknown floating point type");
  return nullptr;
}

/// Convert a integer type to a floating point type
/// of the same size
static inline llvm::Type *IntToFloatTy(llvm::Type *T) {
  assert(T->isIntOrIntVectorTy());
  if (auto ty = llvm::dyn_cast<llvm::VectorType>(T)) {
    return llvm::VectorType::get(IntToFloatTy(ty->getElementType()),
#if LLVM_VERSION_MAJOR >= 11
                                 ty->getElementCount());
#else
                                 ty->getNumElements());
#endif
  }
  if (auto ty = llvm::dyn_cast<llvm::IntegerType>(T)) {
    switch (ty->getBitWidth()) {
    case 16:
      return llvm::Type::getHalfTy(T->getContext());
    case 32:
      return llvm::Type::getFloatTy(T->getContext());
    case 64:
      return llvm::Type::getDoubleTy(T->getContext());
    }
  }
  assert(0 && "unknown int to floating point type");
  return nullptr;
}

// TODO replace/rename
/// Determine whether this function is a certain malloc free
/// debug or lifetime
static inline bool isCertainMallocOrFree(llvm::Function *called) {
  if (called == nullptr)
    return false;
  if (called->getName() == "printf" || called->getName() == "puts" ||
      called->getName() == "malloc" || called->getName() == "_Znwm" ||
      called->getName() == "_ZdlPv" || called->getName() == "_ZdlPvm" ||
      called->getName() == "free" || called->getName() == "swift_allocObject" ||
      called->getName() == "swift_release" ||
      shadowHandlers.find(called->getName().str()) != shadowHandlers.end())
    return true;
  switch (called->getIntrinsicID()) {
  case llvm::Intrinsic::dbg_declare:
  case llvm::Intrinsic::dbg_value:
#if LLVM_VERSION_MAJOR > 6
  case llvm::Intrinsic::dbg_label:
#endif
  case llvm::Intrinsic::dbg_addr:
  case llvm::Intrinsic::lifetime_start:
  case llvm::Intrinsic::lifetime_end:
    return true;
  default:
    break;
  }

  return false;
}

// TODO replace/rename
/// Determine whether this function is a certain print free
/// debug or lifetime
static inline bool isCertainPrintOrFree(llvm::Function *called) {
  if (called == nullptr)
    return false;

  if (called->getName() == "printf" || called->getName() == "puts" ||
      called->getName() == "fprintf" || called->getName() == "putchar" ||
      called->getName().startswith(
          "_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_") ||
      called->getName().startswith("_ZNSolsE") ||
      called->getName().startswith("_ZNSo9_M_insert") ||
      called->getName().startswith("_ZSt16__ostream_insert") ||
      called->getName().startswith("_ZNSo3put") ||
      called->getName().startswith("_ZSt4endl") ||
      called->getName().startswith("_ZN3std2io5stdio6_print") ||
      called->getName().startswith("_ZNSo5flushEv") ||
      called->getName().startswith("_ZN4core3fmt") ||
      called->getName() == "vprintf" || called->getName() == "_ZdlPv" ||
      called->getName() == "_ZdlPvm" || called->getName() == "free" ||
      called->getName() == "swift_release")
    return true;
  switch (called->getIntrinsicID()) {
  case llvm::Intrinsic::dbg_declare:
  case llvm::Intrinsic::dbg_value:
#if LLVM_VERSION_MAJOR > 6
  case llvm::Intrinsic::dbg_label:
#endif
  case llvm::Intrinsic::dbg_addr:
  case llvm::Intrinsic::lifetime_start:
  case llvm::Intrinsic::lifetime_end:
    return true;
  default:
    break;
  }
  return false;
}

// TODO replace/rename
/// Determine whether this function is a certain print malloc free
/// debug or lifetime
static inline bool isCertainPrintMallocOrFree(llvm::Function *called) {
  if (called == nullptr)
    return false;

  if (isCertainPrintOrFree(called))
    return true;

  if (isCertainMallocOrFree(called))
    return true;

  return false;
}

/// Create function for type that performs the derivative memcpy on floating
/// point memory
llvm::Function *
getOrInsertDifferentialFloatMemcpy(llvm::Module &M, llvm::Type *T,
                                   unsigned dstalign, unsigned srcalign,
                                   unsigned dstaddr, unsigned srcaddr);

/// Create function for type that performs memcpy with a stride
llvm::Function *getOrInsertMemcpyStrided(llvm::Module &M, llvm::PointerType *T,
                                         llvm::Type *IT, unsigned dstalign,
                                         unsigned srcalign);

/// Create function for type that performs the derivative memmove on floating
/// point memory
llvm::Function *
getOrInsertDifferentialFloatMemmove(llvm::Module &M, llvm::Type *T,
                                    unsigned dstalign, unsigned srcalign,
                                    unsigned dstaddr, unsigned srcaddr);

llvm::Function *getOrInsertCheckedFree(llvm::Module &M, llvm::CallInst *call,
                                       llvm::Type *Type, unsigned width);

/// Create function for type that performs the derivative MPI_Wait
llvm::Function *getOrInsertDifferentialMPI_Wait(llvm::Module &M,
                                                llvm::ArrayRef<llvm::Type *> T,
                                                llvm::Type *reqType);

/// Create function to computer nearest power of two
llvm::Value *nextPowerOfTwo(llvm::IRBuilder<> &B, llvm::Value *V);

/// Insert into a map
template <typename K, typename V>
static inline typename std::map<K, V>::iterator
insert_or_assign(std::map<K, V> &map, K &key, V &&val) {
  auto found = map.find(key);
  if (found != map.end()) {
    map.erase(found);
  }
  return map.emplace(key, val).first;
}

/// Insert into a map
template <typename K, typename V>
static inline typename std::map<K, V>::iterator
insert_or_assign2(std::map<K, V> &map, K key, V val) {
  auto found = map.find(key);
  if (found != map.end()) {
    map.erase(found);
  }
  return map.emplace(key, val).first;
}

template <typename K, typename V>
static inline V *findInMap(std::map<K, V> &map, K key) {
  auto found = map.find(key);
  if (found == map.end())
    return nullptr;
  V *val = &found->second;
  return val;
}

#include "llvm/IR/CFG.h"
#include <deque>
#include <functional>
/// Call the function f for all instructions that happen after inst
/// If the function returns true, the iteration will early exit
static inline void allFollowersOf(llvm::Instruction *inst,
                                  std::function<bool(llvm::Instruction *)> f) {

  for (auto uinst = inst->getNextNode(); uinst != nullptr;
       uinst = uinst->getNextNode()) {
    if (f(uinst))
      return;
  }

  std::deque<llvm::BasicBlock *> todo;
  std::set<llvm::BasicBlock *> done;
  for (auto suc : llvm::successors(inst->getParent())) {
    todo.push_back(suc);
  }
  while (todo.size()) {
    auto BB = todo.front();
    todo.pop_front();
    if (done.count(BB))
      continue;
    done.insert(BB);
    for (auto &ni : *BB) {
      if (f(&ni))
        return;
      if (&ni == inst)
        break;
    }
    for (auto suc : llvm::successors(BB)) {
      todo.push_back(suc);
    }
  }
}

/// Call the function f for all instructions that happen before inst
/// If the function returns true, the iteration will early exit
static inline void
allPredecessorsOf(llvm::Instruction *inst,
                  std::function<bool(llvm::Instruction *)> f) {

  for (auto uinst = inst->getPrevNode(); uinst != nullptr;
       uinst = uinst->getPrevNode()) {
    if (f(uinst))
      return;
  }

  std::deque<llvm::BasicBlock *> todo;
  std::set<llvm::BasicBlock *> done;
  for (auto suc : llvm::predecessors(inst->getParent())) {
    todo.push_back(suc);
  }
  while (todo.size()) {
    auto BB = todo.front();
    todo.pop_front();
    if (done.count(BB))
      continue;
    done.insert(BB);

    llvm::BasicBlock::reverse_iterator I = BB->rbegin(), E = BB->rend();
    for (; I != E; ++I) {
      if (f(&*I))
        return;
      if (&*I == inst)
        break;
    }
    for (auto suc : llvm::predecessors(BB)) {
      todo.push_back(suc);
    }
  }
}

/// Call the function f for all instructions that happen before inst
/// If the function returns true, the iteration will early exit
static inline void
allDomPredecessorsOf(llvm::Instruction *inst, llvm::DominatorTree &DT,
                     std::function<bool(llvm::Instruction *)> f) {

  for (auto uinst = inst->getPrevNode(); uinst != nullptr;
       uinst = uinst->getPrevNode()) {
    if (f(uinst))
      return;
  }

  std::deque<llvm::BasicBlock *> todo;
  std::set<llvm::BasicBlock *> done;
  for (auto suc : llvm::predecessors(inst->getParent())) {
    todo.push_back(suc);
  }
  while (todo.size()) {
    auto BB = todo.front();
    todo.pop_front();
    if (done.count(BB))
      continue;
    done.insert(BB);

    if (DT.properlyDominates(BB, inst->getParent())) {
      llvm::BasicBlock::reverse_iterator I = BB->rbegin(), E = BB->rend();
      for (; I != E; ++I) {
        if (f(&*I))
          return;
        if (&*I == inst)
          break;
      }
      for (auto suc : llvm::predecessors(BB)) {
        todo.push_back(suc);
      }
    }
  }
}

/// Call the function f for all instructions that happen before inst
/// If the function returns true, the iteration will early exit
static inline void
allUnsyncdPredecessorsOf(llvm::Instruction *inst,
                         std::function<bool(llvm::Instruction *)> f,
                         std::function<void()> preEntry) {

  for (auto uinst = inst->getPrevNode(); uinst != nullptr;
       uinst = uinst->getPrevNode()) {
    if (auto II = llvm::dyn_cast<llvm::IntrinsicInst>(uinst)) {
      if (II->getIntrinsicID() == llvm::Intrinsic::nvvm_barrier0 ||
          II->getIntrinsicID() == llvm::Intrinsic::amdgcn_s_barrier) {
        return;
      }
    }
    if (f(uinst))
      return;
  }

  std::deque<llvm::BasicBlock *> todo;
  std::set<llvm::BasicBlock *> done;
  for (auto suc : llvm::predecessors(inst->getParent())) {
    todo.push_back(suc);
  }
  while (todo.size()) {
    auto BB = todo.front();
    todo.pop_front();
    if (done.count(BB))
      continue;
    done.insert(BB);

    bool syncd = false;
    llvm::BasicBlock::reverse_iterator I = BB->rbegin(), E = BB->rend();
    for (; I != E; ++I) {
      if (auto II = llvm::dyn_cast<llvm::IntrinsicInst>(&*I)) {
        if (II->getIntrinsicID() == llvm::Intrinsic::nvvm_barrier0 ||
            II->getIntrinsicID() == llvm::Intrinsic::amdgcn_s_barrier) {
          syncd = true;
          break;
        }
      }
      if (f(&*I))
        return;
      if (&*I == inst)
        break;
    }
    if (!syncd) {
      for (auto suc : llvm::predecessors(BB)) {
        todo.push_back(suc);
      }
      if (&BB->getParent()->getEntryBlock() == BB) {
        preEntry();
      }
    }
  }
}

#include "llvm/Analysis/LoopInfo.h"

static inline llvm::Loop *getAncestor(llvm::Loop *R1, llvm::Loop *R2) {
  if (!R1 || !R2)
    return nullptr;
  for (llvm::Loop *L1 = R1; L1; L1 = L1->getParentLoop())
    for (llvm::Loop *L2 = R2; L2; L2 = L2->getParentLoop()) {
      if (L1 == L2) {
        return L1;
      }
    }
  return nullptr;
}

// Add all of the stores which may execute after the instruction `inst`
// into the resutls vector.
void mayExecuteAfter(llvm::SmallVectorImpl<llvm::Instruction *> &results,
                     llvm::Instruction *inst,
                     const llvm::SmallPtrSetImpl<llvm::Instruction *> &stores,
                     const llvm::Loop *region);

/// Return whether maybeReader can read from memory written to by maybeWriter
bool writesToMemoryReadBy(llvm::AAResults &AA, llvm::Instruction *maybeReader,
                          llvm::Instruction *maybeWriter);

// A more advanced version of writesToMemoryReadBy, where the writing
// instruction comes after the reading function. Specifically, even if the two
// instructions may access the same location, this variant checks whether
// also checks whether ScalarEvolution ensures that a subsequent write will not
// overwrite the value read by the load.
//   A simple example: the load/store might write/read from the same
//   location. However, no store will overwrite a previous load.
//   for(int i=0; i<N; i++) {
//      load A[i-1]
//      store A[i] = ...
//   }
bool overwritesToMemoryReadBy(llvm::AAResults &AA, llvm::ScalarEvolution &SE,
                              llvm::LoopInfo &LI, llvm::DominatorTree &DT,
                              llvm::Instruction *maybeReader,
                              llvm::Instruction *maybeWriter,
                              llvm::Loop *scope = nullptr);
static inline void
/// Call the function f for all instructions that happen between inst1 and inst2
/// If the function returns true, the iteration will early exit
allInstructionsBetween(llvm::LoopInfo &LI, llvm::Instruction *inst1,
                       llvm::Instruction *inst2,
                       std::function<bool(llvm::Instruction *)> f) {
  assert(inst1->getParent()->getParent() == inst2->getParent()->getParent());
  for (auto uinst = inst1->getNextNode(); uinst != nullptr;
       uinst = uinst->getNextNode()) {
    if (f(uinst))
      return;
    if (uinst == inst2)
      return;
  }

  std::set<llvm::Instruction *> instructions;

  llvm::Loop *l1 = LI.getLoopFor(inst1->getParent());
  while (l1 && !l1->contains(inst2->getParent()))
    l1 = l1->getParentLoop();

  // Do all instructions from inst1 up to first instance of inst2's start block
  {
    std::deque<llvm::BasicBlock *> todo;
    std::set<llvm::BasicBlock *> done;
    for (auto suc : llvm::successors(inst1->getParent())) {
      todo.push_back(suc);
    }
    while (todo.size()) {
      auto BB = todo.front();
      todo.pop_front();
      if (done.count(BB))
        continue;
      done.insert(BB);

      for (auto &ni : *BB) {
        instructions.insert(&ni);
      }
      for (auto suc : llvm::successors(BB)) {
        if (!l1 || suc != l1->getHeader()) {
          todo.push_back(suc);
        }
      }
    }
  }

  allPredecessorsOf(inst2, [&](llvm::Instruction *I) -> bool {
    if (instructions.find(I) == instructions.end())
      return /*earlyReturn*/ false;
    return f(I);
  });
}

enum class MPI_CallType {
  ISEND = 1,
  IRECV = 2,
};

enum class MPI_Elem {
  Buf = 0,
  Count = 1,
  DataType = 2,
  Src = 3,
  Tag = 4,
  Comm = 5,
  Call = 6,
  Old = 7
};

static inline llvm::StructType *getMPIHelper(llvm::LLVMContext &Context) {
  using namespace llvm;
  auto i64 = Type::getInt64Ty(Context);
  Type *types[] = {
      /*buf      0 */ Type::getInt8PtrTy(Context),
      /*count    1 */ i64,
      /*datatype 2 */ Type::getInt8PtrTy(Context),
      /*src      3 */ i64,
      /*tag      4 */ i64,
      /*comm     5 */ Type::getInt8PtrTy(Context),
      /*fn       6 */ Type::getInt8Ty(Context),
      /*old      7 */ Type::getInt8PtrTy(Context),
  };
  return StructType::get(Context, types, false);
}

template <MPI_Elem E, bool Pointer = true>
static inline llvm::Value *getMPIMemberPtr(llvm::IRBuilder<> &B,
                                           llvm::Value *V) {
  using namespace llvm;
  auto i64 = Type::getInt64Ty(V->getContext());
  auto i32 = Type::getInt32Ty(V->getContext());
  auto c0_64 = ConstantInt::get(i64, 0);

  if (Pointer) {
#if LLVM_VERSION_MAJOR > 7
    return B.CreateInBoundsGEP(V->getType()->getPointerElementType(), V,
                               {c0_64, ConstantInt::get(i32, (uint64_t)E)});
#else
    return B.CreateInBoundsGEP(V, {c0_64, ConstantInt::get(i32, (uint64_t)E)});
#endif
  } else {
    return B.CreateExtractValue(V, {(unsigned)E});
  }
}

llvm::Value *getOrInsertOpFloatSum(llvm::Module &M, llvm::Type *OpPtr,
                                   ConcreteType CT, llvm::Type *intType,
                                   llvm::IRBuilder<> &B2);
llvm::Function *getOrInsertExponentialAllocator(llvm::Module &M, bool ZeroInit);

class AssertingReplacingVH : public llvm::CallbackVH {
public:
  AssertingReplacingVH() = default;

  AssertingReplacingVH(llvm::Value *new_value) { setValPtr(new_value); }

  void deleted() override final {
    assert(0 && "attempted to delete value with remaining handle use");
    llvm_unreachable("attempted to delete value with remaining handle use");
  }

  void allUsesReplacedWith(llvm::Value *new_value) override final {
    setValPtr(new_value);
  }
  virtual ~AssertingReplacingVH() {}
};

template <typename T> static inline llvm::Function *getFunctionFromCall(T *op) {
  llvm::Function *called = nullptr;
  using namespace llvm;
  llvm::Value *callVal;
#if LLVM_VERSION_MAJOR >= 11
  callVal = op->getCalledOperand();
#else
  callVal = op->getCalledValue();
#endif

  while (!called) {
    if (auto castinst = dyn_cast<ConstantExpr>(callVal))
      if (castinst->isCast()) {
        callVal = castinst->getOperand(0);
        continue;
      }
    if (auto fn = dyn_cast<Function>(callVal)) {
      called = fn;
      break;
    }
#if LLVM_VERSION_MAJOR >= 11
    if (auto alias = dyn_cast<GlobalAlias>(callVal)) {
      callVal = dyn_cast<Function>(alias->getAliasee());
      continue;
    }
#endif
    break;
  }
  return called;
}

llvm::Function *
getOrInsertDifferentialWaitallSave(llvm::Module &M,
                                   llvm::ArrayRef<llvm::Type *> T,
                                   llvm::PointerType *reqType);

void ErrorIfRuntimeInactive(llvm::IRBuilder<> &B, llvm::Value *primal,
                            llvm::Value *shadow, const char *Message);
#endif
