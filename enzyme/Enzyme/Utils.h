/*
 * Utils.h
 *
 * Copyright (C) 2020 William S. Moses (enzyme@wsmoses.com) - All Rights
 * Reserved
 *
 * For commercial use of this code please contact the author(s) above.
 */

#ifndef ENZYME_UTILS_H
#define ENZYME_UTILS_H

#include "llvm/ADT/SmallPtrSet.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"

#include "llvm/IR/ValueMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <set>

static inline llvm::FastMathFlags getFast() {
  llvm::FastMathFlags f;
  f.set();
  return f;
}

template <typename T> static inline T max(T a, T b) {
  if (a > b)
    return a;
  return b;
}

template <typename T>
static inline std::string to_string(const std::set<T> &us) {
  std::string s = "{";
  for (const auto &y : us)
    s += std::to_string(y) + ",";
  return s + "}";
}

template <typename T, typename N>
static inline void
dumpMap(const llvm::ValueMap<T, N> &o,
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

template <typename T>
static inline void dumpSet(const llvm::SmallPtrSetImpl<T *> &o) {
  llvm::errs() << "<begin dump>\n";
  for (auto a : o)
    llvm::errs() << *a << "\n";
  llvm::errs() << "</end dump>\n";
}

static inline llvm::Instruction *
getNextNonDebugInstructionOrNull(llvm::Instruction *Z) {
  for (llvm::Instruction *I = Z->getNextNode(); I; I = I->getNextNode())
    if (!llvm::isa<llvm::DbgInfoIntrinsic>(I))
      return I;
  return nullptr;
}

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

static inline bool hasMetadata(const llvm::GlobalObject *O,
                               llvm::StringRef kind) {
  return O->getMetadata(kind) != nullptr;
}

static inline bool hasMetadata(const llvm::Instruction *O,
                               llvm::StringRef kind) {
  return O->getMetadata(kind) != nullptr;
}

enum class ReturnType {
  ArgsWithReturn,
  ArgsWithTwoReturns,
  Args,
  TapeAndReturn,
  TapeAndTwoReturns,
  Tape,
};

enum class DIFFE_TYPE {
  OUT_DIFF = 0,  // add differential to output struct
  DUP_ARG = 1,   // duplicate the argument and store differential inside
  CONSTANT = 2,  // no differential
  DUP_NONEED = 3 // duplicate this argument and store differential inside, but
                 // don't need the forward
};

static inline std::string tostring(DIFFE_TYPE t) {
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

#include <set>

// note this doesn't handle recursive types!
static inline DIFFE_TYPE whatType(llvm::Type *arg,
                                  std::set<llvm::Type *> seen = {}) {
  assert(arg);
  if (seen.find(arg) != seen.end())
    return DIFFE_TYPE::CONSTANT;
  seen.insert(arg);

  if (arg->isVoidTy() || arg->isEmptyTy()) {
    return DIFFE_TYPE::CONSTANT;
  }

  if (arg->isPointerTy()) {
    switch (
        whatType(llvm::cast<llvm::PointerType>(arg)->getElementType(), seen)) {
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
    return whatType(llvm::cast<llvm::ArrayType>(arg)->getElementType(), seen);
  } else if (arg->isStructTy()) {
    auto st = llvm::cast<llvm::StructType>(arg);
    if (st->getNumElements() == 0)
      return DIFFE_TYPE::CONSTANT;

    auto ty = DIFFE_TYPE::CONSTANT;
    for (unsigned i = 0; i < st->getNumElements(); i++) {
      switch (whatType(st->getElementType(i), seen)) {
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
    return DIFFE_TYPE::OUT_DIFF;
  } else {
    assert(arg);
    llvm::errs() << "arg: " << *arg << "\n";
    assert(0 && "Cannot handle type");
    return DIFFE_TYPE::CONSTANT;
  }
}

static inline bool isReturned(llvm::Instruction *inst) {
  for (const auto &a : inst->users()) {
    if (llvm::isa<llvm::ReturnInst>(a))
      return true;
  }
  return false;
}

static inline llvm::Type *FloatToIntTy(llvm::Type *T) {
  assert(T->isFPOrFPVectorTy());
  if (auto ty = llvm::dyn_cast<llvm::VectorType>(T)) {
    return llvm::VectorType::get(FloatToIntTy(ty->getElementType()),
                                 ty->getNumElements());
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

static inline llvm::Type *IntToFloatTy(llvm::Type *T) {
  assert(T->isIntOrIntVectorTy());
  if (auto ty = llvm::dyn_cast<llvm::VectorType>(T)) {
    return llvm::VectorType::get(IntToFloatTy(ty->getElementType()),
                                 ty->getNumElements());
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

static inline bool isCertainMallocOrFree(llvm::Function *called) {
  if (called == nullptr)
    return false;
  if (called->getName() == "printf" || called->getName() == "puts" ||
      called->getName() == "malloc" || called->getName() == "_Znwm" ||
      called->getName() == "_ZdlPv" || called->getName() == "_ZdlPvm" ||
      called->getName() == "free")
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

static inline bool isCertainPrintOrFree(llvm::Function *called) {
  if (called == nullptr)
    return false;

  if (called->getName() == "printf" || called->getName() == "puts" ||
      called->getName() == "_ZdlPv" || called->getName() == "_ZdlPvm" ||
      called->getName() == "free")
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

static inline bool isCertainPrintMallocOrFree(llvm::Function *called) {
  if (called == nullptr)
    return false;

  if (called->getName() == "printf" || called->getName() == "puts" ||
      called->getName() == "malloc" || called->getName() == "_Znwm" ||
      called->getName() == "_ZdlPv" || called->getName() == "_ZdlPvm" ||
      called->getName() == "free")
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

//! Create function for type that performs the derivative memcpy on floating
//! point memory
llvm::Function *getOrInsertDifferentialFloatMemcpy(llvm::Module &M,
                                                   llvm::PointerType *T,
                                                   unsigned dstalign,
                                                   unsigned srcalign);

//! Create function for type that performs the derivative memmove on floating
//! point memory
llvm::Function *getOrInsertDifferentialFloatMemmove(llvm::Module &M,
                                                    llvm::PointerType *T,
                                                    unsigned dstalign,
                                                    unsigned srcalign);

template <typename K, typename V, typename K2>
static inline typename std::map<K, V>::iterator
insert_or_assign(std::map<K, V> &map, K2 key, V &&val) {
  // map.insert_or_assign(key, val);
  auto found = map.find(key);
  if (found == map.end()) {
  }
  return map.emplace(key, val).first;

  map.at(key) = val;
  // map->second = val;
  return map.find(key);
}

template <typename K, typename V, typename K2>
static inline typename std::map<K, V>::iterator
insert_or_assign(std::map<K, V> &map, K2 key, const V &val) {
  // map.insert_or_assign(key, val);
  auto found = map.find(key);
  if (found == map.end()) {
  }
  return map.emplace(key, val).first;

  map.at(key) = val;
  // map->second = val;
  return map.find(key);
}

#include "llvm/IR/CFG.h"
#include <deque>
#include <functional>
// Return true if should break early
static inline void allFollowersOf(llvm::Instruction *inst,
                                  std::function<bool(llvm::Instruction *)> f) {

  // llvm::errs() << "all followers of: " << *inst << "\n";
  for (auto uinst = inst->getNextNode(); uinst != nullptr;
       uinst = uinst->getNextNode()) {
    // llvm::errs() << " + bb1: " << *uinst << "\n";
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

static inline void
allPredecessorsOf(llvm::Instruction *inst,
                  std::function<bool(llvm::Instruction *)> f) {

  // llvm::errs() << "all followers of: " << *inst << "\n";
  for (auto uinst = inst->getPrevNode(); uinst != nullptr;
       uinst = uinst->getPrevNode()) {
    // llvm::errs() << " + bb1: " << *uinst << "\n";
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
    for (; I != E; I++) {
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

#include "llvm/Analysis/LoopInfo.h"
static inline void
allInstructionsBetween(llvm::LoopInfo &LI, llvm::Instruction *inst1,
                       llvm::Instruction *inst2,
                       std::function<bool(llvm::Instruction *)> f) {
  for (auto uinst = inst1->getNextNode(); uinst != nullptr;
       uinst = uinst->getNextNode()) {
    // llvm::errs() << " + bb1: " << *uinst << "\n";
    if (f(uinst))
      return;
    if (uinst == inst2)
      return;
  }

  std::set<llvm::Instruction *> instructions;

  llvm::Loop *l1 = LI.getLoopFor(inst1->getParent());
  while (l1 && !l1->contains(inst2->getParent()))
    l1 = l1->getParentLoop();
  /*
  llvm::errs() << " l1: " << l1;
  if (l1) llvm::errs() << " " << *l1;
  llvm::errs() << "\n";
  */

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

      // llvm::errs() << " block: " << BB->getName() << "\n";
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

#endif
