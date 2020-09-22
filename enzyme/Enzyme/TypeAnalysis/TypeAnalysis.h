/*
 * TypeAnalysis.h - Type Analysis Detection Utilities
 *
 * Copyright (C) 2020 William S. Moses (enzyme@wsmoses.com) - All Rights
 * Reserved
 *
 * For commercial use of this code please contact the author(s) above.
 */

#ifndef ENZYME_TYPE_ANALYSIS_H
#define ENZYME_TYPE_ANALYSIS_H 1

#include <cstdint>
#include <deque>

#include <llvm/Config/llvm-config.h>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/IR/InstVisitor.h"

#include "llvm/IR/Dominators.h"

#include "TypeTree.h"

class FnTypeInfo {
public:
  llvm::Function *function;
  FnTypeInfo(llvm::Function *fn) : function(fn) {}
  FnTypeInfo(const FnTypeInfo &) = default;
  FnTypeInfo &operator=(FnTypeInfo &) = default;
  FnTypeInfo &operator=(FnTypeInfo &&) = default;

  // arguments:type
  std::map<llvm::Argument *, TypeTree> first;
  // return type
  TypeTree second;
  // the specific constant of an argument, if it is constant
  std::map<llvm::Argument *, std::set<int64_t>> knownValues;

  std::set<int64_t>
  knownIntegralValues(llvm::Value *val, const llvm::DominatorTree &DT,
                std::map<llvm::Value *, std::set<int64_t>> &intseen) const;
};

static inline bool operator<(const FnTypeInfo &lhs,
                             const FnTypeInfo &rhs) {

  if (lhs.function < rhs.function)
    return true;
  if (rhs.function < lhs.function)
    return false;

  if (lhs.first < rhs.first)
    return true;
  if (rhs.first < lhs.first)
    return false;
  if (lhs.second < rhs.second)
    return true;
  if (rhs.second < lhs.second)
    return false;
  return lhs.knownValues < rhs.knownValues;
}

class TypeAnalyzer;
class TypeAnalysis;

class TypeResults {
public:
  TypeAnalysis &analysis;
  const FnTypeInfo info;

public:
  TypeResults(TypeAnalysis &analysis, const FnTypeInfo &fn);
  ConcreteType intType(llvm::Value *val, bool errIfNotFound = true);

  //! Returns whether in the first num bytes there is pointer, int, float, or
  //! none If pointerIntSame is set to true, then consider either as the same
  //! (and thus mergable)
  ConcreteType firstPointer(size_t num, llvm::Value *val, bool errIfNotFound = true,
                        bool pointerIntSame = false);

  TypeTree query(llvm::Value *val);
  FnTypeInfo getAnalyzedTypeInfo();
  TypeTree getReturnAnalysis();
  void dump();
  std::set<int64_t> knownIntegralValues(llvm::Value *val) const;
};

class TypeAnalyzer : public llvm::InstVisitor<TypeAnalyzer> {
public:
  // List of value's which should be re-analyzed now with new information
  std::deque<llvm::Value *> workList;

private:
  void addToWorkList(llvm::Value *val);
  std::map<llvm::Value *, std::set<int64_t>> intseen;

public:
  // Calling context
  const FnTypeInfo fntypeinfo;

  TypeAnalysis &interprocedural;

  std::map<llvm::Value *, TypeTree> analysis;

  llvm::DominatorTree DT;

  TypeAnalyzer(const FnTypeInfo &fn, TypeAnalysis &TA);

  TypeTree getAnalysis(llvm::Value *val);

  void updateAnalysis(llvm::Value *val, BaseType data, llvm::Value *origin);
  void updateAnalysis(llvm::Value *val, ConcreteType data, llvm::Value *origin);
  void updateAnalysis(llvm::Value *val, TypeTree data, llvm::Value *origin);

  void prepareArgs();

  void considerTBAA();

  void run();

  bool runUnusedChecks();

  void visitValue(llvm::Value &val);

  void visitCmpInst(llvm::CmpInst &I);

  void visitAllocaInst(llvm::AllocaInst &I);

  void visitLoadInst(llvm::LoadInst &I);

  void visitStoreInst(llvm::StoreInst &I);

  void visitGetElementPtrInst(llvm::GetElementPtrInst &gep);

  void visitPHINode(llvm::PHINode &phi);

  void visitTruncInst(llvm::TruncInst &I);

  void visitZExtInst(llvm::ZExtInst &I);

  void visitSExtInst(llvm::SExtInst &I);

  void visitAddrSpaceCastInst(llvm::AddrSpaceCastInst &I);

  void visitFPTruncInst(llvm::FPTruncInst &I);

  void visitFPToUIInst(llvm::FPToUIInst &I);

  void visitFPToSIInst(llvm::FPToSIInst &I);

  void visitUIToFPInst(llvm::UIToFPInst &I);

  void visitSIToFPInst(llvm::SIToFPInst &I);

  void visitPtrToIntInst(llvm::PtrToIntInst &I);

  void visitIntToPtrInst(llvm::IntToPtrInst &I);

  void visitBitCastInst(llvm::BitCastInst &I);

  void visitSelectInst(llvm::SelectInst &I);

  void visitExtractElementInst(llvm::ExtractElementInst &I);

  void visitInsertElementInst(llvm::InsertElementInst &I);

  void visitShuffleVectorInst(llvm::ShuffleVectorInst &I);

  void visitExtractValueInst(llvm::ExtractValueInst &I);

  void visitInsertValueInst(llvm::InsertValueInst &I);

  void visitBinaryOperator(llvm::BinaryOperator &I);

  void visitIPOCall(llvm::CallInst &call, llvm::Function &fn);

  void visitCallInst(llvm::CallInst &call);

  void visitMemTransferInst(llvm::MemTransferInst &MTI);

  void visitIntrinsicInst(llvm::IntrinsicInst &II);

  TypeTree getReturnAnalysis();

  void dump();

  std::set<int64_t> knownIntegralValues(llvm::Value *val);

  //TODO handle fneg on LLVM 10+
};

class TypeAnalysis {
public:
  std::map<FnTypeInfo, TypeAnalyzer> analyzedFunctions;

  TypeResults analyzeFunction(const FnTypeInfo &fn);

  TypeTree query(llvm::Value *val, const FnTypeInfo &fn);

  ConcreteType intType(llvm::Value *val, const FnTypeInfo &fn,
                   bool errIfNotFound = true);
  ConcreteType firstPointer(size_t num, llvm::Value *val, const FnTypeInfo &fn,
                        bool errIfNotFound = true, bool pointerIntSame = false);

  inline TypeTree getReturnAnalysis(const FnTypeInfo &fn) {
    analyzeFunction(fn);
    return analyzedFunctions.find(fn)->second.getReturnAnalysis();
  }
};

#endif
