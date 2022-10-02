//===- TypeAnalysis.h - Declaration of Type Analysis   ------------===//
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
// This file contains the declaration of Type Analysis, a utility for
// computing the underlying data type of LLVM values.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_TYPE_ANALYSIS_H
#define ENZYME_TYPE_ANALYSIS_H 1

#include <cstdint>
#include <deque>

#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/SetVector.h"

#include "llvm/Analysis/TargetLibraryInfo.h"

#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Dominators.h"

#include "TypeTree.h"

extern const std::map<std::string, llvm::Intrinsic::ID> LIBM_FUNCTIONS;

static inline bool isMemFreeLibMFunction(llvm::StringRef str,
                                         llvm::Intrinsic::ID *ID = nullptr) {
  if (str.startswith("__") && str.endswith("_finite")) {
    str = str.substr(2, str.size() - 2 - 7);
  } else if (str.startswith("__fd_") && str.endswith("_1")) {
    str = str.substr(5, str.size() - 5 - 2);
  } else if (str.startswith("__nv_")) {
    str = str.substr(5, str.size() - 5);
  }
  if (LIBM_FUNCTIONS.find(str.str()) != LIBM_FUNCTIONS.end()) {
    if (ID)
      *ID = LIBM_FUNCTIONS.find(str.str())->second;
    return true;
  }
  if (str.endswith("f") || str.endswith("l")) {
    if (LIBM_FUNCTIONS.find(str.substr(0, str.size() - 1).str()) !=
        LIBM_FUNCTIONS.end()) {
      if (ID)
        *ID = LIBM_FUNCTIONS.find(str.substr(0, str.size() - 1).str())->second;
      return true;
    }
  }
  return false;
}

/// Struct containing all contextual type information for a
/// particular function call
struct FnTypeInfo {
  /// Function being analyzed
  llvm::Function *Function;

  FnTypeInfo(llvm::Function *fn) : Function(fn) {}
  FnTypeInfo(const FnTypeInfo &) = default;
  FnTypeInfo &operator=(FnTypeInfo &) = default;
  FnTypeInfo &operator=(FnTypeInfo &&) = default;

  /// Types of arguments
  std::map<llvm::Argument *, TypeTree> Arguments;

  /// Type of return
  TypeTree Return;

  /// The specific constant(s) known to represented by an argument, if constant
  std::map<llvm::Argument *, std::set<int64_t>> KnownValues;

  /// The set of known values val will take
  std::set<int64_t>
  knownIntegralValues(llvm::Value *val, const llvm::DominatorTree &DT,
                      std::map<llvm::Value *, std::set<int64_t>> &intseen,
                      llvm::ScalarEvolution &SE) const;
};

static inline bool operator<(const FnTypeInfo &lhs, const FnTypeInfo &rhs) {

  if (lhs.Function < rhs.Function)
    return true;
  if (rhs.Function < lhs.Function)
    return false;

  if (lhs.Return < rhs.Return)
    return true;
  if (rhs.Return < lhs.Return)
    return false;

  for (auto &arg : lhs.Function->args()) {
    {
      auto foundLHS = lhs.Arguments.find(&arg);
      assert(foundLHS != lhs.Arguments.end());
      auto foundRHS = rhs.Arguments.find(&arg);
      assert(foundRHS != rhs.Arguments.end());
      if (foundLHS->second < foundRHS->second)
        return true;
      if (foundRHS->second < foundLHS->second)
        return false;
    }

    {
      auto foundLHS = lhs.KnownValues.find(&arg);
      assert(foundLHS != lhs.KnownValues.end());
      auto foundRHS = rhs.KnownValues.find(&arg);
      assert(foundRHS != rhs.KnownValues.end());
      if (foundLHS->second < foundRHS->second)
        return true;
      if (foundRHS->second < foundLHS->second)
        return false;
    }
  }
  // equal;
  return false;
}

class TypeAnalyzer;
class TypeAnalysis;

/// A holder class representing the results of running TypeAnalysis
/// on a given function
class TypeResults {
public:
  TypeAnalyzer &analyzer;

public:
  TypeResults(TypeAnalyzer &analyzer);
  ConcreteType intType(size_t num, llvm::Value *val, bool errIfNotFound = true,
                       bool pointerIntSame = false) const;
  llvm::Type *addingType(size_t num, llvm::Value *val) const;

  /// Returns whether in the first num bytes there is pointer, int, float, or
  /// none If pointerIntSame is set to true, then consider either as the same
  /// (and thus mergable)
  ConcreteType firstPointer(size_t num, llvm::Value *val, llvm::Instruction *I,
                            bool errIfNotFound = true,
                            bool pointerIntSame = false) const;

  /// The TypeTree of a particular Value
  TypeTree query(llvm::Value *val) const;

  /// The TypeInfo calling convention
  FnTypeInfo getAnalyzedTypeInfo() const;

  /// The Type of the return
  TypeTree getReturnAnalysis() const;

  /// Prints all known information
  void dump() const;

  /// The set of values val will take on during this program
  std::set<int64_t> knownIntegralValues(llvm::Value *val) const;

  FnTypeInfo getCallInfo(llvm::CallInst &CI, llvm::Function &fn) const;

  llvm::Function *getFunction() const;
};

/// Helper class that computes the fixed-point type results of a given function
class TypeAnalyzer : public llvm::InstVisitor<TypeAnalyzer> {
public:
  /// List of value's which should be re-analyzed now with new information
  llvm::SetVector<llvm::Value *, std::deque<llvm::Value *>> workList;

  const llvm::SmallPtrSet<llvm::BasicBlock *, 4> notForAnalysis;

private:
  /// Tell TypeAnalyzer to reanalyze this value
  void addToWorkList(llvm::Value *val);

  /// Map of Value to known integer constants that it will take on
  std::map<llvm::Value *, std::set<int64_t>> intseen;

  std::map<llvm::Value *, std::pair<bool, bool>> mriseen;
  bool mustRemainInteger(llvm::Value *val, bool *returned = nullptr);

public:
  /// Calling context
  const FnTypeInfo fntypeinfo;

  /// Calling TypeAnalysis to be used in the case of calls to other
  /// functions
  TypeAnalysis &interprocedural;

  /// Directionality of checks
  uint8_t direction;

  /// Whether an inconsistent update has been found
  /// This will only be set when direction != Both, erring otherwise
  bool Invalid;

  bool PHIRecur;

  // propagate from instruction to operand
  static constexpr uint8_t UP = 1;
  // propagate from operand to instruction
  static constexpr uint8_t DOWN = 2;
  static constexpr uint8_t BOTH = UP | DOWN;

  /// Intermediate conservative, but correct Type analysis results
  std::map<llvm::Value *, TypeTree> analysis;

  llvm::TargetLibraryInfo &TLI;
  llvm::DominatorTree &DT;
  llvm::PostDominatorTree &PDT;

  llvm::LoopInfo &LI;
  llvm::ScalarEvolution &SE;

  FnTypeInfo getCallInfo(llvm::CallInst &CI, llvm::Function &fn);

  TypeAnalyzer(const FnTypeInfo &fn, TypeAnalysis &TA,
               uint8_t direction = BOTH);

  TypeAnalyzer(const FnTypeInfo &fn, TypeAnalysis &TA,
               const llvm::SmallPtrSetImpl<llvm::BasicBlock *> &notForAnalysis,
               const TypeAnalyzer &Prev, uint8_t direction = BOTH,
               bool PHIRecur = false);

  /// Get the current results for a given value
  TypeTree getAnalysis(llvm::Value *Val);

  /// Add additional information to the Type info of val, readding it to the
  /// work queue as necessary
  void updateAnalysis(llvm::Value *val, BaseType data, llvm::Value *origin);
  void updateAnalysis(llvm::Value *val, ConcreteType data, llvm::Value *origin);
  void updateAnalysis(llvm::Value *val, TypeTree data, llvm::Value *origin);

  /// Analyze type info given by the arguments, possibly adding to work queue
  void prepareArgs();

  /// Analyze type info given by the TBAA, possibly adding to work queue
  void considerTBAA();

  /// Parse the debug info generated by rustc and retrieve useful type info if
  /// possible
  void considerRustDebugInfo();

  /// Run the interprocedural type analysis starting from this function
  void run();

  /// Hypothesize that undefined phi's are integers and try to prove
  /// that they are really integral
  void runPHIHypotheses();

  void visitValue(llvm::Value &val);

  void visitConstantExpr(llvm::ConstantExpr &CE);

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

  void visitFPExtInst(llvm::FPExtInst &I);

  void visitFPTruncInst(llvm::FPTruncInst &I);

  void visitFPToUIInst(llvm::FPToUIInst &I);

  void visitFPToSIInst(llvm::FPToSIInst &I);

  void visitUIToFPInst(llvm::UIToFPInst &I);

  void visitSIToFPInst(llvm::SIToFPInst &I);

  void visitPtrToIntInst(llvm::PtrToIntInst &I);

  void visitIntToPtrInst(llvm::IntToPtrInst &I);

  void visitBitCastInst(llvm::BitCastInst &I);

#if LLVM_VERSION_MAJOR >= 10
  void visitFreezeInst(llvm::FreezeInst &I);
#endif

  void visitSelectInst(llvm::SelectInst &I);

  void visitExtractElementInst(llvm::ExtractElementInst &I);

  void visitInsertElementInst(llvm::InsertElementInst &I);

  void visitShuffleVectorInst(llvm::ShuffleVectorInst &I);

  void visitExtractValueInst(llvm::ExtractValueInst &I);

  void visitInsertValueInst(llvm::InsertValueInst &I);

  void visitAtomicRMWInst(llvm::AtomicRMWInst &I);

  void visitBinaryOperator(llvm::BinaryOperator &I);
  void visitBinaryOperation(const llvm::DataLayout &DL, llvm::Type *T,
                            llvm::Instruction::BinaryOps, llvm::Value *Args[2],
                            TypeTree &Ret, TypeTree &LHS, TypeTree &RHS);

  void visitIPOCall(llvm::CallInst &call, llvm::Function &fn);

  void visitInvokeInst(llvm::InvokeInst &call);
  void visitCallInst(llvm::CallInst &call);

  void visitMemTransferInst(llvm::MemTransferInst &MTI);
  void visitMemTransferCommon(llvm::CallInst &MTI);

  void visitIntrinsicInst(llvm::IntrinsicInst &II);

  TypeTree getReturnAnalysis();

  void dump(llvm::raw_ostream &ss = llvm::errs());

  std::set<int64_t> knownIntegralValues(llvm::Value *val);

  // TODO handle fneg on LLVM 10+
};

/// Full interprocedural TypeAnalysis
class TypeAnalysis {
public:
  llvm::FunctionAnalysisManager &FAM;
  TypeAnalysis(llvm::FunctionAnalysisManager &FAM) : FAM(FAM) {}
  /// Map of custom function call handlers
  std::map<std::string,
           std::function<bool(int /*direction*/, TypeTree & /*returnTree*/,
                              llvm::ArrayRef<TypeTree> /*argTrees*/,
                              llvm::ArrayRef<std::set<int64_t>> /*knownValues*/,
                              llvm::CallInst * /*call*/)>>
      CustomRules;

  /// Map of possible query states to TypeAnalyzer intermediate results
  std::map<FnTypeInfo, std::shared_ptr<TypeAnalyzer>> analyzedFunctions;

  /// Analyze a particular function, returning the results
  TypeResults analyzeFunction(const FnTypeInfo &fn);

  /// Clear existing analyses
  void clear();
};

#endif
