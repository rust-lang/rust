
//===- MustExitScalarEvolution.h - ScalarEvolution assuming loops terminate-=//
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
// This file declares MustExitScalarEvolution, a subclass of ScalarEvolution
// that assumes that all loops terminate (and don't loop forever).
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MUST_EXIT_SCALAR_EVOLUTION_H_
#define ENZYME_MUST_EXIT_SCALAR_EVOLUTION_H_

#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"

#include "llvm/IR/Dominators.h"

class MustExitScalarEvolution final : public llvm::ScalarEvolution {
public:
  llvm::SmallPtrSet<llvm::BasicBlock *, 4> GuaranteedUnreachable;
  using ScalarEvolution::ScalarEvolution;

  MustExitScalarEvolution(llvm::Function &F, llvm::TargetLibraryInfo &TLI,
                          llvm::AssumptionCache &AC, llvm::DominatorTree &DT,
                          llvm::LoopInfo &LI);
  ScalarEvolution::ExitLimit computeExitLimit(const llvm::Loop *L,
                                              llvm::BasicBlock *ExitingBlock,
                                              bool AllowPredicates);

#if LLVM_VERSION_MAJOR >= 7
  ScalarEvolution::ExitLimit computeExitLimitFromCond(const llvm::Loop *L,
                                                      llvm::Value *ExitCond,
                                                      bool ExitIfTrue,
                                                      bool ControlsExit,
                                                      bool AllowPredicates);

  ScalarEvolution::ExitLimit
  computeExitLimitFromCondCached(ExitLimitCacheTy &Cache, const llvm::Loop *L,
                                 llvm::Value *ExitCond, bool ExitIfTrue,
                                 bool ControlsExit, bool AllowPredicates);
  ScalarEvolution::ExitLimit
  computeExitLimitFromCondImpl(ExitLimitCacheTy &Cache, const llvm::Loop *L,
                               llvm::Value *ExitCond, bool ExitIfTrue,
                               bool ControlsExit, bool AllowPredicates);

  ScalarEvolution::ExitLimit
  computeExitLimitFromICmp(const llvm::Loop *L, llvm::ICmpInst *ExitCond,
                           bool ExitIfTrue, bool ControlsExit,
                           bool AllowPredicates = false);
#else
  ScalarEvolution::ExitLimit
  computeExitLimitFromCond(const llvm::Loop *L, llvm::Value *ExitCond,
                           llvm::BasicBlock *TBB, llvm::BasicBlock *FBB,
                           bool ControlsExit, bool AllowPredicates);

  ScalarEvolution::ExitLimit
  computeExitLimitFromCondCached(ExitLimitCacheTy &Cache, const llvm::Loop *L,
                                 llvm::Value *ExitCond, llvm::BasicBlock *TBB,
                                 llvm::BasicBlock *FBB, bool ControlsExit,
                                 bool AllowPredicates);

  ScalarEvolution::ExitLimit
  computeExitLimitFromCondImpl(ExitLimitCacheTy &Cache, const llvm::Loop *L,
                               llvm::Value *ExitCond, llvm::BasicBlock *TBB,
                               llvm::BasicBlock *FBB, bool ControlsExit,
                               bool AllowPredicates);

  ScalarEvolution::ExitLimit
  computeExitLimitFromICmp(const llvm::Loop *L, llvm::ICmpInst *ExitCond,
                           llvm::BasicBlock *TBB, llvm::BasicBlock *FBB,
                           bool ControlsExit, bool AllowPredicates = false);
#endif

  bool loopIsFiniteByAssumption(const llvm::Loop *L);

  ScalarEvolution::ExitLimit howManyLessThans(const llvm::SCEV *LHS,
                                              const llvm::SCEV *RHS,
                                              const llvm::Loop *L,
                                              bool IsSigned, bool ControlsExit,
                                              bool AllowPredicates);
};

#endif
