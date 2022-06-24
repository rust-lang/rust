//===- DifferentialUseAnalysis.h - Determine values needed in reverse pass-===//
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
// This file contains the declaration of Differential USe Analysis -- an
// AD-specific analysis that deduces if a given value is needed in the reverse
// pass.
//
//===----------------------------------------------------------------------===//
#include <deque>
#include <map>
#include <set>

#include "GradientUtils.h"

typedef std::pair<const Value *, ValueType> UsageKey;

// Determine if a value is needed directly to compute the adjoint
// of the given instruction user
static inline bool is_use_directly_needed_in_reverse(
    const GradientUtils *gutils, const Value *val, const Instruction *user,
    const SmallPtrSetImpl<BasicBlock *> &oldUnreachable) {
  TypeResults const &TR = gutils->TR;
  if (auto ainst = dyn_cast<Instruction>(val)) {
    assert(ainst->getParent()->getParent() == gutils->oldFunc);
  }
  assert(user->getParent()->getParent() == gutils->oldFunc);

  if (oldUnreachable.count(user->getParent()))
    return false;

  if (isa<LoadInst>(user) || isa<CastInst>(user) || isa<PHINode>(user) ||
      isa<GetElementPtrInst>(user)) {
    return false;
  }

  // We don't need any of the input operands to compute the adjoint of a store
  // instance
  if (auto SI = dyn_cast<StoreInst>(user)) {
    // The one exception to this is stores to the loop bounds.
    if (SI->getValueOperand() == val) {
      for (auto U : SI->getPointerOperand()->users()) {
        if (auto CI = dyn_cast<CallInst>(U)) {
          if (auto F = CI->getCalledFunction()) {
            if (F->getName() == "__kmpc_for_static_init_4" ||
                F->getName() == "__kmpc_for_static_init_4u" ||
                F->getName() == "__kmpc_for_static_init_8" ||
                F->getName() == "__kmpc_for_static_init_8u") {
              if (CI->getArgOperand(4) == val || CI->getArgOperand(5) == val ||
                  CI->getArgOperand(6))
                return true;
            }
          }
        }
      }

      // Preserve any non-floating point values that are stored in an active
      // backwards creation shadow.
      if (!TR.query(const_cast<Value *>(SI->getValueOperand()))[{-1}].isFloat())
        for (auto pair : gutils->backwardsOnlyShadows)
          if (pair.second.stores.count(SI)) {
            return true;
          }
    }
    return false;
  }

  // If memtransfer, only the size may need preservation for the reverse pass
  if (auto MTI = dyn_cast<MemTransferInst>(user)) {
    // Unless we're storing into a backwards only shadow store
    if (MTI->getArgOperand(1) == val || MTI->getArgOperand(2) == val) {
      for (auto pair : gutils->backwardsOnlyShadows)
        if (pair.second.stores.count(MTI)) {
          return true;
        }
    }
    if (MTI->getArgOperand(2) != val)
      return false;
    return !gutils->isConstantInstruction(MTI);
  }

  // Preserve the length of memsets of backward creation shadows
  if (auto MS = dyn_cast<MemSetInst>(user)) {
    if (MS->getArgOperand(1) == val || MS->getArgOperand(2) == val) {
      for (auto pair : gutils->backwardsOnlyShadows)
        if (pair.second.stores.count(MS)) {
          return true;
        }
    }
  }

  if (isa<CmpInst>(user) || isa<BranchInst>(user) || isa<ReturnInst>(user) ||
      isa<FPExtInst>(user) || isa<FPTruncInst>(user) ||
      (isa<InsertElementInst>(user) &&
       cast<InsertElementInst>(user)->getOperand(2) != val) ||
      (isa<ExtractElementInst>(user) &&
       cast<ExtractElementInst>(user)->getIndexOperand() != val)
#if LLVM_VERSION_MAJOR >= 10
      || isa<FreezeInst>(user)
#endif
      // isa<ExtractElement>(use) ||
      // isa<InsertElementInst>(use) || isa<ShuffleVectorInst>(use) ||
      // isa<ExtractValueInst>(use) || isa<AllocaInst>(use)
      /*|| isa<StoreInst>(use)*/) {
    return false;
  }

  Intrinsic::ID ID = Intrinsic::not_intrinsic;
  if (auto II = dyn_cast<IntrinsicInst>(user)) {
    ID = II->getIntrinsicID();
  } else if (auto CI = dyn_cast<CallInst>(user)) {
    if (auto called = getFunctionFromCall(const_cast<CallInst *>(CI))) {
      StringRef funcName;
      if (called->hasFnAttribute("enzyme_math"))
        funcName = called->getFnAttribute("enzyme_math").getValueAsString();
      else
        funcName = called->getName();
      isMemFreeLibMFunction(funcName, &ID);
    }
  }

  if (ID != Intrinsic::not_intrinsic) {
    if (ID == Intrinsic::lifetime_start || ID == Intrinsic::lifetime_end ||
        ID == Intrinsic::stacksave || ID == Intrinsic::stackrestore) {
      return false;
    }
    if (ID == Intrinsic::fma || ID == Intrinsic::fmuladd) {
      bool needed = false;
      if (user->getOperand(0) == val &&
          !gutils->isConstantValue(user->getOperand(1)))
        needed = true;
      if (user->getOperand(1) == val &&
          !gutils->isConstantValue(user->getOperand(0)))
        needed = true;
      return needed;
    }
  }

  if (auto op = dyn_cast<BinaryOperator>(user)) {
    if (op->getOpcode() == Instruction::FAdd ||
        op->getOpcode() == Instruction::FSub) {
      return false;
    } else if (op->getOpcode() == Instruction::FMul) {
      bool needed = false;
      if (op->getOperand(0) == val &&
          !gutils->isConstantValue(op->getOperand(1)))
        needed = true;
      if (op->getOperand(1) == val &&
          !gutils->isConstantValue(op->getOperand(0)))
        needed = true;
      return needed;
    } else if (op->getOpcode() == Instruction::FDiv) {
      bool needed = false;
      if (op->getOperand(1) == val &&
          !gutils->isConstantValue(op->getOperand(1)))
        needed = true;
      if (op->getOperand(1) == val &&
          !gutils->isConstantValue(op->getOperand(0)))
        needed = true;
      if (op->getOperand(0) == val &&
          !gutils->isConstantValue(op->getOperand(1)))
        needed = true;
      return needed;
    }
  }

  if (auto si = dyn_cast<SelectInst>(user)) {
    // Only would potentially need the condition
    if (si->getCondition() != val) {
      return false;
    }

    // only need the condition if select is active
    return !gutils->isConstantValue(const_cast<SelectInst *>(si));
  }

  if (auto CI = dyn_cast<CallInst>(user)) {
    if (auto F = getFunctionFromCall(const_cast<CallInst *>(CI))) {
      auto funcName = F->getName();
      if (F->hasFnAttribute("enzyme_math"))
        funcName = F->getFnAttribute("enzyme_math").getValueAsString();

      // Only need primal (and shadow) request for reverse
      if (funcName == "MPI_Isend" || funcName == "MPI_Irecv" ||
          funcName == "PMPI_Isend" || funcName == "PMPI_Irecv") {
        if (val != CI->getArgOperand(6)) {
          return false;
        }
      }

      // Only need the primal request.
      if (funcName == "MPI_Wait" || funcName == "PMPI_Wait")
        if (val != CI->getArgOperand(0))
          return false;

      // Only need element count for reverse of waitall
      if (funcName == "MPI_Waitall" || funcName == "PMPI_Waitall")
        if (val != CI->getArgOperand(0) || val != CI->getOperand(1))
          return false;
      // Since adjoint of barrier is another barrier in reverse
      // we still need even if instruction is inactive
      if (funcName == "__kmpc_barrier" || funcName == "MPI_Barrier")
        return true;

      // Since adjoint of GC preserve is another preserve in reverse
      // we still need even if instruction is inactive
      if (funcName == "llvm.julia.gc_preserve_begin")
        return true;
    }
  }

  return !gutils->isConstantInstruction(user) ||
         !gutils->isConstantValue(const_cast<Instruction *>(user));
}

template <ValueType VT, bool OneLevel = false>
static inline bool is_value_needed_in_reverse(
    const GradientUtils *gutils, const Value *inst, DerivativeMode mode,
    std::map<UsageKey, bool> &seen,
    const SmallPtrSetImpl<BasicBlock *> &oldUnreachable) {
  TypeResults const &TR = gutils->TR;
  static_assert(VT == ValueType::Primal || VT == ValueType::Shadow);
  auto idx = UsageKey(inst, VT);
  if (seen.find(idx) != seen.end())
    return seen[idx];
  if (auto ainst = dyn_cast<Instruction>(inst)) {
    assert(ainst->getParent()->getParent() == gutils->oldFunc);
  }

  // Inductively claim we aren't needed (and try to find contradiction)
  seen[idx] = false;

  if (VT != ValueType::Shadow) {
    if (auto op = dyn_cast<BinaryOperator>(inst)) {
      if (op->getOpcode() == Instruction::FDiv) {
        if (!gutils->isConstantValue(const_cast<Value *>(inst)) &&
            !gutils->isConstantValue(op->getOperand(1))) {
          return seen[idx] = true;
        }
      }
    }
  }

  // Consider all users of this value, do any of them need this in the reverse?
  for (auto use : inst->users()) {
    if (use == inst)
      continue;

    const Instruction *user = dyn_cast<Instruction>(use);

    // A shadow value is only needed in reverse if it or one of its descendants
    // is used in an active instruction.
    // If inst is a constant value, the primal may be used in its place and
    // thus required.
    if (VT == ValueType::Shadow ||
        (gutils->isConstantValue(const_cast<Value *>(inst)) &&
         !TR.query(const_cast<Value *>(inst))[{-1}].isFloat())) {
      if (!user)
        return seen[idx] = true;

      if (auto SI = dyn_cast<StoreInst>(user)) {
        if (mode == DerivativeMode::ReverseModeGradient ||
            mode == DerivativeMode::ForwardModeSplit) {

          bool rematerialized = false;
          for (auto pair : gutils->backwardsOnlyShadows)
            if (pair.second.stores.count(SI)) {
              rematerialized = true;
              break;
            }

          if (SI->getValueOperand() == inst) {
            // storing an active pointer into a location
            // doesn't require the shadow pointer for the
            // reverse pass
            // Unless the store is into a backwards store, which would
            // would then be performed in the reverse if the stored value was
            // a possible pointer.
            if (!rematerialized)
              goto endShadow;
          } else {
            // Likewise, if not rematerializing in reverse pass, you
            // don't need to keep the pointer operand for known pointers
            if (!rematerialized &&
                TR.query(const_cast<Value *>(SI->getValueOperand()))[{-1}] ==
                    BaseType::Pointer)
              goto endShadow;
          }
        }

        if (!gutils->isConstantValue(
                const_cast<Value *>(SI->getPointerOperand())))
          return seen[idx] = true;
        else
          goto endShadow;
      }

      if (auto MTI = dyn_cast<MemTransferInst>(user)) {
        if (MTI->getArgOperand(0) != inst && MTI->getArgOperand(1) != inst)
          goto endShadow;

        if (!gutils->isConstantValue(
                const_cast<Value *>(MTI->getArgOperand(0))))
          return seen[idx] = true;
        else
          goto endShadow;
      }

      if (auto CI = dyn_cast<CallInst>(user)) {
        {
          SmallVector<OperandBundleDef, 2> OrigDefs;
          CI->getOperandBundlesAsDefs(OrigDefs);
          SmallVector<OperandBundleDef, 2> Defs;
          for (auto bund : OrigDefs) {
            for (auto inp : bund.inputs()) {
              if (inp == inst)
                return seen[idx] = true;
            }
          }
        }
        if (auto F = getFunctionFromCall(const_cast<CallInst *>(CI))) {
          StringRef funcName = F->getName();
          if (F->hasFnAttribute("enzyme_math"))
            funcName = F->getFnAttribute("enzyme_math").getValueAsString();

          // Only need shadow request for reverse
          if (funcName == "MPI_Irecv" || funcName == "PMPI_Irecv") {
            if (gutils->isConstantInstruction(const_cast<Instruction *>(user)))
              goto endShadow;
            // Need shadow request
            if (inst == CI->getArgOperand(6))
              return seen[idx] = true;
            // Need shadow buffer in forward pass
            if (mode != DerivativeMode::ReverseModeGradient)
              if (inst == CI->getArgOperand(0))
                return seen[idx] = true;
            goto endShadow;
          }
          if (funcName == "MPI_Isend" || funcName == "PMPI_Isend") {
            if (gutils->isConstantInstruction(const_cast<Instruction *>(user)))
              goto endShadow;
            // Need shadow request
            if (inst == CI->getArgOperand(6))
              return seen[idx] = true;
            // Need shadow buffer in reverse pass or forward mode
            if (inst == CI->getArgOperand(0))
              return seen[idx] = true;
            goto endShadow;
          }

          // Don't need shadow of anything (all via cache for reverse),
          // but need shadow of request for primal.
          if (funcName == "MPI_Wait" || funcName == "PMPI_Wait") {
            if (gutils->isConstantInstruction(const_cast<Instruction *>(user)))
              goto endShadow;
            // Need shadow request in forward pass only
            if (mode != DerivativeMode::ReverseModeGradient)
              if (inst == CI->getArgOperand(0))
                return seen[idx] = true;
            goto endShadow;
          }

          // Don't need shadow of anything (all via cache for reverse),
          // but need shadow of request for primal.
          if (funcName == "MPI_Waitall" || funcName == "PMPI_Waitall") {
            if (gutils->isConstantInstruction(const_cast<Instruction *>(user)))
              goto endShadow;
            // Need shadow request in forward pass
            if (mode != DerivativeMode::ReverseModeGradient)
              if (inst == CI->getArgOperand(1))
                return seen[idx] = true;
            goto endShadow;
          }

          // Use in a write barrier requires the shadow in the forward, even
          // though the instruction is active.
          if (mode != DerivativeMode::ReverseModeGradient &&
              funcName == "julia.write_barrier") {
            return seen[idx] = true;
          }
        }
#if LLVM_VERSION_MAJOR >= 11
        const Value *F = CI->getCalledOperand();
#else
        const Value *F = CI->getCalledValue();
#endif
        if (F == inst) {
          if (!gutils->isConstantInstruction(const_cast<Instruction *>(user)) ||
              !gutils->isConstantValue(const_cast<Value *>((Value *)user))) {
            return seen[idx] = true;
          }
        }
      }

      if (isa<ReturnInst>(user)) {
        if (gutils->ATA->ActiveReturns == DIFFE_TYPE::DUP_ARG ||
            gutils->ATA->ActiveReturns == DIFFE_TYPE::DUP_NONEED)
          return seen[idx] = true;
        else
          goto endShadow;
      }

      // Assume active instructions require the operand.
      if (!gutils->isConstantInstruction(const_cast<Instruction *>(user))) {
        return seen[idx] = true;
      }

      // Now the remaining instructions are inactive, however note that
      // a constant instruction may still require the use of the shadow
      // in the forward pass, for example double* x = load double** y
      // is a constant instruction, but needed in the forward
      if (user->getType()->isVoidTy())
        goto endShadow;

      if (!TR.query(const_cast<Instruction *>(user))
               .Inner0()
               .isPossiblePointer())
        goto endShadow;

      if (!OneLevel && is_value_needed_in_reverse<ValueType::Shadow>(
                           gutils, user, mode, seen, oldUnreachable)) {
        return seen[idx] = true;
      }
    endShadow:
      if (VT != ValueType::Primal)
        continue;
    }

    assert(VT == ValueType::Primal);

    // If a sub user needs, we need
    if (!OneLevel && is_value_needed_in_reverse<VT>(gutils, user, mode, seen,
                                                    oldUnreachable)) {
      return seen[idx] = true;
    }

    // Anything we may try to rematerialize requires its store operands for
    // the reverse pass.
    if (!OneLevel) {
      if (isa<StoreInst>(user) || isa<MemTransferInst>(user) ||
          isa<MemSetInst>(user)) {
        for (auto pair : gutils->rematerializableAllocations) {
          // Directly consider all the load uses to avoid an illegal inductive
          // recurrence. Specifically if we're asking if the alloca is used,
          // we'll set it to unused, then check the gep, then here we'll
          // directly say unused by induction instead of checking the final
          // loads.
          if (pair.second.stores.count(user)) {
            for (LoadInst *L : pair.second.loads)
              if (is_value_needed_in_reverse<VT>(gutils, L, mode, seen,
                                                 oldUnreachable)) {
                return seen[idx] = true;
              }
            for (auto &pair : pair.second.loadLikeCalls)
              if (is_use_directly_needed_in_reverse(
                      gutils, pair.operand, pair.loadCall, oldUnreachable) ||
                  is_value_needed_in_reverse<VT>(gutils, pair.loadCall, mode,
                                                 seen, oldUnreachable)) {
                return seen[idx] = true;
              }
          }
        }
      }
    }

    // One may need to this value in the computation of loop
    // bounds/comparisons/etc (which even though not active -- will be used for
    // the reverse pass)
    //   We could potentially optimize this to avoid caching if in combined mode
    //   and the instruction dominates all returns
    //   otherwise it will use the local cache (rather than save for a separate
    //   backwards cache)
    //   We also don't need this if looking at the shadow rather than primal
    {
      // Proving that none of the uses (or uses' uses) are used in control flow
      // allows us to safely not do this load

      // TODO save loop bounds for dynamic loop

      // TODO make this more aggressive and dont need to save loop latch
      if (isa<BranchInst>(use) || isa<SwitchInst>(use)) {
        size_t num = 0;
        for (auto suc : successors(cast<Instruction>(use)->getParent())) {
          if (!oldUnreachable.count(suc)) {
            num++;
          }
        }
        if (num <= 1)
          continue;
        return seen[idx] = true;
      }

      if (auto CI = dyn_cast<CallInst>(use)) {
        if (auto F = CI->getCalledFunction()) {
          if (F->getName() == "__kmpc_for_static_init_4" ||
              F->getName() == "__kmpc_for_static_init_4u" ||
              F->getName() == "__kmpc_for_static_init_8" ||
              F->getName() == "__kmpc_for_static_init_8u") {
            return seen[idx] = true;
          }
        }
      }
    }

    // The following are types we know we don't need to compute adjoints

    // If a primal value is needed to compute a shadow pointer (e.g. int offset
    // in gep), it needs preserving.
    bool primalUsedInShadowPointer = true;
    if (isa<CastInst>(user) || isa<LoadInst>(user))
      primalUsedInShadowPointer = false;
    if (auto GEP = dyn_cast<GetElementPtrInst>(user)) {
      bool idxUsed = false;
      for (auto &idx : GEP->indices()) {
        if (idx.get() == inst)
          idxUsed = true;
      }
      if (!idxUsed)
        primalUsedInShadowPointer = false;
    }

    if (primalUsedInShadowPointer)
      if (!user->getType()->isVoidTy() &&
          TR.query(const_cast<Instruction *>(user))
              .Inner0()
              .isPossiblePointer()) {
        if (is_value_needed_in_reverse<ValueType::Shadow>(
                gutils, user, mode, seen, oldUnreachable)) {
          return seen[idx] = true;
        }
      }

    bool direct =
        is_use_directly_needed_in_reverse(gutils, inst, user, oldUnreachable);
    if (!direct)
      continue;

    if (inst->getType()->isTokenTy()) {
      llvm::errs() << " need " << *inst << " via " << *user << "\n";
    }
    assert(!inst->getType()->isTokenTy());

    return seen[idx] = true;
  }
  return false;
}

template <ValueType VT>
static inline bool is_value_needed_in_reverse(
    const GradientUtils *gutils, const Value *inst, DerivativeMode mode,
    const SmallPtrSetImpl<BasicBlock *> &oldUnreachable) {
  static_assert(VT == ValueType::Primal || VT == ValueType::Shadow);
  std::map<UsageKey, bool> seen;
  return is_value_needed_in_reverse<VT>(gutils, inst, mode, seen,
                                        oldUnreachable);
}

struct Node {
  Value *V;
  bool outgoing;
  Node(Value *V, bool outgoing) : V(V), outgoing(outgoing){};
  bool operator<(const Node N) const {
    if (V < N.V)
      return true;
    return !(N.V < V) && outgoing < N.outgoing;
  }
  void dump() {
    if (V)
      llvm::errs() << "[" << *V << ", " << (int)outgoing << "]\n";
    else
      llvm::errs() << "[" << V << ", " << (int)outgoing << "]\n";
  }
};

typedef std::map<Node, std::set<Node>> Graph;

static inline void dump(Graph &G) {
  for (auto &pair : G) {
    llvm::errs() << "[" << *pair.first.V << ", " << (int)pair.first.outgoing
                 << "]\n";
    for (auto N : pair.second) {
      llvm::errs() << "\t[" << *N.V << ", " << (int)N.outgoing << "]\n";
    }
  }
}

/* Returns true if there is a path from source 's' to sink 't' in
  residual graph. Also fills parent[] to store the path */
static inline void bfs(const Graph &G,
                       const SmallPtrSetImpl<Value *> &Recompute,
                       std::map<Node, Node> &parent) {
  std::deque<Node> q;
  for (auto V : Recompute) {
    Node N(V, false);
    parent.emplace(N, Node(nullptr, true));
    q.push_back(N);
  }

  // Standard BFS Loop
  while (!q.empty()) {
    auto u = q.front();
    q.pop_front();
    auto found = G.find(u);
    if (found == G.end())
      continue;
    for (auto v : found->second) {
      if (parent.find(v) == parent.end()) {
        q.push_back(v);
        parent.emplace(v, u);
      }
    }
  }
}

// Return 1 if next is better
// 0 if equal
// -1 if prev is better, or unknown
static inline int cmpLoopNest(Loop *prev, Loop *next) {
  if (next == prev)
    return 0;
  if (next == nullptr)
    return 1;
  else if (prev == nullptr)
    return -1;
  for (Loop *L = prev; L != nullptr; L = L->getParentLoop()) {
    if (L == next)
      return 1;
  }
  return -1;
}

static inline void minCut(const DataLayout &DL, LoopInfo &OrigLI,
                          const SmallPtrSetImpl<Value *> &Recomputes,
                          const SmallPtrSetImpl<Value *> &Intermediates,
                          SmallPtrSetImpl<Value *> &Required,
                          SmallPtrSetImpl<Value *> &MinReq,
                          const ValueMap<Value *, GradientUtils::Rematerializer>
                              &rematerializableAllocations) {
  Graph G;
  for (auto V : Intermediates) {
    G[Node(V, false)].insert(Node(V, true));
    for (auto U : V->users()) {
      if (auto I = dyn_cast<Instruction>(U)) {
        for (auto pair : rematerializableAllocations) {
          if (Intermediates.count(pair.first) && pair.second.stores.count(I))
            G[Node(V, true)].insert(Node(pair.first, false));
        }
      }
      if (Intermediates.count(U)) {
        G[Node(V, true)].insert(Node(U, false));
      }
    }
  }
  for (auto pair : rematerializableAllocations) {
    if (Intermediates.count(pair.first)) {
      for (LoadInst *L : pair.second.loads) {
        if (Intermediates.count(L)) {
          G[Node(pair.first, true)].insert(Node(L, false));
        }
      }
      for (auto L : pair.second.loadLikeCalls) {
        if (Intermediates.count(L.loadCall)) {
          G[Node(pair.first, true)].insert(Node(L.loadCall, false));
        }
      }
    }
  }
  for (auto R : Required) {
    assert(Intermediates.count(R));
  }
  for (auto R : Recomputes) {
    assert(Intermediates.count(R));
  }

  Graph Orig = G;

  // Augment the flow while there is a path from source to sink
  while (1) {
    std::map<Node, Node> parent;
    bfs(G, Recomputes, parent);
    Node end(nullptr, false);
    for (auto req : Required) {
      if (parent.find(Node(req, true)) != parent.end()) {
        end = Node(req, true);
        break;
      }
    }
    if (end.V == nullptr)
      break;
    // update residual capacities of the edges and reverse edges
    // along the path
    Node v = end;
    while (1) {
      assert(parent.find(v) != parent.end());
      Node u = parent.find(v)->second;
      assert(u.V != nullptr);
      assert(G[u].count(v) == 1);
      assert(G[v].count(u) == 0);
      G[u].erase(v);
      G[v].insert(u);
      if (Recomputes.count(u.V) && u.outgoing == false)
        break;
      v = u;
    }
  }

  // Flow is maximum now, find vertices reachable from s

  std::map<Node, Node> parent;
  bfs(G, Recomputes, parent);

  // Print all edges that are from a reachable vertex to
  // non-reachable vertex in the original graph
  for (auto &pair : Orig) {
    if (parent.find(pair.first) != parent.end())
      for (auto N : pair.second) {
        if (parent.find(N) == parent.end()) {
          assert(pair.first.outgoing == 0 && N.outgoing == 1);
          assert(pair.first.V == N.V);
          MinReq.insert(N.V);
        }
      }
  }

  // When ambiguous, push to cache the last value in a computation chain
  // This should be considered in a cost for the max flow
  std::deque<Value *> todo(MinReq.begin(), MinReq.end());
  while (todo.size()) {
    auto V = todo.front();
    todo.pop_front();
    auto found = Orig.find(Node(V, true));
    if (found->second.size() == 1 && !Required.count(V)) {
      bool potentiallyRecursive =
          isa<PHINode>((*found->second.begin()).V) &&
          OrigLI.isLoopHeader(
              cast<PHINode>((*found->second.begin()).V)->getParent());
      int moreOuterLoop = cmpLoopNest(
          OrigLI.getLoopFor(cast<Instruction>(V)->getParent()),
          OrigLI.getLoopFor(
              cast<Instruction>(((*found->second.begin()).V))->getParent()));
      if (potentiallyRecursive)
        continue;
      if (moreOuterLoop == -1)
        continue;
      if (auto ASC = dyn_cast<AddrSpaceCastInst>((*found->second.begin()).V)) {
        if (ASC->getDestAddressSpace() == 11 ||
            ASC->getDestAddressSpace() == 13)
          continue;
      }
      if (moreOuterLoop == 1 ||
          (moreOuterLoop == 0 &&
           DL.getTypeSizeInBits(V->getType()) >=
               DL.getTypeSizeInBits((*found->second.begin()).V->getType()))) {
        MinReq.erase(V);
        MinReq.insert((*found->second.begin()).V);
        todo.push_back((*found->second.begin()).V);
      }
    }
  }
  return;
}
