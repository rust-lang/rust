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
#include "GradientUtils.h"

enum ValueType { Primal, Shadow };

typedef std::pair<const Value *, bool> UsageKey;

// Determine if a value is needed directly to compute the adjoint
// of the given instruction user
bool is_use_directly_needed_in_reverse(
    TypeResults &TR, const GradientUtils *gutils, const Value *val,
    const Instruction *user,
    const SmallPtrSetImpl<BasicBlock *> &oldUnreachable) {
  if (auto ainst = dyn_cast<Instruction>(val)) {
    assert(ainst->getParent()->getParent() == gutils->oldFunc);
  }
  assert(user->getParent()->getParent() == gutils->oldFunc);

  if (oldUnreachable.count(user->getParent())) return false;

  if (isa<LoadInst>(user) || isa<CastInst>(user) || isa<PHINode>(user) || isa<GetElementPtrInst>(user)) {
    return false;
  }

  // We don't need any of the input operands to compute the adjoint of a store
  // instance
  if (isa<StoreInst>(user)) {
    return false;
  }

  if (isa<CmpInst>(user) || isa<BranchInst>(user) ||
      isa<ReturnInst>(user) || isa<FPExtInst>(user) || isa<FPTruncInst>(user) ||
      (isa<InsertElementInst>(user) &&
        cast<InsertElementInst>(user)->getOperand(2) != val) ||
      (isa<ExtractElementInst>(user) &&
        cast<ExtractElementInst>(user)->getIndexOperand() != val)
       // isa<ExtractElement>(use) ||
      // isa<InsertElementInst>(use) || isa<ShuffleVectorInst>(use) ||
      // isa<ExtractValueInst>(use) || isa<AllocaInst>(use)
      /*|| isa<StoreInst>(use)*/) {
    return false;
  }

  if (auto II = dyn_cast<IntrinsicInst>(user)) {
    if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
        II->getIntrinsicID() == Intrinsic::lifetime_end ||
        II->getIntrinsicID() == Intrinsic::stacksave ||
        II->getIntrinsicID() == Intrinsic::stackrestore) {
      return false;
    }
    if (II->getIntrinsicID() == Intrinsic::fma) {
      bool needed = false;
      if (II->getArgOperand(0) == val &&
          !gutils->isConstantValue(II->getArgOperand(1)))
        needed = true;
      if (II->getArgOperand(1) == val &&
          !gutils->isConstantValue(II->getArgOperand(0)))
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

  return !gutils->isConstantInstruction(user);
}

template <ValueType VT, bool OneLevel=false>
bool is_value_needed_in_reverse(
    TypeResults &TR, const GradientUtils *gutils, const Value *inst,
    bool topLevel, std::map<UsageKey, bool> &seen,
    const SmallPtrSetImpl<BasicBlock *> &oldUnreachable) {
  auto idx = UsageKey(inst, topLevel);
  if (seen.find(idx) != seen.end())
    return seen[idx];
  if (auto ainst = dyn_cast<Instruction>(inst)) {
    assert(ainst->getParent()->getParent() == gutils->oldFunc);
  }

  // Inductively claim we aren't needed (and try to find contradiction)
  seen[idx] = false;

  if (VT != Shadow) {
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
    //llvm::errs() << " considering user: " << *user << " ici: " << gutils->isConstantInstruction(const_cast<Instruction *>(user)) << "\n";

    // A shadow value is only needed in reverse if it or one of its descendants
    // is used in an active instruction
    if (VT == Shadow) {
      if (user)
        if (!gutils->isConstantInstruction(const_cast<Instruction *>(user)))
          return true;
      if (!OneLevel && is_value_needed_in_reverse<Shadow>(TR, gutils, use, topLevel, seen,
                                             oldUnreachable)) {
        return true;
      }
      continue;
    }

    assert(VT == Primal);

    // If a sub user needs, we need
    if (!OneLevel && is_value_needed_in_reverse<VT>(TR, gutils, user, topLevel, seen,
                                         oldUnreachable)) {
      return seen[idx] = true;
    }

    // One may need to this value in the computation of loop
    // bounds/comparisons/etc (which even though not active -- will be used for
    // the reverse pass)
    //   We only need this if we're not doing the combined forward/reverse since
    //   otherwise it will use the local cache (rather than save for a separate
    //   backwards cache)
    //   We also don't need this if looking at the shadow rather than primal
    if (!topLevel) {
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

    // A pointer is only needed in the reverse pass if its non-store uses are
    // needed in the reverse pass
    //   Moreover, we only need this pointer in the reverse pass if all of its
    //   non-store users are not already cached for the reverse pass
    if (!inst->getType()->isFPOrFPVectorTy() &&
        TR.query(const_cast<Value *>(inst)).Inner0().isPossiblePointer()) {
      // continue;
      bool unknown = true;
      for (auto zu : inst->users()) {
        // Stores to a pointer are not needed for the reverse pass
        if (auto si = dyn_cast<StoreInst>(zu)) {
          if (si->getPointerOperand() == inst) {
            continue;
          }
        }

        if (isa<LoadInst>(zu) || isa<CastInst>(zu) || isa<PHINode>(zu)) {
          if (!OneLevel && is_value_needed_in_reverse<VT>(TR, gutils, zu, topLevel, seen,
                                             oldUnreachable)) {
            return seen[idx] = true;
          }
          continue;
        }

        if (auto II = dyn_cast<IntrinsicInst>(zu)) {
          if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
              II->getIntrinsicID() == Intrinsic::lifetime_end ||
              II->getIntrinsicID() == Intrinsic::stacksave ||
              II->getIntrinsicID() == Intrinsic::stackrestore ||
              II->getIntrinsicID() == Intrinsic::nvvm_barrier0_popc ||
              II->getIntrinsicID() == Intrinsic::nvvm_barrier0_and ||
              II->getIntrinsicID() == Intrinsic::nvvm_barrier0_or) {
            continue;
          }
        }

        if (auto ci = dyn_cast<CallInst>(zu)) {
          // If this instruction isn't constant (and thus we need the argument
          // to propagate to its adjoint)
          //   it may write memory and is topLevel (and thus we need to do the
          //   write in reverse) or we need this value for the reverse pass (we
          //   conservatively assume that if legal it is recomputed and not
          //   stored)
          IRBuilder<> IB(gutils->getNewFromOriginal(ci->getParent()));
          if (!gutils->isConstantInstruction(ci) ||
              !gutils->isConstantValue(
                  const_cast<Value *>((const Value *)ci)) ||
              (ci->mayWriteToMemory() && topLevel) ||
              (gutils->legalRecompute(ci, ValueToValueMapTy(), &IB,
                                      /*reverse*/ true) &&
               !OneLevel && is_value_needed_in_reverse<VT>(TR, gutils, ci, topLevel, seen,
                                              oldUnreachable))) {
            return seen[idx] = true;
          }
          continue;
        }

        // TODO add handling of call and allow interprocedural
        unknown = true;
      }
      if (!unknown)
        continue;
      // return seen[inst] = false;
    }

    if (!is_use_directly_needed_in_reverse(TR, gutils, inst, user, oldUnreachable))
      continue;

    return seen[idx] = true;
  }
  return false;
}

template <ValueType VT>
bool is_value_needed_in_reverse(
    TypeResults &TR, const GradientUtils *gutils, const Value *inst,
    bool topLevel, const SmallPtrSetImpl<BasicBlock *> &oldUnreachable) {
  std::map<std::pair<const Value *, bool>, bool> seen;
  return is_value_needed_in_reverse<VT>(TR, gutils, inst, topLevel, seen,
                                        oldUnreachable);
}

#include <map>
#include <set>
#include <deque>

struct Node {
  Value *V;
  bool outgoing;
  Node(Value* V, bool outgoing) : V(V), outgoing(outgoing) {};
  bool operator<(const Node N) const {
    if (V < N.V)
      return true;
    return !(N.V < V) && outgoing < N.outgoing;
  }
  void dump() {
    llvm::errs() << "[" << *V << ", " << (int)outgoing << "]\n";
  }
};

typedef std::map<Node, std::set<Node>> Graph;

void dump(Graph &G) {
  for (auto &pair : G) {
    llvm::errs() << "[" << *pair.first.V << ", " << (int)pair.first.outgoing << "]\n";
    for (auto N : pair.second) {
      llvm::errs() << "\t[" << *N.V << ", " << (int)N.outgoing << "]\n";
    }
  }
}
  
/* Returns true if there is a path from source 's' to sink 't' in
  residual graph. Also fills parent[] to store the path */
void bfs(Graph &G, SmallPtrSetImpl<Value*> &Recompute, std::map<Node, Node> &parent)
{
    std::deque <Node> q;
    for (auto V : Recompute) {
      Node N(V, false);
      parent.emplace(N, Node(nullptr, true));
      q.push_back(N);
    }
  
    // Standard BFS Loop
    while (!q.empty())
    {
        auto u = q.front();
        q.pop_front();
  
        for (auto v : G[u])
        {
            if (parent.find(v) == parent.end())
            {
                q.push_back(v);
                parent.emplace(v, u);
            }
        }
    }
}
  
// Prints the minimum s-t cut
void minCut(SmallPtrSetImpl<Value*> &Recomputes, SmallPtrSetImpl<Value*> &Intermediates, SmallPtrSetImpl<Value*> &Required, SmallPtrSetImpl<Value*> &MinReq)
{
    int u, v;
    Graph G;
    for (auto V : Intermediates) {
      G[Node(V, false)].insert(Node(V, true));
      for (auto U : V->users()) {
        if (Intermediates.count(U)) {
          G[Node(V, true)].insert(Node(U, false));
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
    while (1)
    {
        std::map<Node, Node> parent;
        bfs(G, Recomputes, parent);
        Node end(nullptr, false);
        for (auto req : Required) {
          if (parent.find(Node(req, true)) != parent.end()) {
            end = Node(req, true);
            break;
          }
        }
        if (end.V == nullptr) break;
        // update residual capacities of the edges and reverse edges
        // along the path
        Node v = end;
        while (1)
        {
          assert(parent.find(v) != parent.end());
          Node u = parent.find(v)->second;
            assert(u.V != nullptr);
            assert(G[u].count(v) == 1);
            G[u].erase(v);
            assert(G[v].count(u) == 0);
            G[v].insert(u);
            if (Recomputes.count(u.V) && u.outgoing == false) break;
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

    return;
}