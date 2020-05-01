
/*
 * GradientUtils.h - Gradient Utility data structures and functions
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

#ifndef ENZYME_GUTILS_H_
#define ENZYME_GUTILS_H_

#include <algorithm>
#include <deque>
#include <map>

#include <llvm/Config/llvm-config.h>

#include "Utils.h"
#include "ActiveVariable.h"
#include "SCEV/ScalarEvolutionExpander.h"

#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/Dominators.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/LoopInfo.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"

#include "llvm/Support/Casting.h"

#include "llvm/Transforms/Utils/ValueMapper.h"

#include "llvm/Support/ErrorHandling.h"

#include "ActiveVariable.h"
#include "EnzymeLogic.h"

using namespace llvm;

enum class AugmentedStruct;
typedef struct {
  PHINode* var;
  Instruction* incvar;
  AllocaInst* antivaralloc;
  BasicBlock* latchMerge;
  BasicBlock* header;
  BasicBlock* preheader;
  bool dynamic;
  //limit is last value, iters is number of iters (thus iters = limit + 1)
  Value* limit;
  SmallPtrSet<BasicBlock*, 8> exitBlocks;
  Loop* parent;
} LoopContext;

enum class UnwrapMode {
  LegalFullUnwrap,
  AttemptFullUnwrapWithLookup,
  AttemptFullUnwrap,
  AttemptSingleUnwrap,
};

static inline bool operator==(const LoopContext& lhs, const LoopContext &rhs) {
    return lhs.parent == rhs.parent;
}

class GradientUtils {
public:
  llvm::Function *newFunc;
  llvm::Function *oldFunc;
  ValueToValueMapTy invertedPointers;
  DominatorTree DT;
  DominatorTree OrigDT;
  SmallPtrSet<Value*,4> constants;
  SmallPtrSet<Value*,20> nonconstant;
  SmallPtrSet<Value*,4> constant_values;
  SmallPtrSet<Value*,2> nonconstant_values;
  LoopInfo LI;
  AssumptionCache AC;
  ScalarEvolution SE;
  std::map<Loop*, LoopContext> loopContexts;
  SmallPtrSet<Instruction*, 10> originalInstructions;
  SmallVector<BasicBlock*, 12> originalBlocks;
  ValueMap<BasicBlock*,BasicBlock*> reverseBlocks;
  BasicBlock* inversionAllocs;
  ValueToValueMapTy scopeMap;
  std::map<AllocaInst*, std::set<CallInst*>> scopeFrees;
  std::map<AllocaInst*, std::vector<CallInst*>> scopeAllocs;
  std::map<AllocaInst*, std::vector<Value*>> scopeStores;
  SmallVector<PHINode*, 4> fictiousPHIs;
  ValueToValueMapTy originalToNewFn;

  const std::map<Instruction*, bool>* can_modref_map;


  Value* getNewFromOriginal(Value* originst) const {
    assert(originst);
    auto f = originalToNewFn.find(originst);
    if (f == originalToNewFn.end()) {
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        dumpMap(originalToNewFn);
        llvm::errs() << *originst << "\n";
    }
    assert(f != originalToNewFn.end());
    if (f->second == nullptr) {
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << *originst << "\n";
    }
    assert(f->second);
    return f->second;
  }
  Instruction* getNewFromOriginal(Instruction* newinst) const {
    return cast<Instruction>(getNewFromOriginal((Value*)newinst));
  }

  Value* hasUninverted(Value* inverted) const {
    for(auto v: invertedPointers) {
        if (v.second == inverted) return const_cast<Value*>(v.first);
    }
    return nullptr;
  }

  Value* getOriginal(Value* newinst) const {
    for(auto v: originalToNewFn) {
        if (v.second == newinst) return const_cast<Value*>(v.first);
    }
    llvm::errs() << *newinst << "\n";
    assert(0 && "could not invert new inst");
    report_fatal_error("could not invert new inst");
  }
  Instruction* getOriginal(Instruction* newinst) const {
    return cast<Instruction>(getOriginal((Value*)newinst));
  }
  CallInst* getOriginal(CallInst* newinst) const {
    return cast<CallInst>(getOriginal((Value*)newinst));
  }
  BasicBlock* getOriginal(BasicBlock* newinst) const {
    return cast<BasicBlock>(getOriginal((Value*)newinst));
  }

private:
  SmallVector<Value*, 4> addedMallocs;
  unsigned tapeidx;
  Value* tape;

  std::map<std::pair<Value*, int>, MDNode*> invariantGroups;
  std::map<Value*, MDNode*> valueInvariantGroups;
  std::map<std::pair<Value*, BasicBlock*>, Value*> unwrap_cache;
  std::map<std::pair<Value*, BasicBlock*>, Value*> lookup_cache;
public:
  bool legalRecompute(Value* val, const ValueToValueMapTy& available);
  bool shouldRecompute(Value* val, const ValueToValueMapTy& available);

  void replaceAWithB(Value* A, Value* B) {
      for(unsigned i=0; i<addedMallocs.size(); i++) {
        if (addedMallocs[i] == A) {
            addedMallocs[i] = B;
        }
      }

    if (scopeMap.find(A) != scopeMap.end()) {
        scopeMap[B] = scopeMap[A];
        scopeMap.erase(A);
    }
    if (invertedPointers.find(A) != invertedPointers.end()) {
        invertedPointers[B] = invertedPointers[A];
        invertedPointers.erase(A);
    }
    A->replaceAllUsesWith(B);
  }

  void erase(Instruction *I) {
    assert(I);
    invertedPointers.erase(I);
    constants.erase(I);
    constant_values.erase(I);
    nonconstant.erase(I);
    nonconstant_values.erase(I);
    originalInstructions.erase(I);
    if (scopeMap.find(I) != scopeMap.end()) {
        scopeFrees.erase(cast<AllocaInst>(scopeMap[I]));
        scopeAllocs.erase(cast<AllocaInst>(scopeMap[I]));
        scopeStores.erase(cast<AllocaInst>(scopeMap[I]));
    }
    if (auto ai = dyn_cast<AllocaInst>(I)) {
        scopeFrees.erase(ai);
        scopeAllocs.erase(ai);
        scopeStores.erase(ai);
    }
    scopeMap.erase(I);
    SE.eraseValueFromMap(I);
    originalToNewFn.erase(I);
    eraser:
    for(auto v: originalToNewFn) {
        if (v.second == I) {
            originalToNewFn.erase(v.first);
            goto eraser;
        }
    }
    for(auto v: scopeMap) {
        if (v.second == I) {
            llvm::errs() << *oldFunc << "\n";
            llvm::errs() << *newFunc << "\n";
            dumpScope();
            llvm::errs() << *v.first << "\n";
            llvm::errs() << *I << "\n";
            assert(0 && "erasing something in scope map");
        }
    }
    if (auto ci = dyn_cast<CallInst>(I))
    for(auto v: scopeFrees) {
        if (v.second.count(ci)) {
            llvm::errs() << *oldFunc << "\n";
            llvm::errs() << *newFunc << "\n";
            llvm::errs() << *v.first << "\n";
            llvm::errs() << *I << "\n";
            assert(0 && "erasing something in scopeFrees map");
        }
    }
    if (auto ci = dyn_cast<CallInst>(I))
    for(auto v: scopeAllocs) {
        if (std::find(v.second.begin(), v.second.end(), ci) != v.second.end()) {
            llvm::errs() << *oldFunc << "\n";
            llvm::errs() << *newFunc << "\n";
            llvm::errs() << *v.first << "\n";
            llvm::errs() << *I << "\n";
            assert(0 && "erasing something in scopeAllocs map");
        }
    }
    for(auto v: scopeStores) {
        if (std::find(v.second.begin(), v.second.end(), I) != v.second.end()) {
            llvm::errs() << *oldFunc << "\n";
            llvm::errs() << *newFunc << "\n";
            llvm::errs() << *v.first << "\n";
            llvm::errs() << *I << "\n";
            assert(0 && "erasing something in scopeStores map");
        }
    }
    for(auto v: invertedPointers) {
        if (v.second == I) {
            llvm::errs() << *oldFunc << "\n";
            llvm::errs() << *newFunc << "\n";
            dumpPointers();
            llvm::errs() << *v.first << "\n";
            llvm::errs() << *I << "\n";
            assert(0 && "erasing something in invertedPointers map");
        }
    }

    {
        std::vector<std::pair<Value*, BasicBlock*>> unwrap_cache_pairs;
        for(auto& a : unwrap_cache) {
            if (a.second == I) {
                unwrap_cache_pairs.push_back(a.first);
            }
            if (a.first.first == I) {
                unwrap_cache_pairs.push_back(a.first);
            }
        }
        for(auto a : unwrap_cache_pairs) {
            unwrap_cache.erase(a);
        }
    }

    {
        std::vector<std::pair<Value*, BasicBlock*>> lookup_cache_pairs;
        for(auto& a : lookup_cache) {
            if (a.second == I) {
                lookup_cache_pairs.push_back(a.first);
            }
            if (a.first.first == I) {
                lookup_cache_pairs.push_back(a.first);
            }
        }
        for(auto a : lookup_cache_pairs) {
            lookup_cache.erase(a);
        }
    }

    if (!I->use_empty()) {
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << *I << "\n";
    }
    assert(I->use_empty());
    I->eraseFromParent();
  }
  //TODO consider invariant group and/or valueInvariant group

  void setTape(Value* newtape) {
    assert(tape == nullptr);
    assert(newtape != nullptr);
    assert(tapeidx == 0);
    assert(addedMallocs.size() == 0);
    if (!newtape->getType()->isStructTy()) {
        llvm::errs() << "incorrect tape type: " << *newtape << "\n";
    }
    assert(newtape->getType()->isStructTy());
    tape = newtape;
  }

  void dumpPointers() {
    llvm::errs() << "invertedPointers:\n";
    for(auto a : invertedPointers) {
        llvm::errs() << "   invertedPointers[" << *a.first << "] = " << *a.second << "\n";
    }
    llvm::errs() << "end invertedPointers\n";
  }

  void dumpScope() {
    llvm::errs() << "scope:\n";
    for(auto a : scopeMap) {
        llvm::errs() << "   scopeMap[" << *a.first << "] = " << *a.second << "\n";
    }
    llvm::errs() << "end scope\n";
  }

  Instruction* createAntiMalloc(CallInst *call, unsigned idx) {
    assert(call->getParent()->getParent() == newFunc);
    PHINode* placeholder = cast<PHINode>(invertedPointers[call]);

    assert(placeholder->getParent()->getParent() == newFunc);
	placeholder->setName("");
    IRBuilder<> bb(placeholder);

	SmallVector<Value*, 8> args;
	for(unsigned i=0;i < call->getNumArgOperands(); i++) {
		args.push_back(call->getArgOperand(i));
	}
    Instruction* anti = bb.CreateCall(call->getCalledFunction(), args, call->getName()+"'mi");
    cast<CallInst>(anti)->setAttributes(call->getAttributes());
    cast<CallInst>(anti)->setCallingConv(call->getCallingConv());
    cast<CallInst>(anti)->setTailCallKind(call->getTailCallKind());
    cast<CallInst>(anti)->setDebugLoc(call->getDebugLoc());
    cast<CallInst>(anti)->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
    cast<CallInst>(anti)->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
    cast<CallInst>(anti)->setMetadata("enzyme_activity_inst", MDNode::get(placeholder->getContext(), MDString::get(placeholder->getContext(), "active")));
    cast<CallInst>(anti)->setMetadata("enzyme_activity_value", MDNode::get(placeholder->getContext(), MDString::get(placeholder->getContext(), "active")));

    invertedPointers[call] = anti;
    assert(placeholder != anti);
    bb.SetInsertPoint(placeholder->getNextNode());
    replaceAWithB(placeholder, anti);
    erase(placeholder);

    anti = cast<Instruction>(addMalloc(bb, anti, idx));
    invertedPointers[call] = anti;

    if (tape == nullptr) {
        auto dst_arg = bb.CreateBitCast(anti,Type::getInt8PtrTy(call->getContext()));
        auto val_arg = ConstantInt::get(Type::getInt8Ty(call->getContext()), 0);
        auto len_arg = bb.CreateZExtOrTrunc(call->getArgOperand(0), Type::getInt64Ty(call->getContext()));
        auto volatile_arg = ConstantInt::getFalse(call->getContext());

#if LLVM_VERSION_MAJOR == 6
        auto align_arg = ConstantInt::get(Type::getInt32Ty(call->getContext()), 0);
        Value *nargs[] = { dst_arg, val_arg, len_arg, align_arg, volatile_arg };
#else
        Value *nargs[] = { dst_arg, val_arg, len_arg, volatile_arg };
#endif

        Type *tys[] = {dst_arg->getType(), len_arg->getType()};

        auto memset = cast<CallInst>(bb.CreateCall(Intrinsic::getDeclaration(newFunc->getParent(), Intrinsic::memset, tys), nargs));
        //memset->addParamAttr(0, Attribute::getWithAlignment(Context, inst->getAlignment()));
        memset->addParamAttr(0, Attribute::NonNull);
    }

    return anti;
  }

  unsigned getIndex(std::pair<Instruction*, CacheType> idx, std::map<std::pair<Instruction*, CacheType>, unsigned> &mapping) {
    if (tape) {
        if (mapping.find(idx) == mapping.end()) {
            llvm::errs() << "oldFunc: " <<*oldFunc << "\n";
            llvm::errs() << "newFunc: " <<*newFunc << "\n";
            llvm::errs() << " <mapping>\n";
            for(auto &p : mapping) {
                llvm::errs() << "   idx: " << *p.first.first << ", " << p.first.second << " pos=" << p.second << "\n";
            }
            llvm::errs() << " </mapping>\n";

            if (mapping.find(idx) == mapping.end()) {
                llvm::errs() << "idx: " << *idx.first << ", " << idx.second << "\n";
                assert(0 && "could not find index in mapping");
            }
        }
        return mapping[idx];
    } else {
        if (mapping.find(idx) != mapping.end()) {
            return mapping[idx];
        }
        //llvm::errs() << "adding to map: ";
        //    llvm::errs() << "idx: " << *idx.first << ", " << idx.second << " pos= " << tapeidx << "\n";
        mapping[idx] = tapeidx;
        tapeidx++;
        return mapping[idx];
    }
  }

  Instruction* addMalloc(IRBuilder<> &BuilderQ, Instruction* malloc, unsigned idx) {
      assert(malloc);
      return cast<Instruction>(addMalloc(BuilderQ, (Value*)malloc, idx));
  }

  Value* addMalloc(IRBuilder<> &BuilderQ, Value* malloc, unsigned idx) {

    if (tape) {
      if (!tape->getType()->isStructTy()) {
          llvm::errs() << "addMalloc incorrect tape type: " << *tape << "\n";
      }
      assert(tape->getType()->isStructTy());
      if (idx >= cast<StructType>(tape->getType())->getNumElements()) {
        llvm::errs() << "oldFunc: " <<*oldFunc << "\n";
        llvm::errs() << "newFunc: " <<*newFunc << "\n";
        if (malloc)
          llvm::errs() << "malloc: " <<*malloc << "\n";
        llvm::errs() << "tape: " <<*tape << "\n";
        llvm::errs() << "idx: " << idx << "\n";
      }
      assert(idx < cast<StructType>(tape->getType())->getNumElements());
      Instruction* ret = cast<Instruction>(BuilderQ.CreateExtractValue(tape, {idx}));

      if (auto inst = dyn_cast_or_null<Instruction>(malloc)) {
        if (MDNode* md = inst->getMetadata("enzyme_activity_value")) {
            ret->setMetadata("enzyme_activity_value", md);
        }
        ret->setMetadata("enzyme_activity_inst", MDNode::get(ret->getContext(), {MDString::get(ret->getContext(), "const")}));
        //llvm::errs() << "replacing " << *malloc << " with " << *ret << "\n";
        originalInstructions.insert(ret);
      }

      if (ret->getType()->isEmptyTy()) {
        if (auto inst = dyn_cast_or_null<Instruction>(malloc)) {
          if (inst->getType() != ret->getType()) {
              llvm::errs() << "oldFunc: " <<*oldFunc << "\n";
              llvm::errs() << "newFunc: " <<*newFunc << "\n";
              llvm::errs() << "inst==malloc: " <<*inst << "\n";
              llvm::errs() << "ret: " <<*ret << "\n";

          }
          assert(inst->getType() == ret->getType());
          inst->replaceAllUsesWith(UndefValue::get(ret->getType()));
          erase(inst);
        }

        return UndefValue::get(ret->getType());
      }

      BasicBlock* parent = BuilderQ.GetInsertBlock();
	  	if (Instruction* inst = dyn_cast_or_null<Instruction>(malloc)) {
			 parent = inst->getParent();
		  }

	    LoopContext lc;
    	bool inLoop = getContext(parent, lc);

      if (!inLoop) {
        if (malloc) ret->setName(malloc->getName()+"_fromtape");
      } else {
        erase(ret);
        IRBuilder<> entryBuilder(inversionAllocs);
        entryBuilder.setFastMathFlags(getFast());
        ret = cast<Instruction>(entryBuilder.CreateExtractValue(tape, {idx}));


        //scopeMap[inst] = cache;
        Type* innerType = ret->getType();
        for(const auto unused : getSubLimits(BuilderQ.GetInsertBlock()) ) {
          if (!isa<PointerType>(innerType)) {
            llvm::errs() << "fn: " << *BuilderQ.GetInsertBlock()->getParent() << "\n";
            llvm::errs() << "bq insertblock: " << *BuilderQ.GetInsertBlock() << "\n";
            llvm::errs() << "ret: " << *ret << " type: " << *ret->getType() << "\n";
            llvm::errs() << "innerType: " << *innerType << "\n";
            if (malloc) llvm::errs() << " malloc: " << *malloc << "\n";
          }
          assert(isa<PointerType>(innerType));
          innerType = cast<PointerType>(innerType)->getElementType();
        }
        if (malloc) {
          if (innerType != malloc->getType()) {
            llvm::errs() << *cast<Instruction>(malloc)->getParent()->getParent() << "\n";
            llvm::errs() << "innerType: " << *innerType << "\n";
            llvm::errs() << "malloc->getType(): " << *malloc->getType() << "\n";
            llvm::errs() << "ret: " << *ret << "\n";
            llvm::errs() << "malloc: " << *malloc << "\n";
          }
          assert(innerType == malloc->getType());
        }

        AllocaInst* cache = createCacheForScope(BuilderQ.GetInsertBlock(), innerType, "mdyncache_fromtape", true, false);
        entryBuilder.CreateStore(ret, cache);

        auto v = lookupValueFromCache(BuilderQ, BuilderQ.GetInsertBlock(), cache);
        if (malloc) {
          assert(v->getType() == malloc->getType());
          if (auto inst = dyn_cast<Instruction>(malloc)) {
            if (MDNode* md = inst->getMetadata("enzyme_activity_value")) {
                ret->setMetadata("enzyme_activity_value", md);
            }
            ret->setMetadata("enzyme_activity_inst", MDNode::get(ret->getContext(), {MDString::get(ret->getContext(), "const")}));
            originalInstructions.insert(inst);
          }
        }
        scopeMap[v] = cache;
        originalInstructions.erase(ret);
        ret = cast<Instruction>(v);
      }

      if (malloc && !isa<UndefValue>(malloc)) {
        if (malloc->getType() != ret->getType()) {
          llvm::errs() << *oldFunc << "\n";
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << *malloc << "\n";
          llvm::errs() << *ret << "\n";
        }
        assert(malloc->getType() == ret->getType());

        if (invertedPointers.find(malloc) != invertedPointers.end()) {
  				invertedPointers[ret] = invertedPointers[malloc];
	   			invertedPointers.erase(malloc);
		  	}

        if (scopeMap.find(malloc) != scopeMap.end()) {
          // There already exists an alloaction for this, we should fully remove it
          if (!inLoop) {

            // Remove stores into
            auto stores = scopeStores[cast<AllocaInst>(scopeMap[malloc])];
            scopeStores.erase(cast<AllocaInst>(scopeMap[malloc]));
            for(int i=stores.size()-1; i>=0; i--) {
                if (auto inst = dyn_cast<Instruction>(stores[i])) {
                    erase(inst);
                }
            }

            std::vector<User*> users;
            for (auto u : scopeMap[malloc]->users()) {
                users.push_back(u);
            }
            for(auto u : users) {
              if (auto li = dyn_cast<LoadInst>(u)) {
                IRBuilder<> lb(li);
                ValueToValueMapTy empty;
                li->replaceAllUsesWith(unwrapM(ret, lb, empty, UnwrapMode::LegalFullUnwrap));
                erase(li);
              } else {
                llvm::errs() << "newFunc: " << *newFunc << "\n";
                llvm::errs() << "malloc: " << *malloc << "\n";
                llvm::errs() << "scopeMap[malloc]: " << *scopeMap[malloc] << "\n";
                llvm::errs() << "u: " << *u << "\n";
                assert(0 && "illegal use for out of loop scopeMap");
              }
            }

            {
            Instruction* preerase = cast<Instruction>(scopeMap[malloc]);
            scopeMap.erase(malloc);
            erase(preerase);
            }
          } else {
            // Remove stores into
            auto stores = scopeStores[cast<AllocaInst>(scopeMap[malloc])];
            scopeStores.erase(cast<AllocaInst>(scopeMap[malloc]));
            for(int i=stores.size()-1; i>=0; i--) {
              if (auto inst = dyn_cast<Instruction>(stores[i])) {
                erase(inst);
              }
            }


                    //Remove allocations for scopealloc since it is already allocated by the augmented forward pass
                    auto allocs = scopeAllocs[cast<AllocaInst>(scopeMap[malloc])];
                    scopeAllocs.erase(cast<AllocaInst>(scopeMap[malloc]));
                    for(auto allocinst : allocs) {
                        CastInst* cast = nullptr;
                        StoreInst* store = nullptr;
                        for(auto use : allocinst->users()) {
                            if (auto ci = dyn_cast<CastInst>(use)) {
                                assert(cast == nullptr);
                                cast = ci;
                            }
                            if (auto si = dyn_cast<StoreInst>(use)) {
                                if (si->getValueOperand() == allocinst) {
                                    assert(store == nullptr);
                                    store = si;
                                }
                            }
                        }
                        if (cast) {
                            assert(store == nullptr);
                            for(auto use : cast->users()) {
                                if (auto si = dyn_cast<StoreInst>(use)) {
                                    if (si->getValueOperand() == cast) {
                                        assert(store == nullptr);
                                        store = si;
                                    }
                                }
                            }
                        }
                        /*
                        if (!store) {
                            allocinst->getParent()->getParent()->dump();
                            allocinst->dump();
                        }
                        assert(store);
                        erase(store);
                        */

                        Instruction* storedinto = cast ? (Instruction*)cast : (Instruction*)allocinst;
                        for(auto use : storedinto->users()) {
                            //llvm::errs() << " found use of " << *storedinto << " of " << use << "\n";
                            if (auto si = dyn_cast<StoreInst>(use)) erase(si);
                        }

                        if (cast) erase(cast);
                        //llvm::errs() << "considering inner loop for malloc: " << *malloc << " allocinst " << *allocinst << "\n";
                        erase(allocinst);
                    }

                    // Remove frees
                    auto tofree = scopeFrees[cast<AllocaInst>(scopeMap[malloc])];
                    scopeFrees.erase(cast<AllocaInst>(scopeMap[malloc]));
                    for(auto freeinst : tofree) {
                        std::deque<Value*> ops = { freeinst->getArgOperand(0) };
                        erase(freeinst);

                        while(ops.size()) {
                            auto z = dyn_cast<Instruction>(ops[0]);
                            ops.pop_front();
                            if (z && z->getNumUses() == 0) {
                                for(unsigned i=0; i<z->getNumOperands(); i++) {
                                    ops.push_back(z->getOperand(i));
                                }
                                erase(z);
                            }
                        }
                    }

                    // uses of the alloc
                    std::vector<User*> users;
                    for (auto u : scopeMap[malloc]->users()) {
                        users.push_back(u);
                    }
                    for( auto u : users) {
                        if (auto li = dyn_cast<LoadInst>(u)) {
                            IRBuilder<> lb(li);
                            //llvm::errs() << "fixing li: " << *li << "\n";
                            auto replacewith = lb.CreateExtractValue(tape, {idx});
                            //llvm::errs() << "fixing with rw: " << *replacewith << "\n";
                            li->replaceAllUsesWith(replacewith);
                            erase(li);
                        } else {
                            llvm::errs() << "newFunc: " << *newFunc << "\n";
                            llvm::errs() << "malloc: " << *malloc << "\n";
                            llvm::errs() << "scopeMap[malloc]: " << *scopeMap[malloc] << "\n";
                            llvm::errs() << "u: " << *u << "\n";
                            assert(0 && "illegal use for out of loop scopeMap");
                        }
                    }

                    //cast<Instruction>(scopeMap[malloc])->getParent()->getParent()->dump();

                    //llvm::errs() << "did erase for malloc: " << *malloc << " " << *scopeMap[malloc] << "\n";

                    Instruction* preerase = cast<Instruction>(scopeMap[malloc]);
                    scopeMap.erase(malloc);
                    erase(preerase);


                }
            }
            //llvm::errs() << "replacing " << *malloc << " with " << *ret << "\n";
            cast<Instruction>(malloc)->replaceAllUsesWith(ret);
            std::string n = malloc->getName().str();
            erase(cast<Instruction>(malloc));
            ret->setName(n);
        }
        return ret;
    } else {
      assert(malloc);
      //assert(!isa<PHINode>(malloc));

      assert(idx == addedMallocs.size());

      if (isa<UndefValue>(malloc)) {
        addedMallocs.push_back(malloc);
        return malloc;
      }

	  BasicBlock* parent = BuilderQ.GetInsertBlock();
	  if (Instruction* inst = dyn_cast_or_null<Instruction>(malloc)) {
			parent = inst->getParent();
	  }
	  LoopContext lc;
      bool inLoop = getContext(parent, lc);

      if (!inLoop) {
	    addedMallocs.push_back(malloc);
        return malloc;
      }

      ensureLookupCached(cast<Instruction>(malloc), /*shouldFree=*/reverseBlocks.size() > 0);
      assert(scopeMap[malloc]);

      Instruction* toadd = scopeAllocs[cast<AllocaInst>(scopeMap[malloc])][0];
      for(auto u : toadd->users()) {
          if (auto ci = dyn_cast<CastInst>(u)) {
             toadd = ci;
          }
      }

      //llvm::errs() << " malloc: " << *malloc << "\n";
      //llvm::errs() << " toadd: " << *toadd << "\n";
      Type* innerType = toadd->getType();
      for(const auto unused : getSubLimits(BuilderQ.GetInsertBlock()) ) {
        innerType = cast<PointerType>(innerType)->getElementType();
      }
      assert(innerType == malloc->getType());

      addedMallocs.push_back(toadd);
      return malloc;
    }
    llvm::errs() << "Fell through on addMalloc. This should never happen.\n";
    assert(false);
  }

  const SmallVectorImpl<Value*> & getMallocs() const {
    return addedMallocs;
  }
public:
  TargetLibraryInfo &TLI;
  AAResults &AA;
  TypeAnalysis &TA;
  GradientUtils(Function* newFunc_, Function* oldFunc_, TargetLibraryInfo &TLI_, TypeAnalysis &TA_, AAResults &AA_, ValueToValueMapTy& invertedPointers_, const SmallPtrSetImpl<Value*> &constants_, const SmallPtrSetImpl<Value*> &nonconstant_, const SmallPtrSetImpl<Value*> &constantvalues_, const SmallPtrSetImpl<Value*> &returnvals_, ValueToValueMapTy& originalToNewFn_) :
      newFunc(newFunc_), oldFunc(oldFunc_), invertedPointers(), DT(*newFunc_), OrigDT(*oldFunc_), constants(constants_.begin(), constants_.end()), nonconstant(nonconstant_.begin(), nonconstant_.end()), constant_values(constantvalues_.begin(), constantvalues_.end()), nonconstant_values(returnvals_.begin(), returnvals_.end()), LI(DT), AC(*newFunc_), SE(*newFunc_, TLI_, AC, DT, LI), inversionAllocs(nullptr), TLI(TLI_), AA(AA_), TA(TA_) {
        invertedPointers.insert(invertedPointers_.begin(), invertedPointers_.end());
        originalToNewFn.insert(originalToNewFn_.begin(), originalToNewFn_.end());
          for (BasicBlock &BB: *newFunc) {
            originalBlocks.emplace_back(&BB);
            for(Instruction &I : BB) {
                originalInstructions.insert(&I);
            }
          }
        tape = nullptr;
        tapeidx = 0;
        assert(originalBlocks.size() > 0);
        inversionAllocs = BasicBlock::Create(newFunc_->getContext(), "allocsForInversion", newFunc);
    }

public:
  static GradientUtils* CreateFromClone(Function *todiff, TargetLibraryInfo &TLI, TypeAnalysis &TA, AAResults &AA, const std::set<unsigned> & constant_args, bool returnUsed, bool differentialReturn, std::map<AugmentedStruct, unsigned>& returnMapping);

  StoreInst* setPtrDiffe(Value* ptr, Value* newval, IRBuilder<> &BuilderM) {
      ptr = invertPointerM(ptr, BuilderM);
      return BuilderM.CreateStore(newval, ptr);
  }

  void prepareForReverse() {
    assert(reverseBlocks.size() == 0);
    for (BasicBlock *BB: originalBlocks) {
      reverseBlocks[BB] = BasicBlock::Create(BB->getContext(), "invert" + BB->getName(), newFunc);
    }
    assert(reverseBlocks.size() != 0);
  }

  BasicBlock* originalForReverseBlock(BasicBlock& BB2) const {
    assert(reverseBlocks.size() != 0);
    for(auto BB : originalBlocks) {
        auto it = reverseBlocks.find(BB);
        assert(it != reverseBlocks.end());
        if (it->second == &BB2) {
            return BB;
        }
    }
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << BB2 << "\n";
    assert(0 && "could not find original block for given reverse block");
    report_fatal_error("could not find original block for given reverse block");
  }

  //! This cache stores blocks we may insert as part of getReverseOrLatchMerge to handle inverse iv iteration
  //  As we don't want to create redundant blocks, we use this convenient cache
  std::map<std::tuple<BasicBlock*, BasicBlock*>, BasicBlock*> newBlocksForLoop_cache;
  BasicBlock* getReverseOrLatchMerge(BasicBlock* BB, BasicBlock* branchingBlock);

  void forceContexts();

  bool getContext(BasicBlock* BB, LoopContext& loopContext);

  bool isOriginalBlock(const BasicBlock &BB) const {
    for(auto A : originalBlocks) {
        if (A == &BB) return true;
    }
    return false;
  }

  bool isConstantValueInternal(Value* val, AAResults &AA, TypeResults &TR) {
	  cast<Value>(val);
    return isconstantValueM(TR, val, constants, nonconstant, constant_values, nonconstant_values, AA);
  };

  bool isConstantInstructionInternal(Instruction* val, AAResults &AA, TypeResults &TR) {
    cast<Instruction>(val);
    return isconstantM(TR, val, constants, nonconstant, constant_values, nonconstant_values, AA);
  }

  SmallPtrSet<Instruction*,4> replaceableCalls;
  void eraseStructuralStoresAndCalls() {

      for(auto pp : fictiousPHIs) {
        pp->replaceAllUsesWith(ConstantPointerNull::get(cast<PointerType>(pp->getType())));
        erase(pp);
      }
      fictiousPHIs.clear();

      for(BasicBlock* BB: this->originalBlocks) {
        auto term = BB->getTerminator();
        if (isa<UnreachableInst>(term)) continue;

        for (auto I = BB->begin(), E = BB->end(); I != E;) {
          Instruction* inst = &*I;
          assert(inst);
          I++;

          if (originalInstructions.find(inst) == originalInstructions.end()) continue;

          if (isa<StoreInst>(inst)) {
            erase(inst);
            continue;
          }
        }
      }

      for(BasicBlock* BB: this->originalBlocks) {
        auto term = BB->getTerminator();
        if (isa<UnreachableInst>(term)) continue;

        for (auto I = BB->begin(), E = BB->end(); I != E;) {
          Instruction* inst = &*I;
          assert(inst);
          I++;

          if (originalInstructions.find(inst) == originalInstructions.end()) continue;

          if (!(isa<SwitchInst>(inst) || isa<BranchInst>(inst) || isa<ReturnInst>(inst)) && isConstantInstruction(inst) && isConstantValue(inst) ) {
            if (inst->getNumUses() == 0) {
                erase(inst);
			    continue;
            }
          } else {
            if (auto inti = dyn_cast<IntrinsicInst>(inst)) {
                if (inti->getIntrinsicID() == Intrinsic::memset || inti->getIntrinsicID() == Intrinsic::memcpy || inti->getIntrinsicID() == Intrinsic::memmove) {
                    erase(inst);
                    continue;
                }
            }
            if (replaceableCalls.find(inst) != replaceableCalls.end()) {
                if (inst->getNumUses() != 0) {
                } else {
                    erase(inst);
                    continue;
                }
            }
          }
        }
      }
  }

  void forceActiveDetection(AAResults &AA, TypeResults &TR) {
      for(auto a = oldFunc->arg_begin(); a != oldFunc->arg_end(); a++) {
        if (constants.find(a) == constants.end() && nonconstant.find(a) == nonconstant.end()) continue;

        bool const_value = isConstantValueInternal(a, AA, TR);
        //a->addAttr(llvm::Attribute::get(a->getContext(), "enzyme_activity_value", const_value ? "const" : "active"));
        cast<Argument>(getNewFromOriginal(a))->addAttr(llvm::Attribute::get(a->getContext(), "enzyme_activity_value", const_value ? "const" : "active"));
      }

      for(BasicBlock& BB: *oldFunc) {
          for(Instruction &I : BB) {
              bool const_inst = isConstantInstructionInternal(&I, AA, TR);

              getNewFromOriginal(&I)->setMetadata("enzyme_activity_inst", MDNode::get(I.getContext(), MDString::get(I.getContext(), const_inst ? "const" : "active")));
              //I.setMetadata(const_inst ? "enzyme_constinst" : "enzyme_activeinst", MDNode::get(I.getContext(), {}));

              //I.addAttr(llvm::Attribute::get(I.getContext(), "enzyme_activity_inst", const_inst ? "const" : "active"));
              bool const_value = isConstantValueInternal(&I, AA, TR);
              //I.setMetadata(const_value ? "enzyme_constvalue" : "enzyme_activevalue", MDNode::get(I.getContext(), {}));
              getNewFromOriginal(&I)->setMetadata("enzyme_activity_value", MDNode::get(I.getContext(), MDString::get(I.getContext(), const_value ? "const" : "active")));
              //I.addAttr(llvm::Attribute::get(I.getContext(), "enzyme_activity_value", const_value ? "const" : "active"));
          }
      }
  }

  void cleanupActiveDetection() {
      //llvm::errs() << "pre cleanup: " << *newFunc << "\n";

      for(auto a = newFunc->arg_begin(); a != newFunc->arg_end(); a++) {
        a->getParent()->removeParamAttr(a->getArgNo(), "enzyme_activity_value");
        //a->getParent()->getAttributes().removeParamAttribute(a->getContext(), a->getArgNo(), "enzyme_activity_value");
      }

      for(BasicBlock& BB: *newFunc) {
          for(Instruction &I : BB) {
              I.setMetadata("enzyme_activity_inst", nullptr);
              I.setMetadata("enzyme_activity_value", nullptr);
          }
      }
      //llvm::errs() << "post cleanup: " << *newFunc << "\n";
  }

  llvm::StringRef getAttribute(Argument* arg, std::string attr) const {
    return arg->getParent()->getAttributes().getParamAttr(arg->getArgNo(), attr).getValueAsString();
  }

  bool isConstantValue(Value* val) const {
    if (auto inst = dyn_cast<Instruction>(val)) {
        if (originalInstructions.find(inst) == originalInstructions.end()) return true;
        if (auto md = inst->getMetadata("enzyme_activity_value")) {
            auto res = cast<MDString>(md->getOperand(0))->getString();
            if (res == "const") return true;
            if (res == "active") return false;
        }
    }

    if (auto arg = dyn_cast<Argument>(val)) {
        auto res = getAttribute(arg, "enzyme_activity_value");
        if (res == "const") return true;
        if (res == "active") return false;
    }

    //! False so we can replace function with augmentation
    if (isa<Function>(val)) {
        return false;
    }

    if (auto gv = dyn_cast<GlobalVariable>(val)) {
        if (hasMetadata(gv, "enzyme_shadow")) return false;
        if (auto md = gv->getMetadata("enzyme_activity_value")) {
            auto res = cast<MDString>(md->getOperand(0))->getString();
            if (res == "const") return true;
            if (res == "active") return false;
        }
        if (nonmarkedglobals_inactive) return true;
        goto err;
    }
    if (isa<GlobalValue>(val)) {
        if (nonmarkedglobals_inactive) return true;
        goto err;
    }

    //TODO allow gv/inline asm
    //if (isa<GlobalValue>(val) || isa<InlineAsm>(val)) return isConstantValueInternal(val);
    if (isa<Constant>(val) || isa<UndefValue>(val) || isa<MetadataAsValue>(val)) return true;

    err:;
    llvm::errs() << *oldFunc << "\n";
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << *val << "\n";
    llvm::errs() << "  unknown did status attribute\n";
    assert(0 && "bad");
    exit(1);
  }

  bool isConstantInstruction(Instruction* inst) const {
    if (originalInstructions.find(inst) == originalInstructions.end()) return true;

    if (MDNode* md = inst->getMetadata("enzyme_activity_inst")) {
        auto res = cast<MDString>(md->getOperand(0))->getString();
        if (res == "const") return true;
        if (res == "active") return false;
    }

    llvm::errs() << *oldFunc << "\n";
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << *inst << "\n";
    llvm::errs() << "  unknown did status attribute\n";
    assert(0 && "bad");
    exit(1);
  }


  void forceAugmentedReturns(TypeResults &TR, const SmallPtrSetImpl<BasicBlock*>& guaranteedUnreachable) {
    assert(TR.info.function == oldFunc);
      for(BasicBlock* BB: this->originalBlocks) {
        LoopContext loopContext;
        this->getContext(BB, loopContext);

        // Don't create derivatives for code that results in termination
        if (guaranteedUnreachable.find(getOriginal(BB)) != guaranteedUnreachable.end()) continue;

        for (auto I = BB->begin(), E = BB->end(); I != E;) {
          Instruction* inst = &*I;
          assert(inst);
          I++;

          if (originalInstructions.find(inst) == originalInstructions.end()) {
              continue;
          }
          if (this->invertedPointers.find(inst) != this->invertedPointers.end()) {
              continue;
          }

          if (inst->getType()->isEmptyTy()) continue;

          if (inst->getType()->isFPOrFPVectorTy()) continue; //!op->getType()->isPointerTy() && !op->getType()->isIntegerTy()) {

          if (!TR.query(getOriginal(inst))[{}].isPossiblePointer()) continue;

          if (isa<LoadInst>(inst)) {
              IRBuilder<> BuilderZ(getNextNonDebugInstruction(inst));
              BuilderZ.setFastMathFlags(getFast());
              PHINode* anti = BuilderZ.CreatePHI(inst->getType(), 1, inst->getName() + "'il_phi");
              anti->setMetadata("enzyme_activity_value", MDNode::get(anti->getContext(), MDString::get(anti->getContext(), "active")));
              invertedPointers[inst] = anti;
              continue;
          }

          if (!isa<CallInst>(inst)) {
              continue;
          }

          if (isa<IntrinsicInst>(inst)) {
              continue;
          }

          CallInst* op = dyn_cast<CallInst>(inst);

          if (this->isConstantValue(op)) {
              continue;
          }

          Function *called = op->getCalledFunction();

          if (called && isCertainPrintOrFree(called)) {
              continue;
          }

          //if (!op->getType()->isPointerTy() && !op->getType()->isIntegerTy()) {
          //    continue;
          //}


            IRBuilder<> BuilderZ(getNextNonDebugInstruction(op));
            BuilderZ.setFastMathFlags(getFast());
            PHINode* anti = BuilderZ.CreatePHI(op->getType(), 1, op->getName() + "'ip_phi");
            anti->setMetadata("enzyme_activity_value", MDNode::get(anti->getContext(), MDString::get(anti->getContext(), "active")));
            invertedPointers[op] = anti;

			if ( called && (called->getName() == "malloc" || called->getName() == "_Znwm")) {
				invertedPointers[op]->setName(op->getName()+"'mi");
			}
        }
      }
  }

  //! if full unwrap, don't just unwrap this instruction, but also its operands, etc

  Value* unwrapM(Value* const val, IRBuilder<>& BuilderM, const ValueToValueMapTy& available, UnwrapMode mode) {//bool lookupIfAble, bool fullUnwrap=true) {
    assert(val);
    assert(val->getName() != "<badref>");

    if (isa<LoadInst>(val) && cast<LoadInst>(val)->getMetadata("enzyme_mustcache")) {
      return val;
    }

    //assert(!val->getName().startswith("$tapeload"));


    auto cidx = std::make_pair(val, BuilderM.GetInsertBlock());
    if (unwrap_cache.find(cidx) != unwrap_cache.end()) {
      if(unwrap_cache[cidx]->getType() != val->getType()) {
          llvm::errs() << "val: " << *val << "\n";
          llvm::errs() << "unwrap_cache[cidx]: " << *unwrap_cache[cidx]<< "\n";
      }
      assert(unwrap_cache[cidx]->getType() == val->getType());
      return unwrap_cache[cidx];
    }

    if (available.count(val)) {
      if(available.lookup(val)->getType() != val->getType()) {
              llvm::errs() << "val: " << *val << "\n";
              llvm::errs() << "available[val]: " << *available.lookup(val) << "\n";
      }
      assert(available.lookup(val)->getType() == val->getType());
      return available.lookup(val);
    }

    if (auto inst = dyn_cast<Instruction>(val)) {
      if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
          if (BuilderM.GetInsertBlock()->size() && BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
              if (DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
                  //llvm::errs() << "allowed " << *inst << "from domination\n";
                  assert(inst->getType() == val->getType());
                  return inst;
              }
          } else {
              if (DT.dominates(inst, BuilderM.GetInsertBlock())) {
                  //llvm::errs() << "allowed " << *inst << "from block domination\n";
                  assert(inst->getType() == val->getType());
                  return inst;
              }
          }
      }
    }

    //llvm::errs() << "uwval: " << *val << "\n";
    auto getOp = [&](Value* v) -> Value* {
      if (mode == UnwrapMode::LegalFullUnwrap || mode == UnwrapMode::AttemptFullUnwrap || mode == UnwrapMode::AttemptFullUnwrapWithLookup) {
        return unwrapM(v, BuilderM, available, mode);
      } else {
        assert(mode == UnwrapMode::AttemptSingleUnwrap);
        return lookupM(v, BuilderM, available);
      }
    };

    if (isa<Argument>(val) || isa<Constant>(val)) {
      unwrap_cache[std::make_pair(val, BuilderM.GetInsertBlock())] = val;
      return val;
    } else if (isa<AllocaInst>(val)) {
      unwrap_cache[std::make_pair(val, BuilderM.GetInsertBlock())] = val;
      return val;
    } else if (auto op = dyn_cast<CastInst>(val)) {
      auto op0 = getOp(op->getOperand(0));
      if (op0 == nullptr) goto endCheck;
      auto toreturn = BuilderM.CreateCast(op->getOpcode(), op0, op->getDestTy(), op->getName()+"_unwrap");
      unwrap_cache[cidx] = toreturn;
      assert(val->getType() == toreturn->getType());
      return toreturn;
    } else if (auto op = dyn_cast<ExtractValueInst>(val)) {
      auto op0 = getOp(op->getAggregateOperand());
      if (op0 == nullptr) goto endCheck;
      auto toreturn = BuilderM.CreateExtractValue(op0, op->getIndices(), op->getName()+"_unwrap");
      unwrap_cache[cidx] = toreturn;
      assert(val->getType() == toreturn->getType());
      return toreturn;
    } else if (auto op = dyn_cast<BinaryOperator>(val)) {
      auto op0 = getOp(op->getOperand(0));
      if (op0 == nullptr) goto endCheck;
      auto op1 = getOp(op->getOperand(1));
      if (op1 == nullptr) goto endCheck;
      auto toreturn = BuilderM.CreateBinOp(op->getOpcode(), op0, op1, op->getName()+"_unwrap");
      cast<BinaryOperator>(toreturn)->copyIRFlags(op);
      unwrap_cache[cidx] = toreturn;
      assert(val->getType() == toreturn->getType());
      return toreturn;
    } else if (auto op = dyn_cast<ICmpInst>(val)) {
      auto op0 = getOp(op->getOperand(0));
      if (op0 == nullptr) goto endCheck;
      auto op1 = getOp(op->getOperand(1));
      if (op1 == nullptr) goto endCheck;
      auto toreturn = BuilderM.CreateICmp(op->getPredicate(), op0, op1);
      unwrap_cache[cidx] = toreturn;
      assert(val->getType() == toreturn->getType());
      return toreturn;
    } else if (auto op = dyn_cast<FCmpInst>(val)) {
      auto op0 = getOp(op->getOperand(0));
      if (op0 == nullptr) goto endCheck;
      auto op1 = getOp(op->getOperand(1));
      if (op1 == nullptr) goto endCheck;
      auto toreturn = BuilderM.CreateFCmp(op->getPredicate(), op0, op1);
      unwrap_cache[cidx] = toreturn;
      assert(val->getType() == toreturn->getType());
      return toreturn;
    } else if (auto op = dyn_cast<SelectInst>(val)) {
      auto op0 = getOp(op->getOperand(0));
      if (op0 == nullptr) goto endCheck;
      auto op1 = getOp(op->getOperand(1));
      if (op1 == nullptr) goto endCheck;
      auto op2 = getOp(op->getOperand(2));
      if (op2 == nullptr) goto endCheck;
      auto toreturn = BuilderM.CreateSelect(op0, op1, op2);
      unwrap_cache[cidx] = toreturn;
      assert(val->getType() == toreturn->getType());
      return toreturn;
    } else if (auto inst = dyn_cast<GetElementPtrInst>(val)) {
      auto ptr = getOp(inst->getPointerOperand());
      if (ptr == nullptr) goto endCheck;
      SmallVector<Value*,4> ind;
      //llvm::errs() << "inst: " << *inst << "\n";
      for(auto& a : inst->indices()) {
        assert(a->getName() != "<badref>");
        auto op = getOp(a);
        if (op == nullptr) goto endCheck;
        ind.push_back(op);
      }
      auto toreturn = BuilderM.CreateGEP(ptr, ind, inst->getName() + "_unwrap");
      cast<GetElementPtrInst>(toreturn)->setIsInBounds(inst->isInBounds());
      unwrap_cache[cidx] = toreturn;
      assert(val->getType() == toreturn->getType());
      return toreturn;
    } else if (auto load = dyn_cast<LoadInst>(val)) {
      if (load->getMetadata("enzyme_noneedunwrap")) return load;

      bool legalMove = mode == UnwrapMode::LegalFullUnwrap;
      if (mode != UnwrapMode::LegalFullUnwrap) {
        //TODO actually consider whether this is legal to move to the new location, rather than recomputable anywhere
        legalMove = legalRecompute(load, available);
      }
      if (!legalMove) return nullptr;


      Value* idx = getOp(load->getOperand(0));
      if (idx == nullptr) goto endCheck;

      if(idx->getType() != load->getOperand(0)->getType()) {
          llvm::errs() << "load: " << *load << "\n";
          llvm::errs() << "load->getOperand(0): " << *load->getOperand(0) << "\n";
          llvm::errs() << "idx: " << *idx << "\n";
      }
      assert(idx->getType() == load->getOperand(0)->getType());
      auto toreturn = BuilderM.CreateLoad(idx, load->getName()+"_unwrap");
      toreturn->setAlignment(load->getAlignment());
      toreturn->setVolatile(load->isVolatile());
      toreturn->setOrdering(load->getOrdering());
      toreturn->setSyncScopeID(load->getSyncScopeID());
      toreturn->setMetadata(LLVMContext::MD_tbaa, load->getMetadata(LLVMContext::MD_tbaa));
      //toreturn->setMetadata(LLVMContext::MD_invariant, load->getMetadata(LLVMContext::MD_invariant));
      toreturn->setMetadata(LLVMContext::MD_invariant_group, load->getMetadata(LLVMContext::MD_invariant_group));
      //TODO adding to cache only legal if no alias of any future writes
      unwrap_cache[cidx] = toreturn;
      assert(val->getType() == toreturn->getType());
      return toreturn;
    } else if (auto op = dyn_cast<IntrinsicInst>(val)) {
      switch(op->getIntrinsicID()) {
          case Intrinsic::sin: {
            Value *args[] = {getOp(op->getOperand(0))};
            if (args[0] == nullptr) goto endCheck;
            Type *tys[] = {op->getOperand(0)->getType()};
            return BuilderM.CreateCall(Intrinsic::getDeclaration(op->getParent()->getParent()->getParent(), Intrinsic::sin, tys), args);
          }
          case Intrinsic::cos: {
            Value *args[] = {getOp(op->getOperand(0))};
            if (args[0] == nullptr) goto endCheck;
            Type *tys[] = {op->getOperand(0)->getType()};
            return BuilderM.CreateCall(Intrinsic::getDeclaration(op->getParent()->getParent()->getParent(), Intrinsic::cos, tys), args);
          }
          default:;

      }
    } else if (auto phi = dyn_cast<PHINode>(val)) {
      if (phi->getNumIncomingValues () == 1) {
          assert(phi->getIncomingValue(0) != phi);
          llvm::errs() << " unwrap of " << *phi << " retrieves" << phi->getIncomingValue(0) << "\n";
          auto toreturn = getOp(phi->getIncomingValue(0));
          if (toreturn == nullptr) goto endCheck;
          assert(val->getType() == toreturn->getType());
          return toreturn;
      }
    }


endCheck:
            assert(val);
            if (mode == UnwrapMode::LegalFullUnwrap || mode == UnwrapMode::AttemptFullUnwrapWithLookup) {
                assert(val->getName() != "<badref>");
                auto toreturn = lookupM(val, BuilderM);
                assert(val->getType() == toreturn->getType());
                return toreturn;
            }

            //llvm::errs() << "cannot unwrap following " << *val << "\n";

          if (auto inst = dyn_cast<Instruction>(val)) {
            //LoopContext lc;
            // if (BuilderM.GetInsertBlock() != inversionAllocs && !( (reverseBlocks.find(BuilderM.GetInsertBlock()) != reverseBlocks.end())  && /*inLoop*/getContext(inst->getParent(), lc)) ) {
            if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
                if (BuilderM.GetInsertBlock()->size() && BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
                    if (DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
                        //llvm::errs() << "allowed " << *inst << "from domination\n";
                        assert(inst->getType() == val->getType());
                        return inst;
                    }
                } else {
                    if (DT.dominates(inst, BuilderM.GetInsertBlock())) {
                        //llvm::errs() << "allowed " << *inst << "from block domination\n";
                        assert(inst->getType() == val->getType());
                        return inst;
                    }
                }
            }
          }
      return nullptr;
    }

    //! returns true indices
    std::vector<std::pair</*sublimit*/Value*, /*loop limits*/std::vector<std::pair<LoopContext, Value*>>>> getSubLimits(BasicBlock* ctx) {
        std::vector<LoopContext> contexts;
        for (BasicBlock* blk = ctx; blk != nullptr; ) {
            LoopContext idx;
            if (!getContext(blk, idx)) {
                break;
            }
            contexts.emplace_back(idx);
            blk = idx.preheader;
        }

        std::vector<BasicBlock*> allocationPreheaders(contexts.size(), nullptr);
        std::vector<Value*> limits(contexts.size(), nullptr);
        for(int i=contexts.size()-1; i >= 0; i--) {
            if ((unsigned)i == contexts.size() - 1) {
                allocationPreheaders[i] = contexts[i].preheader;
            } else if (contexts[i].dynamic) {
                allocationPreheaders[i] = contexts[i].preheader;
            } else {
                allocationPreheaders[i] = allocationPreheaders[i+1];
            }

            if (contexts[i].dynamic) {
                limits[i] = ConstantInt::get(Type::getInt64Ty(ctx->getContext()), 1);
            } else {
                ValueToValueMapTy prevMap;

                for(int j=contexts.size()-1; ; j--) {
                  if (allocationPreheaders[i] == contexts[j].preheader) break;
                  prevMap[contexts[j].var] = prevMap[contexts[j].var];
                }

                IRBuilder <> allocationBuilder(&allocationPreheaders[i]->back());
                Value* limitMinus1 = nullptr;

                //llvm::errs() << " considering limit: " << *contexts[i].limit << "\n";

                //for(auto pm : prevMap) {
                //  llvm::errs() << "    + " << pm.first << "\n";
                //}

                //TODO ensure unwrapM considers the legality of illegal caching / etc
                //   legalRecompute does not fulfill this need as its whether its legal at a certain location, where as legalRecompute
                //   specifies it being recomputable anywhere
                //if (legalRecompute(contexts[i].limit, prevMap)) {
                    limitMinus1 = unwrapM(contexts[i].limit, allocationBuilder, prevMap, UnwrapMode::AttemptFullUnwrap);
                //}

                //if (limitMinus1)
                //  llvm::errs() << " + considering limit: " << *contexts[i].limit << " - " << *limitMinus1 << "\n";
                //else
                //  llvm::errs() << " + considering limit: " << *contexts[i].limit << " - " << limitMinus1 << "\n";

                // We have a loop with static bounds, but whose limit is not available to be computed at the current loop preheader (such as the innermost loop of triangular iteration domain)
                // Handle this case like a dynamic loop
                if (limitMinus1 == nullptr) {
                    allocationPreheaders[i] = contexts[i].preheader;
                    allocationBuilder.SetInsertPoint(&allocationPreheaders[i]->back());
                    limitMinus1 = unwrapM(contexts[i].limit, allocationBuilder, prevMap, UnwrapMode::AttemptFullUnwrap);
                }
                assert(limitMinus1 != nullptr);
                static std::map<std::pair<Value*, BasicBlock*>, Value*> limitCache;
                auto cidx = std::make_pair(limitMinus1, allocationPreheaders[i]);
                if (limitCache.find(cidx) == limitCache.end()) {
                    limitCache[cidx] = allocationBuilder.CreateNUWAdd(limitMinus1, ConstantInt::get(limitMinus1->getType(), 1));
                }
                limits[i] = limitCache[cidx];
            }
        }

        std::vector<std::pair<Value*, std::vector<std::pair<LoopContext,Value*>>>> sublimits;

        Value* size = nullptr;
        std::vector<std::pair<LoopContext, Value*>> lims;
        for(unsigned i=0; i < contexts.size(); i++) {
          IRBuilder <> allocationBuilder(&allocationPreheaders[i]->back());
          lims.push_back(std::make_pair(contexts[i], limits[i]));
          if (size == nullptr) {
              size = limits[i];
          } else {
              static std::map<std::pair<Value*, BasicBlock*>, Value*> sizeCache;
              auto cidx = std::make_pair(size, allocationPreheaders[i]);
              if (sizeCache.find(cidx) == sizeCache.end()) {
                    sizeCache[cidx] = allocationBuilder.CreateNUWMul(size, limits[i]);
              }
              size = sizeCache[cidx];
          }

          // We are now starting a new allocation context
          if ( (i+1 < contexts.size()) && (allocationPreheaders[i] != allocationPreheaders[i+1]) ) {
            sublimits.push_back(std::make_pair(size, lims));
            size = nullptr;
            lims.clear();
          }
        }

        if (size != nullptr) {
          sublimits.push_back(std::make_pair(size, lims));
          lims.clear();
        }
        return sublimits;
    }

    //! Caching mechanism: creates a cache of type T in a scope given by ctx (where if ctx is in a loop there will be a corresponding number of slots)
    AllocaInst* createCacheForScope(BasicBlock* ctx, Type* T, StringRef name, bool shouldFree, bool allocateInternal=true) {
        assert(ctx);
        assert(T);

        auto sublimits = getSubLimits(ctx);

        /* goes from inner loop to outer loop*/
        std::vector<Type*> types = {T};
        for(const auto sublimit: sublimits) {
            types.push_back(PointerType::getUnqual(types.back()));
        }

        assert(inversionAllocs && "must be able to allocate inverted caches");
        IRBuilder<> entryBuilder(inversionAllocs);
        entryBuilder.setFastMathFlags(getFast());
        AllocaInst* alloc = entryBuilder.CreateAlloca(types.back(), nullptr, name+"_cache");
        {
            ConstantInt* byteSizeOfType = ConstantInt::get(Type::getInt64Ty(ctx->getContext()), newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(types.back())/8);
            unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
            if ((bsize & (bsize - 1)) == 0) {
                alloc->setAlignment(bsize);
            }
        }

        Type *BPTy = Type::getInt8PtrTy(ctx->getContext());
        auto realloc = newFunc->getParent()->getOrInsertFunction("realloc", BPTy, BPTy, Type::getInt64Ty(ctx->getContext()));

        Value* storeInto = alloc;

        for(int i=sublimits.size()-1; i>=0; i--) {
            const auto& containedloops = sublimits[i].second;

            Value* size = sublimits[i].first;
            Type* myType = types[i];

            ConstantInt* byteSizeOfType = ConstantInt::get(Type::getInt64Ty(ctx->getContext()), newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(myType)/8);

            if (allocateInternal) {

                IRBuilder <> allocationBuilder(&containedloops.back().first.preheader->back());

                StoreInst* storealloc = nullptr;
                if (!sublimits[i].second.back().first.dynamic) {
                    auto firstallocation = CallInst::CreateMalloc(
                            &allocationBuilder.GetInsertBlock()->back(),
                            size->getType(),
                            myType, byteSizeOfType, size, nullptr, name+"_malloccache");
                    CallInst* malloccall = dyn_cast<CallInst>(firstallocation);
                    if (malloccall == nullptr) {
                        malloccall = cast<CallInst>(cast<Instruction>(firstallocation)->getOperand(0));
                    }
                    malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
                    malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);

                    storealloc = allocationBuilder.CreateStore(firstallocation, storeInto);
                    //storealloc->setMetadata("enzyme_cache_static_store", MDNode::get(storealloc->getContext(), {}));

                    scopeAllocs[alloc].push_back(malloccall);

                    //allocationBuilder.GetInsertBlock()->getInstList().push_back(cast<Instruction>(allocation));
                    //cast<Instruction>(firstallocation)->moveBefore(allocationBuilder.GetInsertBlock()->getTerminator());
                    //mallocs.push_back(firstallocation);
                } else {
                    auto zerostore = allocationBuilder.CreateStore(ConstantPointerNull::get(PointerType::getUnqual(myType)), storeInto);
                    scopeStores[alloc].push_back(zerostore);

                    //auto mdpair = MDNode::getDistinct(zerostore->getContext(), {});
                    //zerostore->setMetadata("enzyme_cache_dynamiczero_store", mdpair);

                    /*
                    if (containedloops.back().first.incvar != containedloops.back().first.header->getFirstNonPHI()) {
                        llvm::errs() << "blk:" << *containedloops.back().first.header << "\n";
                        llvm::errs() << "nonphi:" << *containedloops.back().first.header->getFirstNonPHI() << "\n";
                        llvm::errs() << "incvar:" << *containedloops.back().first.incvar << "\n";
                    }
                    assert(containedloops.back().first.incvar == containedloops.back().first.header->getFirstNonPHI());
                    */
                    IRBuilder <> build(containedloops.back().first.incvar->getNextNode());
                    Value* allocation = build.CreateLoad(storeInto);
                    //Value* foo = build.CreateNUWAdd(containedloops.back().first.var, ConstantInt::get(Type::getInt64Ty(ctx->getContext()), 1));
                    Value* realloc_size = nullptr;
                    if (isa<ConstantInt>(sublimits[i].first) && cast<ConstantInt>(sublimits[i].first)->isOne()) {
                        realloc_size = containedloops.back().first.incvar;
                    } else {
                        realloc_size = build.CreateMul(containedloops.back().first.incvar, sublimits[i].first, "", /*NUW*/true, /*NSW*/true);
                    }

                    Value* idxs[2] = {
                        build.CreatePointerCast(allocation, BPTy),
                        build.CreateMul(
                            ConstantInt::get(size->getType(), newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(myType)/8), realloc_size,
                            "", /*NUW*/true, /*NSW*/true
                        )
                    };

                    Value* realloccall = nullptr;
                    allocation = build.CreatePointerCast(realloccall = build.CreateCall(realloc, idxs, name+"_realloccache"), allocation->getType(), name+"_realloccast");
                    scopeAllocs[alloc].push_back(cast<CallInst>(realloccall));
                    storealloc = build.CreateStore(allocation, storeInto);
                    //storealloc->setMetadata("enzyme_cache_dynamic_store", mdpair);
                }

                if (invariantGroups.find(std::make_pair((Value*)alloc, i)) == invariantGroups.end()) {
                    MDNode* invgroup = MDNode::getDistinct(alloc->getContext(), {});
                    invariantGroups[std::make_pair((Value*)alloc, i)] = invgroup;
                }
                storealloc->setMetadata(LLVMContext::MD_invariant_group, invariantGroups[std::make_pair((Value*)alloc, i)]);
                unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
                if ((bsize & (bsize - 1)) == 0) {
                    storealloc->setAlignment(bsize);
                }
                scopeStores[alloc].push_back(storealloc);

            }

            if (shouldFree) {
                assert(reverseBlocks.size());

                IRBuilder<> tbuild(reverseBlocks[containedloops.back().first.preheader]);
                tbuild.setFastMathFlags(getFast());

                // ensure we are before the terminator if it exists
                if (tbuild.GetInsertBlock()->size()) {
                      tbuild.SetInsertPoint(tbuild.GetInsertBlock()->getFirstNonPHI());
                }

                ValueToValueMapTy antimap;
                for(int j = sublimits.size()-1; j>=i; j--) {
                    auto& innercontainedloops = sublimits[j].second;
                    for(auto riter = innercontainedloops.rbegin(), rend = innercontainedloops.rend(); riter != rend; riter++) {
                        const auto& idx = riter->first;
                        antimap[idx.var] = tbuild.CreateLoad(idx.antivaralloc);
                    }
                }

                auto forfree = cast<LoadInst>(tbuild.CreateLoad(unwrapM(storeInto, tbuild, antimap, UnwrapMode::LegalFullUnwrap)));
                forfree->setMetadata(LLVMContext::MD_invariant_group, invariantGroups[std::make_pair((Value*)alloc, i)]);
                forfree->setMetadata(LLVMContext::MD_dereferenceable, MDNode::get(forfree->getContext(), {ConstantAsMetadata::get(byteSizeOfType)}));
                unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
                if ((bsize & (bsize - 1)) == 0) {
                    forfree->setAlignment(bsize);
                }
                //forfree->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(forfree->getContext(), {}));
                auto ci = cast<CallInst>(CallInst::CreateFree(tbuild.CreatePointerCast(forfree, Type::getInt8PtrTy(ctx->getContext())), tbuild.GetInsertBlock()));
                ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
                if (ci->getParent()==nullptr) {
                    tbuild.Insert(ci);
                }
                scopeFrees[alloc].insert(ci);
            }

            if (i != 0) {
                IRBuilder <>v(&sublimits[i-1].second.back().first.preheader->back());
                //TODO
                if (!sublimits[i].second.back().first.dynamic) {
                    storeInto = v.CreateGEP(v.CreateLoad(storeInto), sublimits[i].second.back().first.var);
                    cast<GetElementPtrInst>(storeInto)->setIsInBounds(true);
                } else {
                    storeInto = v.CreateGEP(v.CreateLoad(storeInto), sublimits[i].second.back().first.var);
                    cast<GetElementPtrInst>(storeInto)->setIsInBounds(true);
                }
            }
        }
        return alloc;
    }

    Value* getCachePointer(IRBuilder <>& BuilderM, BasicBlock* ctx, Value* cache, bool storeInStoresMap=false) {
        assert(ctx);
        assert(cache);

        auto sublimits = getSubLimits(ctx);

        ValueToValueMapTy available;

        Value* next = cache;
        assert(next->getType()->isPointerTy());
        for(int i=sublimits.size()-1; i>=0; i--) {
            next = BuilderM.CreateLoad(next);
            if (storeInStoresMap && isa<AllocaInst>(cache)) scopeStores[cast<AllocaInst>(cache)].push_back(next);

            if (!next->getType()->isPointerTy()) {
                llvm::errs() << *oldFunc << "\n";
                llvm::errs() << *newFunc << "\n";
                llvm::errs() << "cache: " << *cache << "\n";
                llvm::errs() << "next: " << *next << "\n";
            }
            assert(next->getType()->isPointerTy());
            //cast<LoadInst>(next)->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(next->getContext(), {}));
            if (invariantGroups.find(std::make_pair(cache, i)) == invariantGroups.end()) {
                MDNode* invgroup = MDNode::getDistinct(cache->getContext(), {});
                invariantGroups[std::make_pair(cache, i)] = invgroup;
            }
            cast<LoadInst>(next)->setMetadata(LLVMContext::MD_invariant_group, invariantGroups[std::make_pair(cache, i)]);
            ConstantInt* byteSizeOfType = ConstantInt::get(Type::getInt64Ty(cache->getContext()),
                            ctx->getParent()->getParent()->getDataLayout().getTypeAllocSizeInBits(next->getType())/8);
            cast<LoadInst>(next)->setMetadata(LLVMContext::MD_dereferenceable, MDNode::get(cache->getContext(), {ConstantAsMetadata::get(byteSizeOfType)}));
            unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
            if ((bsize & (bsize - 1)) == 0) {
                cast<LoadInst>(next)->setAlignment(bsize);
            }

            const auto& containedloops = sublimits[i].second;

            SmallVector<Value*,3> indices;
            SmallVector<Value*,3> limits;
            for(auto riter = containedloops.rbegin(), rend = containedloops.rend(); riter != rend; riter++) {
              // Only include dynamic index on last iteration (== skip dynamic index on non-last iterations)
              //if (i != 0 && riter+1 == rend) break;

              const auto &idx = riter->first;
              if (!isOriginalBlock(*BuilderM.GetInsertBlock())) {
                Value* av = BuilderM.CreateLoad(idx.antivaralloc);
                indices.push_back(av);
                available[idx.var] = av;
              } else {
                indices.push_back(idx.var);
                available[idx.var] = idx.var;
              }

              Value* lim = unwrapM(riter->second, BuilderM, available, UnwrapMode::AttemptFullUnwrapWithLookup);
              assert(lim);
              if (limits.size() == 0) {
                limits.push_back(lim);
              } else {
                limits.push_back(BuilderM.CreateMul(lim, limits.back(), "", /*NUW*/true, /*NSW*/true));
              }
            }

            if (indices.size() > 0) {
                Value* idx = indices[0];
                for(unsigned ind=1; ind<indices.size(); ind++) {
                  idx = BuilderM.CreateAdd(idx, BuilderM.CreateMul(indices[ind], limits[ind-1], "", /*NUW*/true, /*NSW*/true), "", /*NUW*/true, /*NSW*/true);
                }
                next = BuilderM.CreateGEP(next, {idx});
                cast<GetElementPtrInst>(next)->setIsInBounds(true);
                if (storeInStoresMap && isa<AllocaInst>(cache)) scopeStores[cast<AllocaInst>(cache)].push_back(next);
            }
            assert(next->getType()->isPointerTy());
        }
        return next;
    }

    LoadInst* lookupValueFromCache(IRBuilder<>& BuilderM, BasicBlock* ctx, Value* cache) {
        auto result = BuilderM.CreateLoad(getCachePointer(BuilderM, ctx, cache));

        if (valueInvariantGroups.find(cache) == valueInvariantGroups.end()) {
            MDNode* invgroup = MDNode::getDistinct(cache->getContext(), {});
            valueInvariantGroups[cache] = invgroup;
        }
        result->setMetadata("enzyme_fromcache", MDNode::get(result->getContext(), {}));
        result->setMetadata(LLVMContext::MD_invariant_group, valueInvariantGroups[cache]);
        ConstantInt* byteSizeOfType = ConstantInt::get(Type::getInt64Ty(cache->getContext()),
                        ctx->getParent()->getParent()->getDataLayout().getTypeAllocSizeInBits(result->getType())/8);
        //result->setMetadata(LLVMContext::MD_dereferenceable, MDNode::get(cache->getContext(), {ConstantAsMetadata::get(byteSizeOfType)}));
        unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
        if ((bsize & (bsize - 1)) == 0) {
            result->setAlignment(bsize);
        }
        if (auto inst = dyn_cast<Instruction>(cache)) {
            if (MDNode* md = inst->getMetadata("enzyme_activity_value")) {
                result->setMetadata("enzyme_activity_value", md);
            }
            result->setMetadata("enzyme_activity_inst", MDNode::get(result->getContext(), {MDString::get(result->getContext(), "const")}));
        }
        return result;
    }

    void storeInstructionInCache(BasicBlock* ctx, IRBuilder <>& BuilderM, Value* val, AllocaInst* cache) {
        IRBuilder <> v(BuilderM);
        v.setFastMathFlags(getFast());

        //Note for dynamic loops where the allocation is stored somewhere inside the loop,
        // we must ensure that we load the allocation after the store ensuring memory exists
        // to simplify things and ensure we always store after a potential realloc occurs in this loop
        // This is okay as there should be no load to the cache in the same block where this instruction is defined (since we will just use this instruction)
        for (auto I = BuilderM.GetInsertBlock()->rbegin(), E = BuilderM.GetInsertBlock()->rend(); I != E; I++) {
            if (&*I == &*BuilderM.GetInsertPoint()) break;
            if (auto si = dyn_cast<StoreInst>(&*I)) {
                v.SetInsertPoint(getNextNonDebugInstruction(si));
            }
        }
        Value* loc = getCachePointer(v, ctx, cache, /*storeinstorecache*/true);
        assert(cast<PointerType>(loc->getType())->getElementType() == val->getType());
        StoreInst* storeinst = v.CreateStore(val, loc);
        if (valueInvariantGroups.find(cache) == valueInvariantGroups.end()) {
            MDNode* invgroup = MDNode::getDistinct(cache->getContext(), {});
            valueInvariantGroups[cache] = invgroup;
        }
        storeinst->setMetadata(LLVMContext::MD_invariant_group, valueInvariantGroups[cache]);
        ConstantInt* byteSizeOfType = ConstantInt::get(Type::getInt64Ty(cache->getContext()),
                        ctx->getParent()->getParent()->getDataLayout().getTypeAllocSizeInBits(val->getType())/8);
        unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
        if ((bsize & (bsize - 1)) == 0) {
            storeinst->setAlignment(bsize);
        }
        scopeStores[cache].push_back(storeinst);
    }

    void storeInstructionInCache(BasicBlock* ctx, Instruction* inst, AllocaInst* cache) {
        assert(ctx);
        assert(inst);
        assert(cache);

        IRBuilder <> v(inst->getParent());

        if (&*inst->getParent()->rbegin() != inst) {
            auto pn = dyn_cast<PHINode>(inst);
            Instruction* putafter = ( pn && pn->getNumIncomingValues()>0 )? (inst->getParent()->getFirstNonPHI() ): getNextNonDebugInstruction(inst);
            assert(putafter);
            v.SetInsertPoint(putafter);
        }
        v.setFastMathFlags(getFast());
        storeInstructionInCache(ctx, v, inst, cache);
    }

    void ensureLookupCached(Instruction* inst, bool shouldFree=true) {
        assert(inst);
        if (scopeMap.find(inst) != scopeMap.end()) return;
        AllocaInst* cache = createCacheForScope(inst->getParent(), inst->getType(), inst->getName(), shouldFree);
        assert(cache);
        scopeMap[inst] = cache;
        storeInstructionInCache(inst->getParent(), inst, cache);
    }

    Instruction* fixLCSSA(Instruction* inst, const IRBuilder <>& BuilderM) {
        assert(inst->getName() != "<badref>");
        LoopContext lc;
        bool inLoop = getContext(inst->getParent(), lc);
        if (inLoop) {
            bool isChildLoop = false;

            BasicBlock* forwardBlock = BuilderM.GetInsertBlock();

            if (!isOriginalBlock(*forwardBlock)) {
                forwardBlock = originalForReverseBlock(*forwardBlock);
            }

            auto builderLoop = LI.getLoopFor(forwardBlock);
            while (builderLoop) {
              if (builderLoop->getHeader() == lc.header) {
                isChildLoop = true;
                break;
              }
              builderLoop = builderLoop->getParentLoop();
            }

            if (!isChildLoop) {
                llvm::errs() << "manually performing lcssa for instruction" << *inst << " in block " << BuilderM.GetInsertBlock()->getName() << "\n";
                if (!DT.dominates(inst, forwardBlock)) {
                    llvm::errs() << *this->newFunc << "\n";
                    llvm::errs() << *forwardBlock << "\n";
                    llvm::errs() << *BuilderM.GetInsertBlock() << "\n";
                    llvm::errs() << *inst << "\n";
                }
                assert(DT.dominates(inst, forwardBlock));

                IRBuilder<> lcssa(&forwardBlock->front());
                auto lcssaPHI = lcssa.CreatePHI(inst->getType(), 1, inst->getName()+"!manual_lcssa");
                for(auto pred : predecessors(forwardBlock))
                    lcssaPHI->addIncoming(inst, pred);
                return lcssaPHI;

            }
        }
        return inst;
    }

    Value* lookupM(Value* val, IRBuilder<>& BuilderM, const ValueToValueMapTy &incoming_availalble=ValueToValueMapTy());

    Value* invertPointerM(Value* val, IRBuilder<>& BuilderM);

    void branchToCorrespondingTarget(BasicBlock* ctx, IRBuilder <>& BuilderM, const std::map<BasicBlock*, std::vector<std::pair</*pred*/BasicBlock*,/*successor*/BasicBlock*>>> &targetToPreds, const std::map<BasicBlock*,PHINode*>* replacePHIs = nullptr);

};

class DiffeGradientUtils : public GradientUtils {
  DiffeGradientUtils(Function* newFunc_, Function* oldFunc_, TargetLibraryInfo &TLI, TypeAnalysis &TA, AAResults &AA, ValueToValueMapTy& invertedPointers_, const SmallPtrSetImpl<Value*> &constants_, const SmallPtrSetImpl<Value*> &nonconstant_, const SmallPtrSetImpl<Value*> &constantvalues_, const SmallPtrSetImpl<Value*> &returnvals_, ValueToValueMapTy &origToNew_)
      : GradientUtils(newFunc_, oldFunc_, TLI, TA, AA, invertedPointers_, constants_, nonconstant_, constantvalues_, returnvals_, origToNew_) {
        prepareForReverse();
    }

public:
  ValueToValueMapTy differentials;
  static DiffeGradientUtils* CreateFromClone(bool topLevel, Function *todiff, TargetLibraryInfo &TLI, TypeAnalysis &TA, AAResults &AA, const std::set<unsigned> & constant_args, ReturnType returnValue, bool differentialReturn, Type* additionalArg);

private:
  Value* getDifferential(Value *val) {
    assert(val);
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);
    assert(inversionAllocs);
    if (differentials.find(val) == differentials.end()) {
        IRBuilder<> entryBuilder(inversionAllocs);
        entryBuilder.setFastMathFlags(getFast());
        differentials[val] = entryBuilder.CreateAlloca(val->getType(), nullptr, val->getName()+"'de");
        entryBuilder.CreateStore(Constant::getNullValue(val->getType()), differentials[val]);
    }
    assert(cast<PointerType>(differentials[val]->getType())->getElementType() == val->getType());
    return differentials[val];
  }

public:
  Value* diffe(Value* val, IRBuilder<> &BuilderM) {
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);

    if (isConstantValue(getNewFromOriginal(val))) {
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << *val << "\n";
    }
    if (val->getType()->isPointerTy()) {
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << *val << "\n";
    }
    assert(!val->getType()->isPointerTy());
    assert(!val->getType()->isVoidTy());
    return BuilderM.CreateLoad(getDifferential(val));
  }

  //Returns created select instructions, if any
  std::vector<SelectInst*> addToDiffe(Value* val, Value* dif, IRBuilder<> &BuilderM, Type* addingType) {
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);

    std::vector<SelectInst*> addedSelects;


      auto faddForSelect = [&](Value* old, Value* dif) -> Value* {

        //! optimize fadd of select to select of fadd
        if (SelectInst* select = dyn_cast<SelectInst>(dif)) {
            if (Constant* ci = dyn_cast<Constant>(select->getTrueValue())) {
                if (ci->isZeroValue()) {
                    SelectInst* res = cast<SelectInst>(BuilderM.CreateSelect(select->getCondition(), old, BuilderM.CreateFAdd(old, select->getFalseValue())));
                    addedSelects.emplace_back(res);
                    return res;
                }
            }
            if (Constant* ci = dyn_cast<Constant>(select->getFalseValue())) {
                if (ci->isZeroValue()) {
                    SelectInst* res = cast<SelectInst>(BuilderM.CreateSelect(select->getCondition(), BuilderM.CreateFAdd(old, select->getTrueValue()), old));
                    addedSelects.emplace_back(res);
                    return res;
                }
            }
        }

        //! optimize fadd of bitcast select to select of bitcast fadd
        if (BitCastInst* bc = dyn_cast<BitCastInst>(dif)) {
            if (SelectInst* select = dyn_cast<SelectInst>(bc->getOperand(0))) {
                if (Constant* ci = dyn_cast<Constant>(select->getTrueValue())) {
                    if (ci->isZeroValue()) {
                        SelectInst* res = cast<SelectInst>(BuilderM.CreateSelect(select->getCondition(), old, BuilderM.CreateFAdd(old, BuilderM.CreateCast(bc->getOpcode(), select->getFalseValue(), bc->getDestTy()))));
                        addedSelects.emplace_back(res);
                        return res;
                    }
                }
                if (Constant* ci = dyn_cast<Constant>(select->getFalseValue())) {
                    if (ci->isZeroValue()) {
                        SelectInst* res = cast<SelectInst>(BuilderM.CreateSelect(select->getCondition(), BuilderM.CreateFAdd(old, BuilderM.CreateCast(bc->getOpcode(), select->getTrueValue(), bc->getDestTy())), old));
                        addedSelects.emplace_back(res);
                        return res;
                    }
                }
            }
        }

        // fallback
        return BuilderM.CreateFAdd(old, dif);
      };

      if (val->getType()->isPointerTy()) {
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << *val << "\n";
      }
      if (isConstantValue(getNewFromOriginal(val))) {
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << *val << "\n";
      }
      assert(!val->getType()->isPointerTy());
      assert(!isConstantValue(getNewFromOriginal(val)));
      assert(val->getType() == dif->getType());
      auto old = diffe(val, BuilderM);
      assert(val->getType() == old->getType());
      Value* res = nullptr;
      if (val->getType()->isIntOrIntVectorTy()) {
        assert(addingType);
        assert(addingType->isFPOrFPVectorTy());

        auto oldBitSize = oldFunc->getParent()->getDataLayout().getTypeSizeInBits(old->getType());
        auto newBitSize = oldFunc->getParent()->getDataLayout().getTypeSizeInBits(addingType);

        if ( oldBitSize > newBitSize && oldBitSize % newBitSize == 0 && !addingType->isVectorTy()) {
          addingType = VectorType::get(addingType, oldBitSize / newBitSize);
        }

        Value* bcold = BuilderM.CreateBitCast(old, addingType);
        Value* bcdif = BuilderM.CreateBitCast(dif, addingType);

        res = faddForSelect(bcold, bcdif);
        if (Instruction* oldinst = dyn_cast<Instruction>(bcold)) {
            if (oldinst->getNumUses() == 0) {
                //if (oldinst == &*BuilderM.GetInsertPoint()) BuilderM.SetInsertPoint(oldinst->getNextNode());
                //oldinst->eraseFromParent();
            }
        }
        if (Instruction* difinst = dyn_cast<Instruction>(bcdif)) {
            if (difinst->getNumUses() == 0) {
                //if (difinst == &*BuilderM.GetInsertPoint()) BuilderM.SetInsertPoint(difinst->getNextNode());
                //difinst->eraseFromParent();
            }
        }
        if (SelectInst* select = dyn_cast<SelectInst>(res)) {
            assert(addedSelects.back() == select);
            addedSelects.erase(addedSelects.end()-1);
            res = BuilderM.CreateSelect(select->getCondition(), BuilderM.CreateBitCast(select->getTrueValue(), val->getType()), BuilderM.CreateBitCast(select->getFalseValue(), val->getType()));
            assert(select->getNumUses() == 0);
            //if (select == &*BuilderM.GetInsertPoint()) BuilderM.SetInsertPoint(select->getNextNode());
            //select->eraseFromParent();
        } else {
            res = BuilderM.CreateBitCast(res, val->getType());
        }
        BuilderM.CreateStore(res, getDifferential(val));
        //store->setAlignment(align);
        return addedSelects;
      } else if (val->getType()->isFPOrFPVectorTy()) {
        //TODO consider adding type
        res = faddForSelect(old, dif);

        BuilderM.CreateStore(res, getDifferential(val));
        //store->setAlignment(align);
        return addedSelects;
      } else if (val->getType()->isStructTy()) {
        auto st = cast<StructType>(val->getType());
        for(unsigned i=0; i<st->getNumElements(); i++) {
            Value* v = ConstantInt::get(Type::getInt32Ty(st->getContext()), i);
            SelectInst* addedSelect = addToDiffeIndexed(val, BuilderM.CreateExtractValue(dif,{i}), {v}, BuilderM);
            if (addedSelect) {
                addedSelects.push_back(addedSelect);
            }
        }
        return addedSelects;
      } else {
        llvm_unreachable("unknown type to add to diffe");
        exit(1);
      }
  }

  void setDiffe(Value* val, Value* toset, IRBuilder<> &BuilderM) {
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);
    if (isConstantValue(getNewFromOriginal(val))) {
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << *val << "\n";
      }
      assert(!isConstantValue(getNewFromOriginal(val)));
      Value* tostore = getDifferential(val);
      if (toset->getType() != cast<PointerType>(tostore->getType())->getElementType()) {
        llvm::errs() << "toset:" << *toset << "\n";
        llvm::errs() << "tostore:" << *tostore << "\n";
      }
      assert(toset->getType() == cast<PointerType>(tostore->getType())->getElementType());
      BuilderM.CreateStore(toset, tostore);
  }

  SelectInst* addToDiffeIndexed(Value* val, Value* dif, ArrayRef<Value*> idxs, IRBuilder<> &BuilderM) {
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);
      assert(!isConstantValue(getNewFromOriginal(val)));
      SmallVector<Value*,4> sv;
      sv.push_back(ConstantInt::get(Type::getInt32Ty(val->getContext()), 0));
      for(auto i : idxs)
        sv.push_back(i);
      Value* ptr = BuilderM.CreateGEP(getDifferential(val), sv);
      cast<GetElementPtrInst>(ptr)->setIsInBounds(true);
      Value* old = BuilderM.CreateLoad(ptr);

      Value* res = nullptr;

      if (old->getType()->isIntOrIntVectorTy()) {
        res = BuilderM.CreateFAdd(BuilderM.CreateBitCast(old, IntToFloatTy(old->getType())), BuilderM.CreateBitCast(dif, IntToFloatTy(dif->getType())));
        res = BuilderM.CreateBitCast(res, old->getType());
      } else if(old->getType()->isFPOrFPVectorTy()) {
        res = BuilderM.CreateFAdd(old, dif);
      } else {
        assert(old);
        assert(dif);
        llvm::errs() << *newFunc << "\n" << "cannot handle type " << *old << "\n" << *dif;
        assert(0 && "cannot handle type");
        report_fatal_error("cannot handle type");
      }


      SelectInst* addedSelect = nullptr;

        //! optimize fadd of select to select of fadd
        // TODO: Handle Selects of ints
        if (SelectInst* select = dyn_cast<SelectInst>(dif)) {
            if (ConstantFP* ci = dyn_cast<ConstantFP>(select->getTrueValue())) {
                if (ci->isZero()) {
                    cast<Instruction>(res)->eraseFromParent();
                    res = BuilderM.CreateSelect(select->getCondition(), old, BuilderM.CreateFAdd(old, select->getFalseValue()));
                    addedSelect = cast<SelectInst>(res);
                    goto endselect;
                }
            }
            if (ConstantFP* ci = dyn_cast<ConstantFP>(select->getFalseValue())) {
                if (ci->isZero()) {
                    cast<Instruction>(res)->eraseFromParent();
                    res = BuilderM.CreateSelect(select->getCondition(), BuilderM.CreateFAdd(old, select->getTrueValue()), old);
                    addedSelect = cast<SelectInst>(res);
                    goto endselect;
                }
            }
        }
        endselect:;

      BuilderM.CreateStore(res, ptr);
      return addedSelect;
  }

  //! align is the alignment that should be specified for load/store to pointer
  void addToInvertedPtrDiffe(Value* ptr, Value* dif, IRBuilder<> &BuilderM, unsigned align) {
      if (!(ptr->getType()->isPointerTy()) || !(cast<PointerType>(ptr->getType())->getElementType() == dif->getType())) {
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << "Ptr: " << *ptr << "\n";
        llvm::errs() << "Diff: " << *dif << "\n";
      }
      assert(ptr->getType()->isPointerTy());
      assert(cast<PointerType>(ptr->getType())->getElementType() == dif->getType());

      assert(ptr->getType()->isPointerTy());
      assert(cast<PointerType>(ptr->getType())->getElementType() == dif->getType());

      Value* res;
      LoadInst* old = BuilderM.CreateLoad(ptr);
      old->setAlignment(align);

      if (old->getType()->isIntOrIntVectorTy()) {
        res = BuilderM.CreateFAdd(BuilderM.CreateBitCast(old, IntToFloatTy(old->getType())), BuilderM.CreateBitCast(dif, IntToFloatTy(dif->getType())));
        res = BuilderM.CreateBitCast(res, old->getType());
      } else if(old->getType()->isFPOrFPVectorTy()) {
        res = BuilderM.CreateFAdd(old, dif);
      } else {
        assert(old);
        assert(dif);
        llvm::errs() << *newFunc << "\n" << "cannot handle type " << *old << "\n" << *dif;
        assert(0 && "cannot handle type");
        report_fatal_error("cannot handle type");
      }
      StoreInst* st = BuilderM.CreateStore(res, ptr);
      st->setAlignment(align);
  }

};
#endif
