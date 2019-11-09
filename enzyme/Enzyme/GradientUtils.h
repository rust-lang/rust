
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

#include <deque>

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

using namespace llvm;

typedef struct {
  PHINode* var;
  PHINode* antivar;
  BasicBlock* latchMerge;
  BasicBlock* header;
  BasicBlock* preheader;
  bool dynamic;
  //limit is last value, iters is number of iters (thus iters = limit + 1)
  Value* limit;
  SmallPtrSet<BasicBlock*, 8> exitBlocks;
  Loop* parent;
} LoopContext;

static inline bool operator==(const LoopContext& lhs, const LoopContext &rhs) {
    return lhs.parent == rhs.parent;
}

class GradientUtils {
public:
  llvm::Function *oldFunc;
  llvm::Function *newFunc;
  ValueToValueMapTy invertedPointers;
  DominatorTree DT;
  SmallPtrSet<Value*,4> constants;
  SmallPtrSet<Value*,20> nonconstant;
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
  ValueToValueMapTy lastScopeAlloc;
  ValueToValueMapTy scopeFrees;
  ValueToValueMapTy originalToNewFn;

  std::map<Instruction*, bool>* can_modref_map;  


  Value* getNewFromOriginal(Value* originst) {
    assert(originst);
    auto f = originalToNewFn.find(originst);
    if (f == originalToNewFn.end()) {
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
  Value* getOriginal(Value* newinst) {
    for(auto v: originalToNewFn) {
        if (v.second == newinst) return const_cast<Value*>(v.first);
    }
    llvm::errs() << *newinst << "\n";
    assert(0 && "could not invert new inst");
    report_fatal_error("could not invert new inst");
  }

  Value* getOriginalPointer(Value* newinst) {
    for(auto v: originalToNewFn) {
        if (invertedPointers[v.second] == newinst) return const_cast<Value*>(v.first);
    }
    assert(0 && "could not invert new pointer inst");
    report_fatal_error("could not invert new pointer inst");
  }

private:
  SmallVector<Value*, 4> addedMallocs;
  unsigned tapeidx;
  Value* tape;
public:
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
    if (scopeFrees.find(A) != scopeFrees.end()) {
        scopeFrees[B] = scopeFrees[A];
        scopeFrees.erase(A);
    }
    if (lastScopeAlloc.find(A) != lastScopeAlloc.end()) {
        lastScopeAlloc[B] = lastScopeAlloc[A];
        lastScopeAlloc.erase(A);
    }

    A->replaceAllUsesWith(B);
  }

  void erase(Instruction *I) {
    assert(I);
    invertedPointers.erase(I);
    constants.erase(I);
    nonconstant.erase(I);
    nonconstant_values.erase(I);
    originalInstructions.erase(I);
    scopeMap.erase(I);
    lastScopeAlloc.erase(I);
    scopeFrees.erase(I);
    SE.eraseValueFromMap(I);
    originalToNewFn.erase(I);
    eraser:
    for(auto v: originalToNewFn) {
        if (v.second == I) {
            originalToNewFn.erase(v.first);
            goto eraser;
        }
    }
    for(auto v: lastScopeAlloc) {
        if (v.second == I) {
            llvm::errs() << *v.first << "\n";
            llvm::errs() << *I << "\n";
            assert(0 && "erasing something in lastScopeAlloc map");
        }
    }
    for(auto v: scopeMap) {
        if (v.second == I) {
            llvm::errs() << *newFunc << "\n";
            dumpScope();
            llvm::errs() << *v.first << "\n";
            llvm::errs() << *I << "\n";
            assert(0 && "erasing something in scope map");
        }
    }
    for(auto v: scopeFrees) {
        if (v.second == I) {
            llvm::errs() << *v.first << "\n";
            llvm::errs() << *I << "\n";
            assert(0 && "erasing something in scopeFrees map");
        }
    }
    for(auto v: invertedPointers) {
        if (v.second == I) {
            llvm::errs() << *newFunc << "\n";
            dumpPointers();
            llvm::errs() << *v.first << "\n";
            llvm::errs() << *I << "\n";
            assert(0 && "erasing something in invertedPointers map");
        }
    }
    if (!I->use_empty()) {
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << *I << "\n";
    }
    assert(I->use_empty());
    I->eraseFromParent();
  }

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

  Instruction* createAntiMalloc(CallInst *call) {
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

    invertedPointers[call] = anti;
    assert(placeholder != anti);
    bb.SetInsertPoint(placeholder->getNextNode());
    replaceAWithB(placeholder, anti);
    erase(placeholder);

    anti = cast<Instruction>(addMalloc(bb, anti));
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

  Value* addMalloc(IRBuilder<> &BuilderQ, Value* malloc) {
    if (tape) {
        if (!tape->getType()->isStructTy()) {
            llvm::errs() << "addMalloc incorrect tape type: " << *tape << "\n";
        }
        assert(tape->getType()->isStructTy());
        if (tapeidx >= cast<StructType>(tape->getType())->getNumElements()) {
            llvm::errs() << "oldFunc: " <<*oldFunc << "\n";
            llvm::errs() << "newFunc: " <<*newFunc << "\n";
            if (malloc)
            llvm::errs() << "malloc: " <<*malloc << "\n";
            llvm::errs() << "tape: " <<*tape << "\n";
            llvm::errs() << "tapeidx: " << tapeidx << "\n";
        }
        assert(tapeidx < cast<StructType>(tape->getType())->getNumElements());
        Instruction* ret = cast<Instruction>(BuilderQ.CreateExtractValue(tape, {tapeidx}));
        Instruction* origret = ret;
        tapeidx++;

        if (ret->getType()->isEmptyTy()) {
            
            if (auto inst = dyn_cast_or_null<Instruction>(malloc)) {
                inst->replaceAllUsesWith(UndefValue::get(ret->getType()));
                erase(inst);
            }
            
            //return ret;
            return UndefValue::get(ret->getType());
        }

        BasicBlock* parent = BuilderQ.GetInsertBlock();
	  	if (Instruction* inst = dyn_cast_or_null<Instruction>(malloc)) {
			parent = inst->getParent();
		}

		LoopContext lc;
      	bool inLoop = getContext(parent, lc);

        if (!inLoop) {
        } else {
            erase(ret);
            IRBuilder<> entryBuilder(inversionAllocs);
            entryBuilder.setFastMathFlags(getFast());
            ret = cast<Instruction>(entryBuilder.CreateExtractValue(tape, {tapeidx-1}));

            if (malloc) assert(cast<PointerType>(ret->getType())->getElementType() == malloc->getType());

            AllocaInst* cache = entryBuilder.CreateAlloca(ret->getType(), nullptr, "mdyncache_fromtape");
            entryBuilder.CreateStore(ret, cache);

            auto v = lookupValueFromCache(BuilderQ, BuilderQ.GetInsertBlock(), cache);
            if (malloc) {
                assert(v->getType() == malloc->getType());
            }
            scopeMap[v] = cache;
            originalInstructions.erase(ret);

            assert(reverseBlocks.size() > 0);

            BasicBlock* outermostPreheader = nullptr;

            for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx) ) {
                if (idx.parent == nullptr) {
                    outermostPreheader = idx.preheader;
                }
                if (idx.parent == nullptr) break;
            }
            assert(outermostPreheader);
                IRBuilder<> tbuild(reverseBlocks[outermostPreheader]);
                tbuild.setFastMathFlags(getFast());

                // ensure we are before the terminator if it exists
                if (tbuild.GetInsertBlock()->size()) {
                      tbuild.SetInsertPoint(tbuild.GetInsertBlock()->getFirstNonPHI());
                }

                CallInst* ci = cast<CallInst>(CallInst::CreateFree(tbuild.CreatePointerCast(tbuild.CreateLoad(scopeMap[v]), Type::getInt8PtrTy(outermostPreheader->getContext())), tbuild.GetInsertBlock()));
                ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
                if (ci->getParent()==nullptr) {
                    tbuild.Insert(ci);
                }

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

                if (!inLoop) {
                    std::vector<User*> users;
                    for (auto u : scopeMap[malloc]->users()) {
                        users.push_back(u);
                    }
                    for( auto u : users) {
                        if (auto li = dyn_cast<LoadInst>(u)) {
                            li->replaceAllUsesWith(ret);
                            erase(li);
                        } else if (auto si = dyn_cast<StoreInst>(u)) {
                            erase(si);
                        } else {
                            assert(0 && "illegal use for out of loop scopeMap");
                        }
                    }

                    {
                    Instruction* preerase = cast<Instruction>(scopeMap[malloc]);
                    scopeMap.erase(malloc);
                    erase(preerase);
                    }
                } else {
                    std::vector<User*> users;
                    for( auto u : scopeMap[malloc]->users()) {
                        users.push_back(u);
                    }
                    Instruction* op0 = nullptr;
                    if (auto ci = dyn_cast<CastInst>(scopeMap[malloc])) {
                        op0 = cast<Instruction>(ci->getOperand(0));
                        for( auto u : op0->users()) {
                            if (u != malloc)
                                users.push_back(u);
                        }
                    }

                    for( auto u : users) {
                        if (auto li = dyn_cast<LoadInst>(u)) {
                            for( auto u0 : li->users()) {
                                Instruction* u2 = dyn_cast<Instruction>(u0);
                                if (u2 == nullptr) continue;
                                if (auto ci = dyn_cast<CastInst>(u2)) {
                                    if (ci->hasOneUse())
                                        u2 = cast<Instruction>(*ci->user_begin());
                                }
                                llvm::errs() << " found use in " << *u2 << "\n";
                                if (auto cali = dyn_cast<CallInst>(u2)) {
                                    auto called = cali->getCalledFunction();
                                    if (called == nullptr) continue;
                                    if (!(called->getName() == "free" || called->getName() == "realloc")) continue;
                                    if (scopeFrees.find(malloc) != scopeFrees.end() && scopeFrees[malloc] == cali)
                                        scopeFrees.erase(malloc);
                                    if (lastScopeAlloc.find(malloc) != lastScopeAlloc.end() && lastScopeAlloc[malloc] == cali)
                                        lastScopeAlloc.erase(malloc);
                                    erase(cali);
                                }
                                if (u0->getNumUses() == 0 && u2 != u0) erase(cast<Instruction>(u0));
                            }

                            li->setOperand(0, scopeMap[ret]);
                            if (li->getNumUses() == 0) erase(li);
                        } else if (auto si = dyn_cast<StoreInst>(u)) {
                            Instruction* u2 = cast<Instruction>(si->getValueOperand());
                            erase(si);

                            u2->replaceAllUsesWith(origret);

                            if (auto ci = dyn_cast<CastInst>(u2)) {
                                u2 = cast<Instruction>(ci->getOperand(0));
                                if (lastScopeAlloc.find(malloc) != lastScopeAlloc.end() && (lastScopeAlloc[malloc] == ci || lastScopeAlloc[malloc] == origret))
                                    lastScopeAlloc.erase(malloc);
                                erase(ci);
                            }

                            auto cali = cast<CallInst>(u2);
                            auto called = cali->getCalledFunction();
                            assert(called);
                            assert(called->getName() == "malloc" || called->getName() == "realloc");

                            if (lastScopeAlloc.find(malloc) != lastScopeAlloc.end() && (lastScopeAlloc[malloc] == cali))
                                lastScopeAlloc.erase(malloc);
                            erase(cali);
                            continue;
                        } else {
                            assert(0 && "illegal use for scopeMap");
                        }
                        //TODO consider realloc/free
                    }

                    {
                    Instruction* preerase = cast<Instruction>(scopeMap[malloc]);
                    scopeMap.erase(malloc);
                    erase(preerase);
                    }

                    if (op0) {
                        if (lastScopeAlloc.find(malloc) != lastScopeAlloc.end() && lastScopeAlloc[malloc] == op0)
                            lastScopeAlloc.erase(malloc);
                        erase(op0);
                    }
                }
            }
            if (scopeFrees.find(malloc) != scopeFrees.end()) {
                llvm::errs() << *newFunc << "\n";
                if (scopeFrees[malloc])
                    llvm::errs() << "scopeFrees[malloc] = " << *scopeFrees[malloc] << "\n";
                else 
                    llvm::errs() << "scopeFrees[malloc] = (nullptr)" << "\n";
            }
            assert(scopeFrees.find(malloc) == scopeFrees.end());
            if (lastScopeAlloc.find(malloc) != lastScopeAlloc.end()) {
                llvm::errs() << *newFunc << "\n";
                if (lastScopeAlloc[malloc])
                    llvm::errs() << "lastScopeAlloc[malloc] = " << *lastScopeAlloc[malloc] << "\n";
                else 
                    llvm::errs() << "lastScopeAlloc[malloc] = (nullptr)" << "\n";
            }
            assert(lastScopeAlloc.find(malloc) == lastScopeAlloc.end());
            cast<Instruction>(malloc)->replaceAllUsesWith(ret);
            std::string n = malloc->getName().str();
            erase(cast<Instruction>(malloc));
            ret->setName(n);
        }
        return ret;
    } else {
      assert(malloc);
      assert(!isa<PHINode>(malloc));

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
      assert(lastScopeAlloc[malloc]);
      addedMallocs.push_back(lastScopeAlloc[malloc]);
      return malloc;
    }
    llvm::errs() << "Fell through on addMalloc. This should never happen.\n";
    assert(false); 
  }

  const SmallVectorImpl<Value*> & getMallocs() const {
    return addedMallocs;
  }
protected:
  AAResults &AA;
  TargetLibraryInfo &TLI;
  GradientUtils(Function* newFunc_, AAResults &AA_, TargetLibraryInfo &TLI_, ValueToValueMapTy& invertedPointers_, const SmallPtrSetImpl<Value*> &constants_, const SmallPtrSetImpl<Value*> &nonconstant_, const SmallPtrSetImpl<Value*> &returnvals_, ValueToValueMapTy& originalToNewFn_) :
      newFunc(newFunc_), invertedPointers(), DT(*newFunc_), constants(constants_.begin(), constants_.end()), nonconstant(nonconstant_.begin(), nonconstant_.end()), nonconstant_values(returnvals_.begin(), returnvals_.end()), LI(DT), AC(*newFunc_), SE(*newFunc_, TLI_, AC, DT, LI), inversionAllocs(nullptr), AA(AA_), TLI(TLI_) {
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
  static GradientUtils* CreateFromClone(Function *todiff, AAResults &AA, TargetLibraryInfo &TLI, const std::set<unsigned> & constant_args, ReturnType returnValue, bool differentialReturn, llvm::Type* additionalArg=nullptr);

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

  BasicBlock* getReverseOrLatchMerge(BasicBlock* BB, BasicBlock* branchingBlock);

  void forceContexts(bool setupMerge=false);

  bool getContext(BasicBlock* BB, LoopContext& loopContext);

  bool isOriginalBlock(const BasicBlock &BB) const {
    for(auto A : originalBlocks) {
        if (A == &BB) return true;
    }
    return false;
  }

  bool isConstantValue(Value* val) {
	cast<Value>(val);
    return isconstantValueM(val, constants, nonconstant, nonconstant_values, originalInstructions);
  };

  bool isConstantInstruction(Instruction* val) {
	cast<Instruction>(val);
    return isconstantM(val, constants, nonconstant, nonconstant_values, originalInstructions);
  }

  SmallPtrSet<Instruction*,4> replaceableCalls;
  void eraseStructuralStoresAndCalls() {

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

          if (!(isa<BranchInst>(inst) || isa<ReturnInst>(inst)) && this->isConstantInstruction(inst)) {
            if (inst->getNumUses() == 0) {
                erase(inst);
			    continue;
            }
          } else {
            if (auto inti = dyn_cast<IntrinsicInst>(inst)) {
                if (inti->getIntrinsicID() == Intrinsic::memset || inti->getIntrinsicID() == Intrinsic::memcpy) {
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

  void forceAugmentedReturns() {
      for(BasicBlock* BB: this->originalBlocks) {
        LoopContext loopContext;
        this->getContext(BB, loopContext);

        auto term = BB->getTerminator();
        if (isa<UnreachableInst>(term)) continue;

        for (auto I = BB->begin(), E = BB->end(); I != E;) {
          Instruction* inst = &*I;
          assert(inst);
          I++;

          if (!isa<CallInst>(inst)) {
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

          if (!op->getType()->isPointerTy() && !op->getType()->isIntegerTy()) {
              continue;
          }

          if (this->invertedPointers.find(op) != this->invertedPointers.end()) {
              continue;
          }

            IRBuilder<> BuilderZ(getNextNonDebugInstruction(op));
            BuilderZ.setFastMathFlags(getFast());
            this->invertedPointers[op] = BuilderZ.CreatePHI(op->getType(), 1, op->getName() + "'ip_phi");

			if ( called && (called->getName() == "malloc" || called->getName() == "_Znwm")) {
				this->invertedPointers[op]->setName(op->getName()+"'mi");
			}
        }
      }
  }

  Value* unwrapM(Value* val, IRBuilder<>& BuilderM, const ValueToValueMapTy& available, bool lookupIfAble) {
      assert(val);

      static std::map<std::pair<Value*, BasicBlock*>, Value*> cache;
      auto cidx = std::make_pair(val, BuilderM.GetInsertBlock());
      if (cache.find(cidx) != cache.end()) {
        return cache[cidx];
      }

          if (available.count(val)) {
            return available.lookup(val);
          }
          
          if (auto inst = dyn_cast<Instruction>(val)) {
            if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
                if (BuilderM.GetInsertBlock()->size() && BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
                    if (DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
                        //llvm::errs() << "allowed " << *inst << "from domination\n";
                        return inst;
                    }
                } else {
                    if (DT.dominates(inst, BuilderM.GetInsertBlock())) {
                        //llvm::errs() << "allowed " << *inst << "from block domination\n";
                        return inst;
                    }
                }
            }
          }

          if (isa<Argument>(val) || isa<Constant>(val)) {
            cache[std::make_pair(val, BuilderM.GetInsertBlock())] = val;
            return val;
          } else if (isa<AllocaInst>(val)) {
            cache[std::make_pair(val, BuilderM.GetInsertBlock())] = val;
            return val;
          } else if (auto op = dyn_cast<CastInst>(val)) {
            auto op0 = unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble);
            if (op0 == nullptr) goto endCheck;
            auto toreturn = BuilderM.CreateCast(op->getOpcode(), op0, op->getDestTy(), op->getName()+"_unwrap");
            if (cache.find(std::make_pair((Value*)op->getOperand(0), BuilderM.GetInsertBlock())) != cache.end()) {
                cache[cidx] = toreturn;
            }
            return toreturn;
          } else if (auto op = dyn_cast<ExtractValueInst>(val)) {
            auto op0 = unwrapM(op->getAggregateOperand(), BuilderM, available, lookupIfAble);
            if (op0 == nullptr) goto endCheck;
            auto toreturn = BuilderM.CreateExtractValue(op0, op->getIndices(), op->getName()+"_unwrap");
            if (cache.find(std::make_pair((Value*)op->getOperand(0), BuilderM.GetInsertBlock())) != cache.end()) {
                cache[cidx] = toreturn;
            }
            return toreturn;
          } else if (auto op = dyn_cast<BinaryOperator>(val)) {
            auto op0 = unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble);
            if (op0 == nullptr) goto endCheck;
            auto op1 = unwrapM(op->getOperand(1), BuilderM, available, lookupIfAble);
            if (op1 == nullptr) goto endCheck;
            auto toreturn = BuilderM.CreateBinOp(op->getOpcode(), op0, op1);
            if (
                    (cache.find(std::make_pair((Value*)op->getOperand(0), BuilderM.GetInsertBlock())) != cache.end()) &&
                    (cache.find(std::make_pair((Value*)op->getOperand(1), BuilderM.GetInsertBlock())) != cache.end()) ) {
                cache[cidx] = toreturn;
            }
            return toreturn;
          } else if (auto op = dyn_cast<ICmpInst>(val)) {
            auto op0 = unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble);
            if (op0 == nullptr) goto endCheck;
            auto op1 = unwrapM(op->getOperand(1), BuilderM, available, lookupIfAble);
            if (op1 == nullptr) goto endCheck;
            auto toreturn = BuilderM.CreateICmp(op->getPredicate(), op0, op1);
            if (
                    (cache.find(std::make_pair((Value*)op->getOperand(0), BuilderM.GetInsertBlock())) != cache.end()) &&
                    (cache.find(std::make_pair((Value*)op->getOperand(1), BuilderM.GetInsertBlock())) != cache.end()) ) {
                cache[cidx] = toreturn;
            }
            return toreturn;
          } else if (auto op = dyn_cast<FCmpInst>(val)) {
            auto op0 = unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble);
            if (op0 == nullptr) goto endCheck;
            auto op1 = unwrapM(op->getOperand(1), BuilderM, available, lookupIfAble);
            if (op1 == nullptr) goto endCheck;
            auto toreturn = BuilderM.CreateFCmp(op->getPredicate(), op0, op1);
            if (
                    (cache.find(std::make_pair((Value*)op->getOperand(0), BuilderM.GetInsertBlock())) != cache.end()) &&
                    (cache.find(std::make_pair((Value*)op->getOperand(1), BuilderM.GetInsertBlock())) != cache.end()) ) {
                cache[cidx] = toreturn;
            }
            return toreturn;
          } else if (auto op = dyn_cast<SelectInst>(val)) {
            auto op0 = unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble);
            if (op0 == nullptr) goto endCheck;
            auto op1 = unwrapM(op->getOperand(1), BuilderM, available, lookupIfAble);
            if (op1 == nullptr) goto endCheck;
            auto op2 = unwrapM(op->getOperand(2), BuilderM, available, lookupIfAble);
            if (op2 == nullptr) goto endCheck;
            auto toreturn = BuilderM.CreateSelect(op0, op1, op2);
            if (
                    (cache.find(std::make_pair((Value*)op->getOperand(0), BuilderM.GetInsertBlock())) != cache.end()) &&
                    (cache.find(std::make_pair((Value*)op->getOperand(1), BuilderM.GetInsertBlock())) != cache.end()) &&
                    (cache.find(std::make_pair((Value*)op->getOperand(2), BuilderM.GetInsertBlock())) != cache.end()) ) {
                cache[cidx] = toreturn;
            }
            return toreturn;
          } else if (auto inst = dyn_cast<GetElementPtrInst>(val)) {
              auto ptr = unwrapM(inst->getPointerOperand(), BuilderM, available, lookupIfAble);
              if (ptr == nullptr) goto endCheck;
              bool cached = cache.find(std::make_pair(inst->getPointerOperand(), BuilderM.GetInsertBlock())) != cache.end();
              SmallVector<Value*,4> ind;
              for(auto& a : inst->indices()) {
                auto op = unwrapM(a, BuilderM,available, lookupIfAble);
                if (op == nullptr) goto endCheck;
                cached &= cache.find(std::make_pair((Value*)a, BuilderM.GetInsertBlock())) != cache.end();
                ind.push_back(op);
              }
              auto toreturn = BuilderM.CreateGEP(ptr, ind, inst->getName() + "_unwrap");
              if (cached) {
                    cache[cidx] = toreturn;
              }
              return toreturn;
          } else if (auto load = dyn_cast<LoadInst>(val)) {
                Value* idx = unwrapM(load->getOperand(0), BuilderM, available, lookupIfAble);
                if (idx == nullptr) goto endCheck;
                auto toreturn = BuilderM.CreateLoad(idx);
                if (cache.find(std::make_pair((Value*)load->getOperand(0), BuilderM.GetInsertBlock())) != cache.end()) {
                    cache[cidx] = toreturn;
                }
                return toreturn;
          } else if (auto op = dyn_cast<IntrinsicInst>(val)) {
            switch(op->getIntrinsicID()) {
                case Intrinsic::sin: {
                  Value *args[] = {unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble)};
                  if (args[0] == nullptr) goto endCheck;
                  Type *tys[] = {op->getOperand(0)->getType()};
                  return BuilderM.CreateCall(Intrinsic::getDeclaration(op->getParent()->getParent()->getParent(), Intrinsic::sin, tys), args);
                }
                case Intrinsic::cos: {
                  Value *args[] = {unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble)};
                  if (args[0] == nullptr) goto endCheck;
                  Type *tys[] = {op->getOperand(0)->getType()};
                  return BuilderM.CreateCall(Intrinsic::getDeclaration(op->getParent()->getParent()->getParent(), Intrinsic::cos, tys), args);
                }
                default:;

            }
          } else if (auto phi = dyn_cast<PHINode>(val)) {
            if (phi->getNumIncomingValues () == 1) {
                return unwrapM(phi->getIncomingValue(0), BuilderM, available, lookupIfAble);
            }
          }


endCheck:
            assert(val);
            llvm::errs() << "cannot unwrap following " << *val << "\n";
            if (lookupIfAble)
                return lookupM(val, BuilderM);

          if (auto inst = dyn_cast<Instruction>(val)) {
            //LoopContext lc;
            // if (BuilderM.GetInsertBlock() != inversionAllocs && !( (reverseBlocks.find(BuilderM.GetInsertBlock()) != reverseBlocks.end())  && /*inLoop*/getContext(inst->getParent(), lc)) ) {
            if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
                if (BuilderM.GetInsertBlock()->size() && BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
                    if (DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
                        //llvm::errs() << "allowed " << *inst << "from domination\n";
                        return inst;
                    }
                } else {
                    if (DT.dominates(inst, BuilderM.GetInsertBlock())) {
                        //llvm::errs() << "allowed " << *inst << "from block domination\n";
                        return inst;
                    }
                }
            }
          }
            return nullptr;
            report_fatal_error("unable to unwrap");
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
                ValueToValueMapTy emptyMap;
                IRBuilder <> allocationBuilder(&allocationPreheaders[i]->back());
                Value* limitMinus1 = unwrapM(contexts[i].limit, allocationBuilder, emptyMap, /*lookupIfAble*/false);
                
                // We have a loop with static bounds, but whose limit is not available to be computed at the current loop preheader (such as the innermost loop of triangular iteration domain)
                // Handle this case like a dynamic loop
                if (limitMinus1 == nullptr) {
                    allocationPreheaders[i] = contexts[i].preheader;
                    allocationBuilder.SetInsertPoint(&allocationPreheaders[i]->back());
                    limitMinus1 = unwrapM(contexts[i].limit, allocationBuilder, emptyMap, /*lookupIfAble*/false);
                }
                assert(limitMinus1 != nullptr);
                limits[i] = allocationBuilder.CreateNUWAdd(limitMinus1, ConstantInt::get(limitMinus1->getType(), 1));
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
              size = allocationBuilder.CreateNUWMul(size, limits[i]);
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
    AllocaInst* createCacheForScope(BasicBlock* ctx, Type* T, StringRef name, CallInst** freeLocation, Instruction** lastScopeAllocLocation) {
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
                
        Type *BPTy = Type::getInt8PtrTy(ctx->getContext());
        auto realloc = newFunc->getParent()->getOrInsertFunction("realloc", BPTy, BPTy, Type::getInt64Ty(ctx->getContext()));

        Value* storeInto = alloc;
        ValueToValueMapTy antimap;

        for(int i=sublimits.size()-1; i>=0; i--) {
            const auto& containedloops = sublimits[i].second;
            for(auto riter = containedloops.rbegin(), rend = containedloops.rend(); riter != rend; riter++) {
                const auto& idx = riter->first;
                antimap[idx.var] = idx.antivar;
            }

            Value* size = sublimits[i].first;
            Type* myType = types[i];

            IRBuilder <> allocationBuilder(&containedloops.back().first.preheader->back());
            if (!sublimits[i].second.back().first.dynamic) {
                auto firstallocation = CallInst::CreateMalloc(
                        &allocationBuilder.GetInsertBlock()->back(),
                        size->getType(),
                        myType,
                        ConstantInt::get(size->getType(), allocationBuilder.GetInsertBlock()->getParent()->getParent()->getDataLayout().getTypeAllocSizeInBits(myType)/8), size, nullptr, name+"_malloccache");
                CallInst* malloccall = dyn_cast<CallInst>(firstallocation);
                if (malloccall == nullptr) {
                    malloccall = cast<CallInst>(cast<Instruction>(firstallocation)->getOperand(0));
                }
                malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
                malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
                
                allocationBuilder.CreateStore(firstallocation, storeInto);
                
                if (lastScopeAllocLocation) {
                    *lastScopeAllocLocation = cast<Instruction>(firstallocation);
                }

                //allocationBuilder.GetInsertBlock()->getInstList().push_back(cast<Instruction>(allocation));
                //cast<Instruction>(firstallocation)->moveBefore(allocationBuilder.GetInsertBlock()->getTerminator());
                //mallocs.push_back(firstallocation);
            } else {
                allocationBuilder.CreateStore(ConstantPointerNull::get(PointerType::getUnqual(myType)), storeInto);

                IRBuilder <> build(containedloops.back().first.header->getFirstNonPHI());
                Value* allocation = build.CreateLoad(storeInto);
                Value* foo = build.CreateNUWAdd(containedloops.back().first.var, ConstantInt::get(Type::getInt64Ty(ctx->getContext()), 1));
                Value* realloc_size = build.CreateNUWMul(foo, sublimits[i].first);
                Value* idxs[2] = {
                    build.CreatePointerCast(allocation, BPTy),
                    build.CreateNUWMul(
                        ConstantInt::get(size->getType(), newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(myType)/8), realloc_size
                    )
                };

                Value* realloccall = nullptr;
                allocation = build.CreatePointerCast(realloccall = build.CreateCall(realloc, idxs, name+"_realloccache"), allocation->getType());
                if (lastScopeAllocLocation) {
                    *lastScopeAllocLocation = cast<Instruction>(allocation);
                }
                build.CreateStore(allocation, storeInto);
            }

            if (freeLocation) {
                assert(reverseBlocks.size());

                IRBuilder<> tbuild(reverseBlocks[containedloops.back().first.preheader]);
                tbuild.setFastMathFlags(getFast());

                // ensure we are before the terminator if it exists
                if (tbuild.GetInsertBlock()->size()) {
                      tbuild.SetInsertPoint(tbuild.GetInsertBlock()->getFirstNonPHI());
                }
                auto forfree = cast<LoadInst>(tbuild.CreateLoad(unwrapM(storeInto, tbuild, antimap, /*lookup*/false)));
                forfree->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(forfree->getContext(), {}));
                auto ci = cast<CallInst>(CallInst::CreateFree(tbuild.CreatePointerCast(forfree, Type::getInt8PtrTy(ctx->getContext())), tbuild.GetInsertBlock()));
                ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
                if (ci->getParent()==nullptr) {
                    tbuild.Insert(ci);
                }
                *freeLocation = ci;
            }
        
            if (i != 0) {
                IRBuilder <>v(&sublimits[i-1].second.back().first.preheader->back());
                //TODO
                if (!sublimits[i].second.back().first.dynamic) {
                    storeInto = v.CreateGEP(v.CreateLoad(storeInto), sublimits[i].second.back().first.var);
                } else {
                    storeInto = v.CreateGEP(v.CreateLoad(storeInto), sublimits[i].second.back().first.var);
                }
            }
        }
        return alloc;
    }

    Value* getCachePointer(IRBuilder <>& BuilderM, BasicBlock* ctx, Value* cache) {
        assert(ctx);
        assert(cache);
        
        auto sublimits = getSubLimits(ctx);
			
        ValueToValueMapTy available;
        
        Value* next = cache;
        for(int i=sublimits.size()-1; i>=0; i--) {
            next = BuilderM.CreateLoad(next);
            cast<LoadInst>(next)->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(next->getContext(), {}));

            const auto& containedloops = sublimits[i].second; 

            SmallVector<Value*,3> indices;
            SmallVector<Value*,3> limits;
            for(auto riter = containedloops.rbegin(), rend = containedloops.rend(); riter != rend; riter++) {
              // Only include dynamic index on last iteration (== skip dynamic index on non-last iterations)
              //if (i != 0 && riter+1 == rend) break;

              const auto &idx = riter->first;
              if (!isOriginalBlock(*BuilderM.GetInsertBlock())) {
                indices.push_back(idx.antivar);
                available[idx.var] = idx.antivar;
              } else {
                indices.push_back(idx.var);
                available[idx.var] = idx.var;
              }

              Value* lim = unwrapM(riter->second, BuilderM, available, /*lookupIfAble*/true);
              assert(lim);
              if (limits.size() == 0) {
                limits.push_back(lim);
              } else {
                limits.push_back(BuilderM.CreateNUWMul(lim, limits.back()));
              }
            }

            if (indices.size() > 0) {
                Value* idx = indices[0];
                for(unsigned ind=1; ind<indices.size(); ind++) {
                  idx = BuilderM.CreateNUWAdd(idx, BuilderM.CreateNUWMul(indices[ind], limits[ind-1]));
                }
                next = BuilderM.CreateGEP(next, {idx});
            }
            
            /*
            if (i != 0) {
                //TODO
                if (!sublimits[i].second.back().first.dynamic) {
                    next = BuilderM.CreateGEP(next, sublimits[i].second.back().first.var);
                } else {
                    next = BuilderM.CreateGEP(next, sublimits[i].second.back().first.var);
                }
            }
            */
        }
        return next;
    }
    
    LoadInst* lookupValueFromCache(IRBuilder<>& BuilderM, BasicBlock* ctx, Value* cache) {
        auto result = BuilderM.CreateLoad(getCachePointer(BuilderM, ctx, cache));
        result->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(ctx->getContext(), {}));
        return result;
    }

    void storeInstructionInCache(BasicBlock* ctx, IRBuilder <>& BuilderM, Value* val, AllocaInst* cache) {
        IRBuilder <> v(BuilderM);
        v.setFastMathFlags(getFast());

        //Note for dynamic loops where the allocation is stored somewhere inside the loop,
        // we must ensure that we load the allocation after the store ensuring memory exists
        // This does not need to occur (and will find no such store) for nondynamic loops
        // as memory is statically allocated in the preheader
        for (auto I = BuilderM.GetInsertBlock()->rbegin(), E = BuilderM.GetInsertBlock()->rend(); I != E; I++) {
            if (&*I == &*BuilderM.GetInsertPoint()) break;
            if (auto si = dyn_cast<StoreInst>(&*I)) {
                if (si->getPointerOperand() == cache) {
                    v.SetInsertPoint(getNextNonDebugInstruction(si));
                }   
            }
        }

        v.CreateStore(val, getCachePointer(v, ctx, cache));
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
        CallInst* free = nullptr;
        Instruction* lastalloc = nullptr;
        AllocaInst* cache = createCacheForScope(inst->getParent(), inst->getType(), inst->getName(), shouldFree ? &free : nullptr, &lastalloc);
        assert(cache);
        scopeMap[inst] = cache;
        if (free) {
            scopeFrees[inst] = free;
        }
        if (lastalloc) {
            lastScopeAlloc[inst] = lastalloc;
        }
        storeInstructionInCache(inst->getParent(), inst, cache);
    }

    Instruction* fixLCSSA(Instruction* inst, const IRBuilder <>& BuilderM) {
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

    Value* lookupM(Value* val, IRBuilder<>& BuilderM);

    Value* invertPointerM(Value* val, IRBuilder<>& BuilderM);
  
    void branchToCorrespondingTarget(BasicBlock* ctx, IRBuilder <>& BuilderM, const std::map<BasicBlock*, std::vector<std::pair</*pred*/BasicBlock*,/*successor*/BasicBlock*>>> &targetToPreds, const std::map<BasicBlock*,PHINode*>* replacePHIs = nullptr);

};

class DiffeGradientUtils : public GradientUtils {
  DiffeGradientUtils(Function* newFunc_, AAResults &AA, TargetLibraryInfo &TLI, ValueToValueMapTy& invertedPointers_, const SmallPtrSetImpl<Value*> &constants_, const SmallPtrSetImpl<Value*> &nonconstant_, const SmallPtrSetImpl<Value*> &returnvals_, ValueToValueMapTy &origToNew_)
      : GradientUtils(newFunc_, AA, TLI, invertedPointers_, constants_, nonconstant_, returnvals_, origToNew_) {
        prepareForReverse();
    }

public:
  ValueToValueMapTy differentials;
  static DiffeGradientUtils* CreateFromClone(Function *todiff, AAResults &AA, TargetLibraryInfo &TLI, const std::set<unsigned> & constant_args, ReturnType returnValue, bool differentialReturn, Type* additionalArg);

private:
  Value* getDifferential(Value *val) {
    assert(val);
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
      if (isConstantValue(val)) {
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
  std::vector<SelectInst*> addToDiffe(Value* val, Value* dif, IRBuilder<> &BuilderM) {
      std::vector<SelectInst*> addedSelects;

      if (val->getType()->isPointerTy()) {
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << *val << "\n";
      }
      if (isConstantValue(val)) {
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << *val << "\n";
      }
      assert(!val->getType()->isPointerTy());
      assert(!isConstantValue(val));
      assert(val->getType() == dif->getType());
      auto old = diffe(val, BuilderM);
      assert(val->getType() == old->getType());
      Value* res;
      if (val->getType()->isIntOrIntVectorTy()) {
        res = BuilderM.CreateFAdd(BuilderM.CreateBitCast(old, IntToFloatTy(old->getType())), BuilderM.CreateBitCast(dif, IntToFloatTy(dif->getType())));
        res = BuilderM.CreateBitCast(res, val->getType());
        BuilderM.CreateStore(res, getDifferential(val));
        return addedSelects;
      } else if (val->getType()->isFPOrFPVectorTy()) {
        
        res = BuilderM.CreateFAdd(old, dif);

        //! optimize fadd of select to select of fadd
        if (SelectInst* select = dyn_cast<SelectInst>(dif)) {
            if (Constant* ci = dyn_cast<Constant>(select->getTrueValue())) {
                if (ci->isZeroValue()) {
                    cast<Instruction>(res)->eraseFromParent();
                    res = BuilderM.CreateSelect(select->getCondition(), old, BuilderM.CreateFAdd(old, select->getFalseValue()));
                    addedSelects.emplace_back(cast<SelectInst>(res));
                    goto endselect;
                }
            }
            if (Constant* ci = dyn_cast<Constant>(select->getFalseValue())) {
                if (ci->isZeroValue()) {
                    cast<Instruction>(res)->eraseFromParent();
                    res = BuilderM.CreateSelect(select->getCondition(), BuilderM.CreateFAdd(old, select->getTrueValue()), old);
                    addedSelects.emplace_back(cast<SelectInst>(res));
                    goto endselect;
                }
            }
        }
        endselect:;

        BuilderM.CreateStore(res, getDifferential(val));
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
      if (isConstantValue(val)) {
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << *val << "\n";
      }
      assert(!isConstantValue(val));
      Value* tostore = getDifferential(val);
      if (toset->getType() != cast<PointerType>(tostore->getType())->getElementType()) {
        llvm::errs() << "toset:" << *toset << "\n";
        llvm::errs() << "tostore:" << *tostore << "\n";
      }
      assert(toset->getType() == cast<PointerType>(tostore->getType())->getElementType());
      BuilderM.CreateStore(toset, tostore);
  }

  SelectInst* addToDiffeIndexed(Value* val, Value* dif, ArrayRef<Value*> idxs, IRBuilder<> &BuilderM) {
      assert(!isConstantValue(val));
      SmallVector<Value*,4> sv;
      sv.push_back(ConstantInt::get(Type::getInt32Ty(val->getContext()), 0));
      for(auto i : idxs)
        sv.push_back(i);
      Value* ptr = BuilderM.CreateGEP(getDifferential(val), sv);
      Value* old = BuilderM.CreateLoad(ptr);
        
      Value* res = BuilderM.CreateFAdd(old, dif);
      SelectInst* addedSelect = nullptr;

        //! optimize fadd of select to select of fadd
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

  void addToPtrDiffe(Value* val, Value* dif, IRBuilder<> &BuilderM) {
      if (!(val->getType()->isPointerTy()) || !(cast<PointerType>(val->getType())->getElementType() == dif->getType())) {
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << "Val: " << *val << "\n";
        llvm::errs() << "Diff: " << *dif << "\n";
      }
      assert(val->getType()->isPointerTy());
      assert(cast<PointerType>(val->getType())->getElementType() == dif->getType());

      auto ptr = invertPointerM(val, BuilderM);
      assert(ptr->getType()->isPointerTy());
      assert(cast<PointerType>(ptr->getType())->getElementType() == dif->getType());

      Value* res;
      Value* old = BuilderM.CreateLoad(ptr);
      if (old->getType()->isIntOrIntVectorTy()) {
        res = BuilderM.CreateFAdd(BuilderM.CreateBitCast(old, IntToFloatTy(old->getType())), BuilderM.CreateBitCast(dif, IntToFloatTy(dif->getType())));
        res = BuilderM.CreateBitCast(res, old->getType());
      } else if(old->getType()->isFPOrFPVectorTy()) {
        res = BuilderM.CreateFAdd(old, dif);
      } else {
        assert(old);
        assert(dif);
        llvm::errs() << *newFunc << "\n" << "cannot handle type " << *old << "\n" << *dif;
        report_fatal_error("cannot handle type");
      }
      BuilderM.CreateStore(res, ptr);
  }

  StoreInst* setPtrDiffe(Value* ptr, Value* newval, IRBuilder<> &BuilderM) {
      ptr = invertPointerM(ptr, BuilderM);
      return BuilderM.CreateStore(newval, ptr);
  }

};
#endif
