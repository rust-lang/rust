
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

    invertedPointers[call] = anti;
    assert(placeholder != anti);
    bb.SetInsertPoint(placeholder->getNextNode());
    replaceAWithB(placeholder, anti);
    erase(placeholder);

    anti = addMalloc<Instruction>(bb, anti);
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

  template<typename T>
  T* addMalloc(IRBuilder<> &BuilderQ, T* malloc) {
    if (tape) {
        Instruction* ret = cast<Instruction>(BuilderQ.CreateExtractValue(tape, {tapeidx}));
        Instruction* origret = ret;
        tapeidx++;

        if (ret->getType()->isEmptyTy()) {
            /*
            if (auto inst = dyn_cast<Instruction>(malloc)) {
                inst->replaceAllUsesWith(UndefValue::get(ret->getType()));
                erase(inst);
            }
            */
            return ret;
            //UndefValue::get(ret->getType());
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
            auto n = malloc->getName();
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

      llvm::errs() << " added malloc " << *malloc << "\n";

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
            this->invertedPointers[op] = BuilderZ.CreatePHI(op->getType(), 1);

			if ( called && (called->getName() == "malloc" || called->getName() == "_Znwm")) {
				this->invertedPointers[op]->setName(op->getName()+"'mi");
			}
        }
      }
  }

  Value* unwrapM(Value* val, IRBuilder<>& BuilderM, const ValueToValueMapTy& available, bool lookupIfAble) {
          assert(val);
          if (available.count(val)) {
            return available.lookup(val);
          }

          if (isa<Argument>(val) || isa<Constant>(val)) {
            return val;
          } else if (isa<AllocaInst>(val)) {
            return val;
          } else if (auto op = dyn_cast<CastInst>(val)) {
            auto op0 = unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble);
            if (op0 == nullptr) goto endCheck;
            return BuilderM.CreateCast(op->getOpcode(), op0, op->getDestTy(), op->getName()+"_unwrap");
          } else if (auto op = dyn_cast<ExtractValueInst>(val)) {
            auto op0 = unwrapM(op->getAggregateOperand(), BuilderM, available, lookupIfAble);
            if (op0 == nullptr) goto endCheck;
            return BuilderM.CreateExtractValue(op0, op->getIndices(), op->getName()+"_unwrap");
          } else if (auto op = dyn_cast<BinaryOperator>(val)) {
            auto op0 = unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble);
            if (op0 == nullptr) goto endCheck;
            auto op1 = unwrapM(op->getOperand(1), BuilderM, available, lookupIfAble);
            if (op1 == nullptr) goto endCheck;
            return BuilderM.CreateBinOp(op->getOpcode(), op0, op1);
          } else if (auto op = dyn_cast<ICmpInst>(val)) {
            auto op0 = unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble);
            if (op0 == nullptr) goto endCheck;
            auto op1 = unwrapM(op->getOperand(1), BuilderM, available, lookupIfAble);
            if (op1 == nullptr) goto endCheck;
            return BuilderM.CreateICmp(op->getPredicate(), op0, op1);
          } else if (auto op = dyn_cast<FCmpInst>(val)) {
            auto op0 = unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble);
            if (op0 == nullptr) goto endCheck;
            auto op1 = unwrapM(op->getOperand(1), BuilderM, available, lookupIfAble);
            if (op1 == nullptr) goto endCheck;
            return BuilderM.CreateFCmp(op->getPredicate(), op0, op1);
          } else if (auto op = dyn_cast<SelectInst>(val)) {
            auto op0 = unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble);
            if (op0 == nullptr) goto endCheck;
            auto op1 = unwrapM(op->getOperand(1), BuilderM, available, lookupIfAble);
            if (op1 == nullptr) goto endCheck;
            auto op2 = unwrapM(op->getOperand(2), BuilderM, available, lookupIfAble);
            if (op2 == nullptr) goto endCheck;
            return BuilderM.CreateSelect(op0, op1, op2);
          } else if (auto inst = dyn_cast<GetElementPtrInst>(val)) {
              auto ptr = unwrapM(inst->getPointerOperand(), BuilderM, available, lookupIfAble);
              if (ptr == nullptr) goto endCheck;
              SmallVector<Value*,4> ind;
              for(auto& a : inst->indices()) {
                auto op = unwrapM(a, BuilderM,available, lookupIfAble);
                if (op == nullptr) goto endCheck;
                ind.push_back(op);
              }
              return BuilderM.CreateGEP(ptr, ind);
          } else if (auto load = dyn_cast<LoadInst>(val)) {
                Value* idx = unwrapM(load->getOperand(0), BuilderM, available, lookupIfAble);
                if (idx == nullptr) goto endCheck;
                return BuilderM.CreateLoad(idx);
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
    
    AllocaInst* createCacheForScope(BasicBlock* ctx, Type* T, StringRef name, CallInst** freeLocation, Instruction** lastScopeAllocLocation) {
        assert(ctx);
        assert(T);
        LoopContext lc;
        bool inLoop = getContext(ctx, lc);

        assert(inversionAllocs && "must be able to allocate inverted caches");
        IRBuilder<> entryBuilder(inversionAllocs);
        entryBuilder.setFastMathFlags(getFast());

        if (!inLoop) {
            return entryBuilder.CreateAlloca(T, nullptr, name+"_cache");
        } else {
            Value* size = nullptr;

            BasicBlock* outermostPreheader = nullptr;

            for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx) ) {
                if (idx.parent == nullptr) {
                    outermostPreheader = idx.preheader;
                }
                if (idx.parent == nullptr) break;
            }
            assert(outermostPreheader);

            IRBuilder <> allocationBuilder(&outermostPreheader->back());

            for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx) ) {
              //TODO handle allocations for dynamic loops
              if (idx.dynamic && idx.parent != nullptr) {
                assert(idx.var);
                assert(idx.var->getParent());
                assert(idx.var->getParent()->getParent());
                llvm::errs() << *idx.var->getParent()->getParent() << "\n"
                    << "idx.var=" <<*idx.var << "\n"
                    << "idx.limit=" <<*idx.limit << "\n";
                llvm::errs() << "cannot handle non-outermost dynamic loop\n";
                assert(0 && "cannot handle non-outermost dynamic loop");
              }
              Value* ns = nullptr;
              Type* intT = idx.dynamic ? cast<PointerType>(idx.limit->getType())->getElementType() : idx.limit->getType();
              if (idx.dynamic) {
                ns = ConstantInt::get(intT, 1);
              } else {
                Value* limitm1 = nullptr;
                ValueToValueMapTy emptyMap;
                limitm1 = unwrapM(idx.limit, allocationBuilder, emptyMap, /*lookupIfAble*/false);
                if (limitm1 == nullptr) {
                    assert(outermostPreheader);
                    assert(outermostPreheader->getParent());
                    llvm::errs() << *outermostPreheader->getParent() << "\n";
                    llvm::errs() << "needed value " << *idx.limit << " at " << allocationBuilder.GetInsertBlock()->getName() << "\n";
                }
                assert(limitm1);
                ns = allocationBuilder.CreateNUWAdd(limitm1, ConstantInt::get(intT, 1));
              }
              if (size == nullptr) size = ns;
              else size = allocationBuilder.CreateNUWMul(size, ns);
              if (idx.parent == nullptr) break;
            }

            auto firstallocation = CallInst::CreateMalloc(
                    &allocationBuilder.GetInsertBlock()->back(),
                    size->getType(),
                    T,
                    ConstantInt::get(size->getType(), allocationBuilder.GetInsertBlock()->getParent()->getParent()->getDataLayout().getTypeAllocSizeInBits(T)/8), size, nullptr, name+"_malloccache");
            CallInst* malloccall = dyn_cast<CallInst>(firstallocation);
            if (malloccall == nullptr) {
                malloccall = cast<CallInst>(cast<Instruction>(firstallocation)->getOperand(0));
            }
            malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
            malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
            //allocationBuilder.GetInsertBlock()->getInstList().push_back(cast<Instruction>(allocation));
            cast<Instruction>(firstallocation)->moveBefore(allocationBuilder.GetInsertBlock()->getTerminator());
            AllocaInst* holderAlloc = entryBuilder.CreateAlloca(firstallocation->getType(), nullptr, name+"_mdyncache");
            if (lastScopeAllocLocation)
                *lastScopeAllocLocation = firstallocation;
            allocationBuilder.CreateStore(firstallocation, holderAlloc);


            if (freeLocation) {
                assert(reverseBlocks.size());

                IRBuilder<> tbuild(reverseBlocks[outermostPreheader]);
                tbuild.setFastMathFlags(getFast());

                // ensure we are before the terminator if it exists
                if (tbuild.GetInsertBlock()->size()) {
                      tbuild.SetInsertPoint(tbuild.GetInsertBlock()->getFirstNonPHI());
                }

                auto ci = cast<CallInst>(CallInst::CreateFree(tbuild.CreatePointerCast(tbuild.CreateLoad(holderAlloc), Type::getInt8PtrTy(outermostPreheader->getContext())), tbuild.GetInsertBlock()));
                ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
                if (ci->getParent()==nullptr) {
                    tbuild.Insert(ci);
                }
                *freeLocation = ci;
            }

            IRBuilder <> v(ctx->getFirstNonPHI());
            v.setFastMathFlags(getFast());

            SmallVector<Value*,3> indices;
            SmallVector<Value*,3> limits;
            PHINode* dynamicPHI = nullptr;

            for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx) ) {
              indices.push_back(idx.var);

              if (idx.dynamic) {
                dynamicPHI = idx.var;
                assert(dynamicPHI);
                llvm::errs() << "saw idx.dynamic:" << *dynamicPHI << "\n";
                assert(idx.parent == nullptr);
                break;
              }

              if (idx.parent == nullptr) break;
              ValueToValueMapTy emptyMap;
              auto limitm1 = unwrapM(idx.limit, v, emptyMap, /*lookupIfAble*/false);
              assert(limitm1);
              Type* intT = idx.dynamic ? cast<PointerType>(idx.limit->getType())->getElementType() : idx.limit->getType();
              auto lim = v.CreateNUWAdd(limitm1, ConstantInt::get(intT, 1));
              if (limits.size() != 0) {
                lim = v.CreateNUWMul(lim, limits.back());
              }
              limits.push_back(lim);
            }

            Value* idx = nullptr;
            for(unsigned i=0; i<indices.size(); i++) {
              if (i == 0) {
                idx = indices[i];
              } else {
                auto mul = v.CreateNUWMul(indices[i], limits[i-1]);
                idx = v.CreateNUWAdd(idx, mul);
              }
            }

            if (dynamicPHI != nullptr) {
                Type *BPTy = Type::getInt8PtrTy(v.GetInsertBlock()->getContext());
                auto realloc = newFunc->getParent()->getOrInsertFunction("realloc", BPTy, BPTy, size->getType());
                Value* allocation = v.CreateLoad(holderAlloc);
                auto foo = v.CreateNUWAdd(dynamicPHI, ConstantInt::get(dynamicPHI->getType(), 1));
                Value* idxs[2] = {
                    v.CreatePointerCast(allocation, BPTy),
                    v.CreateNUWMul(
                        ConstantInt::get(size->getType(), newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(T)/8),
                        v.CreateNUWMul(
                            size, foo
                        )
                    )
                };

                Value* realloccall = nullptr;
                allocation = v.CreatePointerCast(realloccall = v.CreateCall(realloc, idxs, name+"_realloccache"), allocation->getType());
                if (lastScopeAllocLocation) {
                    *lastScopeAllocLocation = cast<Instruction>(allocation);
                }
                v.CreateStore(allocation, holderAlloc);
            }
            return holderAlloc;
        }
    }

    void storeInstructionInCache(BasicBlock* ctx, Instruction* inst, AllocaInst* cache) {
        assert(ctx);
        assert(inst);
        assert(cache);
        LoopContext lc;
        bool inLoop = getContext(ctx, lc);

        if (!inLoop) {
            auto pn = dyn_cast<PHINode>(inst);
            Instruction* putafter = ( pn && pn->getNumIncomingValues()>0 )? (inst->getParent()->getFirstNonPHI() ): getNextNonDebugInstruction(inst);
            assert(putafter);
            IRBuilder <> v(putafter);
            v.setFastMathFlags(getFast());
            v.CreateStore(inst, cache);
        } else {
            auto pn = dyn_cast<PHINode>(inst);

            Instruction* putafter = ( pn && pn->getNumIncomingValues()>0 )? (inst->getParent()->getFirstNonPHI() ): getNextNonDebugInstruction(inst);
            IRBuilder <> v(putafter);
            v.setFastMathFlags(getFast());

            //Note for dynamic loops where the allocation is stored somewhere inside the loop,
            // we must ensure that we load the allocation after the store ensuring memory exists
            // This does not need to occur (and will find no such store) for nondynamic loops
            // as memory is statically allocated in the preheader
            for (auto I = inst->getParent()->rbegin(), E = inst->getParent()->rend(); I != E; I++) {
                if (&*I == inst) break;
                if (auto si = dyn_cast<StoreInst>(&*I)) {
                    if (si->getPointerOperand() == cache) {
                        //if (&*inst->getParent()->rbegin() == si) {
                        //    v.SetInsertPoint(inst->getParent());
                        //} else {
                            v.SetInsertPoint(getNextNonDebugInstruction(si));
                        //}
                    }   
                }
            }

            SmallVector<Value*,3> indices;
            SmallVector<Value*,3> limits;
            PHINode* dynamicPHI = nullptr;

            for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx) ) {
              indices.push_back(idx.var);

              if (idx.dynamic) {
                dynamicPHI = idx.var;
                assert(dynamicPHI);
                llvm::errs() << "saw idx.dynamic:" << *dynamicPHI << "\n";
                assert(idx.parent == nullptr);
                break;
              }

              if (idx.parent == nullptr) break;
              ValueToValueMapTy emptyMap;
              auto limitm1 = unwrapM(idx.limit, v, emptyMap, /*lookupIfAble*/false);
              assert(limitm1);
              auto lim = v.CreateNUWAdd(limitm1, ConstantInt::get(idx.limit->getType(), 1));
              if (limits.size() != 0) {
                lim = v.CreateNUWMul(lim, limits.back());
              }
              limits.push_back(lim);
            }

            Value* idx = nullptr;
            for(unsigned i=0; i<indices.size(); i++) {
              if (i == 0) {
                idx = indices[i];
              } else {
                auto mul = v.CreateNUWMul(indices[i], limits[i-1]);
                idx = v.CreateNUWAdd(idx, mul);
              }
            }

            Value* allocation = nullptr;
            if (dynamicPHI == nullptr) {
				BasicBlock* outermostPreheader = nullptr;

				for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx) ) {
					if (idx.parent == nullptr) {
						outermostPreheader = idx.preheader;
					}
					if (idx.parent == nullptr) break;
				}
				assert(outermostPreheader);

                IRBuilder<> outerBuilder(&outermostPreheader->back());
                allocation = outerBuilder.CreateLoad(cache);
            } else {
                allocation = v.CreateLoad(cache);
            }

            Value* idxs[] = {idx};
            auto gep = v.CreateGEP(allocation, idxs);
            v.CreateStore(inst, gep);
        }

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

    LoadInst* lookupValueFromCache(IRBuilder<>& BuilderM, BasicBlock* ctx, Value* cache) {
        assert(ctx);
        assert(cache);
        LoopContext lc;
        bool inLoop = getContext(ctx, lc);

        if (!inLoop) {
            auto result = BuilderM.CreateLoad(cache);
            result->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(ctx->getContext(), {}));
            return result;
        } else {

			ValueToValueMapTy available;
			for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx)) {
			  if (!isOriginalBlock(*BuilderM.GetInsertBlock())) {
				available[idx.var] = idx.antivar;
			  } else {
				available[idx.var] = idx.var;
			  }
			  if (idx.parent == nullptr) break;
			}

            SmallVector<Value*,3> indices;
            SmallVector<Value*,3> limits;
            for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx) ) {
              indices.push_back(unwrapM(idx.var, BuilderM, available, /*lookupIfAble*/false));
              if (idx.parent == nullptr) break;

              auto limitm1 = unwrapM(idx.limit, BuilderM, available, /*lookupIfAble*/true);
              assert(limitm1);
              auto lim = BuilderM.CreateNUWAdd(limitm1, ConstantInt::get(idx.limit->getType(), 1));
              if (limits.size() != 0) {
                lim = BuilderM.CreateNUWMul(lim, limits.back());
              }
              limits.push_back(lim);
            }

            Value* idx = nullptr;
            for(unsigned i=0; i<indices.size(); i++) {
              if (i == 0) {
                idx = indices[i];
              } else {
                idx = BuilderM.CreateNUWAdd(idx, BuilderM.CreateNUWMul(indices[i], limits[i-1]));
              }
            }

            Value* idxs[] = {idx};
            Value* tolookup = BuilderM.CreateLoad(cache);
            auto result = BuilderM.CreateLoad(BuilderM.CreateGEP(tolookup, idxs));
            result->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(result->getContext(), {}));
            return result;
        }
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

                for (BasicBlock* exit : lc.exitBlocks) {
                    if (exit == forwardBlock || DT.dominates(exit, forwardBlock)) {
                        IRBuilder<> lcssa(&exit->front());
                        auto lcssaPHI = lcssa.CreatePHI(inst->getType(), 1, inst->getName()+"!manual_lcssa");
                        for(auto pred : predecessors(exit))
                            lcssaPHI->addIncoming(inst, pred);
                        return lcssaPHI;
                    }
                }
                llvm::errs() << "unable to do lcssa for multi exit block loop with no exit dominating use\n";
                exit(1);
            }
        }
        return inst;
    }

    Value* lookupM(Value* val, IRBuilder<>& BuilderM);

    Value* invertPointerM(Value* val, IRBuilder<>& BuilderM);
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

  void addToDiffe(Value* val, Value* dif, IRBuilder<> &BuilderM) {
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
      } else if (val->getType()->isFPOrFPVectorTy()) {
        
        res = BuilderM.CreateFAdd(old, dif);

        //! optimize fadd of select to select of fadd
        if (SelectInst* select = dyn_cast<SelectInst>(dif)) {
            if (ConstantFP* ci = dyn_cast<ConstantFP>(select->getTrueValue())) {
                if (ci->isZero()) {
                    cast<Instruction>(res)->eraseFromParent();
                    res = BuilderM.CreateSelect(select->getCondition(), old, BuilderM.CreateFAdd(old, select->getFalseValue()));
                    goto endselect;
                }
            }
            if (ConstantFP* ci = dyn_cast<ConstantFP>(select->getFalseValue())) {
                if (ci->isZero()) {
                    cast<Instruction>(res)->eraseFromParent();
                    res = BuilderM.CreateSelect(select->getCondition(), BuilderM.CreateFAdd(old, select->getTrueValue()), old);
                    goto endselect;
                }
            }
        }
        endselect:;

        BuilderM.CreateStore(res, getDifferential(val));
      } else if (val->getType()->isStructTy()) {
        auto st = cast<StructType>(val->getType());
        for(unsigned i=0; i<st->getNumElements(); i++) {
            Value* v = ConstantInt::get(Type::getInt32Ty(st->getContext()), i);
            addToDiffeIndexed(val, BuilderM.CreateExtractValue(dif,{i}), {v}, BuilderM);
        }
      } else {
        assert(0 && "lol");
        exit(1);
      }
  }

  void setDiffe(Value* val, Value* toset, IRBuilder<> &BuilderM) {
      if (isConstantValue(val)) {
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << *val << "\n";
      }
      assert(!isConstantValue(val));
      BuilderM.CreateStore(toset, getDifferential(val));
  }

  void addToDiffeIndexed(Value* val, Value* dif, ArrayRef<Value*> idxs, IRBuilder<> &BuilderM) {
      assert(!isConstantValue(val));
      SmallVector<Value*,4> sv;
      sv.push_back(ConstantInt::get(Type::getInt32Ty(val->getContext()), 0));
      for(auto i : idxs)
        sv.push_back(i);
      auto ptr = BuilderM.CreateGEP(getDifferential(val), sv);
      BuilderM.CreateStore(BuilderM.CreateFAdd(BuilderM.CreateLoad(ptr), dif), ptr);
  }

  void addToPtrDiffe(Value* val, Value* dif, IRBuilder<> &BuilderM) {
      auto ptr = invertPointerM(val, BuilderM);
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

  void setPtrDiffe(Value* ptr, Value* newval, IRBuilder<> &BuilderM) {
      ptr = invertPointerM(ptr, BuilderM);
      BuilderM.CreateStore(newval, ptr);
  }
};
#endif
