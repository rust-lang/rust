/*
 * LowerAutodiffIntrinsic.cpp - Lower autodiff intrinsic
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


#include <llvm/Config/llvm-config.h>

#include "SCEV/ScalarEvolutionExpander.h"

#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"

#if LLVM_VERSION_MAJOR > 6
#include "llvm/Analysis/PhiValues.h"
#include "llvm/Transforms/Utils.h"
#endif

#include "llvm/IR/DebugInfoMetadata.h"

#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Verifier.h"
//#include "llvm/Transforms/Utils/EaryCSE.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"

#include "llvm/Analysis/ScopedNoAliasAA.h"

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"

#include "llvm/ADT/SmallSet.h"

#include <utility>
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"

#if LLVM_VERSION_MAJOR > 6
#include "llvm/Transforms/Scalar/InstSimplifyPass.h"
#endif
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Transforms/Scalar/CorrelatedValuePropagation.h"
#include "llvm/Transforms/Scalar/LoopIdiomRecognize.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"

#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"

#include "llvm/IR/InstrTypes.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "lower-autodiff-intrinsic"

static cl::opt<bool> autodiff_inline(
            "enzyme_inline", cl::init(false), cl::Hidden,
                cl::desc("Force inlining of autodiff"));

static cl::opt<int> autodiff_inline_count(
            "enzyme_inline_count", cl::init(10000), cl::Hidden,
                cl::desc("Limit of number of functions to inline"));

static cl::opt<bool> printconst(
            "enzyme_printconst", cl::init(false), cl::Hidden,
                cl::desc("Print constant detection algorithm"));

static cl::opt<bool> autodiff_print(
            "enzyme_print", cl::init(false), cl::Hidden,
                cl::desc("Print before and after fns for autodiff"));

static cl::opt<bool> enzyme_preopt(
            "enzyme_preopt", cl::init(true), cl::Hidden,
                cl::desc("Run enzyme preprocessing optimizations"));

enum class DIFFE_TYPE {
  OUT_DIFF=0, // add differential to output struct
  DUP_ARG=1,  // duplicate the argument and store differential inside
  CONSTANT=2  // no differential
};

std::string tostring(DIFFE_TYPE t) {
    switch(t) {
      case DIFFE_TYPE::OUT_DIFF:
        return "OUT_DIFF";
      case DIFFE_TYPE::CONSTANT:
        return "CONSTANT";
      case DIFFE_TYPE::DUP_ARG:
        return "DUP_ARG";
      default:
        assert(0 && "illegal diffetype");
        return "";
    }
}

static inline FastMathFlags getFast() {
    FastMathFlags f;
    f.set();
    return f;
}

Instruction *getNextNonDebugInstruction(Instruction* Z) {
   for (Instruction *I = Z->getNextNode(); I; I = I->getNextNode())
     if (!isa<DbgInfoIntrinsic>(I))
       return I;
   return nullptr;
}

bool hasMetadata(const GlobalObject* O, StringRef kind) {
    return O->getMetadata(kind) != nullptr;
}

//note this doesn't handle recursive types!
static inline DIFFE_TYPE whatType(llvm::Type* arg) {
  if (arg->isPointerTy()) {
    switch(whatType(cast<llvm::PointerType>(arg)->getElementType())) {
      case DIFFE_TYPE::OUT_DIFF:
        return DIFFE_TYPE::DUP_ARG;
      case DIFFE_TYPE::CONSTANT:
        return DIFFE_TYPE::CONSTANT;
      case DIFFE_TYPE::DUP_ARG:
        return DIFFE_TYPE::DUP_ARG;
    }
    assert(arg);
    llvm::errs() << "arg: " << *arg << "\n";
    assert(0 && "Cannot handle type0");
    return DIFFE_TYPE::CONSTANT;
  } else if (arg->isArrayTy()) {
    return whatType(cast<llvm::ArrayType>(arg)->getElementType());
  } else if (arg->isStructTy()) {
    auto st = cast<llvm::StructType>(arg);
    if (st->getNumElements() == 0) return DIFFE_TYPE::CONSTANT;

    auto ty = DIFFE_TYPE::CONSTANT;
    for(unsigned i=0; i<st->getNumElements(); i++) {
      switch(whatType(st->getElementType(i))) {
        case DIFFE_TYPE::OUT_DIFF:
              switch(ty) {
                case DIFFE_TYPE::OUT_DIFF:
                case DIFFE_TYPE::CONSTANT:
                  ty = DIFFE_TYPE::OUT_DIFF;
                  break;
                case DIFFE_TYPE::DUP_ARG:
                  ty = DIFFE_TYPE::DUP_ARG;
                  return ty;
              }
        case DIFFE_TYPE::CONSTANT:
              switch(ty) {
                case DIFFE_TYPE::OUT_DIFF:
                  ty = DIFFE_TYPE::OUT_DIFF;
                  break;
                case DIFFE_TYPE::CONSTANT:
                  break;
                case DIFFE_TYPE::DUP_ARG:
                  ty = DIFFE_TYPE::DUP_ARG;
                  return ty;
              }
        case DIFFE_TYPE::DUP_ARG:
            return DIFFE_TYPE::DUP_ARG;
      }
    }

    return ty;
  } else if (arg->isIntOrIntVectorTy() || arg->isFunctionTy ()) {
    return DIFFE_TYPE::CONSTANT;
  } else if  (arg->isFPOrFPVectorTy()) {
    return DIFFE_TYPE::OUT_DIFF;
  } else {
    assert(arg);
    llvm::errs() << "arg: " << *arg << "\n";
    assert(0 && "Cannot handle type");
    return DIFFE_TYPE::CONSTANT;
  }
}

bool isReturned(Instruction *inst) {
	for (const auto &a:inst->users()) {
		if(isa<ReturnInst>(a))
			return true;
	}
	return false;
}

bool isIntASecretFloat(Value* val) {
    assert(val->getType()->isIntegerTy());

    if (isa<UndefValue>(val)) return true;
      
    if (auto cint = dyn_cast<ConstantInt>(val)) {
		if (!cint->isZero()) return false;
        assert(0 && "unsure if constant or not because constantint");
		 //if (cint->isOne()) return cint;
	}


    if (auto inst = dyn_cast<Instruction>(val)) {
        bool floatingUse = false;
        bool pointerUse = false;
        SmallPtrSet<Value*, 4> seen;

        std::function<void(Value*)> trackPointer = [&](Value* v) {
            if (seen.find(v) != seen.end()) return;
            seen.insert(v);
            llvm::errs() << " consider val " << *val << " for v " << * v << "\n";
                do { 
                    Type* let = cast<PointerType>(v->getType())->getElementType();
                    if (let->isFloatingPointTy()) {
                        floatingUse = true;
                    }
                    if (auto ci = dyn_cast<CastInst>(v)) {
                        if (auto cal = dyn_cast<CallInst>(ci->getOperand(0))) {
                            if (cal->getCalledFunction()->getName() == "malloc")
                                break;
                        }
                        v = ci->getOperand(0);
                        continue;
                    } 
                    if (auto gep = dyn_cast<GetElementPtrInst>(v)) {
                        v = gep->getOperand(0);
                        continue;
                    } 
                    if (auto phi = dyn_cast<PHINode>(v)) {
                        for(auto &a : phi->incoming_values()) {
                            trackPointer(a.get());
                        }
                        return;
                    }
                    break;
                } while(1);
                    
                Type* et = cast<PointerType>(v->getType())->getElementType();

                    do {
                        if (auto st = dyn_cast<CompositeType>(et)) {
                            et = st->getTypeAtIndex((unsigned int)0);
                            continue;
                        } 
                        break;
                    } while(1);
                    llvm::errs() << " for val " << *v  << *et << "\n";

                    if (et->isFloatingPointTy()) {
                        floatingUse = true;
                    }
                    if (et->isPointerTy()) {
                        pointerUse = true;
                    }
        };

        for(User* use: inst->users()) {
            if (auto ci = dyn_cast<BitCastInst>(use)) {
                if (ci->getDestTy()->isPointerTy()) {
                    pointerUse = true;
                    continue;
                }
                if (ci->getDestTy()->isFloatingPointTy()) {
                    floatingUse = true;
                    continue;
                }
            }
                
            
            if (isa<IntToPtrInst>(use)) {
                pointerUse = true;
                continue;
            }
            
            if (auto si = dyn_cast<StoreInst>(use)) {
                assert(inst == si->getValueOperand());

                trackPointer(si->getPointerOperand());
            }
        }

        if (auto li = dyn_cast<LoadInst>(inst)) {
            trackPointer(li->getOperand(0));
        }

        if (auto ci = dyn_cast<BitCastInst>(inst)) {
            if (ci->getSrcTy()->isPointerTy()) {
                pointerUse = true;
            }
            if (ci->getSrcTy()->isFloatingPointTy()) {
                floatingUse = true;
            }
        }
            
        
        if (isa<PtrToIntInst>(inst)) {
            pointerUse = true;
        }

        if (pointerUse && !floatingUse) return false; 
        if (!pointerUse && floatingUse) return true;
        llvm::errs() << *inst->getParent()->getParent() << "\n";
        llvm::errs() << " val:" << *val << " pointer:" << pointerUse << " floating:" << floatingUse << "\n";
        assert(0 && "ambiguous unsure if constant or not");
    }

    llvm::errs() << *val << "\n";
    assert(0 && "unsure if constant or not");
}
/*
bool isIntASecretPointer(Value* val, const DataLayout & DL) {
    assert(val->getType()->isIntegerTy());

    if ( DL.getTypeAllocSizeInBits(val->getType()) != Type::getInt8PtrTy(val->getContext()) )
        return false;

    if (isa<UndefValue>(val))false;
      
    if (auto cint = dyn_cast<ConstantInt>(val)) {
        assert(0 && "unsure if constant or not because constantint");
		 //if (cint->isZero()) return cint;
		 //if (cint->isOne()) return cint;
	}

    if (auto inst = dyn_cast<Instruction>(val)) {
        bool floatingUse = false;
        bool pointerUse = false;

        for(auto &use: inst->uses()) {
            if (auto ci = dyn_cast<BitCastInst>(val)) {
                if (ci->getDestTy()->isPointerTy()) {
                    pointerUse = true;
                    continue;
                }
                if (ci->getDestTy()->isFloatingPointTy()) {
                    floatingUse = true;
                    continue;
                }
            }
                
            
            if (isa<IntToPtrInst>(use))
                pointerUse = true;
                continue;
            }
            
            if (auto si = dyn_cast<StoreInst>(val)) {

                pointerUse = true;
                continue;
            }
        }

    }

    
    ConstantInt::get(size->getType(), allocationBuilder.GetInsertBlock()->getParent()->getParent()->getDataLayout().getTypeAllocSizeInBits(val->getType())/8), size, nullptr, val->getName()+"_malloccache");


    if (isa<Constant>(va
    for(
    assert(0 && "unsure if constant or not");
}
*/

void dumpSet(const SmallPtrSetImpl<Instruction*> &o) {
    llvm::errs() << "<begin dump>\n";
    for(auto a : o) llvm::errs() << *a << "\n";
    llvm::errs() << "</end dump>\n";
}

bool isconstantValueM(Value* val, SmallPtrSetImpl<Value*> &constants, SmallPtrSetImpl<Value*> &nonconstant, const SmallPtrSetImpl<Value*> &retvals, const SmallPtrSetImpl<Instruction*> &originalInstructions, uint8_t directions=3);

// TODO separate if the instruction is constant (i.e. could change things)
//    from if the value is constant (the value is something that could be differentiated)
bool isconstantM(Instruction* inst, SmallPtrSetImpl<Value*> &constants, SmallPtrSetImpl<Value*> &nonconstant, const SmallPtrSetImpl<Value*> &retvals, const SmallPtrSetImpl<Instruction*> &originalInstructions, uint8_t directions=3) {
    assert(inst);
	constexpr uint8_t UP = 1;
	constexpr uint8_t DOWN = 2;
	//assert(directions >= 0);
	assert(directions <= 3);
    if (isa<ReturnInst>(inst)) return true;

	if(isa<UnreachableInst>(inst) || isa<BranchInst>(inst) || (constants.find(inst) != constants.end()) || (originalInstructions.find(inst) == originalInstructions.end()) ) {
    	return true;
    }

    if((nonconstant.find(inst) != nonconstant.end())) {
        return false;
    }

	if (auto op = dyn_cast<CallInst>(inst)) {
		if(auto called = op->getCalledFunction()) {
			if (called->getName() == "printf" || called->getName() == "puts") {
			//if (called->getName() == "printf" || called->getName() == "puts" || called->getName() == "__assert_fail") {
				nonconstant.insert(inst);
				return false;
			}
		}
	}
	
    if (auto op = dyn_cast<CallInst>(inst)) {
		if(auto called = op->getCalledFunction()) {
			if (called->getName() == "__assert_fail" || called->getName() == "free" || called->getName() == "_ZdlPv" || called->getName() == "_ZdlPvm") {
				constants.insert(inst);
				return true;
			}
		}
	}
	
    if (auto op = dyn_cast<IntrinsicInst>(inst)) {
		switch(op->getIntrinsicID()) {
			case Intrinsic::assume:
			case Intrinsic::stacksave:
			case Intrinsic::stackrestore:
			case Intrinsic::lifetime_start:
			case Intrinsic::lifetime_end:
			case Intrinsic::dbg_addr:
			case Intrinsic::dbg_declare:
			case Intrinsic::dbg_value:
			case Intrinsic::invariant_start:
			case Intrinsic::invariant_end:
			case Intrinsic::var_annotation:
			case Intrinsic::ptr_annotation:
			case Intrinsic::annotation:
			case Intrinsic::codeview_annotation:
			case Intrinsic::expect:
			case Intrinsic::type_test:
			case Intrinsic::donothing:
			//case Intrinsic::is_constant:
				constants.insert(inst);
				return true;
			default:
				break;
		}
	}

	if (isa<CmpInst>(inst)) {
		constants.insert(inst);
		return true;
	}

    if (printconst)
	  llvm::errs() << "checking if is constant[" << (int)directions << "] " << *inst << "\n";

	SmallPtrSet<Value*, 20> constants_tmp;

    if (inst->getType()->isPointerTy()) {
		//Proceed assuming this is constant, can we prove this should be constant otherwise
		SmallPtrSet<Value*, 20> constants2;
		constants2.insert(constants.begin(), constants.end());
		SmallPtrSet<Value*, 20> nonconstant2;
		nonconstant2.insert(nonconstant.begin(), nonconstant.end());
		constants2.insert(inst);

		if (printconst)
			llvm::errs() << " < MEMSEARCH" << (int)directions << ">" << *inst << "\n";

		for (const auto &a:inst->users()) {
		  if(auto store = dyn_cast<StoreInst>(a)) {
			if (inst == store->getPointerOperand() && !isconstantValueM(store->getValueOperand(), constants2, nonconstant2, retvals, originalInstructions, directions)) {
				if (directions == 3)
				  nonconstant.insert(inst);
    			if (printconst)
				  llvm::errs() << "memory(" << (int)directions << ")  erase 1: " << *inst << "\n";
				return false;
			}
			if (inst == store->getValueOperand() && !isconstantValueM(store->getPointerOperand(), constants2, nonconstant2, retvals, originalInstructions, directions)) {
				if (directions == 3)
				  nonconstant.insert(inst);
    			if (printconst)
				  llvm::errs() << "memory(" << (int)directions << ")  erase 2: " << *inst << "\n";
				return false;
			}
		  } else if (isa<LoadInst>(a)) continue;
		  else {
			if (!isconstantM(cast<Instruction>(a), constants2, nonconstant2, retvals, originalInstructions, directions)) {
				if (directions == 3)
				  nonconstant.insert(inst);
    			if (printconst)
				  llvm::errs() << "memory(" << (int)directions << ") erase 3: " << *inst << " op " << *a << "\n";
				return false;
			}
		  }

		}
		
		if (printconst)
			llvm::errs() << " </MEMSEARCH" << (int)directions << ">" << *inst << "\n";
		
        constants_tmp.insert(constants2.begin(), constants2.end());
	}

	if (!inst->getType()->isPointerTy() && ( !inst->mayWriteToMemory() || isa<BinaryOperator>(inst) ) && (directions & DOWN) ) { 
		//Proceed assuming this is constant, can we prove this should be constant otherwise
		SmallPtrSet<Value*, 20> constants2;
		constants2.insert(constants.begin(), constants.end());
		SmallPtrSet<Value*, 20> nonconstant2;
		nonconstant2.insert(nonconstant.begin(), nonconstant.end());
		constants2.insert(inst);

		if (printconst)
			llvm::errs() << " < USESEARCH" << (int)directions << ">" << *inst << "\n";

		assert(!inst->mayWriteToMemory());
		assert(!isa<StoreInst>(inst));
		bool seenuse = false;
		for (const auto &a:inst->users()) {
			if (auto gep = dyn_cast<GetElementPtrInst>(a)) {
				assert(inst != gep->getPointerOperand());
				continue;
			}
			if (auto call = dyn_cast<CallInst>(a)) {
                auto fnp = call->getCalledFunction();
                if (fnp) {
                    auto fn = fnp->getName();
                    // todo realloc consider?
                    if (fn == "malloc" || fn == "_Znwm")
				        continue;
                    if (fnp->getIntrinsicID() == Intrinsic::memset && call->getArgOperand(0) != inst && call->getArgOperand(1) != inst)
                        continue;
                    if (fnp->getIntrinsicID() == Intrinsic::memcpy && call->getArgOperand(0) != inst && call->getArgOperand(1) != inst)
                        continue;
                }
			}

		  	if (!isconstantM(cast<Instruction>(a), constants2, nonconstant2, retvals, originalInstructions, DOWN)) {
    			if (printconst)
			      llvm::errs() << "nonconstant(" << (int)directions << ") inst (uses):" << *inst << " user " << *a << "\n";
				seenuse = true;
				break;
			} else {
               if (printconst)
			     llvm::errs() << "found constant(" << (int)directions << ")  inst use:" << *inst << " user " << *a << "\n";
			}
		}
		if (!seenuse) {
			constants.insert(inst);
			constants.insert(constants2.begin(), constants2.end());
            constants.insert(constants_tmp.begin(), constants_tmp.end());

			// not here since if had full updown might not have been nonconstant
			//nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
    		if (printconst)
			  llvm::errs() << "constant(" << (int)directions << ") inst (uses):" << *inst << "\n";
			return true;
		}
		
        if (printconst)
			llvm::errs() << " </USESEARCH" << (int)directions << ">" << *inst << "\n";
        constants_tmp.insert(constants2.begin(), constants2.end());
	}

	SmallPtrSet<Value*, 20> constants2;
	constants2.insert(constants.begin(), constants.end());
	SmallPtrSet<Value*, 20> nonconstant2;
	nonconstant2.insert(nonconstant.begin(), nonconstant.end());
	constants2.insert(inst);
		
	if (directions & UP) {
        if (printconst)
		    llvm::errs() << " < UPSEARCH" << (int)directions << ">" << *inst << "\n";
        if (auto gep = dyn_cast<GetElementPtrInst>(inst)) {
            // Handled uses above
            if (!isconstantValueM(gep->getPointerOperand(), constants2, nonconstant2, retvals, originalInstructions, UP)) {
                if (directions == 3)
                  nonconstant.insert(inst);
                if (printconst)
                  llvm::errs() << "nonconstant(" << (int)directions << ") gep " << *inst << " op " << *gep->getPointerOperand() << "\n";
                return false;
            }
            constants.insert(inst);
            constants.insert(constants2.begin(), constants2.end());
            constants.insert(constants_tmp.begin(), constants_tmp.end());
            //if (directions == 3)
            //  nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
            if (printconst)
              llvm::errs() << "constant(" << (int)directions << ") gep:" << *inst << "\n";
            return true;
        } else if (auto ci = dyn_cast<CallInst>(inst)) {
            for(auto& a: ci->arg_operands()) {
                if (!isconstantValueM(a, constants2, nonconstant2, retvals, originalInstructions, UP)) {
                    if (directions == 3)
                      nonconstant.insert(inst);
                    if (printconst)
                      llvm::errs() << "nonconstant(" << (int)directions << ")  call " << *inst << " op " << *a << "\n";
                    return false;
                }
            }

            constants.insert(inst);
            constants.insert(constants2.begin(), constants2.end());
            constants.insert(constants_tmp.begin(), constants_tmp.end());
            //if (directions == 3)
            //  nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
            if (printconst)
              llvm::errs() << "constant(" << (int)directions << ")  call:" << *inst << "\n";
            return true;
        } else {
            for(auto& a: inst->operands()) {
                if (!isconstantValueM(a, constants2, nonconstant2, retvals, originalInstructions, UP)) {
                    if (directions == 3)
                      nonconstant.insert(inst);
                    if (printconst)
                      llvm::errs() << "nonconstant(" << (int)directions << ")  inst " << *inst << " op " << *a << "\n";
                    return false;
                }
            }

            constants.insert(inst);
            constants.insert(constants2.begin(), constants2.end());
            constants.insert(constants_tmp.begin(), constants_tmp.end());
            //if (directions == 3)
            //  nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
            if (printconst)
              llvm::errs() << "constant(" << (int)directions << ")  inst:" << *inst << "\n";
            return true;
        }
        if (printconst)
		    llvm::errs() << " </UPSEARCH" << (int)directions << ">" << *inst << "\n";
	}


    if (directions == 3)
	  nonconstant.insert(inst);
    if (printconst)
	  llvm::errs() << "couldnt decide nonconstants(" << (int)directions << "):" << *inst << "\n";
	return false;
}

// TODO separate if the instruction is constant (i.e. could change things)
//    from if the value is constant (the value is something that could be differentiated)
bool isconstantValueM(Value* val, SmallPtrSetImpl<Value*> &constants, SmallPtrSetImpl<Value*> &nonconstant, const SmallPtrSetImpl<Value*> &retvals, const SmallPtrSetImpl<Instruction*> &originalInstructions, uint8_t directions) {
    assert(val);
	//constexpr uint8_t UP = 1;
	constexpr uint8_t DOWN = 2;
	//assert(directions >= 0);
	assert(directions <= 3);
    
    if (val->getType()->isVoidTy()) return true;

    //! False so we can replace function with augmentation
    if (isa<Function>(val)) {
        return false;
    }

    if (isa<ConstantData>(val) || isa<ConstantAggregate>(val) || isa<Function>(val)) return true;
	if (isa<BasicBlock>(val)) return true;
    assert(!isa<InlineAsm>(val));

    if((constants.find(val) != constants.end())) {
        return true;
    }
    if((retvals.find(val) != retvals.end())) {
        if (printconst) {
		    llvm::errs() << " VALUE nonconst from retval " << *val << "\n";
        }
        return false;
    }

    //All arguments should be marked constant/nonconstant ahead of time
    if (isa<Argument>(val)) {
        if((nonconstant.find(val) != nonconstant.end())) {
		    if (printconst)
		      llvm::errs() << " VALUE nonconst from arg nonconst " << *val << "\n";
            return false;
        }
        assert(0 && "must've put arguments in constant/nonconstant");
    }
    
    if (auto inst = dyn_cast<Instruction>(val)) {
        if (isconstantM(inst, constants, nonconstant, retvals, originalInstructions, directions)) return true;
    }
	
    if (!val->getType()->isPointerTy() && (directions & DOWN) ) { 
		auto &constants2 = constants;
		auto &nonconstant2 = nonconstant;

		if (printconst)
			llvm::errs() << " <Value USESEARCH" << (int)directions << ">" << *val << "\n";

		bool seenuse = false;
		
        for (const auto &a:val->users()) {
		    if (printconst)
			  llvm::errs() << "      considering use of " << *val << " - " << *a << "\n";

			if (auto gep = dyn_cast<GetElementPtrInst>(a)) {
				assert(val != gep->getPointerOperand());
				continue;
			}
			if (auto call = dyn_cast<CallInst>(a)) {
                auto fnp = call->getCalledFunction();
                if (fnp) {
                    auto fn = fnp->getName();
                    // todo realloc consider?
                    if (fn == "malloc" || fn == "_Znwm")
				        continue;
                    if (fnp->getIntrinsicID() == Intrinsic::memset && call->getArgOperand(0) != val && call->getArgOperand(1) != val)
                        continue;
                    if (fnp->getIntrinsicID() == Intrinsic::memcpy && call->getArgOperand(0) != val && call->getArgOperand(1) != val)
                        continue;
                }
			}
            
		  	if (!isconstantM(cast<Instruction>(a), constants2, nonconstant2, retvals, originalInstructions, DOWN)) {
    			if (printconst)
			      llvm::errs() << "Value nonconstant inst (uses):" << *val << " user " << *a << "\n";
				seenuse = true;
				break;
			} else {
               if (printconst)
			     llvm::errs() << "Value found constant inst use:" << *val << " user " << *a << "\n";
			}
		}

		if (!seenuse) {
    		if (printconst)
			  llvm::errs() << "Value constant inst (uses):" << *val << "\n";
			return true;
		}
		
        if (printconst)
			llvm::errs() << " </Value USESEARCH" << (int)directions << ">" << *val << "\n";
	}

    if (printconst)
	   llvm::errs() << " Value nonconstant (couldn't disprove)[" << (int)directions << "]" << *val << "\n";
    return false;
}

 static bool promoteMemoryToRegister(Function &F, DominatorTree &DT,
                                     AssumptionCache &AC) {
   std::vector<AllocaInst *> Allocas;
   BasicBlock &BB = F.getEntryBlock(); // Get the entry node for the function
   bool Changed = false;
 
   while (true) {
     Allocas.clear();
 
     // Find allocas that are safe to promote, by looking at all instructions in
     // the entry node
     for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
       if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) // Is it an alloca?
         if (isAllocaPromotable(AI))
           Allocas.push_back(AI);
 
     if (Allocas.empty())
       break;
 
     PromoteMemToReg(Allocas, DT, &AC);
     Changed = true;
   }
   return Changed;
 }

enum class ReturnType {
    ArgsWithReturn, Args, TapeAndReturns
};

void forceRecursiveInlining(Function *NewF, const Function* F) {
   int count = 0;
   remover:
     SmallPtrSet<Instruction*, 10> originalInstructions;
     for (inst_iterator I = inst_begin(NewF), E = inst_end(NewF); I != E; ++I) {
         originalInstructions.insert(&*I);
     }
   for (inst_iterator I = inst_begin(NewF), E = inst_end(NewF); I != E; ++I)
     if (auto call = dyn_cast<CallInst>(&*I)) {
        //if (isconstantM(call, constants, nonconstant, returnvals, originalInstructions)) continue;
        if (call->getCalledFunction() == nullptr) continue;
        if (call->getCalledFunction()->empty()) continue;
        /*
        if (call->getCalledFunction()->hasFnAttribute(Attribute::NoInline)) {
            llvm::errs() << "can't inline noinline " << call->getCalledFunction()->getName() << "\n";
            continue;
        }
        */
        if (call->getCalledFunction()->hasFnAttribute(Attribute::ReturnsTwice)) continue;
        if (call->getCalledFunction() == F || call->getCalledFunction() == NewF) {
            llvm::errs() << "can't inline recursive " << call->getCalledFunction()->getName() << "\n";
            continue;
        }
        llvm::errs() << "inlining " << call->getCalledFunction()->getName() << "\n";
        InlineFunctionInfo IFI;
        InlineFunction(call, IFI);
        count++;
        if (count >= autodiff_inline_count)
            break;
        else
          goto remover;
     }
}

class GradientUtils;

PHINode* canonicalizeIVs(Type *Ty, Loop *L, ScalarEvolution &SE, DominatorTree &DT, GradientUtils* gutils);

Function* preprocessForClone(Function *F, AAResults &AA, TargetLibraryInfo &TLI) {
 static std::map<Function*,Function*> cache;
 if (cache.find(F) != cache.end()) return cache[F];

 Function *NewF = Function::Create(F->getFunctionType(), F->getLinkage(), "preprocess_" + F->getName(), F->getParent());
 
 ValueToValueMapTy VMap;
 for (auto i=F->arg_begin(), j=NewF->arg_begin(); i != F->arg_end(); ) {
     VMap[i] = j;
     j->setName(i->getName());
     i++;
     j++;
 }

 SmallVector <ReturnInst*,4> Returns;
 CloneFunctionInto(NewF, F, VMap, F->getSubprogram() != nullptr, Returns, "",
                   nullptr);
 NewF->setAttributes(F->getAttributes());

 {
    FunctionAnalysisManager AM;
 AM.registerPass([] { return LoopAnalysis(); });
 AM.registerPass([] { return DominatorTreeAnalysis(); });
 AM.registerPass([] { return ScalarEvolutionAnalysis(); });
 AM.registerPass([] { return AssumptionAnalysis(); });

#if LLVM_VERSION_MAJOR >= 8
 AM.registerPass([] { return PassInstrumentationAnalysis(); });
#endif
    LoopSimplifyPass().run(*NewF, AM);

 }

 if (enzyme_preopt) {

  if(autodiff_inline) {
      llvm::errs() << "running inlining process\n";
      forceRecursiveInlining(NewF, F);

      {
         DominatorTree DT(*NewF);
         AssumptionCache AC(*NewF);
         promoteMemoryToRegister(*NewF, DT, AC);
      }

      {
         FunctionAnalysisManager AM;
         AM.registerPass([] { return AAManager(); });
         AM.registerPass([] { return ScalarEvolutionAnalysis(); });
         AM.registerPass([] { return AssumptionAnalysis(); });
         AM.registerPass([] { return TargetLibraryAnalysis(); });
         AM.registerPass([] { return TargetIRAnalysis(); });
         AM.registerPass([] { return MemorySSAAnalysis(); });
         AM.registerPass([] { return DominatorTreeAnalysis(); });
         AM.registerPass([] { return MemoryDependenceAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
        AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
         AM.registerPass([] { return LazyValueAnalysis(); });

        GVN().run(*NewF, AM);

        SROA().run(*NewF, AM);
      }
 }

 bool repeat = false;
 do {
     repeat = false;
 for(auto& BB: *NewF) {
 for(Instruction &I : BB) {
    if (auto bc = dyn_cast<BitCastInst>(&I)) {
        if (auto bc2 = dyn_cast<BitCastInst>(bc->getOperand(0))) {
            if (bc2->getNumUses() == 1) {
                IRBuilder<> b(bc2);
                auto c = b.CreateBitCast(bc2->getOperand(0), I.getType());
                bc->replaceAllUsesWith(c);
                bc->eraseFromParent();
                bc2->eraseFromParent();
                repeat = true;
                break;
            }
        } else if (auto pt = dyn_cast<PointerType>(bc->getOperand(0)->getType())) {
          if (auto st = dyn_cast<StructType>(pt->getElementType())) {
          
          if (auto pt2 = dyn_cast<PointerType>(bc->getType())) {
            if (st->getNumElements() && st->getElementType(0) == pt2->getElementType()) {
                IRBuilder<> b(bc);
                auto c = b.CreateGEP(bc->getOperand(0), {
                        ConstantInt::get(Type::getInt64Ty(I.getContext()), 0), 
                        ConstantInt::get(Type::getInt32Ty(I.getContext()), 0),
                        });
                bc->replaceAllUsesWith(c);
                bc->eraseFromParent();
                repeat = true;
                break;
            }}
          }
        }
    }
 }
 if (repeat) break;
 }
 } while(repeat);

 {
     FunctionAnalysisManager AM;
     AM.registerPass([] { return AAManager(); });
     AM.registerPass([] { return ScalarEvolutionAnalysis(); });
     AM.registerPass([] { return AssumptionAnalysis(); });
     AM.registerPass([] { return TargetLibraryAnalysis(); });
     AM.registerPass([] { return TargetIRAnalysis(); });
     AM.registerPass([] { return MemorySSAAnalysis(); });
     AM.registerPass([] { return DominatorTreeAnalysis(); });
     AM.registerPass([] { return MemoryDependenceAnalysis(); });
     AM.registerPass([] { return LoopAnalysis(); });
     AM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
     AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
#if LLVM_VERSION_MAJOR >= 8
 AM.registerPass([] { return PassInstrumentationAnalysis(); });
#endif
     AM.registerPass([] { return LazyValueAnalysis(); });
     InstCombinePass().run(*NewF, AM);
#if LLVM_VERSION_MAJOR > 6
     InstSimplifyPass().run(*NewF, AM);
#endif
     InstCombinePass().run(*NewF, AM);
     
     EarlyCSEPass(/*memoryssa*/true).run(*NewF, AM);
     
     GVN().run(*NewF, AM);
     SROA().run(*NewF, AM);

     CorrelatedValuePropagationPass().run(*NewF, AM);

     DCEPass().run(*NewF, AM);
 }

 do {
     repeat = false;
 for(Instruction &I : NewF->getEntryBlock()) {
    if (auto ci = dyn_cast<ICmpInst>(&I)) {
        for(Instruction &J : *ci->getParent()) {
            if (&J == &I) break;
            if (auto ci2 = dyn_cast<ICmpInst>(&J)) {
                if (  (ci->getPredicate() == ci2->getInversePredicate()) &&
                        (
                         ( ci->getOperand(0) == ci2->getOperand(0) && ci->getOperand(1) == ci2->getOperand(1) ) 
                         || 
                         ( (ci->isEquality() || ci2->isEquality()) && ci->getOperand(0) == ci2->getOperand(1) && ci->getOperand(1) == ci2->getOperand(0) ) 
                            ) ) {
                    IRBuilder<> b(ci);
                    Value* tonot = ci2;
                    for(User* a : ci2->users()) {
                        if (auto ii = dyn_cast<IntrinsicInst>(a)) {
                            if (ii->getIntrinsicID() == Intrinsic::assume) {
                                tonot = ConstantInt::getTrue(ii->getContext());
                                break;
                            }
                        }
                    }
                    auto c = b.CreateNot(tonot);
                    
                    ci->replaceAllUsesWith(c);
                    ci->eraseFromParent();
                    repeat = true;
                    break;
                }
            }
        }
        if (repeat) break;
    }
 }
 } while(repeat);
  

 {
     FunctionAnalysisManager AM;
     AM.registerPass([] { return AAManager(); });
     AM.registerPass([] { return ScalarEvolutionAnalysis(); });
     AM.registerPass([] { return AssumptionAnalysis(); });
     AM.registerPass([] { return TargetLibraryAnalysis(); });
     AM.registerPass([] { return TargetIRAnalysis(); });
     AM.registerPass([] { return MemorySSAAnalysis(); });
     AM.registerPass([] { return DominatorTreeAnalysis(); });
     AM.registerPass([] { return MemoryDependenceAnalysis(); });
     AM.registerPass([] { return LoopAnalysis(); });
     AM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
     AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
#if LLVM_VERSION_MAJOR >= 8
 AM.registerPass([] { return PassInstrumentationAnalysis(); });
#endif
     AM.registerPass([] { return LazyValueAnalysis(); });
 
     DSEPass().run(*NewF, AM);
 }

 {
    FunctionAnalysisManager AM;
     AM.registerPass([] { return AAManager(); });
     AM.registerPass([] { return ScalarEvolutionAnalysis(); });
     AM.registerPass([] { return AssumptionAnalysis(); });
     AM.registerPass([] { return TargetLibraryAnalysis(); });
     AM.registerPass([] { return TargetIRAnalysis(); });
     AM.registerPass([] { return MemorySSAAnalysis(); });
     AM.registerPass([] { return DominatorTreeAnalysis(); });
     AM.registerPass([] { return MemoryDependenceAnalysis(); });
     AM.registerPass([] { return LoopAnalysis(); });
     AM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
     AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
#if LLVM_VERSION_MAJOR >= 8
 AM.registerPass([] { return PassInstrumentationAnalysis(); });
#endif
     AM.registerPass([] { return LazyValueAnalysis(); });
    LoopAnalysisManager LAM;
     AM.registerPass([&] { return LoopAnalysisManagerFunctionProxy(LAM); });
     LAM.registerPass([&] { return FunctionAnalysisManagerLoopProxy(AM); });

 SimplifyCFGOptions scfgo(/*unsigned BonusThreshold=*/1, /*bool ForwardSwitchCond=*/false, /*bool SwitchToLookup=*/false, /*bool CanonicalLoops=*/true, /*bool SinkCommon=*/true, /*AssumptionCache *AssumpCache=*/nullptr);
 SimplifyCFGPass(scfgo).run(*NewF, AM);
 LoopSimplifyPass().run(*NewF, AM);
  
 if (autodiff_inline) {
     createFunctionToLoopPassAdaptor(LoopIdiomRecognizePass()).run(*NewF, AM);
 }
 DSEPass().run(*NewF, AM);   
 LoopSimplifyPass().run(*NewF, AM);

 } 
 }

 {
    FunctionAnalysisManager AM;
     AM.registerPass([] { return AAManager(); });
     AM.registerPass([] { return ScalarEvolutionAnalysis(); });
     AM.registerPass([] { return AssumptionAnalysis(); });
     AM.registerPass([] { return TargetLibraryAnalysis(); });
     AM.registerPass([] { return TargetIRAnalysis(); });
     AM.registerPass([] { return LoopAnalysis(); });
     AM.registerPass([] { return MemorySSAAnalysis(); });
     AM.registerPass([] { return DominatorTreeAnalysis(); });
     AM.registerPass([] { return MemoryDependenceAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
     AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
#if LLVM_VERSION_MAJOR >= 8
 AM.registerPass([] { return PassInstrumentationAnalysis(); });
#endif
     
     ModuleAnalysisManager MAM;
     AM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });
     MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(AM); });

 BasicAA ba;
 auto baa = new BasicAAResult(ba.run(*NewF, AM));
 AA.addAAResult(*baa);
 
 ScopedNoAliasAA sa;
 auto saa = new ScopedNoAliasAAResult(sa.run(*NewF, AM));
 AA.addAAResult(*saa);

 }

  if (autodiff_print)
      llvm::errs() << "after simplification :\n" << *NewF << "\n";
  
  if (llvm::verifyFunction(*NewF, &llvm::errs())) {
      llvm::errs() << *NewF << "\n";
      report_fatal_error("function failed verification (1)");
  }
  cache[F] = NewF;
  return NewF;
}

Function *CloneFunctionWithReturns(Function *&F, AAResults &AA, TargetLibraryInfo &TLI, ValueToValueMapTy& ptrInputs, const std::set<unsigned>& constant_args, SmallPtrSetImpl<Value*> &constants, SmallPtrSetImpl<Value*> &nonconstant, SmallPtrSetImpl<Value*> &returnvals, ReturnType returnValue, bool differentialReturn, Twine name, ValueToValueMapTy *VMapO, bool diffeReturnArg, llvm::Type* additionalArg = nullptr) {
 assert(!F->empty());
 F = preprocessForClone(F, AA, TLI);
 diffeReturnArg &= differentialReturn;
 std::vector<Type*> RetTypes;
 if (returnValue == ReturnType::ArgsWithReturn)
   RetTypes.push_back(F->getReturnType());
 std::vector<Type*> ArgTypes;

 ValueToValueMapTy VMap;

 // The user might be deleting arguments to the function by specifying them in
 // the VMap.  If so, we need to not add the arguments to the arg ty vector
 //
 unsigned argno = 0;
 for (const Argument &I : F->args()) {
     ArgTypes.push_back(I.getType());
     if (constant_args.count(argno)) {
        argno++;
        continue;
     }
     if (I.getType()->isPointerTy() || I.getType()->isIntegerTy()) {
        ArgTypes.push_back(I.getType());
        /*
        if (I.getType()->isPointerTy() && !(I.hasAttribute(Attribute::ReadOnly) || I.hasAttribute(Attribute::ReadNone) ) ) {
          llvm::errs() << "Cannot take derivative of function " <<F->getName()<< " input argument to function " << I.getName() << " is not marked read-only\n";
          exit(1);
        }
        */
     } else { 
       RetTypes.push_back(I.getType());
     }
     argno++;
 }

 if (diffeReturnArg && !F->getReturnType()->isPointerTy() && !F->getReturnType()->isIntegerTy()) {
    assert(!F->getReturnType()->isVoidTy());
    ArgTypes.push_back(F->getReturnType());
 }
 if (additionalArg) {
    ArgTypes.push_back(additionalArg);
 }
 Type* RetType = StructType::get(F->getContext(), RetTypes);
 if (returnValue == ReturnType::TapeAndReturns) {
     RetTypes.clear();
     RetTypes.push_back(Type::getInt8PtrTy(F->getContext()));
  if (!F->getReturnType()->isVoidTy()) {
    RetTypes.push_back(F->getReturnType());
    if (F->getReturnType()->isPointerTy() || F->getReturnType()->isIntegerTy())
      RetTypes.push_back(F->getReturnType());
  }
    RetType = StructType::get(F->getContext(), RetTypes);
 }

 // Create a new function type...
 FunctionType *FTy = FunctionType::get(RetType,
                                   ArgTypes, F->getFunctionType()->isVarArg());

 // Create the new function...
 Function *NewF = Function::Create(FTy, F->getLinkage(), name, F->getParent());
 if (diffeReturnArg && !F->getReturnType()->isPointerTy() && !F->getReturnType()->isIntegerTy()) {
    auto I = NewF->arg_end();
    I--;
    if(additionalArg)
        I--;
    I->setName("differeturn");
 }
 if (additionalArg) {
    auto I = NewF->arg_end();
    I--;
    I->setName("tapeArg");
 }

 bool hasPtrInput = false;

 unsigned ii = 0, jj = 0;
 for (auto i=F->arg_begin(), j=NewF->arg_begin(); i != F->arg_end(); ) {
   bool isconstant = (constant_args.count(ii) > 0);

   if (isconstant) {
      constants.insert(j);
      if (printconst)
        llvm::errs() << "in new function " << NewF->getName() << " constant arg " << *j << "\n";
   } else {
	  nonconstant.insert(j);
      if (printconst)
        llvm::errs() << "in new function " << NewF->getName() << " nonconstant arg " << *j << "\n";
   }

   if (!isconstant && ( i->getType()->isPointerTy() || i->getType()->isIntegerTy()) ) {
     VMap[i] = j;
     hasPtrInput = true;
     ptrInputs[j] = (j+1);
     if (F->hasParamAttribute(ii, Attribute::NoCapture)) {
       NewF->addParamAttr(jj, Attribute::NoCapture);
       NewF->addParamAttr(jj+1, Attribute::NoCapture);
     }
     if (F->hasParamAttribute(ii, Attribute::NoAlias)) {
       NewF->addParamAttr(jj, Attribute::NoAlias);
       NewF->addParamAttr(jj+1, Attribute::NoAlias);
     }

     j->setName(i->getName());
     j++;
     j->setName(i->getName()+"'");
	 nonconstant.insert(j);
     j++;
     jj+=2;

     i++;
     ii++;

   } else {
     VMap[i] = j;
     j->setName(i->getName());

     j++;
     jj++;
     i++;
     ii++;
   }
 }

 // Loop over the arguments, copying the names of the mapped arguments over...
 Function::arg_iterator DestI = NewF->arg_begin();


 for (const Argument & I : F->args())
   if (VMap.count(&I) == 0) {     // Is this argument preserved?
     DestI->setName(I.getName()); // Copy the name over...
     VMap[&I] = &*DestI++;        // Add mapping to VMap
   }
 SmallVector <ReturnInst*,4> Returns;
 CloneFunctionInto(NewF, F, VMap, F->getSubprogram() != nullptr, Returns, "",
                   nullptr);
 if (VMapO) VMapO->insert(VMap.begin(), VMap.end());

 if (hasPtrInput) {
    if (NewF->hasFnAttribute(Attribute::ReadNone)) {
    NewF->removeFnAttr(Attribute::ReadNone);
    }
    if (NewF->hasFnAttribute(Attribute::ReadOnly)) {
    NewF->removeFnAttr(Attribute::ReadOnly);
    }
 }
 NewF->setLinkage(Function::LinkageTypes::InternalLinkage);
 assert(NewF->hasLocalLinkage());

 if (differentialReturn) {
   for(auto& r : Returns) {
     if (auto a = r->getReturnValue()) {
       nonconstant.insert(a);
       returnvals.insert(a);
       if (printconst)
         llvm::errs() << "in new function " << NewF->getName() << " nonconstant retval " << *a << "\n";
     }
   }
 }

 //SmallPtrSet<Value*,4> constants2;
 //for (auto a :constants){
 //   constants2.insert(a);
// }
 //for (auto a :nonconstant){
 //   nonconstant2.insert(a);
 //}
 
 return NewF;
}

#include "llvm/IR/Constant.h"
#include <deque>
#include "llvm/IR/CFG.h"

bool shouldRecompute(Value* val, const ValueToValueMapTy& available) {
          if (available.count(val)) return false;
          if (isa<Argument>(val) || isa<Constant>(val) ) {
            return false;
          } else if (auto op = dyn_cast<CastInst>(val)) {
            return shouldRecompute(op->getOperand(0), available);
          } else if (isa<AllocaInst>(val)) {
            return true;
          } else if (auto op = dyn_cast<BinaryOperator>(val)) {
            bool a0 = shouldRecompute(op->getOperand(0), available);
            if (a0) {
                //llvm::errs() << "need recompute: " << *op->getOperand(0) << "\n";
            }
            bool a1 = shouldRecompute(op->getOperand(1), available);
            if (a1) {
                //llvm::errs() << "need recompute: " << *op->getOperand(1) << "\n";
            }
            return a0 || a1;
          } else if (auto op = dyn_cast<CmpInst>(val)) {
            return shouldRecompute(op->getOperand(0), available) || shouldRecompute(op->getOperand(1), available);
          } else if (auto op = dyn_cast<SelectInst>(val)) {
            return shouldRecompute(op->getOperand(0), available) || shouldRecompute(op->getOperand(1), available) || shouldRecompute(op->getOperand(2), available);
          } else if (auto load = dyn_cast<LoadInst>(val)) {
                Value* idx = load->getOperand(0);
                while (!isa<Argument>(idx)) {
                    if (auto gep = dyn_cast<GetElementPtrInst>(idx)) {
                        for(auto &a : gep->indices()) {
                            if (shouldRecompute(a, available)) {
                                //llvm::errs() << "not recomputable: " << *a << "\n";
                                return true;
                            }
                        }
                        idx = gep->getPointerOperand();
                    } else if(auto cast = dyn_cast<CastInst>(idx)) {
                        idx = cast->getOperand(0);
                    } else if(isa<CallInst>(idx)) {
                    //} else if(auto call = dyn_cast<CallInst>(idx)) {
                        //if (call->getCalledFunction()->getName() == "malloc")
                        //    return false;
                        //else
                        {
                            //llvm::errs() << "unknown call " << *call << "\n";
                            return true;
                        }
                    } else {
                      //llvm::errs() << "not a gep " << *idx << "\n";
                      return true;
                    }
                }
                Argument* arg = cast<Argument>(idx);
                if (! ( arg->hasAttribute(Attribute::ReadOnly) || arg->hasAttribute(Attribute::ReadNone)) ) {
                    //llvm::errs() << "argument " << *arg << " not marked read only\n";
                    return true;
                }
                return false;
          } else if (auto phi = dyn_cast<PHINode>(val)) {
            if (phi->getNumIncomingValues () == 1) {
                bool b = shouldRecompute(phi->getIncomingValue(0) , available);
                if (b) {
                    //llvm::errs() << "phi need recompute: " <<*phi->getIncomingValue(0) << "\n";
                }
                return b;
            }

            return true;
          } else if (auto op = dyn_cast<IntrinsicInst>(val)) {
            switch(op->getIntrinsicID()) {
                case Intrinsic::sin:
                case Intrinsic::cos:
                    return false;
                    return shouldRecompute(op->getOperand(0), available);
                default:
                    return true;
            }
        }

          //llvm::errs() << "unknown inst " << *val << " unable to recompute\n";
          return true;
}

    Type* FloatToIntTy(Type* T) {
        assert(T->isFPOrFPVectorTy());
        if (auto ty = dyn_cast<VectorType>(T)) {
            return VectorType::get(FloatToIntTy(ty->getElementType()), ty->getNumElements());
        }
        if (T->isHalfTy()) return IntegerType::get(T->getContext(), 16); 
        if (T->isFloatTy()) return IntegerType::get(T->getContext(), 32); 
        if (T->isDoubleTy()) return IntegerType::get(T->getContext(), 64);
        assert(0 && "unknown floating point type");
        return nullptr;
    }

    Type* IntToFloatTy(Type* T) {
        assert(T->isIntOrIntVectorTy());
        if (auto ty = dyn_cast<VectorType>(T)) {
            return VectorType::get(IntToFloatTy(ty->getElementType()), ty->getNumElements());
        }
        if (auto ty = dyn_cast<IntegerType>(T)) {
            switch(ty->getBitWidth()) {
                case 16: return Type::getHalfTy(T->getContext());
                case 32: return Type::getFloatTy(T->getContext());
                case 64: return Type::getDoubleTy(T->getContext());
            }
        }
        assert(0 && "unknown int to floating point type");
        return nullptr;
    }

typedef struct {
  PHINode* var;
  PHINode* antivar;
  BasicBlock* latch;
  BasicBlock* header;
  BasicBlock* preheader;
  bool dynamic;
  //limit is last value, iters is number of iters (thus iters = limit + 1)
  Value* limit;
  BasicBlock* exit;
  Loop* parent;
} LoopContext;

bool operator==(const LoopContext& lhs, const LoopContext &rhs) {
    return lhs.parent == rhs.parent;
}

bool getContextM(BasicBlock *BB, LoopContext &loopContext, std::map<Loop*,LoopContext> &loopContexts, LoopInfo &LI,ScalarEvolution &SE,DominatorTree &DT, GradientUtils &gutils);

bool isCertainMallocOrFree(Function* called) {
    if (called == nullptr) return false;
    if (called->getName() == "printf" || called->getName() == "puts" || called->getName() == "malloc" || called->getName() == "_Znwm" || called->getName() == "_ZdlPv" || called->getName() == "_ZdlPvm" || called->getName() == "free") return true;
    switch(called->getIntrinsicID()) {
            case Intrinsic::dbg_declare:
            case Intrinsic::dbg_value:
            #if LLVM_VERSION_MAJOR > 6
            case Intrinsic::dbg_label:
            #endif
            case Intrinsic::dbg_addr:
            case Intrinsic::lifetime_start:
            case Intrinsic::lifetime_end:
                return true;
            default:
                break;
    }

    return false;
}

bool isCertainPrintOrFree(Function* called) {
    if (called == nullptr) return false;
    
    if (called->getName() == "printf" || called->getName() == "puts" || called->getName() == "_ZdlPv" || called->getName() == "_ZdlPvm" || called->getName() == "free") return true;
    switch(called->getIntrinsicID()) {
            case Intrinsic::dbg_declare:
            case Intrinsic::dbg_value:
            #if LLVM_VERSION_MAJOR > 6
            case Intrinsic::dbg_label:
            #endif
            case Intrinsic::dbg_addr:
            case Intrinsic::lifetime_start:
            case Intrinsic::lifetime_end:
                return true;
            default:
                break;
    }
    return false;
}

bool isCertainPrintMallocOrFree(Function* called) {
    if (called == nullptr) return false;
    
    if (called->getName() == "printf" || called->getName() == "puts" || called->getName() == "malloc" || called->getName() == "_Znwm" || called->getName() == "_ZdlPv" || called->getName() == "_ZdlPvm" || called->getName() == "free") return true;
    switch(called->getIntrinsicID()) {
            case Intrinsic::dbg_declare:
            case Intrinsic::dbg_value:
            #if LLVM_VERSION_MAJOR > 6
            case Intrinsic::dbg_label:
            #endif
            case Intrinsic::dbg_addr:
            case Intrinsic::lifetime_start:
            case Intrinsic::lifetime_end:
                return true;
            default:
                break;
    }
    return false;
}

Function* CreatePrimalAndGradient(Function* todiff, const std::set<unsigned>& constant_args, TargetLibraryInfo &TLI, AAResults &AA, bool returnValue, bool differentialReturn, bool topLevel, llvm::Type* additionalArg);

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
            v.first->dump();
            I->dump();
            assert(0 && "erasing something in lastScopeAlloc map");
        }
    }
    for(auto v: scopeMap) {
        if (v.second == I) {
            newFunc->dump();
            dumpScope();
            v.first->dump();
            I->dump();
            assert(0 && "erasing something in scope map");
        }
    }
    for(auto v: scopeFrees) {
        if (v.second == I) {
            v.first->dump();
            I->dump();
            assert(0 && "erasing something in scopeFrees map");
        }
    }
    for(auto v: invertedPointers) {
        if (v.second == I) {
            newFunc->dump();
            dumpPointers();
            v.first->dump();
            I->dump();
            assert(0 && "erasing something in invertedPointers map");
        }
    }
    if (!I->use_empty()) {
        newFunc->dump();
        I->dump();
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

            PHINode* phi = BuilderQ.CreatePHI(cast<PointerType>(ret->getType())->getElementType(), 1);
            if (malloc) assert(phi->getType() == malloc->getType());
            
            assert(scopeMap.find(phi) == scopeMap.end());
            scopeMap[phi] = entryBuilder.CreateAlloca(ret->getType(), nullptr, phi->getName()+"_mdyncache_fromtape");
            entryBuilder.CreateStore(ret, scopeMap[phi]);

            auto v = lookupM(phi, BuilderQ, /*forceLookup*/true);
            assert(v != phi);
            if (malloc) {
                assert(v->getType() == malloc->getType());
            }
            scopeMap[v] = scopeMap[phi];
            scopeMap.erase(phi);
            originalInstructions.erase(ret);
            erase(phi);

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
                newFunc->dump();
                llvm::errs() << *scopeFrees[malloc] << "\n";
            }
            assert(scopeFrees.find(malloc) == scopeFrees.end());
            if (lastScopeAlloc.find(malloc) != lastScopeAlloc.end()) {
                newFunc->dump();
                llvm::errs() << *lastScopeAlloc[malloc] << "\n";
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
  static GradientUtils* CreateFromClone(Function *todiff, AAResults &AA, TargetLibraryInfo &TLI, const std::set<unsigned> & constant_args, ReturnType returnValue, bool differentialReturn, llvm::Type* additionalArg=nullptr) {
    assert(!todiff->empty());
    ValueToValueMapTy invertedPointers;
    SmallPtrSet<Value*,4> constants;
    SmallPtrSet<Value*,20> nonconstant;
    SmallPtrSet<Value*,2> returnvals;
    ValueToValueMapTy originalToNew;
    auto newFunc = CloneFunctionWithReturns(todiff, AA, TLI, invertedPointers, constant_args, constants, nonconstant, returnvals, /*returnValue*/returnValue, /*differentialReturn*/differentialReturn, "fakeaugmented_"+todiff->getName(), &originalToNew, /*diffeReturnArg*/false, additionalArg);
    auto res = new GradientUtils(newFunc, AA, TLI, invertedPointers, constants, nonconstant, returnvals, originalToNew);
    res->oldFunc = todiff;
    return res;
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
    report_fatal_error("could not find original block for given reverse block");
  }

  void forceContexts() {
    LoopContext lc;
    for(auto BB : originalBlocks) {
        getContext(BB, lc);
    }
  }
 
  bool getContext(BasicBlock* BB, LoopContext& loopContext) {
    return getContextM(BB, loopContext, this->loopContexts, this->LI, this->SE, this->DT, *this);
  }

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

    void ensureLookupCached(Instruction* inst, bool shouldFree=true) {
        if (scopeMap.find(inst) != scopeMap.end()) return;

        LoopContext lc;
        bool inLoop = getContext(inst->getParent(), lc);
            
        assert(inversionAllocs && "must be able to allocate inverted caches");
        IRBuilder<> entryBuilder(inversionAllocs);
        entryBuilder.setFastMathFlags(getFast());

        if (!inLoop) {
            scopeMap[inst] = entryBuilder.CreateAlloca(inst->getType(), nullptr, inst->getName()+"_cache");
            auto pn = dyn_cast<PHINode>(inst);
            Instruction* putafter = ( pn && pn->getNumIncomingValues()>0 )? (inst->getParent()->getFirstNonPHI() ): getNextNonDebugInstruction(inst);
            assert(putafter);
            IRBuilder <> v(putafter);
            v.setFastMathFlags(getFast());
            v.CreateStore(inst, scopeMap[inst]);
        } else {

            ValueToValueMapTy valmap;
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
              if (idx.dynamic) {
                ns = ConstantInt::get(idx.limit->getType(), 1);
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
                ns = allocationBuilder.CreateNUWAdd(limitm1, ConstantInt::get(idx.limit->getType(), 1));
              }
              if (size == nullptr) size = ns;
              else size = allocationBuilder.CreateNUWMul(size, ns);
              if (idx.parent == nullptr) break;
            }

            auto firstallocation = CallInst::CreateMalloc(
                    &allocationBuilder.GetInsertBlock()->back(),
                    size->getType(),
                    inst->getType(),
                    ConstantInt::get(size->getType(), allocationBuilder.GetInsertBlock()->getParent()->getParent()->getDataLayout().getTypeAllocSizeInBits(inst->getType())/8), size, nullptr, inst->getName()+"_malloccache");
            CallInst* malloccall = dyn_cast<CallInst>(firstallocation);
            if (malloccall == nullptr) {
                malloccall = cast<CallInst>(cast<Instruction>(firstallocation)->getOperand(0));
            }
            malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
            malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
            //allocationBuilder.GetInsertBlock()->getInstList().push_back(cast<Instruction>(allocation));
            cast<Instruction>(firstallocation)->moveBefore(allocationBuilder.GetInsertBlock()->getTerminator());
            scopeMap[inst] = entryBuilder.CreateAlloca(firstallocation->getType(), nullptr, inst->getName()+"_mdyncache");
            lastScopeAlloc[inst] = firstallocation;
            allocationBuilder.CreateStore(firstallocation, scopeMap[inst]);	


            if (shouldFree) {
                assert(reverseBlocks.size());

                IRBuilder<> tbuild(reverseBlocks[outermostPreheader]);
                tbuild.setFastMathFlags(getFast());

                // ensure we are before the terminator if it exists
                if (tbuild.GetInsertBlock()->size()) {
                      tbuild.SetInsertPoint(tbuild.GetInsertBlock()->getFirstNonPHI());
                }
                
                auto ci = cast<CallInst>(CallInst::CreateFree(tbuild.CreatePointerCast(tbuild.CreateLoad(scopeMap[inst]), Type::getInt8PtrTy(outermostPreheader->getContext())), tbuild.GetInsertBlock()));
                ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
                if (ci->getParent()==nullptr) {
                    tbuild.Insert(ci);
                }
                scopeFrees[inst] = ci;
            }

            auto pn = dyn_cast<PHINode>(inst);
            Instruction* putafter = ( pn && pn->getNumIncomingValues()>0 )? (inst->getParent()->getFirstNonPHI() ): getNextNonDebugInstruction(inst);
            IRBuilder <> v(putafter);
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
                IRBuilder<> outerBuilder(&outermostPreheader->back());
                allocation = outerBuilder.CreateLoad(scopeMap[inst]);
            } else {
                Type *BPTy = Type::getInt8PtrTy(v.GetInsertBlock()->getContext());
                auto realloc = newFunc->getParent()->getOrInsertFunction("realloc", BPTy, BPTy, size->getType());
                allocation = v.CreateLoad(scopeMap[inst]);
                auto foo = v.CreateNUWAdd(dynamicPHI, ConstantInt::get(dynamicPHI->getType(), 1));
                Value* idxs[2] = {
                    v.CreatePointerCast(allocation, BPTy),
                    v.CreateNUWMul(
                        ConstantInt::get(size->getType(), newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(inst->getType())/8), 
                        v.CreateNUWMul(
                            size, foo
                        ) 
                    )
                };

                Value* realloccall = nullptr;
                allocation = v.CreatePointerCast(realloccall = v.CreateCall(realloc, idxs, inst->getName()+"_realloccache"), allocation->getType());
                lastScopeAlloc[inst] = allocation;
                v.CreateStore(allocation, scopeMap[inst]);
            }

            Value* idxs[] = {idx};
            auto gep = v.CreateGEP(allocation, idxs);
            v.CreateStore(inst, gep);
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
                IRBuilder<> lcssa(&lc.exit->front());
                auto lcssaPHI = lcssa.CreatePHI(inst->getType(), 1, inst->getName()+"!manual_lcssa");
                for(auto pred : predecessors(lc.exit))
                    lcssaPHI->addIncoming(inst, pred);
                return lcssaPHI;
            }
        }
        return inst;
    }

    Value* lookupM(Value* val, IRBuilder<>& BuilderM, bool forceLookup=false) {
        if (isa<Constant>(val)) {
            return val;
        }
        if (isa<BasicBlock>(val)) {
            return val;
        }
        if (isa<Function>(val)) {
            return val;
        }
        if (isa<UndefValue>(val)) {
            return val;
        }
        if (isa<Argument>(val)) {
            return val;
        }
        if (isa<MetadataAsValue>(val)) {
            return val;
        }
        if (!isa<Instruction>(val)) {
            llvm::errs() << *val << "\n";
        }

        auto inst = cast<Instruction>(val);
        if (!forceLookup && inversionAllocs && inst->getParent() == inversionAllocs) {
            return val;
        }
        
        if (!forceLookup) {
            if (this->isOriginalBlock(*BuilderM.GetInsertBlock())) {
                if (BuilderM.GetInsertBlock()->size() && BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
                    if (this->DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
                        //llvm::errs() << "allowed " << *inst << "from domination\n";
                        return inst;
                    }
                } else {
                    if (this->DT.dominates(inst, BuilderM.GetInsertBlock())) {
                        //llvm::errs() << "allowed " << *inst << "from block domination\n";
                        return inst;
                    }
                }
            }
            val = inst = fixLCSSA(inst, BuilderM);
        }

        assert(!this->isOriginalBlock(*BuilderM.GetInsertBlock()) || forceLookup);
        LoopContext lc;
        bool inLoop = getContext(inst->getParent(), lc);

        ValueToValueMapTy available;
        if (inLoop) {
            for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx)) {
              if (!isOriginalBlock(*BuilderM.GetInsertBlock())) {
                available[idx.var] = idx.antivar;
              } else {
                available[idx.var] = idx.var;
              }
              if (idx.parent == nullptr) break;
            }    
        }
        
        if (!forceLookup) {
            if (!shouldRecompute(inst, available)) {
                auto op = unwrapM(inst, BuilderM, available, /*lookupIfAble*/true);
                assert(op);
                return op;
            }
            /*
            if (!inLoop) {
                if (!isOriginalBlock(*BuilderM.GetInsertBlock()) && inst->getParent() == BuilderM.GetInsertBlock());
                todo here/re
            }
            */
        }

        ensureLookupCached(inst);

        if (!inLoop) {
            auto result = BuilderM.CreateLoad(scopeMap[inst]);
            assert(result->getType() == inst->getType());
            return result;
        } else {
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
            Value* tolookup = BuilderM.CreateLoad(scopeMap[inst]);
            auto result = BuilderM.CreateLoad(BuilderM.CreateGEP(tolookup, idxs));
            assert(result->getType() == inst->getType());
            return result;
        }
    };
    
    Value* invertPointerM(Value* val, IRBuilder<>& BuilderM) {
      if (isa<ConstantPointerNull>(val)) {
         return val;
      } else if (isa<UndefValue>(val)) {
         return val;
      } else if (auto cint = dyn_cast<ConstantInt>(val)) {
		 if (cint->isZero()) return cint;
         //this is extra
		 if (cint->isOne()) return cint;
	  }

      if(isConstantValue(val)) {
        dumpSet(this->originalInstructions);
        if (auto arg = dyn_cast<Instruction>(val)) {
            llvm::errs() << *arg->getParent()->getParent() << "\n";
        }
        llvm::errs() << *val << "\n";
      }
      assert(!isConstantValue(val));
      auto M = BuilderM.GetInsertBlock()->getParent()->getParent();
      assert(val);

      if (invertedPointers.find(val) != invertedPointers.end()) {
        return lookupM(invertedPointers[val], BuilderM);
      }

      if (auto arg = dyn_cast<GlobalVariable>(val)) {
          if (!hasMetadata(arg, "enzyme_shadow")) {
              llvm::errs() << *arg << "\n";
              report_fatal_error("cannot compute with global variable that doesn't have marked shadow global");
          }
          auto md = arg->getMetadata("enzyme_shadow");
          if (!isa<MDTuple>(md)) {
              llvm::errs() << *arg << "\n";
              llvm::errs() << *md << "\n";
              report_fatal_error("cannot compute with global variable that doesn't have marked shadow global (metadata incorrect type)");
          }
          auto md2 = cast<MDTuple>(md);
          assert(md2->getNumOperands() == 1);
          auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
          auto cs = gvemd->getValue();
          return invertedPointers[val] = cs;
      } else if (auto fn = dyn_cast<Function>(val)) {
          //! Todo allow tape propagation
          auto newf = CreatePrimalAndGradient(fn, /*constant_args*/{}, TLI, AA, /*returnValue*/false, /*differentialReturn*/true, /*topLevel*/false, /*additionalArg*/nullptr);
          return BuilderM.CreatePointerCast(newf, fn->getType());
      } else if (auto arg = dyn_cast<CastInst>(val)) {
        auto result = BuilderM.CreateCast(arg->getOpcode(), invertPointerM(arg->getOperand(0), BuilderM), arg->getDestTy(), arg->getName()+"'ipc");
        return result;
      } else if (auto arg = dyn_cast<ExtractValueInst>(val)) {
        IRBuilder<> bb(arg);
        auto result = bb.CreateExtractValue(invertPointerM(arg->getOperand(0), bb), arg->getIndices(), arg->getName()+"'ipev");
        invertedPointers[arg] = result;
        return lookupM(invertedPointers[arg], BuilderM);
      } else if (auto arg = dyn_cast<InsertValueInst>(val)) {
        IRBuilder<> bb(arg);
        auto result = bb.CreateInsertValue(invertPointerM(arg->getOperand(0), bb), invertPointerM(arg->getOperand(1), bb), arg->getIndices(), arg->getName()+"'ipiv");
        invertedPointers[arg] = result;
        return lookupM(invertedPointers[arg], BuilderM);
      } else if (auto arg = dyn_cast<SelectInst>(val)) {
        IRBuilder<> bb(arg);
        auto result = bb.CreateSelect(arg->getCondition(), invertPointerM(arg->getTrueValue(), bb), invertPointerM(arg->getFalseValue(), bb), arg->getName()+"'ipse");
        invertedPointers[arg] = result;
        return lookupM(invertedPointers[arg], BuilderM);
      } else if (auto arg = dyn_cast<LoadInst>(val)) {
		IRBuilder <> bb(arg);
        auto li = bb.CreateLoad(invertPointerM(arg->getOperand(0), bb), arg->getName()+"'ipl");
        li->setAlignment(arg->getAlignment());
		return lookupM(li, BuilderM);
      } else if (auto arg = dyn_cast<GetElementPtrInst>(val)) {
          if (arg->getParent() == &arg->getParent()->getParent()->getEntryBlock()) {
            IRBuilder<> bb(arg);
            SmallVector<Value*,4> invertargs;
            for(auto &a: arg->indices()) {
                auto b = lookupM(a, bb);
                invertargs.push_back(b);
            }
            auto result = bb.CreateGEP(invertPointerM(arg->getPointerOperand(), bb), invertargs, arg->getName()+"'ipge");
            invertedPointers[arg] = result;
            return lookupM(invertedPointers[arg], BuilderM);
          }

        SmallVector<Value*,4> invertargs;
        for(auto &a: arg->indices()) {
            auto b = lookupM(a, BuilderM);
            invertargs.push_back(b);
        }
        auto result = BuilderM.CreateGEP(invertPointerM(arg->getPointerOperand(), BuilderM), invertargs, arg->getName()+"'ipg");
        return result;
      } else if (auto inst = dyn_cast<AllocaInst>(val)) {
            IRBuilder<> bb(inst);
            AllocaInst* antialloca = bb.CreateAlloca(inst->getAllocatedType(), inst->getType()->getPointerAddressSpace(), inst->getArraySize(), inst->getName()+"'ipa");
            invertedPointers[val] = antialloca;
            antialloca->setAlignment(inst->getAlignment());
            
            auto dst_arg = bb.CreateBitCast(antialloca,Type::getInt8PtrTy(val->getContext()));
            auto val_arg = ConstantInt::get(Type::getInt8Ty(val->getContext()), 0);
            auto len_arg = bb.CreateNUWMul(bb.CreateZExtOrTrunc(inst->getArraySize(),Type::getInt64Ty(val->getContext())), ConstantInt::get(Type::getInt64Ty(val->getContext()), M->getDataLayout().getTypeAllocSizeInBits(inst->getAllocatedType())/8 ) );
            auto volatile_arg = ConstantInt::getFalse(val->getContext());

#if LLVM_VERSION_MAJOR == 6
            auto align_arg = ConstantInt::get(Type::getInt32Ty(val->getContext()), antialloca->getAlignment());
            Value *args[] = { dst_arg, val_arg, len_arg, align_arg, volatile_arg };
#else
            Value *args[] = { dst_arg, val_arg, len_arg, volatile_arg };
#endif
            Type *tys[] = {dst_arg->getType(), len_arg->getType()};
            auto memset = cast<CallInst>(bb.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args));
            memset->addParamAttr(0, Attribute::getWithAlignment(inst->getContext(), inst->getAlignment()));
            memset->addParamAttr(0, Attribute::NonNull);
            return lookupM(invertedPointers[inst], BuilderM);
      } else if (auto phi = dyn_cast<PHINode>(val)) {
		 std::map<Value*,std::set<BasicBlock*>> mapped;
		 for(unsigned int i=0; i<phi->getNumIncomingValues(); i++) {
			mapped[phi->getIncomingValue(i)].insert(phi->getIncomingBlock(i));
		 }

		 if (false && mapped.size() == 1) {
         	return invertPointerM(phi->getIncomingValue(0), BuilderM);
		 }    
#if 0
         else if (false && mapped.size() == 2) {
			 IRBuilder <> bb(phi);
			 auto which = bb.CreatePHI(Type::getInt1Ty(phi->getContext()), phi->getNumIncomingValues());
             //TODO this is not recursive

			 int cnt = 0;
			 Value* vals[2];
			 for(auto v : mapped) {
				assert( cnt <= 1 );
				vals[cnt] = v.first;
				for (auto b : v.second) {
					which->addIncoming(ConstantInt::get(which->getType(), cnt), b);
				}
				cnt++;
			 }
			 
			 auto which2 = lookupM(which, BuilderM);
			 auto result = BuilderM.CreateSelect(which2, invertPointerM(vals[1], BuilderM), invertPointerM(vals[0], BuilderM));
             return result;
		 } 
#endif
         
         else {
			 IRBuilder <> bb(phi);
			 auto which = bb.CreatePHI(phi->getType(), phi->getNumIncomingValues());
             invertedPointers[val] = which;
		 
             for(unsigned int i=0; i<phi->getNumIncomingValues(); i++) {
				IRBuilder <>pre(phi->getIncomingBlock(i)->getTerminator());
				which->addIncoming(invertPointerM(phi->getIncomingValue(i), pre), phi->getIncomingBlock(i));
             }

			 return lookupM(which, BuilderM);
		 }
        }
        assert(BuilderM.GetInsertBlock());
        assert(BuilderM.GetInsertBlock()->getParent());
        assert(val);
        llvm::errs() << "fn:" << *BuilderM.GetInsertBlock()->getParent() << "\nval=" << *val << "\n";
        for(auto z : invertedPointers) {
          llvm::errs() << "available inversion for " << *z.first << " of " << *z.second << "\n"; 
        }
        assert(0 && "cannot find deal with ptr that isnt arg");
        report_fatal_error("cannot find deal with ptr that isnt arg");
      
    };
};
  
class DiffeGradientUtils : public GradientUtils {
  DiffeGradientUtils(Function* newFunc_, AAResults &AA, TargetLibraryInfo &TLI, ValueToValueMapTy& invertedPointers_, const SmallPtrSetImpl<Value*> &constants_, const SmallPtrSetImpl<Value*> &nonconstant_, const SmallPtrSetImpl<Value*> &returnvals_, ValueToValueMapTy &origToNew_)
      : GradientUtils(newFunc_, AA, TLI, invertedPointers_, constants_, nonconstant_, returnvals_, origToNew_) {
        prepareForReverse();
    }

public:
  ValueToValueMapTy differentials;
  static DiffeGradientUtils* CreateFromClone(Function *todiff, AAResults &AA, TargetLibraryInfo &TLI, const std::set<unsigned> & constant_args, ReturnType returnValue, bool differentialReturn, Type* additionalArg) {
    assert(!todiff->empty());
    ValueToValueMapTy invertedPointers;
    SmallPtrSet<Value*,4> constants;
    SmallPtrSet<Value*,20> nonconstant;
    SmallPtrSet<Value*,2> returnvals;
    ValueToValueMapTy originalToNew;
    auto newFunc = CloneFunctionWithReturns(todiff, AA, TLI, invertedPointers, constant_args, constants, nonconstant, returnvals, returnValue, differentialReturn, "diffe"+todiff->getName(), &originalToNew, /*diffeReturnArg*/true, additionalArg);
    auto res = new DiffeGradientUtils(newFunc, AA, TLI, invertedPointers, constants, nonconstant, returnvals, originalToNew);
    res->oldFunc = todiff;
    return res;
  }

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

static cl::opt<bool> autodiff_optimize(
            "enzyme_optimize", cl::init(false), cl::Hidden,
                cl::desc("Force inlining of autodiff"));
void optimizeIntermediate(GradientUtils* gutils, bool topLevel, Function *F) {
    if (!autodiff_optimize) return;

    {
        DominatorTree DT(*F);
        AssumptionCache AC(*F);
        promoteMemoryToRegister(*F, DT, AC);
    }

    FunctionAnalysisManager AM;
     AM.registerPass([] { return AAManager(); });
     AM.registerPass([] { return ScalarEvolutionAnalysis(); });
     AM.registerPass([] { return AssumptionAnalysis(); });
     AM.registerPass([] { return TargetLibraryAnalysis(); });
     AM.registerPass([] { return TargetIRAnalysis(); });
     AM.registerPass([] { return MemorySSAAnalysis(); });
     AM.registerPass([] { return DominatorTreeAnalysis(); });
     AM.registerPass([] { return MemoryDependenceAnalysis(); });
     AM.registerPass([] { return LoopAnalysis(); });
     AM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
     AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
     AM.registerPass([] { return LazyValueAnalysis(); });
     LoopAnalysisManager LAM;
     AM.registerPass([&] { return LoopAnalysisManagerFunctionProxy(LAM); });
     LAM.registerPass([&] { return FunctionAnalysisManagerLoopProxy(AM); });
    //LoopSimplifyPass().run(*F, AM);

 //TODO function attributes
 //PostOrderFunctionAttrsPass().run(*F, AM);
 GVN().run(*F, AM);
 SROA().run(*F, AM);
 EarlyCSEPass(/*memoryssa*/true).run(*F, AM);
#if LLVM_VERSION_MAJOR > 6
 InstSimplifyPass().run(*F, AM);
#endif
 CorrelatedValuePropagationPass().run(*F, AM);

 DCEPass().run(*F, AM);
 DSEPass().run(*F, AM);

 createFunctionToLoopPassAdaptor(LoopDeletionPass()).run(*F, AM);
 
 SimplifyCFGOptions scfgo(/*unsigned BonusThreshold=*/1, /*bool ForwardSwitchCond=*/false, /*bool SwitchToLookup=*/false, /*bool CanonicalLoops=*/true, /*bool SinkCommon=*/true, /*AssumptionCache *AssumpCache=*/nullptr);
 SimplifyCFGPass(scfgo).run(*F, AM);
    
 if (!topLevel) {
 for(BasicBlock& BB: *F) { 
      
        for (auto I = BB.begin(), E = BB.end(); I != E;) {
          Instruction* inst = &*I;
          assert(inst);
          I++;

          if (gutils->originalInstructions.find(inst) == gutils->originalInstructions.end()) continue;

          if (gutils->replaceableCalls.find(inst) != gutils->replaceableCalls.end()) {
            if (inst->getNumUses() != 0 && !cast<CallInst>(inst)->getCalledFunction()->hasFnAttribute(Attribute::ReadNone) ) {
                llvm::errs() << "found call ripe for replacement " << *inst;
            } else {
                    gutils->erase(inst);
                    continue;
            }
          }
        }
      }
 }
 //LCSSAPass().run(*NewF, AM);
}

//! return structtype if recursive function
std::pair<Function*,StructType*> CreateAugmentedPrimal(Function* todiff, AAResults &AA, const std::set<unsigned>& constant_args, TargetLibraryInfo &TLI, bool differentialReturn) {
  static std::map<std::tuple<Function*,std::set<unsigned>, bool/*differentialReturn*/>, std::pair<Function*,StructType*>> cachedfunctions;
  static std::map<std::tuple<Function*,std::set<unsigned>, bool/*differentialReturn*/>, bool> cachedfinished;
  auto tup = std::make_tuple(todiff, std::set<unsigned>(constant_args.begin(), constant_args.end()),  differentialReturn);
  if (cachedfunctions.find(tup) != cachedfunctions.end()) {
    return cachedfunctions[tup];
  }

    if (constant_args.size() == 0 && hasMetadata(todiff, "enzyme_augment")) {
      auto md = todiff->getMetadata("enzyme_augment");
      if (!isa<MDTuple>(md)) {
          llvm::errs() << *todiff << "\n";
          llvm::errs() << *md << "\n";
          report_fatal_error("unknown augment for noninvertible function -- metadata incorrect");
      }
      auto md2 = cast<MDTuple>(md);
      assert(md2->getNumOperands() == 1);
      auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
      auto foundcalled = cast<Function>(gvemd->getValue());

      if (foundcalled->getReturnType() == todiff->getReturnType()) {
        FunctionType *FTy = FunctionType::get(StructType::get(todiff->getContext(), {StructType::get(todiff->getContext(), {}), foundcalled->getReturnType()}),
                                   foundcalled->getFunctionType()->params(), foundcalled->getFunctionType()->isVarArg());
        Function *NewF = Function::Create(FTy, Function::LinkageTypes::InternalLinkage, "fixaugmented_"+todiff->getName(), todiff->getParent());
        NewF->setAttributes(foundcalled->getAttributes());
        if (NewF->hasFnAttribute(Attribute::NoInline)) {
            NewF->removeFnAttr(Attribute::NoInline);
        }
        for (auto i=foundcalled->arg_begin(), j=NewF->arg_begin(); i != foundcalled->arg_end(); ) {
             j->setName(i->getName());
             if (j->hasAttribute(Attribute::Returned))
                 j->removeAttr(Attribute::Returned);
             if (j->hasAttribute(Attribute::StructRet))
                 j->removeAttr(Attribute::StructRet);
             i++;
             j++;
         }
        BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
        IRBuilder <>bb(BB);
        SmallVector<Value*,4> args;
        for(auto &a : NewF->args()) args.push_back(&a);
        auto cal = bb.CreateCall(foundcalled, args);
        cal->setCallingConv(foundcalled->getCallingConv());
        auto ut = UndefValue::get(NewF->getReturnType());
        auto val = bb.CreateInsertValue(ut, cal, {1u});
        bb.CreateRet(val);
        return cachedfunctions[tup] = std::pair<Function*,StructType*>(NewF, nullptr);
      }

      //assert(st->getNumElements() > 0);
      return cachedfunctions[tup] = std::pair<Function*,StructType*>(foundcalled, nullptr); //dyn_cast<StructType>(st->getElementType(0)));
    }
  assert(!todiff->empty());
   
  GradientUtils *gutils = GradientUtils::CreateFromClone(todiff, AA, TLI, constant_args, /*returnValue*/ReturnType::TapeAndReturns, /*differentialReturn*/differentialReturn);
  cachedfunctions[tup] = std::pair<Function*,StructType*>(gutils->newFunc, nullptr);
  cachedfinished[tup] = false;
  llvm::errs() << "function with differential return " << todiff->getName() << " " << differentialReturn << "\n";

  gutils->forceContexts();
  gutils->forceAugmentedReturns();

  for(BasicBlock* BB: gutils->originalBlocks) {
      auto term = BB->getTerminator();
      assert(term);
      if(auto ri = dyn_cast<ReturnInst>(term)) {
        auto oldval = ri->getReturnValue();
        Value* rt = UndefValue::get(gutils->newFunc->getReturnType());
        IRBuilder <>ib(ri);
        if (oldval)
            rt = ib.CreateInsertValue(rt, oldval, {1});
        term = ib.CreateRet(rt);
        gutils->erase(ri);
      } else if (isa<BranchInst>(term) || isa<SwitchInst>(term)) {

      } else if (isa<UnreachableInst>(term)) {
      
      } else {
        assert(BB);
        assert(BB->getParent());
        assert(term);
        llvm::errs() << *BB->getParent() << "\n";
        llvm::errs() << "unknown terminator instance " << *term << "\n";
        assert(0 && "unknown terminator inst");
      }

      if (!isa<UnreachableInst>(term))
      for (auto I = BB->rbegin(), E = BB->rend(); I != E;) {
        Instruction* inst = &*I;
        assert(inst);
        I++;
        if (gutils->originalInstructions.find(inst) == gutils->originalInstructions.end()) continue;

        if(auto op = dyn_cast_or_null<IntrinsicInst>(inst)) {
          switch(op->getIntrinsicID()) {
            case Intrinsic::memcpy: {
                if (gutils->isConstantInstruction(inst)) continue;
                assert(0 && "TODO: memcpy has bug that needs fixing (per int double vs ptr)");
                /*
                SmallVector<Value*, 4> args;
                args.push_back(invertPointer(op->getOperand(0)));
                args.push_back(invertPointer(op->getOperand(1)));
                args.push_back(lookup(op->getOperand(2)));
                args.push_back(lookup(op->getOperand(3)));

                Type *tys[] = {args[0]->getType(), args[1]->getType(), args[2]->getType()};
                auto cal = Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::memcpy, tys), args);
                cal->setAttributes(op->getAttributes());
                */
                break;
            }
            case Intrinsic::memset: {
                if (gutils->isConstantInstruction(inst)) continue;
                /*
                if (!gutils->isConstantValue(op->getOperand(1))) {
                    assert(inst);
                    llvm::errs() << "couldn't handle non constant inst in memset to propagate differential to\n" << *inst;
                    report_fatal_error("non constant in memset");
                }
                auto ptx = invertPointer(op->getOperand(0));
                SmallVector<Value*, 4> args;
                args.push_back(ptx);
                args.push_back(lookup(op->getOperand(1)));
                args.push_back(lookup(op->getOperand(2)));
                args.push_back(lookup(op->getOperand(3)));

                Type *tys[] = {args[0]->getType(), args[2]->getType()};
                auto cal = Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args);
                cal->setAttributes(op->getAttributes());
                */
                break;
            }
            case Intrinsic::stacksave:
            case Intrinsic::stackrestore:
            case Intrinsic::dbg_declare:
            case Intrinsic::dbg_value:
            #if LLVM_VERSION_MAJOR > 6
            case Intrinsic::dbg_label:
            #endif
            case Intrinsic::dbg_addr:
            case Intrinsic::lifetime_start:
            case Intrinsic::lifetime_end:
            case Intrinsic::assume:
            case Intrinsic::fabs:
            case Intrinsic::maxnum:
            case Intrinsic::log:
            case Intrinsic::log2:
            case Intrinsic::log10:
            case Intrinsic::exp:
            case Intrinsic::exp2:
            case Intrinsic::pow:
            case Intrinsic::sin:
            case Intrinsic::cos:
                break;
            default:
              if (gutils->isConstantInstruction(inst)) continue;
              assert(inst);
              llvm::errs() << "cannot handle (augmented) unknown intrinsic\n" << *inst;
              report_fatal_error("(augmented) unknown intrinsic");
          }

        } else if(auto op = dyn_cast_or_null<CallInst>(inst)) {

            Function *called = op->getCalledFunction();
            
            if (auto castinst = dyn_cast<ConstantExpr>(op->getCalledValue())) {
                if (castinst->isCast())
                if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                    if (fn->getName() == "malloc" || fn->getName() == "free" || fn->getName() == "_Znwm" || fn->getName() == "_ZdlPv" || fn->getName() == "_ZdlPvm") {
                        called = fn;
                    }
                }
            }

            if (called && (called->getName() == "printf" || called->getName() == "puts"))
                continue;

            if (called && (called->getName()=="malloc" || called->getName()=="_Znwm")) {
		  		//TODO enable this if we need to free the memory
                if (false && op->getNumUses() != 0) {
                    IRBuilder<> BuilderZ(op);
                    gutils->addMalloc<Instruction>(BuilderZ, op);
                }
                if (!gutils->isConstantValue(op)) {
                    auto placeholder = cast<PHINode>(gutils->invertedPointers[op]);
                    gutils->createAntiMalloc(op);
                    if (I != E && placeholder == &*I) I++;
                }
                continue;
            }

            if (called && (called->getName()=="free" ||
                called->getName()=="_ZdlPv" || called->getName()=="_ZdlPvm"))
                continue;

            if (op->getNumUses() != 0 && !op->doesNotAccessMemory()) {
                IRBuilder<> BuilderZ(op);
                gutils->addMalloc<Instruction>(BuilderZ, op);
            }

            if (gutils->isConstantInstruction(op)) {
                continue;
            }

            if (called == nullptr) {
              assert(op);
              llvm::errs() << "cannot handle augment non constant function\n" << *op << "\n";
              report_fatal_error("unknown augment non constant function");
            }
            
              std::set<unsigned> subconstant_args;

              SmallVector<Value*, 8> args;
              SmallVector<DIFFE_TYPE, 8> argsInverted;
              bool modifyPrimal = !called->hasFnAttribute(Attribute::ReadNone);
              IRBuilder<> BuilderZ(op);
              BuilderZ.setFastMathFlags(getFast());

              if ( (op->getType()->isPointerTy() || op->getType()->isIntegerTy()) && !gutils->isConstantValue(op) ) {
                 modifyPrimal = true;
                 //llvm::errs() << "primal modified " << called->getName() << " modified via return" << "\n";
              }

              if (called->empty()) modifyPrimal = true;

              for(unsigned i=0;i<op->getNumArgOperands(); i++) {
                args.push_back(op->getArgOperand(i));

                if (gutils->isConstantValue(op->getArgOperand(i)) && !called->empty()) {
                    subconstant_args.insert(i);
                    argsInverted.push_back(DIFFE_TYPE::CONSTANT);
                    continue;
                }

                auto argType = op->getArgOperand(i)->getType();

                if (argType->isPointerTy() || argType->isIntegerTy()) {
                    argsInverted.push_back(DIFFE_TYPE::DUP_ARG);
                    args.push_back(gutils->invertPointerM(op->getArgOperand(i), BuilderZ));

                    if (! ( called->hasParamAttribute(i, Attribute::ReadOnly) || called->hasParamAttribute(i, Attribute::ReadNone)) ) {
                        modifyPrimal = true;
                        //llvm::errs() << "primal modified " << called->getName() << " modified via arg " << i << "\n";
                    }
                    //Note sometimes whattype mistakenly says something should be constant [because composed of integer pointers alone]
                    assert(whatType(argType) == DIFFE_TYPE::DUP_ARG || whatType(argType) == DIFFE_TYPE::CONSTANT);
                } else {
                    argsInverted.push_back(DIFFE_TYPE::OUT_DIFF);
                    assert(whatType(argType) == DIFFE_TYPE::OUT_DIFF || whatType(argType) == DIFFE_TYPE::CONSTANT);
                }
              }

              //TODO create augmented primal
              if (modifyPrimal) {
                auto newcalled = CreateAugmentedPrimal(dyn_cast<Function>(called), AA, subconstant_args, TLI, /*differentialReturn*/!gutils->isConstantValue(op)).first;
                auto augmentcall = BuilderZ.CreateCall(newcalled, args);
                augmentcall->setCallingConv(op->getCallingConv());
                augmentcall->setDebugLoc(inst->getDebugLoc());
                 
                if (!op->getType()->isVoidTy()) {
                  auto rv = cast<Instruction>(BuilderZ.CreateExtractValue(augmentcall, {1}));
                  gutils->originalInstructions.insert(rv);
                  gutils->originalInstructions.insert(augmentcall);
                  gutils->nonconstant.insert(rv);
                  gutils->nonconstant.insert(augmentcall);
                  if (!gutils->isConstantValue(op)) {
                    gutils->nonconstant_values.insert(rv);
                  }
                  assert(op->getType() == rv->getType());
                  llvm::errs() << "augmented considering differential ip of " << called->getName() << " " << *op->getType() << " " << gutils->isConstantValue(op) << "\n";
                  
                  if ((op->getType()->isPointerTy() || op->getType()->isIntegerTy()) && !gutils->isConstantValue(op)) {
                    auto antiptr = cast<Instruction>(BuilderZ.CreateExtractValue(augmentcall, {2}));
                    auto placeholder = cast<PHINode>(gutils->invertedPointers[op]);
                    if (I != E && placeholder == &*I) I++;
                    gutils->invertedPointers.erase(op);
                    placeholder->replaceAllUsesWith(antiptr);
                    gutils->erase(placeholder);
                    gutils->invertedPointers[rv] = antiptr;
                    gutils->addMalloc<Instruction>(BuilderZ, antiptr);
                  }

                  gutils->replaceAWithB(op,rv);
                }

                Value* tp = BuilderZ.CreateExtractValue(augmentcall, {0});
                if (tp->getType()->isEmptyTy()) {
                    auto tpt = tp->getType();
                    gutils->erase(cast<Instruction>(tp));
                    tp = UndefValue::get(tpt);
                }
                gutils->addMalloc<Value>(BuilderZ, tp);
                gutils->erase(op);
              }
        } else if(isa<LoadInst>(inst)) {
          if (gutils->isConstantInstruction(inst)) continue;

           //TODO IF OP IS POINTER
        } else if(auto op = dyn_cast<StoreInst>(inst)) {
          if (gutils->isConstantInstruction(inst)) continue;

          //TODO const
           //TODO IF OP IS POINTER
          if ( op->getValueOperand()->getType()->isPointerTy() || (op->getValueOperand()->getType()->isIntegerTy() && !isIntASecretFloat(op->getValueOperand()) ) ) {
            IRBuilder <> storeBuilder(op);
            llvm::errs() << "a op value: " << *op->getValueOperand() << "\n";
            Value* valueop = gutils->invertPointerM(op->getValueOperand(), storeBuilder);
            llvm::errs() << "a op pointer: " << *op->getPointerOperand() << "\n";
            Value* pointerop = gutils->invertPointerM(op->getPointerOperand(), storeBuilder);
            storeBuilder.CreateStore(valueop, pointerop);
            //llvm::errs() << "ignoring store bc pointer of " << *op << "\n";
          }
        }
     }
  }

  auto nf = gutils->newFunc;
  
  ValueToValueMapTy invertedRetPs;
  if ((gutils->oldFunc->getReturnType()->isPointerTy() || gutils->oldFunc->getReturnType()->isIntegerTy()) && differentialReturn) {
    for (inst_iterator I = inst_begin(nf), E = inst_end(nf); I != E; ++I) {
      if (ReturnInst* ri = dyn_cast<ReturnInst>(&*I)) {
        IRBuilder <>builder(ri);
        invertedRetPs[ri] = gutils->invertPointerM(cast<InsertValueInst>(ri->getReturnValue())->getInsertedValueOperand(), builder);
        assert(invertedRetPs[ri]);
      }
    }
  }
  
  while(gutils->inversionAllocs->size() > 0) {
    gutils->inversionAllocs->back().moveBefore(gutils->newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  }

  (IRBuilder <>(gutils->inversionAllocs)).CreateUnreachable();
  DeleteDeadBlock(gutils->inversionAllocs);
  
  for (Argument &Arg : gutils->newFunc->args()) {
      if (Arg.hasAttribute(Attribute::Returned))
          Arg.removeAttr(Attribute::Returned);
      if (Arg.hasAttribute(Attribute::StructRet))
          Arg.removeAttr(Attribute::StructRet);
  }

  if (gutils->newFunc->hasFnAttribute(Attribute::OptimizeNone))
    gutils->newFunc->removeFnAttr(Attribute::OptimizeNone);

  if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
      llvm::errs() << *gutils->oldFunc << "\n";
      llvm::errs() << *gutils->newFunc << "\n";
      report_fatal_error("function failed verification (2)");
  }

  std::vector<Type*> RetTypes;

  std::vector<Type*> MallocTypes;

  for(auto a:gutils->getMallocs()) { 
      MallocTypes.push_back(a->getType());
  }

  StructType* tapeType = StructType::get(nf->getContext(), MallocTypes);

  bool recursive = cachedfunctions[tup].first->getNumUses() > 0;

  if (recursive) {
    RetTypes.push_back(Type::getInt8PtrTy(nf->getContext()));
  } else {
    RetTypes.push_back(tapeType);
  }

  if (!gutils->oldFunc->getReturnType()->isVoidTy()) {
    RetTypes.push_back(gutils->oldFunc->getReturnType());
    if (gutils->oldFunc->getReturnType()->isPointerTy() || gutils->oldFunc->getReturnType()->isIntegerTy())
      RetTypes.push_back(gutils->oldFunc->getReturnType());
  }

  Type* RetType = StructType::get(nf->getContext(), RetTypes);
  
 ValueToValueMapTy VMap;
 std::vector<Type*> ArgTypes;
 for (const Argument &I : nf->args()) {
     ArgTypes.push_back(I.getType());
 }

 // Create a new function type...
 FunctionType *FTy = FunctionType::get(RetType,
                                   ArgTypes, nf->getFunctionType()->isVarArg());

 // Create the new function...
 Function *NewF = Function::Create(FTy, nf->getLinkage(), "augmented_"+todiff->getName(), nf->getParent());

 unsigned ii = 0, jj = 0;
 for (auto i=nf->arg_begin(), j=NewF->arg_begin(); i != nf->arg_end(); ) {
    VMap[i] = j;
    if (nf->hasParamAttribute(ii, Attribute::NoCapture)) {
      NewF->addParamAttr(jj, Attribute::NoCapture);
    }
    if (nf->hasParamAttribute(ii, Attribute::NoAlias)) {
       NewF->addParamAttr(jj, Attribute::NoAlias);
    }

     j->setName(i->getName());
     j++;
     jj++;
     
     i++;
     ii++;
 }

  SmallVector <ReturnInst*,4> Returns;
  CloneFunctionInto(NewF, nf, VMap, nf->getSubprogram() != nullptr, Returns, "",
                   nullptr);

  IRBuilder<> ib(NewF->getEntryBlock().getFirstNonPHI());

  Value* ret = ib.CreateAlloca(RetType);

  Value* tapeMemory;
  if (recursive) {
      auto i64 = Type::getInt64Ty(NewF->getContext());
      tapeMemory = CallInst::CreateMalloc(NewF->getEntryBlock().getFirstNonPHI(),
                    i64,
                    tapeType,
                    ConstantInt::get(i64, NewF->getParent()->getDataLayout().getTypeAllocSizeInBits(tapeType)/8),
                    nullptr,
                    nullptr,
                    "tapemem"
                    );
            CallInst* malloccall = dyn_cast<CallInst>(tapeMemory);
            if (malloccall == nullptr) {
                malloccall = cast<CallInst>(cast<Instruction>(tapeMemory)->getOperand(0));
            }
            malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
            malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
    Value *Idxs[] = {
        ib.getInt32(0),
        ib.getInt32(0),
    };
    ib.CreateStore(malloccall, ib.CreateGEP(ret, Idxs, ""));
  } else {
    Value *Idxs[] = {
        ib.getInt32(0),
        ib.getInt32(0),
    };
    tapeMemory = ib.CreateGEP(ret, Idxs, "");
  }
  
  unsigned i=0;
  for (auto v: gutils->getMallocs()) {
      if (!isa<UndefValue>(v)) {
          IRBuilder <>ib(cast<Instruction>(VMap[v])->getNextNode());
          Value *Idxs[] = {
            ib.getInt32(0),
            ib.getInt32(i)
          };
          auto gep = ib.CreateGEP(tapeMemory, Idxs, "");
          ib.CreateStore(VMap[v], gep);
      }
      i++;
  }

  for (inst_iterator I = inst_begin(nf), E = inst_end(nf); I != E; ++I) {
      if (ReturnInst* ri = dyn_cast<ReturnInst>(&*I)) {
          ReturnInst* rim = cast<ReturnInst>(VMap[ri]);
          Value* rv = rim->getReturnValue();
          Type* oldretTy = gutils->oldFunc->getReturnType();
          IRBuilder <>ib(rim);
          if (!oldretTy->isVoidTy()) {
            ib.CreateStore(cast<InsertValueInst>(rv)->getInsertedValueOperand(), ib.CreateConstGEP2_32(RetType, ret, 0, 1, ""));
            
            if ((oldretTy->isPointerTy() || oldretTy->isIntegerTy()) && differentialReturn) {
              assert(invertedRetPs[ri]);
              assert(VMap[invertedRetPs[ri]]);
              ib.CreateStore( VMap[invertedRetPs[ri]], ib.CreateConstGEP2_32(RetType, ret, 0, 2, ""));
            }
            i++;
          }
          ib.CreateRet(ib.CreateLoad(ret));
          gutils->erase(cast<Instruction>(VMap[ri]));
      }
  }

  for (Argument &Arg : NewF->args()) {
      if (Arg.hasAttribute(Attribute::Returned))
          Arg.removeAttr(Attribute::Returned);
      if (Arg.hasAttribute(Attribute::StructRet))
          Arg.removeAttr(Attribute::StructRet);
  }
  if (NewF->hasFnAttribute(Attribute::OptimizeNone))
    NewF->removeFnAttr(Attribute::OptimizeNone);
  
  if (auto bytes = NewF->getDereferenceableBytes(llvm::AttributeList::ReturnIndex)) {
    AttrBuilder ab;
    ab.addDereferenceableAttr(bytes);
    NewF->removeAttributes(llvm::AttributeList::ReturnIndex, ab);
  }
  if (NewF->hasAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias)) {
    NewF->removeAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias);
  }

  if (llvm::verifyFunction(*NewF, &llvm::errs())) {
      llvm::errs() << *gutils->oldFunc << "\n";
      llvm::errs() << *NewF << "\n";
      report_fatal_error("augmented function failed verification (3)");
  }

  SmallVector<User*,4> fnusers;
  for(auto user : cachedfunctions[tup].first->users()) {
    fnusers.push_back(user);
  }
  for(auto user : fnusers) {
    cast<CallInst>(user)->setCalledFunction(NewF);
  }
  cachedfunctions[tup].first = NewF;
  if (recursive)
      cachedfunctions[tup].second = tapeType;
  cachedfinished[tup] = true;
  
  gutils->newFunc->eraseFromParent();

  delete gutils;
  if (autodiff_print)
    llvm::errs() << *NewF << "\n";
  return std::pair<Function*,StructType*>(NewF, recursive ? tapeType : nullptr);
}
  
void createInvertedTerminator(DiffeGradientUtils* gutils, BasicBlock *BB, AllocaInst* retAlloca, unsigned extraArgs) { 
    LoopContext loopContext;
    bool inLoop = gutils->getContext(BB, loopContext);
    BasicBlock* BB2 = gutils->reverseBlocks[BB];
    assert(BB2);
    IRBuilder<> Builder(BB2);
    Builder.setFastMathFlags(getFast());

      SmallVector<BasicBlock*,4> preds;
      for(auto B : predecessors(BB)) {
        preds.push_back(B);
      }

      if (preds.size() == 0) {
        SmallVector<Value *,4> retargs;

        if (retAlloca) {
          retargs.push_back(Builder.CreateLoad(retAlloca));
          assert(retargs[0]);
        }

        auto endidx = gutils->newFunc->arg_end();
        for(unsigned i=0; i<extraArgs; i++)
            endidx--;

        for (auto& I: gutils->newFunc->args()) {
          if (&I == endidx) {
              break;
          }
          if (!gutils->isConstantValue(&I) && whatType(I.getType()) == DIFFE_TYPE::OUT_DIFF ) {
            retargs.push_back(gutils->diffe((Value*)&I, Builder));
          }
        }

        Value* toret = UndefValue::get(gutils->newFunc->getReturnType());
        for(unsigned i=0; i<retargs.size(); i++) {
          unsigned idx[] = { i };
          toret = Builder.CreateInsertValue(toret, retargs[i], idx);
        }
        Builder.SetInsertPoint(Builder.GetInsertBlock());
        Builder.CreateRet(toret);
      } else if (preds.size() == 1) {
        for (auto I = BB->begin(), E = BB->end(); I != E; I++) {
            if(auto PN = dyn_cast<PHINode>(&*I)) {
                if (gutils->isConstantValue(PN)) continue;
                //TODO consider whether indeed we can skip differential phi pointers
                if (PN->getType()->isPointerTy()) continue;
                auto prediff = gutils->diffe(PN, Builder);
                gutils->setDiffe(PN, Constant::getNullValue(PN->getType()), Builder);
                if (!gutils->isConstantValue(PN->getIncomingValueForBlock(preds[0]))) {
                    gutils->addToDiffe(PN->getIncomingValueForBlock(preds[0]), prediff, Builder);
                }
            } else break;
        }

        Builder.SetInsertPoint(Builder.GetInsertBlock());
        Builder.CreateBr(gutils->reverseBlocks[preds[0]]);

      } else if (preds.size() == 2) {
        IRBuilder <> pbuilder(&BB->front());
        pbuilder.setFastMathFlags(getFast());
        Value* phi = nullptr;

        if (inLoop && BB2 == gutils->reverseBlocks[loopContext.var->getParent()]) {
          assert( ((preds[0] == loopContext.latch) && (preds[1] == loopContext.preheader)) || ((preds[1] == loopContext.latch) && (preds[0] == loopContext.preheader)) );
          if (preds[0] == loopContext.latch)
            phi = Builder.CreateICmpNE(loopContext.antivar, Constant::getNullValue(loopContext.antivar->getType()));
          else if(preds[1] == loopContext.latch)
            phi = Builder.CreateICmpEQ(loopContext.antivar, Constant::getNullValue(loopContext.antivar->getType()));
          else {
            llvm::errs() << "weird behavior for loopContext\n";
            assert(0 && "illegal loopcontext behavior");
          }
        } else {
          std::map<BasicBlock*,std::set<unsigned>> seen;
          std::map<BasicBlock*,std::set<BasicBlock*>> done;
          std::deque<std::tuple<BasicBlock*,unsigned,BasicBlock*>> Q; // newblock, prednum, pred
          Q.push_back(std::tuple<BasicBlock*,unsigned,BasicBlock*>(preds[0], 0, BB));
          Q.push_back(std::tuple<BasicBlock*,unsigned,BasicBlock*>(preds[1], 1, BB));
          //done.insert(BB);

          while(Q.size()) {
                auto trace = Q.front();
                auto block = std::get<0>(trace);
                auto num = std::get<1>(trace);
                auto predblock = std::get<2>(trace);
                Q.pop_front();

                if (seen[block].count(num) && done[block].count(predblock)) {
                  continue;
                }

                seen[block].insert(num);
                done[block].insert(predblock);

                if (seen[block].size() == 1) {
                  for (BasicBlock *Pred : predecessors(block)) {
                    Q.push_back(std::tuple<BasicBlock*,unsigned,BasicBlock*>(Pred, (*seen[block].begin()), block ));
                  }
                }

                SmallVector<BasicBlock*,4> succs;
                bool allDone = true;
                for (BasicBlock *Succ : successors(block)) {
                    succs.push_back(Succ);
                    if (done[block].count(Succ) == 0) {
                      allDone = false;
                    }
                }

                if (!allDone) {
                  continue;
                }

                if (seen[block].size() == preds.size() && succs.size() == preds.size()) {
                  //TODO below doesnt generalize past 2
                  bool hasSingle = false;
                  for(auto a : succs) {
                    if (seen[a].size() == 1) {
                      hasSingle = true;
                    }
                  }
                  if (!hasSingle)
                      goto continueloop;
                  if (auto branch = dyn_cast<BranchInst>(block->getTerminator())) {
                    assert(branch->getCondition());
                    phi = gutils->lookupM(branch->getCondition(), Builder);
                    for(unsigned i=0; i<preds.size(); i++) {
                        auto s = branch->getSuccessor(i);
                        assert(s == succs[i]);
                        if (seen[s].size() == 1) {
                            if ( (*seen[s].begin()) != i) {
                                phi = Builder.CreateNot(phi);
                                break;
                            } else {
                                break;
                            }
                        }
                    }
                    goto endPHI;
                  }

                  break;
                }
                continueloop:;
          }

          phi = pbuilder.CreatePHI(Type::getInt1Ty(Builder.getContext()), 2);
          cast<PHINode>(phi)->addIncoming(ConstantInt::getTrue(phi->getType()), preds[0]);
          cast<PHINode>(phi)->addIncoming(ConstantInt::getFalse(phi->getType()), preds[1]);
          phi = gutils->lookupM(phi, Builder);
          goto endPHI;

          endPHI:;
        }

        for (auto I = BB->begin(), E = BB->end(); I != E; I++) {
            if(auto PN = dyn_cast<PHINode>(&*I)) {

                // POINTER TYPE THINGS
                if (PN->getType()->isPointerTy()) continue;
                if (gutils->isConstantValue(PN)) continue; 
                auto prediff = gutils->diffe(PN, Builder);
                gutils->setDiffe(PN, Constant::getNullValue(PN->getType()), Builder);
                if (!gutils->isConstantValue(PN->getIncomingValueForBlock(preds[0]))) {
                    auto dif = Builder.CreateSelect(phi, prediff, Constant::getNullValue(prediff->getType()));
                    gutils->addToDiffe(PN->getIncomingValueForBlock(preds[0]), dif, Builder);
                }
                if (!gutils->isConstantValue(PN->getIncomingValueForBlock(preds[1]))) {
                    auto dif = Builder.CreateSelect(phi, Constant::getNullValue(prediff->getType()), prediff);
                    gutils->addToDiffe(PN->getIncomingValueForBlock(preds[1]), dif, Builder);
                }
            } else break;
        }
        BasicBlock* f0 = cast<BasicBlock>(gutils->reverseBlocks[preds[0]]);
        BasicBlock* f1 = cast<BasicBlock>(gutils->reverseBlocks[preds[1]]);
        while (auto bo = dyn_cast<BinaryOperator>(phi)) {
            if (bo->getOpcode() == BinaryOperator::Xor) {
                if (auto ci = dyn_cast<ConstantInt>(bo->getOperand(1))) {
                    if (ci->isOne()) {
                        phi = bo->getOperand(0);
                        auto ft = f0;
                        f0 = f1;
                        f1 = ft;
                        continue;
                    }
                }

                if (auto ci = dyn_cast<ConstantInt>(bo->getOperand(0))) {
                    if (ci->isOne()) {
                        phi = bo->getOperand(1);
                        auto ft = f0;
                        f0 = f1;
                        f1 = ft;
                        continue;
                    }
                }
                break;
            } else break;
        }
        Builder.SetInsertPoint(Builder.GetInsertBlock());
        Builder.CreateCondBr(phi, f0, f1);
      } else {
        IRBuilder <> pbuilder(&BB->front());
        pbuilder.setFastMathFlags(getFast());
        Value* phi = nullptr;

        if (true) {
          phi = pbuilder.CreatePHI(Type::getInt8Ty(Builder.getContext()), preds.size());
          for(unsigned i=0; i<preds.size(); i++) {
            cast<PHINode>(phi)->addIncoming(ConstantInt::get(phi->getType(), i), preds[i]);
          }
          phi = gutils->lookupM(phi, Builder);
        }

        for (auto I = BB->begin(), E = BB->end(); I != E; I++) {
            if(auto PN = dyn_cast<PHINode>(&*I)) {
              if (gutils->isConstantValue(PN)) continue;

              // POINTER TYPE THINGS
              if (PN->getType()->isPointerTy()) continue;
              auto prediff = gutils->diffe(PN, Builder);
              gutils->setDiffe(PN, Constant::getNullValue(PN->getType()), Builder);
              for(unsigned i=0; i<preds.size(); i++) {
                if (!gutils->isConstantValue(PN->getIncomingValueForBlock(preds[i]))) {
                    auto cond = Builder.CreateICmpEQ(phi, ConstantInt::get(phi->getType(), i));
                    auto dif = Builder.CreateSelect(cond, prediff, Constant::getNullValue(prediff->getType()));
                    gutils->addToDiffe(PN->getIncomingValueForBlock(preds[i]), dif, Builder);
                }
              }
            } else break;
        }

        Builder.SetInsertPoint(Builder.GetInsertBlock());
        auto swit = Builder.CreateSwitch(phi, gutils->reverseBlocks[preds.back()], preds.size()-1);
        for(unsigned i=0; i<preds.size()-1; i++) {
          swit->addCase(ConstantInt::get(cast<IntegerType>(phi->getType()), i), gutils->reverseBlocks[preds[i]]);
        }
      }
}

//! assuming not top level
std::pair<SmallVector<Type*,4>,SmallVector<Type*,4>> getDefaultFunctionTypeForGradient(FunctionType* called, bool differentialReturn) {
    SmallVector<Type*, 4> args;
    SmallVector<Type*, 4> outs;
    for(auto &argType : called->params()) {
        args.push_back(argType);

        if ( argType->isPointerTy() || argType->isIntegerTy() ) {
            args.push_back(argType);
        } else {
            outs.push_back(argType);
        }
    }

    auto ret = called->getReturnType();
    if (!ret->isVoidTy()) {
        if (differentialReturn) {
            args.push_back(ret);
        }
    }

    return std::pair<SmallVector<Type*,4>,SmallVector<Type*,4>>(args, outs);
}

Function* CreatePrimalAndGradient(Function* todiff, const std::set<unsigned>& constant_args, TargetLibraryInfo &TLI, AAResults &AA, bool returnValue, bool differentialReturn, bool topLevel, llvm::Type* additionalArg) { 
  static std::map<std::tuple<Function*,std::set<unsigned>, bool/*retval*/, bool/*differentialReturn*/, bool/*topLevel*/, llvm::Type*>, Function*> cachedfunctions;
  auto tup = std::make_tuple(todiff, std::set<unsigned>(constant_args.begin(), constant_args.end()), returnValue, differentialReturn, topLevel, additionalArg);
  if (cachedfunctions.find(tup) != cachedfunctions.end()) {
    return cachedfunctions[tup];
  }

  if (constant_args.size() == 0 && !topLevel && !returnValue && hasMetadata(todiff, "enzyme_gradient")) {

      auto md = todiff->getMetadata("enzyme_gradient");
      if (!isa<MDTuple>(md)) {
          llvm::errs() << *todiff << "\n";
          llvm::errs() << *md << "\n";
          report_fatal_error("unknown gradient for noninvertible function -- metadata incorrect");
      }
      auto md2 = cast<MDTuple>(md);
      assert(md2->getNumOperands() == 1);
      auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
      auto foundcalled = cast<Function>(gvemd->getValue());
            
      auto res = getDefaultFunctionTypeForGradient(todiff->getFunctionType(), differentialReturn);

      bool hasTape = false;

      if (foundcalled->arg_size() == res.first.size() + 1 /*tape*/) {
        auto lastarg = foundcalled->arg_end();
        lastarg--;
        res.first.push_back(lastarg->getType());
        hasTape = true;
      } else if (foundcalled->arg_size() == res.first.size()) {
        res.first.push_back(StructType::get(todiff->getContext(), {}));
      } else {
          llvm::errs() << *foundcalled << "\n";
          assert(0 && "bad type for custom gradient");
      }

      auto st = dyn_cast<StructType>(foundcalled->getReturnType());
      bool wrongRet = st == nullptr;
      if (wrongRet || !hasTape) {
        FunctionType *FTy = FunctionType::get(StructType::get(todiff->getContext(), {res.second}), res.first, todiff->getFunctionType()->isVarArg());
        Function *NewF = Function::Create(FTy, Function::LinkageTypes::InternalLinkage, "fixgradient_"+todiff->getName(), todiff->getParent());
        NewF->setAttributes(foundcalled->getAttributes());
        if (NewF->hasFnAttribute(Attribute::NoInline)) {
            NewF->removeFnAttr(Attribute::NoInline);
        }
          for (Argument &Arg : NewF->args()) {
              if (Arg.hasAttribute(Attribute::Returned))
                  Arg.removeAttr(Attribute::Returned);
              if (Arg.hasAttribute(Attribute::StructRet))
                  Arg.removeAttr(Attribute::StructRet);
          }

        BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
        IRBuilder <>bb(BB);
        SmallVector<Value*,4> args;
        for(auto &a : NewF->args()) args.push_back(&a);
        if (!hasTape) {
            args.pop_back();
        }
        llvm::errs() << *NewF << "\n";
        llvm::errs() << *foundcalled << "\n";
        auto cal = bb.CreateCall(foundcalled, args);
        cal->setCallingConv(foundcalled->getCallingConv());
        Value* val = cal;
        if (wrongRet) {
            auto ut = UndefValue::get(NewF->getReturnType());
            if (val->getType()->isEmptyTy() && res.second.size() == 0) {
                val = ut;
            } else if (res.second.size() == 1 && res.second[0] == val->getType()) {
                val = bb.CreateInsertValue(ut, cal, {0u});
            } else {
                llvm::errs() << *foundcalled << "\n";
                assert(0 && "illegal type for reverse");
            }
        }
        bb.CreateRet(val);
        foundcalled = NewF;
      } 
      return cachedfunctions[tup] = foundcalled;
  }

  assert(!todiff->empty());
  auto M = todiff->getParent();

  auto& Context = M->getContext();

  DiffeGradientUtils *gutils = DiffeGradientUtils::CreateFromClone(todiff, AA, TLI, constant_args, returnValue ? ReturnType::ArgsWithReturn : ReturnType::Args, differentialReturn, additionalArg);
  cachedfunctions[tup] = gutils->newFunc;
  
  gutils->forceContexts(); 
  gutils->forceAugmentedReturns();
  
  Argument* additionalValue = nullptr;
  if (additionalArg) {
    auto v = gutils->newFunc->arg_end();
    v--;
    additionalValue = v;
    gutils->setTape(additionalValue);
  }

  Argument* differetval = nullptr;
  if (differentialReturn) {
    auto endarg = gutils->newFunc->arg_end();
    endarg--;
    if (additionalArg) endarg--;
    differetval = endarg;
  }

  llvm::AllocaInst* retAlloca = nullptr;
  if (returnValue && differentialReturn) {
	retAlloca = IRBuilder<>(&gutils->newFunc->getEntryBlock().front()).CreateAlloca(todiff->getReturnType(), nullptr, "toreturn");
  }

  std::map<ReturnInst*,StoreInst*> replacedReturns;

  for(BasicBlock* BB: gutils->originalBlocks) {

    LoopContext loopContext;
    bool inLoop = gutils->getContext(BB, loopContext);

    auto BB2 = gutils->reverseBlocks[BB];
    assert(BB2);

    IRBuilder<> Builder2(BB2);
    if (BB2->size() > 0) {
        Builder2.SetInsertPoint(BB2->getFirstNonPHI());
    }
    Builder2.setFastMathFlags(getFast());

    std::map<Value*,Value*> alreadyLoaded;

    std::function<Value*(Value*)> lookup = [&](Value* val) -> Value* {
      if (alreadyLoaded.find(val) != alreadyLoaded.end()) {
        return alreadyLoaded[val];
      }
      return alreadyLoaded[val] = gutils->lookupM(val, Builder2);
    };

    auto diffe = [&Builder2,&gutils](Value* val) -> Value* {
        return gutils->diffe(val, Builder2);
    };

    auto addToDiffe = [&Builder2,&gutils](Value* val, Value* dif) -> void {
      gutils->addToDiffe(val, dif, Builder2);
    };

    auto setDiffe = [&](Value* val, Value* toset) -> void {
      gutils->setDiffe(val, toset, Builder2);
    };

    auto addToDiffeIndexed = [&](Value* val, Value* dif, ArrayRef<Value*> idxs) -> void{
      gutils->addToDiffeIndexed(val, dif, idxs, Builder2);
    };

    auto invertPointer = [&](Value* val) -> Value* {
        return gutils->invertPointerM(val, Builder2);
    };

    auto addToPtrDiffe = [&](Value* val, Value* dif) {
      gutils->addToPtrDiffe(val, dif, Builder2);
    };
    
    auto setPtrDiffe = [&](Value* val, Value* dif) {
      gutils->setPtrDiffe(val, dif, Builder2);
    };

  auto term = BB->getTerminator();
  assert(term);
  bool unreachableTerminator = false;
  if(auto op = dyn_cast<ReturnInst>(term)) {
      auto retval = op->getReturnValue();
      IRBuilder<> rb(op);
      rb.setFastMathFlags(getFast());
	  if (retAlloca) {
		auto si = rb.CreateStore(retval, retAlloca);
        replacedReturns[cast<ReturnInst>(gutils->getOriginal(op))] = si;
      }
	 
      rb.CreateBr(BB2);

      gutils->erase(op);

      if (differentialReturn && !gutils->isConstantValue(retval)) {
        setDiffe(retval, differetval);
      } else {
		assert (retAlloca == nullptr);
      }
  } else if (isa<BranchInst>(term) || isa<SwitchInst>(term)) {

  } else if (isa<UnreachableInst>(term)) {
    unreachableTerminator = true;
    continue;
  } else {
    assert(BB);
    assert(BB->getParent());
    assert(term);
    llvm::errs() << *BB->getParent() << "\n";
    llvm::errs() << "unknown terminator instance " << *term << "\n";
    assert(0 && "unknown terminator inst");
  }

  if (inLoop && loopContext.latch==BB) {
    BB2->getInstList().push_front(loopContext.antivar);

    IRBuilder<> tbuild(gutils->reverseBlocks[loopContext.exit]);
    tbuild.setFastMathFlags(getFast());

    // ensure we are before the terminator if it exists
    if (gutils->reverseBlocks[loopContext.exit]->size()) {
      tbuild.SetInsertPoint(&gutils->reverseBlocks[loopContext.exit]->back());
    }

    auto sub = Builder2.CreateSub(loopContext.antivar, ConstantInt::get(loopContext.antivar->getType(), 1));
    for(BasicBlock* in: successors(loopContext.latch) ) {
        if (loopContext.exit == in) {
            loopContext.antivar->addIncoming(gutils->lookupM(loopContext.limit, tbuild), gutils->reverseBlocks[in]);
        } else if (gutils->LI.getLoopFor(in) == gutils->LI.getLoopFor(BB)) {
            loopContext.antivar->addIncoming(sub, gutils->reverseBlocks[in]);
        }
    }
  }

  if (!unreachableTerminator)
  for (auto I = BB->rbegin(), E = BB->rend(); I != E;) {
    Instruction* inst = &*I;
    assert(inst);
    I++;
    if (gutils->originalInstructions.find(inst) == gutils->originalInstructions.end()) continue;

    if (auto op = dyn_cast<BinaryOperator>(inst)) {
      if (gutils->isConstantInstruction(inst)) continue;
      Value* dif0 = nullptr;
      Value* dif1 = nullptr;
      switch(op->getOpcode()) {
        case Instruction::FMul:{
          auto idiff = diffe(inst);
          if (!gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFMul(idiff, lookup(op->getOperand(1)), "m0diffe"+op->getOperand(0)->getName());
          if (!gutils->isConstantValue(op->getOperand(1)))
            dif1 = Builder2.CreateFMul(idiff, lookup(op->getOperand(0)), "m1diffe"+op->getOperand(1)->getName());
          break;
        }
        case Instruction::FAdd:{
          auto idiff = diffe(inst);
          if (!gutils->isConstantValue(op->getOperand(0)))
            dif0 = idiff;
          if (!gutils->isConstantValue(op->getOperand(1)))
            dif1 = idiff;
          break;
        }
        case Instruction::FSub:{
          auto idiff = diffe(inst);
          if (!gutils->isConstantValue(op->getOperand(0)))
            dif0 = idiff;
          if (!gutils->isConstantValue(op->getOperand(1)))
            dif1 = Builder2.CreateFNeg(idiff);
          break;
        }
        case Instruction::FDiv:{
          auto idiff = diffe(inst);
          if (!gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFDiv(idiff, lookup(op->getOperand(1)), "d0diffe"+op->getOperand(0)->getName());
          if (!gutils->isConstantValue(op->getOperand(1)))
            dif1 = Builder2.CreateFNeg(
              Builder2.CreateFDiv(
                Builder2.CreateFMul(idiff, lookup(op)),
                lookup(op->getOperand(1)))
            );
          break;
        }
        default:
          assert(op);
          llvm::errs() << *gutils->newFunc << "\n";
          llvm::errs() << "cannot handle unknown binary operator: " << *op << "\n";
          report_fatal_error("unknown binary operator");
      }

      if (dif0 || dif1) setDiffe(inst, Constant::getNullValue(inst->getType()));
      if (dif0) addToDiffe(op->getOperand(0), dif0);
      if (dif1) addToDiffe(op->getOperand(1), dif1);
    } else if(auto op = dyn_cast_or_null<IntrinsicInst>(inst)) {
      Value* dif0 = nullptr;
      Value* dif1 = nullptr;
      switch(op->getIntrinsicID()) {
        case Intrinsic::memcpy: {
            if (gutils->isConstantInstruction(inst)) continue;
            assert(0 && "TODO: memcpy has bug that needs fixing (per int double vs ptr)");
            SmallVector<Value*, 4> args;
            // source and dest are swapped
            args.push_back(invertPointer(op->getOperand(1)));
            args.push_back(invertPointer(op->getOperand(0)));
            args.push_back(lookup(op->getOperand(2)));
            args.push_back(lookup(op->getOperand(3)));

            Type *tys[] = {args[0]->getType(), args[1]->getType(), args[2]->getType()};
            auto cal = Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::memcpy, tys), args);
            cal->setAttributes(op->getAttributes());
            cal->setCallingConv(op->getCallingConv());
            cal->setTailCallKind(op->getTailCallKind());

            break;
        }
        case Intrinsic::memset: {
            if (gutils->isConstantInstruction(inst)) continue;
            if (!gutils->isConstantValue(op->getOperand(1))) {
                assert(inst);
                llvm::errs() << "couldn't handle non constant inst in memset to propagate differential to\n" << *inst;
                report_fatal_error("non constant in memset");
            }
            auto ptx = invertPointer(op->getOperand(0));
            SmallVector<Value*, 4> args;
            args.push_back(ptx);
            for(unsigned i=1; i<op->getNumArgOperands(); i++) {
                args.push_back(lookup(op->getOperand(i)));
            }

            Type *tys[] = {args[0]->getType(), args[2]->getType()};
            auto cal = Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args);
            cal->setAttributes(op->getAttributes());
            cal->setCallingConv(op->getCallingConv());
            cal->setTailCallKind(op->getTailCallKind());
            
            break;
        }
        case Intrinsic::assume:
        case Intrinsic::stacksave:
        case Intrinsic::stackrestore:
        case Intrinsic::dbg_declare:
        case Intrinsic::dbg_value:
        #if LLVM_VERSION_MAJOR > 6
        case Intrinsic::dbg_label:
        #endif
        case Intrinsic::dbg_addr:
            break;
        case Intrinsic::lifetime_start:{
            if (gutils->isConstantInstruction(inst)) continue;
            SmallVector<Value*, 2> args = {lookup(op->getOperand(0)), lookup(op->getOperand(1))};
            Type *tys[] = {args[1]->getType()};
            auto cal = Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::lifetime_end, tys), args);
            cal->setAttributes(op->getAttributes());
            cal->setCallingConv(op->getCallingConv());
            cal->setTailCallKind(op->getTailCallKind());
            break;
        }
        case Intrinsic::lifetime_end:
            gutils->erase(op);
            break;
        case Intrinsic::sqrt: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateBinOp(Instruction::FDiv, diffe(inst),
              Builder2.CreateFMul(ConstantFP::get(op->getType(), 2.0), lookup(op))
            );
          break;
        }
        case Intrinsic::fabs: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0))) {
            auto cmp = Builder2.CreateFCmpOLT(lookup(op->getOperand(0)), ConstantFP::get(op->getOperand(0)->getType(), 0));
            dif0 = Builder2.CreateFMul(Builder2.CreateSelect(cmp, ConstantFP::get(op->getOperand(0)->getType(), -1), ConstantFP::get(op->getOperand(0)->getType(), 1)), diffe(inst));
          }
          break;
        }
        case Intrinsic::maxnum: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0))) {
            auto cmp = Builder2.CreateFCmpOLT(lookup(op->getOperand(0)), lookup(op->getOperand(1)));
            dif0 = Builder2.CreateSelect(cmp, ConstantFP::get(op->getOperand(0)->getType(), 0), diffe(inst));
          }
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(1))) {
            auto cmp = Builder2.CreateFCmpOLT(lookup(op->getOperand(0)), lookup(op->getOperand(1)));
            dif1 = Builder2.CreateSelect(cmp, diffe(inst), ConstantFP::get(op->getOperand(0)->getType(), 0));
          }
          break;
        }

        case Intrinsic::log: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFDiv(diffe(inst), lookup(op->getOperand(0)));
          break;
        }
        case Intrinsic::log2: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFDiv(diffe(inst),
              Builder2.CreateFMul(ConstantFP::get(op->getType(), 0.6931471805599453), lookup(op->getOperand(0)))
            );
          break;
        }
        case Intrinsic::log10: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFDiv(diffe(inst),
              Builder2.CreateFMul(ConstantFP::get(op->getType(), 2.302585092994046), lookup(op->getOperand(0)))
            );
          break;
        }
        case Intrinsic::exp: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFMul(diffe(inst), lookup(op));
          break;
        }
        case Intrinsic::exp2: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFMul(
              Builder2.CreateFMul(diffe(inst), lookup(op)), ConstantFP::get(op->getType(), 0.6931471805599453)
            );
          break;
        }
        case Intrinsic::pow: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0))) {

            /*
            dif0 = Builder2.CreateFMul(
              Builder2.CreateFMul(diffe(inst),
                Builder2.CreateFDiv(lookup(op), lookup(op->getOperand(0)))), lookup(op->getOperand(1))
            );
            */
            SmallVector<Value*, 2> args = {lookup(op->getOperand(0)), Builder2.CreateFSub(lookup(op->getOperand(1)), ConstantFP::get(op->getType(), 1.0))};
            Type *tys[] = {args[1]->getType()};
            auto cal = Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::pow, tys), args);
            cal->setAttributes(op->getAttributes());
            cal->setCallingConv(op->getCallingConv());
            cal->setTailCallKind(op->getTailCallKind());
            dif0 = Builder2.CreateFMul(
              Builder2.CreateFMul(diffe(inst), cal)
              , lookup(op->getOperand(1))
            );
          }

          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(1))) {
            Value *args[] = {lookup(op->getOperand(1))};
            Type *tys[] = {op->getOperand(1)->getType()};

            dif1 = Builder2.CreateFMul(
              Builder2.CreateFMul(diffe(inst), lookup(op)),
              Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::log, tys), args)
            );
          }

          break;
        }
        case Intrinsic::sin: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0))) {
            Value *args[] = {lookup(op->getOperand(0))};
            Type *tys[] = {op->getOperand(0)->getType()};
            dif0 = Builder2.CreateFMul(diffe(inst),
              Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::cos, tys), args) );
          }
          break;
        }
        case Intrinsic::cos: {
          if (!gutils->isConstantInstruction(op) && !gutils->isConstantValue(op->getOperand(0))) {
            Value *args[] = {lookup(op->getOperand(0))};
            Type *tys[] = {op->getOperand(0)->getType()};
            dif0 = Builder2.CreateFMul(diffe(inst),
              Builder2.CreateFNeg(
                Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::sin, tys), args) )
            );
          }
          break;
        }
        default:
          if (gutils->isConstantInstruction(inst)) continue;
          assert(inst);
          llvm::errs() << "cannot handle unknown intrinsic\n" << *inst;
          report_fatal_error("unknown intrinsic");
      }

      if (dif0 || dif1) setDiffe(inst, Constant::getNullValue(inst->getType()));
      if (dif0) addToDiffe(op->getOperand(0), dif0);
      if (dif1) addToDiffe(op->getOperand(1), dif1);
    } else if(auto op = dyn_cast_or_null<CallInst>(inst)) {

        Function *called = op->getCalledFunction();
        
        if (auto castinst = dyn_cast<ConstantExpr>(op->getCalledValue())) {
            if (castinst->isCast())
            if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                if (fn->getName() == "malloc" || fn->getName() == "free" || fn->getName() == "_Znwm" || fn->getName() == "_ZdlPv" || fn->getName() == "_ZdlPvm") {
                    called = fn;
                }
            }
        }

        if (called && (called->getName() == "printf" || called->getName() == "puts")) {
            SmallVector<Value*, 4> args;
            for(unsigned i=0; i<op->getNumArgOperands(); i++) {
                args.push_back(lookup(op->getArgOperand(i)));
            }
            auto cal = Builder2.CreateCall(called, args);
            cal->setAttributes(op->getAttributes());
            cal->setCallingConv(op->getCallingConv());
            cal->setTailCallKind(op->getTailCallKind());
            continue;
        }

        if (called && called->getName()=="malloc") {

          if (!gutils->isConstantValue(inst)) { 
            auto placeholder = cast<PHINode>(gutils->invertedPointers[op]);
            auto anti = gutils->createAntiMalloc(op);
            if (I != E && placeholder == &*I) I++; 
            auto lu = lookup(anti);
            auto ci = cast<CallInst>(CallInst::CreateFree(Builder2.CreatePointerCast(lu, Type::getInt8PtrTy(Context)), Builder2.GetInsertBlock()));
            ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
            if (ci->getParent()==nullptr) Builder2.Insert(ci);
          }
          
		  //TODO enable this if we need to free the memory
		  // NOTE THAT TOPLEVEL IS THERE SIMPLY BECAUSE THAT WAS PREVIOUS ATTITUTE TO FREE'ing
		  if (topLevel) {
		     if (!topLevel && op->getNumUses() != 0) {
               IRBuilder<> BuilderZ(op);
               inst = gutils->addMalloc<Instruction>(BuilderZ, op);
             }
             auto ci = CallInst::CreateFree(Builder2.CreatePointerCast(lookup(inst), Type::getInt8PtrTy(Context)), Builder2.GetInsertBlock());
             if (ci->getParent()==nullptr) Builder2.Insert(ci);
          }

          continue;
        } 
        
        if (called && called->getName()=="_Znwm") {
          //TODO _ZdlPv or _ZdlPvm
          Type *VoidTy = Type::getVoidTy(Context);
          Type *IntPtrTy = Type::getInt8PtrTy(Context);
          auto FreeFunc = M->getOrInsertFunction("_ZdlPv", VoidTy, IntPtrTy);

          if (!gutils->isConstantValue(op)) { 
            auto placeholder = cast<PHINode>(gutils->invertedPointers[op]);
            auto anti = gutils->createAntiMalloc(op);
            if (I != E && placeholder == &*I) I++;
              
            auto ci = cast<CallInst>(CallInst::Create(FreeFunc, {Builder2.CreatePointerCast(lookup(anti), Type::getInt8PtrTy(Context))}, "", Builder2.GetInsertBlock()));
            //ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
            //ci->setTailCall();
            if (Function *F = dyn_cast<Function>(FreeFunc)) ci->setCallingConv(F->getCallingConv());
            if (ci->getParent()==nullptr) Builder2.Insert(ci);
          }
          
		  //TODO enable this if we need to free the memory
		  // NOTE THAT TOPLEVEL IS THERE SIMPLY BECAUSE THAT WAS PREVIOUS ATTITUTE TO FREE'ing
          if (topLevel) {
		      if (!topLevel && op->getNumUses() != 0) {
                IRBuilder<> BuilderZ(op);
                inst = gutils->addMalloc<Instruction>(BuilderZ, op);
              }
              auto ci = cast<CallInst>(CallInst::Create(FreeFunc, {Builder2.CreatePointerCast(lookup(inst), Type::getInt8PtrTy(Context))}, "", Builder2.GetInsertBlock()));
              ci->setTailCall();
              if (Function *F = dyn_cast<Function>(FreeFunc)) ci->setCallingConv(F->getCallingConv());
              if (ci->getParent()==nullptr) Builder2.Insert(ci);
          }
          continue;
        }
        
        if (called && called->getName()=="free") {
            llvm::Value* val = op->getArgOperand(0);
            while(auto cast = dyn_cast<CastInst>(val)) val = cast->getOperand(0);
            if (auto dc = dyn_cast<CallInst>(val)) {
                if (dc->getCalledFunction()->getName() == "malloc") {
                    gutils->erase(op);
                    continue;
                }
            }
            if (isa<ConstantPointerNull>(val)) {
                gutils->erase(op);
                llvm::errs() << "removing free of null pointer\n";
                continue;
            }
            llvm::errs() << "freeing without malloc " << *val << "\n";
            gutils->erase(op);
            continue;
            //TODO HANDLE FREE
        }
        
        if (called && (called->getName()=="_ZdlPv" || called->getName()=="_ZdlPvm")) {
            llvm::Value* val = op->getArgOperand(0);
            while(auto cast = dyn_cast<CastInst>(val)) val = cast->getOperand(0);
            if (auto dc = dyn_cast<CallInst>(val)) {
                if (dc->getCalledFunction()->getName() == "_Znwm") {
                    gutils->erase(op);
                    continue;
                }
            }
            llvm::errs() << "deleting without new " << *val << "\n";
            gutils->erase(op);
            continue;
            //TODO HANDLE FREE/DELETE
        }
        
        if (gutils->isConstantInstruction(op)) {
          if (!topLevel && op->getNumUses() != 0 && !op->doesNotAccessMemory()) {
            IRBuilder<> BuilderZ(op);
            gutils->addMalloc<Instruction>(BuilderZ, op);
          }
          continue;
        }

        bool modifyPrimal = false;
        bool foreignFunction = false;

        if (called && !called->hasFnAttribute(Attribute::ReadNone)) {
            modifyPrimal = true;
        }

        if (called == nullptr || called->empty()) {
            foreignFunction = true;
            modifyPrimal = true;
        }

          std::set<unsigned> subconstant_args;

          SmallVector<Value*, 8> args;
          SmallVector<Value*, 8> pre_args;
          SmallVector<DIFFE_TYPE, 8> argsInverted;
          IRBuilder<> BuilderZ(op);
          std::vector<Instruction*> postCreate;
          BuilderZ.setFastMathFlags(getFast());

          if ( (op->getType()->isPointerTy() || op->getType()->isIntegerTy()) && !gutils->isConstantValue(op)) {
              //llvm::errs() << "augmented modified " << called->getName() << " modified via return" << "\n";
              modifyPrimal = true;
          }

          for(unsigned i=0;i<op->getNumArgOperands(); i++) {
            args.push_back(lookup(op->getArgOperand(i)));
            pre_args.push_back(op->getArgOperand(i));

            if (gutils->isConstantValue(op->getArgOperand(i)) && !foreignFunction) {
                subconstant_args.insert(i);
                argsInverted.push_back(DIFFE_TYPE::CONSTANT);
                continue;
            }

            auto argType = op->getArgOperand(i)->getType();

            if ( (argType->isPointerTy() || argType->isIntegerTy()) && !gutils->isConstantValue(op->getArgOperand(i)) ) {
                argsInverted.push_back(DIFFE_TYPE::DUP_ARG);
                args.push_back(gutils->invertPointerM(op->getArgOperand(i), Builder2));
				pre_args.push_back(gutils->invertPointerM(op->getArgOperand(i), BuilderZ));

                //TODO this check should consider whether this pointer has allocation/etc modifications and so on
                if (called && ! ( called->hasParamAttribute(i, Attribute::ReadOnly) || called->hasParamAttribute(i, Attribute::ReadNone)) ) {
                    //llvm::errs() << "augmented modified " << called->getName() << " modified via arg " << i << "\n";
                    modifyPrimal = true;
                }

                //Note sometimes whattype mistakenly says something should be constant [because composed of integer pointers alone]
                assert(whatType(argType) == DIFFE_TYPE::DUP_ARG || whatType(argType) == DIFFE_TYPE::CONSTANT);
            } else {
                argsInverted.push_back(DIFFE_TYPE::OUT_DIFF);
                assert(whatType(argType) == DIFFE_TYPE::OUT_DIFF || whatType(argType) == DIFFE_TYPE::CONSTANT);
            }
          }

          bool replaceFunction = false;
          if (topLevel && BB->getSingleSuccessor() == BB2 && !foreignFunction) {
              auto origop = cast<CallInst>(gutils->getOriginal(op));
              auto OBB = cast<BasicBlock>(gutils->getOriginal(BB));
              //TODO fix this to be more accurate
              auto iter = OBB->rbegin();
              SmallPtrSet<Instruction*,4> usetree;
              usetree.insert(origop);
              for(auto uinst = origop->getNextNode(); uinst != nullptr; uinst = uinst->getNextNode()) {
                bool usesInst = false;
                for(auto &operand : uinst->operands()) {
                    if (auto usedinst = dyn_cast<Instruction>(operand.get())) {
                        if (usetree.find(usedinst) != usetree.end()) {
                            usesInst = true;
                            break;
                        }
                    }
                }
                if (usesInst) {
                    usetree.insert(uinst);
                }

              }

              while(iter != OBB->rend() && &*iter != origop) {
                if (auto call = dyn_cast<CallInst>(&*iter)) {
                    if (isCertainMallocOrFree(call->getCalledFunction())) {
                        iter++;
                        continue;
                    }
                }
                
                if (auto ri = dyn_cast<ReturnInst>(&*iter)) {
                    auto fd = replacedReturns.find(ri);
                    if (fd != replacedReturns.end()) {
                        auto si = fd->second;
                        if (usetree.find(dyn_cast<Instruction>(gutils->getOriginal(si->getValueOperand()))) != usetree.end()) {
                            postCreate.push_back(si);
                        }
                    }
                    iter++;
                    continue;
                }
                
                //TODO usesInst has created a bug here
                // if it is used by the reverse pass before this call creates it
                // and thus it loads an undef from cache
                bool usesInst = usetree.find(&*iter) != usetree.end();

                //TODO remove this upon start of more accurate)
                //if (usesInst) break;

                if (!usesInst && (!iter->mayReadOrWriteMemory() || isa<BinaryOperator>(&*iter))) {
                    iter++;
                    continue;
                }

                ModRefInfo mri = ModRefInfo::NoModRef;
                if (iter->mayReadOrWriteMemory()) {
                    mri = AA.getModRefInfo(&*iter, origop);
                }

                if (mri == ModRefInfo::NoModRef && !usesInst) {
                    iter++;
                    continue;
                }

                //load that follows the original
                if (auto li = dyn_cast<LoadInst>(&*iter)) {
                    bool modref = false;
                        for(Instruction* it = li; it != nullptr; it = it->getNextNode()) {
                            if (auto call = dyn_cast<CallInst>(it)) {
                                 if (isCertainMallocOrFree(call->getCalledFunction())) {
                                     continue;
                                 }
                            }
                            if (AA.canInstructionRangeModRef(*it, *it, MemoryLocation::get(li), ModRefInfo::Mod)) {
                                modref = true;
                        llvm::errs() << " inst  found mod " << *iter << " " << *it << "\n";
                            }
                        }

                    if (modref)
                        break;
                    postCreate.push_back(cast<Instruction>(gutils->getNewFromOriginal(&*iter)));
                    iter++;
                    continue;
                }
                
                //call that follows the original
                if (auto li = dyn_cast<IntrinsicInst>(&*iter)) {
                  if (li->getIntrinsicID() == Intrinsic::memcpy) {
                   auto mem0 = AA.getModRefInfo(&*iter, li->getOperand(0), MemoryLocation::UnknownSize);
                   auto mem1 = AA.getModRefInfo(&*iter, li->getOperand(1), MemoryLocation::UnknownSize);

                   llvm::errs() << "modrefinfo for mem0 " << *li->getOperand(0) << " " << (unsigned int)mem0 << "\n";
                   llvm::errs() << "modrefinfo for mem1 " << *li->getOperand(1) << " " << (unsigned int)mem1 << "\n";
                    //TODO TEMPORARILY REMOVED THIS TO TEST SOMETHING
                    // PUT BACK THIS CONDITION!!
                    //if ( !isModSet(mem0) )
                    {
                    bool modref = false;
                        for(Instruction* it = li; it != nullptr; it = it->getNextNode()) {
                            if (auto call = dyn_cast<CallInst>(it)) {
                                 if (isCertainMallocOrFree(call->getCalledFunction())) {
                                     continue;
                                 }
                            }
                            if (AA.canInstructionRangeModRef(*it, *it, li->getOperand(1), MemoryLocation::UnknownSize, ModRefInfo::Mod)) {
                                modref = true;
                        llvm::errs() << " inst  found mod " << *iter << " " << *it << "\n";
                            }
                        }

                    if (modref)
                        break;
                    postCreate.push_back(cast<Instruction>(gutils->getNewFromOriginal(&*iter)));
                    iter++;
                    continue;
                    }
                }
                }
                
                if (usesInst) {
                    bool modref = false;
                    //auto start = &*iter;
                    if (mri != ModRefInfo::NoModRef) {
                        modref = true;
                        /*
                        for(Instruction* it = start; it != nullptr; it = it->getNextNode()) {
                            if (auto call = dyn_cast<CallInst>(it)) {
                                 if (isCertainMallocOrFree(call->getCalledFunction())) {
                                     continue;
                                 }
                            }
                            if ( (isa<StoreInst>(start) || isa<CallInst>(start)) && AA.canInstructionRangeModRef(*it, *it, MemoryLocation::get(start), ModRefInfo::Mod)) {
                                modref = true;
                        llvm::errs() << " inst  found mod " << *iter << " " << *it << "\n";
                            }
                        }
                        */
                    }

                    if (modref)
                        break;
                    postCreate.push_back(cast<Instruction>(gutils->getNewFromOriginal(&*iter)));
                    iter++;
                    continue;
                }
                
                break;
              }
              if (&*iter == gutils->getOriginal(op)) {
                  bool outsideuse = false;
                  for(auto user : op->users()) {
                    if (gutils->originalInstructions.find(cast<Instruction>(user)) == gutils->originalInstructions.end()) {
                        outsideuse = true;
                    }
                  }

                  if (!outsideuse) {
                      if (called)
                        llvm::errs() << " choosing to replace function " << (called->getName()) << " and do both forward/reverse\n";
                      else
                          llvm::errs() << " choosing to replace function " << (*op->getCalledValue()) << " and do both forward/reverse\n";

                      replaceFunction = true;
                      modifyPrimal = false;
                  } else {
                  if (called)
                      llvm::errs() << " failed to replace function (cacheuse)" << (called->getName()) << " due to " << *iter << "\n";
                  else 
                      llvm::errs() << " failed to replace function (cacheuse)" << (*op->getCalledValue()) << " due to " << *iter << "\n";

                  }
              } else {
                  if (called)
                      llvm::errs() << " failed to replace function " << (called->getName()) << " due to " << *iter << "\n";
                  else 
                      llvm::errs() << " failed to replace function " << (*op->getCalledValue()) << " due to " << *iter << "\n";
              }
          }

          Value* tape = nullptr;
          CallInst* augmentcall = nullptr;
          Instruction* cachereplace = nullptr;
          //TODO consider what to do if called == nullptr for augmentation
          if (modifyPrimal && called) {
            auto fnandtapetype = CreateAugmentedPrimal(cast<Function>(called), AA, subconstant_args, TLI, /*differentialReturns*/!gutils->isConstantValue(op));
            if (topLevel) {
              Function* newcalled = fnandtapetype.first;
              augmentcall = BuilderZ.CreateCall(newcalled, pre_args);
              augmentcall->setCallingConv(op->getCallingConv());
              augmentcall->setDebugLoc(inst->getDebugLoc());
          
              gutils->originalInstructions.insert(augmentcall);
              gutils->nonconstant.insert(augmentcall);

              tape = BuilderZ.CreateExtractValue(augmentcall, {0});
              if (tape->getType()->isEmptyTy()) {
                auto tt = tape->getType();
                gutils->erase(cast<Instruction>(tape));
                tape = UndefValue::get(tt);
              }

              if( (op->getType()->isPointerTy() || op->getType()->isIntegerTy()) && !gutils->isConstantValue(op) ) {
                auto newip = cast<Instruction>(BuilderZ.CreateExtractValue(augmentcall, {2}));
                auto placeholder = cast<PHINode>(gutils->invertedPointers[op]);
                if (I != E && placeholder == &*I) I++;
                placeholder->replaceAllUsesWith(newip);
                gutils->erase(placeholder);
                gutils->invertedPointers[op] = newip;
              }
            } else {
			  //TODO unknown type for tape phi
              assert(additionalValue);
              if (!topLevel && op->getNumUses() != 0 && !op->doesNotAccessMemory()) {
					cachereplace = IRBuilder<>(op).CreatePHI(op->getType(), 1);
                    cachereplace = gutils->addMalloc<Instruction>(BuilderZ, cachereplace);
              }
              if( (op->getType()->isPointerTy() || op->getType()->isIntegerTy()) && !gutils->isConstantValue(op) ) {
                IRBuilder<> bb(op);
                auto placeholder = cast<PHINode>(gutils->invertedPointers[op]);
                if (I != E && placeholder == &*I) I++;
                auto newip = gutils->addMalloc<Instruction>(bb, placeholder);
                gutils->invertedPointers[op] = newip;
              }
            }

            IRBuilder<> builder(op);
            tape = gutils->addMalloc<Value>(builder, tape);
            if (fnandtapetype.second) {
              auto tapep = BuilderZ.CreatePointerCast(tape, PointerType::getUnqual(fnandtapetype.second));
              auto truetape = BuilderZ.CreateLoad(tapep);
                
              CallInst* ci = cast<CallInst>(CallInst::CreateFree(tape, &*BuilderZ.GetInsertPoint()));
              ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
              tape = truetape;
            }

          } else {
              if (!topLevel && op->getNumUses() != 0 && !op->doesNotAccessMemory()) {
					cachereplace = IRBuilder<>(op).CreatePHI(op->getType(), 1);
                    cachereplace = gutils->addMalloc<Instruction>(BuilderZ, cachereplace);
              }
          }
 
          bool retUsed = replaceFunction && (op->getNumUses() > 0);

          Value* newcalled = nullptr;
          
          if (called) {
            newcalled = CreatePrimalAndGradient(cast<Function>(called), subconstant_args, TLI, AA, retUsed, !gutils->isConstantValue(inst) && !inst->getType()->isPointerTy(), /*topLevel*/replaceFunction, tape ? tape->getType() : nullptr);//, LI, DT);
          } else {
            newcalled = gutils->invertPointerM(op->getCalledValue(), Builder2);
            auto ft = cast<FunctionType>(cast<PointerType>(op->getCalledValue()->getType())->getElementType());
            auto res = getDefaultFunctionTypeForGradient(ft, differentialReturn);
            //TODO Note there is empty tape added here, replace with generic
            //res.first.push_back(StructType::get(newcalled->getContext(), {}));
            newcalled = Builder2.CreatePointerCast(newcalled, PointerType::getUnqual(FunctionType::get(StructType::get(newcalled->getContext(), res.second), res.first, ft->isVarArg())));
          }

          if (!gutils->isConstantValue(inst) && !inst->getType()->isPointerTy()) {
            args.push_back(diffe(inst));
          }

          if (tape) {
              args.push_back(lookup(tape));
          }

          CallInst* diffes = Builder2.CreateCall(newcalled, args);
          diffes->setCallingConv(op->getCallingConv());
          diffes->setDebugLoc(inst->getDebugLoc());

          unsigned structidx = retUsed ? 1 : 0;

          for(unsigned i=0;i<op->getNumArgOperands(); i++) {
            if (argsInverted[i] == DIFFE_TYPE::OUT_DIFF) {
              Value* diffeadd = Builder2.CreateExtractValue(diffes, {structidx});
              structidx++;
              addToDiffe(op->getArgOperand(i), diffeadd);
            }
          }

          //TODO this shouldn't matter because this can't use itself, but setting null should be done before other sets but after load of diffe
          if (inst->getNumUses() != 0 && !gutils->isConstantValue(inst))
            setDiffe(inst, Constant::getNullValue(inst->getType()));
          
          gutils->originalInstructions.insert(diffes);
          gutils->nonconstant.insert(diffes);
          if (!gutils->isConstantValue(op))
            gutils->nonconstant_values.insert(diffes);

          if (replaceFunction) {
            ValueToValueMapTy mapp;
            if (op->getNumUses() != 0) {
              auto retval = cast<Instruction>(Builder2.CreateExtractValue(diffes, {0}));
              gutils->originalInstructions.insert(retval);
              gutils->nonconstant.insert(retval);
              if (!gutils->isConstantValue(op))
                gutils->nonconstant_values.insert(retval);
              op->replaceAllUsesWith(retval);
              mapp[op] = retval;
            }
            for (auto &a : *BB) {
              if (&a != op) {
                mapp[&a] = &a;
              }
            }
            for (auto &a : *BB2) {
                mapp[&a] = &a;
            }
            std::reverse(postCreate.begin(), postCreate.end());
            for(auto a : postCreate) {
                for(unsigned i=0; i<a->getNumOperands(); i++) {
                    a->setOperand(i, gutils->unwrapM(a->getOperand(i), Builder2, mapp, true));
                }
                a->moveBefore(*Builder2.GetInsertBlock(), Builder2.GetInsertPoint());
            }
            gutils->erase(op);
            continue;
          }

          if (augmentcall || cachereplace) {

            if (!op->getType()->isVoidTy()) {
              Instruction* dcall = nullptr;
              if (augmentcall) {
                dcall = cast<Instruction>(BuilderZ.CreateExtractValue(augmentcall, {1}));
              } 
              if (cachereplace) {
                assert(dcall == nullptr);
                dcall = cast<Instruction>(cachereplace);
              }

              gutils->originalInstructions.insert(dcall);
              gutils->nonconstant.insert(dcall);
              if (!gutils->isConstantValue(op))
                gutils->nonconstant_values.insert(dcall);

              if (called)
                 llvm::errs() << "augmented considering differential ip of " << (called->getName()) << " " << *op->getType() << " " << gutils->isConstantValue(op) << "\n";
              else
                  llvm::errs() << "augmented considering differential ip of " << (*op->getCalledValue()) << " " << *op->getType() << " " << gutils->isConstantValue(op) << "\n";
              if (!gutils->isConstantValue(op)) {
                  if (op->getType()->isPointerTy() || op->getType()->isIntegerTy()) {
                    gutils->invertedPointers[dcall] = gutils->invertedPointers[op];
                    gutils->invertedPointers.erase(op);
                  } else {
                    gutils->differentials[dcall] = gutils->differentials[op];
                    gutils->differentials.erase(op);
                  }
              }
              op->replaceAllUsesWith(dcall);
            }

            gutils->erase(op);

            if (augmentcall)
               gutils->replaceableCalls.insert(augmentcall);
          } else {
            gutils->replaceableCalls.insert(op);
          }
    } else if(auto op = dyn_cast_or_null<SelectInst>(inst)) {
      if (gutils->isConstantValue(inst)) continue;
      if (op->getType()->isPointerTy()) continue;

      Value* dif1 = nullptr;
      Value* dif2 = nullptr;

      if (!gutils->isConstantValue(op->getOperand(1)))
        dif1 = Builder2.CreateSelect(lookup(op->getOperand(0)), diffe(inst), Constant::getNullValue(op->getOperand(1)->getType()), "diffe"+op->getOperand(1)->getName());
      if (!gutils->isConstantValue(op->getOperand(2)))
        dif2 = Builder2.CreateSelect(lookup(op->getOperand(0)), Constant::getNullValue(op->getOperand(2)->getType()), diffe(inst), "diffe"+op->getOperand(2)->getName());

      setDiffe(inst, Constant::getNullValue(inst->getType()));
      if (dif1) addToDiffe(op->getOperand(1), dif1);
      if (dif2) addToDiffe(op->getOperand(2), dif2);
    } else if(auto op = dyn_cast<LoadInst>(inst)) {
      if (gutils->isConstantValue(inst)) continue;

       //TODO IF OP IS POINTER
      if (!op->getType()->isPointerTy()) {
        auto prediff = diffe(inst);
        setDiffe(inst, Constant::getNullValue(inst->getType()));
        addToPtrDiffe(op->getOperand(0), prediff);
      } else {
        //Builder2.CreateStore(diffe(inst), invertPointer(op->getOperand(0)));//, op->getName()+"'psweird");
        //addToNPtrDiffe(op->getOperand(0), diffe(inst));
        //assert(0 && "cannot handle non const pointer load inversion");
        assert(op);
		llvm::errs() << "ignoring load bc pointer of " << *op << "\n";
      }
    } else if(auto op = dyn_cast<StoreInst>(inst)) {
      if (gutils->isConstantInstruction(inst)) continue;

      //TODO const
       //TODO IF OP IS POINTER
      if (! ( op->getValueOperand()->getType()->isPointerTy() || (op->getValueOperand()->getType()->isIntegerTy() && !isIntASecretFloat(op->getValueOperand()) ) ) ) {
		  if (!gutils->isConstantValue(op->getValueOperand())) {
			auto dif1 = Builder2.CreateLoad(invertPointer(op->getPointerOperand()));
            setPtrDiffe(op->getPointerOperand(), Constant::getNullValue(op->getValueOperand()->getType()));
			addToDiffe(op->getValueOperand(), dif1);
		  }
	  } else if (topLevel) {
        IRBuilder <> storeBuilder(op);
        //llvm::errs() << "op value: " << *op->getValueOperand() << "\n";
        Value* valueop = gutils->invertPointerM(op->getValueOperand(), storeBuilder);
        //llvm::errs() << "op pointer: " << *op->getPointerOperand() << "\n";
        Value* pointerop = gutils->invertPointerM(op->getPointerOperand(), storeBuilder);
        storeBuilder.CreateStore(valueop, pointerop);
		//llvm::errs() << "ignoring store bc pointer of " << *op << "\n";
	  }
      //?necessary if pointer is readwrite
      /*
      IRBuilder<> BuilderZ(inst);
      Builder2.CreateStore(
        lookup(BuilderZ.CreateLoad(op->getPointerOperand())), lookup(op->getPointerOperand()));
      */
    } else if(auto op = dyn_cast<ExtractValueInst>(inst)) {
      if (gutils->isConstantValue(inst)) continue;
      if (op->getType()->isPointerTy()) continue;

      auto prediff = diffe(inst);
      //todo const
      if (!gutils->isConstantValue(op->getOperand(0))) {
		SmallVector<Value*,4> sv;
      	for(auto i : op->getIndices())
        	sv.push_back(ConstantInt::get(Type::getInt32Ty(Context), i));
        addToDiffeIndexed(op->getOperand(0), prediff, sv);
      }
      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(auto op = dyn_cast<InsertValueInst>(inst)) {
      if (gutils->isConstantValue(inst)) continue;
      auto st = cast<StructType>(op->getType());
      bool hasNonPointer = false;
      for(unsigned i=0; i<st->getNumElements(); i++) {
        if (!st->getElementType(i)->isPointerTy()) {
           hasNonPointer = true; 
        }
      }
      if (!hasNonPointer) continue;

      if (!gutils->isConstantValue(op->getInsertedValueOperand()) && !op->getInsertedValueOperand()->getType()->isPointerTy()) {
        auto prediff = gutils->diffe(inst, Builder2);
        auto dindex = Builder2.CreateExtractValue(prediff, op->getIndices());
        gutils->addToDiffe(op->getOperand(1), dindex, Builder2);
      }
      
      if (!gutils->isConstantValue(op->getAggregateOperand()) && !op->getAggregateOperand()->getType()->isPointerTy()) {
        auto prediff = gutils->diffe(inst, Builder2);
        auto dindex = Builder2.CreateInsertValue(prediff, Constant::getNullValue(op->getInsertedValueOperand()->getType()), op->getIndices());
        gutils->addToDiffe(op->getAggregateOperand(), dindex, Builder2);
      }

      gutils->setDiffe(inst, Constant::getNullValue(inst->getType()), Builder2);
    } else if (auto op = dyn_cast<ShuffleVectorInst>(inst)) {
      if (gutils->isConstantValue(inst)) continue;

      auto loaded = diffe(inst);
      size_t l1 = cast<VectorType>(op->getOperand(0)->getType())->getNumElements();
      uint64_t instidx = 0;
      for( size_t idx : op->getShuffleMask()) {
        auto opnum = (idx < l1) ? 0 : 1;
        auto opidx = (idx < l1) ? idx : (idx-l1);
        SmallVector<Value*,4> sv;
        sv.push_back(ConstantInt::get(Type::getInt32Ty(Context), opidx));
		if (!gutils->isConstantValue(op->getOperand(opnum)))
          addToDiffeIndexed(op->getOperand(opnum), Builder2.CreateExtractElement(loaded, instidx), sv);
        instidx++;
      }
      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(auto op = dyn_cast<ExtractElementInst>(inst)) {
      if (gutils->isConstantValue(inst)) continue;

	  if (!gutils->isConstantValue(op->getVectorOperand())) {
        SmallVector<Value*,4> sv;
        sv.push_back(op->getIndexOperand());
        addToDiffeIndexed(op->getVectorOperand(), diffe(inst), sv);
      }
      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(auto op = dyn_cast<InsertElementInst>(inst)) {
      if (gutils->isConstantValue(inst)) continue;

      auto dif1 = diffe(inst);

      if (!gutils->isConstantValue(op->getOperand(0)))
        addToDiffe(op->getOperand(0), Builder2.CreateInsertElement(dif1, Constant::getNullValue(op->getOperand(1)->getType()), lookup(op->getOperand(2)) ));

      if (!gutils->isConstantValue(op->getOperand(1)))
        addToDiffe(op->getOperand(1), Builder2.CreateExtractElement(dif1, lookup(op->getOperand(2))));

      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(auto op = dyn_cast<CastInst>(inst)) {
      if (gutils->isConstantValue(inst)) continue;
      if (op->getType()->isPointerTy()) continue;

	  if (!gutils->isConstantValue(op->getOperand(0))) {
        if (op->getOpcode()==CastInst::CastOps::FPTrunc || op->getOpcode()==CastInst::CastOps::FPExt) {
          addToDiffe(op->getOperand(0), Builder2.CreateFPCast(diffe(inst), op->getOperand(0)->getType()));
        } else if (op->getOpcode()==CastInst::CastOps::BitCast) {
          addToDiffe(op->getOperand(0), Builder2.CreateBitCast(diffe(inst), op->getOperand(0)->getType()));
        } else {
            llvm::errs() << *inst->getParent()->getParent() << "\n" << *inst->getParent() << "\n";
            llvm::errs() << "cannot handle above cast " << *inst << "\n";
            report_fatal_error("unknown instruction");
        }
      }
      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(isa<CmpInst>(inst) || isa<PHINode>(inst) || isa<BranchInst>(inst) || isa<SwitchInst>(inst) || isa<AllocaInst>(inst) || isa<CastInst>(inst) || isa<GetElementPtrInst>(inst)) {
        continue;
    } else {
      assert(inst);
      assert(inst->getParent());
      assert(inst->getParent()->getParent());
      llvm::errs() << *inst->getParent()->getParent() << "\n" << *inst->getParent() << "\n";
      llvm::errs() << "cannot handle above inst " << *inst << "\n";
      report_fatal_error("unknown instruction");
    }
  }
 
    createInvertedTerminator(gutils, BB, retAlloca, 0 + (additionalArg ? 1 : 0) + (differentialReturn ? 1 : 0));

  }
  
  if (!topLevel)
    gutils->eraseStructuralStoresAndCalls();

  while(gutils->inversionAllocs->size() > 0) {
    gutils->inversionAllocs->back().moveBefore(gutils->newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  }

  (IRBuilder <>(gutils->inversionAllocs)).CreateUnreachable();
  DeleteDeadBlock(gutils->inversionAllocs);
  for(auto BBs : gutils->reverseBlocks) {
    if (pred_begin(BBs.second) == pred_end(BBs.second)) {
        (IRBuilder <>(BBs.second)).CreateUnreachable();
        DeleteDeadBlock(BBs.second);
    }
  }
  
  for (Argument &Arg : gutils->newFunc->args()) {
      if (Arg.hasAttribute(Attribute::Returned))
          Arg.removeAttr(Attribute::Returned);
      if (Arg.hasAttribute(Attribute::StructRet))
          Arg.removeAttr(Attribute::StructRet);
  }
  if (gutils->newFunc->hasFnAttribute(Attribute::OptimizeNone))
    gutils->newFunc->removeFnAttr(Attribute::OptimizeNone);

  if (auto bytes = gutils->newFunc->getDereferenceableBytes(llvm::AttributeList::ReturnIndex)) {
    AttrBuilder ab;
    ab.addDereferenceableAttr(bytes);
    gutils->newFunc->removeAttributes(llvm::AttributeList::ReturnIndex, ab);
  }
  if (gutils->newFunc->hasAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias)) {
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias);
  }

  if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
      llvm::errs() << *gutils->oldFunc << "\n";
      llvm::errs() << *gutils->newFunc << "\n";
      report_fatal_error("function failed verification (4)");
  }

  optimizeIntermediate(gutils, topLevel, gutils->newFunc);

  auto nf = gutils->newFunc;
  delete gutils;

  return nf;
}

void HandleAutoDiff(CallInst *CI, TargetLibraryInfo &TLI, AAResults &AA) {//, LoopInfo& LI, DominatorTree& DT) {
  

  Value* fn = CI->getArgOperand(0);

  while (auto ci = dyn_cast<CastInst>(fn)) {
    fn = ci->getOperand(0);
  }
  while (auto ci = dyn_cast<BlockAddress>(fn)) {
    fn = ci->getFunction();
  }
  while (auto ci = dyn_cast<ConstantExpr>(fn)) {
    fn = ci->getOperand(0);
  }
  auto FT = cast<Function>(fn)->getFunctionType();
  assert(fn);
   
  if (autodiff_print)
      llvm::errs() << "prefn:\n" << *fn << "\n";

  std::set<unsigned> constants;
  SmallVector<Value*,2> args;

  unsigned truei = 0;
  IRBuilder<> Builder(CI);

  for(unsigned i=1; i<CI->getNumArgOperands(); i++) {
    Value* res = CI->getArgOperand(i);

    auto PTy = FT->getParamType(truei);
    DIFFE_TYPE ty = DIFFE_TYPE::CONSTANT;

    if (auto av = dyn_cast<MetadataAsValue>(res)) {
        auto MS = cast<MDString>(av->getMetadata())->getString();
        if (MS == "diffe_dup") {
            ty = DIFFE_TYPE::DUP_ARG;
        } else if(MS == "diffe_out") {
            llvm::errs() << "saw metadata for diffe_out\n";
            ty = DIFFE_TYPE::OUT_DIFF;
        } else if (MS == "diffe_const") {
            ty = DIFFE_TYPE::CONSTANT;
        } else {
            assert(0 && "illegal diffe metadata string");
        }
        i++;
        res = CI->getArgOperand(i);
    } else 
      ty = whatType(PTy);

    if (ty == DIFFE_TYPE::CONSTANT)
      constants.insert(truei);

    assert(truei < FT->getNumParams());
    if (PTy != res->getType()) {
        if (auto ptr = dyn_cast<PointerType>(res->getType())) {
            if (auto PT = dyn_cast<PointerType>(PTy)) {
                if (ptr->getAddressSpace() != PT->getAddressSpace()) {
                    res = Builder.CreateAddrSpaceCast(res, PointerType::get(ptr->getElementType(), PT->getAddressSpace()));
                    assert(res);
                    assert(PTy);
                    assert(FT);
                    llvm::errs() << "Warning cast(1) __builtin_autodiff argument " << i << " " << *res <<"|" << *res->getType()<< " to argument " << truei << " " << *PTy << "\n" << "orig: " << *FT << "\n";
                }
            }
        }
      if (!res->getType()->canLosslesslyBitCastTo(PTy)) {
        llvm::errs() << "Cannot cast(1) __builtin_autodiff argument " << i << " " << *res << "|"<< *res->getType() << " to argument " << truei << " " << *PTy << "\n" << "orig: " << *FT << "\n";
        report_fatal_error("Illegal cast(1)");
      }
      res = Builder.CreateBitCast(res, PTy);
    }

    args.push_back(res);
    if (ty == DIFFE_TYPE::DUP_ARG) {
      i++;

      Value* res = CI->getArgOperand(i);
      if (PTy != res->getType()) {
        if (auto ptr = dyn_cast<PointerType>(res->getType())) {
            if (auto PT = dyn_cast<PointerType>(PTy)) {
                if (ptr->getAddressSpace() != PT->getAddressSpace()) {
                    res = Builder.CreateAddrSpaceCast(res, PointerType::get(ptr->getElementType(), PT->getAddressSpace()));
                    assert(res);
                    assert(PTy);
                    assert(FT);
                    llvm::errs() << "Warning cast(2) __builtin_autodiff argument " << i << " " << *res <<"|" << *res->getType()<< " to argument " << truei << " " << *PTy << "\n" << "orig: " << *FT << "\n";
                }
            }
        }
        if (!res->getType()->canLosslesslyBitCastTo(PTy)) {
          assert(res);
          assert(res->getType());
          assert(PTy);
          assert(FT);
          llvm::errs() << "Cannot cast(2) __builtin_autodiff argument " << i << " " << *res <<"|" << *res->getType()<< " to argument " << truei << " " << *PTy << "\n" << "orig: " << *FT << "\n";
          report_fatal_error("Illegal cast(2)");
        }
        res = Builder.CreateBitCast(res, PTy);
      }
      args.push_back(res);
    }

    truei++;
  }

  bool differentialReturn = cast<Function>(fn)->getReturnType()->isFPOrFPVectorTy();
  
  auto newFunc = CreatePrimalAndGradient(cast<Function>(fn), constants, TLI, AA, /*should return*/false, differentialReturn, /*topLevel*/true, /*addedType*/nullptr);//, LI, DT);
  
  if (differentialReturn)
    args.push_back(ConstantFP::get(cast<Function>(fn)->getReturnType(), 1.0));
  assert(newFunc);
  if (autodiff_print)
    llvm::errs() << "postfn:\n" << *newFunc << "\n";
  Builder.setFastMathFlags(getFast());

  CallInst* diffret = cast<CallInst>(Builder.CreateCall(newFunc, args));
  diffret->setCallingConv(CI->getCallingConv());
  diffret->setDebugLoc(CI->getDebugLoc());
  if (!diffret->getType()->isEmptyTy()) {
    unsigned idxs[] = {0};
    auto diffreti = Builder.CreateExtractValue(diffret, idxs);
    CI->replaceAllUsesWith(diffreti);
  } else {
    CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
  }
  CI->eraseFromParent();
}

static bool lowerAutodiffIntrinsic(Function &F, TargetLibraryInfo &TLI, AAResults &AA) {//, LoopInfo& LI, DominatorTree& DT) {

  bool Changed = false;

reset:
  for (BasicBlock &BB : F) {

    for (auto BI = BB.rbegin(), BE = BB.rend(); BI != BE; BI++) {
      CallInst *CI = dyn_cast<CallInst>(&*BI);
      if (!CI) continue;

      Function *Fn = CI->getCalledFunction();
            
      if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledValue())) {
        if (castinst->isCast())
            if (auto fn = dyn_cast<Function>(castinst->getOperand(0)))
                Fn = fn;
      }

      if (Fn && ( Fn->getName() == "__enzyme_autodiff" || Fn->getName().startswith("__enzyme_autodiff")) ) {
        HandleAutoDiff(CI, TLI, AA);//, LI, DT);
        Changed = true;
        goto reset;
      }
    }
  }

  return Changed;
}

PHINode* canonicalizeIVs(Type *Ty, Loop *L, ScalarEvolution &SE, DominatorTree &DT, GradientUtils* gutils) {
    
    fake::SCEVExpander e(SE, L->getHeader()->getParent()->getParent()->getDataLayout(), "ad");
    
    PHINode *CanonicalIV = e.getOrInsertCanonicalInductionVariable(L, Ty);
    
    assert (CanonicalIV && "canonicalizing IV");
  
  SmallVector<WeakTrackingVH, 16> DeadInst0;
  e.replaceCongruentIVs(L, &DT, DeadInst0);
  for (WeakTrackingVH V : DeadInst0) {
    gutils->erase(cast<Instruction>(V)); //->eraseFromParent();
  }
  

  return CanonicalIV;
  
}

bool getContextM(BasicBlock *BB, LoopContext &loopContext, std::map<Loop*,LoopContext> &loopContexts, LoopInfo &LI,ScalarEvolution &SE,DominatorTree &DT, GradientUtils &gutils) {
    if (auto L = LI.getLoopFor(BB)) {
        if (loopContexts.find(L) != loopContexts.end()) {
            loopContext = loopContexts.find(L)->second;
            return true;
        }

        SmallVector<BasicBlock *, 8> PotentialExitBlocks;
        SmallPtrSet<BasicBlock *, 8> ExitBlocks;
        L->getExitBlocks(PotentialExitBlocks);
        for(auto a:PotentialExitBlocks) {

            SmallVector<BasicBlock*, 4> tocheck;
            SmallPtrSet<BasicBlock*, 4> checked;
            tocheck.push_back(a);

            bool isExit = false;

            while(tocheck.size()) {
                auto foo = tocheck.back();
                tocheck.pop_back();
                if (checked.count(foo)) {
                    isExit = true;
                    goto exitblockcheck;
                }
                checked.insert(foo);
                if(auto bi = dyn_cast<BranchInst>(foo->getTerminator())) {
                    for(auto nb : bi->successors()) {
                        if (L->contains(nb)) continue;
                        tocheck.push_back(nb);
                    }
                } else if (isa<UnreachableInst>(foo->getTerminator())) {
                    continue;
                } else {
                    isExit = true;
                    goto exitblockcheck;
                }
            }

            
            exitblockcheck:
            if (isExit) {
				ExitBlocks.insert(a);
            }
        }

        if (ExitBlocks.size() != 1) {
            assert(BB);
            assert(BB->getParent());
            assert(L);
            llvm::errs() << *BB->getParent() << "\n";
            llvm::errs() << *L << "\n";
			for(auto b:ExitBlocks) {
                assert(b);
                llvm::errs() << *b << "\n";
            }
			llvm::errs() << "offending: \n";
			llvm::errs() << "No unique exit block (1)\n";
        }

        BasicBlock* ExitBlock = *ExitBlocks.begin(); //[0];

        BasicBlock *Header = L->getHeader();
        BasicBlock *Preheader = L->getLoopPreheader();
        assert(Preheader && "requires preheader");
        BasicBlock *Latch = L->getLoopLatch();

        const SCEV *Limit = SE.getExitCount(L, Latch);
		SmallVector<PHINode*, 8> IVsToRemove;
        
		PHINode *CanonicalIV = nullptr;
		Value *LimitVar = nullptr;

        {

		if (SE.getCouldNotCompute() != Limit) {

        	CanonicalIV = canonicalizeIVs(Limit->getType(), L, SE, DT, &gutils);
        	if (!CanonicalIV) {
                report_fatal_error("Couldn't get canonical IV.");
        	}
        	
			const SCEVAddRecExpr *CanonicalSCEV = cast<const SCEVAddRecExpr>(SE.getSCEV(CanonicalIV));

        	assert(SE.isLoopBackedgeGuardedByCond(L, ICmpInst::ICMP_ULT,
                                              CanonicalSCEV, Limit) &&
               "Loop backedge is not guarded by canonical comparison with limit.");
        
            fake::SCEVExpander Exp(SE, Preheader->getParent()->getParent()->getDataLayout(), "ad");
			LimitVar = Exp.expandCodeFor(Limit, CanonicalIV->getType(),
                                            Preheader->getTerminator());

			loopContext.dynamic = false;
		} else {
          
          //llvm::errs() << "Se has any info: " << SE.getBackedgeTakenInfo(L).hasAnyInfo() << "\n";
          llvm::errs() << "SE could not compute loop limit.\n";

		  IRBuilder <>B(&Header->front());
		  CanonicalIV = B.CreatePHI(Type::getInt64Ty(Header->getContext()), 1); // should be Header->getNumPredecessors());

		  B.SetInsertPoint(Header->getTerminator());
		  auto inc = B.CreateNUWAdd(CanonicalIV, ConstantInt::get(CanonicalIV->getType(), 1));
		  CanonicalIV->addIncoming(inc, Latch);
		  for (BasicBlock *Pred : predecessors(Header)) {
			  if (Pred != Latch) {
				  CanonicalIV->addIncoming(ConstantInt::get(CanonicalIV->getType(), 0), Pred);
			  }
		  }

		  B.SetInsertPoint(&ExitBlock->front());
		  LimitVar = B.CreatePHI(CanonicalIV->getType(), 1); // should be ExitBlock->getNumPredecessors());

		  for (BasicBlock *Pred : predecessors(ExitBlock)) {
    		if (LI.getLoopFor(Pred) == L)
		    	cast<PHINode>(LimitVar)->addIncoming(CanonicalIV, Pred);
			else
				cast<PHINode>(LimitVar)->addIncoming(ConstantInt::get(CanonicalIV->getType(), 0), Pred);
		  }
		  loopContext.dynamic = true;
		}
	
		// Remove Canonicalizable IV's
		{
            fake::SCEVExpander Exp(SE, Preheader->getParent()->getParent()->getDataLayout(), "ad");
		  for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
			PHINode *PN = cast<PHINode>(II);
			if (PN == CanonicalIV) continue;
			if (!SE.isSCEVable(PN->getType())) continue;
			const SCEV *S = SE.getSCEV(PN);
			if (SE.getCouldNotCompute() == S) continue;
			Value *NewIV = Exp.expandCodeFor(S, S->getType(), CanonicalIV);
			if (NewIV == PN) {
				llvm::errs() << "TODO: odd case need to ensure replacement\n";
				continue;
			}
			PN->replaceAllUsesWith(NewIV);
			IVsToRemove.push_back(PN);
		  }
		}

        }

		for (PHINode *PN : IVsToRemove) {
		  gutils.erase(PN);
		}

        //if (SE.getCouldNotCompute() == Limit) {
        //Limit = SE.getMaxBackedgeTakenCount(L);
        //}
		assert(CanonicalIV);
		assert(LimitVar);
        loopContext.var = CanonicalIV;
        loopContext.limit = LimitVar;
        loopContext.antivar = PHINode::Create(CanonicalIV->getType(), CanonicalIV->getNumIncomingValues(), CanonicalIV->getName()+"'phi");
        loopContext.exit = ExitBlock;
        loopContext.latch = Latch;
        loopContext.preheader = Preheader;
		loopContext.header = Header;
        loopContext.parent = L->getParentLoop();

        loopContexts[L] = loopContext;
        return true;
    }
    return false;
  }


namespace {
/// Legacy pass for lowering expect intrinsics out of the IR.
///
/// When this pass is run over a function it uses expect intrinsics which feed
/// branches and switches to provide branch weight metadata for those
/// terminators. It then removes the expect intrinsics from the IR so the rest
/// of the optimizer can ignore them.
class Enzyme : public FunctionPass {
public:
  static char ID;
  Enzyme() : FunctionPass(ID) {
    //initializeLowerAutodiffIntrinsicPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<GlobalsAAWrapperPass>();
    AU.addRequiredID(LoopSimplifyID);
    //AU.addRequiredID(LCSSAID);
    
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
    auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    
    /*
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    */
  

    return lowerAutodiffIntrinsic(F, TLI, AA);
  }
};
}

char Enzyme::ID = 0;

static RegisterPass<Enzyme> X("enzyme", "Enzyme Pass");

FunctionPass *createEnzymePass() {
  return new Enzyme();
}

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>
#include <llvm/IR/Value.h>

#include "llvm/Transforms/Scalar.h"
#include "llvm-c/Initialization.h"
#include "llvm-c/Transforms/Scalar.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"

extern "C" void AddEnzymePass(LLVMPassManagerRef PM) {
    unwrap(PM)->add(createEnzymePass());
}
