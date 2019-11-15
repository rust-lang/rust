/*
 * ActiveVariable.cpp - Active Varaible Detection Utilities
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

#include <cstdint>

#include <llvm/Config/llvm-config.h>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/Support/raw_ostream.h"

#include "ActiveVariable.h"
#include "Utils.h"

using namespace llvm;

cl::opt<bool> printconst(
            "enzyme_printconst", cl::init(false), cl::Hidden,
            cl::desc("Print constant detection algorithm"));

cl::opt<bool> nonmarkedglobals_inactive(
            "enzyme_nonmarkedglobals_inactive", cl::init(false), cl::Hidden,
            cl::desc("Consider all nonmarked globals to be inactive"));

bool isKnownIntegerTBAA(Instruction* inst) {
  if (MDNode* md = inst->getMetadata(LLVMContext::MD_tbaa)) {
	if (md->getNumOperands() != 3) return false;
	Metadata* metadata = md->getOperand(1).get();
	if (auto mda = dyn_cast<MDNode>(metadata)) {
	  if (mda->getNumOperands() == 0) return false;
	  Metadata* metadata2 = mda->getOperand(0).get();
	  if (auto typeName = dyn_cast<MDString>(metadata2)) {
	    auto typeNameStringRef = typeName->getString();
	    if (typeNameStringRef == "long") {
		  return true; 
	    }
	  }
	}
  }
  return false;
}

void trackType(Type* et, SmallPtrSet<Type*, 4>& seen, Type*& floatingUse, bool& pointerUse) {
    if (seen.find(et) != seen.end()) return;
    seen.insert(et);
    
    if (et->isFloatingPointTy()) {
        if (floatingUse == nullptr) {
            floatingUse = et;
        } else {
            assert(floatingUse == et);
        }
    } else if (et->isPointerTy()) {
        pointerUse = true;
    }

    if (auto st = dyn_cast<SequentialType>(et)) {
        trackType(st->getElementType(), seen, floatingUse, pointerUse);
    } 
    
    if (auto st = dyn_cast<StructType>(et)) {
        for (auto innerType : st->elements()) {
            trackType(innerType, seen, floatingUse, pointerUse);
        }
    }
}
        
void trackPointer(Value* v, SmallPtrSet<Value*, 4> seen, SmallPtrSet<Type*, 4> typeseen, Type*& floatingUse, bool& pointerUse) {
    if (seen.find(v) != seen.end()) return;
    seen.insert(v);

    assert(v->getType()->isPointerTy());
    
    Type* et = cast<PointerType>(v->getType())->getElementType();
    trackType(et, typeseen, floatingUse, pointerUse);
            
    if (auto phi = dyn_cast<PHINode>(v)) {
        for(auto &a : phi->incoming_values()) {
            trackPointer(a.get(), seen, typeseen, floatingUse, pointerUse);
        }
    }
    if (auto ci = dyn_cast<CastInst>(v)) {
        if (ci->getSrcTy()->isPointerTy())
            trackPointer(ci->getOperand(0), seen, typeseen, floatingUse, pointerUse);
    } 
    if (auto gep = dyn_cast<GetElementPtrInst>(v)) {
        trackPointer(gep->getOperand(0), seen, typeseen, floatingUse, pointerUse);
    }
    if (auto inst = dyn_cast<Instruction>(v)) {
        for(User* use: inst->users()) {
            if (auto ci = dyn_cast<CastInst>(use)) {
                if (ci->getDestTy()->isPointerTy()) {
                    trackPointer(ci, seen, typeseen, floatingUse, pointerUse);
                }
            }
        }
    }
}

bool isIntASecretFloat(Value* val) {
    assert(val->getType()->isIntegerTy());

    if (isa<UndefValue>(val)) return true;
      
    if (auto cint = dyn_cast<ConstantInt>(val)) {
		if (!cint->isZero()) return false;
        return false;
        //llvm::errs() << *val << "\n";
        //assert(0 && "unsure if constant or not because constantint");
		 //if (cint->isOne()) return cint;
	}

    if (auto inst = dyn_cast<Instruction>(val)) {
        Type* floatingUse = nullptr;
        bool pointerUse = false;
        bool intUse = false;
        SmallPtrSet<Value*, 4> seen;
        
        SmallPtrSet<Type*, 4> typeseen;

        for(User* use: inst->users()) {
            if (auto ci = dyn_cast<BitCastInst>(use)) {
                if (ci->getDestTy()->isPointerTy()) {
                    pointerUse = true;
                    continue;
                }
                if (ci->getDestTy()->isFloatingPointTy()) {
                    floatingUse = ci->getDestTy();
                    continue;
                }
            }
                
            
            if (isa<IntToPtrInst>(use)) {
                pointerUse = true;
                continue;
            }
            
            if (auto si = dyn_cast<StoreInst>(use)) {
                assert(inst == si->getValueOperand());
				if (isKnownIntegerTBAA(si)) intUse = true;
                trackPointer(si->getPointerOperand(), seen, typeseen, floatingUse, pointerUse);
            }
        }

        if (auto li = dyn_cast<LoadInst>(inst)) {
			if (isKnownIntegerTBAA(li)) intUse = true;
            trackPointer(li->getOperand(0), seen, typeseen, floatingUse, pointerUse);
        }

        if (auto ci = dyn_cast<BitCastInst>(inst)) {
            if (ci->getSrcTy()->isPointerTy()) {
                pointerUse = true;
            }
            if (ci->getSrcTy()->isFloatingPointTy()) {
                floatingUse = ci->getSrcTy();
            }
        }
        
        if (isa<PtrToIntInst>(inst)) {
            pointerUse = true;
        }

        if (intUse  && !pointerUse && !floatingUse) return false; 
        if (!intUse && pointerUse && !floatingUse) return false; 
        if (!intUse && !pointerUse && floatingUse) return true;
        llvm::errs() << *inst->getParent()->getParent() << "\n";
        llvm::errs() << " val:" << *val << " pointer:" << pointerUse << " floating:" << floatingUse << " int:" << intUse << "\n";
        assert(0 && "ambiguous unsure if constant or not");
    }

    llvm::errs() << *val << "\n";
    assert(0 && "unsure if constant or not");
}

//! return the secret float type if found, otherwise nullptr
Type* isIntPointerASecretFloat(Value* val) {
    assert(val->getType()->isPointerTy());
    assert(cast<PointerType>(val->getType())->getElementType()->isIntegerTy());

    if (isa<UndefValue>(val)) return nullptr;
      
    if (auto cint = dyn_cast<ConstantInt>(val)) {
		if (!cint->isZero()) return nullptr;
        assert(0 && "unsure if constant or not because constantint");
		 //if (cint->isOne()) return cint;
	}

    Type* floatingUse = nullptr;
    bool pointerUse = false;

    SmallPtrSet<Type*, 4> typeseen;

    SmallPtrSet<Value*, 4> seen;

    trackPointer(val, seen, typeseen, floatingUse, pointerUse);

    if (pointerUse && (floatingUse == nullptr)) return nullptr; 
    if (!pointerUse && (floatingUse != nullptr)) return floatingUse;

    if (auto inst = dyn_cast<Instruction>(val)) {
        llvm::errs() << *inst->getParent()->getParent() << "\n";
    }
    llvm::errs() << " val:" << *val << " pointer:" << pointerUse << " floating:" << floatingUse << "\n";
    assert(0 && "ambiguous unsure if constant or not");
}

cl::opt<bool> ipoconst(
            "enzyme_ipoconst", cl::init(false), cl::Hidden,
            cl::desc("Interprocedural constant detection"));

cl::opt<bool> emptyfnconst(
            "enzyme_emptyfnconst", cl::init(false), cl::Hidden,
            cl::desc("Empty functions are considered constant"));

#include <set>
#include <map>
#include <unordered_map>
#include "llvm/IR/InstIterator.h"

/*
namespace std {
template <typename T>
struct hash<SmallPtrSetImpl<T>>
{
std::size_t operator()(const SmallPtrSetImpl<T>& k) const
{
  return std::hash<std::set<T>>(std::set<T>(k.begin(), k.end()));
}
};

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
};

template <typename A, typename B, typename C, typename D>
struct hash<tuple<A, B, C, D>>
{
std::size_t operator()(const tuple<A, B, C, D>& k) const
{
  return hash_combine(hash_combine(hash_combine(
					std::hash<A>(std::get<0>(k)),
					std::get<1>(k)),
					std::get<2>(k)),
					std::get<3>(k));
}
};

}
*/

bool isFunctionArgumentConstant(CallInst* CI, Value* val, SmallPtrSetImpl<Value*> &constants, SmallPtrSetImpl<Value*> &nonconstant, const SmallPtrSetImpl<Value*> &retvals, const SmallPtrSetImpl<Instruction*> &originalInstructions, int directions) {
    Function* F = CI->getCalledFunction();
    if (F == nullptr) return false;
    
    auto fn = F->getName();
    // todo realloc consider?
    // For known library functions, special case how derivatives flow to allow for more aggressive active variable detection
    if (fn == "malloc" || fn == "free" || fn == "_Znwm" || fn == "__cxa_guard_acquire" || fn == "__cxa_guard_release" || fn == "__cxa_guard_abort")
        return true;
    if (F->getIntrinsicID() == Intrinsic::memset && CI->getArgOperand(0) != val && CI->getArgOperand(1) != val)
        return true;
    if (F->getIntrinsicID() == Intrinsic::memcpy && CI->getArgOperand(0) != val && CI->getArgOperand(1) != val)
        return true;
    if (F->getIntrinsicID() == Intrinsic::memmove && CI->getArgOperand(0) != val && CI->getArgOperand(1) != val)
        return true;

    if (F->empty()) return false;

    return false;
    if (fn.startswith("augmented")) return false;
    if (fn.startswith("fakeaugmented")) return false;
    if (fn.startswith("diffe")) return false;
    if (val->getType()->isPointerTy()) return false;

    assert(retvals.find(val) == retvals.end());

    //static std::unordered_map<std::tuple<Function*, Value*, SmallPtrSet<Value*,20>, std::set<Value*> >, bool> metacache;
    static std::map<std::tuple<CallInst*, Value*, std::set<Value*>, std::set<Value*>, std::set<Value*> >, bool> metacache;
    //auto metatuple = std::make_tuple(F, val, SmallPtrSet<Value*,20>(constants.begin(), constants.end()), std::set<Value*>(nonconstant.begin(), nonconstant.end()));
    auto metatuple = std::make_tuple(CI, val, std::set<Value*>(constants.begin(), constants.end()), std::set<Value*>(nonconstant.begin(), nonconstant.end()), std::set<Value*>(retvals.begin(), retvals.end()));
    if (metacache.find(metatuple) != metacache.end()) {
		if (printconst)
        llvm::errs() << " < SUBFN metacache const " << F->getName() << "> arg: " << *val << " ci:" << *CI << "\n";
        return metacache[metatuple];
    }
    if (printconst)
       llvm::errs() << " < METAINDUCTIVE SUBFN const " << F->getName() << "> arg: " << *val << " ci:" << *CI << "\n";
    
    //metacache[metatuple] = true;
    //Note that the base case of true broke the up/down variant so have to be very conservative
    //  as a consequence we cannot detect const of recursive functions :'( [in that will be too conservative]
    metacache[metatuple] = false;

    SmallPtrSet<Value*, 20> constants2;
    constants2.insert(constants.begin(), constants.end());
    SmallPtrSet<Value*, 20> nonconstant2;
    nonconstant2.insert(nonconstant.begin(), nonconstant.end());
    SmallPtrSet<Value*, 20> retvals2;
    retvals2.insert(retvals.begin(), retvals.end());

    //Ask the question, even if is this is active, are all its uses inactive (meaning this use does not impact its activity)
    nonconstant2.insert(val);
    //retvals2.insert(val);
    
    //constants2.insert(val);

    if (printconst) {
        llvm::errs() << " < SUBFN " << F->getName() << "> arg: " << *val << " ci:" << *CI << "\n";
    }
    

    auto a = F->arg_begin();
    
    std::set<int> arg_constants;
    std::set<int> idx_findifactive;
    SmallPtrSet<Argument*, 20> arg_findifactive;
    
    SmallPtrSet<Value*, 20> newconstants;
    SmallPtrSet<Value*, 20> newnonconstant;
    
    for(unsigned i=0; i<CI->getNumArgOperands(); i++) {
        if (CI->getArgOperand(i) == val) {
            arg_findifactive.insert(a);
            idx_findifactive.insert(i);
            newnonconstant.insert(a);
            a++;
            continue;
        }

        if (isconstantValueM(CI->getArgOperand(i), constants2, nonconstant2, retvals2, originalInstructions), directions) {
            newconstants.insert(a);
            arg_constants.insert(i);
        } else {
            newnonconstant.insert(a);
        }
        a++;
    }

    bool constret = isconstantValueM(CI, constants2, nonconstant2, retvals2, originalInstructions, directions);
    
    if (constret) arg_constants.insert(-1);

    static std::map<std::tuple<std::set<int>, Function*, std::set<int> >, bool> cache;

    auto tuple = std::make_tuple(arg_constants, F, idx_findifactive);
    if (cache.find(tuple) != cache.end()) {
		if (printconst) 
        llvm::errs() << " < SUBFN cache const " << F->getName() << "> arg: " << *val << " ci:" << *CI << "\n";
        return cache[tuple];
    }
    
    //! inductively assume that it is constant, it should be deduced nonconstant elsewhere if this is not the case
    if (printconst)
       llvm::errs() << " < INDUCTIVE SUBFN const " << F->getName() << "> arg: " << *val << " ci:" << *CI << "\n";
    cache[tuple] = true;
    //Note that the base case of true broke the up/down variant so have to be very conservative
    //  as a consequence we cannot detect const of recursive functions :'( [in that will be too conservative]
    cache[tuple] = false;
    
    SmallPtrSet<Instruction*,4> newinsts;
    SmallPtrSet<Value*,4> newretvals;
	newretvals.insert(retvals2.begin(), retvals2.end());
    for (llvm::inst_iterator I = llvm::inst_begin(F), E = llvm::inst_end(F); I != E; ++I) {
        newinsts.insert(&*I);
        if (!constret) {
            if (auto ri = dyn_cast<ReturnInst>(&*I)) {
                newretvals.insert(ri->getReturnValue());
            }
        }
    }
    
    
    for(auto specialarg : arg_findifactive) {
        for(auto user : specialarg->users()) {
			if (printconst)
			llvm::errs() << " going to consider user " << *user << "\n";
            if (!isconstantValueM(user, newconstants, newnonconstant, newretvals, newinsts, 3)) {
                if (printconst)
                    llvm::errs() << " < SUBFN nonconst " << F->getName() << "> arg: " << *val << " ci:" << *CI << "  from sf: " << *user << "\n";
				metacache.erase(metatuple);
                return cache[tuple] = false;
            }
        }
    }

    constants.insert(constants2.begin(), constants2.end());
    nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
    if (printconst) {
        llvm::errs() << " < SUBFN const " << F->getName() << "> arg: " << *val << " ci:" << *CI << "\n";
    }
	metacache.erase(metatuple);
    return cache[tuple] = true;
}


// TODO separate if the instruction is constant (i.e. could change things)
//    from if the value is constant (the value is something that could be differentiated)
bool isconstantM(Instruction* inst, SmallPtrSetImpl<Value*> &constants, SmallPtrSetImpl<Value*> &nonconstant, SmallPtrSetImpl<Value*> &retvals, const SmallPtrSetImpl<Instruction*> &originalInstructions, uint8_t directions) {
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
			if (called->getName() == "__assert_fail" || called->getName() == "free" || called->getName() == "_ZdlPv" || called->getName() == "_ZdlPvm"
                    || called->getName() == "__cxa_guard_acquire" || called->getName() == "__cxa_guard_release" || called->getName() == "__cxa_guard_abort") {
				constants.insert(inst);
				return true;
			}
            if (called->empty() && !hasMetadata(called, "enzyme_gradient") && emptyfnconst) {
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
	
	if (isa<LoadInst>(inst) || isa<StoreInst>(inst)) {
		if (isKnownIntegerTBAA(inst)) {
			if (printconst)
				llvm::errs() << " constant instruction from TBAA " << *inst << "\n";
			constants.insert(inst);
			return true;
		}
	}

    /* TODO consider constant stores
    if (auto si = dyn_cast<StoreInst>(inst)) {
		SmallPtrSet<Value*, 20> constants2;
		constants2.insert(constants.begin(), constants.end());
		SmallPtrSet<Value*, 20> nonconstant2;
		nonconstant2.insert(nonconstant.begin(), nonconstant.end());
		SmallPtrSet<Value*, 20> retvals2;
		retvals2.insert(retvals.begin(), retvals.end());
		constants2.insert(inst);
        if (isconstantValueM(si->getValueOperand(), constants2, nonconstant2, retvals2, originalInstructions, directions)) {
			constants.insert(inst);
			constants.insert(constants2.begin(), constants2.end());
            constants.insert(constants_tmp.begin(), constants_tmp.end());

			// not here since if had full updown might not have been nonconstant
			//nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
    		if (printconst)
			  llvm::errs() << "constant(" << (int)directions << ") store:" << *inst << "\n";
			return true;
        }
    }
    */

    if (printconst)
	  llvm::errs() << "checking if is constant[" << (int)directions << "] " << *inst << "\n";

	SmallPtrSet<Value*, 20> constants_tmp;

    if (inst->getType()->isPointerTy()) {
		//Proceed assuming this is constant, can we prove this should be constant otherwise
		SmallPtrSet<Value*, 20> constants2;
		constants2.insert(constants.begin(), constants.end());
		SmallPtrSet<Value*, 20> nonconstant2;
		nonconstant2.insert(nonconstant.begin(), nonconstant.end());
		SmallPtrSet<Value*, 20> retvals2;
		retvals2.insert(retvals.begin(), retvals.end());
		constants2.insert(inst);

		if (printconst)
			llvm::errs() << " < MEMSEARCH" << (int)directions << ">" << *inst << "\n";

        {
		SmallPtrSet<Value*, 20> constants2;
		constants2.insert(constants.begin(), constants.end());
		SmallPtrSet<Value*, 20> nonconstant2;
		nonconstant2.insert(nonconstant.begin(), nonconstant.end());
		SmallPtrSet<Value*, 20> retvals2;
		retvals2.insert(retvals.begin(), retvals.end());
		nonconstant2.insert(inst);
		for (const auto &a:inst->users()) {
		  if (isa<LoadInst>(a)) {
		      if (!isconstantValueM(a, constants2, nonconstant2, retvals2, originalInstructions, directions)) {
				if (directions == 3)
				  nonconstant.insert(inst);
    			if (printconst)
				  llvm::errs() << "memory(" << (int)directions << ")  erase 3: " << *inst << "\n";
				return false;
              }
              continue;
          }
		}

        }

		for (const auto &a:inst->users()) {
		  if(auto store = dyn_cast<StoreInst>(a)) {

			if (inst == store->getPointerOperand() && !isconstantValueM(store->getValueOperand(), constants2, nonconstant2, retvals2, originalInstructions, directions)) {
				if (directions == 3)
				  nonconstant.insert(inst);
    			if (printconst)
				  llvm::errs() << "memory(" << (int)directions << ")  erase 1: " << *inst << "\n";
				return false;
			}
			if (inst == store->getValueOperand() && !isconstantValueM(store->getPointerOperand(), constants2, nonconstant2, retvals2, originalInstructions, directions)) {
				if (directions == 3)
				  nonconstant.insert(inst);
    			if (printconst)
				  llvm::errs() << "memory(" << (int)directions << ")  erase 2: " << *inst << "\n";
				return false;
			}
		  } else if (isa<LoadInst>(a)) {
              /*
		      if (!isconstantValueM(a, constants2, nonconstant2, retvals2, originalInstructions, directions)) {
				if (directions == 3)
				  nonconstant.insert(inst);
    			if (printconst)
				  llvm::errs() << "memory(" << (int)directions << ")  erase 3: " << *inst << "\n";
				return false;
              }
              */
              continue;
          } else {
			if (!isconstantM(cast<Instruction>(a), constants2, nonconstant2, retvals2, originalInstructions, directions)) {
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

	if (!inst->getType()->isPointerTy() && ( !inst->mayWriteToMemory() || isa<BinaryOperator>(inst) ) && (directions & DOWN) && (retvals.find(inst) == retvals.end()) ) { 
		//Proceed assuming this is constant, can we prove this should be constant otherwise
		SmallPtrSet<Value*, 20> constants2;
		constants2.insert(constants.begin(), constants.end());
		SmallPtrSet<Value*, 20> nonconstant2;
		nonconstant2.insert(nonconstant.begin(), nonconstant.end());
		SmallPtrSet<Value*, 20> retvals2;
		retvals2.insert(retvals.begin(), retvals.end());
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
            if (isa<AllocaInst>(a)) {
               if (printconst)
			     llvm::errs() << "found constant(" << (int)directions << ")  allocainst use:" << *inst << " user " << *a << "\n";
               continue;
            }

			if (auto call = dyn_cast<CallInst>(a)) {
                if (isFunctionArgumentConstant(call, inst, constants2, nonconstant2, retvals2, originalInstructions, DOWN)) {
                    continue;
                }
			}

		  	if (!isconstantM(cast<Instruction>(a), constants2, nonconstant2, retvals2, originalInstructions, DOWN)) {
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
	SmallPtrSet<Value*, 20> retvals2;
	retvals2.insert(retvals.begin(), retvals.end());
	constants2.insert(inst);
		
	if (directions & UP) {
        if (printconst)
		    llvm::errs() << " < UPSEARCH" << (int)directions << ">" << *inst << "\n";
        if (auto gep = dyn_cast<GetElementPtrInst>(inst)) {
            // Handled uses above
            if (!isconstantValueM(gep->getPointerOperand(), constants2, nonconstant2, retvals2, originalInstructions, UP)) {
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
                if (!isconstantValueM(a, constants2, nonconstant2, retvals2, originalInstructions, UP)) {
                    if (directions == 3)
                      nonconstant.insert(inst);
                    if (printconst)
                      llvm::errs() << "nonconstant(" << (int)directions << ")  call " << *inst << " op " << *a << "\n";
                    return false;
                }
            }
            
            //! TODO consider calling interprocedural here
            //! TODO: Really need an attribute that determines whether a function can access a global (not even necessarily read)
            //if (ci->hasFnAttr(Attribute::ReadNone) || ci->hasFnAttr(Attribute::ArgMemOnly)) 
            {
                constants.insert(inst);
                constants.insert(constants2.begin(), constants2.end());
                constants.insert(constants_tmp.begin(), constants_tmp.end());
                //if (directions == 3)
                //  nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
                if (printconst)
                  llvm::errs() << "constant(" << (int)directions << ")  call:" << *inst << "\n";
                return true;
            }
        } else {
            for(auto& a: inst->operands()) {
                if (!isconstantValueM(a, constants2, nonconstant2, retvals2, originalInstructions, UP)) {
                    if (directions == 3)
                      nonconstant.insert(inst);
                    if (printconst)
                      llvm::errs() << "nonconstant(" << (int)directions << ")  inst " << *inst << " op " << *a << "\n";
                    return false;
                }
            }

            //if (!isa<StoreInst>(inst) && !inst->getType()->isPointerTy()) {
                constants.insert(inst);
                constants.insert(constants2.begin(), constants2.end());
                constants.insert(constants_tmp.begin(), constants_tmp.end());
                //if (directions == 3)
                //  nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
                if (printconst)
                  llvm::errs() << "constant(" << (int)directions << ")  inst:" << *inst << "\n";
                return true;
            //}
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
bool isconstantValueM(Value* val, SmallPtrSetImpl<Value*> &constants, SmallPtrSetImpl<Value*> &nonconstant, SmallPtrSetImpl<Value*> &retvals, const SmallPtrSetImpl<Instruction*> &originalInstructions, uint8_t directions) {
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

    //All arguments should be marked constant/nonconstant ahead of time
    if (isa<Argument>(val)) {
        if((nonconstant.find(val) != nonconstant.end())) {
		    if (printconst)
		      llvm::errs() << " VALUE nonconst from arg nonconst " << *val << "\n";
            return false;
        }
        llvm::errs() << *(cast<Argument>(val)->getParent()) << "\n";
	llvm::errs() << *val << "\n";
        assert(0 && "must've put arguments in constant/nonconstant");
    }

    if (auto gi = dyn_cast<GlobalVariable>(val)) {
        if (!hasMetadata(gi, "enzyme_shadow") && nonmarkedglobals_inactive) {
            constants.insert(val);
            return true;
        }
        //TODO consider this more
        if (gi->isConstant() && isconstantValueM(gi->getInitializer(), constants, nonconstant, retvals, originalInstructions, directions)) {
            constants.insert(val);
            return true;
        }
    }

    if (auto ce = dyn_cast<ConstantExpr>(val)) {
        if (ce->isCast()) {
            if (isconstantValueM(ce->getOperand(0), constants, nonconstant, retvals, originalInstructions, directions)) {
                constants.insert(val);
                return true;
            }
        }
        if (ce->isGEPWithNoNotionalOverIndexing()) {
            if (isconstantValueM(ce->getOperand(0), constants, nonconstant, retvals, originalInstructions, directions)) {
                constants.insert(val);
                return true;
            }
        }
    }
    
    if (auto inst = dyn_cast<Instruction>(val)) {
        if (isconstantM(inst, constants, nonconstant, retvals, originalInstructions, directions)) return true;
    }
   
    if (!val->getType()->isPointerTy() && (directions & DOWN) && (retvals.find(val) == retvals.end()) ) { 
		auto &constants2 = constants;
		auto &nonconstant2 = nonconstant;
		auto &retvals2 = retvals;

		if (printconst)
			llvm::errs() << " <Value USESEARCH" << (int)directions << ">" << *val << "\n";

		bool seenuse = false;
		
        for (const auto &a:val->users()) {
			if (isa<Instruction>(a) && originalInstructions.find(cast<Instruction>(a)) == originalInstructions.end()) continue;

		    if (printconst)
			  llvm::errs() << "      considering use of " << *val << " - " << *a << "\n";

			if (auto gep = dyn_cast<GetElementPtrInst>(a)) {
				assert(val != gep->getPointerOperand());
				continue;
			}
			if (auto call = dyn_cast<CallInst>(a)) {
                if (isFunctionArgumentConstant(call, val, constants2, nonconstant2, retvals2, originalInstructions, DOWN)) {
                    if (printconst)
			          llvm::errs() << "Value found constant callinst use:" << *val << " user " << *call << "\n";
                    continue;
                }
			}
            if (isa<AllocaInst>(a)) {
               if (printconst)
			     llvm::errs() << "Value found constant allocainst use:" << *val << " user " << *a << "\n";
               continue;
            }
            
		  	if (!isconstantM(cast<Instruction>(a), constants2, nonconstant2, retvals2, originalInstructions, DOWN)) {
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
    retvals.insert(val);
    return false;
}
