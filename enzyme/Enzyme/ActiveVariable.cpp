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

#include "llvm/IR/InstIterator.h"

#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/InlineAsm.h"

#include "ActiveVariable.h"
#include "Utils.h"

#include "TBAA.h"

using namespace llvm;

cl::opt<bool> printconst(
            "enzyme_printconst", cl::init(false), cl::Hidden,
            cl::desc("Print constant detection algorithm"));

cl::opt<bool> nonmarkedglobals_inactive(
            "enzyme_nonmarkedglobals_inactive", cl::init(false), cl::Hidden,
            cl::desc("Consider all nonmarked globals to be inactive"));

bool isKnownIntegerTBAA(Instruction* inst) {
    auto typeNameStringRef = getAccessNameTBAA(inst);
    if (typeNameStringRef == "long long" || typeNameStringRef == "long" || typeNameStringRef == "int" || typeNameStringRef == "bool") {// || typeNameStringRef == "omnipotent char") {
        if (printconst) {
            llvm::errs() << "known tbaa " << *inst << " " << typeNameStringRef << "\n";
        }
	    return true; 
    } else {
        //if (printconst)
        //    llvm::errs() << "unknown tbaa " << *inst << " " << typeNameStringRef << "\n";
    }
    return false;
}

bool isKnownPointerTBAA(Instruction* inst) {
    auto typeNameStringRef = getAccessNameTBAA(inst);
    if (typeNameStringRef == "any pointer" || typeNameStringRef == "vtable pointer") {// || typeNameStringRef == "omnipotent char") {
        if (printconst) {
            llvm::errs() << "known tbaa " << *inst << " " << typeNameStringRef << "\n";
        }
        return true; 
    } else {
        //if (printconst)
        //    llvm::errs() << "unknown tbaa " << *inst << " " << typeNameStringRef << "\n";
    }
    return false;
}

Type* isKnownFloatTBAA(Instruction* inst) {
    auto typeNameStringRef = getAccessNameTBAA(inst);
    if (typeNameStringRef == "float") {
        if (printconst)
            llvm::errs() << "known tbaa " << *inst << " " << typeNameStringRef << "\n";
        return Type::getFloatTy(inst->getContext());
    } else if (typeNameStringRef == "double") {
        if (printconst)
            llvm::errs() << "known tbaa " << *inst << " " << typeNameStringRef << "\n";
        return Type::getDoubleTy(inst->getContext());
    } else {
        //if (printconst)
        //    llvm::errs() << "unknown tbaa " << *inst << " " << typeNameStringRef << "\n";
    }
    return nullptr;
}

cl::opt<bool> fast_tracking(
            "enzyme_fast_tracking", cl::init(true), cl::Hidden,
            cl::desc("Enable fast tracking (e.g. don't verify if inconsistent state)"));

// returns whether it found something
bool trackType(Type* et, SmallPtrSet<Type*, 4>& seen, Type*& floatingUse, bool& pointerUse, bool onlyFirst, std::vector<int> indices = {}) {
    if (seen.find(et) != seen.end()) return false;
    seen.insert(et);
    
    /*
    llvm::errs() << "  track type of saw " << *et << "\n";
    llvm::errs() << "       indices = [";
    for(auto a: indices) {
        llvm::errs() << a << ",";
    }
    llvm::errs() << "] of:" << onlyFirst << "\n";
    */
    
    
    if (et->isFloatingPointTy()) {
        if (floatingUse == nullptr) {
            //llvm::errs() << "  tract type saw(f) " << *et << " " << *et << "\n";
            floatingUse = et;
        } else {
            assert(floatingUse == et);
        }
        if (fast_tracking) return true;
    } else if (et->isPointerTy()) {
        //llvm::errs() << "  tract type saw(p) " << *et << "\n";
        pointerUse = true;
        if (fast_tracking) return true;
    }

    if (auto st = dyn_cast<SequentialType>(et)) {
        if (trackType(st->getElementType(), seen, floatingUse, pointerUse, onlyFirst, indices)) {
            if (fast_tracking) return true;
        }
    } 
    
    if (auto st = dyn_cast<StructType>(et)) {
        int index = -1;
        auto nindices = indices;
        if (indices.size() > 0) {
            index = indices[0];
            nindices.erase(nindices.begin());
        }
        if (onlyFirst) {
            if (index >= 0) {
                if (index >= st->getNumElements()) {
                    llvm::errs() << *st << " " << index << "\n";
                }
                assert(index < st->getNumElements());

                if(trackType(st->getElementType(index), seen, floatingUse, pointerUse, onlyFirst, nindices)) {
                    if (fast_tracking) return true;
                }
            } else {
                for (auto innerType : st->elements()) {
                    if (trackType(innerType, seen, floatingUse, pointerUse, onlyFirst, nindices)) {
                        if (fast_tracking) return true;
                    }
                }
            }
        } else {
            for (auto innerType : st->elements()) {
                if (trackType(innerType, seen, floatingUse, pointerUse, false, {})) {
                    if (fast_tracking) return true;
                }
            }
        }
    }

    return false;
}

// This function looks for recusion in the trace and removes the cycle
// We should always prefer using the older version of this if possible (to ensure that type info is kept coming in)
void addCallRemovingCycle(std::vector<CallInst*>& newtrace, CallInst* call) {
    for(int i=newtrace.size()-1; i>=0; i--) {
        if (newtrace[i] == call) {
            bool failedcycle = false;
            for(int j=0; ;j++) {
                // finished cycle
                if (newtrace.size()-1-j == i) break;

                // out of bounds
                if (i-1-j < 0) {
                    failedcycle = true;
                    break;
                }

                if (newtrace[i-1-j] != newtrace[newtrace.size()-1-j]) {
                    //llvm::errs() << "i: " << i << "j: " << j << " size:" << newtrace.size() << " nt[i-j]" << *newtrace[i-j] << " nt[s-1-j]" << *newtrace[newtrace.size()-1-j] << "\n";
                    failedcycle = true;
                    break;
                }
            }
            if (!failedcycle) {
                //erase all elements after i
                newtrace.resize(i+1);
                assert(newtrace.back() == call);
                return;
            }
        }
    }
    newtrace.push_back(call);
    //assert(newtrace.size() < 12);
}


bool trackInt(const std::map<Argument*, DataType> typeInfo, const std::vector<CallInst*> trace, Value* v, std::map<std::tuple<const std::vector<CallInst*>, Value*,bool>, IntType> intseen, std::set<std::tuple<const std::vector<CallInst*>, Value*>> ptrseen, SmallPtrSet<Type*, 4> typeseen, Type*& floatingUse, bool& pointerUse, bool& intUse, bool& unknownUse, bool shouldConsiderUnknownUse=false, bool* sawReturn=nullptr /*if sawReturn != nullptr, we can ignore uses of returninst, setting the bool to true if we see one*/);
        
bool trackPointer(const std::map<Argument*, DataType> typeInfo, const std::vector<CallInst*> trace, Value* v, std::map<std::tuple<const std::vector<CallInst*>, Value*,bool>, IntType> intseen, std::set<std::tuple<const std::vector<CallInst*>, Value*>> seen, SmallPtrSet<Type*, 4> typeseen, Type*& floatingUse, bool& pointerUse, bool &intUse, bool* sawReturn, bool onlyFirst, std::vector<int> indices = {}) {
    auto idx = std::tuple<const std::vector<CallInst*>, Value*>(trace, v);
    if (seen.find(idx) != seen.end()) return false;
    seen.insert(idx);

    assert(v->getType()->isPointerTy());
    
    Type* et = cast<PointerType>(v->getType())->getElementType();

    if (!onlyFirst) {
        if (auto arg = dyn_cast<Argument>(v)) {
            auto fd = typeInfo.find(arg);
            if (fd != typeInfo.end()) {
                switch(fd->second.typeEnum) {
                    case IntType::Unknown:
                        break;
                    case IntType::Float:
                        floatingUse = fd->second.type;
                        if (fast_tracking) return true;
                        break;
                    case IntType::Integer:
                        intUse = true;
                        if (fast_tracking) return true;
                        break;
                    case IntType::Pointer:
                        pointerUse = true;
                        if (fast_tracking) return true;
                        break;
                }
            } else if (trace.size() == 0) {
                llvm::errs() << "couldn't find arg: " << *arg << "(" << arg->getParent()->getName() << ") " << "\n";
                for(auto &pair : typeInfo) {
                    llvm::errs() << "    + option: " << *pair.first << "(" << pair.first->getParent()->getName() << ") " << "\n";
                }
                if (typeInfo.size() != 0) //} && arg->getParent()->getName()) 
                    assert(0 && "no arg for tracking");
            }
        }
    }

    /*
    llvm::errs() << "  tract pointer of saw " << *v << " et:" << *et << "\n";
    llvm::errs() << "       indices = [";
    for(auto a: indices) {
        llvm::errs() << a << ",";
    }
    llvm::errs() << "]\n";
    */
    
    

    if (trackType(et, typeseen, floatingUse, pointerUse, onlyFirst, indices)) {
        if (fast_tracking) return true;
    }
            
    if (auto phi = dyn_cast<PHINode>(v)) {
        for(auto &a : phi->incoming_values()) {
            if (trackPointer(typeInfo, trace, a.get(), intseen, seen, typeseen, floatingUse, pointerUse, intUse, sawReturn, onlyFirst, indices)) {
                if (fast_tracking) return true;
            }
        }
    }

    if (auto ci = dyn_cast<CastInst>(v)) {
        std::vector<int> idnext;
        //only propagate type information if it is safe to do so (e.g. we either look at any memory inside of [thus not onlyFirst], or we don't have any indexing inside of [thus indices.size() == 0])
        if (!onlyFirst || indices.size() == 0) {
            if (ci->getSrcTy()->isPointerTy()) {
                if (trackPointer(typeInfo, trace, ci->getOperand(0), intseen, seen, typeseen, floatingUse, pointerUse, intUse, sawReturn, onlyFirst, {})) {
                    if (fast_tracking) return true;
                }
            } else {
                //If the parent has another cast to a pointer type, let us consider that cast as well
                //  Specifically, given x is an integer (representing a pointer)
                //    we know ci is (int*)x; if there is later a (float*)x; we can use the type information from (float*)x
                for(User* use2 : ci->getOperand(0)->users()) {
                    //only consider pointer results
                    if (!use2->getType()->isPointerTy()) continue;

                    //we don't need to (but can) revisit this cast
                    if (use2 == ci) continue;

                    if (auto ci2 = dyn_cast<CastInst>(use2)) {
                        if (trackPointer(typeInfo, trace, ci2, intseen, seen, typeseen, floatingUse, pointerUse, intUse, sawReturn, onlyFirst, {})) {
                            if (fast_tracking) return true;
                        } 
                    }
                }
            }
        }
    } 
    if (auto gep = dyn_cast<GetElementPtrInst>(v)) {
        bool first = true;
        
        std::vector<int> idnext;

        for(auto& a : gep->indices()) {
            if (first) { first = false; continue; }
            if (auto ci = dyn_cast<ConstantInt>(a)) {
                idnext.push_back((int)ci->getLimitedValue());
            } else {
                idnext.push_back(-1);
            }
        }
        idnext.insert(idnext.end(), indices.begin(), indices.end());
        if (trackPointer(typeInfo, trace, gep->getOperand(0), intseen, seen, typeseen, floatingUse, pointerUse, intUse, sawReturn, onlyFirst, idnext)) {
            if (fast_tracking) return true;
        }
    }
    
    if (auto call = dyn_cast<CallInst>(v)) {
        if (isKnownIntegerTBAA(call)) {
            intUse = true;  
            if (fast_tracking) return true;
        }

        if (isKnownPointerTBAA(call)) {
            pointerUse = true;
            if (fast_tracking) return true;
        }

        if (Type* t = isKnownFloatTBAA(call)) {
            if (floatingUse == nullptr) {
                floatingUse = t;
            } else {
                assert(floatingUse == t);
            }
            if (fast_tracking) return true;
        }
    }

    if (!isa<Constant>(v)) {
        for(User* use: v->users()) {
            if (auto ci = dyn_cast<CastInst>(use)) {
                std::vector<int> idnext;//TODO insert old onlyFirst & indices from above

                //only propagate type information if it is safe to do so (e.g. we either look at any memory inside of [thus not onlyFirst], or we don't have any indexing inside of [thus indices.size() == 0])
                if (!onlyFirst || indices.size() == 0) {
                    if (ci->getDestTy()->isPointerTy()) {
                        if (trackPointer(typeInfo, trace, ci, intseen, seen, typeseen, floatingUse, pointerUse, intUse, sawReturn, onlyFirst, idnext)) {
                            if (fast_tracking) return true;
                        }
                    } else {
                    }
                }
            }

            if (isa<ReturnInst>(use)) {
                if (sawReturn) *sawReturn = true;
                continue;
            }

            if (auto gep = dyn_cast<GetElementPtrInst>(use)) {
                //TODO consider implication of gep indices on onlyfirst/indices

                std::vector<int> idnext;
                size_t cnt = 0;
                size_t vcnt = 0;

                //We should not track geps that we no [from indices] do not point to the same memory location as we may get incorrect type information
                bool notrack = false;
                for(auto &a : gep->indices()) {
                    if (vcnt == indices.size()) break;
                    if (cnt != 0) {
                        if (auto ci = dyn_cast<ConstantInt>(&a)) {
                            if (indices[vcnt] != ci->getLimitedValue()) {
                                notrack = true;
                            }
                        } else {
                            if (indices[vcnt] != -1) notrack = true;
                        }
                        vcnt++;
                    }
                    cnt++;
                }
                if (notrack) continue;

                for(; vcnt < indices.size(); vcnt++) {
                    idnext.push_back(indices[vcnt]);
                }

                if (trackPointer(typeInfo, trace, gep, intseen, seen, typeseen, floatingUse, pointerUse, intUse, sawReturn, onlyFirst, idnext)) {
                    if (fast_tracking) return true;
                }
            }

            if (auto call = dyn_cast<CallInst>(use)) {
                if (Function* ci = call->getCalledFunction()) {

                    //If memcpy / memmove of pointer, we can propagate type information from src to dst and vice versa
                    if (ci->getIntrinsicID() == Intrinsic::memcpy || ci->getIntrinsicID() == Intrinsic::memmove) {
                        if (call->getArgOperand(0) == v) {
                            //! If TBAA lets us ascertain information, let's use it
                            if (isKnownIntegerTBAA(call)) {
                                intUse = true;  
                                if (fast_tracking) return true;
                            }

                            if (isKnownPointerTBAA(call)) {
                                pointerUse = true;
                                if (fast_tracking) return true;
                            }

                            if (Type* t = isKnownFloatTBAA(call)) {
                                if (floatingUse == nullptr) {
                                    floatingUse = t;
                                } else {
                                    assert(floatingUse == t);
                                }
                                if (fast_tracking) return true;
                            }

                            if (trackPointer(typeInfo, trace, call->getArgOperand(1), intseen, seen, typeseen, floatingUse, pointerUse, intUse, sawReturn, onlyFirst, indices)) {
                                if (fast_tracking) return true;
                            }
                        }
                        if (call->getArgOperand(1) == v) {
                            //! If TBAA lets us ascertain information, let's use it
                            if (isKnownIntegerTBAA(call)) {
                                intUse = true;  
                                if (fast_tracking) return true;
                            }

                            if (isKnownPointerTBAA(call)) {
                                pointerUse = true;
                                if (fast_tracking) return true;
                            }

                            if (Type* t = isKnownFloatTBAA(call)) {
                                if (floatingUse == nullptr) {
                                    floatingUse = t;
                                } else {
                                    assert(floatingUse == t);
                                }
                                if (fast_tracking) return true;
                            }

                            if (trackPointer(typeInfo, trace, call->getArgOperand(0), intseen, seen, typeseen, floatingUse, pointerUse, intUse, sawReturn, onlyFirst, indices)) {
                                if (fast_tracking) return true;
                            }
                        }
                    }

                    //TODO we should handle calls interprocedurally, allowing better propagation of type information
                    if (!ci->empty()) {
                        auto a = ci->arg_begin();
                        bool subsawReturn=false;
                        std::vector<CallInst*> newtrace(trace);
                        addCallRemovingCycle(newtrace, call);
                        for(size_t i=0; i<call->getNumArgOperands(); i++) {
                            if (call->getArgOperand(i) == v) {
                                //TODO consider allowing return to be ignored as an unknown use below, so long as we also then look at use of call value itself
                                if(trackPointer(typeInfo, newtrace, a, intseen, seen, typeseen, floatingUse, pointerUse, intUse, &subsawReturn, onlyFirst, indices)) {
                                    if (fast_tracking) return true;
                                }
                            }
                            a++;
                        }
                        if (subsawReturn) {
                            if(trackPointer(typeInfo, trace, call, intseen, seen, typeseen, floatingUse, pointerUse, intUse, sawReturn, onlyFirst, indices)) {
                                if (fast_tracking) return true;
                            }
                        }
                    }
                }
            }

            if (auto li = dyn_cast<LoadInst>(use)) {
                bool unknownuse;
                if (li->getType()->isIntOrIntVectorTy())
                if (trackInt(typeInfo, trace, li, intseen, seen, typeseen, floatingUse, pointerUse, intUse, unknownuse, /*shouldconsiderunknownuse*/false)) {
                    if (fast_tracking) return true; 
                }
            }
            if (auto si = dyn_cast<StoreInst>(use)) {
                if (si->getPointerOperand() == v && si->getValueOperand()->getType()->isIntOrIntVectorTy()) {
                    //! If TBAA lets us ascertain information, let's use it
                    if (isKnownIntegerTBAA(si)) {
                        intUse = true;  
                        if (fast_tracking) return true;
                    }

                    if (isKnownPointerTBAA(si)) {
                        pointerUse = true;
                        if (fast_tracking) return true;
                    }

                    if (Type* t = isKnownFloatTBAA(si)) {
                        if (floatingUse == nullptr) {
                            floatingUse = t;
                        } else {
                            assert(floatingUse == t);
                        }
                        if (fast_tracking) return true;
                    }


                    //! Storing a constant integer into memory does not tell us that this memory must be integral
                    //  since we may store the constant representation of a floating point (in hex). It is, however, unlikely
                    //  to be a pointer (but unclear whether it is an integer vs float)
                    if (auto c = dyn_cast<Constant>(si->getValueOperand())) {
                        // Zero could be float zero, nullptr, or integer zero, no information should be propagated
                        
                        // follow-up, we apparently are insufficiently aggressive on type analysis if this is withheld =/
                        //if (c->isZeroValue())
                        //    continue;

                        if (auto ci = dyn_cast<ConstantInt>(c)) {

                            //it is unlikely a value from 1-4096 is not meant to be an integer, let us assume that it is
                            if (ci->getLimitedValue() <= 4096) {
                                intUse = true;
                                if (fast_tracking) return true;
                            }
                        }

                        continue;
                    }
                    bool unknownuse;
                    if (trackInt(typeInfo, trace, si->getValueOperand(), intseen, seen, typeseen, floatingUse, pointerUse, intUse, unknownuse, /*shouldconsiderunknownuse*/false)) {
                        if (fast_tracking) return true; 
                    }
                }
            }
        }
    }
    return false;
}

void appendArgumentInformation(std::map<Argument*, DataType> &typeInfo, const std::vector<CallInst*> newtrace, CallInst* call, std::map<std::tuple<const std::vector<CallInst*>, Value*,bool>, IntType> intseen, std::set<std::tuple<const std::vector<CallInst*>, Value*>> ptrseen, SmallPtrSet<Type*, 4> typeseen) {
        int argnum = 0;

        for(auto &arg : call->getCalledFunction()->args()) {
            DataType dt(IntType::Unknown);
            Type* floatingUse = nullptr;
            bool pointerUse = false;
            bool intUse = false;
            if (auto pt = dyn_cast<PointerType>(arg.getType())) {
                if (pt->getElementType()->isIntOrIntVectorTy()) {
                    trackPointer(typeInfo, newtrace, call->getArgOperand(argnum), intseen, ptrseen, typeseen, floatingUse, pointerUse, intUse, /*sawreturn*/nullptr, /*onlyfirst*/false);
                }
            } else if (arg.getType()->isIntOrIntVectorTy()) {
                bool unknownuse = false;
                trackInt(typeInfo, newtrace, call->getArgOperand(argnum), intseen, ptrseen, typeseen, floatingUse, pointerUse, intUse, unknownuse, /*shouldconsiderunknown*/false, nullptr);
            }
            if (floatingUse && !pointerUse && !intUse) dt = DataType(floatingUse);
            if (!floatingUse && pointerUse && !intUse) dt = DataType(IntType::Pointer);
            if (!floatingUse && !pointerUse && intUse) dt = DataType(IntType::Integer);
            typeInfo.insert(std::pair<Argument*, DataType>(&arg, dt));
            argnum++;
        }
}

bool trackInt(const std::map<Argument*, DataType> typeInfo, const std::vector<CallInst*> trace, Value* v, std::map<std::tuple<const std::vector<CallInst*>, Value*,bool>, IntType> intseen, std::set<std::tuple<const std::vector<CallInst*>, Value*>> ptrseen, SmallPtrSet<Type*, 4> typeseen, Type*& floatingUse, bool& pointerUse, bool& intUse, bool& unknownUse, bool shouldConsiderUnknownUse, bool* sawReturn/*if sawReturn != nullptr, we can ignore uses of returninst, setting the bool to true if we see one*/) {
    auto idx = std::tuple<const std::vector<CallInst*>, Value*, bool>(trace, v, shouldConsiderUnknownUse);
    //llvm::errs() << "trackingInt" << *v << " [";
    //for(auto a : trace) llvm::errs() << a << "," ;
    //llvm::errs() << "]\n";
    if (intseen.find(idx) != intseen.end()) {
        if (intseen[idx] == IntType::Integer) {
            intUse = true;
        }
        return false;
    }
    intseen[idx] = IntType::Unknown;
    
    assert(v);
    assert(v->getType());
    assert(v->getType()->isIntOrIntVectorTy());
        
    if (auto arg = dyn_cast<Argument>(v)) {
        auto fd = typeInfo.find(arg);
        if (fd != typeInfo.end()) {
            switch(fd->second.typeEnum) {
                case IntType::Unknown:
                    break;
                case IntType::Float:
                    floatingUse = fd->second.type;
                    if (fast_tracking) return true;
                    break;
                case IntType::Integer:
                    intUse = true;
                    if (fast_tracking) return true;
                    break;
                case IntType::Pointer:
                    pointerUse = true;
                    if (fast_tracking) return true;
                    break;
            }
        } else if (trace.size() == 0) {
            llvm::errs() << "couldn't find arg: " << *arg << "(" << arg->getParent()->getName() << ") " << "\n";
            bool sameNAme = false;
            for(auto &pair : typeInfo) {
                if (arg->getName() == pair.first->getName()) sameNAme = true;
                llvm::errs() << "    + option: " << *pair.first << "(" << pair.first->getParent()->getName() << ") " << "\n";
            }

            if (typeInfo.size() != 0 && sameNAme) //} && arg->getParent()->getName()) 
                assert(0 && "no arg for tracking");
            //assert(0 && "no arg for tracking");
        }
    }

    if (isa<UndefValue>(v)) {
        intUse = true;
        intseen[idx] = IntType::Integer;
        //llvm::errs() << "find int use of " << *v << " undef\n";
        return true;
    }
      
    if (isa<Constant>(v)) {
        intUse = true;
        intseen[idx] = IntType::Integer;
        //llvm::errs() << "find int use of " << *v << " constant\n";
        return true;
    }

    //consider booleans to be integers
    if (cast<IntegerType>(v->getType()->getScalarType())->getBitWidth() == 1) {
        intUse = true;
        intseen[idx] = IntType::Integer;
        //llvm::errs() << "find int use of " << *v << " is bool\n";
        return true;
    }
    bool oldunknownUse = unknownUse;
    unknownUse = false;

    for(User* use: v->users()) {
        if (auto ci = dyn_cast<CastInst>(use)) {

            if (isa<SIToFPInst>(use) || isa<UIToFPInst>(use)) {
                intUse = true;
                intseen[idx] = IntType::Integer;
                if (fast_tracking) return true;
                continue;
            }

            if (isa<FPToSIInst>(use) || isa<FPToUIInst>(use)) {
                if (floatingUse == nullptr) {
                    floatingUse = ci->getSrcTy();
                } else {
                    assert(floatingUse == ci->getSrcTy());
                }
                intseen[idx] = IntType::Float;
                if (fast_tracking) return true;
                continue;
            }

            if (ci->getDestTy()->isPointerTy()) {
                //llvm::errs() << "saw(p use) of " << *v << " use " << *ci << "\n";
                pointerUse = true;
                if (fast_tracking) return true;
                continue;
            }
            if (ci->getDestTy()->isFPOrFPVectorTy()) {
                if (isa<BitCastInst>(ci)) {
                    if (floatingUse == nullptr) {
                        floatingUse = ci->getDestTy();
                    } else {
                        assert(floatingUse == ci->getDestTy());
                    }
                    if (fast_tracking) return true;
                }
                continue;
            }
            if (ci->getDestTy()->isIntOrIntVectorTy()) {
              if (cast<IntegerType>(ci->getDestTy()->getScalarType())->getBitWidth() < 8) continue;

              if (trackInt(typeInfo, trace, ci, intseen, ptrseen, typeseen, floatingUse, pointerUse, intUse, unknownUse, false, sawReturn)) {
                if (fast_tracking) return true;
              }
              continue;
            } else {
              assert(0 && "illegal use of cast");
            }
        }

        if (auto bi = dyn_cast<BinaryOperator>(use)) {
            //And's should not have their int uses propagated to v (e.g. if ptr & 1 is used somewhere as an integer/index into an array, this does not mean that ptr is an integer)
            if (bi->getOpcode() == BinaryOperator::And) continue;
            if (bi->getOpcode() == BinaryOperator::Or) continue;

            if (trackInt(typeInfo, trace, bi, intseen, ptrseen, typeseen, floatingUse, pointerUse, intUse, unknownUse, false, sawReturn)) {
                if (fast_tracking) {
                    //if (floatingUse)
                    //llvm::errs() << "use predefault val:" << *v << " use: " << *bi << " pointer:" << pointerUse << " floating:" << *floatingUse << " int:" << intUse << "\n";
                    //else
                    //llvm::errs() << "use predefault val:" << *v << " use: " << *bi << " pointer:" << pointerUse << " floating:" << floatingUse << " int:" << intUse << "\n";
                    return true;
                }
            }
            continue;
        }

        if (auto pn = dyn_cast<PHINode>(use)) {
            if (trackInt(typeInfo, trace, pn, intseen, ptrseen, typeseen, floatingUse, pointerUse, intUse, unknownUse, false, sawReturn)) {
                if (fast_tracking) return true;
            }
            continue;
        }
        
        if (auto seli = dyn_cast<SelectInst>(use)) {
            assert(seli->getCondition() != v);
            if (trackInt(typeInfo, trace, seli, intseen, ptrseen, typeseen, floatingUse, pointerUse, intUse, unknownUse, /*shouldconsiderUnknownUse=*/false, sawReturn)) {
                //llvm::errs() << "find select use of " << *v << " in " << *seli << " intUse: " << intUse << "\n";
                if (fast_tracking) return true;
            }
            continue;
        }

        if (auto gep = dyn_cast<GetElementPtrInst>(use)) {
            if (gep->getPointerOperand() != v) {
                //llvm::errs() << "find int use of " << *v << " in " << *gep << "\n";
                intUse = true;
                intseen[idx] = IntType::Integer;
                continue;
            }
        }
        
        if (auto call = dyn_cast<CallInst>(use)) {
            if (Function* ci = call->getCalledFunction()) {
                if (ci->getName() == "malloc") {
                    intUse = true;
                    intseen[idx] = IntType::Integer;
                    //llvm::errs() << "find int use of " << *v << " in " << *use << "\n";
                    continue;
                }

                //These function calls are known uses that do not potentially have an inactive use
                if (ci->getName() == "__cxa_guard_acquire" || ci->getName() == "__cxa_guard_release" || ci->getName() == "__cxa_guard_abort" || ci->getName() == "printf" || ci->getName() == "fprintf") {
                    continue;
                }

                if (ci->getIntrinsicID() == Intrinsic::memset) {
                    if (call->getArgOperand(0) != v) {
                        intUse = true;
                        intseen[idx] = IntType::Integer;
                        //llvm::errs() << "find int use of " << *v << " in " << *use << "\n";
                        continue;
                    }
                }
                if (ci->getIntrinsicID() == Intrinsic::memcpy || ci->getIntrinsicID() == Intrinsic::memmove) {
                    if (call->getArgOperand(0) != v && call->getArgOperand(1) != v) {
                        intUse = true;
                        intseen[idx] = IntType::Integer;
                        //llvm::errs() << "find int use of " << *v << " in " << *use << "\n";    
                        continue;
                    }
                }
                if (!ci->empty()) {
                    auto a = ci->arg_begin();
                    bool shouldHandleReturn=false;
                    std::vector<CallInst*> newtrace(trace);
                    addCallRemovingCycle(newtrace, call);
                    for(size_t i=0; i<call->getNumArgOperands(); i++) {
                        if (call->getArgOperand(i) == v) {
                            //TODO consider allowing return to be ignored as an unknown use below, so long as we also then look at use of call value itself
                            if(trackInt(typeInfo, newtrace, a, intseen, ptrseen, typeseen, floatingUse, pointerUse, intUse, unknownUse, /*shouldUnknownReturn*/false, &shouldHandleReturn)) {
                                if (fast_tracking) return true;
                            }
                        }
                        a++;
                    }

                    if (shouldHandleReturn) {
                        if(trackInt(typeInfo, trace, call, intseen, ptrseen, typeseen, floatingUse, pointerUse, intUse, unknownUse, /*shouldUnknownReturn*/false, sawReturn)) {
                            if (fast_tracking) return true;
                        }
                    }
                    continue;
                }
            }
        }

        if (isa<AllocaInst>(use)) {
            intUse = true;
            intseen[idx] = IntType::Integer;
            //llvm::errs() << "find int use of " << *v << " in " << *use << "\n";    
            if (fast_tracking) return true;
            continue;
        }
        /*
        if (isa<ExtractValueInst>(use) || isa<InsertValueInst>(use)) {
            intUse = true;
            intseen[idx] = IntType::Integer;
            if (fast_tracking) return true;
            continue;
        }
        */
        
        if (isa<IntToPtrInst>(use)) {
            //llvm::errs() << "saw(p use) of " << *v << " use " << *use << "\n";
            pointerUse = true;
            if (fast_tracking) return true;
            continue;
        }
        
        if (auto si = dyn_cast<StoreInst>(use)) {
            assert(v == si->getValueOperand());
            if (isKnownIntegerTBAA(si)) {
                intUse = true;  
                intseen[idx] = IntType::Integer;
                if (fast_tracking) return true;
            }

            if (isKnownPointerTBAA(si)) {
                pointerUse = true;
                if (fast_tracking) return true;
            }

            if (Type* t = isKnownFloatTBAA(si)) {
                if (floatingUse == nullptr) {
                    floatingUse = t;
                } else {
                    assert(floatingUse == t);
                }
                if (fast_tracking) return true;
            }

            if (trackPointer(typeInfo, trace, si->getPointerOperand(), intseen, ptrseen, typeseen, floatingUse, pointerUse, intUse, nullptr, true, {})) {
                if (fast_tracking) return true;
            }

            if (auto ai = dyn_cast<AllocaInst>(si->getPointerOperand())) {
                bool baduse = false;
                for(auto user : ai->users()) {
                    if (auto si = dyn_cast<StoreInst>(user)) {
                        if (si->getValueOperand() == ai) {
                            baduse = true;
                            break;
                        }
                    } else if (auto li = dyn_cast<LoadInst>(user)) {
                        if (trackInt(typeInfo, trace, li, intseen, ptrseen, typeseen, floatingUse, pointerUse, intUse, unknownUse, false, sawReturn)) {
                            if (fast_tracking) return true;
                        } 
                    } else baduse = true;
                }
                if (!baduse) continue;
            }
        }

        if (isa<CmpInst>(use)) continue;
        if (isa<SwitchInst>(use)) continue;

        if (sawReturn && isa<ReturnInst>(use)) { 
            *sawReturn = true;
            continue;
        }

        unknownUse = true;
        //llvm::errs() << "unknown use : " << *use << " of v: " << *v << "\n";
        continue;
    }

    //! This code isnt great if hit by a caller
    
    if (shouldConsiderUnknownUse &&unknownUse == false) {
        intUse = true;
        intseen[idx] = IntType::Integer;
        //llvm::errs() << "find int use of " << *v << " from lack of unknown use\n";
        if (fast_tracking) return true;
    }
    

    unknownUse |= oldunknownUse;

    if (auto li = dyn_cast<LoadInst>(v)) {
        if (isKnownIntegerTBAA(li)) {
            intUse = true;
            intseen[idx] = IntType::Integer;
            if (fast_tracking) return true;
        }
        if (isKnownPointerTBAA(li)) {
            pointerUse = true;
            if (fast_tracking) return true;
        }
        if (Type* t = isKnownFloatTBAA(li)) {
            floatingUse = t;
            if (fast_tracking) return true;
        }
        if (trackPointer(typeInfo, trace, li->getOperand(0), intseen, ptrseen, typeseen, floatingUse, pointerUse, intUse, nullptr, true, {})) {
            if (fast_tracking) return true;
        }
    }
   
    if (isa<FPToSIInst>(v) || isa<FPToUIInst>(v)) {
        intUse = true;
        intseen[idx] = IntType::Integer;
        if (fast_tracking) return true;
    }

    if (auto ci = dyn_cast<BitCastInst>(v)) {
        if (ci->getSrcTy()->isPointerTy()) {
            //llvm::errs() << "saw(p) " << *ci << "\n";
            pointerUse = true;
            //llvm::errs() << "saw(p use) of " << *v << " use " << *ci << "\n";
            if (fast_tracking) return true;
        }
        if (ci->getSrcTy()->isFloatingPointTy()) {
            //llvm::errs() << "saw(p) " << *ci << "\n";
            floatingUse = ci->getSrcTy();
            if (fast_tracking) return true;
        }
        if (ci->getSrcTy()->isIntOrIntVectorTy()) {
          bool fakeunknownuse = false;
          if (trackInt(typeInfo, trace, ci->getOperand(0), intseen, ptrseen, typeseen, floatingUse, pointerUse, intUse, fakeunknownuse)) {
            if (fast_tracking) return true;
          }
        }
    }
    
    if (auto seli = dyn_cast<SelectInst>(v)) {
          bool fakeunknownuse = false;
          if (trackInt(typeInfo, trace, seli->getOperand(1), intseen, ptrseen, typeseen, floatingUse, pointerUse, intUse, fakeunknownuse)) {
            if (fast_tracking) return true;
          }
          if (trackInt(typeInfo, trace, seli->getOperand(2), intseen, ptrseen, typeseen, floatingUse, pointerUse, intUse, fakeunknownuse)) {
            if (fast_tracking) return true;
          }
    }
    
    if (auto bi = dyn_cast<BinaryOperator>(v)) {

        bool intUse0 = false, intUse1 = false;
        std::map<std::tuple<const std::vector<CallInst*>, Value*,bool>, IntType> intseen0(intseen.begin(), intseen.end());
        std::set<std::tuple<const std::vector<CallInst*>, Value*>> ptrseen0(ptrseen.begin(), ptrseen.end());
        SmallPtrSet<Type*, 4> typeseen0(typeseen.begin(), typeseen.end());
        bool fakeunknownuse0 = false;
        
        if (trackInt(typeInfo, trace, bi->getOperand(0), intseen0, ptrseen0, typeseen0, floatingUse, pointerUse, intUse0, fakeunknownuse0, true) && (floatingUse || pointerUse) ) {
            if (fast_tracking) return true;
        }

        if (intUse0) {
            if (trackInt(typeInfo, trace, bi->getOperand(1), intseen0, ptrseen0, typeseen0, floatingUse, pointerUse, intUse1, fakeunknownuse0, true) && (floatingUse || pointerUse)) {
                if (fast_tracking) return true;
            }
        }
        
        //llvm::errs() << " considering binary operator " << *bi << " iu0: " << intUse0 << " iu1: " << intUse1 << "\n";
        
        if (intUse0 && intUse1) {
            intUse = true;
            intseen[idx] = IntType::Integer;
            /*
            if (floatingUse0) {
                if (floatingUse != nullptr) assert(floatingUse == floatingUse0);
                floatingUse = floatingUse0;
            }
            pointerUse |= pointer
                */
            intseen.insert(intseen0.begin(), intseen0.end());
            
            ptrseen.insert(ptrseen0.begin(), ptrseen0.end());
            
            typeseen.insert(typeseen0.begin(), typeseen0.end());
            if (fast_tracking) return true;
        }
    }
    
    if (auto phi = dyn_cast<PHINode>(v)) {
        std::map<std::tuple<const std::vector<CallInst*>, Value*,bool>, IntType> intseen0(intseen.begin(), intseen.end());
        std::set<std::tuple<const std::vector<CallInst*>, Value*>> ptrseen0(ptrseen.begin(), ptrseen.end());
        SmallPtrSet<Type*, 4> typeseen0(typeseen.begin(), typeseen.end());
        bool allintUse = true;
        for (auto& val : phi->incoming_values() ) {
            if (!isa<Value>(&val)) continue;

            bool intUse0 = false;
            bool fakeunknownuse0 = false;
            if ( trackInt(typeInfo, trace, cast<Value>(&val), intseen0, ptrseen0, typeseen0, floatingUse, pointerUse, intUse0, fakeunknownuse0, true) && (floatingUse || pointerUse) ) {
                intseen.insert(intseen0.begin(), intseen0.end());
                ptrseen.insert(ptrseen0.begin(), ptrseen0.end());
                typeseen.insert(typeseen0.begin(), typeseen0.end());

                if (fast_tracking) return true;
            }
            if (!intUse0) {
                allintUse = false;
            }
        }

        if (allintUse) {
            intUse = true;
            intseen[idx] = IntType::Integer;
            
            intseen.insert(intseen0.begin(), intseen0.end());
            
            ptrseen.insert(ptrseen0.begin(), ptrseen0.end());
            
            typeseen.insert(typeseen0.begin(), typeseen0.end());
            if (fast_tracking) return true;
        }
    }
    
    if (auto ci = dyn_cast<CallInst>(v)) {
        if (auto F = ci->getCalledFunction()) {
            std::map<std::tuple<const std::vector<CallInst*>, Value*,bool>, IntType> intseen0(intseen.begin(), intseen.end());
            std::set<std::tuple<const std::vector<CallInst*>, Value*>> ptrseen0(ptrseen.begin(), ptrseen.end());
            SmallPtrSet<Type*, 4> typeseen0(typeseen.begin(), typeseen.end());
            bool allintUse = true;
            std::vector<CallInst*> newtrace(trace);
            addCallRemovingCycle(newtrace, ci);
            std::map<Argument*, DataType> newTypeInfo(typeInfo);
            appendArgumentInformation(newTypeInfo, newtrace, ci, intseen, ptrseen, typeseen);

            for (llvm::inst_iterator I = llvm::inst_begin(F), E = llvm::inst_end(F); I != E; ++I) {
                if (auto ri = dyn_cast<ReturnInst>(&*I)) {
                    auto rv = ri->getReturnValue();
                    bool intUse0 = false;
                    bool fakeunknownuse0 = false;
                    if ( trackInt(newTypeInfo, newtrace, rv, intseen0, ptrseen0, typeseen0, floatingUse, pointerUse, intUse0, fakeunknownuse0, true) && (floatingUse || pointerUse) ) {
                        intseen.insert(intseen0.begin(), intseen0.end());
                        ptrseen.insert(ptrseen0.begin(), ptrseen0.end());
                        typeseen.insert(typeseen0.begin(), typeseen0.end());

                        if (fast_tracking) return true;
                    }
                    if (!intUse0) {
                        //llvm::errs() << "could not intify " << *v << " as nonint return: " << *ri << "\n";
                        allintUse = false;
                    }
                }
            }

            if (allintUse) {
                intUse = true;
                intseen[idx] = IntType::Integer;
                
                intseen.insert(intseen0.begin(), intseen0.end());
                
                ptrseen.insert(ptrseen0.begin(), ptrseen0.end());
                
                typeseen.insert(typeseen0.begin(), typeseen0.end());
                if (fast_tracking) return true;
            }
        }
    }

    if (auto ev = dyn_cast<ExtractValueInst>(v)) {
        if (auto ci = dyn_cast<CallInst>(ev->getOperand(0))) {
            if (auto iasm = dyn_cast<InlineAsm>(ci->getCalledValue())) {
                if (iasm->getAsmString() == "cpuid") {
                    intseen[idx] = IntType::Integer;
                    intUse = true;
                    if (fast_tracking) return true;
                }
            }
        }
    }
    
    if (isa<PtrToIntInst>(v)) {
        //llvm::errs() << "saw(p) " << *v << "\n";
        pointerUse = true;
        if (fast_tracking) return true;
    }
    return false;
}

DataType isIntASecretFloat(const std::map<Argument*, DataType> typeInfo, Value* val, IntType defaultType, bool errIfNotFound) {
    //llvm::errs() << "starting isint a secretfloat for " << *val << "\n";

    assert(val->getType()->isIntOrIntVectorTy());

    if (isa<UndefValue>(val)) {
        if (defaultType != IntType::Unknown) return defaultType;
        return IntType::Integer;
    }
      
    if (isa<ConstantInt>(val)) {
        if (defaultType != IntType::Unknown) return defaultType;
        return IntType::Integer;

		//if (!cint->isZero()) return false;
        //return false;
        //llvm::errs() << *val << "\n";
        //assert(0 && "unsure if constant or not because constantint");
		 //if (cint->isOne()) return cint;
	}


        Type* floatingUse = nullptr;
        bool pointerUse = false;
        bool intUse = false;
        std::map<std::tuple<const std::vector<CallInst*>, Value*,bool>, IntType> intseen;
        std::set<std::tuple<const std::vector<CallInst*>, Value*>> ptrseen;
        
        SmallPtrSet<Type*, 4> typeseen;

        std::vector<CallInst*> trace;

        bool fakeunknownuse = false;
        trackInt(typeInfo, trace, val, intseen, ptrseen, typeseen, floatingUse, pointerUse, intUse, fakeunknownuse, /*shouldConsiderUnknownUse*/true);
        
        /*
        if (floatingUse)
        llvm::errs() << " val:" << *val << " pointer:" << pointerUse << " floating:" << *floatingUse << " int:" << intUse << "\n";
        else
        llvm::errs() << " val:" << *val << " pointer:" << pointerUse << " floating:" << floatingUse << " int:" << intUse << "\n";
        */
        
        //if (floatingUse)
        //llvm::errs() << "predefault val:" << *val << " pointer:" << pointerUse << " floating:" << *floatingUse << " int:" << intUse << "\n";
        //else
        //llvm::errs() << "predefault val:" << *val << " pointer:" << pointerUse << " floating:" << floatingUse << " int:" << intUse << "\n";

        if (!intUse && pointerUse && !floatingUse) { return IntType::Pointer; }
        if (!intUse && !pointerUse && floatingUse) { return DataType(floatingUse); }
        if (intUse && !pointerUse && !floatingUse) { return IntType::Integer; }
        

        if (defaultType != IntType::Unknown) return defaultType;

        if (!errIfNotFound) return IntType::Unknown;

        if(auto inst = dyn_cast<Instruction>(val)) {
            llvm::errs() << *inst->getParent()->getParent()->getParent() << "\n";
            llvm::errs() << *inst->getParent()->getParent() << "\n";
        }
        
        if(auto arg = dyn_cast<Argument>(val)) {
            llvm::errs() << *arg->getParent()->getParent() << "\n";
            llvm::errs() << *arg->getParent() << "\n";
        }

        if (floatingUse)
        llvm::errs() << " val:" << *val << " pointer:" << pointerUse << " floating:" << *floatingUse << " int:" << intUse << "\n";
        else
        llvm::errs() << " val:" << *val << " pointer:" << pointerUse << " floating:" << floatingUse << " int:" << intUse << "\n";
        assert(0 && "ambiguous unsure if constant or not");
}

//! return the secret float type if found, otherwise nullptr
DataType isIntPointerASecretFloat(const std::map<Argument*, DataType> typeInfo, Value* val, bool onlyFirst, bool errIfNotFound) {
    assert(val->getType()->isPointerTy());
    assert(cast<PointerType>(val->getType())->getElementType()->isIntOrIntVectorTy());

    if (isa<UndefValue>(val)) return nullptr;
      
    if (auto cint = dyn_cast<ConstantInt>(val)) {
		if (!cint->isZero()) return nullptr;
        assert(0 && "unsure if constant or not because constantint");
		 //if (cint->isOne()) return cint;
	}

    Type* floatingUse = nullptr;
    bool pointerUse = false;
    bool intUse = false;

    std::map<std::tuple<const std::vector<CallInst*>, Value*,bool>, IntType> intseen;
    SmallPtrSet<Type*, 4> typeseen;

    std::set<std::tuple<const std::vector<CallInst*>, Value*>> seen;
    std::vector<CallInst*> trace;

    trackPointer(typeInfo, trace, val, intseen, seen, typeseen, floatingUse, pointerUse, intUse, nullptr, onlyFirst);

    if (pointerUse && (floatingUse == nullptr) && !intUse) return IntType::Pointer; 
    if (!pointerUse && (floatingUse != nullptr) && !intUse) return floatingUse;
    if (!pointerUse && (floatingUse == nullptr) && intUse) return IntType::Integer; 

    if (errIfNotFound) {
        if (auto inst = dyn_cast<Instruction>(val)) {
            llvm::errs() << *inst->getParent()->getParent()->getParent() << "\n";
            llvm::errs() << *inst->getParent()->getParent() << "\n";
        }

        if(auto arg = dyn_cast<Argument>(val)) {
            llvm::errs() << *arg->getParent()->getParent() << "\n";
            llvm::errs() << *arg->getParent() << "\n";
        }
            
        if (floatingUse)
        llvm::errs() << " val:" << *val << " pointer:" << pointerUse << " floating:" << *floatingUse << " int:" << intUse << "\n";
        else
        llvm::errs() << " val:" << *val << " pointer:" << pointerUse << " floating:" << floatingUse << " int:" << intUse << "\n";
        assert(0 && "ambiguous unsure what type of ptr or not");
    }
    return IntType::Unknown;
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

bool isFunctionArgumentConstant(CallInst* CI, Value* val, SmallPtrSetImpl<Value*> &constants, SmallPtrSetImpl<Value*> &nonconstant, SmallPtrSetImpl<Value*> &constantvals, const SmallPtrSetImpl<Value*> &retvals, AAResults &AA, int directions) {
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

    //return false;
    if (fn.startswith("augmented")) return false;
    if (fn.startswith("fakeaugmented")) return false;
    if (fn.startswith("diffe")) return false;
    //if (val->getType()->isPointerTy()) return false;
    if (!val->getType()->isIntOrIntVectorTy()) return false;

    assert(retvals.find(val) == retvals.end());

    //static std::unordered_map<std::tuple<Function*, Value*, SmallPtrSet<Value*,20>, std::set<Value*> >, bool> metacache;
    static std::map<std::tuple<CallInst*, Value*, std::set<Value*>, std::set<Value*>, std::set<Value*>, std::set<Value*> >, bool> metacache;
    //auto metatuple = std::make_tuple(F, val, SmallPtrSet<Value*,20>(constants.begin(), constants.end()), std::set<Value*>(nonconstant.begin(), nonconstant.end()));
    auto metatuple = std::make_tuple(CI, val, std::set<Value*>(constants.begin(), constants.end()), std::set<Value*>(nonconstant.begin(), nonconstant.end()), std::set<Value*>(constantvals.begin(), constantvals.end()), std::set<Value*>(retvals.begin(), retvals.end()));
    if (metacache.find(metatuple) != metacache.end()) {
		if (printconst)
        llvm::errs() << " < SUBFN metacache const " << F->getName() << "> arg: " << *val << " ci:" << *CI << "\n";
        return metacache[metatuple];
    }
    if (printconst)
       llvm::errs() << " < METAINDUCTIVE SUBFN const " << F->getName() << "> arg: " << *val << " ci:" << *CI << "\n";
    
    metacache[metatuple] = true;
    //Note that the base case of true broke the up/down variant so have to be very conservative
    //  as a consequence we cannot detect const of recursive functions :'( [in that will be too conservative]
    //metacache[metatuple] = false;

    SmallPtrSet<Value*, 20> constants2;
    constants2.insert(constants.begin(), constants.end());
    SmallPtrSet<Value*, 20> nonconstant2;
    nonconstant2.insert(nonconstant.begin(), nonconstant.end());
    SmallPtrSet<Value*, 20> constantvals2;
    constantvals2.insert(constantvals.begin(), constantvals.end());
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
    SmallPtrSet<Value*, 20> arg_findifactive;
    
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

        if (isconstantValueM(CI->getArgOperand(i), constants2, nonconstant2, constantvals2, retvals2, AA), directions) {
            newconstants.insert(a);
            arg_constants.insert(i);
        } else {
            newnonconstant.insert(a);
        }
        a++;
    }

    bool constret;
    
    //allow return index as valid entry as well
    if (CI != val) {
        constret = isconstantValueM(CI, constants2, nonconstant2, constantvals2, retvals2, AA, directions);
        if (constret) arg_constants.insert(-1);
    } else {
        constret = false;
        arg_findifactive.insert(a);
        idx_findifactive.insert(-1);
        
    }

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
    //cache[tuple] = false;
        
    SmallPtrSet<Value*,4> newconstantvals;
	newconstantvals.insert(constantvals2.begin(), constantvals2.end());
    
    SmallPtrSet<Value*,4> newretvals;
	newretvals.insert(retvals2.begin(), retvals2.end());

    for (llvm::inst_iterator I = llvm::inst_begin(F), E = llvm::inst_end(F); I != E; ++I) {
        if (auto ri = dyn_cast<ReturnInst>(&*I)) {
            if (!constret) {
                newretvals.insert(ri->getReturnValue());
                if (CI == val) arg_findifactive.insert(ri->getReturnValue());
            } else {
                //newconstantvals.insert(ri->getReturnValue());
            }
        }
    }
    
    
    for(auto specialarg : arg_findifactive) {
        for(auto user : specialarg->users()) {
			if (printconst)
			llvm::errs() << " going to consider user " << *user << "\n";
            if (!isconstantValueM(user, newconstants, newnonconstant, newconstantvals, newretvals, AA, 3)) {
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


static bool isGuaranteedConstantValue(Value* val, const SmallPtrSetImpl<Value*> *constantvals) {
    //This result of this instruction is certainly an integer (and only and integer, not a pointer or float). Therefore its value is inactive
    // Note that this is correct, but not aggressive as it should be (we should call isConstantValue(inst) here, but we need to be careful to not have an infinite recursion)
    //  TODO: make this more aggressive
    if (val->getType()->isIntOrIntVectorTy()) {
        //TODO propagate interprocedural type info here instead of using empty map {}
        //  This does not affect correctness, but can do better constant analysis if we have
        if (isIntASecretFloat({}, val, /*default*/IntType::Pointer)==IntType::Integer) {
            if (printconst)
                llvm::errs() << " -> known integer " << *val << "\n";
            return true;
        }
    // if we happen to have already deduced this instruction constant, we might as well use the information
    } else if (constantvals && constantvals->find(val) != constantvals->end()) {
        if (printconst)
            llvm::errs() << " -> previous constant value " << *val << "\n";
        return true;
    // if we know the subtype contains no derivative information, we can assure that this is a constant value
    } else if (val->getType()->isVoidTy() || val->getType()->isEmptyTy()) {
        return true;
    }
    return false;
}

// TODO separate if the instruction is constant (i.e. could change things)
//    from if the value is constant (the value is something that could be differentiated)
bool isconstantM(Instruction* inst, SmallPtrSetImpl<Value*> &constants, SmallPtrSetImpl<Value*> &nonconstant, SmallPtrSetImpl<Value*> &constantvals, SmallPtrSetImpl<Value*> &retvals, AAResults &AA, uint8_t directions) {
    assert(inst);
	constexpr uint8_t UP = 1;
	constexpr uint8_t DOWN = 2;
	//assert(directions >= 0);
	assert(directions <= 3);
    if (isa<ReturnInst>(inst)) return true;

	if(isa<UnreachableInst>(inst) || isa<BranchInst>(inst) || (constants.find(inst) != constants.end())) {
    	return true;
    }

    if((nonconstant.find(inst) != nonconstant.end())) {
        return false;
    }

    // If this instruction does not write memory to memory that outlives itself (therefore propagating derivative information), and the return value of this instruction is known to be inactive
    // this instruction is inactive as it cannot propagate derivative information
    if (isGuaranteedConstantValue(inst, directions == 3 ? &constantvals : nullptr)) {
        if (!inst->mayWriteToMemory() || (isa<CallInst>(inst) && AA.onlyReadsMemory(cast<CallInst>(inst)) ) ) {
            if (printconst)
                llvm::errs() << " constant instruction from known constant non-writing instruction " << *inst << "\n";
            constants.insert(inst);
            return true;
        } else {
            if (printconst)
                llvm::errs() << " may be active inst as could write to memory " << *inst << "\n";
        }
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
            if (!isCertainPrintMallocOrFree(called) && called->empty() && !hasMetadata(called, "enzyme_gradient") && !isa<IntrinsicInst>(op) && emptyfnconst) {
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

    if (isa<FPToSIInst>(inst) || isa<FPToUIInst>(inst)) {
            constants.insert(inst);
            return true;
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
    if (auto li = dyn_cast<LoadInst>(inst)) {
        if (constants.find(li->getPointerOperand()) != constants.end()) {
            constants.insert(li);
            return true;
        }
    }

    if (printconst)
	  llvm::errs() << "checking if is constant[" << (int)directions << "] " << *inst << "\n";
    
    // For specific instructions, if for some reason or another we know that the value is a constant, this pointer instruction must be constant, indepedent of whether it could be memory or not
    if (constantvals.find(inst) != constantvals.end() && UP) {
        if (isa<CastInst>(inst) || isa<PHINode>(inst) || isa<SelectInst>(inst)) {
            if (printconst)
              llvm::errs() << "constant value becomes constant instruction " << *inst << "\n";
            constants.insert(inst);
            return true;
        }
    }
    
	SmallPtrSet<Value*, 20> constants_tmp;
	SmallPtrSet<Value*, 20> constantvals_tmp;

    //Handle types that could contain pointers
    //  Consider all types except
    //   * floating point types (since those are assumed not pointers)
    //   * integers that we know are not pointers
    bool containsPointer = true;
    if (inst->getType()->isFPOrFPVectorTy()) containsPointer = false;
    // TODO propagate typeInfo here so can do more aggressive constant analysis rather than using empty map {}
    if (inst->getType()->isIntOrIntVectorTy() && isIntASecretFloat({}, inst, /*default*/IntType::Pointer).typeEnum != IntType::Pointer) containsPointer = false;

    if (containsPointer) {

		//Proceed assuming this is constant, can we prove this should be constant otherwise
		SmallPtrSet<Value*, 20> constants2;
		constants2.insert(constants.begin(), constants.end());
		SmallPtrSet<Value*, 20> nonconstant2;
		nonconstant2.insert(nonconstant.begin(), nonconstant.end());
		SmallPtrSet<Value*, 20> constantvals2;
		constantvals2.insert(constantvals.begin(), constantvals.end());
		SmallPtrSet<Value*, 20> retvals2;
		retvals2.insert(retvals.begin(), retvals.end());
		constants2.insert(inst);

		if (printconst)
			llvm::errs() << " < MEMSEARCH" << (int)directions << ">" << *inst << "\n";


        //This pointer is returned (or value is nonconstant other places) and nonconstant, be conservative and assume it is used in a nonconstant instruction
        //  This check is only necessary when considering a pointer returned as this should be deduced as nonconstant elsewhere otherwise
        if (retvals.find(inst) != retvals.end()) {
            if (directions == 3)
              nonconstant.insert(inst);
            if (printconst)
              llvm::errs() << "memory(" << (int)directions << ")  ret: " << *inst << "\n";
            return false;
        }

        {
		SmallPtrSet<Value*, 20> constants2;
		constants2.insert(constants.begin(), constants.end());
		SmallPtrSet<Value*, 20> nonconstant2;
		nonconstant2.insert(nonconstant.begin(), nonconstant.end());
		SmallPtrSet<Value*, 20> constantvals2;
		constantvals2.insert(constantvals.begin(), constantvals.end());
		SmallPtrSet<Value*, 20> retvals2;
		retvals2.insert(retvals.begin(), retvals.end());
		nonconstant2.insert(inst);
		for (const auto &a:inst->users()) {
		  if (isa<LoadInst>(a)) {
		      if (!isconstantValueM(a, constants2, nonconstant2, constantvals2, retvals2, AA, directions & DOWN)) {
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

			if (inst == store->getPointerOperand() && !isconstantValueM(store->getValueOperand(), constants2, nonconstant2, constantvals2, retvals2, AA, directions & DOWN)) {
				if (directions == 3)
				  nonconstant.insert(inst);
    			if (printconst)
				  llvm::errs() << "memory(" << (int)directions << ")  erase 1: " << *inst << "\n";
				return false;
			}
			if (inst == store->getValueOperand() && !isconstantValueM(store->getPointerOperand(), constants2, nonconstant2, constantvals2, retvals2, AA, directions & DOWN)) {
				if (directions == 3)
				  nonconstant.insert(inst);
    			if (printconst)
				  llvm::errs() << "memory(" << (int)directions << ")  erase 2: " << *inst << "\n";
				return false;
			}
		  } else if (isa<LoadInst>(a)) {
              /*
		      if (!isconstantValueM(a, constants2, nonconstant2, constantvals2, retvals2, AA, directions)) {
				if (directions == 3)
				  nonconstant.insert(inst);
    			if (printconst)
				  llvm::errs() << "memory(" << (int)directions << ")  erase 3: " << *inst << "\n";
				return false;
              }
              */
              continue;
          } else if (auto ci = dyn_cast<CallInst>(a)) {
			if (!isconstantM(ci, constants2, nonconstant2, constantvals2, retvals2, AA, directions & DOWN)) {
				if (directions == 3)
				  nonconstant.insert(inst);
    			if (printconst)
				  llvm::errs() << "memory(" << (int)directions << ") erase 5: " << *inst << " op " << *a << "\n";
				return false;
			}
          } else {
			if (!isconstantM(cast<Instruction>(a), constants2, nonconstant2, constantvals2, retvals2, AA, directions & DOWN)) {
				if (directions == 3)
				  nonconstant.insert(inst);
    			if (printconst)
				  llvm::errs() << "memory(" << (int)directions << ") erase 4: " << *inst << " op " << *a << "\n";
				return false;
			}
		  }

		}
		
		if (printconst)
			llvm::errs() << " </MEMSEARCH" << (int)directions << ">" << *inst << "\n";
		
        constants_tmp.insert(constants2.begin(), constants2.end());
        constantvals_tmp.insert(constantvals2.begin(), constantvals2.end());
	}
    
    {
	SmallPtrSet<Value*, 20> constants2;
	constants2.insert(constants.begin(), constants.end());
	SmallPtrSet<Value*, 20> nonconstant2;
	nonconstant2.insert(nonconstant.begin(), nonconstant.end());
	SmallPtrSet<Value*, 20> constantvals2;
	constantvals2.insert(constantvals.begin(), constantvals.end());
	SmallPtrSet<Value*, 20> retvals2;
	retvals2.insert(retvals.begin(), retvals.end());
	constants2.insert(inst);
		
	if (directions & UP) {
        if (printconst)
		    llvm::errs() << " < UPSEARCH" << (int)directions << ">" << *inst << "\n";

        if (auto gep = dyn_cast<GetElementPtrInst>(inst)) {
            if (isconstantValueM(gep->getPointerOperand(), constants2, nonconstant2, constantvals2, retvals2, AA, UP)) {
                constants.insert(inst);
                constants.insert(constants2.begin(), constants2.end());
                constants.insert(constants_tmp.begin(), constants_tmp.end());
                constantvals.insert(constantvals2.begin(), constantvals2.end());
                constantvals.insert(constantvals_tmp.begin(), constantvals_tmp.end());
                if (printconst)
                  llvm::errs() << "constant(" << (int)directions << ") up-gep " << *inst << "\n";
                return true;
            }

        } else if (auto ci = dyn_cast<CallInst>(inst)) {
            bool seenuse = false;
             
            if (!seenuse) {
            for(auto& a: ci->arg_operands()) {
                if (!isconstantValueM(a, constants2, nonconstant2, constantvals2, retvals2, AA, UP)) {
                    seenuse = true;
                    if (printconst)
                      llvm::errs() << "nonconstant(" << (int)directions << ")  up-call " << *inst << " op " << *a << "\n";
                    break;
                    /*
                    if (directions == 3)
                      nonconstant.insert(inst);
                    if (printconst)
                      llvm::errs() << "nonconstant(" << (int)directions << ")  call " << *inst << " op " << *a << "\n";
                    //return false;
                    break;
                    */
                }
            }
            }

            
            //! TODO consider calling interprocedural here
            //! TODO: Really need an attribute that determines whether a function can access a global (not even necessarily read)
            //if (ci->hasFnAttr(Attribute::ReadNone) || ci->hasFnAttr(Attribute::ArgMemOnly)) 
            if (!seenuse) {
                constants.insert(inst);
                constants.insert(constants2.begin(), constants2.end());
                constants.insert(constants_tmp.begin(), constants_tmp.end());
                constantvals.insert(constantvals2.begin(), constantvals2.end());
                constantvals.insert(constantvals_tmp.begin(), constantvals_tmp.end());
                //constants.insert(constants_tmp.begin(), constants_tmp.end());
                //if (directions == 3)
                //  nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
                if (printconst)
                  llvm::errs() << "constant(" << (int)directions << ")  up-call:" << *inst << "\n";
                return true;
            }
        } else if (auto si = dyn_cast<StoreInst>(inst)) {
            SmallPtrSet<Value*, 20> constants2;
            constants2.insert(constants.begin(), constants.end());
            SmallPtrSet<Value*, 20> nonconstant2;
            nonconstant2.insert(nonconstant.begin(), nonconstant.end());
            SmallPtrSet<Value*, 20> constantvals2;
            constantvals2.insert(constantvals.begin(), constantvals.end());
            SmallPtrSet<Value*, 20> retvals2;
            retvals2.insert(retvals.begin(), retvals.end());
            constants2.insert(inst);

            if (isconstantValueM(si->getPointerOperand(), constants2, nonconstant2, constantvals2, retvals2, AA, UP)) {
                constants.insert(inst);
                constants.insert(constants2.begin(), constants2.end());
                constants.insert(constants_tmp.begin(), constants_tmp.end());
                constantvals.insert(constantvals2.begin(), constantvals2.end());
                constantvals.insert(constantvals_tmp.begin(), constantvals_tmp.end());
                // Note: not adding nonconstant here since if had full updown might not have been nonconstant

                if (printconst)
                  llvm::errs() << "constant(" << (int)directions << ") up-store:" << *inst << "\n";
                return true;

            }

            /* TODO consider stores of constant values 
            if (isconstantValueM(si->getValueOperand(), constants2, nonconstant2, constantvals2, retvals2, originalInstructions, directions)) {
                constants.insert(inst);
                constants.insert(constants2.begin(), constants2.end());
                constants.insert(constants_tmp.begin(), constants_tmp.end());

                // not here since if had full updown might not have been nonconstant
                //nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
                if (printconst)
                  llvm::errs() << "constant(" << (int)directions << ") store:" << *inst << "\n";
                return true;
            }
            */
        } else {
            bool seenuse = false;

            for(auto& a: inst->operands()) {
                if (!isconstantValueM(a, constants2, nonconstant2, constantvals2, retvals2, AA, UP)) {
                    //if (directions == 3)
                    //  nonconstant.insert(inst);
                    if (printconst)
                      llvm::errs() << "nonconstant(" << (int)directions << ")  up-inst " << *inst << " op " << *a << "\n";
                    seenuse = true;
                    break;
                    //return false;
                }
            }

            if (!seenuse) {
            //if (!isa<StoreInst>(inst) && !inst->getType()->isPointerTy()) {
                constants.insert(inst);
                constants.insert(constants2.begin(), constants2.end());
                constants.insert(constants_tmp.begin(), constants_tmp.end());
                constantvals.insert(constantvals2.begin(), constantvals2.end());
                constantvals.insert(constantvals_tmp.begin(), constantvals_tmp.end());
                //constants.insert(constants_tmp.begin(), constants_tmp.end());
                //if (directions == 3)
                //  nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
                if (printconst)
                  llvm::errs() << "constant(" << (int)directions << ")  up-inst:" << *inst << "\n";
                return true;
            //}
            }
        }
        if (printconst)
		    llvm::errs() << " </UPSEARCH" << (int)directions << ">" << *inst << "\n";
	}
    }

    //TODO use typeInfo for more aggressive activity analysis
	if (!(inst->getType()->isPointerTy() || (inst->getType()->isIntOrIntVectorTy() && isIntASecretFloat({}, inst, /*default*/IntType::Pointer)==IntType::Pointer) ) && ( !inst->mayWriteToMemory() || isa<BinaryOperator>(inst) ) && (directions & DOWN) && (retvals.find(inst) == retvals.end()) ) { 
		//Proceed assuming this is constant, can we prove this should be constant otherwise
		SmallPtrSet<Value*, 20> constants2;
		constants2.insert(constants.begin(), constants.end());
		SmallPtrSet<Value*, 20> nonconstant2;
		nonconstant2.insert(nonconstant.begin(), nonconstant.end());
		SmallPtrSet<Value*, 20> constantvals2;
		constantvals2.insert(constantvals.begin(), constantvals.end());
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
                if (isFunctionArgumentConstant(call, inst, constants2, nonconstant2, constantvals2, retvals2, AA, DOWN)) {
                    if (printconst)
			          llvm::errs() << "found constant(" << (int)directions << ")  callinst use:" << *inst << " user " << *a << "\n";
                    continue;
                } else {
                    if (printconst)
			          llvm::errs() << "found seminonconstant(" << (int)directions << ")  callinst use:" << *inst << " user " << *a << "\n";
                    //seenuse = true;
                    //break;
                }
			}

		  	if (!isconstantM(cast<Instruction>(a), constants2, nonconstant2, constantvals2, retvals2, AA, DOWN)) {
    			if (printconst)
			      llvm::errs() << "nonconstant(" << (int)directions << ") inst (uses):" << *inst << " user " << *a << " " << &seenuse << "\n";
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
			  llvm::errs() << "constant(" << (int)directions << ") inst (uses):" << *inst << "   seenuse:" << &seenuse << "\n";
			return true;
		}
		
        if (printconst)
			llvm::errs() << " </USESEARCH" << (int)directions << ">" << *inst << "\n";
        constants_tmp.insert(constants2.begin(), constants2.end());
	}

    if (directions == 3)
	  nonconstant.insert(inst);
    if (printconst)
	  llvm::errs() << "couldnt decide nonconstants(" << (int)directions << "):" << *inst << "\n";
	return false;
}

// TODO separate if the instruction is constant (i.e. could change things)
//    from if the value is constant (the value is something that could be differentiated)
bool isconstantValueM(Value* val, SmallPtrSetImpl<Value*> &constants, SmallPtrSetImpl<Value*> &nonconstant, SmallPtrSetImpl<Value*> &constantvals, SmallPtrSetImpl<Value*> &retvals, AAResults &AA, uint8_t directions) {
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
    
    if((constantvals.find(val) != constantvals.end())) {
        if (printconst)
		    llvm::errs() << " VALUE const from precomputation " << *val << "\n";
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
            constantvals.insert(val);
            gi->setMetadata("enzyme_activity_value", MDNode::get(gi->getContext(), MDString::get(gi->getContext(), "const")));
            return true;
        }
        //TODO consider this more
        if (gi->isConstant() && isconstantValueM(gi->getInitializer(), constants, nonconstant, constantvals, retvals, AA, directions)) {
            constantvals.insert(val);
            gi->setMetadata("enzyme_activity_value", MDNode::get(gi->getContext(), MDString::get(gi->getContext(), "const")));
            return true;
        }
    }

    if (auto ce = dyn_cast<ConstantExpr>(val)) {
        if (ce->isCast()) {
            if (isconstantValueM(ce->getOperand(0), constants, nonconstant, constantvals, retvals, AA, directions)) {
                constantvals.insert(val);
                return true;
            }
        }
        if (ce->isGEPWithNoNotionalOverIndexing()) {
            if (isconstantValueM(ce->getOperand(0), constants, nonconstant, constantvals, retvals, AA, directions)) {
                constantvals.insert(val);
                return true;
            }
        }
    }
    
    if (auto inst = dyn_cast<Instruction>(val)) {
        if (isconstantM(inst, constants, nonconstant, constantvals, retvals, AA, directions)) {
            constantvals.insert(val);
            return true;
        }
    }
    
    //! This instruction is certainly an integer (and only and integer, not a pointer or float). Therefore its value is constant
    //TODO use typeInfo for more aggressive activity analysis
    if (val->getType()->isIntOrIntVectorTy() && isIntASecretFloat({}, val, /*default*/IntType::Pointer)==IntType::Integer) {
		if (printconst)
			llvm::errs() << " Value const as integral " << (int)directions << " " << *val << "\n";
        constantvals.insert(val);
        return true;
    }
   
    if ((directions & DOWN) && (retvals.find(val) == retvals.end()) ) { 
    //if (val->getType()->isFPOrFPVectorTy() && (directions & DOWN) && (retvals.find(val) == retvals.end()) ) { 
		auto &constants2 = constants;
		auto &nonconstant2 = nonconstant;
		auto &constantvals2 = constantvals;
		auto &retvals2 = retvals;

		if (printconst)
			llvm::errs() << " <Value USESEARCH" << (int)directions << ">" << *val << "\n";

		bool seenuse = false;
		
        for (const auto &a:val->users()) {

		    if (printconst)
			  llvm::errs() << "      considering use of " << *val << " - " << *a << "\n";

			if (auto call = dyn_cast<CallInst>(a)) {
                if (isFunctionArgumentConstant(call, val, constants2, nonconstant2, constantvals2, retvals2, AA, DOWN)) {
                    if (printconst) {
			          llvm::errs() << "Value found constant callinst use:" << *val << " user " << *call << "\n";
                    }
                    continue;
                }
			}
            
		  	if (!isconstantM(cast<Instruction>(a), constants2, nonconstant2, constantvals2, retvals2, AA, DOWN)) {
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
