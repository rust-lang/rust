/*
 * TypeAnalysis.h - Type Analysis Detection Utilities
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

#ifndef ENZYME_TYPE_ANALYSIS_H
#define ENZYME_TYPE_ANALYSIS_H 1

#include <cstdint>
#include <deque>

#include <llvm/Config/llvm-config.h>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/IR/InstVisitor.h"

enum class IntType {
    //integral type
    Integer,
    //floating point
    Float,
    //pointer
    Pointer,
    //can be anything of users choosing [usually result of a constant]
    Anything,
    //insufficient information
    Unknown
};


static inline std::string to_string(IntType t) {
    switch(t) {
        case IntType::Integer: return "Integer";
        case IntType::Float: return "Float";
        case IntType::Pointer: return "Pointer";
        case IntType::Anything: return "Anything";
        case IntType::Unknown: return "Unknown";
    }
    llvm_unreachable("unknown inttype");
}

static inline IntType parseIntType(std::string str) {
    if (str == "Integer") return IntType::Integer;
    if (str == "Float") return IntType::Float;
    if (str == "Pointer") return IntType::Pointer;
    if (str == "Anything") return IntType::Anything;
    if (str == "Unknown") return IntType::Unknown;
    llvm_unreachable("unknown inttype str");
}

class DataType {
public:
    llvm::Type* type;
    IntType typeEnum;

    DataType(llvm::Type* type) : type(type), typeEnum(IntType::Float) {
        assert(type != nullptr);
    }

    DataType(IntType typeEnum) : type(nullptr), typeEnum(typeEnum) {
        assert(typeEnum != IntType::Float);
    }

    DataType(std::string str, llvm::LLVMContext &C) {
        auto fd = str.find('@');
        if (fd != std::string::npos) {
            typeEnum = IntType::Float;
            assert(str.substr(0, fd) == "Float");
            auto subt = str.substr(fd+1);
            if (subt == "half") {
                type = llvm::Type::getHalfTy(C);
            } else if (subt == "float") {
                type = llvm::Type::getFloatTy(C);
            } else if (subt == "double") {
                type = llvm::Type::getDoubleTy(C);
            } else if (subt == "fp80") {
                type = llvm::Type::getX86_FP80Ty(C);
            } else if (subt == "fp128") {
                type = llvm::Type::getFP128Ty(C);
            } else if (subt == "ppc128") {
                type = llvm::Type::getPPC_FP128Ty(C);
            } else {
                llvm_unreachable("unknown data type");
            }
        } else {
            type = nullptr;
            typeEnum = parseIntType(str);
        }
    }

    bool isKnown() const {
        return typeEnum != IntType::Unknown;
    }
    
    llvm::Type* isFloat() const {
        return type;
    }
    
	bool operator==(const IntType dt) const {
        return typeEnum == dt;
    }
	
	bool operator!=(const IntType dt) const {
        return typeEnum != dt;
    }

    bool operator==(const DataType dt) const {
        return type == dt.type && typeEnum == dt.typeEnum;
    }
    bool operator!=(const DataType dt) const {
        return !(*this == dt);
    }
    
    //returns whether changed
    bool operator=(const DataType dt) {
        bool changed = false;
        if (typeEnum != dt.typeEnum) changed = true;
        typeEnum = dt.typeEnum;
        if (type != dt.type) changed = true;
        type = dt.type;
        return changed;
    }
    
    //returns whether changed
    bool operator|=(const DataType dt) {
        if (typeEnum == IntType::Anything) {
            return false;
        }
        if (dt.typeEnum == IntType::Anything) {
            return *this = dt;
        }
        if (typeEnum == IntType::Unknown) {
            return *this = dt;
        }
        if (dt.typeEnum == IntType::Unknown) {
            return false;
        }
		if (dt.typeEnum != typeEnum) {
			llvm::errs() << "typeEnum: " << to_string(typeEnum) << " dt.typeEnum.str(): " << to_string(dt.typeEnum) << "\n";
		}
        assert(dt.typeEnum == typeEnum);
		if (dt.type != type) {
			llvm::errs() << "type: " << *type << " dt.type: " << *dt.type << "\n";
		}
        assert(dt.type == type);
        return false;
    }

    bool pointerIntMerge(const DataType dt) {
        //TODO consider &= pointer/int implications
        if (dt.typeEnum == IntType::Pointer && typeEnum == IntType::Integer) {
            typeEnum = IntType::Pointer;
            return true;
        }
        if (typeEnum == IntType::Pointer && dt.typeEnum == IntType::Integer) {
            return false;
        }

        return *this &= dt;
    }
    
    //returns whether changed
    bool operator&=(const DataType dt) {
        if (typeEnum == IntType::Anything) {
            return *this = dt;
        }
        if (dt.typeEnum == IntType::Anything) {
            return false;
        }
        if (typeEnum == IntType::Unknown) {
            return false;
        }
        if (dt.typeEnum == IntType::Unknown) {
            return *this = dt;
        }

		if (dt.typeEnum != typeEnum) {
			llvm::errs() << "&= typeEnum: " << to_string(typeEnum) << " dt.typeEnum.str(): " << to_string(dt.typeEnum) << "\n";
            return *this = IntType::Unknown;
		}
        assert(dt.typeEnum == typeEnum);
		if (dt.type != type) {
			llvm::errs() << "type: " << *type << " dt.type: " << *dt.type << "\n";
		}
        assert(dt.type == type);
        return false;
    }
     
    bool operator<(const DataType dt) const {
        if (typeEnum == dt.typeEnum) {
            return type < dt.type;
        } else {
            return typeEnum < dt.typeEnum;
        }
    }
	std::string str() const {
		std::string res = to_string(typeEnum);
		if (typeEnum == IntType::Float) {
			if (type->isHalfTy()) {
				res += "@half";
			} else if (type->isFloatTy()) {
				res += "@float";
			} else if (type->isDoubleTy()) {
				res += "@double";
			} else if (type->isX86_FP80Ty()) {
				res += "@fp80";
			} else if (type->isFP128Ty()) {
				res += "@fp128";
			} else if (type->isPPC_FP128Ty()) {
				res += "@ppc128";
			} else {
				llvm_unreachable("unknown data type");
			}
		}
		return res;
	}
};
    
static inline std::string to_string(const DataType dt) {
	return dt.str();
}


static inline std::string to_string(const std::vector<int> x) {
    std::string out = "[";
    for(unsigned i=0; i<x.size(); i++) {
        if (i != 0) out +=",";
        out += std::to_string(x[i]); 
    }
    out +="]";
    return out;
}

class ValueData {
    private:
        //mapping of known indices to type if one exists
        std::map<const std::vector<int>, DataType> mapping;
public:

    public:
        DataType operator[] (const std::vector<int> v) const {
            auto found = mapping.find(v);
            if (found != mapping.end()) {
                return found->second;
            }
            return IntType::Unknown;
        }

        void insert(const std::vector<int> v, DataType d) {
            if (v.size() > 0) {
                std::vector<int> tmp(v.begin(), v.end()-1);
                auto found = mapping.find(tmp);
                if (found != mapping.end()) {
                    if (!(found->second == IntType::Pointer || found->second== IntType::Anything)) {
                        llvm::errs() << "FAILED dt: " << str() << " adding v: " << to_string(v) << ": " << d.str() << "\n";
                    }
                    assert(found->second == IntType::Pointer || found->second== IntType::Anything);
                }
            }
            if (v.size() > 6) {
                llvm::errs() << "not handling more than 6 pointer lookups deep dt:" << str() << " adding v: " << to_string(v) << ": " << d.str() << "\n";
                return;
            }
            for(auto a : v) {
                if (a > 1000) {
                    //llvm::errs() << "not handling more than 1000B offset pointer dt:" << str() << " adding v: " << to_string(v) << ": " << d.str() << "\n";
                    return; 
                }
            }
            mapping.insert(std::pair<const std::vector<int>, DataType>(v, d));
        }

        bool operator<(const ValueData& vd) const {
            return mapping < vd.mapping;
        }

        ValueData() {}
        ValueData(DataType dat) {
            if (dat != DataType(IntType::Unknown)) {
                insert({}, dat);
            }
        }
        
        llvm::Type* getKnownFloat(llvm::Value* val) const {
            for(const auto &pair : mapping) {
                auto s = pair.second;
                if (s == IntType::Anything) continue;
                if (s == IntType::Unknown) continue;
                return s.isFloat();
            }
            if (auto inst = llvm::dyn_cast<llvm::Instruction>(val)) {
                llvm::errs() << *inst->getParent()->getParent() << "\n";
            }
            llvm::errs() << " could not find known for value " << *val << " dt: " << str() << "\n";
            assert(0 && "could not find known for value");
            exit(1);
        }
        
        static ValueData Unknown() {
            return ValueData();
        }

		ValueData JustInt() const {
			ValueData vd;
			for(auto &pair : mapping) {
				if (pair.second.typeEnum == IntType::Integer) {
					vd.insert(pair.first, pair.second);
				}
			}	

			return vd;
		}
        
        //TODO keep type information that is striated
        // e.g. if you have an i8* [0:Int, 8:Int] => i64* [0:Int, 1:Int]
        // After a depth len into the index tree, prune any lookups that are not {0} or {-1}
        ValueData KeepForCast(const llvm::DataLayout& dl, llvm::Type* from, llvm::Type* to) const;

        static std::vector<int> appendIndices(std::vector<int> first, const std::vector<int> &second) {
            for(unsigned i=0; i<second.size(); i++)
                first.push_back(second[i]);
            return first;
        }

        ValueData Only(std::vector<int> indices) const {
            ValueData dat;

            for(const auto &pair : mapping) {
                dat.insert(appendIndices(indices, pair.first), pair.second);
                if (pair.first.size() > 0) {
                    dat.insert(indices, DataType(IntType::Pointer));
                }
            }

            return dat;
        }
        
        static bool lookupIndices(std::vector<int> &first, const std::vector<int> &second) {
            if (first.size() > second.size()) return false;

            auto fs = first.size();
            for(unsigned i=0; i<fs; i++) {
                if (first[i] == -1) continue;
                if (second[i] == -1) continue;
                if (first[i] != second[i]) return false;
            }

            first.clear();
            for(auto i=fs; i<second.size(); i++) {
                first.push_back(second[i]);
            }
			return true;
        }

        ValueData Lookup(std::vector<int> indices) const {

            ValueData dat;

            for(const auto &pair : mapping) {
                std::vector<int> next = indices;
                if (lookupIndices(next, pair.first)) 
                    dat.insert(next, pair.second);
            }

            return dat;
        }
        
        static std::vector<int> mergeIndices(int offset, const std::vector<int> &second) {
            assert(second.size() > 0);
            
            std::vector<int> next(second);
            //-1 represents all elements in that range
            if (offset == -1 || next[0] == -1) {
                next[0] = -1;
            } else {
                next[0] += offset;
            }

            assert(next.size() > 0);
            return next;
        }

        ValueData MergeIndices(int offset) const {
            ValueData dat;

            for(const auto &pair : mapping) {
                ValueData dat2;

                if (pair.first.size() == 0) {
                    if (pair.second == IntType::Pointer || pair.second == IntType::Anything) continue;

                    llvm::errs() << "could not merge test  " << str() << "\n";
                }
                dat2.insert(mergeIndices(offset, pair.first), pair.second);
                dat |= dat2;
            }

            return dat;
        }

        static llvm::Type* indexIntoType(llvm::Type* ty, int idx) {
            if (ty == nullptr) return nullptr;
            if (idx == -1) idx = 0;
            if (auto st = llvm::dyn_cast<llvm::StructType>(ty)) {
                return st->getElementType(idx);
            }
            if (auto at = llvm::dyn_cast<llvm::ArrayType>(ty)) {
                return at->getElementType();
            }
            return nullptr;
        }

        // given previous [0, 1, 2], index[0, 1] we should get back [0, 2]
        // we should also have type dependent [2], index[1], if index.type[1] cast to index.type[2] permits
        static bool unmergeIndices(std::vector<int>& next, int offset, const std::vector<int> &previous) {
            assert(next.size() == 0);

            assert(previous.size() > 0);

            next.assign(previous.begin(), previous.end());

            if (next[0] == -1) {
                return true;
            }

            if (next[0] < offset) {
                return false;
            }

            next[0] -= offset;
            return true;
        }

        //We want all the data from this value, given that we are indexing with indices
        // E.g. we might have a { [0, 1, 2]: Int, [5, 10, 30]: Pointer}, we may index [0, 1] and should get back [0, 2]:Int
        ValueData UnmergeIndices(int offset) const {
            ValueData dat;

            for(const auto &pair : mapping) {
                ValueData dat2;
                std::vector<int> next;

                if (pair.first.size() == 0) {
                    if (pair.second == IntType::Pointer) continue;

                    llvm::errs() << "could not unmerge " << str() << "\n";
                }
                assert(pair.first.size() > 0);

                if (unmergeIndices(next, offset, pair.first)) {
                    //llvm::errs() << "next: " << to_string(next) << " indices: " << to_string(indices) << " pair.first: " << to_string(pair.first) << "\n";
                    dat2.insert(next, pair.second);
                }
                dat |= dat2;
            }

            return dat;
        }

        //Removes any anything types
        ValueData PurgeAnything() const {
            ValueData dat;
            for(const auto &pair : mapping) {
                if (pair.second == DataType(IntType::Anything)) continue;
                dat.insert(pair.first, pair.second);
            }
            return dat;
        }

        ValueData AtMost(int max) const {
            assert(max > 0);
            ValueData dat;
            for(const auto &pair : mapping) {
                if (pair.first.size() == 0 || pair.first[0] == -1 || pair.first[0] < max) {
                    dat.insert(pair.first, pair.second);
                }
            }
            return dat;
        }

        static ValueData Argument(DataType type, llvm::Value* v) {
            if (v->getType()->isIntOrIntVectorTy()) return ValueData(type);
            return ValueData(type).Only({0});
        }

        bool operator==(const ValueData &v) const {
            return mapping == v.mapping;
        }

        // Return if changed
        bool operator=(const ValueData& v) {
            if (*this == v) return false;
            mapping = v.mapping;
            return true;
        }

        bool operator|=(const ValueData &v) {
            //! Detect recursive merge
            //if (v.mapping.size() == mapping.size() && mapping.size() ) {
            //    if 
            //}
            
            
            bool changed = false;

            if (v[{-1}] != IntType::Unknown) {
                for(auto &pair : mapping) {
                    if (pair.first.size() == 1 && pair.first[0] != -1) {
                        pair.second |= v[{-1}];
                        //if (pair.second == ) // NOTE DELETE the non -1 
                    }
                }
            }
            
            for(auto &pair : v.mapping) {
                assert(pair.second != IntType::Unknown);
                DataType dt = operator[](pair.first);
                //llvm::errs() << "merging @ " << to_string(pair.first) << " old " << dt.str() << pair.second.str() << "\n";
                changed |= (dt |= pair.second);
                insert(pair.first, dt);
            }

            return changed;
        }

        bool operator&=(const ValueData &v) {
            bool changed = false;
            
            std::vector<std::vector<int>> keystodelete;
            for(auto &pair : mapping) {
                DataType other = IntType::Unknown;
                auto fd = v.mapping.find(pair.first);
                if (fd != v.mapping.end()) {
                    other = fd->second;
                }
                changed = (pair.second &= other);
                if (pair.second == IntType::Unknown) {
                    keystodelete.push_back(pair.first);
                }
            }

            for(auto &key : keystodelete) {
                mapping.erase(key);
            }
            
            return changed;
        }


        bool pointerIntMerge(const ValueData &v) {
            bool changed = false;
            
            std::vector<std::vector<int>> keystodelete;
            for(auto &pair : mapping) {
                DataType other = IntType::Unknown;
                auto fd = v.mapping.find(pair.first);
                if (fd != v.mapping.end()) {
                    other = fd->second;
                }
                changed = (pair.second.pointerIntMerge(other));
                if (pair.second == IntType::Unknown) {
                    keystodelete.push_back(pair.first);
                }
            }

            for(auto &key : keystodelete) {
                mapping.erase(key);
            }
            
            return changed;
        }

	std::string str() const {
		std::string out = "{";
		bool first = true;
		for(auto &pair : mapping) {
			if (!first) {
				out += ", ";
			}
			out += "[";
			for(unsigned i=0; i<pair.first.size(); i++) {
				if (i != 0) out +=",";
				out += std::to_string(pair.first[i]); 
			}
			out +="]:" + pair.second.str();
			first = false;
		}
		out += "}";
		return out;
	}
};

typedef std::map<llvm::Argument*, DataType> FnTypeInfo; 

//First is arguments, then return type
typedef std::pair<std::map<llvm::Argument*, ValueData>, ValueData> NewFnTypeInfo;

class TypeAnalyzer;
class TypeAnalysis;

class TypeResults {
	TypeAnalysis &analysis;
	const NewFnTypeInfo info;
    llvm::Function* function;
public:
	TypeResults(TypeAnalysis &analysis, const NewFnTypeInfo& fn, llvm::Function* function);
	DataType intType(llvm::Value* val);
    NewFnTypeInfo getAnalyzedTypeInfo();
    ValueData getReturnAnalysis();
};

class TypeAnalyzer : public llvm::InstVisitor<TypeAnalyzer> {
private:
    //List of value's which should be re-analyzed now with new information
    std::deque<llvm::Value*> workList;
    void addToWorkList(llvm::Value* val);
public:
    llvm::Function* function;

    //Calling context
    const NewFnTypeInfo fntypeinfo;
    

	TypeAnalysis &interprocedural;
    
	std::map<llvm::Value*, ValueData> analysis;

    TypeAnalyzer(llvm::Function* function, const NewFnTypeInfo& fn, TypeAnalysis& TA);

    ValueData getAnalysis(llvm::Value* val);

    void updateAnalysis(llvm::Value* val, IntType data, llvm::Value* origin);

    void updateAnalysis(llvm::Value* val, ValueData data, llvm::Value* origin);

    void prepareArgs();

    void considerTBAA();
    
	void run();

    void visitValue(llvm::Value& val);

    void visitAllocaInst(llvm::AllocaInst &I);
    
    void visitLoadInst(llvm::LoadInst &I);
    
	void visitStoreInst(llvm::StoreInst &I);

    void visitGetElementPtrInst(llvm::GetElementPtrInst &gep);

    void visitPHINode(llvm::PHINode& phi);

	void visitTruncInst(llvm::TruncInst &I);
	
	void visitZExtInst(llvm::ZExtInst &I);
	
	void visitSExtInst(llvm::SExtInst &I);

	void visitAddrSpaceCastInst(llvm::AddrSpaceCastInst &I);
	
	void visitFPToUIInst(llvm::FPToUIInst &I);
	
	void visitFPToSIInst(llvm::FPToSIInst &I);
	
	void visitUIToFPInst(llvm::UIToFPInst &I);
	
    void visitSIToFPInst(llvm::SIToFPInst &I);
	
	void visitPtrToIntInst(llvm::PtrToIntInst &I);
    
    void visitIntToPtrInst(llvm::IntToPtrInst &I);

    void visitBitCastInst(llvm::BitCastInst &I);

    void visitSelectInst(llvm::SelectInst &I);

    void visitExtractElementInst(llvm::ExtractElementInst &I);

    void visitInsertElementInst(llvm::InsertElementInst &I);

    void visitShuffleVectorInst(llvm::ShuffleVectorInst &I);

    void visitExtractValueInst(llvm::ExtractValueInst &I);

    void visitInsertValueInst(llvm::InsertValueInst &I);

    void visitBinaryOperator(llvm::BinaryOperator &I);

	void visitIPOCall(llvm::CallInst& call, llvm::Function& fn);

    void visitCallInst(llvm::CallInst &call);

    ValueData getReturnAnalysis();
};


class TypeAnalysis {
public:

    std::map<NewFnTypeInfo, TypeAnalyzer > analyzedFunctions;

    TypeResults analyzeFunction(const NewFnTypeInfo& fn, llvm::Function* function);
    
	ValueData query(llvm::Value* val, const NewFnTypeInfo& fn);

    DataType intType(llvm::Value* val, const NewFnTypeInfo& fn, bool errIfNotFound=true);
    DataType firstPointer(llvm::Value* val, const NewFnTypeInfo& fn, bool errIfNotFound=true);
    
    //err if not known; if found float anywhere return the type
    inline llvm::Type* getKnownFloat(llvm::Value* val, const NewFnTypeInfo &fn) {
        return query(val, fn).getKnownFloat(val);
    }
    inline ValueData getReturnAnalysis(const NewFnTypeInfo &fn, llvm::Function* function) {
        analyzeFunction(fn, function);
        return analyzedFunctions.find(fn)->second.getReturnAnalysis();
    }
};

#endif
