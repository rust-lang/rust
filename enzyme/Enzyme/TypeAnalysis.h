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

    DataType(IntType typeEnum=IntType::Unknown) : type(nullptr), typeEnum(typeEnum) {
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
			llvm::errs() << "typeEnum: " << to_string(typeEnum) << " dt.typeEnum.str(): " << to_string(dt.typeEnum) << "\n";
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

class ValueData {
    private:
public:
        //mapping of known indices to type if one exists
        std::map<const std::vector<int>, DataType> mapping;

    public:
        bool operator<(const ValueData& vd) const {
            return mapping < vd.mapping;
        }

        ValueData() {}
        ValueData(DataType dat) {
            if (dat != DataType(IntType::Unknown)) {
                mapping[{}] = dat;
            }
        }
        
        static ValueData Unknown() {
            return ValueData();
        }

		ValueData JustInt() const {
			ValueData vd;
			for(auto &pair : mapping) {
				if (pair.second.typeEnum == IntType::Integer) {
					vd.mapping[pair.first] = pair.second;
				}
			}	

			return vd;
		}
        
		//needed for handling casts
		ValueData KeepFirst() const {
			ValueData vd;
			if (mapping.find({0}) != mapping.end()) {
				vd.mapping[{0}] = mapping.find({0})->second;
			}
			if (mapping.find({-1}) != mapping.end()) {
				vd.mapping[{-1}] = mapping.find({-1})->second;
			}

			return vd;
		}

        static std::vector<int> mergeIndices(std::vector<int> first, const std::vector<int> &second) {
            //assert(first.size() > 0);
            //assert(second.size() > 0);

            //-1 represents all elements in that range
            //if (first[0] >= 0 && second[0] >= 0) {
            //    first[0] += second[0];
            //} else {
            //    first[0] = -1;
            //}

            for(unsigned i=0; i<second.size(); i++)
                first.push_back(second[i]);
            return first;
        }

        ValueData Only(std::vector<int> indices) const {
            ValueData dat;

            for(const auto &pair : mapping) {
                dat.mapping[mergeIndices(indices, pair.first)] = pair.second;
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
                    dat.mapping[next] = pair.second;
            }

            return dat;
        }

        //Removes any anything types
        ValueData PurgeAnything() const {
            ValueData dat;
            for(const auto &pair : mapping) {
                if (pair.second == DataType(IntType::Anything)) continue;
                dat.mapping[pair.first] = pair.second;
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
            bool changed = false;
            
            for(auto &pair : v.mapping) {
                changed |= ( mapping[pair.first] |= pair.second );
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
				out += std::to_string(i); 
			}
			out +="]:" + pair.second.str();
			first = false;
		}
		out += "}";
		return out;
	}
};

typedef std::map<llvm::Argument*, DataType> FnTypeInfo; 

typedef std::map<llvm::Argument*, ValueData> NewFnTypeInfo;

class TypeAnalyzer;
class TypeAnalysis;

class TypeResults {
	TypeAnalysis &analysis;
	const NewFnTypeInfo info;
public:
	TypeResults(TypeAnalysis &analysis, const NewFnTypeInfo& fn);
	DataType intType(llvm::Value* val);
};

class TypeAnalyzer : public llvm::InstVisitor<TypeAnalyzer> {
public:
    llvm::Function* function;

    //Calling context
    const NewFnTypeInfo fntypeinfo;
    
    //List of value's which should be re-analyzed now with new information
    std::deque<llvm::Value*> workList;

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
};


class TypeAnalysis {
public:

    std::map<NewFnTypeInfo, TypeAnalyzer > analyzedFunctions;

    TypeResults analyzeFunction(const NewFnTypeInfo& fn, llvm::Function* function);
    
	ValueData query(llvm::Value* val, const NewFnTypeInfo& fn);

    DataType intType(llvm::Value* val, const NewFnTypeInfo& fn, bool errIfNotFound=true);
    DataType firstPointer(llvm::Value* val, const NewFnTypeInfo& fn, bool errIfNotFound=true);


    inline TypeResults analyzeFunction(const FnTypeInfo& fn, llvm::Function* function) {
        NewFnTypeInfo nti;
        for(auto &pair : fn) {
            nti[pair.first] = ValueData::Argument(pair.second, pair.first);
        }
        return analyzeFunction(nti, function);
    }
    inline DataType intType(llvm::Value* val, const FnTypeInfo& fn, bool errIfNotFound=true) {
        NewFnTypeInfo nti;
        for(auto &pair : fn) {
            nti[pair.first] = ValueData::Argument(pair.second, pair.first);
        }
        return intType(val, nti, errIfNotFound);
    }
    inline DataType firstPointer(llvm::Value* val, const FnTypeInfo& fn, bool errIfNotFound=true) {
        NewFnTypeInfo nti;
        for(auto &pair : fn) {
            nti[pair.first] = ValueData::Argument(pair.second, pair.first);
        }
        return firstPointer(val, nti, errIfNotFound);
    }
};

#endif
