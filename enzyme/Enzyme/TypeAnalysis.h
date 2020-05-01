/*
 * TypeAnalysis.h - Type Analysis Detection Utilities
 *
 * Copyright (C) 2020 William S. Moses (enzyme@wsmoses.com) - All Rights Reserved
 *
 * For commercial use of this code please contact the author(s) above.
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

#include "llvm/IR/Dominators.h"

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
        assert(!llvm::isa<llvm::VectorType>(type));
        if (!type->isFloatingPointTy()) {
            llvm::errs() << " passing in non FP type: " << *type << "\n";
        }
        assert(type->isFloatingPointTy());
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

    bool isIntegral() const {
        return typeEnum == IntType::Integer || typeEnum == IntType::Anything;
    }

    bool isKnown() const {
        return typeEnum != IntType::Unknown;
    }

    bool isPossiblePointer() const {
        return !isKnown() || typeEnum == IntType::Pointer;
    }

    bool isPossibleFloat() const {
        return !isKnown() || typeEnum == IntType::Float;
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
    bool legalMergeIn(const DataType dt, bool pointerIntSame, bool& legal) {
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
            if (pointerIntSame) {
                if ((typeEnum == IntType::Pointer && dt.typeEnum == IntType::Integer) ||
                    (typeEnum == IntType::Integer && dt.typeEnum == IntType::Pointer)) {
                    return false;
                }
            }
            legal = false;
            return false;
        }
        assert(dt.typeEnum == typeEnum);
        if (dt.type != type) {
            legal = false;
            return false;
        }
        assert(dt.type == type);
        return false;
    }

    //returns whether changed
    bool mergeIn(const DataType dt, bool pointerIntSame) {
        bool legal = true;
        bool res = legalMergeIn(dt, pointerIntSame, legal);
        if (!legal) {
            llvm::errs() << "me: " << str() << " right: " << dt.str() << "\n";
        }
        assert(legal);
        return res;
    }

    //returns whether changed
    bool operator|=(const DataType dt) {
        return mergeIn(dt, /*pointerIntSame*/false);
    }

    bool pointerIntMerge(const DataType dt, llvm::BinaryOperator::BinaryOps op) {
        bool changed = false;
        using namespace llvm;

        if (typeEnum == IntType::Anything && dt.typeEnum == IntType::Anything) {
            return changed;
        }

        if ((typeEnum == IntType::Unknown && dt.typeEnum == IntType::Anything) ||
            (typeEnum == IntType::Anything && dt.typeEnum == IntType::Unknown)) {
            if (typeEnum != IntType::Unknown) {
                typeEnum = IntType::Unknown;
                changed = true;
            }
            return changed;
        }

        if ((typeEnum == IntType::Integer && dt.typeEnum == IntType::Integer) ||
            (typeEnum == IntType::Unknown && dt.typeEnum == IntType::Integer) ||
            (typeEnum == IntType::Integer && dt.typeEnum == IntType::Unknown) ||
            (typeEnum == IntType::Anything && dt.typeEnum == IntType::Integer) ||
            (typeEnum == IntType::Integer && dt.typeEnum == IntType::Anything)
            ) {
            switch(op) {
                case BinaryOperator::Add:
                case BinaryOperator::Sub:
                    // if one of these is unknown we cannot deduce the result
                    // e.g. pointer + int = pointer and int + int = int
                    if (typeEnum == IntType::Unknown || dt.typeEnum == IntType::Unknown) {
                        if (typeEnum != IntType::Unknown) {
                            typeEnum = IntType::Unknown;
                            changed = true;
                        }
                        return changed;
                    }

                case BinaryOperator::Mul:
                case BinaryOperator::UDiv:
                case BinaryOperator::SDiv:
                case BinaryOperator::URem:
                case BinaryOperator::SRem:
                case BinaryOperator::And:
                case BinaryOperator::Or:
                case BinaryOperator::Xor:
                case BinaryOperator::Shl:
                case BinaryOperator::AShr:
                case BinaryOperator::LShr:
                    //! Anything << 16   ==> Anything
                    if (typeEnum == IntType::Anything) {
                        break;
                    }
                    if (typeEnum != IntType::Integer) {
                        typeEnum = IntType::Integer;
                        changed = true;
                    }
                    break;
                default:
                    llvm_unreachable("unknown binary operator");
            }
            return changed;
        }

        if (typeEnum == IntType::Pointer && dt.typeEnum == IntType::Pointer) {
            switch(op) {
                case BinaryOperator::Sub:
                    typeEnum = IntType::Integer;
                    changed = true;
                    break;
                case BinaryOperator::Add:
                case BinaryOperator::Mul:
                case BinaryOperator::UDiv:
                case BinaryOperator::SDiv:
                case BinaryOperator::URem:
                case BinaryOperator::SRem:
                case BinaryOperator::And:
                case BinaryOperator::Or:
                case BinaryOperator::Xor:
                case BinaryOperator::Shl:
                case BinaryOperator::AShr:
                case BinaryOperator::LShr:
                    llvm_unreachable("illegal pointer/pointer operation");
                    break;
                default:
                    llvm_unreachable("unknown binary operator");
            }
            return changed;
        }

        if ((typeEnum == IntType::Integer && dt.typeEnum == IntType::Pointer) ||
            (typeEnum == IntType::Pointer && dt.typeEnum == IntType::Integer) ||
            (typeEnum == IntType::Integer && dt.typeEnum == IntType::Pointer) ||
            (typeEnum == IntType::Pointer && dt.typeEnum == IntType::Unknown) ||
            (typeEnum == IntType::Unknown && dt.typeEnum == IntType::Pointer) ||
            (typeEnum == IntType::Pointer && dt.typeEnum == IntType::Anything) ||
            (typeEnum == IntType::Anything && dt.typeEnum == IntType::Pointer)
            ){

            switch(op) {
                case BinaryOperator::Sub:
                    if (typeEnum == IntType::Anything || dt.typeEnum == IntType::Anything) {
                        if (typeEnum != IntType::Unknown) {
                            typeEnum = IntType::Unknown;
                            changed = true;
                        }
                        break;
                    }
                case BinaryOperator::Add:
                case BinaryOperator::Mul:
                    if (typeEnum != IntType::Pointer) {
                        typeEnum = IntType::Pointer;
                        changed = true;
                    }
                    break;
                case BinaryOperator::UDiv:
                case BinaryOperator::SDiv:
                case BinaryOperator::URem:
                case BinaryOperator::SRem:
                    if (dt.typeEnum == IntType::Pointer) {
                        llvm_unreachable("cannot divide integer by pointer");
                    } else if (typeEnum != IntType::Unknown) {
                        typeEnum = IntType::Unknown;
                        changed = true;
                    }
                    break;
                case BinaryOperator::And:
                case BinaryOperator::Or:
                case BinaryOperator::Xor:
                case BinaryOperator::Shl:
                case BinaryOperator::AShr:
                case BinaryOperator::LShr:
                    if (typeEnum != IntType::Unknown) {
                        typeEnum = IntType::Unknown;
                        changed = true;
                    }
                    break;
                default:
                    llvm_unreachable("unknown binary operator");
            }
            return changed;
        }

        if (dt.typeEnum == IntType::Integer) {
            switch(op) {
                case BinaryOperator::Shl:
                case BinaryOperator::AShr:
                case BinaryOperator::LShr:
                    if (typeEnum != IntType::Unknown) {
                        typeEnum = IntType::Unknown;
                        changed = true;
                        return changed;
                    }
                    break;
                default: break;
            }
        }

        llvm::errs() << "self: " << str() << " other: " << dt.str() << " op: " << op << "\n";
        llvm_unreachable("unknown case");
    }

    bool andIn(const DataType dt, bool assertIfIllegal=true) {
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
            if (!assertIfIllegal) {
                return *this = IntType::Unknown;
            }
			llvm::errs() << "&= typeEnum: " << to_string(typeEnum) << " dt.typeEnum.str(): " << to_string(dt.typeEnum) << "\n";
            return *this = IntType::Unknown;
		}
        assert(dt.typeEnum == typeEnum);
		if (dt.type != type) {
            if (!assertIfIllegal) {
                return *this = IntType::Unknown;
            }
			llvm::errs() << "type: " << *type << " dt.type: " << *dt.type << "\n";
		}
        assert(dt.type == type);
        return false;
    }

    //returns whether changed
    bool operator&=(const DataType dt) {
        return andIn(dt, /*assertIfIllegal*/true);
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

class ValueData;

typedef std::shared_ptr<const ValueData> TypeResult;
typedef std::map<const std::vector<int>, DataType> DataTypeMapType;
typedef std::map<const std::vector<int>, const TypeResult> ValueDataMapType;

class ValueData : public std::enable_shared_from_this<ValueData> {
private:
    //mapping of known indices to type if one exists
    DataTypeMapType mapping;

    //mapping of known indices to type if one exists
    //ValueDataMapType recur_mapping;

    static std::map<std::pair<DataTypeMapType, ValueDataMapType>, TypeResult> cache;
public:
    DataType operator[] (const std::vector<int> v) const {
        auto found = mapping.find(v);
        if (found != mapping.end()) {
            return found->second;
        }
        for(const auto& pair : mapping) {
            if (pair.first.size() != v.size()) continue;
            bool match = true;
            for(unsigned i=0; i<pair.first.size(); i++) {
                if (pair.first[i] == -1) continue;
                if (pair.first[i] != v[i]) {
                    match = false;
                    break;
                }
            }
            if (!match) continue;
            return pair.second;
        }
        return IntType::Unknown;
    }

    void erase(const std::vector<int> v) {
        mapping.erase(v);
    }

    void insert(const std::vector<int> v, DataType d, bool intsAreLegalSubPointer=false) {
        if (v.size() > 0) {
            //check pointer abilities from before
            {
            std::vector<int> tmp(v.begin(), v.end()-1);
            auto found = mapping.find(tmp);
            if (found != mapping.end()) {
                if (!(found->second == IntType::Pointer || found->second== IntType::Anything)) {
                    llvm::errs() << "FAILED dt: " << str() << " adding v: " << to_string(v) << ": " << d.str() << "\n";
                }
                assert(found->second == IntType::Pointer || found->second== IntType::Anything);
            }
            }

            //don't insert if there's an existing ending -1
            {
            std::vector<int> tmp(v.begin(), v.end()-1);
            tmp.push_back(-1);
            auto found = mapping.find(tmp);
            if (found != mapping.end()) {

                if (found->second != d) {
                    if (d == IntType::Anything) {
                        found->second = d;
                    } else {
                        llvm::errs() << "FAILED dt: " << str() << " adding v: " << to_string(v) << ": " << d.str() << "\n";
                    }
                }
                assert(found->second == d);
                return;
            }
            }

            //don't insert if there's an existing starting -1
            {
            std::vector<int> tmp(v.begin(), v.end());
            tmp[0] = -1;
            auto found = mapping.find(tmp);
            if (found != mapping.end()) {
                if (found->second != d) {
                    if (d == IntType::Anything) {
                        found->second = d;
                    } else {
                        llvm::errs() << "FAILED dt: " << str() << " adding v: " << to_string(v) << ": " << d.str() << "\n";
                    }
                }
                assert(found->second == d);
                return;
            }
            }

            //if this is a ending -1, remove other -1's
            if (v.back() == -1){
                std::set<std::vector<int>> toremove;
                for(const auto& pair : mapping) {
                    if (pair.first.size() == v.size()) {
                        bool matches = true;
                        for(unsigned i=0; i<pair.first.size()-1; i++) {
                            if (pair.first[i] != v[i]) {
                                matches = false;
                                break;
                            }
                        }
                        if (!matches) continue;

                        if (intsAreLegalSubPointer && pair.second.typeEnum == IntType::Integer && d.typeEnum == IntType::Pointer) {

                        } else {
                            if (pair.second != d) {
                                llvm::errs() << "inserting into : " << str() << " with " << to_string(v) << " of " << d.str() << "\n";
                            }
                            assert(pair.second == d);
                        }
                        toremove.insert(pair.first);
                    }
                }

                for(const auto & val : toremove) {
                    mapping.erase(val);
                }

            }

            //if this is a starting -1, remove other -1's
            if (v[0] == -1){
                std::set<std::vector<int>> toremove;
                for(const auto& pair : mapping) {
                    if (pair.first.size() == v.size()) {
                        bool matches = true;
                        for(unsigned i=1; i<pair.first.size(); i++) {
                            if (pair.first[i] != v[i]) {
                                matches = false;
                                break;
                            }
                        }
                        if (!matches) continue;
                        assert(pair.second == d);
                        toremove.insert(pair.first);
                    }
                }

                for(const auto & val : toremove) {
                    mapping.erase(val);
                }

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

    bool isKnown() {
        for(auto &pair : mapping) {
            // we should assert here as we shouldn't keep any unknown maps for efficiency
            assert(pair.second.isKnown());
        }
        return mapping.size() != 0;
    }

    bool isKnownPastPointer() {
        for(auto &pair : mapping) {
            // we should assert here as we shouldn't keep any unknown maps for efficiency
            assert(pair.second.isKnown());
            if (pair.first.size() == 0) {
                assert(pair.second == IntType::Pointer);
                continue;
            }
            return true;
        }
        return false;
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

    static std::vector<int> appendIndex(int off, const std::vector<int> &first) {
        std::vector<int> out;
        out.push_back(off);
        for(auto a : first) out.push_back(a);
        return out;
    }

    ValueData Only(int off) const {
        ValueData dat;

        for(const auto &pair : mapping) {
            dat.insert(appendIndex(off, pair.first), pair.second);
            //if (pair.first.size() > 0) {
            //    dat.insert(indices, DataType(IntType::Pointer));
            //}
        }

        return dat;
    }

    static bool lookupIndices(std::vector<int> &first, int idx, const std::vector<int> &second) {
        if (second.size() == 0) return false;

        assert(first.size() == 0);
        
        if (idx == -1) {
        } else if (second[0] == -1) {
        } else if(idx != second[0]) {
            return false;
        }

        for(size_t i=1; i<second.size(); i++) {
            first.push_back(second[i]);
        }
		return true;
    }

    ValueData Data0() const {
        ValueData dat;

        for(const auto &pair : mapping) {
            assert(pair.first.size() != 0);

            if (pair.first[0] == 0 || pair.first[0] == -1) {
                std::vector<int> next;
                for(size_t i=1; i<pair.first.size(); i++)
                    next.push_back(pair.first[i]);
                ValueData dat2;
                dat2.insert(next, pair.second);
                dat |= dat2;
            }
        }

        return dat;
    }

    ValueData Clear(size_t start, size_t end, size_t len) const {
        ValueData dat;

        for(const auto &pair : mapping) {
            assert(pair.first.size() != 0);

            if (pair.first[0] == -1) {
                ValueData dat2;
                auto next = pair.first;
                for(size_t i=0; i<start; i++) {
                    next[0] = i;
                    dat2.insert(next, pair.second);
                }
                for(size_t i=end; i<len; i++) {
                    next[0] = i;
                    dat2.insert(next, pair.second);
                }
                dat |= dat2;
            } else if ((size_t)pair.first[0] > start && (size_t)pair.first[0] >= end && (size_t)pair.first[0] < len) {
                ValueData dat2;
                dat2.insert(pair.first, pair.second);
                dat |= dat2;
            }
        }

        //TODO canonicalize this
        return dat; 
    }

    ValueData Lookup(size_t len, const llvm::DataLayout &dl) const {

        // Map of indices[1:] => ( End => possible Index[0] )
        std::map<std::vector<int>, std::map<DataType, std::set<int> >> staging;

        for(const auto &pair : mapping) {
            assert(pair.first.size() != 0);

            // Pointer is at offset 0 from this object
            if (pair.first[0] != 0 && pair.first[0] != -1) continue;

            if (pair.first.size() == 1) {
                assert(pair.second == DataType(IntType::Pointer) || pair.second == DataType(IntType::Anything));
                continue;
            }
            
            if (pair.first[1] == -1) {
            } else {
                if ((size_t)pair.first[1] >= len) continue;
            }

            std::vector<int> next;
            for(size_t i=2; i<pair.first.size(); i++) {
                next.push_back(pair.first[i]);
            }

            staging[next][pair.second].insert(pair.first[1]);
        }

        ValueData dat;
        for(auto & pair : staging) {
            auto &pnext = pair.first;
            for(auto & pair2 : pair.second) {
                auto &dt = pair2.first;
                auto &set = pair2.second;

                bool legalCombine = set.count(-1);

                // See if we can canonicalize the outermost index into a -1
                if (!legalCombine) {
                    size_t chunk = 1;
                    if (auto flt = dt.isFloat()) {
                        if (flt->isFloatTy()) {
                            chunk = 4;
                        } else if(flt->isDoubleTy()) {
                            chunk = 8;
                        } else if(flt->isHalfTy()) {
                            chunk = 2;
                        } else {
                            llvm::errs() << *flt << "\n";
                            assert(0 && "unhandled float type");
                        }
                    } else if (dt.typeEnum == IntType::Pointer) {
                        chunk = dl.getPointerSizeInBits() / 8;
                    }

                    legalCombine = true;
                    for(size_t i=0; i<len; i+=chunk) {
                        if (!set.count(i)) {
                            legalCombine = false;
                            break;
                        }
                    }
                }

                std::vector<int> next;
                next.push_back(-1);
                for(auto v : pnext) next.push_back(v);

                if (legalCombine) {
                    dat.insert(next, dt, /*intsAreLegalPointerSub*/true);
                } else {
                    for(auto e : set) {
                        next[0] = e;
                        dat.insert(next, dt);
                    }
                }

            }
        }

        return dat;
    }

    ValueData CanonicalizeValue(size_t len, const llvm::DataLayout& dl) const {

        // Map of indices[1:] => ( End => possible Index[0] )
        std::map<std::vector<int>, std::map<DataType, std::set<int> >> staging;

        for(const auto &pair : mapping) {
            assert(pair.first.size() != 0);

            std::vector<int> next;
            for(size_t i=1; i<pair.first.size(); i++) {
                next.push_back(pair.first[i]);
            }

            staging[next][pair.second].insert(pair.first[0]);
        }

        ValueData dat;
        for(auto & pair : staging) {
            auto &pnext = pair.first;
            for(auto & pair2 : pair.second) {
                auto &dt = pair2.first;
                auto &set = pair2.second;

                bool legalCombine = set.count(-1);

                // See if we can canonicalize the outermost index into a -1
                if (!legalCombine) {
                    size_t chunk = 1;
                    if (auto flt = dt.isFloat()) {
                        if (flt->isFloatTy()) {
                            chunk = 4;
                        } else if(flt->isDoubleTy()) {
                            chunk = 8;
                        } else if(flt->isHalfTy()) {
                            chunk = 2;
                        } else {
                            llvm::errs() << *flt << "\n";
                            assert(0 && "unhandled float type");
                        }
                    } else if (dt.typeEnum == IntType::Pointer) {
                        chunk = dl.getPointerSizeInBits() / 8;
                    }

                    legalCombine = true;
                    for(size_t i=0; i<len; i+=chunk) {
                        if (!set.count(i)) {
                            legalCombine = false;
                            break;
                        }
                    }
                }

                std::vector<int> next;
                next.push_back(-1);
                for(auto v : pnext) next.push_back(v);

                if (legalCombine) {
                    dat.insert(next, dt, /*intsAreLegalPointerSub*/true);
                } else {
                    for(auto e : set) {
                        next[0] = e;
                        dat.insert(next, dt);
                    }
                }

            }
        }

        return dat;
    }

    ValueData KeepMinusOne() const {
        ValueData dat;

        for(const auto &pair : mapping) {

            assert(pair.first.size() != 0);

            // Pointer is at offset 0 from this object
            if (pair.first[0] != 0 && pair.first[0] != -1) continue;

            if (pair.first.size() == 1) {
                if (pair.second == IntType::Pointer || pair.second == IntType::Anything) {
                    dat.insert(pair.first, pair.second);
                    continue;
                }
                llvm::errs() << "could not merge test  " << str() << "\n";
            }

            if (pair.first[1] == -1) {
                dat.insert(pair.first, pair.second);
            }
        }

        return dat;
    }

    //! Replace offsets in [offset, offset+maxSize] with [addOffset, addOffset + maxSize]
    ValueData ShiftIndices(const llvm::DataLayout& dl, int offset, int maxSize, size_t addOffset=0) const {
        ValueData dat;

        for(const auto &pair : mapping) {
            if (pair.first.size() == 0) {
                if (pair.second == IntType::Pointer || pair.second == IntType::Anything) {
                    dat.insert(pair.first, pair.second);
                    continue;
                }

                llvm::errs() << "could not unmerge " << str() << "\n";
            }
            assert(pair.first.size() > 0);

            std::vector<int> next(pair.first);

            if (next[0] == -1) {
                if (maxSize == -1) {
                    // Max size does not clip the next index

                    // If we have a follow up offset add, we lose the -1 since we only represent [0, inf) with -1 not the [addOffset, inf) required here
                    if (addOffset != 0) {
                        next[0] = addOffset;
                    }

                } else {
                    // This needs to become 0...maxSize as seen below
                }
            } else {
                // Too small for range
                if (next[0] < offset) {
                    continue;
                }
                next[0] -= offset;

                if (maxSize != -1) {
                    if (next[0] >= maxSize) continue;
                }

                next[0] += addOffset;
            }

            ValueData dat2;
            //llvm::errs() << "next: " << to_string(next) << " indices: " << to_string(indices) << " pair.first: " << to_string(pair.first) << "\n";
            if (next[0] == -1 && maxSize != -1) {
                size_t chunk = 1;
                auto op = operator[]({ pair.first[0] });
                if (auto flt = op.isFloat()) {
                    if (flt->isFloatTy()) {
                        chunk = 4;
                    } else if(flt->isDoubleTy()) {
                        chunk = 8;
                    } else if(flt->isHalfTy()) {
                        chunk = 2;
                    } else {
                        llvm::errs() << *flt << "\n";
                        assert(0 && "unhandled float type");
                    }
                } else if (op.typeEnum == IntType::Pointer) {
                    chunk = dl.getPointerSizeInBits() / 8;
                }

                for(int i=0; i<maxSize; i+= chunk) {
                    next[0] = i + addOffset;
                    dat2.insert(next, pair.second);
                }
            } else {
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

    // TODO note that this keeps -1's
    ValueData AtMost(size_t max) const {
        assert(max > 0);
        ValueData dat;
        for(const auto &pair : mapping) {
            if (pair.first.size() == 0 || pair.first[0] == -1 || (size_t)pair.first[0] < max) {
                dat.insert(pair.first, pair.second);
            }
        }
        return dat;
    }

    static ValueData Argument(DataType type, llvm::Value* v) {
        if (v->getType()->isIntOrIntVectorTy()) return ValueData(type);
        return ValueData(type).Only(0);
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

    bool mergeIn(const ValueData &v, bool pointerIntSame) {
        //! Todo detect recursive merge

        bool changed = false;

        if (v[{-1}] != IntType::Unknown) {
            for(auto &pair : mapping) {
                if (pair.first.size() == 1 && pair.first[0] != -1) {
                    pair.second.mergeIn(v[{-1}], pointerIntSame);
                    //if (pair.second == ) // NOTE DELETE the non -1
                }
            }
        }

        for(auto &pair : v.mapping) {
            assert(pair.second != IntType::Unknown);
            DataType dt = operator[](pair.first);
            //llvm::errs() << "merging @ " << to_string(pair.first) << " old:" << dt.str() << " new:" << pair.second.str() << "\n";
            changed |= (dt.mergeIn(pair.second, pointerIntSame));

            /*
            if (dt == IntType::Integer && pair.first.size() > 0 && pair.first.back() != -1) {
                auto p2(pair.first);
                for(unsigned i=max((int)pair.first.back()-4, 0); i<(unsigned)pair.first.back(); i++) {
                    p2[p2.size()-1] == i;
                    if (operator[](p2).typeEnum == IntType::Float) {
                        llvm::errs() << " illegal merge of " << v.str() << " into " << str() << "\n";
                        assert(0 && "badmerge");
                        exit(1);
                    }
                }
            }

            if (dt == IntType::Float && pair.first.size() > 0 && pair.first.back() != -1) {
                auto p2(pair.first);
                for(unsigned i=pair.first.back(); i<(unsigned)pair.first.back()+4; i++) {
                    p2[p2.size()-1] == i;
                    if (operator[](p2).typeEnum == IntType::Integer) {
                        llvm::errs() << " illegal merge of " << v.str() << " into " << str() << "\n";
                        assert(0 && "badmerg2");
                        exit(1);
                    }
                }
            }
            */


            insert(pair.first, dt);
        }
        return changed;
    }

    bool operator|=(const ValueData &v) {
        return mergeIn(v, /*pointerIntSame*/false);
    }

    bool operator&=(const ValueData &v) {
        return andIn(v, /*assertIfIllegal*/true);
    }

    bool andIn(const ValueData &v, bool assertIfIllegal=true) {
        bool changed = false;

        std::vector<std::vector<int>> keystodelete;
        for(auto &pair : mapping) {
            DataType other = IntType::Unknown;
            auto fd = v.mapping.find(pair.first);
            if (fd != v.mapping.end()) {
                other = fd->second;
            }
            changed = (pair.second.andIn(other, assertIfIllegal));
            if (pair.second == IntType::Unknown) {
                keystodelete.push_back(pair.first);
            }
        }

        for(auto &key : keystodelete) {
            mapping.erase(key);
        }

        return changed;
    }


    bool pointerIntMerge(const ValueData &v, llvm::BinaryOperator::BinaryOps op) {
        bool changed = false;

        auto found = mapping.find({});
        if (found != mapping.end()) {
            changed |= ( found->second.pointerIntMerge(v[{}], op) );
            if (found->second == IntType::Unknown) {
                mapping.erase(std::vector<int>({}));
            }
        } else if (v.mapping.find({}) != v.mapping.end()) {
            DataType dt(IntType::Unknown);
            dt.pointerIntMerge(v[{}], op);
            if (dt != IntType::Unknown) {
                changed = true;
                mapping.emplace(std::vector<int>({}), dt);
            }
        }

        std::vector<std::vector<int>> keystodelete;

        for(auto &pair : mapping) {
            if (pair.first != std::vector<int>({})) keystodelete.push_back(pair.first);
        }

        for(auto &key : keystodelete) {
            mapping.erase(key);
            changed = true;
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

//First is ; then ; then
class NewFnTypeInfo {
public:
    llvm::Function* function;
    NewFnTypeInfo(llvm::Function* fn) : function(fn) {}
    NewFnTypeInfo(const NewFnTypeInfo&) = default;
    NewFnTypeInfo& operator=(NewFnTypeInfo&) = default;
    NewFnTypeInfo& operator=(NewFnTypeInfo&&) = default;

    // arguments:type
    std::map<llvm::Argument*, ValueData> first;
    // return type
    ValueData second;
    // the specific constant of an argument, if it is constant
    std::map<llvm::Argument*, std::set<int64_t>> knownValues;

    std::set<int64_t> isConstantInt(llvm::Value* val, const llvm::DominatorTree& DT, std::map<llvm::Value*, std::set<int64_t>>& intseen) const;
};

static inline bool operator<(const NewFnTypeInfo& lhs, const NewFnTypeInfo& rhs) {

    if (lhs.function < rhs.function) return true;
    if (rhs.function < lhs.function) return false;

    if (lhs.first < rhs.first) return true;
    if (rhs.first < lhs.first) return false;
    if (lhs.second < rhs.second) return true;
    if (rhs.second < lhs.second) return false;
    return lhs.knownValues < rhs.knownValues;
}

class TypeAnalyzer;
class TypeAnalysis;

class TypeResults {
public:
	TypeAnalysis &analysis;
	const NewFnTypeInfo info;
public:
	TypeResults(TypeAnalysis &analysis, const NewFnTypeInfo& fn);
	DataType intType(llvm::Value* val, bool errIfNotFound=true);

    //! Returns whether in the first num bytes there is pointer, int, float, or none
    //! If pointerIntSame is set to true, then consider either as the same (and thus mergable)
    DataType firstPointer(size_t num, llvm::Value* val, bool errIfNotFound=true, bool pointerIntSame=false);

    ValueData query(llvm::Value* val);
    NewFnTypeInfo getAnalyzedTypeInfo();
    ValueData getReturnAnalysis();
    void dump();
    std::set<int64_t> isConstantInt(llvm::Value* val) const;
};


class TypeAnalyzer : public llvm::InstVisitor<TypeAnalyzer> {
public:
    //List of value's which should be re-analyzed now with new information
    std::deque<llvm::Value*> workList;
private:
    void addToWorkList(llvm::Value* val);
    std::map<llvm::Value*, std::set<int64_t>> intseen;
public:
    //Calling context
    const NewFnTypeInfo fntypeinfo;

	TypeAnalysis &interprocedural;

	std::map<llvm::Value*, ValueData> analysis;

    llvm::DominatorTree DT;

    TypeAnalyzer(const NewFnTypeInfo& fn, TypeAnalysis& TA);

    ValueData getAnalysis(llvm::Value* val);

    void updateAnalysis(llvm::Value* val, IntType data, llvm::Value* origin);
    void updateAnalysis(llvm::Value* val, DataType data, llvm::Value* origin);
    void updateAnalysis(llvm::Value* val, ValueData data, llvm::Value* origin);

    void prepareArgs();

    void considerTBAA();

	void run();

    bool runUnusedChecks();

    void visitValue(llvm::Value& val);

    void visitCmpInst(llvm::CmpInst &I);

    void visitAllocaInst(llvm::AllocaInst &I);

    void visitLoadInst(llvm::LoadInst &I);

	void visitStoreInst(llvm::StoreInst &I);

    void visitGetElementPtrInst(llvm::GetElementPtrInst &gep);

    void visitPHINode(llvm::PHINode& phi);

	void visitTruncInst(llvm::TruncInst &I);

	void visitZExtInst(llvm::ZExtInst &I);

	void visitSExtInst(llvm::SExtInst &I);

	void visitAddrSpaceCastInst(llvm::AddrSpaceCastInst &I);

    void visitFPTruncInst(llvm::FPTruncInst &I);

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

    void visitMemTransferInst(llvm::MemTransferInst& MTI);

    void visitIntrinsicInst(llvm::IntrinsicInst &II);

    ValueData getReturnAnalysis();

    void dump();

    std::set<int64_t> isConstantInt(llvm::Value* val);
};


class TypeAnalysis {
public:
    std::map<NewFnTypeInfo, TypeAnalyzer > analyzedFunctions;

    TypeResults analyzeFunction(const NewFnTypeInfo& fn);

	ValueData query(llvm::Value* val, const NewFnTypeInfo& fn);

    DataType intType(llvm::Value* val, const NewFnTypeInfo& fn, bool errIfNotFound=true);
    DataType firstPointer(size_t num, llvm::Value* val, const NewFnTypeInfo& fn, bool errIfNotFound=true, bool pointerIntSame=false);

    inline ValueData getReturnAnalysis(const NewFnTypeInfo &fn) {
        analyzeFunction(fn);
        return analyzedFunctions.find(fn)->second.getReturnAnalysis();
    }
};

#endif
