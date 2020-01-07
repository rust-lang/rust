/*
 * ActiveVariable.h - Active Varaible Detection Utilities
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
#ifndef ENZYME_ACTIVE_VAR_H
#define ENZYME_ACTIVE_VAR_H 1

#include <cstdint>

#include <llvm/Config/llvm-config.h>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Analysis/AliasAnalysis.h"

#include "llvm/Support/CommandLine.h"

extern llvm::cl::opt<bool> printconst;
extern llvm::cl::opt<bool> nonmarkedglobals_inactive;

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
    if (str == "Anything") return IntType::Everything;
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

    bool operator==(const DataType dt) const {
        return type == dt.type && typeEnum == dt.typeEnum;
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
        assert(dt.typeEnum == typeEnum);
        assert(dt.type == type);
        return false;
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
        assert(dt.typeEnum == typeEnum);
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
};
    
static inline std::string to_string(const DataType dt) {
    std::string res = to_string(dt.typeEnum);
    if (dt.typeEnum == IntType::Float) {
        if (dt.type->isHalfTy()) {
            res += "@half";
        } else if (dt.type->isFloatTy()) {
            res += "@float";
        } else if (dt.type->isDoubleTy()) {
            res += "@double";
        } else if (dt.type->isX86_FP80Ty()) {
            res += "@fp80";
        } else if (dt.type->isFP128Ty()) {
            res += "@fp128";
        } else if (dt.type->isPPC_FP128Ty()) {
            res += "@ppc128";
        } else {
            llvm_unreachable("unknown data type");
        }
    }
    return res;
}

llvm::Type* isKnownFloatTBAA(llvm::Instruction* inst);

DataType isIntASecretFloat(const std::map<llvm::Argument*, DataType> typeInfo, llvm::Value* val, IntType defaultType=IntType::Unknown, bool errIfNotFound=false);

//! return the secret float type of val if found, otherwise nullptr
//!   if onlyFirst is set, consider only the first element of the pointer val (e.g. if we have {int, double, double}, consider only the int)
//    onlyFirst should typically be used for store operands, whereas not used for memcpy
DataType isIntPointerASecretFloat(const std::map<llvm::Argument*, DataType> typeInfo, llvm::Value* val, bool onlyFirst, bool errIfNotFound=true);

bool isconstantValueM(llvm::Value* val, llvm::SmallPtrSetImpl<llvm::Value*> &constants, llvm::SmallPtrSetImpl<llvm::Value*> &nonconstant, llvm::SmallPtrSetImpl<llvm::Value*> &constantvals, llvm::SmallPtrSetImpl<llvm::Value*> &retvals, llvm::AAResults &AA, uint8_t directions=3);

// TODO separate if the instruction is constant (i.e. could change things)
//    from if the value is constant (the value is something that could be differentiated)
bool isconstantM(llvm::Instruction* inst, llvm::SmallPtrSetImpl<llvm::Value*> &constants, llvm::SmallPtrSetImpl<llvm::Value*> &nonconstant, llvm::SmallPtrSetImpl<llvm::Value*> &constantvals, llvm::SmallPtrSetImpl<llvm::Value*> &retvals, llvm::AAResults &AA, uint8_t directions=3);

#endif
