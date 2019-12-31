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

#include "llvm/Support/CommandLine.h"

extern llvm::cl::opt<bool> printconst;

enum class IntType {
    Integer,
    Float,
    Pointer,
    Unknown
};

static inline std::string to_string(IntType t) {
    switch(t) {
        case IntType::Integer: return "Integer";
        case IntType::Float: return "Float";
        case IntType::Pointer: return "Pointer";
        case IntType::Unknown: return "Unknown";
    }
    llvm_unreachable("unknown inttype");
}

static inline IntType parseIntType(std::string str) {
    if (str == "Integer") return IntType::Integer;
    if (str == "Float") return IntType::Float;
    if (str == "Pointer") return IntType::Pointer;
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

    llvm::Type* isFloat() const {
        return type;
    }
    bool operator==(const DataType dt) const {
        return type == dt.type && typeEnum == dt.typeEnum;
    }
    bool operator<(const DataType dt) const {
        if (typeEnum == dt.typeEnum) {
            return type < dt.type;
        } else {
            return typeEnum < dt.typeEnum;
        }
    }
};
    
static inline std::string str(DataType&& dt) {
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

IntType isIntASecretFloat(const std::map<llvm::Argument*, DataType> typeInfo, llvm::Value* val, IntType defaultType=IntType::Unknown);

//! return the secret float type of val if found, otherwise nullptr
//!   if onlyFirst is set, consider only the first element of the pointer val (e.g. if we have {int, double, double}, consider only the int)
//    onlyFirst should typically be used for store operands, whereas not used for memcpy
DataType isIntPointerASecretFloat(const std::map<llvm::Argument*, DataType> typeInfo, llvm::Value* val, bool onlyFirst, bool errIfNotFound=true);

bool isconstantValueM(llvm::Value* val, llvm::SmallPtrSetImpl<llvm::Value*> &constants, llvm::SmallPtrSetImpl<llvm::Value*> &nonconstant, llvm::SmallPtrSetImpl<llvm::Value*> &constantvals, llvm::SmallPtrSetImpl<llvm::Value*> &retvals, const llvm::SmallPtrSetImpl<llvm::Instruction*> &originalInstructions, uint8_t directions=3);

// TODO separate if the instruction is constant (i.e. could change things)
//    from if the value is constant (the value is something that could be differentiated)
bool isconstantM(llvm::Instruction* inst, llvm::SmallPtrSetImpl<llvm::Value*> &constants, llvm::SmallPtrSetImpl<llvm::Value*> &nonconstant, llvm::SmallPtrSetImpl<llvm::Value*> &constantvals, llvm::SmallPtrSetImpl<llvm::Value*> &retvals, const llvm::SmallPtrSetImpl<llvm::Instruction*> &originalInstructions, uint8_t directions=3);

#endif
