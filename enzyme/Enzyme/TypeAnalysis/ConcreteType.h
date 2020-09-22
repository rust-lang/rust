/*
 * Concrete.h
 *
 * Copyright (C) 2020 William S. Moses (enzyme@wsmoses.com) - All Rights
 * Reserved
 *
 * For commercial use of this code please contact the author(s) above.
 */
#ifndef ENZYME_TYPE_ANALYSIS_CONCRETE_TYPE_H
#define ENZYME_TYPE_ANALYSIS_CONCRETE_TYPE_H 1

#include <string>

#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/ErrorHandling.h"

#include "BaseType.h"

class ConcreteType {
public:
  llvm::Type *type;
  BaseType typeEnum;

  ConcreteType(const ConcreteType&) = default;
  ConcreteType(ConcreteType&&) = default;
  ConcreteType(llvm::Type *type) : type(type), typeEnum(BaseType::Float) {
    assert(type != nullptr);
    assert(!llvm::isa<llvm::VectorType>(type));
    if (!type->isFloatingPointTy()) {
      llvm::errs() << " passing in non FP type: " << *type << "\n";
    }
    assert(type->isFloatingPointTy());
  }

  ConcreteType(BaseType typeEnum) : type(nullptr), typeEnum(typeEnum) {
    assert(typeEnum != BaseType::Float);
  }

  ConcreteType(std::string str, llvm::LLVMContext &C) {
    auto fd = str.find('@');
    if (fd != std::string::npos) {
      typeEnum = BaseType::Float;
      assert(str.substr(0, fd) == "Float");
      auto subt = str.substr(fd + 1);
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
      typeEnum = parseBaseType(str);
    }
  }

  bool isIntegral() const {
    return typeEnum == BaseType::Integer || typeEnum == BaseType::Anything;
  }

  bool isKnown() const { return typeEnum != BaseType::Unknown; }

  bool isPossiblePointer() const {
    return !isKnown() || typeEnum == BaseType::Pointer;
  }

  bool isPossibleFloat() const {
    return !isKnown() || typeEnum == BaseType::Float;
  }

  llvm::Type *isFloat() const { return type; }

  bool operator==(const BaseType dt) const { return typeEnum == dt; }

  bool operator!=(const BaseType dt) const { return typeEnum != dt; }

  bool operator==(const ConcreteType dt) const {
    return type == dt.type && typeEnum == dt.typeEnum;
  }
  bool operator!=(const ConcreteType dt) const { return !(*this == dt); }


  bool operator=(const BaseType bt) {
    assert(bt != BaseType::Float);
    bool changed = false;
    if (typeEnum != bt)
      changed = true;
    typeEnum = bt;
    if (type != nullptr)
      changed = true;
    type = nullptr;
    return changed;
  }

  // returns whether changed
  bool operator=(const ConcreteType dt) {
    bool changed = false;
    if (typeEnum != dt.typeEnum)
      changed = true;
    typeEnum = dt.typeEnum;
    if (type != dt.type)
      changed = true;
    type = dt.type;
    return changed;
  }
  bool operator=(ConcreteType&& dt) {
    bool changed = false;
    if (typeEnum != dt.typeEnum)
      changed = true;
    typeEnum = dt.typeEnum;
    if (type != dt.type)
      changed = true;
    type = dt.type;
    return changed;
  }

  // returns whether changed
  bool legalMergeIn(const ConcreteType dt, bool pointerIntSame, bool &legal) {
    if (typeEnum == BaseType::Anything) {
      return false;
    }
    if (dt.typeEnum == BaseType::Anything) {
      return *this = dt;
    }
    if (typeEnum == BaseType::Unknown) {
      return *this = dt;
    }
    if (dt.typeEnum == BaseType::Unknown) {
      return false;
    }
    if (dt.typeEnum != typeEnum) {
      if (pointerIntSame) {
        if ((typeEnum == BaseType::Pointer && dt.typeEnum == BaseType::Integer) ||
            (typeEnum == BaseType::Integer && dt.typeEnum == BaseType::Pointer)) {
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

  // returns whether changed
  bool mergeIn(const ConcreteType dt, bool pointerIntSame) {
    bool legal = true;
    bool res = legalMergeIn(dt, pointerIntSame, legal);
    if (!legal) {
      llvm::errs() << "me: " << str() << " right: " << dt.str() << "\n";
    }
    assert(legal);
    return res;
  }

  // returns whether changed
  bool operator|=(const ConcreteType dt) {
    return mergeIn(dt, /*pointerIntSame*/ false);
  }

  bool pointerIntMerge(const ConcreteType dt, llvm::BinaryOperator::BinaryOps op) {
    bool changed = false;
    using namespace llvm;

    if (typeEnum == BaseType::Anything && dt.typeEnum == BaseType::Anything) {
      return changed;
    }

    if (op == BinaryOperator::And &&
        (((typeEnum == BaseType::Anything || typeEnum == BaseType::Integer) &&
          dt.isFloat()) ||
         (isFloat() && (dt.typeEnum == BaseType::Anything ||
                        dt.typeEnum == BaseType::Integer)))) {
      typeEnum = BaseType::Unknown;
      type = nullptr;
      changed = true;
      return changed;
    }

    if ((typeEnum == BaseType::Unknown && dt.typeEnum == BaseType::Anything) ||
        (typeEnum == BaseType::Anything && dt.typeEnum == BaseType::Unknown)) {
      if (typeEnum != BaseType::Unknown) {
        typeEnum = BaseType::Unknown;
        changed = true;
      }
      return changed;
    }

    if ((typeEnum == BaseType::Integer && dt.typeEnum == BaseType::Integer) ||
        (typeEnum == BaseType::Unknown && dt.typeEnum == BaseType::Integer) ||
        (typeEnum == BaseType::Integer && dt.typeEnum == BaseType::Unknown) ||
        (typeEnum == BaseType::Anything && dt.typeEnum == BaseType::Integer) ||
        (typeEnum == BaseType::Integer && dt.typeEnum == BaseType::Anything)) {
      switch (op) {
      case BinaryOperator::Add:
      case BinaryOperator::Sub:
        // if one of these is unknown we cannot deduce the result
        // e.g. pointer + int = pointer and int + int = int
        if (typeEnum == BaseType::Unknown || dt.typeEnum == BaseType::Unknown) {
          if (typeEnum != BaseType::Unknown) {
            typeEnum = BaseType::Unknown;
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
        if (typeEnum == BaseType::Anything) {
          break;
        }
        if (typeEnum != BaseType::Integer) {
          typeEnum = BaseType::Integer;
          changed = true;
        }
        break;
      default:
        llvm_unreachable("unknown binary operator");
      }
      return changed;
    }

    if (typeEnum == BaseType::Pointer && dt.typeEnum == BaseType::Pointer) {
      switch (op) {
      case BinaryOperator::Sub:
        typeEnum = BaseType::Integer;
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

    if ((typeEnum == BaseType::Integer && dt.typeEnum == BaseType::Pointer) ||
        (typeEnum == BaseType::Pointer && dt.typeEnum == BaseType::Integer) ||
        (typeEnum == BaseType::Integer && dt.typeEnum == BaseType::Pointer) ||
        (typeEnum == BaseType::Pointer && dt.typeEnum == BaseType::Unknown) ||
        (typeEnum == BaseType::Unknown && dt.typeEnum == BaseType::Pointer) ||
        (typeEnum == BaseType::Pointer && dt.typeEnum == BaseType::Anything) ||
        (typeEnum == BaseType::Anything && dt.typeEnum == BaseType::Pointer)) {

      switch (op) {
      case BinaryOperator::Sub:
        if (typeEnum == BaseType::Anything || dt.typeEnum == BaseType::Anything) {
          if (typeEnum != BaseType::Unknown) {
            typeEnum = BaseType::Unknown;
            changed = true;
          }
          break;
        }
      case BinaryOperator::Add:
      case BinaryOperator::Mul:
        if (typeEnum != BaseType::Pointer) {
          typeEnum = BaseType::Pointer;
          changed = true;
        }
        break;
      case BinaryOperator::UDiv:
      case BinaryOperator::SDiv:
      case BinaryOperator::URem:
      case BinaryOperator::SRem:
        if (dt.typeEnum == BaseType::Pointer) {
          llvm_unreachable("cannot divide integer by pointer");
        } else if (typeEnum != BaseType::Unknown) {
          typeEnum = BaseType::Unknown;
          changed = true;
        }
        break;
      case BinaryOperator::And:
      case BinaryOperator::Or:
      case BinaryOperator::Xor:
      case BinaryOperator::Shl:
      case BinaryOperator::AShr:
      case BinaryOperator::LShr:
        if (typeEnum != BaseType::Unknown) {
          typeEnum = BaseType::Unknown;
          changed = true;
        }
        break;
      default:
        llvm_unreachable("unknown binary operator");
      }
      return changed;
    }

    if (dt.typeEnum == BaseType::Integer) {
      switch (op) {
      case BinaryOperator::Shl:
      case BinaryOperator::AShr:
      case BinaryOperator::LShr:
        if (typeEnum != BaseType::Unknown) {
          typeEnum = BaseType::Unknown;
          changed = true;
          return changed;
        }
        break;
      default:
        break;
      }
    }

    llvm::errs() << "self: " << str() << " other: " << dt.str() << " op: " << op
                 << "\n";
    llvm_unreachable("unknown case");
  }

  bool andIn(const ConcreteType dt, bool assertIfIllegal = true) {
    if (typeEnum == BaseType::Anything) {
      return *this = dt;
    }
    if (dt.typeEnum == BaseType::Anything) {
      return false;
    }
    if (typeEnum == BaseType::Unknown) {
      return false;
    }
    if (dt.typeEnum == BaseType::Unknown) {
      return *this = dt;
    }

    if (dt.typeEnum != typeEnum) {
      if (!assertIfIllegal) {
        return *this = BaseType::Unknown;
      }
      llvm::errs() << "&= typeEnum: " << to_string(typeEnum)
                   << " dt.typeEnum.str(): " << to_string(dt.typeEnum) << "\n";
      return *this = BaseType::Unknown;
    }
    assert(dt.typeEnum == typeEnum);
    if (dt.type != type) {
      if (!assertIfIllegal) {
        return *this = BaseType::Unknown;
      }
      llvm::errs() << "type: " << *type << " dt.type: " << *dt.type << "\n";
    }
    assert(dt.type == type);
    return false;
  }

  // returns whether changed
  bool operator&=(const ConcreteType dt) {
    return andIn(dt, /*assertIfIllegal*/ true);
  }

  bool operator<(const ConcreteType dt) const {
    if (typeEnum == dt.typeEnum) {
      return type < dt.type;
    } else {
      return typeEnum < dt.typeEnum;
    }
  }
  std::string str() const {
    std::string res = to_string(typeEnum);
    if (typeEnum == BaseType::Float) {
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

static inline std::string to_string(const ConcreteType dt) { return dt.str(); }

#endif
