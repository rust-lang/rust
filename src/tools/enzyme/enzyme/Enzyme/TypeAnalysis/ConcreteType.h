//===- ConcreteType.h - Underlying SubType used in Type Analysis
//------------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of an a class representing all
// potential end SubTypes used in Type Analysis. This ``ConcreteType`` contains
// an the SubType category ``BaseType`` as well as the SubType of float, if
// relevant. This also contains several helper utility functions.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_TYPE_ANALYSIS_CONCRETE_TYPE_H
#define ENZYME_TYPE_ANALYSIS_CONCRETE_TYPE_H 1

#include <string>

#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/ErrorHandling.h"

#include "BaseType.h"

/// Concrete SubType of a given value. Consists of a category `BaseType` and the
/// particular floating point value, if relevant.
class ConcreteType {
public:
  /// Category of underlying type
  BaseType SubTypeEnum;
  /// Floating point type, if relevant, otherwise nullptr
  llvm::Type *SubType;

  /// Construct a ConcreteType from an existing FloatingPoint Type
  ConcreteType(llvm::Type *SubType)
      : SubTypeEnum(BaseType::Float), SubType(SubType) {
    assert(SubType != nullptr);
    assert(!llvm::isa<llvm::VectorType>(SubType));
    if (!SubType->isFloatingPointTy()) {
      llvm::errs() << " passing in non FP SubType: " << *SubType << "\n";
    }
    assert(SubType->isFloatingPointTy());
  }

  /// Construct a non-floating Concrete type from a BaseType
  ConcreteType(BaseType SubTypeEnum)
      : SubTypeEnum(SubTypeEnum), SubType(nullptr) {
    assert(SubTypeEnum != BaseType::Float);
  }

  /// Construct a ConcreteType from a string
  ///  A Concrete Type's string representation is given by the string of the
  ///  enum If it is a floating point it is given by Float@<specific_type>
  ConcreteType(std::string Str, llvm::LLVMContext &C) {
    auto Sep = Str.find('@');
    if (Sep != std::string::npos) {
      SubTypeEnum = BaseType::Float;
      assert(Str.substr(0, Sep) == "Float");
      auto SubName = Str.substr(Sep + 1);
      if (SubName == "half") {
        SubType = llvm::Type::getHalfTy(C);
      } else if (SubName == "float") {
        SubType = llvm::Type::getFloatTy(C);
      } else if (SubName == "double") {
        SubType = llvm::Type::getDoubleTy(C);
      } else if (SubName == "fp80") {
        SubType = llvm::Type::getX86_FP80Ty(C);
      } else if (SubName == "fp128") {
        SubType = llvm::Type::getFP128Ty(C);
      } else if (SubName == "ppc128") {
        SubType = llvm::Type::getPPC_FP128Ty(C);
      } else {
        llvm_unreachable("unknown data SubType");
      }
    } else {
      SubType = nullptr;
      SubTypeEnum = parseBaseType(Str);
    }
  }

  /// Convert the ConcreteType to a string
  std::string str() const {
    std::string Result = to_string(SubTypeEnum);
    if (SubTypeEnum == BaseType::Float) {
      if (SubType->isHalfTy()) {
        Result += "@half";
      } else if (SubType->isFloatTy()) {
        Result += "@float";
      } else if (SubType->isDoubleTy()) {
        Result += "@double";
      } else if (SubType->isX86_FP80Ty()) {
        Result += "@fp80";
      } else if (SubType->isFP128Ty()) {
        Result += "@fp128";
      } else if (SubType->isPPC_FP128Ty()) {
        Result += "@ppc128";
      } else {
        llvm_unreachable("unknown data SubType");
      }
    }
    return Result;
  }

  /// Whether this ConcreteType has information (is not unknown)
  bool isKnown() const { return SubTypeEnum != BaseType::Unknown; }

  /// Whether this ConcreteType must an integer
  bool isIntegral() const { return SubTypeEnum == BaseType::Integer; }

  /// Whether this ConcreteType could be a pointer (SubTypeEnum is unknown or a
  /// pointer)
  bool isPossiblePointer() const {
    return SubTypeEnum == BaseType::Pointer ||
           SubTypeEnum == BaseType::Anything ||
           SubTypeEnum == BaseType::Unknown;
  }

  /// Whether this ConcreteType could be a float (SubTypeEnum is unknown or a
  /// float)
  bool isPossibleFloat() const {
    return SubTypeEnum == BaseType::Float ||
           SubTypeEnum == BaseType::Anything ||
           SubTypeEnum == BaseType::Unknown;
  }

  /// Return the floating point type, if this is a float
  llvm::Type *isFloat() const { return SubType; }

  /// Return if this is known to be the BaseType BT
  /// This cannot be called with BaseType::Float as it lacks information
  bool operator==(const BaseType BT) const {
    if (BT == BaseType::Float) {
      assert(0 &&
             "Cannot do comparision between ConcreteType and BaseType::Float");
      llvm_unreachable(
          "Cannot do comparision between ConcreteType and BaseType::Float");
    }
    return SubTypeEnum == BT;
  }

  /// Return if this is known not to be the BaseType BT
  /// This cannot be called with BaseType::Float as it lacks information
  bool operator!=(const BaseType BT) const {
    if (BT == BaseType::Float) {
      assert(0 &&
             "Cannot do comparision between ConcreteType and BaseType::Float");
      llvm_unreachable(
          "Cannot do comparision between ConcreteType and BaseType::Float");
    }
    return SubTypeEnum != BT;
  }

  /// Return if this is known to be the ConcreteType CT
  bool operator==(const ConcreteType CT) const {
    return SubType == CT.SubType && SubTypeEnum == CT.SubTypeEnum;
  }

  /// Return if this is known not to be the ConcreteType CT
  bool operator!=(const ConcreteType CT) const { return !(*this == CT); }

  /// Set this to the given ConcreteType, returning true if
  /// this ConcreteType has changed
  bool operator=(const ConcreteType CT) {
    bool changed = false;
    if (SubTypeEnum != CT.SubTypeEnum)
      changed = true;
    SubTypeEnum = CT.SubTypeEnum;
    if (SubType != CT.SubType)
      changed = true;
    SubType = CT.SubType;
    return changed;
  }

  /// Set this to the given BaseType, returning true if
  /// this ConcreteType has changed
  bool operator=(const BaseType BT) {
    assert(BT != BaseType::Float);
    return ConcreteType::operator=(ConcreteType(BT));
  }

  /// Set this to the logical or of itself and CT, returning whether this value
  /// changed Setting `PointerIntSame` considers pointers and integers as
  /// equivalent If this is an illegal operation, `LegalOr` will be set to false
  bool checkedOrIn(const ConcreteType CT, bool PointerIntSame, bool &LegalOr) {
    LegalOr = true;
    if (SubTypeEnum == BaseType::Anything) {
      return false;
    }
    if (CT.SubTypeEnum == BaseType::Anything) {
      return *this = CT;
    }
    if (SubTypeEnum == BaseType::Unknown) {
      return *this = CT;
    }
    if (CT.SubTypeEnum == BaseType::Unknown) {
      return false;
    }
    if (CT.SubTypeEnum != SubTypeEnum) {
      if (PointerIntSame) {
        if ((SubTypeEnum == BaseType::Pointer &&
             CT.SubTypeEnum == BaseType::Integer) ||
            (SubTypeEnum == BaseType::Integer &&
             CT.SubTypeEnum == BaseType::Pointer)) {
          return false;
        }
      }
      LegalOr = false;
      return false;
    }
    assert(CT.SubTypeEnum == SubTypeEnum);
    if (CT.SubType != SubType) {
      LegalOr = false;
      return false;
    }
    assert(CT.SubType == SubType);
    return false;
  }

  /// Set this to the logical or of itself and CT, returning whether this value
  /// changed Setting `PointerIntSame` considers pointers and integers as
  /// equivalent This function will error if doing an illegal Operation
  bool orIn(const ConcreteType CT, bool PointerIntSame) {
    bool Legal = true;
    bool Result = checkedOrIn(CT, PointerIntSame, Legal);
    if (!Legal) {
      llvm::errs() << "Illegal orIn: " << str() << " right: " << CT.str()
                   << " PointerIntSame=" << PointerIntSame << "\n";
      assert(0 && "Performed illegal ConcreteType::orIn");
      llvm_unreachable("Performed illegal ConcreteType::orIn");
    }
    return Result;
  }

  /// Set this to the logical or of itself and CT, returning whether this value
  /// changed This assumes that pointers and integers are distinct This function
  /// will error if doing an illegal Operation
  bool operator|=(const ConcreteType CT) {
    return orIn(CT, /*pointerIntSame*/ false);
  }

  /// Set this to the logical and of itself and CT, returning whether this value
  /// changed If this and CT are incompatible, the result will be
  /// BaseType::Unknown
  bool andIn(const ConcreteType CT) {
    if (SubTypeEnum == BaseType::Anything) {
      return *this = CT;
    }
    if (CT.SubTypeEnum == BaseType::Anything) {
      return false;
    }
    if (SubTypeEnum == BaseType::Unknown) {
      return false;
    }
    if (CT.SubTypeEnum == BaseType::Unknown) {
      return *this = CT;
    }

    if (CT.SubTypeEnum != SubTypeEnum) {
      return *this = BaseType::Unknown;
    }
    if (CT.SubType != SubType) {
      return *this = BaseType::Unknown;
    }
    return false;
  }

  /// Set this to the logical and of itself and CT, returning whether this value
  /// changed If this and CT are incompatible, the result will be
  /// BaseType::Unknown
  bool operator&=(const ConcreteType CT) { return andIn(CT); }

  /// Keep only mappings where the type is not an `Anything`
  ConcreteType PurgeAnything() const {
    if (SubTypeEnum == BaseType::Anything)
      return BaseType::Unknown;
    return *this;
  }

  /// Set this to the logical `binop` of itself and RHS, using the Binop Op,
  /// returning true if this was changed.
  /// This function will error on an invalid type combination
  bool binopIn(const ConcreteType RHS, llvm::BinaryOperator::BinaryOps Op) {
    bool Changed = false;
    using namespace llvm;

    // Anything op Anything => Anything
    if (SubTypeEnum == BaseType::Anything &&
        RHS.SubTypeEnum == BaseType::Anything) {
      return Changed;
    }

    // [?] op float => Unknown
    if ((((SubTypeEnum == BaseType::Anything ||
           SubTypeEnum == BaseType::Integer ||
           SubTypeEnum == BaseType::Unknown) &&
          RHS.isFloat()) ||
         (isFloat() && (RHS.SubTypeEnum == BaseType::Anything ||
                        RHS.SubTypeEnum == BaseType::Integer ||
                        RHS.SubTypeEnum == BaseType::Unknown)))) {
      SubTypeEnum = BaseType::Unknown;
      SubType = nullptr;
      Changed = true;
      return Changed;
    }

    // Unknown op Anything => Unknown
    if ((SubTypeEnum == BaseType::Unknown &&
         RHS.SubTypeEnum == BaseType::Anything) ||
        (SubTypeEnum == BaseType::Anything &&
         RHS.SubTypeEnum == BaseType::Unknown)) {
      if (SubTypeEnum != BaseType::Unknown) {
        SubTypeEnum = BaseType::Unknown;
        Changed = true;
      }
      return Changed;
    }

    // Integer op Integer => Integer
    if (SubTypeEnum == BaseType::Integer &&
        RHS.SubTypeEnum == BaseType::Integer) {
      return Changed;
    }

    // Integer op Anything => {Anything, Integer}
    if ((SubTypeEnum == BaseType::Anything &&
         RHS.SubTypeEnum == BaseType::Integer) ||
        (SubTypeEnum == BaseType::Integer &&
         RHS.SubTypeEnum == BaseType::Anything)) {

      switch (Op) {
      // The result of these operands mix data between LHS/RHS
      // Therefore there is some "anything" data in the result
      case BinaryOperator::Add:
      case BinaryOperator::Sub:
      case BinaryOperator::Mul:
      case BinaryOperator::And:
      case BinaryOperator::Or:
      case BinaryOperator::Xor:
        if (SubTypeEnum != BaseType::Anything) {
          SubTypeEnum = BaseType::Anything;
          Changed = true;
        }
        break;

      // The result of these operands only use data from LHS
      case BinaryOperator::UDiv:
      case BinaryOperator::SDiv:
      case BinaryOperator::URem:
      case BinaryOperator::SRem:
      case BinaryOperator::Shl:
      case BinaryOperator::AShr:
      case BinaryOperator::LShr:
        // No change since we retain data from LHS
        break;
      default:
        llvm_unreachable("unknown binary operator");
      }
      return Changed;
    }

    // Integer op Unknown => Unknown
    // e.g. pointer + int = pointer and int + int = int
    if ((SubTypeEnum == BaseType::Unknown &&
         RHS.SubTypeEnum == BaseType::Integer) ||
        (SubTypeEnum == BaseType::Integer &&
         RHS.SubTypeEnum == BaseType::Unknown)) {
      if (SubTypeEnum != BaseType::Unknown) {
        SubTypeEnum = BaseType::Unknown;
        Changed = true;
      }
      return Changed;
    }

    // Pointer op Pointer => {Integer, Illegal}
    if (SubTypeEnum == BaseType::Pointer &&
        RHS.SubTypeEnum == BaseType::Pointer) {
      switch (Op) {
      case BinaryOperator::Sub:
        SubTypeEnum = BaseType::Integer;
        Changed = true;
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
      return Changed;
    }

    // Pointer - Unknown => Unknown
    //   This is because Pointer - Pointer => Integer
    //              and  Pointer - Integer => Pointer
    if (Op == BinaryOperator::Sub && SubTypeEnum == BaseType::Pointer &&
        RHS.SubTypeEnum == BaseType::Unknown) {
      SubTypeEnum = BaseType::Unknown;
      Changed = true;
      return Changed;
    }

    // Pointer op ? => {Pointer, Unknown}
    if ((SubTypeEnum == BaseType::Integer &&
         RHS.SubTypeEnum == BaseType::Pointer) ||
        (SubTypeEnum == BaseType::Pointer &&
         RHS.SubTypeEnum == BaseType::Integer) ||
        (SubTypeEnum == BaseType::Integer &&
         RHS.SubTypeEnum == BaseType::Pointer) ||
        (SubTypeEnum == BaseType::Pointer &&
         RHS.SubTypeEnum == BaseType::Unknown) ||
        (SubTypeEnum == BaseType::Unknown &&
         RHS.SubTypeEnum == BaseType::Pointer) ||
        (SubTypeEnum == BaseType::Pointer &&
         RHS.SubTypeEnum == BaseType::Anything) ||
        (SubTypeEnum == BaseType::Anything &&
         RHS.SubTypeEnum == BaseType::Pointer)) {

      switch (Op) {
      case BinaryOperator::Sub:
        if (SubTypeEnum == BaseType::Anything ||
            RHS.SubTypeEnum == BaseType::Anything) {
          if (SubTypeEnum != BaseType::Unknown) {
            SubTypeEnum = BaseType::Unknown;
            Changed = true;
          }
          break;
        }
        if (RHS.SubTypeEnum == BaseType::Pointer) {
          if (SubTypeEnum != BaseType::Unknown) {
            SubTypeEnum = BaseType::Unknown;
            Changed = true;
          }
          break;
        }
      case BinaryOperator::Add:
      case BinaryOperator::Mul:
        if (SubTypeEnum != BaseType::Pointer) {
          SubTypeEnum = BaseType::Pointer;
          Changed = true;
        }
        break;
      case BinaryOperator::UDiv:
      case BinaryOperator::SDiv:
      case BinaryOperator::URem:
      case BinaryOperator::SRem:
        if (RHS.SubTypeEnum == BaseType::Pointer) {
          llvm_unreachable("cannot divide integer by pointer");
        } else if (SubTypeEnum != BaseType::Unknown) {
          SubTypeEnum = BaseType::Unknown;
          Changed = true;
        }
        break;
      case BinaryOperator::And:
      case BinaryOperator::Or:
      case BinaryOperator::Xor:
      case BinaryOperator::Shl:
      case BinaryOperator::AShr:
      case BinaryOperator::LShr:
        if (SubTypeEnum != BaseType::Unknown) {
          SubTypeEnum = BaseType::Unknown;
          Changed = true;
        }
        break;
      default:
        llvm_unreachable("unknown binary operator");
      }
      return Changed;
    }

    llvm::errs() << "self: " << str() << " RHS: " << RHS.str() << " Op: " << Op
                 << "\n";
    llvm_unreachable("Unknown ConcreteType::binopIn");
  }

  /// Compare concrete types for use in map's
  bool operator<(const ConcreteType dt) const {
    if (SubTypeEnum == dt.SubTypeEnum) {
      return SubType < dt.SubType;
    } else {
      return SubTypeEnum < dt.SubTypeEnum;
    }
  }
};

// Convert ConcreteType to string
static inline std::string to_string(const ConcreteType dt) { return dt.str(); }

#endif
