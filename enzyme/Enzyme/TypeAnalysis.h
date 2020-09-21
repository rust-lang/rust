/*
 * TypeAnalysis.h - Type Analysis Detection Utilities
 *
 * Copyright (C) 2020 William S. Moses (enzyme@wsmoses.com) - All Rights
 * Reserved
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

enum class BaseType {
  // integral type
  Integer,
  // floating point
  Float,
  // pointer
  Pointer,
  // can be anything of users choosing [usually result of a constant]
  Anything,
  // insufficient information
  Unknown
};

static inline std::string to_string(BaseType t) {
  switch (t) {
  case BaseType::Integer:
    return "Integer";
  case BaseType::Float:
    return "Float";
  case BaseType::Pointer:
    return "Pointer";
  case BaseType::Anything:
    return "Anything";
  case BaseType::Unknown:
    return "Unknown";
  }
  llvm_unreachable("unknown inttype");
}

static inline BaseType parseBaseType(std::string str) {
  if (str == "Integer")
    return BaseType::Integer;
  if (str == "Float")
    return BaseType::Float;
  if (str == "Pointer")
    return BaseType::Pointer;
  if (str == "Anything")
    return BaseType::Anything;
  if (str == "Unknown")
    return BaseType::Unknown;
  llvm_unreachable("unknown inttype str");
}

class ConcreteType {
public:
  llvm::Type *type;
  BaseType typeEnum;

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

static inline std::string to_string(const std::vector<int> x) {
  std::string out = "[";
  for (unsigned i = 0; i < x.size(); i++) {
    if (i != 0)
      out += ",";
    out += std::to_string(x[i]);
  }
  out += "]";
  return out;
}

class TypeTree;

typedef std::shared_ptr<const TypeTree> TypeResult;
typedef std::map<const std::vector<int>, ConcreteType> ConcreteTypeMapType;
typedef std::map<const std::vector<int>, const TypeResult> TypeTreeMapType;

class TypeTree : public std::enable_shared_from_this<TypeTree> {
private:
  // mapping of known indices to type if one exists
  ConcreteTypeMapType mapping;

  // mapping of known indices to type if one exists
  // TypeTreeMapType recur_mapping;

  static std::map<std::pair<ConcreteTypeMapType, TypeTreeMapType>, TypeResult>
      cache;

public:
  ConcreteType operator[](const std::vector<int> v) const {
    auto found = mapping.find(v);
    if (found != mapping.end()) {
      return found->second;
    }
    for (const auto &pair : mapping) {
      if (pair.first.size() != v.size())
        continue;
      bool match = true;
      for (unsigned i = 0; i < pair.first.size(); i++) {
        if (pair.first[i] == -1)
          continue;
        if (pair.first[i] != v[i]) {
          match = false;
          break;
        }
      }
      if (!match)
        continue;
      return pair.second;
    }
    return BaseType::Unknown;
  }

  void erase(const std::vector<int> v) { mapping.erase(v); }

  void insert(const std::vector<int> v, ConcreteType d,
              bool intsAreLegalSubPointer = false) {
    if (v.size() > 0) {
      // check pointer abilities from before
      {
        std::vector<int> tmp(v.begin(), v.end() - 1);
        auto found = mapping.find(tmp);
        if (found != mapping.end()) {
          if (!(found->second == BaseType::Pointer ||
                found->second == BaseType::Anything)) {
            llvm::errs() << "FAILED dt: " << str()
                         << " adding v: " << to_string(v) << ": " << d.str()
                         << "\n";
          }
          assert(found->second == BaseType::Pointer ||
                 found->second == BaseType::Anything);
        }
      }

      // don't insert if there's an existing ending -1
      {
        std::vector<int> tmp(v.begin(), v.end() - 1);
        tmp.push_back(-1);
        auto found = mapping.find(tmp);
        if (found != mapping.end()) {

          if (found->second != d) {
            if (d == BaseType::Anything) {
              found->second = d;
            } else {
              llvm::errs() << "FAILED dt: " << str()
                           << " adding v: " << to_string(v) << ": " << d.str()
                           << "\n";
            }
          }
          assert(found->second == d);
          return;
        }
      }

      // don't insert if there's an existing starting -1
      {
        std::vector<int> tmp(v.begin(), v.end());
        tmp[0] = -1;
        auto found = mapping.find(tmp);
        if (found != mapping.end()) {
          if (found->second != d) {
            if (d == BaseType::Anything) {
              found->second = d;
            } else {
              llvm::errs() << "FAILED dt: " << str()
                           << " adding v: " << to_string(v) << ": " << d.str()
                           << "\n";
            }
          }
          assert(found->second == d);
          return;
        }
      }

      // if this is a ending -1, remove other -1's
      if (v.back() == -1) {
        std::set<std::vector<int>> toremove;
        for (const auto &pair : mapping) {
          if (pair.first.size() == v.size()) {
            bool matches = true;
            for (unsigned i = 0; i < pair.first.size() - 1; i++) {
              if (pair.first[i] != v[i]) {
                matches = false;
                break;
              }
            }
            if (!matches)
              continue;

            if (intsAreLegalSubPointer &&
                pair.second.typeEnum == BaseType::Integer &&
                d.typeEnum == BaseType::Pointer) {

            } else {
              if (pair.second != d) {
                llvm::errs() << "inserting into : " << str() << " with "
                             << to_string(v) << " of " << d.str() << "\n";
              }
              assert(pair.second == d);
            }
            toremove.insert(pair.first);
          }
        }

        for (const auto &val : toremove) {
          mapping.erase(val);
        }
      }

      // if this is a starting -1, remove other -1's
      if (v[0] == -1) {
        std::set<std::vector<int>> toremove;
        for (const auto &pair : mapping) {
          if (pair.first.size() == v.size()) {
            bool matches = true;
            for (unsigned i = 1; i < pair.first.size(); i++) {
              if (pair.first[i] != v[i]) {
                matches = false;
                break;
              }
            }
            if (!matches)
              continue;
            assert(pair.second == d);
            toremove.insert(pair.first);
          }
        }

        for (const auto &val : toremove) {
          mapping.erase(val);
        }
      }
    }
    if (v.size() > 6) {
      llvm::errs() << "not handling more than 6 pointer lookups deep dt:"
                   << str() << " adding v: " << to_string(v) << ": " << d.str()
                   << "\n";
      return;
    }
    for (auto a : v) {
      if (a > 1000) {
        // llvm::errs() << "not handling more than 1000B offset pointer dt:" <<
        // str() << " adding v: " << to_string(v) << ": " << d.str() << "\n";
        return;
      }
    }
    mapping.insert(std::pair<const std::vector<int>, ConcreteType>(v, d));
  }

  bool operator<(const TypeTree &vd) const { return mapping < vd.mapping; }

  TypeTree() {}
  TypeTree(ConcreteType dat) {
    if (dat != ConcreteType(BaseType::Unknown)) {
      insert({}, dat);
    }
  }

  bool isKnown() {
    for (auto &pair : mapping) {
      // we should assert here as we shouldn't keep any unknown maps for
      // efficiency
      assert(pair.second.isKnown());
    }
    return mapping.size() != 0;
  }

  bool isKnownPastPointer() {
    for (auto &pair : mapping) {
      // we should assert here as we shouldn't keep any unknown maps for
      // efficiency
      assert(pair.second.isKnown());
      if (pair.first.size() == 0) {
        assert(pair.second == BaseType::Pointer);
        continue;
      }
      return true;
    }
    return false;
  }

  static TypeTree Unknown() { return TypeTree(); }

  TypeTree JustInt() const {
    TypeTree vd;
    for (auto &pair : mapping) {
      if (pair.second.typeEnum == BaseType::Integer) {
        vd.insert(pair.first, pair.second);
      }
    }

    return vd;
  }

  // TODO keep type information that is striated
  // e.g. if you have an i8* [0:Int, 8:Int] => i64* [0:Int, 1:Int]
  // After a depth len into the index tree, prune any lookups that are not {0}
  // or {-1}
  TypeTree KeepForCast(const llvm::DataLayout &dl, llvm::Type *from,
                        llvm::Type *to) const;

  static std::vector<int> appendIndex(int off, const std::vector<int> &first) {
    std::vector<int> out;
    out.push_back(off);
    for (auto a : first)
      out.push_back(a);
    return out;
  }

  TypeTree Only(int off) const {
    TypeTree dat;

    for (const auto &pair : mapping) {
      dat.insert(appendIndex(off, pair.first), pair.second);
      // if (pair.first.size() > 0) {
      //    dat.insert(indices, ConcreteType(BaseType::Pointer));
      //}
    }

    return dat;
  }

  static bool lookupIndices(std::vector<int> &first, int idx,
                            const std::vector<int> &second) {
    if (second.size() == 0)
      return false;

    assert(first.size() == 0);

    if (idx == -1) {
    } else if (second[0] == -1) {
    } else if (idx != second[0]) {
      return false;
    }

    for (size_t i = 1; i < second.size(); i++) {
      first.push_back(second[i]);
    }
    return true;
  }

  TypeTree Data0() const {
    TypeTree dat;

    for (const auto &pair : mapping) {
      assert(pair.first.size() != 0);

      if (pair.first[0] == 0 || pair.first[0] == -1) {
        std::vector<int> next;
        for (size_t i = 1; i < pair.first.size(); i++)
          next.push_back(pair.first[i]);
        TypeTree dat2;
        dat2.insert(next, pair.second);
        dat |= dat2;
      }
    }

    return dat;
  }

  TypeTree Clear(size_t start, size_t end, size_t len) const {
    TypeTree dat;

    for (const auto &pair : mapping) {
      assert(pair.first.size() != 0);

      if (pair.first[0] == -1) {
        TypeTree dat2;
        auto next = pair.first;
        for (size_t i = 0; i < start; i++) {
          next[0] = i;
          dat2.insert(next, pair.second);
        }
        for (size_t i = end; i < len; i++) {
          next[0] = i;
          dat2.insert(next, pair.second);
        }
        dat |= dat2;
      } else if ((size_t)pair.first[0] > start &&
                 (size_t)pair.first[0] >= end && (size_t)pair.first[0] < len) {
        TypeTree dat2;
        dat2.insert(pair.first, pair.second);
        dat |= dat2;
      }
    }

    // TODO canonicalize this
    return dat;
  }

  TypeTree Lookup(size_t len, const llvm::DataLayout &dl) const {

    // Map of indices[1:] => ( End => possible Index[0] )
    std::map<std::vector<int>, std::map<ConcreteType, std::set<int>>> staging;

    for (const auto &pair : mapping) {
      assert(pair.first.size() != 0);

      // Pointer is at offset 0 from this object
      if (pair.first[0] != 0 && pair.first[0] != -1)
        continue;

      if (pair.first.size() == 1) {
        assert(pair.second == ConcreteType(BaseType::Pointer) ||
               pair.second == ConcreteType(BaseType::Anything));
        continue;
      }

      if (pair.first[1] == -1) {
      } else {
        if ((size_t)pair.first[1] >= len)
          continue;
      }

      std::vector<int> next;
      for (size_t i = 2; i < pair.first.size(); i++) {
        next.push_back(pair.first[i]);
      }

      staging[next][pair.second].insert(pair.first[1]);
    }

    TypeTree dat;
    for (auto &pair : staging) {
      auto &pnext = pair.first;
      for (auto &pair2 : pair.second) {
        auto &dt = pair2.first;
        auto &set = pair2.second;

        bool legalCombine = set.count(-1);

        // See if we can canonicalize the outermost index into a -1
        if (!legalCombine) {
          size_t chunk = 1;
          if (auto flt = dt.isFloat()) {
            if (flt->isFloatTy()) {
              chunk = 4;
            } else if (flt->isDoubleTy()) {
              chunk = 8;
            } else if (flt->isHalfTy()) {
              chunk = 2;
            } else {
              llvm::errs() << *flt << "\n";
              assert(0 && "unhandled float type");
            }
          } else if (dt.typeEnum == BaseType::Pointer) {
            chunk = dl.getPointerSizeInBits() / 8;
          }

          legalCombine = true;
          for (size_t i = 0; i < len; i += chunk) {
            if (!set.count(i)) {
              legalCombine = false;
              break;
            }
          }
        }

        std::vector<int> next;
        next.push_back(-1);
        for (auto v : pnext)
          next.push_back(v);

        if (legalCombine) {
          dat.insert(next, dt, /*intsAreLegalPointerSub*/ true);
        } else {
          for (auto e : set) {
            next[0] = e;
            dat.insert(next, dt);
          }
        }
      }
    }

    return dat;
  }

  TypeTree CanonicalizeValue(size_t len, const llvm::DataLayout &dl) const {

    // Map of indices[1:] => ( End => possible Index[0] )
    std::map<std::vector<int>, std::map<ConcreteType, std::set<int>>> staging;

    for (const auto &pair : mapping) {
      assert(pair.first.size() != 0);

      std::vector<int> next;
      for (size_t i = 1; i < pair.first.size(); i++) {
        next.push_back(pair.first[i]);
      }

      staging[next][pair.second].insert(pair.first[0]);
    }

    TypeTree dat;
    for (auto &pair : staging) {
      auto &pnext = pair.first;
      for (auto &pair2 : pair.second) {
        auto &dt = pair2.first;
        auto &set = pair2.second;

        bool legalCombine = set.count(-1);

        // See if we can canonicalize the outermost index into a -1
        if (!legalCombine) {
          size_t chunk = 1;
          if (auto flt = dt.isFloat()) {
            if (flt->isFloatTy()) {
              chunk = 4;
            } else if (flt->isDoubleTy()) {
              chunk = 8;
            } else if (flt->isHalfTy()) {
              chunk = 2;
            } else {
              llvm::errs() << *flt << "\n";
              assert(0 && "unhandled float type");
            }
          } else if (dt.typeEnum == BaseType::Pointer) {
            chunk = dl.getPointerSizeInBits() / 8;
          }

          legalCombine = true;
          for (size_t i = 0; i < len; i += chunk) {
            if (!set.count(i)) {
              legalCombine = false;
              break;
            }
          }
        }

        std::vector<int> next;
        next.push_back(-1);
        for (auto v : pnext)
          next.push_back(v);

        if (legalCombine) {
          dat.insert(next, dt, /*intsAreLegalPointerSub*/ true);
        } else {
          for (auto e : set) {
            next[0] = e;
            dat.insert(next, dt);
          }
        }
      }
    }

    return dat;
  }

  TypeTree KeepMinusOne() const {
    TypeTree dat;

    for (const auto &pair : mapping) {

      assert(pair.first.size() != 0);

      // Pointer is at offset 0 from this object
      if (pair.first[0] != 0 && pair.first[0] != -1)
        continue;

      if (pair.first.size() == 1) {
        if (pair.second == BaseType::Pointer ||
            pair.second == BaseType::Anything) {
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

  //! Replace offsets in [offset, offset+maxSize] with [addOffset, addOffset +
  //! maxSize]
  TypeTree ShiftIndices(const llvm::DataLayout &dl, int offset, int maxSize,
                         size_t addOffset = 0) const {
    TypeTree dat;

    for (const auto &pair : mapping) {
      if (pair.first.size() == 0) {
        if (pair.second == BaseType::Pointer ||
            pair.second == BaseType::Anything) {
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

          // If we have a follow up offset add, we lose the -1 since we only
          // represent [0, inf) with -1 not the [addOffset, inf) required here
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
          if (next[0] >= maxSize)
            continue;
        }

        next[0] += addOffset;
      }

      TypeTree dat2;
      // llvm::errs() << "next: " << to_string(next) << " indices: " <<
      // to_string(indices) << " pair.first: " << to_string(pair.first) << "\n";
      if (next[0] == -1 && maxSize != -1) {
        size_t chunk = 1;
        auto op = operator[]({pair.first[0]});
        if (auto flt = op.isFloat()) {
          if (flt->isFloatTy()) {
            chunk = 4;
          } else if (flt->isDoubleTy()) {
            chunk = 8;
          } else if (flt->isHalfTy()) {
            chunk = 2;
          } else {
            llvm::errs() << *flt << "\n";
            assert(0 && "unhandled float type");
          }
        } else if (op.typeEnum == BaseType::Pointer) {
          chunk = dl.getPointerSizeInBits() / 8;
        }

        for (int i = 0; i < maxSize; i += chunk) {
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

  // Removes any anything types
  TypeTree PurgeAnything() const {
    TypeTree dat;
    for (const auto &pair : mapping) {
      if (pair.second == ConcreteType(BaseType::Anything))
        continue;
      dat.insert(pair.first, pair.second);
    }
    return dat;
  }

  // TODO note that this keeps -1's
  TypeTree AtMost(size_t max) const {
    assert(max > 0);
    TypeTree dat;
    for (const auto &pair : mapping) {
      if (pair.first.size() == 0 || pair.first[0] == -1 ||
          (size_t)pair.first[0] < max) {
        dat.insert(pair.first, pair.second);
      }
    }
    return dat;
  }

  static TypeTree Argument(ConcreteType type, llvm::Value *v) {
    if (v->getType()->isIntOrIntVectorTy())
      return TypeTree(type);
    return TypeTree(type).Only(0);
  }

  bool operator==(const TypeTree &v) const { return mapping == v.mapping; }

  // Return if changed
  bool operator=(const TypeTree &v) {
    if (*this == v)
      return false;
    mapping = v.mapping;
    return true;
  }

  bool mergeIn(const TypeTree &v, bool pointerIntSame) {
    //! Todo detect recursive merge

    bool changed = false;

    if (v[{-1}] != BaseType::Unknown) {
      for (auto &pair : mapping) {
        if (pair.first.size() == 1 && pair.first[0] != -1) {
          pair.second.mergeIn(v[{-1}], pointerIntSame);
          // if (pair.second == ) // NOTE DELETE the non -1
        }
      }
    }

    for (auto &pair : v.mapping) {
      assert(pair.second != BaseType::Unknown);
      ConcreteType dt = operator[](pair.first);
      // llvm::errs() << "merging @ " << to_string(pair.first) << " old:" <<
      // dt.str() << " new:" << pair.second.str() << "\n";
      changed |= (dt.mergeIn(pair.second, pointerIntSame));

      /*
      if (dt == BaseType::Integer && pair.first.size() > 0 && pair.first.back()
      != -1) { auto p2(pair.first); for(unsigned i=max((int)pair.first.back()-4,
      0); i<(unsigned)pair.first.back(); i++) { p2[p2.size()-1] == i; if
      (operator[](p2).typeEnum == BaseType::Float) { llvm::errs() << " illegal
      merge of " << v.str() << " into " << str() << "\n"; assert(0 &&
      "badmerge"); exit(1);
              }
          }
      }

      if (dt == BaseType::Float && pair.first.size() > 0 && pair.first.back() !=
      -1) { auto p2(pair.first); for(unsigned i=pair.first.back();
      i<(unsigned)pair.first.back()+4; i++) { p2[p2.size()-1] == i; if
      (operator[](p2).typeEnum == BaseType::Integer) { llvm::errs() << " illegal
      merge of " << v.str() << " into " << str() << "\n"; assert(0 &&
      "badmerg2"); exit(1);
              }
          }
      }
      */

      insert(pair.first, dt);
    }
    return changed;
  }

  bool operator|=(const TypeTree &v) {
    return mergeIn(v, /*pointerIntSame*/ false);
  }

  bool operator&=(const TypeTree &v) {
    return andIn(v, /*assertIfIllegal*/ true);
  }

  bool andIn(const TypeTree &v, bool assertIfIllegal = true) {
    bool changed = false;

    std::vector<std::vector<int>> keystodelete;
    for (auto &pair : mapping) {
      ConcreteType other = BaseType::Unknown;
      auto fd = v.mapping.find(pair.first);
      if (fd != v.mapping.end()) {
        other = fd->second;
      }
      changed = (pair.second.andIn(other, assertIfIllegal));
      if (pair.second == BaseType::Unknown) {
        keystodelete.push_back(pair.first);
      }
    }

    for (auto &key : keystodelete) {
      mapping.erase(key);
    }

    return changed;
  }

  bool pointerIntMerge(const TypeTree &v, llvm::BinaryOperator::BinaryOps op) {
    bool changed = false;

    auto found = mapping.find({});
    if (found != mapping.end()) {
      changed |= (found->second.pointerIntMerge(v[{}], op));
      if (found->second == BaseType::Unknown) {
        mapping.erase(std::vector<int>({}));
      }
    } else if (v.mapping.find({}) != v.mapping.end()) {
      ConcreteType dt(BaseType::Unknown);
      dt.pointerIntMerge(v[{}], op);
      if (dt != BaseType::Unknown) {
        changed = true;
        mapping.emplace(std::vector<int>({}), dt);
      }
    }

    std::vector<std::vector<int>> keystodelete;

    for (auto &pair : mapping) {
      if (pair.first != std::vector<int>({}))
        keystodelete.push_back(pair.first);
    }

    for (auto &key : keystodelete) {
      mapping.erase(key);
      changed = true;
    }

    return changed;
  }

  std::string str() const {
    std::string out = "{";
    bool first = true;
    for (auto &pair : mapping) {
      if (!first) {
        out += ", ";
      }
      out += "[";
      for (unsigned i = 0; i < pair.first.size(); i++) {
        if (i != 0)
          out += ",";
        out += std::to_string(pair.first[i]);
      }
      out += "]:" + pair.second.str();
      first = false;
    }
    out += "}";
    return out;
  }
};

class FnTypeInfo {
public:
  llvm::Function *function;
  FnTypeInfo(llvm::Function *fn) : function(fn) {}
  FnTypeInfo(const FnTypeInfo &) = default;
  FnTypeInfo &operator=(FnTypeInfo &) = default;
  FnTypeInfo &operator=(FnTypeInfo &&) = default;

  // arguments:type
  std::map<llvm::Argument *, TypeTree> first;
  // return type
  TypeTree second;
  // the specific constant of an argument, if it is constant
  std::map<llvm::Argument *, std::set<int64_t>> knownValues;

  std::set<int64_t>
  knownIntegralValues(llvm::Value *val, const llvm::DominatorTree &DT,
                std::map<llvm::Value *, std::set<int64_t>> &intseen) const;
};

static inline bool operator<(const FnTypeInfo &lhs,
                             const FnTypeInfo &rhs) {

  if (lhs.function < rhs.function)
    return true;
  if (rhs.function < lhs.function)
    return false;

  if (lhs.first < rhs.first)
    return true;
  if (rhs.first < lhs.first)
    return false;
  if (lhs.second < rhs.second)
    return true;
  if (rhs.second < lhs.second)
    return false;
  return lhs.knownValues < rhs.knownValues;
}

class TypeAnalyzer;
class TypeAnalysis;

class TypeResults {
public:
  TypeAnalysis &analysis;
  const FnTypeInfo info;

public:
  TypeResults(TypeAnalysis &analysis, const FnTypeInfo &fn);
  ConcreteType intType(llvm::Value *val, bool errIfNotFound = true);

  //! Returns whether in the first num bytes there is pointer, int, float, or
  //! none If pointerIntSame is set to true, then consider either as the same
  //! (and thus mergable)
  ConcreteType firstPointer(size_t num, llvm::Value *val, bool errIfNotFound = true,
                        bool pointerIntSame = false);

  TypeTree query(llvm::Value *val);
  FnTypeInfo getAnalyzedTypeInfo();
  TypeTree getReturnAnalysis();
  void dump();
  std::set<int64_t> knownIntegralValues(llvm::Value *val) const;
};

class TypeAnalyzer : public llvm::InstVisitor<TypeAnalyzer> {
public:
  // List of value's which should be re-analyzed now with new information
  std::deque<llvm::Value *> workList;

private:
  void addToWorkList(llvm::Value *val);
  std::map<llvm::Value *, std::set<int64_t>> intseen;

public:
  // Calling context
  const FnTypeInfo fntypeinfo;

  TypeAnalysis &interprocedural;

  std::map<llvm::Value *, TypeTree> analysis;

  llvm::DominatorTree DT;

  TypeAnalyzer(const FnTypeInfo &fn, TypeAnalysis &TA);

  TypeTree getAnalysis(llvm::Value *val);

  void updateAnalysis(llvm::Value *val, BaseType data, llvm::Value *origin);
  void updateAnalysis(llvm::Value *val, ConcreteType data, llvm::Value *origin);
  void updateAnalysis(llvm::Value *val, TypeTree data, llvm::Value *origin);

  void prepareArgs();

  void considerTBAA();

  void run();

  bool runUnusedChecks();

  void visitValue(llvm::Value &val);

  void visitCmpInst(llvm::CmpInst &I);

  void visitAllocaInst(llvm::AllocaInst &I);

  void visitLoadInst(llvm::LoadInst &I);

  void visitStoreInst(llvm::StoreInst &I);

  void visitGetElementPtrInst(llvm::GetElementPtrInst &gep);

  void visitPHINode(llvm::PHINode &phi);

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

  void visitIPOCall(llvm::CallInst &call, llvm::Function &fn);

  void visitCallInst(llvm::CallInst &call);

  void visitMemTransferInst(llvm::MemTransferInst &MTI);

  void visitIntrinsicInst(llvm::IntrinsicInst &II);

  TypeTree getReturnAnalysis();

  void dump();

  std::set<int64_t> knownIntegralValues(llvm::Value *val);
};

class TypeAnalysis {
public:
  std::map<FnTypeInfo, TypeAnalyzer> analyzedFunctions;

  TypeResults analyzeFunction(const FnTypeInfo &fn);

  TypeTree query(llvm::Value *val, const FnTypeInfo &fn);

  ConcreteType intType(llvm::Value *val, const FnTypeInfo &fn,
                   bool errIfNotFound = true);
  ConcreteType firstPointer(size_t num, llvm::Value *val, const FnTypeInfo &fn,
                        bool errIfNotFound = true, bool pointerIntSame = false);

  inline TypeTree getReturnAnalysis(const FnTypeInfo &fn) {
    analyzeFunction(fn);
    return analyzedFunctions.find(fn)->second.getReturnAnalysis();
  }
};

#endif
