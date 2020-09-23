//===- TypeTree.cpp - Declaration of Type Analysis Type Trees   -----------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @misc{enzymeGithub,
//  author = {William S. Moses and Valentin Churavy},
//  title = {Enzyme: High Performance Automatic Differentiation of LLVM},
//  year = {2020},
//  howpublished = {\url{https://github.com/wsmoses/Enzyme}},
//  note = {commit xxxxxxx}
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of TypeTrees -- a class
// representing all of the underlying types of a particular LLVM value. This
// consists of a map of memory offsets to an underlying ConcreteType. This
// permits TypeTrees to represent distinct underlying types at different
// locations. Presently, TypeTree's have both a fixed depth of memory lookups
// and a maximum offset to ensure that Type Analysis eventually terminates.
// In the future this should be modified to better represent recursive types
// rather than limiting the depth.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_TYPE_ANALYSIS_TYPE_TREE_H
#define ENZYME_TYPE_ANALYSIS_TYPE_TREE_H 1

#include <map>
#include <set>
#include <string>
#include <vector>
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"

#include "BaseType.h"
#include "ConcreteType.h"

static inline std::string to_string(const std::vector<int> x) {
  std::string out = "[";
  for (unsigned i = 0; i < x.size(); ++i) {
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
      for (unsigned i = 0; i < pair.first.size(); ++i) {
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
            for (unsigned i = 0; i < pair.first.size() - 1; ++i) {
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
            for (unsigned i = 1; i < pair.first.size(); ++i) {
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

    for (size_t i = 1; i < second.size(); ++i) {
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
        for (size_t i = 1; i < pair.first.size(); ++i)
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
        for (size_t i = 0; i < start; ++i) {
          next[0] = i;
          dat2.insert(next, pair.second);
        }
        for (size_t i = end; i < len; ++i) {
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
      for (size_t i = 2; i < pair.first.size(); ++i) {
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
      for (size_t i = 1; i < pair.first.size(); ++i) {
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
    mapping.clear();
    for(const auto& elems : v.mapping) {
      mapping.emplace(elems);
    }
    //mapping = v.mapping;
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
      for (unsigned i = 0; i < pair.first.size(); ++i) {
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

#endif
