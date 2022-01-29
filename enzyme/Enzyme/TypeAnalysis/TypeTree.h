//===- TypeTree.cpp - Declaration of Type Analysis Type Trees   -----------===//
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

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
#include <string>
#include <vector>

#include "BaseType.h"
#include "ConcreteType.h"

/// Maximum offset for type trees to keep
extern "C" {
extern llvm::cl::opt<int> MaxTypeOffset;
extern llvm::cl::opt<bool> EnzymeTypeWarning;
constexpr int EnzymeMaxTypeDepth = 6;
}

/// Helper function to print a vector of ints to a string
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

/// Class representing the underlying types of values as
/// sequences of offsets to a ConcreteType
class TypeTree : public std::enable_shared_from_this<TypeTree> {
private:
  // mapping of known indices to type if one exists
  ConcreteTypeMapType mapping;
  std::vector<int> minIndices;

public:
  TypeTree() {}
  TypeTree(ConcreteType dat) {
    if (dat != ConcreteType(BaseType::Unknown)) {
      mapping.insert(std::pair<const std::vector<int>, ConcreteType>({}, dat));
    }
  }

  /// Utility helper to lookup the mapping
  const ConcreteTypeMapType &getMapping() const { return mapping; }

  /// Lookup the underlying ConcreteType at a given offset sequence
  /// or Unknown if none exists
  ConcreteType operator[](const std::vector<int> Seq) const {
    auto Found0 = mapping.find(Seq);
    if (Found0 != mapping.end())
      return Found0->second;
    size_t Len = Seq.size();
    if (Len == 0)
      return BaseType::Unknown;

    std::vector<std::vector<int>> todo[2];
    todo[0].push_back({});
    int parity = 0;
    for (size_t i = 0, Len = Seq.size(); i < Len - 1; ++i) {
      for (auto prev : todo[parity]) {
        prev.push_back(-1);
        if (mapping.find(prev) != mapping.end())
          todo[1 - parity].push_back(prev);
        if (Seq[i] != -1) {
          prev.back() = Seq[i];
          if (mapping.find(prev) != mapping.end())
            todo[1 - parity].push_back(prev);
        }
      }
      todo[parity].clear();
      parity = 1 - parity;
    }

    size_t i = Len - 1;
    for (auto prev : todo[parity]) {
      prev.push_back(-1);
      auto Found = mapping.find(prev);
      if (Found != mapping.end())
        return Found->second;
      if (Seq[i] != -1) {
        prev.back() = Seq[i];
        Found = mapping.find(prev);
        if (Found != mapping.end())
          return Found->second;
      }
    }
    return BaseType::Unknown;
  }

  // Return true if this type tree is fully known (i.e. there
  // is no more information which could be added).
  bool IsFullyDetermined() const {
    std::vector<int> offsets = {-1};
    while (1) {
      auto found = mapping.find(offsets);
      if (found == mapping.end())
        return false;
      if (found->second != BaseType::Pointer)
        return true;
      offsets.push_back(-1);
    }
  }

  /// Return if changed
  bool insert(const std::vector<int> Seq, ConcreteType CT,
              bool intsAreLegalSubPointer = false) {
    size_t SeqSize = Seq.size();
    if (SeqSize > EnzymeMaxTypeDepth) {
      if (EnzymeTypeWarning)
        llvm::errs() << "not handling more than " << EnzymeMaxTypeDepth
                     << " pointer lookups deep dt:" << str()
                     << " adding v: " << to_string(Seq) << ": " << CT.str()
                     << "\n";
      return false;
    }
    if (SeqSize == 0) {
      mapping.insert(std::pair<const std::vector<int>, ConcreteType>(Seq, CT));
      return true;
    }

    // check types at lower pointer offsets are either pointer or
    // anything. Don't insert into an anything
    {
      std::vector<int> tmp(Seq);
      while (tmp.size() > 0) {
        tmp.erase(tmp.end() - 1);
        auto found = mapping.find(tmp);
        if (found != mapping.end()) {
          if (found->second == BaseType::Anything)
            return false;
          if (found->second != BaseType::Pointer) {
            llvm::errs() << "FAILED CT: " << str()
                         << " adding Seq: " << to_string(Seq) << ": "
                         << CT.str() << "\n";
          }
          assert(found->second == BaseType::Pointer);
        }
      }
    }

    bool changed = false;

    // if this is a ending -1, remove other elems if no more info
    if (Seq.back() == -1) {
      std::set<std::vector<int>> toremove;
      for (const auto &pair : mapping) {
        if (pair.first.size() != SeqSize)
          continue;
        bool matches = true;
        for (unsigned i = 0; i < SeqSize - 1; ++i) {
          if (pair.first[i] != Seq[i]) {
            matches = false;
            break;
          }
        }
        if (!matches)
          continue;

        if (intsAreLegalSubPointer && pair.second == BaseType::Integer &&
            CT == BaseType::Pointer) {
          toremove.insert(pair.first);
        } else {
          if (CT == pair.second) {
            // previous equivalent values or values overwritten by
            // an anything are removed
            toremove.insert(pair.first);
          } else if (pair.second != BaseType::Anything) {
            llvm::errs() << "inserting into : " << str() << " with "
                         << to_string(Seq) << " of " << CT.str() << "\n";
            llvm_unreachable("illegal insertion");
          }
        }
      }

      for (const auto &val : toremove) {
        mapping.erase(val);
        changed = true;
      }
    }

    // if this is a starting -1, remove other -1's
    if (Seq[0] == -1) {
      std::set<std::vector<int>> toremove;
      for (const auto &pair : mapping) {
        if (pair.first.size() != SeqSize)
          continue;
        bool matches = true;
        for (unsigned i = 1; i < SeqSize; ++i) {
          if (pair.first[i] != Seq[i]) {
            matches = false;
            break;
          }
        }
        if (!matches)
          continue;
        if (intsAreLegalSubPointer && pair.second == BaseType::Integer &&
            CT == BaseType::Pointer) {
          toremove.insert(pair.first);
        } else {
          if (CT == pair.second) {
            // previous equivalent values or values overwritten by
            // an anything are removed
            toremove.insert(pair.first);
          } else if (pair.second != BaseType::Anything) {
            llvm::errs() << "inserting into : " << str() << " with "
                         << to_string(Seq) << " of " << CT.str() << "\n";
            llvm_unreachable("illegal insertion");
          }
        }
      }

      for (const auto &val : toremove) {
        mapping.erase(val);
        changed = true;
      }
    }

    bool possibleDeletion = false;
    size_t minLen =
        (minIndices.size() <= SeqSize) ? minIndices.size() : SeqSize;
    for (size_t i = 0; i < minLen; i++) {
      if (minIndices[i] > Seq[i]) {
        if (minIndices[i] > MaxTypeOffset)
          possibleDeletion = true;
        minIndices[i] = Seq[i];
      }
    }

    if (minIndices.size() < SeqSize) {
      for (size_t i = minIndices.size(), end = SeqSize; i < end; ++i) {
        minIndices.push_back(Seq[i]);
      }
    }

    if (possibleDeletion) {
      std::vector<std::vector<int>> toErase;
      for (const auto &pair : mapping) {
        size_t i = 0;
        bool mustKeep = false;
        bool considerErase = false;
        for (int val : pair.first) {
          if (val > MaxTypeOffset) {
            if (val == minIndices[i]) {
              mustKeep = true;
              break;
            }
            considerErase = true;
          }
          ++i;
        }
        if (!mustKeep && considerErase) {
          toErase.push_back(pair.first);
        }
      }

      for (auto vec : toErase) {
        mapping.erase(vec);
        changed = true;
      }
    }

    size_t i = 0;
    bool keep = false;
    bool considerErase = false;
    for (auto val : Seq) {
      if (val > MaxTypeOffset) {
        if (val == minIndices[i]) {
          keep = true;
          break;
        }
        considerErase = true;
      }
      i++;
    }
    if (considerErase && !keep)
      return changed;
    mapping.insert(std::pair<const std::vector<int>, ConcreteType>(Seq, CT));
    return true;
  }

  /// How this TypeTree compares with another
  bool operator<(const TypeTree &vd) const { return mapping < vd.mapping; }

  /// Whether this TypeTree contains any information
  bool isKnown() const {
    for (const auto &pair : mapping) {
      // we should assert here as we shouldn't keep any unknown maps for
      // efficiency
      assert(pair.second.isKnown());
    }
    return mapping.size() != 0;
  }

  /// Whether this TypeTree knows any non-pointer information
  bool isKnownPastPointer() const {
    for (auto &pair : mapping) {
      // we should assert here as we shouldn't keep any unknown maps for
      // efficiency
      assert(pair.second.isKnown());
      if (pair.first.size() == 0) {
        assert(pair.second == BaseType::Pointer ||
               pair.second == BaseType::Anything);
        continue;
      }
      return true;
    }
    return false;
  }

  /// Select only the Integer ConcreteTypes
  TypeTree JustInt() const {
    TypeTree vd;
    for (auto &pair : mapping) {
      if (pair.second == BaseType::Integer) {
        vd.insert(pair.first, pair.second);
      }
    }

    return vd;
  }

  /// Prepend an offset to all mappings
  TypeTree Only(int Off) const {
    TypeTree Result;
    Result.minIndices.reserve(1 + minIndices.size());
    Result.minIndices.push_back(Off);
    for (auto midx : minIndices)
      Result.minIndices.push_back(midx);

    if (Result.minIndices.size() > EnzymeMaxTypeDepth) {
      Result.minIndices.pop_back();
      if (EnzymeTypeWarning)
        llvm::errs() << "not handling more than " << EnzymeMaxTypeDepth
                     << " pointer lookups deep dt:" << str() << " only(" << Off
                     << "): " << str() << "\n";
    }

    for (const auto &pair : mapping) {
      if (pair.first.size() == EnzymeMaxTypeDepth)
        continue;
      std::vector<int> Vec;
      Vec.reserve(pair.first.size() + 1);
      Vec.push_back(Off);
      for (auto Val : pair.first)
        Vec.push_back(Val);
      Result.mapping.insert(
          std::pair<const std::vector<int>, ConcreteType>(Vec, pair.second));
    }
    return Result;
  }

  /// Peel off the outermost index at offset 0
  TypeTree Data0() const {
    TypeTree Result;

    for (const auto &pair : mapping) {
      if (pair.first.size() == 0) {
        llvm::errs() << str() << "\n";
      }
      assert(pair.first.size() != 0);

      if (pair.first[0] == -1) {
        std::vector<int> next(pair.first.begin() + 1, pair.first.end());
        Result.mapping.insert(
            std::pair<const std::vector<int>, ConcreteType>(next, pair.second));
        for (size_t i = 0, Len = next.size(); i < Len; ++i) {
          if (i == Result.minIndices.size())
            Result.minIndices.push_back(next[i]);
          else if (next[i] < Result.minIndices[i])
            Result.minIndices[i] = next[i];
        }
      }
    }
    for (const auto &pair : mapping) {
      if (pair.first[0] == 0) {
        std::vector<int> next(pair.first.begin() + 1, pair.first.end());
        // We do insertion like this to force an error
        // on the orIn operation if there is an incompatible
        // merge. The insert operation does not error.
        Result.orIn(next, pair.second);
      }
    }

    return Result;
  }

  /// Optimized version of Data0()[{}]
  ConcreteType Inner0() const {
    ConcreteType CT = operator[]({-1});
    CT |= operator[]({0});
    return CT;
  }

  /// Remove any mappings in the range [start, end) or [len, inf)
  /// This function has special handling for -1's
  TypeTree Clear(size_t start, size_t end, size_t len) const {
    TypeTree Result;

    // Note that below do insertion with the orIn operator
    // to force an error if there is an incompatible
    // merge. The insert operation does not error.

    for (const auto &pair : mapping) {
      assert(pair.first.size() != 0);

      if (pair.first[0] == -1) {
        // For "all index" calculations, explicitly
        // add mappings for regions in range
        auto next = pair.first;
        for (size_t i = 0; i < start; ++i) {
          next[0] = i;
          Result.orIn(next, pair.second);
        }
        for (size_t i = end; i < len; ++i) {
          next[0] = i;
          Result.orIn(next, pair.second);
        }
      } else if ((size_t)pair.first[0] < start ||
                 ((size_t)pair.first[0] >= end &&
                  (size_t)pair.first[0] < len)) {
        // Otherwise simply check that the given offset is in range

        Result.insert(pair.first, pair.second);
      }
    }

    // TODO canonicalize this
    return Result;
  }

  /// Select all submappings whose first index is in range [0, len) and remove
  /// the first index. This is the inverse of the `Only` operation
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

      std::vector<int> next(pair.first.begin() + 2, pair.first.end());

      staging[next][pair.second].insert(pair.first[1]);
    }

    TypeTree Result;
    for (auto &pair : staging) {
      auto &pnext = pair.first;
      for (auto &pair2 : pair.second) {
        auto dt = pair2.first;
        const auto &set = pair2.second;

        bool legalCombine = set.count(-1);

        // See if we can canonicalize the outermost index into a -1
        if (!legalCombine) {
          size_t chunk = 1;
          // Implicit pointer
          if (set.size() > 0) {
            chunk = dl.getPointerSizeInBits() / 8;
          } else {
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
            } else if (dt == BaseType::Pointer) {
              chunk = dl.getPointerSizeInBits() / 8;
            }
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
        next.reserve(pnext.size() + 1);
        next.push_back(-1);
        for (auto v : pnext)
          next.push_back(v);

        if (legalCombine) {
          Result.insert(next, dt, /*intsAreLegalPointerSub*/ true);
        } else {
          for (auto e : set) {
            next[0] = e;
            Result.insert(next, dt);
          }
        }
      }
    }

    return Result;
  }

  /// Given that this tree represents something of at most size len,
  /// canonicalize this, creating -1's where possible
  void CanonicalizeInPlace(size_t len, const llvm::DataLayout &dl) {
    bool canonicalized = true;
    for (const auto &pair : mapping) {
      assert(pair.first.size() != 0);
      if (pair.first[0] != -1) {
        canonicalized = false;
        break;
      }
    }
    if (canonicalized)
      return;

    // Map of indices[1:] => ( End => possible Index[0] )
    std::map<const std::vector<int>, std::map<ConcreteType, std::set<int>>>
        staging;

    for (const auto &pair : mapping) {

      std::vector<int> next(pair.first.begin() + 1, pair.first.end());
      if (pair.first[0] != -1) {
        if ((size_t)pair.first[0] >= len) {
          llvm::errs() << str() << "\n";
          llvm::errs() << " canonicalizing " << len << "\n";
        }
        assert((size_t)pair.first[0] < len);
      }
      staging[next][pair.second].insert(pair.first[0]);
    }

    mapping.clear();

    for (auto &pair : staging) {
      auto &pnext = pair.first;
      for (auto &pair2 : pair.second) {
        auto dt = pair2.first;
        const auto &set = pair2.second;

        // llvm::errs() << " - set: {";
        // for(auto s : set) llvm::errs() << s << ", ";
        // llvm::errs() << "} len=" << len << "\n";

        bool legalCombine = set.count(-1);

        // See if we can canonicalize the outermost index into a -1
        if (!legalCombine) {
          size_t chunk = 1;
          if (set.size() > 0) {
            chunk = dl.getPointerSizeInBits() / 8;
          } else {
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
            } else if (dt == BaseType::Pointer) {
              chunk = dl.getPointerSizeInBits() / 8;
            }
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
        next.reserve(pnext.size() + 1);
        next.push_back(-1);
        for (auto v : pnext)
          next.push_back(v);

        if (legalCombine) {
          insert(next, dt, /*intsAreLegalPointerSub*/ true);
        } else {
          for (auto e : set) {
            next[0] = e;
            insert(next, dt);
          }
        }
      }
    }
  }

  /// Keep only pointers (or anything's) to a repeated value (represented by -1)
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
        assert(0 && "could not merge");
        llvm_unreachable("could not merge");
      }

      if (pair.first[1] == -1) {
        dat.insert(pair.first, pair.second);
      }
    }

    return dat;
  }

  llvm::Type *IsAllFloat(const size_t size) const {
    auto m1 = TypeTree::operator[]({-1});
    if (auto FT = m1.isFloat())
      return FT;

    auto m0 = TypeTree::operator[]({0});

    if (auto flt = m0.isFloat()) {
      size_t chunk;
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
      for (size_t i = chunk; i < size; i += chunk) {
        auto mx = TypeTree::operator[]({(int)i});
        if (auto f2 = mx.isFloat()) {
          if (f2 != flt)
            return nullptr;
        } else
          return nullptr;
      }
      return flt;
    } else {
      return nullptr;
    }
  }

  /// Replace mappings in the range in [offset, offset+maxSize] with those in
  // [addOffset, addOffset + maxSize]. In other worse, select all mappings in
  // [offset, offset+maxSize] then add `addOffset`
  TypeTree ShiftIndices(const llvm::DataLayout &dl, const int offset,
                        const int maxSize, size_t addOffset = 0) const {
    TypeTree Result;

    for (const auto &pair : mapping) {
      if (pair.first.size() == 0) {
        if (pair.second == BaseType::Pointer ||
            pair.second == BaseType::Anything) {
          Result.insert(pair.first, pair.second);
          continue;
        }

        llvm::errs() << "could not unmerge " << str() << "\n";
        assert(0 && "ShiftIndices called on a nonpointer/anything");
        llvm_unreachable("ShiftIndices called on a nonpointer/anything");
      }

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
      } else if (op == BaseType::Pointer) {
        chunk = dl.getPointerSizeInBits() / 8;
      }

      if (next[0] == -1 && maxSize != -1) {
        auto offincr = (chunk - offset % chunk) % chunk;
        for (int i = offincr; i < maxSize; i += chunk) {
          next[0] = i + addOffset;
          Result.orIn(next, pair.second);
        }
      } else {
        Result.orIn(next, pair.second);
      }
    }

    return Result;
  }

  /// Keep only mappings where the type is not an `Anything`
  TypeTree PurgeAnything() const {
    TypeTree Result;
    Result.minIndices.reserve(minIndices.size());
    for (const auto &pair : mapping) {
      if (pair.second == ConcreteType(BaseType::Anything))
        continue;
      Result.mapping.insert(pair);
      for (size_t i = 0, Len = pair.first.size(); i < Len; ++i) {
        if (i == Result.minIndices.size())
          Result.minIndices.push_back(pair.first[i]);
        else if (pair.first[i] < Result.minIndices[i])
          Result.minIndices[i] = pair.first[i];
      }
    }
    return Result;
  }

  /// Replace -1 with 0
  TypeTree ReplaceMinus() const {
    TypeTree dat;
    for (const auto pair : mapping) {
      if (pair.second == ConcreteType(BaseType::Anything))
        continue;
      std::vector<int> nex = pair.first;
      for (auto &v : nex)
        if (v == -1)
          v = 0;
      dat.insert(nex, pair.second);
    }
    return dat;
  }

  /// Replace all integer subtypes with anything
  void ReplaceIntWithAnything() {
    for (auto &pair : mapping) {
      if (pair.second == BaseType::Integer) {
        pair.second = BaseType::Anything;
      }
    }
  }

  /// Keep only mappings where the type is an `Anything`
  TypeTree JustAnything() const {
    TypeTree dat;
    for (const auto &pair : mapping) {
      if (pair.second != ConcreteType(BaseType::Anything))
        continue;
      dat.insert(pair.first, pair.second);
    }
    return dat;
  }

  /// Chceck equality of two TypeTrees
  bool operator==(const TypeTree &RHS) const { return mapping == RHS.mapping; }

  /// Set this to another TypeTree, returning if this was changed
  bool operator=(const TypeTree &RHS) {
    if (*this == RHS)
      return false;
    minIndices = RHS.minIndices;
    mapping.clear();
    for (const auto &elems : RHS.mapping) {
      mapping.emplace(elems);
    }
    return true;
  }

  bool checkedOrIn(const std::vector<int> &Seq, ConcreteType RHS,
                   bool PointerIntSame, bool &LegalOr) {
    assert(RHS != BaseType::Unknown);
    ConcreteType CT = operator[](Seq);

    bool subchanged = CT.checkedOrIn(RHS, PointerIntSame, LegalOr);
    if (!subchanged)
      return false;
    if (!LegalOr)
      return subchanged;

    if (Seq.size() > 0) {
      // check pointer abilities from before
      {
        std::vector<int> tmp(Seq.begin(), Seq.end() - 1);
        auto found = mapping.find(tmp);
        if (found != mapping.end()) {
          if (!(found->second == BaseType::Pointer ||
                found->second == BaseType::Anything)) {
            LegalOr = false;
            return false;
          }
        }
      }

      // if this is a ending -1, remove other elems if no more info
      if (Seq.back() == -1) {
        std::set<std::vector<int>> toremove;
        for (const auto &pair : mapping) {
          if (pair.first.size() == Seq.size()) {
            bool matches = true;
            for (unsigned i = 0; i < pair.first.size() - 1; ++i) {
              if (pair.first[i] != Seq[i]) {
                matches = false;
                break;
              }
            }
            if (!matches)
              continue;

            if (CT == BaseType::Anything || CT == pair.second) {
              // previous equivalent values or values overwritten by
              // an anything are removed
              toremove.insert(pair.first);
            } else if (CT != BaseType::Anything &&
                       pair.second == BaseType::Anything) {
              // keep lingering anythings if not being overwritten
            } else {
              LegalOr = false;
              return false;
            }
          }
        }
        for (const auto &val : toremove) {
          mapping.erase(val);
        }
      }

      // if this is a starting -1, remove other -1's
      if (Seq[0] == -1) {
        std::set<std::vector<int>> toremove;
        for (const auto &pair : mapping) {
          if (pair.first.size() == Seq.size()) {
            bool matches = true;
            for (unsigned i = 1; i < pair.first.size(); ++i) {
              if (pair.first[i] != Seq[i]) {
                matches = false;
                break;
              }
            }
            if (!matches)
              continue;

            if (CT == BaseType::Anything || CT == pair.second) {
              // previous equivalent values or values overwritten by
              // an anything are removed
              toremove.insert(pair.first);
            } else if (CT != BaseType::Anything &&
                       pair.second == BaseType::Anything) {
              // keep lingering anythings if not being overwritten
            } else {
              LegalOr = false;
              return false;
            }
          }
        }

        for (const auto &val : toremove) {
          mapping.erase(val);
        }
      }
    }

    return insert(Seq, CT);
  }

  bool orIn(const std::vector<int> &Seq, ConcreteType RHS,
            bool PointerIntSame = false) {
    bool LegalOr = true;
    bool Result = checkedOrIn(Seq, RHS, PointerIntSame, LegalOr);
    assert(LegalOr);
    return Result;
  }

  /// Set this to the logical or of itself and RHS, returning whether this value
  /// changed Setting `PointerIntSame` considers pointers and integers as
  /// equivalent If this is an illegal operation, `LegalOr` will be set to false
  bool checkedOrIn(const TypeTree &RHS, bool PointerIntSame, bool &LegalOr) {
    // TODO detect recursive merge and simplify

    bool changed = false;
    for (auto &pair : RHS.mapping) {
      changed |= checkedOrIn(pair.first, pair.second, PointerIntSame, LegalOr);
    }
    return changed;
  }

  /// Set this to the logical or of itself and RHS, returning whether this value
  /// changed Setting `PointerIntSame` considers pointers and integers as
  /// equivalent This function will error if doing an illegal Operation
  bool orIn(const TypeTree RHS, bool PointerIntSame) {
    bool Legal = true;
    bool Result = checkedOrIn(RHS, PointerIntSame, Legal);
    if (!Legal) {
      llvm::errs() << "Illegal orIn: " << str() << " right: " << RHS.str()
                   << " PointerIntSame=" << PointerIntSame << "\n";
      assert(0 && "Performed illegal ConcreteType::orIn");
      llvm_unreachable("Performed illegal ConcreteType::orIn");
    }
    return Result;
  }

  /// Set this to the logical or of itself and RHS, returning whether this value
  /// changed Setting `PointerIntSame` considers pointers and integers as
  /// equivalent This function will error if doing an illegal Operation
  bool orIn(const std::vector<int> Seq, ConcreteType CT, bool PointerIntSame) {
    bool Legal = true;
    bool Result = checkedOrIn(Seq, CT, PointerIntSame, Legal);
    if (!Legal) {
      llvm::errs() << "Illegal orIn: " << str() << " right: " << to_string(Seq)
                   << " CT: " << CT.str()
                   << " PointerIntSame=" << PointerIntSame << "\n";
      assert(0 && "Performed illegal ConcreteType::orIn");
      llvm_unreachable("Performed illegal ConcreteType::orIn");
    }
    return Result;
  }

  /// Set this to the logical or of itself and RHS, returning whether this value
  /// changed This assumes that pointers and integers are distinct This function
  /// will error if doing an illegal Operation
  bool operator|=(const TypeTree &RHS) {
    return orIn(RHS, /*PointerIntSame*/ false);
  }

  /// Set this to the logical and of itself and RHS, returning whether this
  /// value changed If this and RHS are incompatible at an index, the result
  /// will be BaseType::Unknown
  bool andIn(const TypeTree &RHS) {
    bool changed = false;

    std::vector<std::vector<int>> keystodelete;
    for (auto &pair : mapping) {
      ConcreteType other = BaseType::Unknown;
      auto fd = RHS.mapping.find(pair.first);
      if (fd != RHS.mapping.end()) {
        other = fd->second;
      }
      changed = (pair.second &= other);
      if (pair.second == BaseType::Unknown) {
        keystodelete.push_back(pair.first);
      }
    }

    for (auto &key : keystodelete) {
      mapping.erase(key);
    }

    return changed;
  }

  /// Set this to the logical and of itself and RHS, returning whether this
  /// value changed If this and RHS are incompatible at an index, the result
  /// will be BaseType::Unknown
  bool operator&=(const TypeTree &RHS) { return andIn(RHS); }

  /// Set this to the logical `binop` of itself and RHS, using the Binop Op,
  /// returning true if this was changed.
  /// This function will error on an invalid type combination
  bool binopIn(const TypeTree &RHS, llvm::BinaryOperator::BinaryOps Op) {
    bool changed = false;

    std::vector<std::vector<int>> toErase;

    for (auto &pair : mapping) {
      // TODO propagate non-first level operands:
      // Special handling is necessary here because a pointer to an int
      // binop with something should not apply the binop rules to the
      // underlying data but instead a different rule
      if (pair.first.size() > 0) {
        toErase.push_back(pair.first);
        continue;
      }

      ConcreteType CT(pair.second);
      ConcreteType RightCT(BaseType::Unknown);

      // Mutual mappings
      auto found = RHS.mapping.find(pair.first);
      if (found != RHS.mapping.end()) {
        RightCT = found->second;
      }

      changed |= CT.binopIn(RightCT, Op);
      if (CT == BaseType::Unknown) {
        toErase.push_back(pair.first);
      } else {
        pair.second = CT;
      }
    }

    // mapings just on the right
    for (auto &pair : RHS.mapping) {
      // TODO propagate non-first level operands:
      // Special handling is necessary here because a pointer to an int
      // binop with something should not apply the binop rules to the
      // underlying data but instead a different rule
      if (pair.first.size() > 0) {
        continue;
      }

      if (mapping.find(pair.first) == RHS.mapping.end()) {
        ConcreteType CT = BaseType::Unknown;
        changed |= CT.binopIn(pair.second, Op);
        if (CT != BaseType::Unknown) {
          mapping.insert(std::make_pair(pair.first, CT));
        }
      }
    }

    for (auto vec : toErase) {
      mapping.erase(vec);
    }

    return changed;
  }

  /// Returns a string representation of this TypeTree
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
