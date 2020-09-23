//===- TypeTree.cpp - Implementation of Type Analysis Type Trees-----------===//
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
// This file contains the implementation TypeTrees -- a class
// representing all of the underlying types of a particular LLVM value. This
// consists of a map of memory offsets to an underlying ConcreteType. This
// permits TypeTrees to represent distinct underlying types at different
// locations. Presently, TypeTree's have both a fixed depth of memory lookups
// and a maximum offset to ensure that Type Analysis eventually terminates.
// In the future this should be modified to better represent recursive types
// rather than limiting the depth.
//
//===----------------------------------------------------------------------===//
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"

#include "TypeTree.h"

using namespace llvm;

// TODO keep type information that is striated
// e.g. if you have an i8* [0:Int, 8:Int] => i64* [0:Int, 1:Int]
// After a depth len into the index tree, prune any lookups that are not {0} or
// {-1} Todo handle {double}** to double** where there is a 0 removed
TypeTree TypeTree::KeepForCast(const llvm::DataLayout &dl, llvm::Type *from,
                                 llvm::Type *to) const {
  assert(from);
  assert(to);

  bool fromOpaque = isa<StructType>(from) && cast<StructType>(from)->isOpaque();
  bool toOpaque = isa<StructType>(to) && cast<StructType>(to)->isOpaque();

  TypeTree vd;

  for (auto &pair : mapping) {

    TypeTree vd2;

    // llvm::errs() << " considering casting from " << *from << " to " << *to <<
    // " fromidx: " << to_string(pair.first) << " dt:" << pair.second.str() << "
    // fromsize: " << fromsize << " tosize: " << tosize << "\n";

    if (pair.first.size() == 0) {
      vd2.insert(pair.first, pair.second);
      goto add;
    }

    if (!fromOpaque && !toOpaque) {
      uint64_t fromsize = (dl.getTypeSizeInBits(from) + 7) / 8;
      if (fromsize == 0)
        llvm::errs() << "from: " << *from << "\n";
      assert(fromsize > 0);
      uint64_t tosize = (dl.getTypeSizeInBits(to) + 7) / 8;
      assert(tosize > 0);

      // If the sizes are the same, whatever the original one is okay [ since
      // tomemory[ i*sizeof(from) ] indeed the start of an object of type to
      // since tomemory is "aligned" to type to
      if (fromsize == tosize) {
        vd2.insert(pair.first, pair.second);
        goto add;
      }

      // If the offset doesn't leak into a later element, we're fine to include
      if (pair.first[0] != -1 && (uint64_t)pair.first[0] < tosize) {
        vd2.insert(pair.first, pair.second);
        goto add;
      }

      if (pair.first[0] != -1) {
        vd.insert(pair.first, pair.second);
        goto add;
      } else {
        // pair.first[0] == -1

        if (fromsize < tosize) {
          if (tosize % fromsize == 0) {
            // TODO should really be at each offset do a -1
            vd.insert(pair.first, pair.second);
            goto add;
          } else {
            auto tmp(pair.first);
            tmp[0] = 0;
            vd.insert(tmp, pair.second);
            goto add;
          }
        } else {
          // fromsize > tosize
          // TODO should really insert all indices which are multiples of
          // fromsize
          auto tmp(pair.first);
          tmp[0] = 0;
          vd.insert(tmp, pair.second);
          goto add;
        }
      }
    }

    continue;
  add:;
    vd |= vd2;
  }
  return vd;
}