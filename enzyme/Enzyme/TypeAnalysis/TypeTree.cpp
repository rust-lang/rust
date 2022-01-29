//===- TypeTree.cpp - Implementation of Type Analysis Type Trees-----------===//
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

#include "llvm/Support/CommandLine.h"

#include "TypeTree.h"

using namespace llvm;

extern "C" {
/// Maximum offset for type trees to keep
llvm::cl::opt<int> MaxTypeOffset("enzyme-max-type-offset", cl::init(500),
                                 cl::Hidden,
                                 cl::desc("Maximum type tree offset"));
llvm::cl::opt<bool> EnzymeTypeWarning("enzyme-type-warning", cl::init(true),
                                      cl::Hidden,
                                      cl::desc("Print Type Depth Warning"));
}
