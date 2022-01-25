//===- PreserveNVVM.cpp - Mark NVVM attributes for preservation.  -------===//
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
// This file contains createPreserveNVVM, a transformation pass that marks
// calls to __nv_* functions, marking them as noinline as implementing the llvm
// intrinsic.
//
//===----------------------------------------------------------------------===//
#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/Function.h"

#include "llvm/Pass.h"

#include "llvm/Transforms/Utils.h"

#include <map>

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "preserve-nvvm"

namespace {

class PreserveNVVM : public FunctionPass {
public:
  static char ID;
  bool Begin;
  PreserveNVVM(bool Begin = true) : FunctionPass(ID), Begin(Begin) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {}

  bool runOnFunction(Function &F) override {
    bool changed = false;
    std::map<std::pair<Type *, std::string>,
             std::pair<std::string, std::string>>
        Implements;
    for (Type *T : {Type::getFloatTy(F.getContext()),
                    Type::getDoubleTy(F.getContext())}) {
      // sincos, sinpi, cospi, sincospi, cyl_bessel_i1
      for (std::string name :
           {"sin",        "cos",     "tan",       "log2",   "exp",    "exp2",
            "exp10",      "cosh",    "sinh",      "tanh",   "atan2",  "atan",
            "asin",       "acos",    "log",       "log10",  "log1p",  "acosh",
            "asinh",      "atanh",   "expm1",     "hypot",  "rhypot", "norm3d",
            "rnorm3d",    "norm4d",  "rnorm4d",   "norm",   "rnorm",  "cbrt",
            "rcbrt",      "j0",      "j1",        "y0",     "y1",     "yn",
            "jn",         "erf",     "erfinv",    "erfc",   "erfcx",  "erfcinv",
            "normcdfinv", "normcdf", "lgamma",    "ldexp",  "scalbn", "frexp",
            "modf",       "fmod",    "remainder", "remquo", "powi",   "tgamma",
            "round",      "fdim",    "ilogb",     "logb",   "isinf",  "pow",
            "sqrt"}) {
        std::string nvname = "__nv_" + name;
        std::string llname = "llvm." + name + ".";

        if (T->isFloatTy()) {
          nvname += "f";
          llname += "f32";
        } else {
          llname += "f64";
        }

        Implements[std::make_pair(T, nvname)] = std::make_pair(name, llname);
      }
    }
    auto idx = std::make_pair(F.getReturnType(), F.getName().str());
    auto found = Implements.find(idx);
    if (found != Implements.end()) {
      if (Begin) {
        F.removeFnAttr(Attribute::AlwaysInline);
        F.addFnAttr(Attribute::NoInline);
        // As a side effect, enforces arguments
        // cannot be erased.
        F.setLinkage(Function::LinkageTypes::ExternalLinkage);
        F.addFnAttr("implements", found->second.second);
        F.addFnAttr("enzyme_math", found->second.first);
        changed = true;
      } else {
        F.addFnAttr(Attribute::AlwaysInline);
        F.removeFnAttr(Attribute::NoInline);
        F.setLinkage(Function::LinkageTypes::InternalLinkage);
      }
    }
    return changed;
  }
};

} // namespace

char PreserveNVVM::ID = 0;

static RegisterPass<PreserveNVVM> X("preserve-nvvm", "Preserve NVVM Pass");

FunctionPass *createPreserveNVVMPass(bool Begin) {
  return new PreserveNVVM(Begin);
}

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include "llvm/IR/LegacyPassManager.h"

extern "C" void AddPreserveNVVMPass(LLVMPassManagerRef PM, uint8_t Begin) {
  unwrap(PM)->add(createPreserveNVVMPass((bool)Begin));
}
