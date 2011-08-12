//===- RustGCStrategy.cpp - Rust garbage collection strategy ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the garbage collection strategy for Rust.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCs.h"
#include "llvm/CodeGen/GCStrategy.h"

using namespace llvm;

namespace {
  class RustGCStrategy : public GCStrategy {
  public:
    RustGCStrategy();
  };
}

static GCRegistry::Add<RustGCStrategy>
X("rust", "Rust GC");

RustGCStrategy::RustGCStrategy() {
  NeededSafePoints = 1 << GC::PostCall;
  UsesMetadata = true;
}


