//===- RustGCStrategy.cpp - Rust garbage collection strategy ----*- C++ -*-===
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===
//
// This file defines the garbage collection strategy for Rust.
//
//===----------------------------------------------------------------------===

#include "llvm/CodeGen/GCs.h"
#include "llvm/CodeGen/GCStrategy.h"

using namespace llvm;

class RustGCStrategy : public GCStrategy {
public:
  RustGCStrategy() {
    NeededSafePoints = 1 << GC::PostCall;
    UsesMetadata = true;
    InitRoots = false;  // LLVM crashes with this on due to bitcasts.
  }
};

static GCRegistry::Add<RustGCStrategy>
RustGCStrategyRegistration("rust", "Rust GC");


