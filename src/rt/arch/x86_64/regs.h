// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is loosely kept in sync with src/libstd/rt/context.rs

#define RUSTRT_RBX   0
#define RUSTRT_RSP   1
#define RUSTRT_RBP   2
// RCX on Windows, RDI elsewhere
#define RUSTRT_ARG0  3
#define RUSTRT_R12   4
#define RUSTRT_R13   5
#define RUSTRT_R14   6
#define RUSTRT_R15   7
#define RUSTRT_IP    8
#if defined(__MINGW32__) || defined(_WINDOWS)
    #define RUSTRT_RDI   9
    #define RUSTRT_RSI   10
    #define RUSTRT_ST1   11
    #define RUSTRT_ST2   12
    #define RUSTRT_XMM6  14
    #define RUSTRT_XMM7  16
    #define RUSTRT_XMM8  18
    #define RUSTRT_XMM9  20
    #define RUSTRT_XMM10 22
    #define RUSTRT_XMM11 24
    #define RUSTRT_XMM12 26
    #define RUSTRT_XMM13 28
    #define RUSTRT_XMM14 30
    #define RUSTRT_XMM15 32
    #define RUSTRT_MAX   34
#else
    // Not used, just padding
    #define RUSTRT_XXX   9
    #define RUSTRT_XMM0 10
    #define RUSTRT_XMM1 12
    #define RUSTRT_XMM2 14
    #define RUSTRT_XMM3 16
    #define RUSTRT_XMM4 18
    #define RUSTRT_XMM5 20
    #define RUSTRT_MAX  22
#endif

// ARG0 is the register in which the first argument goes.
// Naturally this depends on your operating system.
#if defined(__MINGW32__) || defined(_WINDOWS)
#   define RUSTRT_ARG0_S %rcx
#   define RUSTRT_ARG1_S %rdx
#   define RUSTRT_ARG2_S %r8
#   define RUSTRT_ARG3_S %r9
#else
#   define RUSTRT_ARG0_S %rdi
#   define RUSTRT_ARG1_S %rsi
#   define RUSTRT_ARG2_S %rdx
#   define RUSTRT_ARG3_S %rcx
#   define RUSTRT_ARG4_S %r8
#   define RUSTRT_ARG5_S %r9
#endif
