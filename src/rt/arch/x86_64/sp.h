// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Getting the stack pointer and getting/setting sp limit.

#ifndef SP_H
#define SP_H

#include "../../rust_globals.h"

// Gets a pointer to the vicinity of the current stack pointer
extern "C" ALWAYS_INLINE uintptr_t get_sp() {
    uintptr_t sp;
    asm volatile (
        "movq %%rsp, %0"
        : "=m"(sp));
    return sp;
}

// Gets the pointer to the end of the Rust stack from a platform-
// specific location in the thread control block
extern "C" CDECL ALWAYS_INLINE uintptr_t get_sp_limit() {
    uintptr_t limit;

#if defined(__linux__)
    asm volatile (
        "movq %%fs:112, %0"
        : "=r"(limit));
#elif defined(__APPLE__)
    asm volatile (
        "movq $0x60+90*8, %%rsi\n\t"
        "movq %%gs:(%%rsi), %0"
        :  "=r"(limit)
        :: "rsi");
#elif defined(__FreeBSD__)
    asm volatile (
        "movq %%fs:24, %0"
        : "=r"(limit));
#elif defined(_WIN64)
    asm volatile (
        "movq %%gs:0x28, %0"
        : "=r"(limit));
#endif

    return limit;
}

// Records the pointer to the end of the Rust stack in a platform-
// specific location in the thread control block
extern "C" CDECL ALWAYS_INLINE void record_sp_limit(void *limit) {
#if defined(__linux__)
    asm volatile (
        "movq %0, %%fs:112"
        :: "r"(limit));
#elif defined(__APPLE__)
    asm volatile (
        "movq $0x60+90*8, %%rsi\n\t"
        "movq %0, %%gs:(%%rsi)"
        :: "r"(limit)
        :  "rsi");
#elif defined(__FreeBSD__)
    asm volatile (
        "movq %0, %%fs:24"
        :: "r"(limit));
#elif defined(_WIN64)
    asm volatile (
        "movq %0, %%gs:0x28"
        :: "r"(limit));
#endif
}

#endif
