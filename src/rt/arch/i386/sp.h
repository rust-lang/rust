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
        "movl %%esp, %0"
        : "=m"(sp));
    return sp;
}

// Gets the pointer to the end of the Rust stack from a platform-
// specific location in the thread control block
extern "C" CDECL ALWAYS_INLINE uintptr_t get_sp_limit() {
    uintptr_t limit;

#if defined(__linux__) || defined(__FreeBSD__)
    asm volatile (
        "movl %%gs:48, %0"
        : "=r"(limit));
#elif defined(__APPLE__)
    asm volatile (
        "movl $0x48+90*4, %%ecx\n\t"
        "movl %%gs:(%%ecx), %0"
        :  "=r"(limit)
        :: "ecx");
#elif defined(_WIN32)
    asm volatile (
        "movl %%fs:0x14, %0"
        : "=r"(limit));
#endif

    return limit;
}

// Records the pointer to the end of the Rust stack in a platform-
// specific location in the thread control block
extern "C" CDECL ALWAYS_INLINE void record_sp_limit(void *limit) {
#if defined(__linux__) || defined(__FreeBSD__)
    asm volatile (
        "movl %0, %%gs:48"
        :: "r"(limit));
#elif defined(__APPLE__)
    asm volatile (
        "movl $0x48+90*4, %%eax\n\t"
        "movl %0, %%gs:(%%eax)"
        :: "r"(limit)
        :  "eax");
#elif defined(_WIN32)
    asm volatile (
        "movl %0, %%fs:0x14"
        :: "r"(limit));
#endif
}

#endif
