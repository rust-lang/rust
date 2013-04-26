// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#ifndef RUST_STACK_H
#define RUST_STACK_H

#include "rust_globals.h"
#include "rust_exchange_alloc.h"
#include "memory_region.h"

struct rust_task;

struct stk_seg {
    stk_seg *prev;
    stk_seg *next;
    uintptr_t end;
    unsigned int valgrind_id;
    uint8_t is_big;

    rust_task *task;
    uintptr_t canary;

    uint8_t data[];
};

stk_seg *
create_stack(memory_region *region, size_t sz);

void
destroy_stack(memory_region *region, stk_seg *stk);

stk_seg *
create_exchange_stack(rust_exchange_alloc *exchange, size_t sz);

void
destroy_exchange_stack(rust_exchange_alloc *exchange, stk_seg *stk);

// Must be called before each time a stack is reused to tell valgrind
// that the stack is accessible.
void
reuse_valgrind_stack(stk_seg *stk, uint8_t *sp);

// Run a sanity check
void
check_stack_canary(stk_seg *stk);

#endif /* RUST_STACK_H */
