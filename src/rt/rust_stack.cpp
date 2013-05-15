// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#include "rust_stack.h"
#include "vg/valgrind.h"
#include "vg/memcheck.h"

#include <cstdio>

#ifdef _LP64
const uintptr_t canary_value = 0xABCDABCDABCDABCD;
#else
const uintptr_t canary_value = 0xABCDABCD;
#endif

void
register_valgrind_stack(stk_seg *stk) {
    stk->valgrind_id =
        VALGRIND_STACK_REGISTER(&stk->data[0],
                                stk->end);
}

void
reuse_valgrind_stack(stk_seg *stk, uint8_t *sp) {
    // Establish that the stack is accessible.  This must be done when reusing
    // old stack segments, since the act of popping the stack previously
    // caused valgrind to consider the whole thing inaccessible.
    assert(sp >= stk->data && sp <= (uint8_t*) stk->end
           && "Stack pointer must be inside stack segment");
    size_t sz = stk->end - (uintptr_t)sp;
    (void) VALGRIND_MAKE_MEM_UNDEFINED(sp, sz);
    (void) sz;
}

void
deregister_valgrind_stack(stk_seg *stk) {
    VALGRIND_STACK_DEREGISTER(stk->valgrind_id);
}

void
add_stack_canary(stk_seg *stk) {
    stk->canary = canary_value;
}

void
check_stack_canary(stk_seg *stk) {
    assert(stk->canary == canary_value && "Somebody killed the canary");
}

// XXX: Duplication here between the local and exchange heap constructors

stk_seg *
create_stack(memory_region *region, size_t sz) {
    size_t total_sz = sizeof(stk_seg) + sz;
    stk_seg *stk = (stk_seg *)region->malloc(total_sz, "stack");
    memset(stk, 0, sizeof(stk_seg));
    stk->end = (uintptr_t) &stk->data[sz];
    stk->is_big = 0;
    add_stack_canary(stk);
    register_valgrind_stack(stk);
    return stk;
}

void
destroy_stack(memory_region *region, stk_seg *stk) {
    deregister_valgrind_stack(stk);
    region->free(stk);
}

stk_seg *
create_exchange_stack(rust_exchange_alloc *exchange, size_t sz) {
    size_t total_sz = sizeof(stk_seg) + sz;
    stk_seg *stk = (stk_seg *)exchange->malloc(total_sz);
    memset(stk, 0, sizeof(stk_seg));
    stk->end = (uintptr_t) &stk->data[sz];
    stk->is_big = 0;
    add_stack_canary(stk);
    register_valgrind_stack(stk);
    return stk;
}

void
destroy_exchange_stack(rust_exchange_alloc *exchange, stk_seg *stk) {
    deregister_valgrind_stack(stk);
    exchange->free(stk);
}


extern "C" CDECL unsigned int
rust_valgrind_stack_register(void *start, void *end) {
  return VALGRIND_STACK_REGISTER(start, end);
}

extern "C" CDECL void
rust_valgrind_stack_deregister(unsigned int id) {
  VALGRIND_STACK_DEREGISTER(id);
}
