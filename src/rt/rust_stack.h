#ifndef RUST_STACK_H
#define RUST_STACK_H

#include "memory_region.h"

struct stk_seg {
    stk_seg *prev;
    stk_seg *next;
    uintptr_t end;
    unsigned int valgrind_id;
#ifndef _LP64
    uint32_t pad;
#endif

    rust_task *task;
    uintptr_t canary;

    uint8_t data[];
};

stk_seg *
create_stack(memory_region *region, size_t sz);

void
destroy_stack(memory_region *region, stk_seg *stk);

// Must be called before each time a stack is reused to tell valgrind
// that the stack is accessible.
void
reuse_valgrind_stack(stk_seg *stk, uint8_t *sp);

// Run a sanity check
void
check_stack_canary(stk_seg *stk);

#endif /* RUST_STACK_H */
