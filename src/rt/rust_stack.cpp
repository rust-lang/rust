#include "rust_internal.h"

#include "vg/valgrind.h"
#include "vg/memcheck.h"

// A value that goes at the end of the stack and must not be touched
const uint8_t stack_canary[] = {0xAB, 0xCD, 0xAB, 0xCD,
                                0xAB, 0xCD, 0xAB, 0xCD,
                                0xAB, 0xCD, 0xAB, 0xCD,
                                0xAB, 0xCD, 0xAB, 0xCD};

void
config_valgrind_stack(stk_seg *stk) {
    stk->valgrind_id =
        VALGRIND_STACK_REGISTER(&stk->data[0],
                                stk->end);
#ifndef NVALGRIND
    // Establish that the stack is accessible.  This must be done when reusing
    // old stack segments, since the act of popping the stack previously
    // caused valgrind to consider the whole thing inaccessible.
    size_t sz = stk->end - (uintptr_t)&stk->data[0];
    VALGRIND_MAKE_MEM_UNDEFINED(stk->data + sizeof(stack_canary),
                                sz - sizeof(stack_canary));
#endif
}

void
unconfig_valgrind_stack(stk_seg *stk) {
    VALGRIND_STACK_DEREGISTER(stk->valgrind_id);
}

void
add_stack_canary(stk_seg *stk) {
    memcpy(stk->data, stack_canary, sizeof(stack_canary));
    assert(sizeof(stack_canary) == 16 && "Stack canary was not the expected size");
}

void
check_stack_canary(stk_seg *stk) {
    assert(!memcmp(stk->data, stack_canary, sizeof(stack_canary))
      && "Somebody killed the canary");
}
