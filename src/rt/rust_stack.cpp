#include "rust_internal.h"

#include "vg/valgrind.h"
#include "vg/memcheck.h"

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
prepare_valgrind_stack(stk_seg *stk) {
#ifndef NVALGRIND
    // Establish that the stack is accessible.  This must be done when reusing
    // old stack segments, since the act of popping the stack previously
    // caused valgrind to consider the whole thing inaccessible.
    size_t sz = stk->end - (uintptr_t)&stk->data[0];
    VALGRIND_MAKE_MEM_UNDEFINED(stk->data, sz);
#endif
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
