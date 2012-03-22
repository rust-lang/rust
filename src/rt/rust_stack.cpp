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
reuse_valgrind_stack(stk_seg *stk, uint8_t *sp) {
    // Establish that the stack is accessible.  This must be done when reusing
    // old stack segments, since the act of popping the stack previously
    // caused valgrind to consider the whole thing inaccessible.
    assert(sp >= stk->data && sp <= (uint8_t*) stk->end
	   && "Stack pointer must be inside stack segment");
    size_t sz = stk->end - (uintptr_t)sp;
    (void) VALGRIND_MAKE_MEM_UNDEFINED(sp, sz);
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

stk_seg *
create_stack(memory_region *region, size_t sz) {
    size_t total_sz = sizeof(stk_seg) + sz;
    stk_seg *stk = (stk_seg *)region->malloc(total_sz, "stack", false);
    memset(stk, 0, sizeof(stk_seg));
    stk->end = (uintptr_t) &stk->data[sz];
    add_stack_canary(stk);
    register_valgrind_stack(stk);
    return stk;
}

void
destroy_stack(memory_region *region, stk_seg *stk) {
    deregister_valgrind_stack(stk);
    region->free(stk);
}
