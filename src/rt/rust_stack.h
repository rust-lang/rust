struct stk_seg {
    stk_seg *prev;
    stk_seg *next;
    uintptr_t end;
    unsigned int valgrind_id;
#ifndef _LP64
    uint32_t pad;
#endif

    uint8_t data[];
};

void
config_valgrind_stack(stk_seg *stk);

void
unconfig_valgrind_stack(stk_seg *stk);

void
add_stack_canary(stk_seg *stk);

void
check_stack_canary(stk_seg *stk);
