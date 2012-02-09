#ifndef RUST_STACK_H
#define RUST_STACK_H

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

template <class T>
stk_seg *
create_stack(T allocer, size_t sz) {
  size_t total_sz = sizeof(stk_seg) + sz;
  stk_seg *stk = (stk_seg *)allocer->malloc(total_sz, "stack");
  memset(stk, 0, sizeof(stk_seg));
  stk->end = (uintptr_t) &stk->data[sz];
  return stk;
}

template <class T>
void
destroy_stack(T allocer, stk_seg *stk) {
  allocer->free(stk);
}

void
config_valgrind_stack(stk_seg *stk);

void
unconfig_valgrind_stack(stk_seg *stk);

void
add_stack_canary(stk_seg *stk);

void
check_stack_canary(stk_seg *stk);

#endif /* RUST_STACK_H */
