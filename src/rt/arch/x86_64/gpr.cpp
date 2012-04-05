#include "gpr.h"

#define LOAD(rn) do { \
    uintptr_t tmp; \
    asm("movq %%" #rn ",%0" : "=r" (tmp) :); \
    this->rn = tmp; \
} while (0)

void rust_gpr::load() {
    LOAD(rax); LOAD(rbx); LOAD(rcx); LOAD(rdx);
    LOAD(rsi); LOAD(rdi); LOAD(rbp); LOAD(rsi);
    LOAD(r8);  LOAD(r9);  LOAD(r10); LOAD(r11);
    LOAD(r12); LOAD(r13); LOAD(r14); LOAD(r15);
}

