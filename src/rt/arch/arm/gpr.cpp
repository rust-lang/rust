// xfail-license

#include "gpr.h"

#define LOAD(rn) do { \
    uintptr_t tmp; \
    asm("mov %%" #rn ",%0" : "=r" (tmp) :); \
    this->rn = tmp; \
} while (0)

void rust_gpr::load() {
    LOAD(r0); LOAD(r1); LOAD(r2); LOAD(r3);
    LOAD(r4); LOAD(r5); LOAD(r6); LOAD(r7);
    LOAD(r8);  LOAD(r9);  LOAD(r10); LOAD(r11);
    LOAD(r12); LOAD(r13); LOAD(r14); LOAD(r15);
}
