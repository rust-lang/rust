#include "gpr.h"

#define LOAD(rn) do { \
    uintptr_t tmp; \
    asm("movl %%" #rn ",%0" : "=r" (tmp) :); \
    this->rn = tmp; \
} while (0)

void rust_gpr::load() {
    LOAD(eax); LOAD(ebx); LOAD(ecx); LOAD(edx);
    LOAD(esi); LOAD(edi); LOAD(ebp); LOAD(esi);
}

