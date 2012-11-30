// General-purpose registers. This structure is used during stack crawling.

#ifndef GPR_H
#define GPR_H

#include "rust_gpr_base.h"

class rust_gpr : public rust_gpr_base {
public:
    uintptr_t r0, r1, r2, r3, r4, r5, r6, r7;
    uintptr_t  r8,  r9, r10, r11, r12, r13, r14, r15;

    inline uintptr_t get_fp() { return r11; }
    inline uintptr_t get_ip() { return r12; }

    inline void set_fp(uintptr_t new_fp) { r11 = new_fp; }
    inline void set_ip(uintptr_t new_ip) { r12 = new_ip; }

    void load();
};

#endif

