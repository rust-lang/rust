// General-purpose registers. This structure is used during stack crawling.

#ifndef GPR_H
#define GPR_H

#include "rust_gpr_base.h"

class rust_gpr : public rust_gpr_base {
public:
    uintptr_t eax, ebx, ecx, edx, esi, edi, ebp, eip;

    inline uintptr_t get_fp() { return ebp; }
    inline uintptr_t get_ip() { return eip; }

    inline void set_fp(uintptr_t new_fp) { ebp = new_fp; }
    inline void set_ip(uintptr_t new_ip) { eip = new_ip; }

    void load();
};

#endif

