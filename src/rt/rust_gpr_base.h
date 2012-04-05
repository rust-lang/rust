// Base class for architecture-specific general-purpose registers. This
// structure is used during stack crawling.

#ifndef GPR_BASE_H
#define GPR_BASE_H

#include <stdint.h>

class rust_gpr_base {
public:
    // Returns the value of a register by number.
    inline uintptr_t &get(uint32_t i) {
        return reinterpret_cast<uintptr_t *>(this)[i];
    }

    // Sets the value of a register by number.
    inline void set(uint32_t i, uintptr_t val) {
        reinterpret_cast<uintptr_t *>(this)[i] = val;
    }
};


#endif

