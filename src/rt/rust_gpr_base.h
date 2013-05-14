// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
