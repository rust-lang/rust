// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Getting the stack pointer and getting/setting sp limit.

#ifndef SP_H
#define SP_H

#include "../../rust_globals.h"

// Gets a pointer to the vicinity of the current stack pointer
extern "C" uintptr_t get_sp();

// Gets the pointer to the end of the Rust stack from a platform-
// specific location in the thread control block
extern "C" CDECL uintptr_t get_sp_limit();

// Records the pointer to the end of the Rust stack in a platform-
// specific location in the thread control block
extern "C" CDECL void record_sp_limit(void *limit);

#endif
