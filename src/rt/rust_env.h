// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#ifndef RUST_ENV_H
#define RUST_ENV_H

#include "rust_globals.h"

// Avoiding 'bool' type here since I'm not sure it has a standard size
typedef uint8_t rust_bool;

struct rust_env {
    size_t num_sched_threads;
    size_t min_stack_size;
    size_t max_stack_size;
    char* logspec;
    rust_bool detailed_leaks;
    char* rust_seed;
    rust_bool poison_on_free;
    int argc;
    char **argv;
    rust_bool debug_mem;
    rust_bool debug_borrow;
};

rust_env* load_env(int argc, char **argv);
void free_env(rust_env *rust_env);

#endif
