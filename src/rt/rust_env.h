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

struct rust_env {
    size_t num_sched_threads;
    size_t min_stack_size;
    size_t max_stack_size;
    char* logspec;
    bool detailed_leaks;
    char* rust_seed;
    bool poison_on_free;
    int argc;
    char **argv;
};

rust_env* load_env(int argc, char **argv);
void free_env(rust_env *rust_env);

#endif
