// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#include "rust_kernel.h"
#include "rust_sched_reaper.h"

// NB: We're using a very small stack here
const size_t STACK_SIZE = 1024*20;

rust_sched_reaper::rust_sched_reaper(rust_kernel *kernel)
    : rust_thread(STACK_SIZE), kernel(kernel) {
}

void
rust_sched_reaper::run() {
    kernel->wait_for_schedulers();
}
