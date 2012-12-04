// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#ifndef RUST_SCHED_REAPER_H
#define RUST_SCHED_REAPER_H

#include "sync/rust_thread.h"

class rust_kernel;

/* Responsible for joining with rust_schedulers */
class rust_sched_reaper : public rust_thread {
private:
    rust_kernel *kernel;
public:
    rust_sched_reaper(rust_kernel *kernel);
    virtual void run();
};

#endif /* RUST_SCHED_REAPER_H */
