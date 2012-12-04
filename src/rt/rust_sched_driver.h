// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#ifndef RUST_SCHED_DRIVER_H
#define RUST_SCHED_DRIVER_H

#include "sync/lock_and_signal.h"
#include "rust_signal.h"

struct rust_sched_loop;

class rust_sched_driver : public rust_signal {
private:
    rust_sched_loop *sched_loop;
    lock_and_signal lock;
    bool signalled;

public:
    rust_sched_driver(rust_sched_loop *sched_loop);

    void start_main_loop();

    virtual void signal();
};

#endif /* RUST_SCHED_DRIVER_H */
