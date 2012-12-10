// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#include "rust_globals.h"
#include "rust_sched_driver.h"
#include "rust_sched_loop.h"

rust_sched_driver::rust_sched_driver(rust_sched_loop *sched_loop)
    : sched_loop(sched_loop),
      signalled(false) {

    assert(sched_loop != NULL);
    sched_loop->on_pump_loop(this);
}

/**
 * Starts the main scheduler loop which performs task scheduling for this
 * domain.
 *
 * Returns once no more tasks can be scheduled and all task ref_counts
 * drop to zero.
 */
void
rust_sched_driver::start_main_loop() {
    assert(sched_loop != NULL);

#ifdef __APPLE__
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "scheduler loop %d", sched_loop->get_id());
        // pthread_setname_np seems to have a different signature and
        // different behavior on different platforms. Thus, this is
        // only for Mac at the moment. There are equivalent versions
        // for Linux that we can add if needed.
        pthread_setname_np(buf);
    }
#endif

    rust_sched_loop_state state = sched_loop_state_keep_going;
    while (state != sched_loop_state_exit) {
        DLOG(sched_loop, dom, "pumping scheduler");
        state = sched_loop->run_single_turn();

        if (state == sched_loop_state_block) {
            scoped_lock with(lock);
            if (!signalled) {
                DLOG(sched_loop, dom, "blocking scheduler");
                lock.wait();
            }
            signalled = false;
        }
    }
}

void
rust_sched_driver::signal() {
    scoped_lock with(lock);
    signalled = true;
    lock.signal();
}
