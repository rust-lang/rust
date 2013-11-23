// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[doc(hidden)];

use comm::{GenericChan, GenericPort};
use comm;
use prelude::*;
use task;
use libc::uintptr_t;

pub mod dynamic_lib;

pub mod finally;
pub mod intrinsics;
pub mod simd;
#[cfg(not(test))]
pub mod lang;
pub mod sync;
pub mod mutex;
pub mod atomics;
pub mod raw;

/**

Start a new thread outside of the current runtime context and wait
for it to terminate.

The executing thread has no access to a task pointer and will be using
a normal large stack.
*/
pub fn run_in_bare_thread(f: proc()) {
    use rt::thread::Thread;

    let (port, chan) = comm::stream();
    // FIXME #4525: Unfortunate that this creates an extra scheduler but it's
    // necessary since rust_raw_thread_join is blocking
    do task::spawn_sched(task::SingleThreaded) {
        Thread::start(f).join();
        chan.send(());
    }
    port.recv();
}

#[test]
fn test_run_in_bare_thread() {
    let i = 100;
    do run_in_bare_thread {
        assert_eq!(i, 100);
    }
}

#[test]
fn test_run_in_bare_thread_exchange() {
    // Does the exchange heap work without the runtime?
    let i = ~100;
    do run_in_bare_thread {
        assert!(i == ~100);
    }
}

/// Dynamically inquire about whether we're running under V.
/// You should usually not use this unless your test definitely
/// can't run correctly un-altered. Valgrind is there to help
/// you notice weirdness in normal, un-doctored code paths!
pub fn running_on_valgrind() -> bool {
    unsafe { rust_running_on_valgrind() != 0 }
}

extern {
    fn rust_running_on_valgrind() -> uintptr_t;
}
