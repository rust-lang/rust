// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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
use libc;
use prelude::*;
use ptr;
use task;

pub mod at_exit;
pub mod global;
pub mod finally;
pub mod weak_task;
pub mod intrinsics;
pub mod simd;
pub mod extfmt;
#[cfg(not(test))]
pub mod lang;
pub mod sync;
pub mod atomics;

/**

Start a new thread outside of the current runtime context and wait
for it to terminate.

The executing thread has no access to a task pointer and will be using
a normal large stack.
*/
pub fn run_in_bare_thread(f: ~fn()) {
    let (port, chan) = comm::stream();
    // FIXME #4525: Unfortunate that this creates an extra scheduler but it's
    // necessary since rust_raw_thread_join_delete is blocking
    do task::spawn_sched(task::SingleThreaded) {
        unsafe {
            let closure: &fn() = || {
                f()
            };
            let thread = rust_raw_thread_start(&closure);
            rust_raw_thread_join_delete(thread);
            chan.send(());
        }
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

#[allow(non_camel_case_types)] // runtime type
pub type raw_thread = libc::c_void;

extern {
    fn rust_raw_thread_start(f: &(&fn())) -> *raw_thread;
    fn rust_raw_thread_join_delete(thread: *raw_thread);
}
