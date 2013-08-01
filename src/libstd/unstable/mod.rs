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

pub mod dynamic_lib;

pub mod finally;
pub mod intrinsics;
pub mod simd;
pub mod extfmt;
#[cfg(not(test))]
pub mod lang;
pub mod sync;
pub mod atomics;
pub mod raw;

/**

Start a new thread outside of the current runtime context and wait
for it to terminate.

The executing thread has no access to a task pointer and will be using
a normal large stack.
*/
pub fn run_in_bare_thread(f: ~fn()) {
    use cell::Cell;
    use rt::thread::Thread;

    let f_cell = Cell::new(f);
    let (port, chan) = comm::stream();
    // FIXME #4525: Unfortunate that this creates an extra scheduler but it's
    // necessary since rust_raw_thread_join is blocking
    do task::spawn_sched(task::SingleThreaded) {
        Thread::start(f_cell.take()).join();
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


/// Changes the current working directory to the specified
/// path while acquiring a global lock, then calls `action`.
/// If the change is successful, releases the lock and restores the
/// CWD to what it was before, returning true.
/// Returns false if the directory doesn't exist or if the directory change
/// is otherwise unsuccessful.
///
/// This is used by test cases to avoid cwd races.
///
/// # Safety Note
///
/// This uses a pthread mutex so descheduling in the action callback
/// can lead to deadlock. Calling change_dir_locked recursively will
/// also deadlock.
pub fn change_dir_locked(p: &Path, action: &fn()) -> bool {
    use os;
    use os::change_dir;
    use unstable::sync::atomically;
    use unstable::finally::Finally;

    unsafe {
        // This is really sketchy. Using a pthread mutex so descheduling
        // in the `action` callback can cause deadlock. Doing it in
        // `task::atomically` to try to avoid that, but ... I don't know
        // this is all bogus.
        return do atomically {
            rust_take_change_dir_lock();

            do (||{
                let old_dir = os::getcwd();
                if change_dir(p) {
                    action();
                    change_dir(&old_dir)
                }
                else {
                    false
                }
            }).finally {
                rust_drop_change_dir_lock();
            }
        }
    }

    extern {
        fn rust_take_change_dir_lock();
        fn rust_drop_change_dir_lock();
    }
}
