// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Runtime services
//!
//! The `rt` module provides a narrow set of runtime services,
//! including the global heap (exported in `heap`) and unwinding and
//! backtrace support. The APIs in this module are highly unstable,
//! and should be considered as private implementation details for the
//! time being.

#![unstable(feature = "std_misc")]

// FIXME: this should not be here.
#![allow(missing_docs)]

#![allow(dead_code)]

use marker::Send;
use ops::FnOnce;
use sys;
use thunk::Thunk;
use usize;

// Reexport some of our utilities which are expected by other crates.
pub use self::util::{default_sched_threads, min_stack, running_on_valgrind};
pub use self::unwind::{begin_unwind, begin_unwind_fmt};

// Reexport some functionality from liballoc.
pub use alloc::heap;

// Simple backtrace functionality (to print on panic)
pub mod backtrace;

// Internals
#[macro_use]
mod macros;

// These should be refactored/moved/made private over time
pub mod util;
pub mod unwind;
pub mod args;

mod at_exit_imp;
mod libunwind;

/// The default error code of the rust runtime if the main thread panics instead
/// of exiting cleanly.
pub const DEFAULT_ERROR_CODE: int = 101;

#[cfg(any(windows, android))]
const OS_DEFAULT_STACK_ESTIMATE: uint = 1 << 20;
#[cfg(all(unix, not(android)))]
const OS_DEFAULT_STACK_ESTIMATE: uint = 2 * (1 << 20);

#[cfg(not(test))]
#[lang = "start"]
fn lang_start(main: *const u8, argc: int, argv: *const *const u8) -> int {
    use prelude::v1::*;

    use mem;
    use env;
    use rt;
    use sys_common::thread_info::{self, NewThread};
    use sys_common;
    use thread::Thread;

    let something_around_the_top_of_the_stack = 1;
    let addr = &something_around_the_top_of_the_stack as *const int;
    let my_stack_top = addr as uint;

    // FIXME #11359 we just assume that this thread has a stack of a
    // certain size, and estimate that there's at most 20KB of stack
    // frames above our current position.
    const TWENTY_KB: uint = 20000;

    // saturating-add to sidestep overflow
    let top_plus_spill = if usize::MAX - TWENTY_KB < my_stack_top {
        usize::MAX
    } else {
        my_stack_top + TWENTY_KB
    };
    // saturating-sub to sidestep underflow
    let my_stack_bottom = if top_plus_spill < OS_DEFAULT_STACK_ESTIMATE {
        0
    } else {
        top_plus_spill - OS_DEFAULT_STACK_ESTIMATE
    };

    let failed = unsafe {
        // First, make sure we don't trigger any __morestack overflow checks,
        // and next set up our stack to have a guard page and run through our
        // own fault handlers if we hit it.
        sys_common::stack::record_os_managed_stack_bounds(my_stack_bottom,
                                                          my_stack_top);
        sys::thread::guard::init();
        sys::stack_overflow::init();

        // Next, set up the current Thread with the guard information we just
        // created. Note that this isn't necessary in general for new threads,
        // but we just do this to name the main thread and to give it correct
        // info about the stack bounds.
        let thread: Thread = NewThread::new(Some("<main>".to_string()));
        thread_info::set(sys::thread::guard::main(), thread);

        // By default, some platforms will send a *signal* when a EPIPE error
        // would otherwise be delivered. This runtime doesn't install a SIGPIPE
        // handler, causing it to kill the program, which isn't exactly what we
        // want!
        //
        // Hence, we set SIGPIPE to ignore when the program starts up in order
        // to prevent this problem.
        #[cfg(windows)] fn ignore_sigpipe() {}
        #[cfg(unix)] fn ignore_sigpipe() {
            use libc;
            use libc::funcs::posix01::signal::signal;
            unsafe {
                assert!(signal(libc::SIGPIPE, libc::SIG_IGN) != -1);
            }
        }
        ignore_sigpipe();

        // Store our args if necessary in a squirreled away location
        args::init(argc, argv);

        // And finally, let's run some code!
        let res = unwind::try(|| {
            let main: fn() = mem::transmute(main);
            main();
        });
        cleanup();
        res.is_err()
    };

    // If the exit code wasn't set, then the try block must have panicked.
    if failed {
        rt::DEFAULT_ERROR_CODE
    } else {
        env::get_exit_status() as isize
    }
}

/// Enqueues a procedure to run when the main thread exits.
///
/// It is forbidden for procedures to register more `at_exit` handlers when they
/// are running, and doing so will lead to a process abort.
///
/// Note that other threads may still be running when `at_exit` routines start
/// running.
pub fn at_exit<F: FnOnce() + Send + 'static>(f: F) {
    at_exit_imp::push(Thunk::new(f));
}

/// One-time runtime cleanup.
///
/// This function is unsafe because it performs no checks to ensure that the
/// runtime has completely ceased running. It is the responsibility of the
/// caller to ensure that the runtime is entirely shut down and nothing will be
/// poking around at the internal components.
///
/// Invoking cleanup while portions of the runtime are still in use may cause
/// undefined behavior.
pub unsafe fn cleanup() {
    args::cleanup();
    sys::stack_overflow::cleanup();
    at_exit_imp::cleanup();
}
