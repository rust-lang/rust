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

#![experimental]

// FIXME: this should not be here.
#![allow(missing_docs)]

#![allow(dead_code)]

use os;
use thunk::Thunk;
use kinds::Send;
use thread::Thread;
use ops::FnOnce;
use sys;
use sys_common;
use sys_common::thread_info::{mod, NewThread};

// Reexport some of our utilities which are expected by other crates.
pub use self::util::{default_sched_threads, min_stack, running_on_valgrind};
pub use self::unwind::{begin_unwind, begin_unwind_fmt};

// Reexport some functionality from liballoc.
pub use alloc::heap;

// Simple backtrace functionality (to print on panic)
pub mod backtrace;

// Internals
mod macros;

// These should be refactored/moved/made private over time
pub mod util;
pub mod unwind;
pub mod args;

mod at_exit_imp;
mod libunwind;

/// The default error code of the rust runtime if the main task panics instead
/// of exiting cleanly.
pub const DEFAULT_ERROR_CODE: int = 101;

#[cfg(any(windows, android))]
const OS_DEFAULT_STACK_ESTIMATE: uint = 1 << 20;
#[cfg(all(unix, not(android)))]
const OS_DEFAULT_STACK_ESTIMATE: uint = 2 * (1 << 20);

#[cfg(not(test))]
#[lang = "start"]
fn lang_start(main: *const u8, argc: int, argv: *const *const u8) -> int {
    use mem;
    use prelude::*;
    use rt;

    let something_around_the_top_of_the_stack = 1;
    let addr = &something_around_the_top_of_the_stack as *const int;
    let my_stack_top = addr as uint;

    // FIXME #11359 we just assume that this thread has a stack of a
    // certain size, and estimate that there's at most 20KB of stack
    // frames above our current position.
    let my_stack_bottom = my_stack_top + 20000 - OS_DEFAULT_STACK_ESTIMATE;

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
        thread_info::set((my_stack_bottom, my_stack_top),
                         sys::thread::guard::main(),
                         thread);

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
        os::get_exit_status()
    }
}

/// Enqueues a procedure to run when the runtime is cleaned up
///
/// The procedure passed to this function will be executed as part of the
/// runtime cleanup phase. For normal rust programs, this means that it will run
/// after all other tasks have exited.
///
/// The procedure is *not* executed with a local `Task` available to it, so
/// primitives like logging, I/O, channels, spawning, etc, are *not* available.
/// This is meant for "bare bones" usage to clean up runtime details, this is
/// not meant as a general-purpose "let's clean everything up" function.
///
/// It is forbidden for procedures to register more `at_exit` handlers when they
/// are running, and doing so will lead to a process abort.
pub fn at_exit<F:FnOnce()+Send>(f: F) {
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
    // FIXME: (#20012): the resources being cleaned up by at_exit
    // currently are not prepared for cleanup to happen asynchronously
    // with detached threads using the resources; for now, we leak.
    // at_exit_imp::cleanup();
}
