// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Runtime services, including the task scheduler and I/O dispatcher
//!
//! The `rt` module provides the private runtime infrastructure necessary to support core language
//! features like the exchange and local heap, logging, local data and unwinding. It also
//! implements the default task scheduler and task model. Initialization routines are provided for
//! setting up runtime resources in common configurations, including that used by `rustc` when
//! generating executables.
//!
//! It is intended that the features provided by `rt` can be factored in a way such that the core
//! library can be built with different 'profiles' for different use cases, e.g. excluding the task
//! scheduler. A number of runtime features though are critical to the functioning of the language
//! and an implementation must be provided regardless of the execution environment.
//!
//! Of foremost importance is the global exchange heap, in the module `heap`. Very little practical
//! Rust code can be written without access to the global heap. Unlike most of `rt` the global heap
//! is truly a global resource and generally operates independently of the rest of the runtime.
//!
//! All other runtime features are task-local, including the local heap, local storage, logging and
//! the stack unwinder.
//!
//! The relationship between `rt` and the rest of the core library is not entirely clear yet and
//! some modules will be moving into or out of `rt` as development proceeds.
//!
//! Several modules in `core` are clients of `rt`:
//!
//! * `std::task` - The user-facing interface to the Rust task model.
//! * `std::local_data` - The interface to local data.
//! * `std::unstable::lang` - Miscellaneous lang items, some of which rely on `std::rt`.
//! * `std::cleanup` - Local heap destruction.
//! * `std::io` - In the future `std::io` will use an `rt` implementation.
//! * `std::logging`
//! * `std::comm`

#![experimental]

// FIXME: this should not be here.
#![allow(missing_docs)]

#![allow(dead_code)]

use borrow::IntoCow;
use failure;
use os;
use thunk::Thunk;
use kinds::Send;
use sys_common;

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
pub mod mutex;
pub mod thread;
pub mod exclusive;
pub mod util;
pub mod bookkeeping;
pub mod local;
pub mod task;
pub mod unwind;

mod args;
mod at_exit_imp;
mod libunwind;
mod local_ptr;
mod thread_local_storage;

/// The default error code of the rust runtime if the main task panics instead
/// of exiting cleanly.
pub const DEFAULT_ERROR_CODE: int = 101;

/// One-time runtime initialization.
///
/// Initializes global state, including frobbing
/// the crate's logging flags, registering GC
/// metadata, and storing the process arguments.
#[allow(experimental)]
pub fn init(argc: int, argv: *const *const u8) {
    // FIXME: Derefing these pointers is not safe.
    // Need to propagate the unsafety to `start`.
    unsafe {
        args::init(argc, argv);
        local_ptr::init();
        thread::init();
        unwind::register(failure::on_fail);
    }
}

#[cfg(any(windows, android))]
static OS_DEFAULT_STACK_ESTIMATE: uint = 1 << 20;
#[cfg(all(unix, not(android)))]
static OS_DEFAULT_STACK_ESTIMATE: uint = 2 * (1 << 20);

#[cfg(not(test))]
#[lang = "start"]
fn lang_start(main: *const u8, argc: int, argv: *const *const u8) -> int {
    use mem;
    start(argc, argv, Thunk::new(move|| {
        let main: extern "Rust" fn() = unsafe { mem::transmute(main) };
        main();
    }))
}

/// Executes the given procedure after initializing the runtime with the given
/// argc/argv.
///
/// This procedure is guaranteed to run on the thread calling this function, but
/// the stack bounds for this rust task will *not* be set. Care must be taken
/// for this function to not overflow its stack.
///
/// This function will only return once *all* native threads in the system have
/// exited.
pub fn start(argc: int, argv: *const *const u8, main: Thunk) -> int {
    use prelude::*;
    use rt;
    use rt::task::Task;
    use str;

    let something_around_the_top_of_the_stack = 1;
    let addr = &something_around_the_top_of_the_stack as *const int;
    let my_stack_top = addr as uint;

    // FIXME #11359 we just assume that this thread has a stack of a
    // certain size, and estimate that there's at most 20KB of stack
    // frames above our current position.
    let my_stack_bottom = my_stack_top + 20000 - OS_DEFAULT_STACK_ESTIMATE;

    // When using libgreen, one of the first things that we do is to turn off
    // the SIGPIPE signal (set it to ignore). By default, some platforms will
    // send a *signal* when a EPIPE error would otherwise be delivered. This
    // runtime doesn't install a SIGPIPE handler, causing it to kill the
    // program, which isn't exactly what we want!
    //
    // Hence, we set SIGPIPE to ignore when the program starts up in order to
    // prevent this problem.
    #[cfg(windows)] fn ignore_sigpipe() {}
    #[cfg(unix)] fn ignore_sigpipe() {
        use libc;
        use libc::funcs::posix01::signal::signal;
        unsafe {
            assert!(signal(libc::SIGPIPE, libc::SIG_IGN) != -1);
        }
    }
    ignore_sigpipe();

    init(argc, argv);
    let mut exit_code = None;
    let mut main = Some(main);
    let mut task = box Task::new(Some((my_stack_bottom, my_stack_top)),
                                 Some(rt::thread::main_guard_page()));
    task.name = Some(str::Slice("<main>"));
    drop(task.run(|| {
        unsafe {
            sys_common::stack::record_os_managed_stack_bounds(my_stack_bottom, my_stack_top);
        }
        (main.take().unwrap()).invoke(());
        exit_code = Some(os::get_exit_status());
    }).destroy());
    unsafe { cleanup(); }
    // If the exit code wasn't set, then the task block must have panicked.
    return exit_code.unwrap_or(rt::DEFAULT_ERROR_CODE);
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
pub fn at_exit(f: proc():Send) {
    at_exit_imp::push(f);
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
    bookkeeping::wait_for_other_tasks();
    args::cleanup();
    thread::cleanup();
    local_ptr::cleanup();
}

// FIXME: these probably shouldn't be public...
#[doc(hidden)]
pub mod shouldnt_be_public {
    #[cfg(not(test))]
    pub use super::local_ptr::native::maybe_tls_key;
    #[cfg(all(not(windows), not(target_os = "android"), not(target_os = "ios")))]
    pub use super::local_ptr::compiled::RT_TLS_PTR;
}
