// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The native I/O and threading crate
//!
//! This crate contains an implementation of 1:1 scheduling for a "native"
//! runtime. In addition, all I/O provided by this crate is the thread blocking
//! version of I/O.
//!
//! # Starting with libnative
//!
//! ```rust
//! extern crate native;
//!
//! #[start]
//! fn start(argc: int, argv: **u8) -> int { native::start(argc, argv, main) }
//!
//! fn main() {
//!     // this code is running on the main OS thread
//! }
//! ```
//!
//! # Force spawning a native task
//!
//! ```rust
//! extern crate native;
//!
//! fn main() {
//!     // We're not sure whether this main function is run in 1:1 or M:N mode.
//!
//!     native::task::spawn(proc() {
//!         // this code is guaranteed to be run on a native thread
//!     });
//! }
//! ```

#![crate_id = "native#0.10-pre"]
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://static.rust-lang.org/doc/master")]
#![deny(unused_result, unused_must_use)]
#![allow(non_camel_case_types)]

#![allow(visible_private_types)] // NOTE: remove after a stage0 snap

// NB this crate explicitly does *not* allow glob imports, please seriously
//    consider whether they're needed before adding that feature here (the
//    answer is that you don't need them)

use std::os;
use std::rt;
use std::str;

pub mod io;
pub mod task;

#[cfg(windows)]
#[cfg(android)]
static OS_DEFAULT_STACK_ESTIMATE: uint = 1 << 20;
#[cfg(unix, not(android))]
static OS_DEFAULT_STACK_ESTIMATE: uint = 2 * (1 << 20);

#[lang = "start"]
#[cfg(not(test))]
pub fn lang_start(main: *u8, argc: int, argv: **u8) -> int {
    use std::cast;
    start(argc, argv, proc() {
        let main: extern "Rust" fn() = unsafe { cast::transmute(main) };
        main();
    })
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
pub fn start(argc: int, argv: **u8, main: proc()) -> int {
    let something_around_the_top_of_the_stack = 1;
    let addr = &something_around_the_top_of_the_stack as *int;
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
        use std::libc;
        use std::libc::funcs::posix01::signal::signal;
        unsafe {
            assert!(signal(libc::SIGPIPE, libc::SIG_IGN) != -1);
        }
    }
    ignore_sigpipe();

    rt::init(argc, argv);
    let mut exit_code = None;
    let mut main = Some(main);
    let mut task = task::new((my_stack_bottom, my_stack_top));
    task.name = Some(str::Slice("<main>"));
    let t = task.run(|| {
        unsafe {
            rt::stack::record_stack_bounds(my_stack_bottom, my_stack_top);
        }
        exit_code = Some(run(main.take_unwrap()));
    });
    drop(t);
    unsafe { rt::cleanup(); }
    // If the exit code wasn't set, then the task block must have failed.
    return exit_code.unwrap_or(rt::DEFAULT_ERROR_CODE);
}

/// Executes a procedure on the current thread in a Rust task context.
///
/// This function has all of the same details as `start` except for a different
/// number of arguments.
pub fn run(main: proc()) -> int {
    main();
    os::get_exit_status()
}
