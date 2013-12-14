// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The native runtime crate
//!
//! This crate contains an implementation of 1:1 scheduling for a "native"
//! runtime. In addition, all I/O provided by this crate is the thread blocking
//! version of I/O.

#[link(name = "native",
       package_id = "native",
       vers = "0.9-pre",
       uuid = "535344a7-890f-5a23-e1f3-e0d118805141",
       url = "https://github.com/mozilla/rust/tree/master/src/native")];

#[license = "MIT/ASL2"];
#[crate_type = "rlib"];
#[crate_type = "dylib"];

#[cfg(stage0, test)] extern mod green;

// NB this crate explicitly does *not* allow glob imports, please seriously
//    consider whether they're needed before adding that feature here (the
//    answer is that you don't need them)

use std::os;
use std::rt;
use stdtask = std::rt::task;

pub mod io;
pub mod task;


// XXX: this should not exist here
#[cfg(stage0, notready)]
#[lang = "start"]
pub fn lang_start(main: *u8, argc: int, argv: **u8) -> int {
    use std::cast;
    use std::task::try;

    do start(argc, argv) {
        // Instead of invoking main directly on this thread, invoke it on
        // another spawned thread that we are guaranteed to know the size of the
        // stack of. Currently, we do not have a method of figuring out the size
        // of the main thread's stack, so for stack overflow detection to work
        // we must spawn the task in a subtask which we know the stack size of.
        let main: extern "Rust" fn() = unsafe { cast::transmute(main) };
        match do try { main() } {
            Ok(()) => { os::set_exit_status(0); }
            Err(..) => { os::set_exit_status(rt::DEFAULT_ERROR_CODE); }
        }
    }
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
    rt::init(argc, argv);
    let exit_code = run(main);
    unsafe { rt::cleanup(); }
    return exit_code;
}

/// Executes a procedure on the current thread in a Rust task context.
///
/// This function has all of the same details as `start` except for a different
/// number of arguments.
pub fn run(main: proc()) -> int {
    // Create a task, run the procedure in it, and then wait for everything.
    task::run(task::new(), main);

    // Block this OS task waiting for everything to finish.
    unsafe { stdtask::wait_for_completion() }

    os::get_exit_status()
}
