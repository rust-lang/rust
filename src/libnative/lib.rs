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

// NB this crate explicitly does *not* allow glob imports, please seriously
//    consider whether they're needed before adding that feature here.

use std::cast;
use std::os;
use std::rt;
use std::task::try;

pub mod io;
pub mod task;

// XXX: this should not exist here
#[cfg(stage0)]
#[lang = "start"]
pub fn start(main: *u8, argc: int, argv: **u8) -> int {
    rt::init(argc, argv);

    // Bootstrap ourselves by installing a local Task and then immediately
    // spawning a thread to run 'main'. Always spawn a new thread for main so
    // the stack size of 'main' is known (and the bounds can be set
    // appropriately).
    //
    // Once the main task has completed, then we wait for everyone else to exit.
    task::run(task::new(), proc() {
        let main: extern "Rust" fn() = unsafe { cast::transmute(main) };
        match do try { main() } {
            Ok(()) => { os::set_exit_status(0); }
            Err(..) => { os::set_exit_status(rt::DEFAULT_ERROR_CODE); }
        }
    });
    task::wait_for_completion();

    unsafe { rt::cleanup(); }
    os::get_exit_status()
}
