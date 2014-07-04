// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_id = "rustrt#0.11.0"]
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/0.11.0/")]

#![feature(macro_rules, phase, globs, thread_local, managed_boxes, asm)]
#![feature(linkage, lang_items, unsafe_destructor)]
#![no_std]
#![experimental]

#[phase(plugin, link)] extern crate core;
extern crate alloc;
extern crate libc;
extern crate collections;

#[cfg(test)] extern crate realrustrt = "rustrt";
#[cfg(test)] extern crate test;
#[cfg(test)] extern crate native;

#[cfg(test)] #[phase(plugin, link)] extern crate std;

pub use self::util::{Stdio, Stdout, Stderr};
pub use self::unwind::{begin_unwind, begin_unwind_fmt};

use core::prelude::*;

use alloc::owned::Box;
use core::any::Any;

use task::{Task, BlockedTask, TaskOpts};

mod macros;

mod at_exit_imp;
mod local_ptr;
mod thread_local_storage;
mod util;
mod libunwind;

pub mod args;
pub mod bookkeeping;
pub mod c_str;
pub mod exclusive;
pub mod local;
pub mod local_data;
pub mod local_heap;
pub mod mutex;
pub mod rtio;
pub mod stack;
pub mod task;
pub mod thread;
pub mod unwind;

/// The interface to the current runtime.
///
/// This trait is used as the abstraction between 1:1 and M:N scheduling. The
/// two independent crates, libnative and libgreen, both have objects which
/// implement this trait. The goal of this trait is to encompass all the
/// fundamental differences in functionality between the 1:1 and M:N runtime
/// modes.
pub trait Runtime {
    // Necessary scheduling functions, used for channels and blocking I/O
    // (sometimes).
    fn yield_now(~self, cur_task: Box<Task>);
    fn maybe_yield(~self, cur_task: Box<Task>);
    fn deschedule(~self, times: uint, cur_task: Box<Task>,
                  f: |BlockedTask| -> Result<(), BlockedTask>);
    fn reawaken(~self, to_wake: Box<Task>);

    // Miscellaneous calls which are very different depending on what context
    // you're in.
    fn spawn_sibling(~self,
                     cur_task: Box<Task>,
                     opts: TaskOpts,
                     f: proc():Send);
    fn local_io<'a>(&'a mut self) -> Option<rtio::LocalIo<'a>>;
    /// The (low, high) edges of the current stack.
    fn stack_bounds(&self) -> (uint, uint); // (lo, hi)
    fn can_block(&self) -> bool;

    // FIXME: This is a serious code smell and this should not exist at all.
    fn wrap(~self) -> Box<Any>;
}

/// The default error code of the rust runtime if the main task fails instead
/// of exiting cleanly.
pub static DEFAULT_ERROR_CODE: int = 101;

/// One-time runtime initialization.
///
/// Initializes global state, including frobbing
/// the crate's logging flags, registering GC
/// metadata, and storing the process arguments.
pub fn init(argc: int, argv: *const *const u8) {
    // FIXME: Derefing these pointers is not safe.
    // Need to propagate the unsafety to `start`.
    unsafe {
        args::init(argc, argv);
        local_ptr::init();
        at_exit_imp::init();
    }

    // FIXME(#14344) this shouldn't be necessary
    collections::fixme_14344_be_sure_to_link_to_collections();
    alloc::fixme_14344_be_sure_to_link_to_collections();
    libc::issue_14344_workaround();
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
    at_exit_imp::run();
    args::cleanup();
    local_ptr::cleanup();
}

// FIXME: these probably shouldn't be public...
#[doc(hidden)]
pub mod shouldnt_be_public {
    #[cfg(not(test))]
    pub use super::local_ptr::native::maybe_tls_key;
    #[cfg(not(windows), not(target_os = "android"), not(target_os = "ios"))]
    pub use super::local_ptr::compiled::RT_TLS_PTR;
}

#[cfg(not(test))]
mod std {
    pub use core::{fmt, option, cmp};
}
