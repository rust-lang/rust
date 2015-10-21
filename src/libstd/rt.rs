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

#![unstable(feature = "rt",
            reason = "this public module should not exist and is highly likely \
                      to disappear",
            issue = "0")]
#![doc(hidden)]

use borrow::ToOwned;
use mem;
use sys;
use sys_common::thread_info::{self, NewThread};
use sys_common;
use thread::{self, Thread};

// Reexport some of our utilities which are expected by other crates.
pub use sys_common::unwind::{begin_unwind, begin_unwind_fmt};

// Rust runtime's startup objects depend on these symbols, so they must be public.
// Since sys_common isn't public, we have to re-export them here.
#[cfg(all(target_os="windows", target_arch = "x86", target_env="gnu"))]
pub use sys_common::unwind::imp::eh_frame_registry::*;

#[cfg(not(test))]
#[lang = "start"]
fn lang_start(main: *const u8, argc: isize, argv: *const *const u8) -> isize {
    sys::init();

    let failed = unsafe {
        let main_guard = sys::thread::guard::init();
        sys::stack_overflow::init();

        // Next, set up the current Thread with the guard information we just
        // created. Note that this isn't necessary in general for new threads,
        // but we just do this to name the main thread and to give it correct
        // info about the stack bounds.
        let thread: Thread = NewThread::new(Some("<main>".to_owned()));
        thread_info::set(main_guard, thread);

        // Store our args if necessary in a squirreled away location
        sys_common::args::init(argc, argv);

        // Let's run some code!
        let res = thread::catch_panic(mem::transmute::<_, fn()>(main));
        sys_common::cleanup();
        res.is_err()
    };

    if failed {
        101
    } else {
        0
    }
}
