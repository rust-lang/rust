// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Runtime services, including the task scheduler and I/O dispatcher

The `rt` module provides the private runtime infrastructure necessary
to support core language features like the exchange and local heap,
logging, local data and unwinding. It also implements the default task
scheduler and task model. Initialization routines are provided for setting
up runtime resources in common configurations, including that used by
`rustc` when generating executables.

It is intended that the features provided by `rt` can be factored in a
way such that the core library can be built with different 'profiles'
for different use cases, e.g. excluding the task scheduler. A number
of runtime features though are critical to the functioning of the
language and an implementation must be provided regardless of the
execution environment.

Of foremost importance is the global exchange heap, in the module
`heap`. Very little practical Rust code can be written without
access to the global heap. Unlike most of `rt` the global heap is
truly a global resource and generally operates independently of the
rest of the runtime.

All other runtime features are task-local, including the local heap,
local storage, logging and the stack unwinder.

The relationship between `rt` and the rest of the core library is
not entirely clear yet and some modules will be moving into or
out of `rt` as development proceeds.

Several modules in `core` are clients of `rt`:

* `std::task` - The user-facing interface to the Rust task model.
* `std::local_data` - The interface to local data.
* `std::unstable::lang` - Miscellaneous lang items, some of which rely on `std::rt`.
* `std::cleanup` - Local heap destruction.
* `std::io` - In the future `std::io` will use an `rt` implementation.
* `std::logging`
* `std::comm`

*/

#![experimental]

// FIXME: this should not be here.
#![allow(missing_docs)]

#![allow(dead_code)]

use failure;
use rustrt;
use os;

// Reexport some of our utilities which are expected by other crates.
pub use self::util::{default_sched_threads, min_stack, running_on_valgrind};

// Reexport functionality from librustrt and other crates underneath the
// standard library which work together to create the entire runtime.
pub use alloc::heap;
pub use rustrt::{begin_unwind, begin_unwind_fmt, at_exit};

// Simple backtrace functionality (to print on panic)
pub mod backtrace;

// Just stuff
mod util;

/// One-time runtime initialization.
///
/// Initializes global state, including frobbing
/// the crate's logging flags, registering GC
/// metadata, and storing the process arguments.
#[allow(experimental)]
pub fn init(argc: int, argv: *const *const u8) { unimplemented!() }

#[cfg(any(windows, android))]
static OS_DEFAULT_STACK_ESTIMATE: uint = 1 << 20;
#[cfg(all(unix, not(android)))]
static OS_DEFAULT_STACK_ESTIMATE: uint = 2 * (1 << 20);

#[cfg(not(test))]
#[lang = "start"]
fn lang_start(main: *const u8, argc: int, argv: *const *const u8) -> int { unimplemented!() }

/// Executes the given procedure after initializing the runtime with the given
/// argc/argv.
///
/// This procedure is guaranteed to run on the thread calling this function, but
/// the stack bounds for this rust task will *not* be set. Care must be taken
/// for this function to not overflow its stack.
///
/// This function will only return once *all* native threads in the system have
/// exited.
pub fn start(argc: int, argv: *const *const u8, main: proc()) -> int { unimplemented!() }

/// One-time runtime cleanup.
///
/// This function is unsafe because it performs no checks to ensure that the
/// runtime has completely ceased running. It is the responsibility of the
/// caller to ensure that the runtime is entirely shut down and nothing will be
/// poking around at the internal components.
///
/// Invoking cleanup while portions of the runtime are still in use may cause
/// undefined behavior.
pub unsafe fn cleanup() { unimplemented!() }
