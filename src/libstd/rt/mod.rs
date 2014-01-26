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
the garbage collector, logging, local data and unwinding. It also
implements the default task scheduler and task model. Initialization
routines are provided for setting up runtime resources in common
configurations, including that used by `rustc` when generating
executables.

It is intended that the features provided by `rt` can be factored in a
way such that the core library can be built with different 'profiles'
for different use cases, e.g. excluding the task scheduler. A number
of runtime features though are critical to the functioning of the
language and an implementation must be provided regardless of the
execution environment.

Of foremost importance is the global exchange heap, in the module
`global_heap`. Very little practical Rust code can be written without
access to the global heap. Unlike most of `rt` the global heap is
truly a global resource and generally operates independently of the
rest of the runtime.

All other runtime features are task-local, including the local heap,
the garbage collector, local storage, logging and the stack unwinder.

The relationship between `rt` and the rest of the core library is
not entirely clear yet and some modules will be moving into or
out of `rt` as development proceeds.

Several modules in `core` are clients of `rt`:

* `std::task` - The user-facing interface to the Rust task model.
* `std::local_data` - The interface to local data.
* `std::gc` - The garbage collector.
* `std::unstable::lang` - Miscellaneous lang items, some of which rely on `std::rt`.
* `std::condition` - Uses local data.
* `std::cleanup` - Local heap destruction.
* `std::io` - In the future `std::io` will use an `rt` implementation.
* `std::logging`
* `std::comm`

*/

// FIXME: this should not be here.
#[allow(missing_doc)];

use any::Any;
use option::Option;
use result::Result;
use task::TaskOpts;

use self::task::{Task, BlockedTask};

// this is somewhat useful when a program wants to spawn a "reasonable" number
// of workers based on the constraints of the system that it's running on.
// Perhaps this shouldn't be a `pub use` though and there should be another
// method...
pub use self::util::default_sched_threads;

// Export unwinding facilities used by the failure macros
pub use self::unwind::{begin_unwind, begin_unwind_raw};

// FIXME: these probably shouldn't be public...
#[doc(hidden)]
pub mod shouldnt_be_public {
    pub use super::local_ptr::native::maybe_tls_key;
    #[cfg(not(windows), not(target_os = "android"))]
    pub use super::local_ptr::compiled::RT_TLS_PTR;
}

// Internal macros used by the runtime.
mod macros;

/// The global (exchange) heap.
pub mod global_heap;

/// Implementations of language-critical runtime features like @.
pub mod task;

/// The EventLoop and internal synchronous I/O interface.
pub mod rtio;

/// The Local trait for types that are accessible via thread-local
/// or task-local storage.
pub mod local;

/// Bindings to system threading libraries.
pub mod thread;

/// The runtime configuration, read from environment variables.
pub mod env;

/// The local, managed heap
pub mod local_heap;

/// The Logger trait and implementations
pub mod logging;

/// Crate map
pub mod crate_map;

/// The runtime needs to be able to put a pointer into thread-local storage.
mod local_ptr;

/// Bindings to pthread/windows thread-local storage.
mod thread_local_storage;

/// Stack unwinding
pub mod unwind;

/// Just stuff
mod util;

// Global command line argument storage
pub mod args;

// Support for running procedures when a program has exited.
mod at_exit_imp;

/// The default error code of the rust runtime if the main task fails instead
/// of exiting cleanly.
pub static DEFAULT_ERROR_CODE: int = 101;

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
    fn yield_now(~self, cur_task: ~Task);
    fn maybe_yield(~self, cur_task: ~Task);
    fn deschedule(~self, times: uint, cur_task: ~Task,
                  f: |BlockedTask| -> Result<(), BlockedTask>);
    fn reawaken(~self, to_wake: ~Task, can_resched: bool);

    // Miscellaneous calls which are very different depending on what context
    // you're in.
    fn spawn_sibling(~self, cur_task: ~Task, opts: TaskOpts, f: proc());
    fn local_io<'a>(&'a mut self) -> Option<rtio::LocalIo<'a>>;
    /// The (low, high) edges of the current stack.
    fn stack_bounds(&self) -> (uint, uint); // (lo, hi)

    // FIXME: This is a serious code smell and this should not exist at all.
    fn wrap(~self) -> ~Any;
}

/// One-time runtime initialization.
///
/// Initializes global state, including frobbing
/// the crate's logging flags, registering GC
/// metadata, and storing the process arguments.
pub fn init(argc: int, argv: **u8) {
    // FIXME: Derefing these pointers is not safe.
    // Need to propagate the unsafety to `start`.
    unsafe {
        args::init(argc, argv);
        env::init();
        logging::init();
        local_ptr::init();
        at_exit_imp::init();
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
pub fn at_exit(f: proc()) {
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
    at_exit_imp::run();
    args::cleanup();
    local_ptr::cleanup();
}
