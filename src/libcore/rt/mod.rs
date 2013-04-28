// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Rust runtime services, including the task scheduler and I/O interface

#[doc(hidden)];

use libc::c_char;

/// The Scheduler and Task types
mod sched;

/// Thread-local access to the current Scheduler
mod local_sched;

/// Synchronous I/O
#[path = "io/mod.rs"]
pub mod io;

/// Thread-local implementations of language-critical runtime features like @
pub mod local_services;

/// The EventLoop and internal synchronous I/O interface, dynamically
/// overridable so that it's primary implementation on libuv can
/// live outside of core.
mod rtio;

/// libuv
#[path = "uv/mod.rs"]
pub mod uv;

// FIXME #5248: The import in `sched` doesn't resolve unless this is pub!
/// Bindings to pthread/windows thread-local storage
pub mod thread_local_storage;

/// A parallel work-stealing queue
mod work_queue;

/// Stack segments and their cacheing
mod stack;

/// CPU context swapping
mod context;

/// Bindings to system threading libraries
mod thread;

/// The runtime configuration, read from environment variables
pub mod env;

/// The local, managed heap
mod local_heap;

/// The Logger trait and implementations
pub mod logging;

/// Tools for testing the runtime
#[cfg(test)]
pub mod test;

/// Set up a default runtime configuration, given compiler-supplied arguments.
///
/// This is invoked by the `start` _language item_ (unstable::lang) to
/// run a Rust executable.
///
/// # Arguments
///
/// * `main` - A C-abi function that takes no arguments and returns `c_void`.
///   It is a wrapper around the user-defined `main` function, and will be run
///   in a task.
/// * `argc` & `argv` - The argument vector. On Unix this information is used
///   by os::args.
/// * `crate_map` - Runtime information about the executing crate, mostly for logging
///
/// # Return value
///
/// The return value is used as the process return code. 0 on success, 101 on error.
pub fn start(main: *u8, _argc: int, _argv: **c_char, _crate_map: *u8) -> int {

    use self::sched::{Scheduler, Task};
    use self::uv::uvio::UvEventLoop;
    use sys::Closure;
    use ptr;
    use cast;

    let loop_ = ~UvEventLoop::new();
    let mut sched = ~Scheduler::new(loop_);

    let main_task = ~do Task::new(&mut sched.stack_pool) {

        unsafe {
            // `main` is an `fn() -> ()` that doesn't take an environment
            // XXX: Could also call this as an `extern "Rust" fn` once they work
            let main = Closure {
                code: main as *(),
                env: ptr::null(),
            };
            let mainfn: &fn() = cast::transmute(main);

            mainfn();
        }
    };

    sched.task_queue.push_back(main_task);
    sched.run();

    return 0;
}

/// Possible contexts in which Rust code may be executing.
/// Different runtime services are available depending on context.
/// Mostly used for determining if we're using the new scheduler
/// or the old scheduler.
#[deriving(Eq)]
pub enum RuntimeContext {
    // Only the exchange heap is available
    GlobalContext,
    // The scheduler may be accessed
    SchedulerContext,
    // Full task services, e.g. local heap, unwinding
    TaskContext,
    // Running in an old-style task
    OldTaskContext
}

/// Determine the current RuntimeContext
pub fn context() -> RuntimeContext {

    use task::rt::rust_task;
    use self::sched::local_sched;

    // XXX: Hitting TLS twice to check if the scheduler exists
    // then to check for the task is not good for perf
    if unsafe { rust_try_get_task().is_not_null() } {
        return OldTaskContext;
    } else {
        if local_sched::exists() {
            let context = ::cell::empty_cell();
            do local_sched::borrow |sched| {
                if sched.in_task_context() {
                    context.put_back(TaskContext);
                } else {
                    context.put_back(SchedulerContext);
                }
            }
            return context.take();
        } else {
            return GlobalContext;
        }
    }

    pub extern {
        #[rust_stack]
        fn rust_try_get_task() -> *rust_task;
    }
}

#[test]
fn test_context() {
    use unstable::run_in_bare_thread;
    use self::sched::{local_sched, Task};
    use rt::uv::uvio::UvEventLoop;
    use cell::Cell;

    assert!(context() == OldTaskContext);
    do run_in_bare_thread {
        assert!(context() == GlobalContext);
        let mut sched = ~UvEventLoop::new_scheduler();
        let task = ~do Task::new(&mut sched.stack_pool) {
            assert!(context() == TaskContext);
            let sched = local_sched::take();
            do sched.deschedule_running_task_and_then() |task| {
                assert!(context() == SchedulerContext);
                let task = Cell(task);
                do local_sched::borrow |sched| {
                    sched.task_queue.push_back(task.take());
                }
            }
        };
        sched.task_queue.push_back(task);
        sched.run();
    }
}
