// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! The Rust runtime, including the scheduler and I/O interface */

#[doc(hidden)];

use libc::c_char;

#[path = "sched/mod.rs"]
mod sched;
mod rtio;
pub mod uvll;
mod uvio;
#[path = "uv/mod.rs"]
mod uv;
#[path = "io/mod.rs"]
mod io;
// FIXME #5248: The import in `sched` doesn't resolve unless this is pub!
pub mod thread_local_storage;
mod work_queue;
mod stack;
mod context;
mod thread;
pub mod env;
pub mod local_services;
mod local_heap;

/// Tools for testing the runtime
#[cfg(test)]
pub mod test;

pub fn start(main: *u8, _argc: int, _argv: **c_char, _crate_map: *u8) -> int {
    use self::sched::{Scheduler, Task};
    use self::uvio::UvEventLoop;

    let loop_ = ~UvEventLoop::new();
    let mut sched = ~Scheduler::new(loop_);
    let main_task = ~do Task::new(&mut sched.stack_pool) {
        // XXX: Can't call a C function pointer from Rust yet
        unsafe { rust_call_nullary_fn(main) };
    };
    sched.task_queue.push_back(main_task);
    sched.run();
    return 0;

    extern {
        fn rust_call_nullary_fn(f: *u8);
    }
}

/// Possible contexts in which Rust code may be executing.
/// Different runtime services are available depending on context.
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
    use self::uvio::UvEventLoop;
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
