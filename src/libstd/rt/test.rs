// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use uint;
use option::{Option, Some, None};
use cell::Cell;
use clone::Clone;
use container::Container;
use old_iter::MutableIter;
use vec::OwnedVector;
use result::{Result, Ok, Err};
use unstable::run_in_bare_thread;
use super::io::net::ip::{IpAddr, Ipv4};
use rt::task::Task;
use rt::thread::Thread;
use rt::local::Local;
use rt::sched::{Scheduler, Coroutine};
use rt::sleeper_list::SleeperList;
use rt::work_queue::WorkQueue;

pub fn new_test_uv_sched() -> Scheduler {
    use rt::uv::uvio::UvEventLoop;
    use rt::work_queue::WorkQueue;
    use rt::sleeper_list::SleeperList;

    let mut sched = Scheduler::new(~UvEventLoop::new(), WorkQueue::new(), SleeperList::new());
    // Don't wait for the Shutdown message
    sched.no_sleep = true;
    return sched;
}

/// Creates a new scheduler in a new thread and runs a task in it,
/// then waits for the scheduler to exit. Failure of the task
/// will abort the process.
pub fn run_in_newsched_task(f: ~fn()) {
    use super::sched::*;
    use unstable::run_in_bare_thread;
    use rt::uv::uvio::UvEventLoop;

    let f = Cell(f);

    do run_in_bare_thread {
        let mut sched = ~new_test_uv_sched();
        let task = ~Coroutine::with_task(&mut sched.stack_pool,
                                         ~Task::without_unwinding(),
                                         f.take());
        sched.enqueue_task(task);
        sched.run();
    }
}

/// Create more than one scheduler and run a function in a task
/// in one of the schedulers. The schedulers will stay alive
/// until the function `f` returns.
pub fn run_in_mt_newsched_task(f: ~fn()) {
    use rt::uv::uvio::UvEventLoop;
    use rt::sched::Shutdown;

    let f_cell = Cell(f);

    do run_in_bare_thread {
        static N: uint = 4;

        let sleepers = SleeperList::new();
        let work_queue = WorkQueue::new();

        let mut handles = ~[];
        let mut scheds = ~[];

        for uint::range(0, N) |i| {
            let loop_ = ~UvEventLoop::new();
            let mut sched = ~Scheduler::new(loop_, work_queue.clone(), sleepers.clone());
            let handle = sched.make_handle();
            handles.push(handle);
            scheds.push(sched);
        }

        let f_cell = Cell(f_cell.take());
        let handles = Cell(handles);
        let main_task = ~do Coroutine::new(&mut scheds[0].stack_pool) {
            f_cell.take()();

            let mut handles = handles.take();
            // Tell schedulers to exit
            for handles.each_mut |handle| {
                handle.send(Shutdown);
            }
        };

        scheds[0].enqueue_task(main_task);

        let mut threads = ~[];

        while !scheds.is_empty() {
            let sched = scheds.pop();
            let sched_cell = Cell(sched);
            let thread = do Thread::start {
                let mut sched = sched_cell.take();
                sched.run();
            };

            threads.push(thread);
        }

        // Wait for schedulers
        let _threads = threads;
    }
}

/// Test tasks will abort on failure instead of unwinding
pub fn spawntask(f: ~fn()) {
    use super::sched::*;

    let mut sched = Local::take::<Scheduler>();
    let task = ~Coroutine::with_task(&mut sched.stack_pool,
                                     ~Task::without_unwinding(),
                                     f);
    sched.schedule_new_task(task);
}

/// Create a new task and run it right now. Aborts on failure
pub fn spawntask_immediately(f: ~fn()) {
    use super::sched::*;

    let mut sched = Local::take::<Scheduler>();
    let task = ~Coroutine::with_task(&mut sched.stack_pool,
                                     ~Task::without_unwinding(),
                                     f);
    do sched.switch_running_tasks_and_then(task) |sched, task| {
        sched.enqueue_task(task);
    }
}

/// Create a new task and run it right now. Aborts on failure
pub fn spawntask_later(f: ~fn()) {
    use super::sched::*;

    let mut sched = Local::take::<Scheduler>();
    let task = ~Coroutine::with_task(&mut sched.stack_pool,
                                     ~Task::without_unwinding(),
                                     f);

    sched.enqueue_task(task);
    Local::put(sched);
}

/// Spawn a task and either run it immediately or run it later
pub fn spawntask_random(f: ~fn()) {
    use super::sched::*;
    use rand::{Rand, rng};

    let mut rng = rng();
    let run_now: bool = Rand::rand(&mut rng);

    let mut sched = Local::take::<Scheduler>();
    let task = ~Coroutine::with_task(&mut sched.stack_pool,
                                     ~Task::without_unwinding(),
                                     f);

    if run_now {
        do sched.switch_running_tasks_and_then(task) |sched, task| {
            sched.enqueue_task(task);
        }
    } else {
        sched.enqueue_task(task);
        Local::put(sched);
    }
}


/// Spawn a task and wait for it to finish, returning whether it completed successfully or failed
pub fn spawntask_try(f: ~fn()) -> Result<(), ()> {
    use cell::Cell;
    use super::sched::*;
    use task;
    use unstable::finally::Finally;

    // Our status variables will be filled in from the scheduler context
    let mut failed = false;
    let failed_ptr: *mut bool = &mut failed;

    // Switch to the scheduler
    let f = Cell(Cell(f));
    let sched = Local::take::<Scheduler>();
    do sched.deschedule_running_task_and_then() |sched, old_task| {
        let old_task = Cell(old_task);
        let f = f.take();
        let new_task = ~do Coroutine::new(&mut sched.stack_pool) {
            do (|| {
                (f.take())()
            }).finally {
                // Check for failure then resume the parent task
                unsafe { *failed_ptr = task::failing(); }
                let sched = Local::take::<Scheduler>();
                do sched.switch_running_tasks_and_then(old_task.take()) |sched, new_task| {
                    sched.enqueue_task(new_task);
                }
            }
        };

        sched.enqueue_task(new_task);
    }

    if !failed { Ok(()) } else { Err(()) }
}

// Spawn a new task in a new scheduler and return a thread handle.
pub fn spawntask_thread(f: ~fn()) -> Thread {
    use rt::sched::*;
    use rt::uv::uvio::UvEventLoop;

    let f = Cell(f);
    let thread = do Thread::start {
        let mut sched = ~new_test_uv_sched();
        let task = ~Coroutine::with_task(&mut sched.stack_pool,
                                         ~Task::without_unwinding(),
                                         f.take());
        sched.enqueue_task(task);
        sched.run();
    };
    return thread;
}

/// Get a port number, starting at 9600, for use in tests
pub fn next_test_port() -> u16 {
    unsafe {
        return rust_dbg_next_port() as u16;
    }
    extern {
        fn rust_dbg_next_port() -> ::libc::uintptr_t;
    }
}

/// Get a unique localhost:port pair starting at 9600
pub fn next_test_ip4() -> IpAddr {
    Ipv4(127, 0, 0, 1, next_test_port())
}

/// Get a constant that represents the number of times to repeat stress tests. Default 1.
pub fn stress_factor() -> uint {
    use os::getenv;

    match getenv("RUST_RT_STRESS") {
        Some(val) => uint::from_str(val).get(),
        None => 1
    }
}

