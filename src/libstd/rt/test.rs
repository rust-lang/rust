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
use option::{Some, None};
use cell::Cell;
use clone::Clone;
use container::Container;
use iterator::IteratorUtil;
use vec::{OwnedVector, MutableVector};
use result::{Result, Ok, Err};
use unstable::run_in_bare_thread;
use super::io::net::ip::{IpAddr, Ipv4};
use rt::comm::oneshot;
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

    let f = Cell::new(f);

    do run_in_bare_thread {
        let mut sched = ~new_test_uv_sched();
        let mut new_task = ~Task::new_root();
        let on_exit: ~fn(bool) = |exit_status| rtassert!(exit_status);
        new_task.on_exit = Some(on_exit);
        let task = ~Coroutine::with_task(&mut sched.stack_pool,
                                         new_task,
                                         f.take());
        sched.enqueue_task(task);
        sched.run();
    }
}

/// Create more than one scheduler and run a function in a task
/// in one of the schedulers. The schedulers will stay alive
/// until the function `f` returns.
pub fn run_in_mt_newsched_task(f: ~fn()) {
    use os;
    use from_str::FromStr;
    use rt::uv::uvio::UvEventLoop;
    use rt::sched::Shutdown;
    use rt::util;

    let f_cell = Cell::new(f);

    do run_in_bare_thread {
        let nthreads = match os::getenv("RUST_TEST_THREADS") {
            Some(nstr) => FromStr::from_str(nstr).get(),
            None => {
                // Using more threads than cores in test code
                // to force the OS to preempt them frequently.
                // Assuming that this help stress test concurrent types.
                util::num_cpus() * 2
            }
        };

        let sleepers = SleeperList::new();
        let work_queue = WorkQueue::new();

        let mut handles = ~[];
        let mut scheds = ~[];

        for uint::range(0, nthreads) |_| {
            let loop_ = ~UvEventLoop::new();
            let mut sched = ~Scheduler::new(loop_, work_queue.clone(), sleepers.clone());
            let handle = sched.make_handle();

            handles.push(handle);
            scheds.push(sched);
        }

        let f_cell = Cell::new(f_cell.take());
        let handles = Cell::new(handles);
        let mut new_task = ~Task::new_root();
        let on_exit: ~fn(bool) = |exit_status| {

            let mut handles = handles.take();
            // Tell schedulers to exit
            for handles.mut_iter().advance |handle| {
                handle.send(Shutdown);
            }

            rtassert!(exit_status);
        };
        new_task.on_exit = Some(on_exit);
        let main_task = ~Coroutine::with_task(&mut scheds[0].stack_pool,
                                              new_task, f_cell.take());
        scheds[0].enqueue_task(main_task);

        let mut threads = ~[];

        while !scheds.is_empty() {
            let sched = scheds.pop();
            let sched_cell = Cell::new(sched);
            let thread = do Thread::start {
                let sched = sched_cell.take();
                sched.run();
            };

            threads.push(thread);
        }

        // Wait for schedulers
        let _threads = threads;
    }

}

// THIS IS AWFUL. Copy-pasted the above initialization function but
// with a number of hacks to make it spawn tasks on a variety of
// schedulers with a variety of homes using the new spawn.

pub fn run_in_mt_newsched_task_random_homed() {
    use libc;
    use os;
    use from_str::FromStr;
    use rt::uv::uvio::UvEventLoop;
    use rt::sched::Shutdown;

    do run_in_bare_thread {
        let nthreads = match os::getenv("RUST_TEST_THREADS") {
            Some(nstr) => FromStr::from_str(nstr).get(),
            None => unsafe {
                // Using more threads than cores in test code to force
                // the OS to preempt them frequently.  Assuming that
                // this help stress test concurrent types.
                rust_get_num_cpus() * 2
            }
        };

        let sleepers = SleeperList::new();
        let work_queue = WorkQueue::new();

        let mut handles = ~[];
        let mut scheds = ~[];

        // create a few special schedulers, those with even indicies
        // will be pinned-only
        for uint::range(0, nthreads) |i| {
            let special = (i % 2) == 0;
            let loop_ = ~UvEventLoop::new();
            let mut sched = ~Scheduler::new_special(
                loop_, work_queue.clone(), sleepers.clone(), special);
            let handle = sched.make_handle();
            handles.push(handle);
            scheds.push(sched);
        }

        // Schedule a pile o tasks
        let n = 5*stress_factor();
        for uint::range(0,n) |_i| {
                rtdebug!("creating task: %u", _i);
                let hf: ~fn() = || { assert!(true) };
                spawntask_homed(&mut scheds, hf);
            }

        // Now we want another pile o tasks that do not ever run on a
        // special scheduler, because they are normal tasks. Because
        // we can we put these in the "main" task.

        let n = 5*stress_factor();

        let f: ~fn() = || {
            for uint::range(0,n) |_| {
                let f: ~fn()  = || {
                    // Borrow the scheduler we run on and check if it is
                    // privileged.
                    do Local::borrow::<Scheduler,()> |sched| {
                        assert!(sched.run_anything);
                    };
                };
                spawntask_random(f);
            };
        };

        let f_cell = Cell::new(f);
        let handles = Cell::new(handles);

        rtdebug!("creating main task");

        let main_task = ~do Coroutine::new_root(&mut scheds[0].stack_pool) {
            f_cell.take()();
            let mut handles = handles.take();
            // Tell schedulers to exit
            for handles.mut_iter().advance |handle| {
                handle.send(Shutdown);
            }
        };

        rtdebug!("queuing main task")

        scheds[0].enqueue_task(main_task);

        let mut threads = ~[];

        while !scheds.is_empty() {
            let sched = scheds.pop();
            let sched_cell = Cell::new(sched);
            let thread = do Thread::start {
                let sched = sched_cell.take();
                rtdebug!("running sched: %u", sched.sched_id());
                sched.run();
            };

            threads.push(thread);
        }

        rtdebug!("waiting on scheduler threads");

        // Wait for schedulers
        let _threads = threads;
    }

    extern {
        fn rust_get_num_cpus() -> libc::uintptr_t;
    }
}


/// Test tasks will abort on failure instead of unwinding
pub fn spawntask(f: ~fn()) {
    use super::sched::*;

    rtdebug!("spawntask taking the scheduler from TLS")
    let task = do Local::borrow::<Task, ~Task>() |running_task| {
        ~running_task.new_child()
    };

    let mut sched = Local::take::<Scheduler>();
    let task = ~Coroutine::with_task(&mut sched.stack_pool,
                                     task, f);
    rtdebug!("spawntask scheduling the new task");
    sched.schedule_task(task);
}

/// Create a new task and run it right now. Aborts on failure
pub fn spawntask_immediately(f: ~fn()) {
    use super::sched::*;

    let task = do Local::borrow::<Task, ~Task>() |running_task| {
        ~running_task.new_child()
    };

    let mut sched = Local::take::<Scheduler>();
    let task = ~Coroutine::with_task(&mut sched.stack_pool,
                                     task, f);
    do sched.switch_running_tasks_and_then(task) |sched, task| {
        sched.enqueue_task(task);
    }
}

/// Create a new task and run it right now. Aborts on failure
pub fn spawntask_later(f: ~fn()) {
    use super::sched::*;

    let task = do Local::borrow::<Task, ~Task>() |running_task| {
        ~running_task.new_child()
    };

    let mut sched = Local::take::<Scheduler>();
    let task = ~Coroutine::with_task(&mut sched.stack_pool,
                                     task, f);

    sched.enqueue_task(task);
    Local::put(sched);
}

/// Spawn a task and either run it immediately or run it later
pub fn spawntask_random(f: ~fn()) {
    use super::sched::*;
    use rand::{Rand, rng};

    let task = do Local::borrow::<Task, ~Task>() |running_task| {
        ~running_task.new_child()
    };

    let mut sched = Local::take::<Scheduler>();
    let task = ~Coroutine::with_task(&mut sched.stack_pool,
                                     task, f);

    let mut rng = rng();
    let run_now: bool = Rand::rand(&mut rng);

    if run_now {
        do sched.switch_running_tasks_and_then(task) |sched, task| {
            sched.enqueue_task(task);
        }
    } else {
        sched.enqueue_task(task);
        Local::put(sched);
    }
}

/// Spawn a task, with the current scheduler as home, and queue it to
/// run later.
pub fn spawntask_homed(scheds: &mut ~[~Scheduler], f: ~fn()) {
    use super::sched::*;
    use rand::{rng, RngUtil};
    let mut rng = rng();

    let task = {
        let sched = &mut scheds[rng.gen_int_range(0,scheds.len() as int)];
        let handle = sched.make_handle();
        let home_id = handle.sched_id;

        // now that we know where this is going, build a new function
        // that can assert it is in the right place
        let af: ~fn() = || {
            do Local::borrow::<Scheduler,()>() |sched| {
                rtdebug!("home_id: %u, runtime loc: %u",
                         home_id,
                         sched.sched_id());
                assert!(home_id == sched.sched_id());
            };
            f()
        };

        ~Coroutine::with_task_homed(&mut sched.stack_pool,
                                    ~Task::new_root(),
                                    af,
                                    Sched(handle))
    };
    let dest_sched = &mut scheds[rng.gen_int_range(0,scheds.len() as int)];
    // enqueue it for future execution
    dest_sched.enqueue_task(task);
}

/// Spawn a task and wait for it to finish, returning whether it completed successfully or failed
pub fn spawntask_try(f: ~fn()) -> Result<(), ()> {
    use cell::Cell;
    use super::sched::*;

    let (port, chan) = oneshot();
    let chan = Cell::new(chan);
    let mut new_task = ~Task::new_root();
    let on_exit: ~fn(bool) = |exit_status| chan.take().send(exit_status);
    new_task.on_exit = Some(on_exit);
    let mut sched = Local::take::<Scheduler>();
    let new_task = ~Coroutine::with_task(&mut sched.stack_pool,
                                         new_task, f);
    do sched.switch_running_tasks_and_then(new_task) |sched, old_task| {
        sched.enqueue_task(old_task);
    }

    let exit_status = port.recv();
    if exit_status { Ok(()) } else { Err(()) }
}

// Spawn a new task in a new scheduler and return a thread handle.
pub fn spawntask_thread(f: ~fn()) -> Thread {
    use rt::sched::*;

    let task = do Local::borrow::<Task, ~Task>() |running_task| {
        ~running_task.new_child()
    };

    let task = Cell::new(task);
    let f = Cell::new(f);
    let thread = do Thread::start {
        let mut sched = ~new_test_uv_sched();
        let task = ~Coroutine::with_task(&mut sched.stack_pool,
                                         task.take(),
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
