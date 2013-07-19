// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cell::Cell;
use uint;
use option::{Some, None};
use rt::sched::Scheduler;
use super::io::net::ip::{IpAddr, Ipv4};
use unstable::run_in_bare_thread;
use rt::thread::Thread;
use rt::task::Task;
use rt::uv::uvio::UvEventLoop;
use rt::work_queue::WorkQueue;
use rt::sleeper_list::SleeperList;
use rt::comm::oneshot;
use result::{Result, Ok, Err};

pub fn new_test_uv_sched() -> Scheduler {

    let mut sched = Scheduler::new(~UvEventLoop::new(),
                                   WorkQueue::new(),
                                   SleeperList::new());

    // Don't wait for the Shutdown message
    sched.no_sleep = true;
    return sched;

}

pub fn run_in_newsched_task(f: ~fn()) {
    let f = Cell::new(f);
    do run_in_bare_thread {
        run_in_newsched_task_core(f.take());
    }
}

pub fn run_in_newsched_task_core(f: ~fn()) {

    use rt::sched::Shutdown;

    let mut sched = ~new_test_uv_sched();
    let exit_handle = Cell::new(sched.make_handle());

    let on_exit: ~fn(bool) = |exit_status| {
        exit_handle.take().send(Shutdown);
        rtassert!(exit_status);
    };
    let mut task = ~Task::new_root(&mut sched.stack_pool, f);
    task.on_exit = Some(on_exit);

    sched.bootstrap(task);
}

/// Create more than one scheduler and run a function in a task
/// in one of the schedulers. The schedulers will stay alive
/// until the function `f` returns.
pub fn run_in_mt_newsched_task(f: ~fn()) {
    use os;
    use from_str::FromStr;
    use rt::sched::Shutdown;
    use rt::util;

    let f = Cell::new(f);

    do run_in_bare_thread {
        let nthreads = match os::getenv("RUST_TEST_THREADS") {
            Some(nstr) => FromStr::from_str(nstr).get(),
            None => {
                // Using more threads than cores in test code
                // to force the OS to preempt them frequently.
                // Assuming that this helps stress test concurrent types.
                util::num_cpus() * 2
            }
        };

        let sleepers = SleeperList::new();
        let work_queue = WorkQueue::new();

        let mut handles = ~[];
        let mut scheds = ~[];

        for uint::range(0, nthreads) |_| {
            let loop_ = ~UvEventLoop::new();
            let mut sched = ~Scheduler::new(loop_,
                                            work_queue.clone(),
                                            sleepers.clone());
            let handle = sched.make_handle();

            handles.push(handle);
            scheds.push(sched);
        }

        let f = Cell::new(f.take());
        let handles = Cell::new(handles);
        let on_exit: ~fn(bool) = |exit_status| {
            let mut handles = handles.take();
            // Tell schedulers to exit
            for handles.mut_iter().advance |handle| {
                handle.send(Shutdown);
            }

            rtassert!(exit_status);
        };
        let mut main_task = ~Task::new_root(&mut scheds[0].stack_pool,
                                        f.take());
        main_task.on_exit = Some(on_exit);

        let mut threads = ~[];
        let main_task = Cell::new(main_task);

        let main_thread = {
            let sched = scheds.pop();
            let sched_cell = Cell::new(sched);
            do Thread::start {
                let sched = sched_cell.take();
                sched.bootstrap(main_task.take());
            }
        };
        threads.push(main_thread);

        while !scheds.is_empty() {
            let mut sched = scheds.pop();
            let bootstrap_task = ~do Task::new_root(&mut sched.stack_pool) || {};
            let bootstrap_task_cell = Cell::new(bootstrap_task);
            let sched_cell = Cell::new(sched);
            let thread = do Thread::start {
                let sched = sched_cell.take();
                sched.bootstrap(bootstrap_task_cell.take());
            };

            threads.push(thread);
        }

        // Wait for schedulers
        let _threads = threads;
    }

}

/// Test tasks will abort on failure instead of unwinding
pub fn spawntask(f: ~fn()) {
    Scheduler::run_task(Task::build_child(f));
}

/// Create a new task and run it right now. Aborts on failure
pub fn spawntask_later(f: ~fn()) {
    Scheduler::run_task_later(Task::build_child(f));
}

pub fn spawntask_random(f: ~fn()) {
    use rand::{Rand, rng};

    let mut rng = rng();
    let run_now: bool = Rand::rand(&mut rng);

    if run_now {
        spawntask(f)
    } else {
        spawntask_later(f)
    }
}

pub fn spawntask_try(f: ~fn()) -> Result<(),()> {

    let (port, chan) = oneshot();
    let chan = Cell::new(chan);
    let on_exit: ~fn(bool) = |exit_status| chan.take().send(exit_status);

    let mut new_task = Task::build_root(f);
    new_task.on_exit = Some(on_exit);

    Scheduler::run_task(new_task);

    let exit_status = port.recv();
    if exit_status { Ok(()) } else { Err(()) }

}

pub fn spawntask_thread(f: ~fn()) -> Thread {

    let f = Cell::new(f);

    let thread = do Thread::start {
        run_in_newsched_task_core(f.take());
    };

    return thread;
}

/// Use to cleanup tasks created for testing but not "run".
pub fn cleanup_task(task: ~Task) {

    let mut task = task;
    task.destroyed = true;

    let local_success = !task.unwinder.unwinding;
    let join_latch = task.join_latch.swap_unwrap();
    match task.on_exit {
        Some(ref on_exit) => {
            let success = join_latch.wait(local_success);
            (*on_exit)(success);
        }
        None => {
            join_latch.release(local_success);
        }
    }
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

/// Get a constant that represents the number of times to repeat
/// stress tests. Default 1.
pub fn stress_factor() -> uint {
    use os::getenv;

    match getenv("RUST_RT_STRESS") {
        Some(val) => uint::from_str(val).get(),
        None => 1
    }
}