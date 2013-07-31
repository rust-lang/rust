// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc;
use uint;
use option::{Some, None};
use cell::Cell;
use clone::Clone;
use container::Container;
use iterator::Iterator;
use vec::{OwnedVector, MutableVector};
use super::io::net::ip::{IpAddr, Ipv4, Ipv6};
use rt::sched::Scheduler;
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
    task.death.on_exit = Some(on_exit);

    sched.bootstrap(task);
}

/// Create more than one scheduler and run a function in a task
/// in one of the schedulers. The schedulers will stay alive
/// until the function `f` returns.
pub fn run_in_mt_newsched_task(f: ~fn()) {
    use os;
    use from_str::FromStr;
    use rt::sched::Shutdown;

    let f = Cell::new(f);

    do run_in_bare_thread {
        let nthreads = match os::getenv("RUST_RT_TEST_THREADS") {
            Some(nstr) => FromStr::from_str(nstr).get(),
            None => {
                // A reasonable number of threads for testing
                // multithreading. NB: It's easy to exhaust OS X's
                // low maximum fd limit by setting this too high (#7772)
                4
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

        let handles = Cell::new(handles);
        let on_exit: ~fn(bool) = |exit_status| {
            let mut handles = handles.take();
            // Tell schedulers to exit
            foreach handle in handles.mut_iter() {
                handle.send(Shutdown);
            }

            rtassert!(exit_status);
        };
        let mut main_task = ~Task::new_root(&mut scheds[0].stack_pool,
                                        f.take());
        main_task.death.on_exit = Some(on_exit);

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
            let bootstrap_task = ~do Task::new_root(&mut sched.stack_pool) || {
                rtdebug!("bootstrapping non-primary scheduler");
            };
            let bootstrap_task_cell = Cell::new(bootstrap_task);
            let sched_cell = Cell::new(sched);
            let thread = do Thread::start {
                let sched = sched_cell.take();
                sched.bootstrap(bootstrap_task_cell.take());
            };

            threads.push(thread);
        }

        // Wait for schedulers
        foreach thread in threads.consume_iter() {
            thread.join();
        }
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
    new_task.death.on_exit = Some(on_exit);

    Scheduler::run_task(new_task);

    let exit_status = port.recv();
    if exit_status { Ok(()) } else { Err(()) }

}

/// Spawn a new task in a new scheduler and return a thread handle.
pub fn spawntask_thread(f: ~fn()) -> Thread {

    let f = Cell::new(f);

    let thread = do Thread::start {
        run_in_newsched_task_core(f.take());
    };

    return thread;
}

/// Get a ~Task for testing purposes other than actually scheduling it.
pub fn with_test_task(blk: ~fn(~Task) -> ~Task) {
    do run_in_bare_thread {
        let mut sched = ~new_test_uv_sched();
        let task = blk(~Task::new_root(&mut sched.stack_pool, ||{}));
        cleanup_task(task);
    }
}

/// Use to cleanup tasks created for testing but not "run".
pub fn cleanup_task(mut task: ~Task) {
    task.destroyed = true;
}

/// Get a port number, starting at 9600, for use in tests
pub fn next_test_port() -> u16 {
    unsafe {
        return rust_dbg_next_port(base_port() as libc::uintptr_t) as u16;
    }
    extern {
        fn rust_dbg_next_port(base: libc::uintptr_t) -> libc::uintptr_t;
    }
}

/// Get a unique IPv4 localhost:port pair starting at 9600
pub fn next_test_ip4() -> IpAddr {
    Ipv4(127, 0, 0, 1, next_test_port())
}

/// Get a unique IPv6 localhost:port pair starting at 9600
pub fn next_test_ip6() -> IpAddr {
    Ipv6(0, 0, 0, 0, 0, 0, 0, 1, next_test_port())
}

/*
XXX: Welcome to MegaHack City.

The bots run multiple builds at the same time, and these builds
all want to use ports. This function figures out which workspace
it is running in and assigns a port range based on it.
*/
fn base_port() -> uint {
    use os;
    use str::StrSlice;
    use to_str::ToStr;
    use vec::ImmutableVector;

    let base = 9600u;
    let range = 1000;

    let bases = [
        ("32-opt", base + range * 1),
        ("32-noopt", base + range * 2),
        ("64-opt", base + range * 3),
        ("64-noopt", base + range * 4),
        ("64-opt-vg", base + range * 5),
        ("all-opt", base + range * 6),
        ("snap3", base + range * 7),
        ("dist", base + range * 8)
    ];

    let path = os::getcwd().to_str();

    let mut final_base = base;

    foreach &(dir, base) in bases.iter() {
        if path.contains(dir) {
            final_base = base;
            break;
        }
    }

    return final_base;
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
