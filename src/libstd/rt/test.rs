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
use iterator::IteratorUtil;
use vec::{OwnedVector, MutableVector};
use super::io::net::ip::{IpAddr, Ipv4, Ipv6};
use rt::sched::Scheduler;
use rt::local::Local;
use unstable::run_in_bare_thread;
use rt::thread::Thread;
use rt::task::Task;
use rt::uv::uvio::UvEventLoop;
use rt::work_queue::WorkQueue;
use rt::sleeper_list::SleeperList;
use rt::task::{Sched};
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

/// Creates a new scheduler in a new thread and runs a task in it,
/// then waits for the scheduler to exit. Failure of the task
/// will abort the process.
pub fn run_in_newsched_task(f: ~fn()) {
    let f = Cell::new(f);

    do run_in_bare_thread {
        let mut sched = ~new_test_uv_sched();
        let on_exit: ~fn(bool) = |exit_status| rtassert!(exit_status);
        let mut task = ~Task::new_root(&mut sched.stack_pool,
                                       f.take());
        rtdebug!("newsched_task: %x", ::borrow::to_uint(task));
        task.death.on_exit = Some(on_exit);
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
    use rt::sched::Shutdown;

    let f_cell = Cell::new(f);

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

        let f_cell = Cell::new(f_cell.take());
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
                                        f_cell.take());
        main_task.death.on_exit = Some(on_exit);
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
        for threads.consume_iter().advance() |thread| {
            thread.join();
        }
    }

}

/// Test tasks will abort on failure instead of unwinding
pub fn spawntask(f: ~fn()) {
    use super::sched::*;
    let f = Cell::new(f);

    let task = unsafe {
        let sched = Local::unsafe_borrow::<Scheduler>();
        rtdebug!("spawntask taking the scheduler from TLS");


        do Local::borrow::<Task, ~Task>() |running_task| {
            ~running_task.new_child(&mut (*sched).stack_pool, f.take())
        }
    };

    rtdebug!("new task pointer: %x", ::borrow::to_uint(task));

    let sched = Local::take::<Scheduler>();
    rtdebug!("spawntask scheduling the new task");
    sched.schedule_task(task);
}


/// Create a new task and run it right now. Aborts on failure
pub fn spawntask_immediately(f: ~fn()) {
    use super::sched::*;

    let f = Cell::new(f);

    let task = unsafe {
        let sched = Local::unsafe_borrow::<Scheduler>();
        do Local::borrow::<Task, ~Task>() |running_task| {
            ~running_task.new_child(&mut (*sched).stack_pool,
                                    f.take())
        }
    };

    let sched = Local::take::<Scheduler>();
    do sched.switch_running_tasks_and_then(task) |sched, task| {
        sched.enqueue_blocked_task(task);
    }
}

/// Create a new task and run it right now. Aborts on failure
pub fn spawntask_later(f: ~fn()) {
    use super::sched::*;
    let f = Cell::new(f);

    let task = unsafe {
        let sched = Local::unsafe_borrow::<Scheduler>();
        do Local::borrow::<Task, ~Task>() |running_task| {
            ~running_task.new_child(&mut (*sched).stack_pool, f.take())
        }
    };

    let mut sched = Local::take::<Scheduler>();
    sched.enqueue_task(task);
    Local::put(sched);
}

/// Spawn a task and either run it immediately or run it later
pub fn spawntask_random(f: ~fn()) {
    use super::sched::*;
    use rand::{Rand, rng};

    let f = Cell::new(f);

    let task = unsafe {
        let sched = Local::unsafe_borrow::<Scheduler>();
        do Local::borrow::<Task, ~Task>() |running_task| {
            ~running_task.new_child(&mut (*sched).stack_pool,
                                    f.take())

        }
    };

    let mut sched = Local::take::<Scheduler>();

    let mut rng = rng();
    let run_now: bool = Rand::rand(&mut rng);

    if run_now {
        do sched.switch_running_tasks_and_then(task) |sched, task| {
            sched.enqueue_blocked_task(task);
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

        ~Task::new_root_homed(&mut sched.stack_pool,
                              Sched(handle),
                              af)
    };
    let dest_sched = &mut scheds[rng.gen_int_range(0,scheds.len() as int)];
    // enqueue it for future execution
    dest_sched.enqueue_task(task);
}

/// Spawn a task and wait for it to finish, returning whether it
/// completed successfully or failed
pub fn spawntask_try(f: ~fn()) -> Result<(), ()> {
    use cell::Cell;
    use super::sched::*;

    let f = Cell::new(f);

    let (port, chan) = oneshot();
    let chan = Cell::new(chan);
    let on_exit: ~fn(bool) = |exit_status| chan.take().send(exit_status);
    let mut new_task = unsafe {
        let sched = Local::unsafe_borrow::<Scheduler>();
        do Local::borrow::<Task, ~Task> |_running_task| {

            // I don't understand why using a child task here fails. I
            // think the fail status is propogating back up the task
            // tree and triggering a fail for the parent, which we
            // aren't correctly expecting.

            // ~running_task.new_child(&mut (*sched).stack_pool,
            ~Task::new_root(&mut (*sched).stack_pool,
                           f.take())
        }
    };
    new_task.death.on_exit = Some(on_exit);

    let sched = Local::take::<Scheduler>();
    do sched.switch_running_tasks_and_then(new_task) |sched, old_task| {
        sched.enqueue_blocked_task(old_task);
    }

    rtdebug!("enqueued the new task, now waiting on exit_status");

    let exit_status = port.recv();
    if exit_status { Ok(()) } else { Err(()) }
}

/// Spawn a new task in a new scheduler and return a thread handle.
pub fn spawntask_thread(f: ~fn()) -> Thread {
    use rt::sched::*;

    let f = Cell::new(f);

    let task = unsafe {
        let sched = Local::unsafe_borrow::<Scheduler>();
        do Local::borrow::<Task, ~Task>() |running_task| {
            ~running_task.new_child(&mut (*sched).stack_pool,
                                    f.take())
        }
    };

    let task = Cell::new(task);

    let thread = do Thread::start {
        let mut sched = ~new_test_uv_sched();
        sched.enqueue_task(task.take());
        sched.run();
    };
    return thread;
}

/// Get a ~Task for testing purposes other than actually scheduling it.
pub fn with_test_task(blk: ~fn(~Task) -> ~Task) {
    do run_in_bare_thread {
        let mut sched = ~new_test_uv_sched();
        let task = blk(~Task::new_root(&mut sched.stack_pool, ||{}));
        sched.enqueue_task(task);
        sched.run();
    }
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

    for bases.iter().advance |&(dir, base)| {
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
