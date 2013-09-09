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
use iter::{Iterator, range};
use super::io::net::ip::{SocketAddr, Ipv4Addr, Ipv6Addr};
use vec::{OwnedVector, MutableVector, ImmutableVector};
use rt::sched::Scheduler;
use unstable::{run_in_bare_thread};
use rt::thread::Thread;
use rt::task::Task;
use rt::uv::uvio::UvEventLoop;
use rt::work_queue::WorkQueue;
use rt::sleeper_list::SleeperList;
use rt::comm::oneshot;
use result::{Result, Ok, Err};

pub fn new_test_uv_sched() -> Scheduler {

    let queue = WorkQueue::new();
    let queues = ~[queue.clone()];

    let mut sched = Scheduler::new(~UvEventLoop::new(),
                                   queue,
                                   queues,
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
    let mut task = ~Task::new_root(&mut sched.stack_pool, None, f);
    task.death.on_exit = Some(on_exit);

    sched.bootstrap(task);
}

#[cfg(target_os="macos")]
#[allow(non_camel_case_types)]
mod darwin_fd_limit {
    /*!
     * darwin_fd_limit exists to work around an issue where launchctl on Mac OS X defaults the
     * rlimit maxfiles to 256/unlimited. The default soft limit of 256 ends up being far too low
     * for our multithreaded scheduler testing, depending on the number of cores available.
     *
     * This fixes issue #7772.
     */

    use libc;
    type rlim_t = libc::uint64_t;
    struct rlimit {
        rlim_cur: rlim_t,
        rlim_max: rlim_t
    }
    #[nolink]
    extern {
        // name probably doesn't need to be mut, but the C function doesn't specify const
        fn sysctl(name: *mut libc::c_int, namelen: libc::c_uint,
                  oldp: *mut libc::c_void, oldlenp: *mut libc::size_t,
                  newp: *mut libc::c_void, newlen: libc::size_t) -> libc::c_int;
        fn getrlimit(resource: libc::c_int, rlp: *mut rlimit) -> libc::c_int;
        fn setrlimit(resource: libc::c_int, rlp: *rlimit) -> libc::c_int;
    }
    static CTL_KERN: libc::c_int = 1;
    static KERN_MAXFILESPERPROC: libc::c_int = 29;
    static RLIMIT_NOFILE: libc::c_int = 8;

    pub unsafe fn raise_fd_limit() {
        #[fixed_stack_segment]; #[inline(never)];

        // The strategy here is to fetch the current resource limits, read the kern.maxfilesperproc
        // sysctl value, and bump the soft resource limit for maxfiles up to the sysctl value.
        use ptr::{to_unsafe_ptr, to_mut_unsafe_ptr, mut_null};
        use sys::size_of_val;
        use os::last_os_error;

        // Fetch the kern.maxfilesperproc value
        let mut mib: [libc::c_int, ..2] = [CTL_KERN, KERN_MAXFILESPERPROC];
        let mut maxfiles: libc::c_int = 0;
        let mut size: libc::size_t = size_of_val(&maxfiles) as libc::size_t;
        if sysctl(to_mut_unsafe_ptr(&mut mib[0]), 2,
                  to_mut_unsafe_ptr(&mut maxfiles) as *mut libc::c_void,
                  to_mut_unsafe_ptr(&mut size),
                  mut_null(), 0) != 0 {
            let err = last_os_error();
            error!("raise_fd_limit: error calling sysctl: %s", err);
            return;
        }

        // Fetch the current resource limits
        let mut rlim = rlimit{rlim_cur: 0, rlim_max: 0};
        if getrlimit(RLIMIT_NOFILE, to_mut_unsafe_ptr(&mut rlim)) != 0 {
            let err = last_os_error();
            error!("raise_fd_limit: error calling getrlimit: %s", err);
            return;
        }

        // Bump the soft limit to the smaller of kern.maxfilesperproc and the hard limit
        rlim.rlim_cur = ::cmp::min(maxfiles as rlim_t, rlim.rlim_max);

        // Set our newly-increased resource limit
        if setrlimit(RLIMIT_NOFILE, to_unsafe_ptr(&rlim)) != 0 {
            let err = last_os_error();
            error!("raise_fd_limit: error calling setrlimit: %s", err);
            return;
        }
    }
}

#[cfg(not(target_os="macos"))]
mod darwin_fd_limit {
    pub unsafe fn raise_fd_limit() {}
}

#[doc(hidden)]
pub fn prepare_for_lots_of_tests() {
    // Bump the fd limit on OS X. See darwin_fd_limit for an explanation.
    unsafe { darwin_fd_limit::raise_fd_limit() }
}

/// Create more than one scheduler and run a function in a task
/// in one of the schedulers. The schedulers will stay alive
/// until the function `f` returns.
pub fn run_in_mt_newsched_task(f: ~fn()) {
    use os;
    use from_str::FromStr;
    use rt::sched::Shutdown;
    use rt::util;

    // see comment in other function (raising fd limits)
    prepare_for_lots_of_tests();

    let f = Cell::new(f);

    do run_in_bare_thread {
        let nthreads = match os::getenv("RUST_RT_TEST_THREADS") {
            Some(nstr) => FromStr::from_str(nstr).unwrap(),
            None => {
                if util::limit_thread_creation_due_to_osx_and_valgrind() {
                    1
                } else {
                    // Using more threads than cores in test code
                    // to force the OS to preempt them frequently.
                    // Assuming that this help stress test concurrent types.
                    util::num_cpus() * 2
                }
            }
        };

        let sleepers = SleeperList::new();

        let mut handles = ~[];
        let mut scheds = ~[];
        let mut work_queues = ~[];

        for _ in range(0u, nthreads) {
            let work_queue = WorkQueue::new();
            work_queues.push(work_queue);
        }

        for i in range(0u, nthreads) {
            let loop_ = ~UvEventLoop::new();
            let mut sched = ~Scheduler::new(loop_,
                                            work_queues[i].clone(),
                                            work_queues.clone(),
                                            sleepers.clone());
            let handle = sched.make_handle();

            handles.push(handle);
            scheds.push(sched);
        }

        let handles = Cell::new(handles);
        let on_exit: ~fn(bool) = |exit_status| {
            let mut handles = handles.take();
            // Tell schedulers to exit
            for handle in handles.mut_iter() {
                handle.send(Shutdown);
            }

            rtassert!(exit_status);
        };
        let mut main_task = ~Task::new_root(&mut scheds[0].stack_pool, None, f.take());
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
            let bootstrap_task = ~do Task::new_root(&mut sched.stack_pool, None) || {
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
        for thread in threads.move_iter() {
            thread.join();
        }
    }

}

/// Test tasks will abort on failure instead of unwinding
pub fn spawntask(f: ~fn()) {
    Scheduler::run_task(Task::build_child(None, f));
}

/// Create a new task and run it right now. Aborts on failure
pub fn spawntask_later(f: ~fn()) {
    Scheduler::run_task_later(Task::build_child(None, f));
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

    let mut new_task = Task::build_root(None, f);
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
        let task = blk(~Task::new_root(&mut sched.stack_pool, None, ||{}));
        cleanup_task(task);
    }
}

/// Use to cleanup tasks created for testing but not "run".
pub fn cleanup_task(mut task: ~Task) {
    task.destroyed = true;
}

/// Get a port number, starting at 9600, for use in tests
#[fixed_stack_segment] #[inline(never)]
pub fn next_test_port() -> u16 {
    unsafe {
        return rust_dbg_next_port(base_port() as libc::uintptr_t) as u16;
    }
    extern {
        fn rust_dbg_next_port(base: libc::uintptr_t) -> libc::uintptr_t;
    }
}

/// Get a unique IPv4 localhost:port pair starting at 9600
pub fn next_test_ip4() -> SocketAddr {
    SocketAddr { ip: Ipv4Addr(127, 0, 0, 1), port: next_test_port() }
}

/// Get a unique IPv6 localhost:port pair starting at 9600
pub fn next_test_ip6() -> SocketAddr {
    SocketAddr { ip: Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 1), port: next_test_port() }
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

    for &(dir, base) in bases.iter() {
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
        Some(val) => uint::from_str(val).unwrap(),
        None => 1
    }
}
