// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The "green scheduling" library
//!
//! This library provides M:N threading for rust programs. Internally this has
//! the implementation of a green scheduler along with context switching and a
//! stack-allocation strategy.
//!
//! This can be optionally linked in to rust programs in order to provide M:N
//! functionality inside of 1:1 programs.

#[link(name = "green",
       package_id = "green",
       vers = "0.9-pre",
       uuid = "20c38f8c-bfea-83ed-a068-9dc05277be26",
       url = "https://github.com/mozilla/rust/tree/master/src/libgreen")];

#[license = "MIT/ASL2"];
#[crate_type = "rlib"];
#[crate_type = "dylib"];

// NB this does *not* include globs, please keep it that way.
#[feature(macro_rules)];

use std::cast;
use std::os;
use std::rt::thread::Thread;
use std::rt;
use std::rt::crate_map;
use std::rt::rtio;
use std::sync::deque;
use std::sync::atomics::{SeqCst, AtomicUint, INIT_ATOMIC_UINT};
use std::task::TaskOpts;
use std::vec;
use std::util;
use stdtask = std::rt::task;

use sched::{Shutdown, Scheduler, SchedHandle, TaskFromFriend, NewNeighbor};
use sleeper_list::SleeperList;
use stack::StackPool;
use task::GreenTask;

mod macros;

pub mod basic;
pub mod context;
pub mod coroutine;
pub mod sched;
pub mod sleeper_list;
pub mod stack;
pub mod task;

#[cfg(test)] mod tests;

#[cfg(stage0)]
#[lang = "start"]
pub fn lang_start(main: *u8, argc: int, argv: **u8) -> int {
    do start(argc, argv) {
        let main: extern "Rust" fn() = unsafe { cast::transmute(main) };
        main();
    }
}

/// Set up a default runtime configuration, given compiler-supplied arguments.
///
/// This function will block the current thread of execution until the entire
/// pool of M:N schedulers have exited.
///
/// # Arguments
///
/// * `argc` & `argv` - The argument vector. On Unix this information is used
///   by os::args.
/// * `main` - The initial procedure to run inside of the M:N scheduling pool.
///            Once this procedure exits, the scheduling pool will begin to shut
///            down. The entire pool (and this function) will only return once
///            all child tasks have finished executing.
///
/// # Return value
///
/// The return value is used as the process return code. 0 on success, 101 on
/// error.
pub fn start(argc: int, argv: **u8, main: proc()) -> int {
    rt::init(argc, argv);
    let exit_code = run(main);
    // unsafe is ok b/c we're sure that the runtime is gone
    unsafe { rt::cleanup() }
    exit_code
}

/// Execute the main function in a pool of M:N schedulers.
///
/// Configures the runtime according to the environment, by default
/// using a task scheduler with the same number of threads as cores.
/// Returns a process exit code.
///
/// This function will not return until all schedulers in the associated pool
/// have returned.
pub fn run(main: proc()) -> int {
    let mut pool = SchedPool::new(PoolConfig::new());
    pool.spawn(TaskOpts::new(), main);
    unsafe { stdtask::wait_for_completion(); }
    pool.shutdown();
    os::get_exit_status()
}

/// Configuration of how an M:N pool of schedulers is spawned.
pub struct PoolConfig {
    /// The number of schedulers (OS threads) to spawn into this M:N pool.
    threads: uint,
}

impl PoolConfig {
    /// Returns the default configuration, as determined the the environment
    /// variables of this process.
    pub fn new() -> PoolConfig {
        PoolConfig {
            threads: rt::default_sched_threads(),
        }
    }
}

/// A structure representing a handle to a pool of schedulers. This handle is
/// used to keep the pool alive and also reap the status from the pool.
pub struct SchedPool {
    priv id: uint,
    priv threads: ~[Thread<()>],
    priv handles: ~[SchedHandle],
    priv stealers: ~[deque::Stealer<~task::GreenTask>],
    priv next_friend: uint,
    priv stack_pool: StackPool,
    priv deque_pool: deque::BufferPool<~task::GreenTask>,
    priv sleepers: SleeperList,
}

impl SchedPool {
    /// Execute the main function in a pool of M:N schedulers.
    ///
    /// This will configure the pool according to the `config` parameter, and
    /// initially run `main` inside the pool of schedulers.
    pub fn new(config: PoolConfig) -> SchedPool {
        static mut POOL_ID: AtomicUint = INIT_ATOMIC_UINT;

        let PoolConfig { threads: nscheds } = config;
        assert!(nscheds > 0);

        // The pool of schedulers that will be returned from this function
        let mut pool = SchedPool {
            threads: ~[],
            handles: ~[],
            stealers: ~[],
            id: unsafe { POOL_ID.fetch_add(1, SeqCst) },
            sleepers: SleeperList::new(),
            stack_pool: StackPool::new(),
            deque_pool: deque::BufferPool::new(),
            next_friend: 0,
        };

        // Create a work queue for each scheduler, ntimes. Create an extra
        // for the main thread if that flag is set. We won't steal from it.
        let arr = vec::from_fn(nscheds, |_| pool.deque_pool.deque());
        let (workers, stealers) = vec::unzip(arr.move_iter());
        pool.stealers = stealers;

        // Now that we've got all our work queues, create one scheduler per
        // queue, spawn the scheduler into a thread, and be sure to keep a
        // handle to the scheduler and the thread to keep them alive.
        for worker in workers.move_iter() {
            rtdebug!("inserting a regular scheduler");

            let mut sched = ~Scheduler::new(pool.id,
                                            new_event_loop(),
                                            worker,
                                            pool.stealers.clone(),
                                            pool.sleepers.clone());
            pool.handles.push(sched.make_handle());
            let sched = sched;
            pool.threads.push(do Thread::start {
                let mut sched = sched;
                let task = do GreenTask::new(&mut sched.stack_pool, None) {
                    rtdebug!("boostraping a non-primary scheduler");
                };
                sched.bootstrap(task);
            });
        }

        return pool;
    }

    pub fn task(&mut self, opts: TaskOpts, f: proc()) -> ~GreenTask {
        GreenTask::configure(&mut self.stack_pool, opts, f)
    }

    pub fn spawn(&mut self, opts: TaskOpts, f: proc()) {
        let task = self.task(opts, f);

        // Figure out someone to send this task to
        let idx = self.next_friend;
        self.next_friend += 1;
        if self.next_friend >= self.handles.len() {
            self.next_friend = 0;
        }

        // Jettison the task away!
        self.handles[idx].send(TaskFromFriend(task));
    }

    /// Spawns a new scheduler into this M:N pool. A handle is returned to the
    /// scheduler for use. The scheduler will not exit as long as this handle is
    /// active.
    ///
    /// The scheduler spawned will participate in work stealing with all of the
    /// other schedulers currently in the scheduler pool.
    pub fn spawn_sched(&mut self) -> SchedHandle {
        let (worker, stealer) = self.deque_pool.deque();
        self.stealers.push(stealer.clone());

        // Tell all existing schedulers about this new scheduler so they can all
        // steal work from it
        for handle in self.handles.mut_iter() {
            handle.send(NewNeighbor(stealer.clone()));
        }

        // Create the new scheduler, using the same sleeper list as all the
        // other schedulers as well as having a stealer handle to all other
        // schedulers.
        let mut sched = ~Scheduler::new(self.id,
                                        new_event_loop(),
                                        worker,
                                        self.stealers.clone(),
                                        self.sleepers.clone());
        let ret = sched.make_handle();
        self.handles.push(sched.make_handle());
        let sched = sched;
        self.threads.push(do Thread::start {
            let mut sched = sched;
            let task = do GreenTask::new(&mut sched.stack_pool, None) {
                rtdebug!("boostraping a non-primary scheduler");
            };
            sched.bootstrap(task);
        });

        return ret;
    }

    pub fn shutdown(mut self) {
        self.stealers = ~[];

        for mut handle in util::replace(&mut self.handles, ~[]).move_iter() {
            handle.send(Shutdown);
        }
        for thread in util::replace(&mut self.threads, ~[]).move_iter() {
            thread.join();
        }
    }
}

impl Drop for SchedPool {
    fn drop(&mut self) {
        if self.threads.len() > 0 {
            fail!("dropping a M:N scheduler pool that wasn't shut down");
        }
    }
}

fn new_event_loop() -> ~rtio::EventLoop {
    match crate_map::get_crate_map() {
        None => {}
        Some(map) => {
            match map.event_loop_factory {
                None => {}
                Some(factory) => return factory()
            }
        }
    }

    // If the crate map didn't specify a factory to create an event loop, then
    // instead just use a basic event loop missing all I/O services to at least
    // get the scheduler running.
    return basic::event_loop();
}
