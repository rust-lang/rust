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
use std::rt::task::Task;
use std::rt::rtio;
use std::sync::deque;
use std::sync::atomics::{SeqCst, AtomicUint, INIT_ATOMIC_UINT};
use std::task::TaskResult;
use std::vec;
use std::util;

use sched::{Shutdown, Scheduler, SchedHandle};
use sleeper_list::SleeperList;
use task::{GreenTask, HomeSched};

mod macros;

pub mod basic;
pub mod context;
pub mod coroutine;
pub mod sched;
pub mod sleeper_list;
pub mod stack;
pub mod task;

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
    let config = Config {
        shutdown_after_main_exits: true,
        ..Config::new()
    };
    Pool::spawn(config, main).wait();
    os::get_exit_status()
}

/// Configuration of how an M:N pool of schedulers is spawned.
pub struct Config {
    /// If this flag is set, then when schedulers are spawned via the `start`
    /// and `run` functions the thread invoking `start` and `run` will have a
    /// scheduler spawned on it. This scheduler will be "special" in that the
    /// main task will be pinned to the scheduler and it will not participate in
    /// work stealing.
    ///
    /// If the `spawn` function is used to create a pool of schedulers, then
    /// this option has no effect.
    use_main_thread: bool,

    /// The number of schedulers (OS threads) to spawn into this M:N pool.
    threads: uint,

    /// When the main function exits, this flag will dictate whether a shutdown
    /// is requested of all schedulers. If this flag is `true`, this means that
    /// schedulers will shut down as soon as possible after the main task exits
    /// (but some may stay alive longer for things like I/O or other tasks).
    ///
    /// If this flag is `false`, then no action is taken when the `main` task
    /// exits. The scheduler pool is then shut down via the `wait()` function.
    shutdown_after_main_exits: bool,
}

impl Config {
    /// Returns the default configuration, as determined the the environment
    /// variables of this process.
    pub fn new() -> Config {
        Config {
            use_main_thread: false,
            threads: rt::default_sched_threads(),
            shutdown_after_main_exits: false,
        }
    }
}

/// A structure representing a handle to a pool of schedulers. This handle is
/// used to keep the pool alive and also reap the status from the pool.
pub struct Pool {
    priv threads: ~[Thread<()>],
    priv handles: Option<~[SchedHandle]>,
}

impl Pool {
    /// Execute the main function in a pool of M:N schedulers.
    ///
    /// This will configure the pool according to the `config` parameter, and
    /// initially run `main` inside the pool of schedulers.
    pub fn spawn(config: Config, main: proc()) -> Pool {
        static mut POOL_ID: AtomicUint = INIT_ATOMIC_UINT;

        let Config {
            threads: nscheds,
            use_main_thread: use_main_sched,
            shutdown_after_main_exits
        } = config;

        let mut main = Some(main);
        let pool_id = unsafe { POOL_ID.fetch_add(1, SeqCst) };

        // The shared list of sleeping schedulers.
        let sleepers = SleeperList::new();

        // Create a work queue for each scheduler, ntimes. Create an extra
        // for the main thread if that flag is set. We won't steal from it.
        let mut pool = deque::BufferPool::new();
        let arr = vec::from_fn(nscheds, |_| pool.deque());
        let (workers, stealers) = vec::unzip(arr.move_iter());

        // The schedulers.
        let mut scheds = ~[];
        // Handles to the schedulers. When the main task ends these will be
        // sent the Shutdown message to terminate the schedulers.
        let mut handles = ~[];

        for worker in workers.move_iter() {
            rtdebug!("inserting a regular scheduler");

            // Every scheduler is driven by an I/O event loop.
            let loop_ = new_event_loop();
            let mut sched = ~Scheduler::new(pool_id,
                                            loop_,
                                            worker,
                                            stealers.clone(),
                                            sleepers.clone());
            let handle = sched.make_handle();

            scheds.push(sched);
            handles.push(handle);
        }

        // If we need a main-thread task then create a main thread scheduler
        // that will reject any task that isn't pinned to it
        let main_sched = if use_main_sched {

            // Create a friend handle.
            let mut friend_sched = scheds.pop();
            let friend_handle = friend_sched.make_handle();
            scheds.push(friend_sched);

            // This scheduler needs a queue that isn't part of the stealee
            // set.
            let (worker, _) = pool.deque();

            let main_loop = new_event_loop();
            let mut main_sched = ~Scheduler::new_special(pool_id,
                                                         main_loop,
                                                         worker,
                                                         stealers.clone(),
                                                         sleepers.clone(),
                                                         false,
                                                         Some(friend_handle));
            let mut main_handle = main_sched.make_handle();
            // Allow the scheduler to exit when the main task exits.
            // Note: sending the shutdown message also prevents the scheduler
            // from pushing itself to the sleeper list, which is used for
            // waking up schedulers for work stealing; since this is a
            // non-work-stealing scheduler it should not be adding itself
            // to the list.
            main_handle.send(Shutdown);
            Some(main_sched)
        } else {
            None
        };

        // The pool of schedulers that will be returned from this function
        let mut pool = Pool { threads: ~[], handles: None };

        // When the main task exits, after all the tasks in the main
        // task tree, shut down the schedulers and set the exit code.
        let mut on_exit = if shutdown_after_main_exits {
            let handles = handles;
            Some(proc(exit_success: TaskResult) {
                let mut handles = handles;
                for handle in handles.mut_iter() {
                    handle.send(Shutdown);
                }
                if exit_success.is_err() {
                    os::set_exit_status(rt::DEFAULT_ERROR_CODE);
                }
            })
        } else {
            pool.handles = Some(handles);
            None
        };

        if !use_main_sched {

            // In the case where we do not use a main_thread scheduler we
            // run the main task in one of our threads.

            let mut main = GreenTask::new(&mut scheds[0].stack_pool, None,
                                          main.take_unwrap());
            let mut main_task = ~Task::new();
            main_task.name = Some(SendStrStatic("<main>"));
            main_task.death.on_exit = on_exit.take();
            main.put_task(main_task);

            let sched = scheds.pop();
            let main = main;
            let thread = do Thread::start {
                sched.bootstrap(main);
            };
            pool.threads.push(thread);
        }

        // Run each remaining scheduler in a thread.
        for sched in scheds.move_rev_iter() {
            rtdebug!("creating regular schedulers");
            let thread = do Thread::start {
                let mut sched = sched;
                let mut task = do GreenTask::new(&mut sched.stack_pool, None) {
                    rtdebug!("boostraping a non-primary scheduler");
                };
                task.put_task(~Task::new());
                sched.bootstrap(task);
            };
            pool.threads.push(thread);
        }

        // If we do have a main thread scheduler, run it now.

        if use_main_sched {
            rtdebug!("about to create the main scheduler task");

            let mut main_sched = main_sched.unwrap();

            let home = HomeSched(main_sched.make_handle());
            let mut main = GreenTask::new_homed(&mut main_sched.stack_pool, None,
                                                home, main.take_unwrap());
            let mut main_task = ~Task::new();
            main_task.name = Some(SendStrStatic("<main>"));
            main_task.death.on_exit = on_exit.take();
            main.put_task(main_task);
            rtdebug!("bootstrapping main_task");

            main_sched.bootstrap(main);
        }

        return pool;
    }

    /// Waits for the pool of schedulers to exit. If the pool was spawned to
    /// shutdown after the main task exits, this will simply wait for all the
    /// scheudlers to exit. If the pool was not spawned like that, this function
    /// will trigger shutdown of all the active schedulers. The schedulers will
    /// exit once all tasks in this pool of schedulers has exited.
    pub fn wait(&mut self) {
        match self.handles.take() {
            Some(mut handles) => {
                for handle in handles.mut_iter() {
                    handle.send(Shutdown);
                }
            }
            None => {}
        }

        for thread in util::replace(&mut self.threads, ~[]).move_iter() {
            thread.join();
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
