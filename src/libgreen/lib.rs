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
//! stack-allocation strategy. This can be optionally linked in to rust
//! programs in order to provide M:N functionality inside of 1:1 programs.
//!
//! # Architecture
//!
//! An M:N scheduling library implies that there are N OS thread upon which M
//! "green threads" are multiplexed. In other words, a set of green threads are
//! all run inside a pool of OS threads.
//!
//! With this design, you can achieve _concurrency_ by spawning many green
//! threads, and you can achieve _parallelism_ by running the green threads
//! simultaneously on multiple OS threads. Each OS thread is a candidate for
//! being scheduled on a different core (the source of parallelism), and then
//! all of the green threads cooperatively schedule amongst one another (the
//! source of concurrency).
//!
//! ## Schedulers
//!
//! In order to coordinate among green threads, each OS thread is primarily
//! running something which we call a Scheduler. Whenever a reference to a
//! Scheduler is made, it is synonymous to referencing one OS thread. Each
//! scheduler is bound to one and exactly one OS thread, and the thread that it
//! is bound to never changes.
//!
//! Each scheduler is connected to a pool of other schedulers (a `SchedPool`)
//! which is the thread pool term from above. A pool of schedulers all share the
//! work that they create. Furthermore, whenever a green thread is created (also
//! synonymously referred to as a green task), it is associated with a
//! `SchedPool` forevermore. A green thread cannot leave its scheduler pool.
//!
//! Schedulers can have at most one green thread running on them at a time. When
//! a scheduler is asleep on its event loop, there are no green tasks running on
//! the OS thread or the scheduler. The term "context switch" is used for when
//! the running green thread is swapped out, but this simply changes the one
//! green thread which is running on the scheduler.
//!
//! ## Green Threads
//!
//! A green thread can largely be summarized by a stack and a register context.
//! Whenever a green thread is spawned, it allocates a stack, and then prepares
//! a register context for execution. The green task may be executed across
//! multiple OS threads, but it will always use the same stack and it will carry
//! its register context across OS threads.
//!
//! Each green thread is cooperatively scheduled with other green threads.
//! Primarily, this means that there is no pre-emption of a green thread. The
//! major consequence of this design is that a green thread stuck in an infinite
//! loop will prevent all other green threads from running on that particular
//! scheduler.
//!
//! Scheduling events for green threads occur on communication and I/O
//! boundaries. For example, if a green task blocks waiting for a message on a
//! channel some other green thread can now run on the scheduler. This also has
//! the consequence that until a green thread performs any form of scheduling
//! event, it will be running on the same OS thread (unconditionally).
//!
//! ## Work Stealing
//!
//! With a pool of schedulers, a new green task has a number of options when
//! deciding where to run initially. The current implementation uses a concept
//! called work stealing in order to spread out work among schedulers.
//!
//! In a work-stealing model, each scheduler maintains a local queue of tasks to
//! run, and this queue is stolen from by other schedulers. Implementation-wise,
//! work stealing has some hairy parts, but from a user-perspective, work
//! stealing simply implies what with M green threads and N schedulers where
//! M > N it is very likely that all schedulers will be busy executing work.
//!
//! # Considerations when using libgreen
//!
//! An M:N runtime has both pros and cons, and there is no one answer as to
//! whether M:N or 1:1 is appropriate to use. As always, there are many
//! advantages and disadvantages between the two. Regardless of the workload,
//! however, there are some aspects of using green thread which you should be
//! aware of:
//!
//! * The largest concern when using libgreen is interoperating with native
//!   code. Care should be taken when calling native code that will block the OS
//!   thread as it will prevent further green tasks from being scheduled on the
//!   OS thread.
//!
//! * Native code using thread-local-storage should be approached
//!   with care. Green threads may migrate among OS threads at any time, so
//!   native libraries using thread-local state may not always work.
//!
//! * Native synchronization primitives (e.g. pthread mutexes) will also not
//!   work for green threads. The reason for this is because native primitives
//!   often operate on a _os thread_ granularity whereas green threads are
//!   operating on a more granular unit of work.
//!
//! * A green threading runtime is not fork-safe. If the process forks(), it
//!   cannot expect to make reasonable progress by continuing to use green
//!   threads.
//!
//! Note that these concerns do not mean that operating with native code is a
//! lost cause. These are simply just concerns which should be considered when
//! invoking native code.
//!
//! # Starting with libgreen
//!
//! ```rust
//! extern crate green;
//!
//! #[start]
//! fn start(argc: int, argv: *const *const u8) -> int {
//!     green::start(argc, argv, green::basic::event_loop, main)
//! }
//!
//! fn main() {
//!     // this code is running in a pool of schedulers
//! }
//! ```
//!
//! > **Note**: This `main` function in this example does *not* have I/O
//! >           support. The basic event loop does not provide any support
//!
//! # Starting with I/O support in libgreen
//!
//! ```rust
//! extern crate green;
//! extern crate rustuv;
//!
//! #[start]
//! fn start(argc: int, argv: *const *const u8) -> int {
//!     green::start(argc, argv, rustuv::event_loop, main)
//! }
//!
//! fn main() {
//!     // this code is running in a pool of schedulers all powered by libuv
//! }
//! ```
//!
//! The above code can also be shortened with a macro from libgreen.
//!
//! ```
//! #![feature(phase)]
//! #[phase(plugin)] extern crate green;
//!
//! green_start!(main)
//!
//! fn main() {
//!     // run inside of a green pool
//! }
//! ```
//!
//! # Using a scheduler pool
//!
//! This library adds a `GreenTaskBuilder` trait that extends the methods
//! available on `std::task::TaskBuilder` to allow spawning a green task,
//! possibly pinned to a particular scheduler thread:
//!
//! ```rust
//! use std::task::TaskBuilder;
//! use green::{SchedPool, PoolConfig, GreenTaskBuilder};
//!
//! let config = PoolConfig::new();
//! let mut pool = SchedPool::new(config);
//!
//! // Spawn tasks into the pool of schedulers
//! TaskBuilder::new().green(&mut pool).spawn(proc() {
//!     // this code is running inside the pool of schedulers
//!
//!     spawn(proc() {
//!         // this code is also running inside the same scheduler pool
//!     });
//! });
//!
//! // Dynamically add a new scheduler to the scheduler pool. This adds another
//! // OS thread that green threads can be multiplexed on to.
//! let mut handle = pool.spawn_sched();
//!
//! // Pin a task to the spawned scheduler
//! TaskBuilder::new().green_pinned(&mut pool, &mut handle).spawn(proc() {
//!     /* ... */
//! });
//!
//! // Handles keep schedulers alive, so be sure to drop all handles before
//! // destroying the sched pool
//! drop(handle);
//!
//! // Required to shut down this scheduler pool.
//! // The task will fail if `shutdown` is not called.
//! pool.shutdown();
//! ```

#![crate_id = "green#0.11.0-pre"]
#![experimental]
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/",
       html_playground_url = "http://play.rust-lang.org/")]

// NB this does *not* include globs, please keep it that way.
#![feature(macro_rules, phase)]
#![allow(visible_private_types)]
#![allow(deprecated)]
#![feature(default_type_params)]

#[cfg(test)] #[phase(plugin, link)] extern crate log;
#[cfg(test)] extern crate rustuv;
extern crate libc;
extern crate alloc;

use alloc::arc::Arc;
use std::mem::replace;
use std::os;
use std::rt::rtio;
use std::rt::thread::Thread;
use std::rt::task::TaskOpts;
use std::rt;
use std::sync::atomics::{SeqCst, AtomicUint, INIT_ATOMIC_UINT};
use std::sync::deque;
use std::task::{TaskBuilder, Spawner};

use sched::{Shutdown, Scheduler, SchedHandle, TaskFromFriend, PinnedTask, NewNeighbor};
use sleeper_list::SleeperList;
use stack::StackPool;
use task::GreenTask;

mod macros;
mod simple;
mod message_queue;

pub mod basic;
pub mod context;
pub mod coroutine;
pub mod sched;
pub mod sleeper_list;
pub mod stack;
pub mod task;

/// A helper macro for booting a program with libgreen
///
/// # Example
///
/// ```
/// #![feature(phase)]
/// #[phase(plugin)] extern crate green;
///
/// green_start!(main)
///
/// fn main() {
///     // running with libgreen
/// }
/// ```
#[macro_export]
macro_rules! green_start( ($f:ident) => (
    mod __start {
        extern crate green;
        extern crate rustuv;

        #[start]
        fn start(argc: int, argv: *const *const u8) -> int {
            green::start(argc, argv, rustuv::event_loop, super::$f)
        }
    }
) )

/// Set up a default runtime configuration, given compiler-supplied arguments.
///
/// This function will block until the entire pool of M:N schedulers have
/// exited. This function also requires a local task to be available.
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
pub fn start(argc: int, argv: *const *const u8,
             event_loop_factory: fn() -> Box<rtio::EventLoop + Send>,
             main: proc():Send) -> int {
    rt::init(argc, argv);
    let mut main = Some(main);
    let mut ret = None;
    simple::task().run(|| {
        ret = Some(run(event_loop_factory, main.take_unwrap()));
    }).destroy();
    // unsafe is ok b/c we're sure that the runtime is gone
    unsafe { rt::cleanup() }
    ret.unwrap()
}

/// Execute the main function in a pool of M:N schedulers.
///
/// Configures the runtime according to the environment, by default using a task
/// scheduler with the same number of threads as cores.  Returns a process exit
/// code.
///
/// This function will not return until all schedulers in the associated pool
/// have returned.
pub fn run(event_loop_factory: fn() -> Box<rtio::EventLoop + Send>,
           main: proc():Send) -> int {
    // Create a scheduler pool and spawn the main task into this pool. We will
    // get notified over a channel when the main task exits.
    let mut cfg = PoolConfig::new();
    cfg.event_loop_factory = event_loop_factory;
    let mut pool = SchedPool::new(cfg);
    let (tx, rx) = channel();
    let mut opts = TaskOpts::new();
    opts.on_exit = Some(proc(r) tx.send(r));
    opts.name = Some("<main>".into_maybe_owned());
    pool.spawn(opts, main);

    // Wait for the main task to return, and set the process error code
    // appropriately.
    if rx.recv().is_err() {
        os::set_exit_status(rt::DEFAULT_ERROR_CODE);
    }

    // Now that we're sure all tasks are dead, shut down the pool of schedulers,
    // waiting for them all to return.
    pool.shutdown();
    os::get_exit_status()
}

/// Configuration of how an M:N pool of schedulers is spawned.
pub struct PoolConfig {
    /// The number of schedulers (OS threads) to spawn into this M:N pool.
    pub threads: uint,
    /// A factory function used to create new event loops. If this is not
    /// specified then the default event loop factory is used.
    pub event_loop_factory: fn() -> Box<rtio::EventLoop + Send>,
}

impl PoolConfig {
    /// Returns the default configuration, as determined the environment
    /// variables of this process.
    pub fn new() -> PoolConfig {
        PoolConfig {
            threads: rt::default_sched_threads(),
            event_loop_factory: basic::event_loop,
        }
    }
}

/// A structure representing a handle to a pool of schedulers. This handle is
/// used to keep the pool alive and also reap the status from the pool.
pub struct SchedPool {
    id: uint,
    threads: Vec<Thread<()>>,
    handles: Vec<SchedHandle>,
    stealers: Vec<deque::Stealer<Box<task::GreenTask>>>,
    next_friend: uint,
    stack_pool: StackPool,
    deque_pool: deque::BufferPool<Box<task::GreenTask>>,
    sleepers: SleeperList,
    factory: fn() -> Box<rtio::EventLoop + Send>,
    task_state: TaskState,
    tasks_done: Receiver<()>,
}

/// This is an internal state shared among a pool of schedulers. This is used to
/// keep track of how many tasks are currently running in the pool and then
/// sending on a channel once the entire pool has been drained of all tasks.
#[deriving(Clone)]
struct TaskState {
    cnt: Arc<AtomicUint>,
    done: Sender<()>,
}

impl SchedPool {
    /// Execute the main function in a pool of M:N schedulers.
    ///
    /// This will configure the pool according to the `config` parameter, and
    /// initially run `main` inside the pool of schedulers.
    pub fn new(config: PoolConfig) -> SchedPool {
        static mut POOL_ID: AtomicUint = INIT_ATOMIC_UINT;

        let PoolConfig {
            threads: nscheds,
            event_loop_factory: factory
        } = config;
        assert!(nscheds > 0);

        // The pool of schedulers that will be returned from this function
        let (p, state) = TaskState::new();
        let mut pool = SchedPool {
            threads: vec![],
            handles: vec![],
            stealers: vec![],
            id: unsafe { POOL_ID.fetch_add(1, SeqCst) },
            sleepers: SleeperList::new(),
            stack_pool: StackPool::new(),
            deque_pool: deque::BufferPool::new(),
            next_friend: 0,
            factory: factory,
            task_state: state,
            tasks_done: p,
        };

        // Create a work queue for each scheduler, ntimes. Create an extra
        // for the main thread if that flag is set. We won't steal from it.
        let mut workers = Vec::with_capacity(nscheds);
        let mut stealers = Vec::with_capacity(nscheds);

        for _ in range(0, nscheds) {
            let (w, s) = pool.deque_pool.deque();
            workers.push(w);
            stealers.push(s);
        }
        pool.stealers = stealers;

        // Now that we've got all our work queues, create one scheduler per
        // queue, spawn the scheduler into a thread, and be sure to keep a
        // handle to the scheduler and the thread to keep them alive.
        for worker in workers.move_iter() {
            rtdebug!("inserting a regular scheduler");

            let mut sched = box Scheduler::new(pool.id,
                                            (pool.factory)(),
                                            worker,
                                            pool.stealers.clone(),
                                            pool.sleepers.clone(),
                                            pool.task_state.clone());
            pool.handles.push(sched.make_handle());
            pool.threads.push(Thread::start(proc() { sched.bootstrap(); }));
        }

        return pool;
    }

    /// Creates a new task configured to run inside of this pool of schedulers.
    /// This is useful to create a task which can then be sent to a specific
    /// scheduler created by `spawn_sched` (and possibly pin it to that
    /// scheduler).
    #[deprecated = "use the green and green_pinned methods of GreenTaskBuilder instead"]
    pub fn task(&mut self, opts: TaskOpts, f: proc():Send) -> Box<GreenTask> {
        GreenTask::configure(&mut self.stack_pool, opts, f)
    }

    /// Spawns a new task into this pool of schedulers, using the specified
    /// options to configure the new task which is spawned.
    ///
    /// New tasks are spawned in a round-robin fashion to the schedulers in this
    /// pool, but tasks can certainly migrate among schedulers once they're in
    /// the pool.
    #[deprecated = "use the green and green_pinned methods of GreenTaskBuilder instead"]
    pub fn spawn(&mut self, opts: TaskOpts, f: proc():Send) {
        let task = self.task(opts, f);

        // Figure out someone to send this task to
        let idx = self.next_friend;
        self.next_friend += 1;
        if self.next_friend >= self.handles.len() {
            self.next_friend = 0;
        }

        // Jettison the task away!
        self.handles.get_mut(idx).send(TaskFromFriend(task));
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
        let mut sched = box Scheduler::new(self.id,
                                        (self.factory)(),
                                        worker,
                                        self.stealers.clone(),
                                        self.sleepers.clone(),
                                        self.task_state.clone());
        let ret = sched.make_handle();
        self.handles.push(sched.make_handle());
        self.threads.push(Thread::start(proc() { sched.bootstrap() }));

        return ret;
    }

    /// Consumes the pool of schedulers, waiting for all tasks to exit and all
    /// schedulers to shut down.
    ///
    /// This function is required to be called in order to drop a pool of
    /// schedulers, it is considered an error to drop a pool without calling
    /// this method.
    ///
    /// This only waits for all tasks in *this pool* of schedulers to exit, any
    /// native tasks or extern pools will not be waited on
    pub fn shutdown(mut self) {
        self.stealers = vec![];

        // Wait for everyone to exit. We may have reached a 0-task count
        // multiple times in the past, meaning there could be several buffered
        // messages on the `tasks_done` port. We're guaranteed that after *some*
        // message the current task count will be 0, so we just receive in a
        // loop until everything is totally dead.
        while self.task_state.active() {
            self.tasks_done.recv();
        }

        // Now that everyone's gone, tell everything to shut down.
        for mut handle in replace(&mut self.handles, vec![]).move_iter() {
            handle.send(Shutdown);
        }
        for thread in replace(&mut self.threads, vec![]).move_iter() {
            thread.join();
        }
    }
}

impl TaskState {
    fn new() -> (Receiver<()>, TaskState) {
        let (tx, rx) = channel();
        (rx, TaskState {
            cnt: Arc::new(AtomicUint::new(0)),
            done: tx,
        })
    }

    fn increment(&mut self) {
        self.cnt.fetch_add(1, SeqCst);
    }

    fn active(&self) -> bool {
        self.cnt.load(SeqCst) != 0
    }

    fn decrement(&mut self) {
        let prev = self.cnt.fetch_sub(1, SeqCst);
        if prev == 1 {
            self.done.send(());
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

/// A spawner for green tasks
pub struct GreenSpawner<'a>{
    pool: &'a mut SchedPool,
    handle: Option<&'a mut SchedHandle>
}

impl<'a> Spawner for GreenSpawner<'a> {
    #[inline]
    fn spawn(self, opts: TaskOpts, f: proc():Send) {
        let GreenSpawner { pool, handle } = self;
        match handle {
            None    => pool.spawn(opts, f),
            Some(h) => h.send(PinnedTask(pool.task(opts, f)))
        }
    }
}

/// An extension trait adding `green` configuration methods to `TaskBuilder`.
pub trait GreenTaskBuilder {
    fn green<'a>(self, &'a mut SchedPool) -> TaskBuilder<GreenSpawner<'a>>;
    fn green_pinned<'a>(self, &'a mut SchedPool, &'a mut SchedHandle)
                        -> TaskBuilder<GreenSpawner<'a>>;
}

impl<S: Spawner> GreenTaskBuilder for TaskBuilder<S> {
    fn green<'a>(self, pool: &'a mut SchedPool) -> TaskBuilder<GreenSpawner<'a>> {
        self.spawner(GreenSpawner {pool: pool, handle: None})
    }

    fn green_pinned<'a>(self, pool: &'a mut SchedPool, handle: &'a mut SchedHandle)
                        -> TaskBuilder<GreenSpawner<'a>> {
        self.spawner(GreenSpawner {pool: pool, handle: Some(handle)})
    }
}

#[cfg(test)]
mod test {
    use std::task::TaskBuilder;
    use super::{SchedPool, PoolConfig, GreenTaskBuilder};

    #[test]
    fn test_green_builder() {
        let mut pool = SchedPool::new(PoolConfig::new());
        let res = TaskBuilder::new().green(&mut pool).try(proc() {
            "Success!".to_string()
        });
        assert_eq!(res.ok().unwrap(), "Success!".to_string());
        pool.shutdown();
    }
}
