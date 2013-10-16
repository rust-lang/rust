// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! The Rust Runtime, including the task scheduler and I/O

The `rt` module provides the private runtime infrastructure necessary
to support core language features like the exchange and local heap,
the garbage collector, logging, local data and unwinding. It also
implements the default task scheduler and task model. Initialization
routines are provided for setting up runtime resources in common
configurations, including that used by `rustc` when generating
executables.

It is intended that the features provided by `rt` can be factored in a
way such that the core library can be built with different 'profiles'
for different use cases, e.g. excluding the task scheduler. A number
of runtime features though are critical to the functioning of the
language and an implementation must be provided regardless of the
execution environment.

Of foremost importance is the global exchange heap, in the module
`global_heap`. Very little practical Rust code can be written without
access to the global heap. Unlike most of `rt` the global heap is
truly a global resource and generally operates independently of the
rest of the runtime.

All other runtime features are task-local, including the local heap,
the garbage collector, local storage, logging and the stack unwinder.

The relationship between `rt` and the rest of the core library is
not entirely clear yet and some modules will be moving into or
out of `rt` as development proceeds.

Several modules in `core` are clients of `rt`:

* `std::task` - The user-facing interface to the Rust task model.
* `std::task::local_data` - The interface to local data.
* `std::gc` - The garbage collector.
* `std::unstable::lang` - Miscellaneous lang items, some of which rely on `std::rt`.
* `std::condition` - Uses local data.
* `std::cleanup` - Local heap destruction.
* `std::io` - In the future `std::io` will use an `rt` implementation.
* `std::logging`
* `std::pipes`
* `std::comm`
* `std::stackwalk`

*/

// XXX: this should not be here.
#[allow(missing_doc)];

use cell::Cell;
use clone::Clone;
use container::Container;
use iter::Iterator;
use option::{Option, None, Some};
use ptr::RawPtr;
use rt::local::Local;
use rt::sched::{Scheduler, Shutdown};
use rt::sleeper_list::SleeperList;
use rt::task::{Task, SchedTask, GreenTask, Sched};
use rt::uv::uvio::UvEventLoop;
use unstable::atomics::{AtomicInt, AtomicBool, SeqCst};
use unstable::sync::UnsafeArc;
use vec;
use vec::{OwnedVector, MutableVector, ImmutableVector};

use self::thread::Thread;
use self::work_queue::WorkQueue;

// the os module needs to reach into this helper, so allow general access
// through this reexport.
pub use self::util::set_exit_status;

// this is somewhat useful when a program wants to spawn a "reasonable" number
// of workers based on the constraints of the system that it's running on.
// Perhaps this shouldn't be a `pub use` though and there should be another
// method...
pub use self::util::default_sched_threads;

// XXX: these probably shouldn't be public...
#[doc(hidden)]
pub mod shouldnt_be_public {
    pub use super::sched::Scheduler;
    pub use super::kill::KillHandle;
    pub use super::thread::Thread;
    pub use super::work_queue::WorkQueue;
    pub use super::select::SelectInner;
    pub use super::rtio::EventLoop;
    pub use super::select::{SelectInner, SelectPortInner};
    pub use super::local_ptr::maybe_tls_key;
}

// Internal macros used by the runtime.
mod macros;

/// The global (exchange) heap.
pub mod global_heap;

/// Implementations of language-critical runtime features like @.
pub mod task;

/// Facilities related to task failure, killing, and death.
mod kill;

/// The coroutine task scheduler, built on the `io` event loop.
mod sched;

/// Synchronous I/O.
pub mod io;

/// The EventLoop and internal synchronous I/O interface.
mod rtio;

/// libuv and default rtio implementation.
pub mod uv;

/// The Local trait for types that are accessible via thread-local
/// or task-local storage.
pub mod local;

/// A parallel work-stealing deque.
mod work_queue;

/// A parallel queue.
mod message_queue;

/// A parallel data structure for tracking sleeping schedulers.
mod sleeper_list;

/// Stack segments and caching.
pub mod stack;

/// CPU context swapping.
mod context;

/// Bindings to system threading libraries.
mod thread;

/// The runtime configuration, read from environment variables.
pub mod env;

/// The local, managed heap
pub mod local_heap;

/// The Logger trait and implementations
pub mod logging;

/// Crate map
pub mod crate_map;

/// Tools for testing the runtime
pub mod test;

/// Reference counting
pub mod rc;

/// A simple single-threaded channel type for passing buffered data between
/// scheduler and task context
pub mod tube;

/// Simple reimplementation of std::comm
pub mod comm;

mod select;

/// The runtime needs to be able to put a pointer into thread-local storage.
mod local_ptr;

/// Bindings to pthread/windows thread-local storage.
mod thread_local_storage;

/// Just stuff
mod util;

// Global command line argument storage
pub mod args;

// Support for dynamic borrowck
pub mod borrowck;

/// Set up a default runtime configuration, given compiler-supplied arguments.
///
/// This is invoked by the `start` _language item_ (unstable::lang) to
/// run a Rust executable.
///
/// # Arguments
///
/// * `argc` & `argv` - The argument vector. On Unix this information is used
///   by os::args.
///
/// # Return value
///
/// The return value is used as the process return code. 0 on success, 101 on error.
pub fn start(argc: int, argv: **u8, main: ~fn()) -> int {

    init(argc, argv);
    let exit_code = run(main);
    cleanup();

    return exit_code;
}

/// Like `start` but creates an additional scheduler on the current thread,
/// which in most cases will be the 'main' thread, and pins the main task to it.
///
/// This is appropriate for running code that must execute on the main thread,
/// such as the platform event loop and GUI.
pub fn start_on_main_thread(argc: int, argv: **u8, main: ~fn()) -> int {
    init(argc, argv);
    let exit_code = run_on_main_thread(main);
    cleanup();

    return exit_code;
}

/// One-time runtime initialization.
///
/// Initializes global state, including frobbing
/// the crate's logging flags, registering GC
/// metadata, and storing the process arguments.
pub fn init(argc: int, argv: **u8) {
    // XXX: Derefing these pointers is not safe.
    // Need to propagate the unsafety to `start`.
    unsafe {
        args::init(argc, argv);
        env::init();
        logging::init();
    }
}

/// One-time runtime cleanup.
pub fn cleanup() {
    args::cleanup();
}

/// Execute the main function in a scheduler.
///
/// Configures the runtime according to the environment, by default
/// using a task scheduler with the same number of threads as cores.
/// Returns a process exit code.
pub fn run(main: ~fn()) -> int {
    run_(main, false)
}

pub fn run_on_main_thread(main: ~fn()) -> int {
    run_(main, true)
}

fn run_(main: ~fn(), use_main_sched: bool) -> int {
    static DEFAULT_ERROR_CODE: int = 101;

    let nscheds = util::default_sched_threads();

    let main = Cell::new(main);

    // The shared list of sleeping schedulers.
    let sleepers = SleeperList::new();

    // Create a work queue for each scheduler, ntimes. Create an extra
    // for the main thread if that flag is set. We won't steal from it.
    let work_queues: ~[WorkQueue<~Task>] = vec::from_fn(nscheds, |_| WorkQueue::new());

    // The schedulers.
    let mut scheds = ~[];
    // Handles to the schedulers. When the main task ends these will be
    // sent the Shutdown message to terminate the schedulers.
    let mut handles = ~[];

    for work_queue in work_queues.iter() {
        rtdebug!("inserting a regular scheduler");

        // Every scheduler is driven by an I/O event loop.
        let loop_ = ~UvEventLoop::new();
        let mut sched = ~Scheduler::new(loop_,
                                        work_queue.clone(),
                                        work_queues.clone(),
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
        let work_queue = WorkQueue::new();

        let main_loop = ~UvEventLoop::new();
        let mut main_sched = ~Scheduler::new_special(main_loop,
                                                     work_queue,
                                                     work_queues.clone(),
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
        main_handle.send_shutdown();
        Some(main_sched)
    } else {
        None
    };

    // Create a shared cell for transmitting the process exit
    // code from the main task to this function.
    let exit_code = UnsafeArc::new(AtomicInt::new(0));
    let exit_code_clone = exit_code.clone();

    // Used to sanity check that the runtime only exits once
    let exited_already = UnsafeArc::new(AtomicBool::new(false));

    // When the main task exits, after all the tasks in the main
    // task tree, shut down the schedulers and set the exit code.
    let handles = Cell::new(handles);
    let on_exit: ~fn(bool) = |exit_success| {
        unsafe {
            assert!(!(*exited_already.get()).swap(true, SeqCst),
                    "the runtime already exited");
        }

        let mut handles = handles.take();
        for handle in handles.mut_iter() {
            handle.send(Shutdown);
        }

        unsafe {
            let exit_code = if exit_success {
                use rt::util;

                // If we're exiting successfully, then return the global
                // exit status, which can be set programmatically.
                util::get_exit_status()
            } else {
                DEFAULT_ERROR_CODE
            };
            (*exit_code_clone.get()).store(exit_code, SeqCst);
        }
    };

    let mut threads = ~[];

    let on_exit = Cell::new(on_exit);

    if !use_main_sched {

        // In the case where we do not use a main_thread scheduler we
        // run the main task in one of our threads.

        let mut main_task = ~Task::new_root(&mut scheds[0].stack_pool, None, main.take());
        main_task.death.on_exit = Some(on_exit.take());
        let main_task_cell = Cell::new(main_task);

        let sched = scheds.pop();
        let sched_cell = Cell::new(sched);
        let thread = do Thread::start {
            let sched = sched_cell.take();
            sched.bootstrap(main_task_cell.take());
        };
        threads.push(thread);
    }

    // Run each remaining scheduler in a thread.
    for sched in scheds.move_rev_iter() {
        rtdebug!("creating regular schedulers");
        let sched_cell = Cell::new(sched);
        let thread = do Thread::start {
            let mut sched = sched_cell.take();
            let bootstrap_task = ~do Task::new_root(&mut sched.stack_pool, None) || {
                rtdebug!("boostraping a non-primary scheduler");
            };
            sched.bootstrap(bootstrap_task);
        };
        threads.push(thread);
    }

    // If we do have a main thread scheduler, run it now.

    if use_main_sched {

        rtdebug!("about to create the main scheduler task");

        let mut main_sched = main_sched.unwrap();

        let home = Sched(main_sched.make_handle());
        let mut main_task = ~Task::new_root_homed(&mut main_sched.stack_pool, None,
                                                  home, main.take());
        main_task.death.on_exit = Some(on_exit.take());
        rtdebug!("bootstrapping main_task");

        main_sched.bootstrap(main_task);
    }

    rtdebug!("waiting for threads");

    // Wait for schedulers
    for thread in threads.move_iter() {
        thread.join();
    }

    // Return the exit code
    unsafe {
        (*exit_code.get()).load(SeqCst)
    }
}

pub fn in_sched_context() -> bool {
    unsafe {
        let task_ptr: Option<*mut Task> = Local::try_unsafe_borrow();
        match task_ptr {
            Some(task) => {
                match (*task).task_type {
                    SchedTask => true,
                    _ => false
                }
            }
            None => false
        }
    }
}

pub fn in_green_task_context() -> bool {
    unsafe {
        let task: Option<*mut Task> = Local::try_unsafe_borrow();
        match task {
            Some(task) => {
                match (*task).task_type {
                    GreenTask(_) => true,
                    _ => false
                }
            }
            None => false
        }
    }
}
