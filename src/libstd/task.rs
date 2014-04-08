// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Utilities for managing and scheduling tasks
 *
 * An executing Rust program consists of a collection of tasks, each with their
 * own stack, and sole ownership of their allocated heap data. Tasks communicate
 * with each other using channels (see `std::comm` for more info about how
 * communication works).
 *
 * Failure in one task does not propagate to any others (not to parent, not to
 * child).  Failure propagation is instead handled by using the channel send()
 * and recv() methods which will fail if the other end has hung up already.
 *
 * Task Scheduling:
 *
 * By default, every task is created with the same "flavor" as the calling task.
 * This flavor refers to the scheduling mode, with two possibilities currently
 * being 1:1 and M:N modes. Green (M:N) tasks are cooperatively scheduled and
 * native (1:1) tasks are scheduled by the OS kernel.
 *
 * # Example
 *
 * ```rust
 * spawn(proc() {
 *     println!("Hello, World!");
 * })
 * ```
 */

use any::Any;
use comm::{Sender, Receiver, channel};
use io::Writer;
use kinds::{Send, marker};
use option::{None, Some, Option};
use result::{Result, Ok, Err};
use rt::local::Local;
use rt::task::Task;
use str::{Str, SendStr, IntoMaybeOwned};

#[cfg(test)] use any::{AnyOwnExt, AnyRefExt};
#[cfg(test)] use result;

/// Indicates the manner in which a task exited.
///
/// A task that completes without failing is considered to exit successfully.
///
/// If you wish for this result's delivery to block until all
/// children tasks complete, recommend using a result future.
pub type TaskResult = Result<(), ~Any:Send>;

/// Task configuration options
pub struct TaskOpts {
    /// Enable lifecycle notifications on the given channel
    pub notify_chan: Option<Sender<TaskResult>>,
    /// A name for the task-to-be, for identification in failure messages
    pub name: Option<SendStr>,
    /// The size of the stack for the spawned task
    pub stack_size: Option<uint>,
    /// Task-local stdout
    pub stdout: Option<~Writer:Send>,
    /// Task-local stderr
    pub stderr: Option<~Writer:Send>,
}

/**
 * The task builder type.
 *
 * Provides detailed control over the properties and behavior of new tasks.
 */
// NB: Builders are designed to be single-use because they do stateful
// things that get weird when reusing - e.g. if you create a result future
// it only applies to a single task, so then you have to maintain Some
// potentially tricky state to ensure that everything behaves correctly
// when you try to reuse the builder to spawn a new task. We'll just
// sidestep that whole issue by making builders uncopyable and making
// the run function move them in.
pub struct TaskBuilder {
    /// Options to spawn the new task with
    pub opts: TaskOpts,
    gen_body: Option<proc(v: proc():Send):Send -> proc():Send>,
    nocopy: Option<marker::NoCopy>,
}

/**
 * Generate the base configuration for spawning a task, off of which more
 * configuration methods can be chained.
 */
pub fn task() -> TaskBuilder {
    TaskBuilder {
        opts: TaskOpts::new(),
        gen_body: None,
        nocopy: None,
    }
}

impl TaskBuilder {
    /// Get a future representing the exit status of the task.
    ///
    /// Taking the value of the future will block until the child task
    /// terminates. The future result return value will be created *before* the task is
    /// spawned; as such, do not invoke .get() on it directly;
    /// rather, store it in an outer variable/list for later use.
    ///
    /// # Failure
    /// Fails if a future_result was already set for this task.
    pub fn future_result(&mut self) -> Receiver<TaskResult> {
        // FIXME (#3725): Once linked failure and notification are
        // handled in the library, I can imagine implementing this by just
        // registering an arbitrary number of task::on_exit handlers and
        // sending out messages.

        if self.opts.notify_chan.is_some() {
            fail!("Can't set multiple future_results for one task!");
        }

        // Construct the future and give it to the caller.
        let (tx, rx) = channel();

        // Reconfigure self to use a notify channel.
        self.opts.notify_chan = Some(tx);

        rx
    }

    /// Name the task-to-be. Currently the name is used for identification
    /// only in failure messages.
    pub fn named<S: IntoMaybeOwned<'static>>(mut self, name: S) -> TaskBuilder {
        self.opts.name = Some(name.into_maybe_owned());
        self
    }

    /**
     * Add a wrapper to the body of the spawned task.
     *
     * Before the task is spawned it is passed through a 'body generator'
     * function that may perform local setup operations as well as wrap
     * the task body in remote setup operations. With this the behavior
     * of tasks can be extended in simple ways.
     *
     * This function augments the current body generator with a new body
     * generator by applying the task body which results from the
     * existing body generator to the new body generator.
     */
    pub fn with_wrapper(mut self,
                        wrapper: proc(v: proc():Send):Send -> proc():Send)
        -> TaskBuilder
    {
        self.gen_body = match self.gen_body.take() {
            Some(prev) => Some(proc(body) { wrapper(prev(body)) }),
            None => Some(wrapper)
        };
        self
    }

    /**
     * Creates and executes a new child task
     *
     * Sets up a new task with its own call stack and schedules it to run
     * the provided unique closure. The task has the properties and behavior
     * specified by the task_builder.
     */
    pub fn spawn(mut self, f: proc():Send) {
        let gen_body = self.gen_body.take();
        let f = match gen_body {
            Some(gen) => gen(f),
            None => f
        };
        let t: ~Task = Local::take();
        t.spawn_sibling(self.opts, f);
    }

    /**
     * Execute a function in another task and return either the return value
     * of the function or result::err.
     *
     * # Return value
     *
     * If the function executed successfully then try returns result::ok
     * containing the value returned by the function. If the function fails
     * then try returns result::err containing nil.
     *
     * # Failure
     * Fails if a future_result was already set for this task.
     */
    pub fn try<T:Send>(mut self, f: proc():Send -> T) -> Result<T, ~Any:Send> {
        let (tx, rx) = channel();

        let result = self.future_result();

        self.spawn(proc() {
            tx.send(f());
        });

        match result.recv() {
            Ok(())     => Ok(rx.recv()),
            Err(cause) => Err(cause)
        }
    }
}

/* Task construction */

impl TaskOpts {
    pub fn new() -> TaskOpts {
        /*!
         * The default task options
         */

        TaskOpts {
            notify_chan: None,
            name: None,
            stack_size: None,
            stdout: None,
            stderr: None,
        }
    }
}

/* Spawn convenience functions */

/// Creates and executes a new child task
///
/// Sets up a new task with its own call stack and schedules it to run
/// the provided unique closure.
///
/// This function is equivalent to `task().spawn(f)`.
pub fn spawn(f: proc():Send) {
    let task = task();
    task.spawn(f)
}

pub fn try<T:Send>(f: proc():Send -> T) -> Result<T, ~Any:Send> {
    /*!
     * Execute a function in another task and return either the return value
     * of the function or result::err.
     *
     * This is equivalent to task().try.
     */

    let task = task();
    task.try(f)
}


/* Lifecycle functions */

/// Read the name of the current task.
pub fn with_task_name<U>(blk: |Option<&str>| -> U) -> U {
    use rt::task::Task;

    let mut task = Local::borrow(None::<Task>);
    match task.get().name {
        Some(ref name) => blk(Some(name.as_slice())),
        None => blk(None)
    }
}

pub fn deschedule() {
    //! Yield control to the task scheduler

    use rt::local::Local;

    // FIXME(#7544): Optimize this, since we know we won't block.
    let task: ~Task = Local::take();
    task.yield_now();
}

pub fn failing() -> bool {
    //! True if the running task has failed

    use rt::task::Task;

    let mut local = Local::borrow(None::<Task>);
    local.get().unwinder.unwinding()
}

// The following 8 tests test the following 2^3 combinations:
// {un,}linked {un,}supervised failure propagation {up,down}wards.

// !!! These tests are dangerous. If Something is buggy, they will hang, !!!
// !!! instead of exiting cleanly. This might wedge the buildbots.       !!!

#[test]
fn test_unnamed_task() {
    spawn(proc() {
        with_task_name(|name| {
            assert!(name.is_none());
        })
    })
}

#[test]
fn test_owned_named_task() {
    task().named(~"ada lovelace").spawn(proc() {
        with_task_name(|name| {
            assert!(name.unwrap() == "ada lovelace");
        })
    })
}

#[test]
fn test_static_named_task() {
    task().named("ada lovelace").spawn(proc() {
        with_task_name(|name| {
            assert!(name.unwrap() == "ada lovelace");
        })
    })
}

#[test]
fn test_send_named_task() {
    task().named("ada lovelace".into_maybe_owned()).spawn(proc() {
        with_task_name(|name| {
            assert!(name.unwrap() == "ada lovelace");
        })
    })
}

#[test]
fn test_run_basic() {
    let (tx, rx) = channel();
    task().spawn(proc() {
        tx.send(());
    });
    rx.recv();
}

#[test]
fn test_with_wrapper() {
    let (tx, rx) = channel();
    task().with_wrapper(proc(body) {
        let result: proc():Send = proc() {
            body();
            tx.send(());
        };
        result
    }).spawn(proc() { });
    rx.recv();
}

#[test]
fn test_future_result() {
    let mut builder = task();
    let result = builder.future_result();
    builder.spawn(proc() {});
    assert!(result.recv().is_ok());

    let mut builder = task();
    let result = builder.future_result();
    builder.spawn(proc() {
        fail!();
    });
    assert!(result.recv().is_err());
}

#[test] #[should_fail]
fn test_back_to_the_future_result() {
    let mut builder = task();
    builder.future_result();
    builder.future_result();
}

#[test]
fn test_try_success() {
    match try(proc() {
        ~"Success!"
    }).as_ref().map(|s| s.as_slice()) {
        result::Ok("Success!") => (),
        _ => fail!()
    }
}

#[test]
fn test_try_fail() {
    match try(proc() {
        fail!()
    }) {
        result::Err(_) => (),
        result::Ok(()) => fail!()
    }
}

#[test]
fn test_spawn_sched() {
    use clone::Clone;

    let (tx, rx) = channel();

    fn f(i: int, tx: Sender<()>) {
        let tx = tx.clone();
        spawn(proc() {
            if i == 0 {
                tx.send(());
            } else {
                f(i - 1, tx);
            }
        });

    }
    f(10, tx);
    rx.recv();
}

#[test]
fn test_spawn_sched_childs_on_default_sched() {
    let (tx, rx) = channel();

    spawn(proc() {
        spawn(proc() {
            tx.send(());
        });
    });

    rx.recv();
}

#[cfg(test)]
fn avoid_copying_the_body(spawnfn: |v: proc():Send|) {
    let (tx, rx) = channel::<uint>();

    let x = ~1;
    let x_in_parent = (&*x) as *int as uint;

    spawnfn(proc() {
        let x_in_child = (&*x) as *int as uint;
        tx.send(x_in_child);
    });

    let x_in_child = rx.recv();
    assert_eq!(x_in_parent, x_in_child);
}

#[test]
fn test_avoid_copying_the_body_spawn() {
    avoid_copying_the_body(spawn);
}

#[test]
fn test_avoid_copying_the_body_task_spawn() {
    avoid_copying_the_body(|f| {
        let builder = task();
        builder.spawn(proc() {
            f();
        });
    })
}

#[test]
fn test_avoid_copying_the_body_try() {
    avoid_copying_the_body(|f| {
        let _ = try(proc() {
            f()
        });
    })
}

#[test]
fn test_child_doesnt_ref_parent() {
    // If the child refcounts the parent task, this will stack overflow when
    // climbing the task tree to dereference each ancestor. (See #1789)
    // (well, it would if the constant were 8000+ - I lowered it to be more
    // valgrind-friendly. try this at home, instead..!)
    static generations: uint = 16;
    fn child_no(x: uint) -> proc():Send {
        return proc() {
            if x < generations {
                task().spawn(child_no(x+1));
            }
        }
    }
    task().spawn(child_no(0));
}

#[test]
fn test_simple_newsched_spawn() {
    spawn(proc()())
}

#[test]
fn test_try_fail_message_static_str() {
    match try(proc() {
        fail!("static string");
    }) {
        Err(e) => {
            type T = &'static str;
            assert!(e.is::<T>());
            assert_eq!(*e.move::<T>().unwrap(), "static string");
        }
        Ok(()) => fail!()
    }
}

#[test]
fn test_try_fail_message_owned_str() {
    match try(proc() {
        fail!(~"owned string");
    }) {
        Err(e) => {
            type T = ~str;
            assert!(e.is::<T>());
            assert_eq!(*e.move::<T>().unwrap(), ~"owned string");
        }
        Ok(()) => fail!()
    }
}

#[test]
fn test_try_fail_message_any() {
    match try(proc() {
        fail!(~413u16 as ~Any:Send);
    }) {
        Err(e) => {
            type T = ~Any:Send;
            assert!(e.is::<T>());
            let any = e.move::<T>().unwrap();
            assert!(any.is::<u16>());
            assert_eq!(*any.move::<u16>().unwrap(), 413u16);
        }
        Ok(()) => fail!()
    }
}

#[test]
fn test_try_fail_message_unit_struct() {
    struct Juju;

    match try(proc() {
        fail!(Juju)
    }) {
        Err(ref e) if e.is::<Juju>() => {}
        Err(_) | Ok(()) => fail!()
    }
}
