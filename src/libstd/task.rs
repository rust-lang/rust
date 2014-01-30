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
 * An executing Rust program consists of a tree of tasks, each with their own
 * stack, and sole ownership of their allocated heap data. Tasks communicate
 * with each other using ports and channels (see std::rt::comm for more info
 * about how communication works).
 *
 * Tasks can be spawned in 3 different modes.
 *
 *  * Bidirectionally linked: This is the default mode and it's what ```spawn``` does.
 *  Failures will be propagated from parent to child and vice versa.
 *
 *  * Unidirectionally linked (parent->child): This type of task can be created with
 *  ```spawn_supervised```. In this case, failures are propagated from parent to child
 *  but not the other way around.
 *
 *  * Unlinked: Tasks can be completely unlinked. These tasks can be created by using
 *  ```spawn_unlinked```. In this case failures are not propagated at all.
 *
 * Tasks' failure modes can be further configured. For instance, parent tasks can (un)watch
 * children failures. Please, refer to TaskBuilder's documentation bellow for more information.
 *
 * When a (bi|uni)directionally linked task fails, its failure will be propagated to all tasks
 * linked to it, this will cause such tasks to fail by a `linked failure`.
 *
 * Task Scheduling:
 *
 * By default, every task is created in the same scheduler as its parent, where it
 * is scheduled cooperatively with all other tasks in that scheduler. Some specialized
 * applications may want more control over their scheduling, in which case they can be
 * spawned into a new scheduler with the specific properties required. See TaskBuilder's
 * documentation bellow for more information.
 *
 * # Example
 *
 * ```
 * spawn(proc() {
 *     log(error, "Hello, World!");
 * })
 * ```
 */

#[allow(missing_doc)];

use any::Any;
use comm::{Chan, Port};
use io::Writer;
use kinds::Send;
use logging::Logger;
use option::{None, Some, Option};
use result::{Result, Ok, Err};
use rt::local::Local;
use rt::task::Task;
use send_str::{SendStr, IntoSendStr};
use str::Str;
use util;

#[cfg(test)] use any::{AnyOwnExt, AnyRefExt};
#[cfg(test)] use comm::SharedChan;
#[cfg(test)] use ptr;
#[cfg(test)] use result;

/// Indicates the manner in which a task exited.
///
/// A task that completes without failing is considered to exit successfully.
/// Supervised ancestors and linked siblings may yet fail after this task
/// succeeds. Also note that in such a case, it may be nondeterministic whether
/// linked failure or successful exit happen first.
///
/// If you wish for this result's delivery to block until all linked and/or
/// children tasks complete, recommend using a result future.
pub type TaskResult = Result<(), ~Any>;

/**
 * Task configuration options
 *
 * # Fields
 *
 * * watched - Make parent task collect exit status notifications from child
 *             before reporting its own exit status. (This delays the parent
 *             task's death and cleanup until after all transitively watched
 *             children also exit.) True by default.
 *
 * * notify_chan - Enable lifecycle notifications on the given channel
 *
 * * name - A name for the task-to-be, for identification in failure messages.
 *
 * * sched - Specify the configuration of a new scheduler to create the task
 *           in. This is of particular importance for libraries which want to call
 *           into foreign code that blocks. Without doing so in a different
 *           scheduler other tasks will be impeded or even blocked indefinitely.
 */
pub struct TaskOpts {
    watched: bool,
    notify_chan: Option<Chan<TaskResult>>,
    name: Option<SendStr>,
    stack_size: Option<uint>,
    logger: Option<~Logger>,
    stdout: Option<~Writer>,
    stderr: Option<~Writer>,
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
    opts: TaskOpts,
    priv gen_body: Option<proc(v: proc()) -> proc()>,
    priv can_not_copy: Option<util::NonCopyable>,
}

/**
 * Generate the base configuration for spawning a task, off of which more
 * configuration methods can be chained.
 * For example, task().unlinked().spawn is equivalent to spawn_unlinked.
 */
pub fn task() -> TaskBuilder {
    TaskBuilder {
        opts: TaskOpts::new(),
        gen_body: None,
        can_not_copy: None,
    }
}

impl TaskBuilder {
    /// Cause the parent task to collect the child's exit status (and that of
    /// all transitively-watched grandchildren) before reporting its own.
    pub fn watched(&mut self) {
        self.opts.watched = true;
    }

    /// Allow the child task to outlive the parent task, at the possible cost
    /// of the parent reporting success even if the child task fails later.
    pub fn unwatched(&mut self) {
        self.opts.watched = false;
    }

    /// Get a future representing the exit status of the task.
    ///
    /// Taking the value of the future will block until the child task
    /// terminates. The future result return value will be created *before* the task is
    /// spawned; as such, do not invoke .get() on it directly;
    /// rather, store it in an outer variable/list for later use.
    ///
    /// Note that the future returned by this function is only useful for
    /// obtaining the value of the next task to be spawning with the
    /// builder. If additional tasks are spawned with the same builder
    /// then a new result future must be obtained prior to spawning each
    /// task.
    ///
    /// # Failure
    /// Fails if a future_result was already set for this task.
    pub fn future_result(&mut self) -> Port<TaskResult> {
        // FIXME (#3725): Once linked failure and notification are
        // handled in the library, I can imagine implementing this by just
        // registering an arbitrary number of task::on_exit handlers and
        // sending out messages.

        if self.opts.notify_chan.is_some() {
            fail!("Can't set multiple future_results for one task!");
        }

        // Construct the future and give it to the caller.
        let (notify_pipe_po, notify_pipe_ch) = Chan::new();

        // Reconfigure self to use a notify channel.
        self.opts.notify_chan = Some(notify_pipe_ch);

        notify_pipe_po
    }

    /// Name the task-to-be. Currently the name is used for identification
    /// only in failure messages.
    pub fn name<S: IntoSendStr>(&mut self, name: S) {
        self.opts.name = Some(name.into_send_str());
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
    pub fn add_wrapper(&mut self, wrapper: proc(v: proc()) -> proc()) {
        let prev_gen_body = self.gen_body.take();
        let prev_gen_body = match prev_gen_body {
            Some(gen) => gen,
            None => {
                let f: proc(proc()) -> proc() = proc(body) body;
                f
            }
        };
        let next_gen_body = {
            let f: proc(proc()) -> proc() = proc(body) {
                wrapper(prev_gen_body(body))
            };
            f
        };
        self.gen_body = Some(next_gen_body);
    }

    /**
     * Creates and executes a new child task
     *
     * Sets up a new task with its own call stack and schedules it to run
     * the provided unique closure. The task has the properties and behavior
     * specified by the task_builder.
     *
     * # Failure
     *
     * When spawning into a new scheduler, the number of threads requested
     * must be greater than zero.
     */
    pub fn spawn(mut self, f: proc()) {
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
    pub fn try<T:Send>(mut self, f: proc() -> T) -> Result<T, ~Any> {
        let (po, ch) = Chan::new();

        let result = self.future_result();

        self.spawn(proc() {
            ch.send(f());
        });

        match result.recv() {
            Ok(())     => Ok(po.recv()),
            Err(cause) => Err(cause)
        }
    }
}

/* Task construction */

impl TaskOpts {
    pub fn new() -> TaskOpts {
        /*!
         * The default task options
         *
         * By default all tasks are supervised by their parent, are spawned
         * into the same scheduler, and do not post lifecycle notifications.
         */

        TaskOpts {
            watched: true,
            notify_chan: None,
            name: None,
            stack_size: None,
            logger: None,
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
pub fn spawn(f: proc()) {
    let task = task();
    task.spawn(f)
}

pub fn try<T:Send>(f: proc() -> T) -> Result<T, ~Any> {
    /*!
     * Execute a function in another task and return either the return value
     * of the function or result::err.
     *
     * This is equivalent to task().supervised().try.
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
    let mut t = task();
    t.name(~"ada lovelace");
    t.spawn(proc() {
        with_task_name(|name| {
            assert!(name.unwrap() == "ada lovelace");
        })
    })
}

#[test]
fn test_static_named_task() {
    let mut t = task();
    t.name("ada lovelace");
    t.spawn(proc() {
        with_task_name(|name| {
            assert!(name.unwrap() == "ada lovelace");
        })
    })
}

#[test]
fn test_send_named_task() {
    let mut t = task();
    t.name("ada lovelace".into_send_str());
    t.spawn(proc() {
        with_task_name(|name| {
            assert!(name.unwrap() == "ada lovelace");
        })
    })
}

#[test]
fn test_run_basic() {
    let (po, ch) = Chan::new();
    task().spawn(proc() {
        ch.send(());
    });
    po.recv();
}

#[test]
fn test_add_wrapper() {
    let (po, ch) = Chan::new();
    let mut b0 = task();
    b0.add_wrapper(proc(body) {
        let ch = ch;
        let result: proc() = proc() {
            body();
            ch.send(());
        };
        result
    });
    b0.spawn(proc() { });
    po.recv();
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
    }) {
        result::Ok(~"Success!") => (),
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

    let (po, ch) = SharedChan::new();

    fn f(i: int, ch: SharedChan<()>) {
        let ch = ch.clone();
        spawn(proc() {
            if i == 0 {
                ch.send(());
            } else {
                f(i - 1, ch);
            }
        });

    }
    f(10, ch);
    po.recv();
}

#[test]
fn test_spawn_sched_childs_on_default_sched() {
    let (po, ch) = Chan::new();

    spawn(proc() {
        let ch = ch;
        spawn(proc() {
            ch.send(());
        });
    });

    po.recv();
}

#[cfg(test)]
fn avoid_copying_the_body(spawnfn: |v: proc()|) {
    let (p, ch) = Chan::<uint>::new();

    let x = ~1;
    let x_in_parent = ptr::to_unsafe_ptr(&*x) as uint;

    spawnfn(proc() {
        let x_in_child = ptr::to_unsafe_ptr(&*x) as uint;
        ch.send(x_in_child);
    });

    let x_in_child = p.recv();
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
        try(proc() {
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
    fn child_no(x: uint) -> proc() {
        return proc() {
            if x < generations {
                let mut t = task();
                t.unwatched();
                t.spawn(child_no(x+1));
            }
        }
    }
    let mut t = task();
    t.unwatched();
    t.spawn(child_no(0));
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
        fail!(~413u16 as ~Any);
    }) {
        Err(e) => {
            type T = ~Any;
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
