// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Task creation
//!
//! An executing Rust program consists of a collection of tasks, each
//! with their own stack and local state. A Rust task is typically
//! backed by an operating system thread, making tasks 'just threads',
//! but may also be implemented via other strategies as well
//! (e.g. Rust comes with the [`green`](../../green/index.html)
//! scheduling crate for creating tasks backed by green threads).
//!
//! Tasks generally have their memory *isolated* from each other by
//! virtue of Rust's owned types (which of course may only be owned by
//! a single task at a time). Communication between tasks is primarily
//! done through [channels](../../std/comm/index.html), Rust's
//! message-passing types, though [other forms of task
//! synchronization](../../std/sync/index.html) are often employed to
//! achieve particular performance goals. In particular, types that
//! are guaranteed to be threadsafe are easily shared between threads
//! using the atomically-reference-counted container,
//! [`Arc`](../../std/sync/struct.Arc.html).
//!
//! Fatal logic errors in Rust cause *task failure*, during which
//! a task will unwind the stack, running destructors and freeing
//! owned resources. Task failure is unrecoverable from within
//! the failing task (i.e. there is no 'try/catch' in Rust), but
//! failure may optionally be detected from a different task. If
//! the main task fails the application will exit with a non-zero
//! exit code.
//!
//! # Basic task scheduling
//!
//! By default, every task is created with the same "flavor" as the calling task.
//! This flavor refers to the scheduling mode, with two possibilities currently
//! being 1:1 and M:N modes. Green (M:N) tasks are cooperatively scheduled and
//! native (1:1) tasks are scheduled by the OS kernel.
//!
//! ## Example
//!
//! ```rust
//! spawn(proc() {
//!     println!("Hello, World!");
//! })
//! ```
//!
//! # Advanced task scheduling
//!
//! Task spawning can also be configured to use a particular scheduler, to
//! redirect the new task's output, or to yield a `future` representing the
//! task's final result. The configuration is established using the
//! `TaskBuilder` API:
//!
//! ## Example
//!
//! ```rust
//! extern crate green;
//! extern crate native;
//!
//! use std::task::TaskBuilder;
//! use green::{SchedPool, PoolConfig, GreenTaskBuilder};
//! use native::NativeTaskBuilder;
//!
//! # fn main() {
//! // Create a green scheduler pool with the default configuration
//! let mut pool = SchedPool::new(PoolConfig::new());
//!
//! // Spawn a task in the green pool
//! let mut fut_green = TaskBuilder::new().green(&mut pool).try_future(proc() {
//!     /* ... */
//! });
//!
//! // Spawn a native task
//! let mut fut_native = TaskBuilder::new().native().try_future(proc() {
//!     /* ... */
//! });
//!
//! // Wait for both tasks to finish, recording their outcome
//! let res_green  = fut_green.unwrap();
//! let res_native = fut_native.unwrap();
//!
//! // Shut down the green scheduler pool
//! pool.shutdown();
//! # }
//! ```

#![stable]

use any::Any;
use comm::channel;
use io::{Writer, stdio};
use kinds::{Send, marker};
use option::{None, Some, Option};
use boxed::Box;
use result::Result;
use rt::local::Local;
use rt::task;
use rt::task::Task;
use str::{Str, SendStr, IntoMaybeOwned};
use string::String;
use sync::Future;
use to_string::ToString;

/// A means of spawning a task
pub trait Spawner {
    /// Spawn a task, given low-level task options.
    fn spawn(self, opts: task::TaskOpts, f: proc():Send);
}

/// The default task spawner, which spawns siblings to the current task.
pub struct SiblingSpawner;

impl Spawner for SiblingSpawner {
    fn spawn(self, opts: task::TaskOpts, f: proc():Send) {
        // bind tb to provide type annotation
        let tb: Option<Box<Task>> = Local::try_take();
        match tb {
            Some(t) => t.spawn_sibling(opts, f),
            None => fail!("need a local task to spawn a sibling task"),
        };
    }
}

/// The task builder type.
///
/// Provides detailed control over the properties and behavior of new tasks.

// NB: Builders are designed to be single-use because they do stateful
// things that get weird when reusing - e.g. if you create a result future
// it only applies to a single task, so then you have to maintain Some
// potentially tricky state to ensure that everything behaves correctly
// when you try to reuse the builder to spawn a new task. We'll just
// sidestep that whole issue by making builders uncopyable and making
// the run function move them in.
pub struct TaskBuilder<S = SiblingSpawner> {
    // A name for the task-to-be, for identification in failure messages
    name: Option<SendStr>,
    // The size of the stack for the spawned task
    stack_size: Option<uint>,
    // Task-local stdout
    stdout: Option<Box<Writer + Send>>,
    // Task-local stderr
    stderr: Option<Box<Writer + Send>>,
    // The mechanics of actually spawning the task (i.e.: green or native)
    spawner: S,
    // Optionally wrap the eventual task body
    gen_body: Option<proc(v: proc():Send):Send -> proc():Send>,
    nocopy: marker::NoCopy,
}

impl TaskBuilder<SiblingSpawner> {
    /// Generate the base configuration for spawning a task, off of which more
    /// configuration methods can be chained.
    pub fn new() -> TaskBuilder<SiblingSpawner> {
        TaskBuilder {
            name: None,
            stack_size: None,
            stdout: None,
            stderr: None,
            spawner: SiblingSpawner,
            gen_body: None,
            nocopy: marker::NoCopy,
        }
    }
}

impl<S: Spawner> TaskBuilder<S> {
    /// Name the task-to-be. Currently the name is used for identification
    /// only in failure messages.
    #[unstable = "IntoMaybeOwned will probably change."]
    pub fn named<T: IntoMaybeOwned<'static>>(mut self, name: T) -> TaskBuilder<S> {
        self.name = Some(name.into_maybe_owned());
        self
    }

    /// Set the size of the stack for the new task.
    pub fn stack_size(mut self, size: uint) -> TaskBuilder<S> {
        self.stack_size = Some(size);
        self
    }

    /// Redirect task-local stdout.
    #[experimental = "May not want to make stdio overridable here."]
    pub fn stdout(mut self, stdout: Box<Writer + Send>) -> TaskBuilder<S> {
        self.stdout = Some(stdout);
        self
    }

    /// Redirect task-local stderr.
    #[experimental = "May not want to make stdio overridable here."]
    pub fn stderr(mut self, stderr: Box<Writer + Send>) -> TaskBuilder<S> {
        self.stderr = Some(stderr);
        self
    }

    /// Set the spawning mechanism for the task.
    ///
    /// The `TaskBuilder` API configures a task to be spawned, but defers to the
    /// "spawner" to actually create and spawn the task. The `spawner` method
    /// should not be called directly by `TaskBuiler` clients. It is intended
    /// for use by downstream crates (like `native` and `green`) that implement
    /// tasks. These downstream crates then add extension methods to the
    /// builder, like `.native()` and `.green(pool)`, that actually set the
    /// spawner.
    pub fn spawner<T: Spawner>(self, spawner: T) -> TaskBuilder<T> {
        // repackage the entire TaskBuilder since its type is changing.
        let TaskBuilder {
            name, stack_size, stdout, stderr, spawner: _, gen_body, nocopy
        } = self;
        TaskBuilder {
            name: name,
            stack_size: stack_size,
            stdout: stdout,
            stderr: stderr,
            spawner: spawner,
            gen_body: gen_body,
            nocopy: nocopy,
        }
    }

    /// Add a wrapper to the body of the spawned task.
    ///
    /// Before the task is spawned it is passed through a 'body generator'
    /// function that may perform local setup operations as well as wrap
    /// the task body in remote setup operations. With this the behavior
    /// of tasks can be extended in simple ways.
    ///
    /// This function augments the current body generator with a new body
    /// generator by applying the task body which results from the
    /// existing body generator to the new body generator.
    #[deprecated = "this function will be removed soon"]
    pub fn with_wrapper(mut self, wrapper: proc(v: proc():Send):Send -> proc():Send)
                        -> TaskBuilder<S> {
        self.gen_body = match self.gen_body.take() {
            Some(prev) => Some(proc(body) { wrapper(prev(body)) }),
            None => Some(wrapper)
        };
        self
    }

    // Where spawning actually happens (whether yielding a future or not)
    fn spawn_internal(self, f: proc():Send,
                      on_exit: Option<proc(Result<(), Box<Any + Send>>):Send>) {
        let TaskBuilder {
            name, stack_size, stdout, stderr, spawner, mut gen_body, nocopy: _
        } = self;
        let f = match gen_body.take() {
            Some(gen) => gen(f),
            None => f
        };
        let opts = task::TaskOpts {
            on_exit: on_exit,
            name: name,
            stack_size: stack_size,
        };
        if stdout.is_some() || stderr.is_some() {
            spawner.spawn(opts, proc() {
                let _ = stdout.map(stdio::set_stdout);
                let _ = stderr.map(stdio::set_stderr);
                f();
            })
        } else {
            spawner.spawn(opts, f)
        }
    }

    /// Creates and executes a new child task.
    ///
    /// Sets up a new task with its own call stack and schedules it to run
    /// the provided proc. The task has the properties and behavior
    /// specified by the `TaskBuilder`.
    pub fn spawn(self, f: proc():Send) {
        self.spawn_internal(f, None)
    }

    /// Execute a proc in a newly-spawned task and return a future representing
    /// the task's result. The task has the properties and behavior
    /// specified by the `TaskBuilder`.
    ///
    /// Taking the value of the future will block until the child task
    /// terminates.
    ///
    /// # Return value
    ///
    /// If the child task executes successfully (without failing) then the
    /// future returns `result::Ok` containing the value returned by the
    /// function. If the child task fails then the future returns `result::Err`
    /// containing the argument to `fail!(...)` as an `Any` trait object.
    #[experimental = "Futures are experimental."]
    pub fn try_future<T:Send>(self, f: proc():Send -> T)
                              -> Future<Result<T, Box<Any + Send>>> {
        // currently, the on_exit proc provided by librustrt only works for unit
        // results, so we use an additional side-channel to communicate the
        // result.

        let (tx_done, rx_done) = channel(); // signal that task has exited
        let (tx_retv, rx_retv) = channel(); // return value from task

        let on_exit = proc(res) { let _ = tx_done.send_opt(res); };
        self.spawn_internal(proc() { let _ = tx_retv.send_opt(f()); },
                            Some(on_exit));

        Future::from_fn(proc() {
            rx_done.recv().map(|_| rx_retv.recv())
        })
    }

    /// Execute a function in a newly-spawnedtask and block until the task
    /// completes or fails. Equivalent to `.try_future(f).unwrap()`.
    #[unstable = "Error type may change."]
    pub fn try<T:Send>(self, f: proc():Send -> T) -> Result<T, Box<Any + Send>> {
        self.try_future(f).unwrap()
    }
}

/* Convenience functions */

/// Creates and executes a new child task
///
/// Sets up a new task with its own call stack and schedules it to run
/// the provided unique closure.
///
/// This function is equivalent to `TaskBuilder::new().spawn(f)`.
pub fn spawn(f: proc(): Send) {
    TaskBuilder::new().spawn(f)
}

/// Execute a function in a newly-spawned task and return either the return
/// value of the function or an error if the task failed.
///
/// This is equivalent to `TaskBuilder::new().try`.
#[unstable = "Error type may change."]
pub fn try<T: Send>(f: proc(): Send -> T) -> Result<T, Box<Any + Send>> {
    TaskBuilder::new().try(f)
}

/// Execute a function in another task and return a future representing the
/// task's result.
///
/// This is equivalent to `TaskBuilder::new().try_future`.
#[experimental = "Futures are experimental."]
pub fn try_future<T:Send>(f: proc():Send -> T) -> Future<Result<T, Box<Any + Send>>> {
    TaskBuilder::new().try_future(f)
}


/* Lifecycle functions */

/// Read the name of the current task.
#[deprecated = "Use `task::name()`."]
pub fn with_task_name<U>(blk: |Option<&str>| -> U) -> U {
    use rt::task::Task;

    let task = Local::borrow(None::<Task>);
    match task.name {
        Some(ref name) => blk(Some(name.as_slice())),
        None => blk(None)
    }
}

/// Read the name of the current task.
#[stable]
pub fn name() -> Option<String> {
    use rt::task::Task;

    let task = Local::borrow(None::<Task>);
    match task.name {
        Some(ref name) => Some(name.as_slice().to_string()),
        None => None
    }
}

/// Yield control to the task scheduler.
#[unstable = "Name will change."]
pub fn deschedule() {
    use rt::local::Local;

    // FIXME(#7544): Optimize this, since we know we won't block.
    let task: Box<Task> = Local::take();
    task.yield_now();
}

/// True if the running task is currently failing (e.g. will return `true` inside a
/// destructor that is run while unwinding the stack after a call to `fail!()`).
#[unstable = "May move to a different module."]
pub fn failing() -> bool {
    use rt::task::Task;
    Local::borrow(None::<Task>).unwinder.unwinding()
}

#[cfg(test)]
mod test {
    use any::{Any, AnyRefExt};
    use boxed::BoxAny;
    use result;
    use result::{Ok, Err};
    use string::String;
    use std::io::{ChanReader, ChanWriter};
    use prelude::*;
    use super::*;

    // !!! These tests are dangerous. If something is buggy, they will hang, !!!
    // !!! instead of exiting cleanly. This might wedge the buildbots.       !!!

    #[test]
    fn test_unnamed_task() {
        try(proc() {
            assert!(name().is_none());
        }).map_err(|_| ()).unwrap();
    }

    #[test]
    fn test_owned_named_task() {
        TaskBuilder::new().named("ada lovelace".to_string()).try(proc() {
            assert!(name().unwrap() == "ada lovelace".to_string());
        }).map_err(|_| ()).unwrap();
    }

    #[test]
    fn test_static_named_task() {
        TaskBuilder::new().named("ada lovelace").try(proc() {
            assert!(name().unwrap() == "ada lovelace".to_string());
        }).map_err(|_| ()).unwrap();
    }

    #[test]
    fn test_send_named_task() {
        TaskBuilder::new().named("ada lovelace".into_maybe_owned()).try(proc() {
            assert!(name().unwrap() == "ada lovelace".to_string());
        }).map_err(|_| ()).unwrap();
    }

    #[test]
    fn test_run_basic() {
        let (tx, rx) = channel();
        TaskBuilder::new().spawn(proc() {
            tx.send(());
        });
        rx.recv();
    }

    #[test]
    fn test_with_wrapper() {
        let (tx, rx) = channel();
        TaskBuilder::new().with_wrapper(proc(body) {
            let result: proc():Send = proc() {
                body();
                tx.send(());
            };
            result
        }).spawn(proc() { });
        rx.recv();
    }

    #[test]
    fn test_try_future() {
        let result = TaskBuilder::new().try_future(proc() {});
        assert!(result.unwrap().is_ok());

        let result = TaskBuilder::new().try_future(proc() -> () {
            fail!();
        });
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn test_try_success() {
        match try(proc() {
            "Success!".to_string()
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

    fn avoid_copying_the_body(spawnfn: |v: proc():Send|) {
        let (tx, rx) = channel::<uint>();

        let x = box 1;
        let x_in_parent = (&*x) as *const int as uint;

        spawnfn(proc() {
            let x_in_child = (&*x) as *const int as uint;
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
            let builder = TaskBuilder::new();
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
        static GENERATIONS: uint = 16;
        fn child_no(x: uint) -> proc(): Send {
            return proc() {
                if x < GENERATIONS {
                    TaskBuilder::new().spawn(child_no(x+1));
                }
            }
        }
        TaskBuilder::new().spawn(child_no(0));
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
                assert_eq!(*e.downcast::<T>().unwrap(), "static string");
            }
            Ok(()) => fail!()
        }
    }

    #[test]
    fn test_try_fail_message_owned_str() {
        match try(proc() {
            fail!("owned string".to_string());
        }) {
            Err(e) => {
                type T = String;
                assert!(e.is::<T>());
                assert_eq!(*e.downcast::<T>().unwrap(), "owned string".to_string());
            }
            Ok(()) => fail!()
        }
    }

    #[test]
    fn test_try_fail_message_any() {
        match try(proc() {
            fail!(box 413u16 as Box<Any + Send>);
        }) {
            Err(e) => {
                type T = Box<Any + Send>;
                assert!(e.is::<T>());
                let any = e.downcast::<T>().unwrap();
                assert!(any.is::<u16>());
                assert_eq!(*any.downcast::<u16>().unwrap(), 413u16);
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

    #[test]
    fn test_stdout() {
        let (tx, rx) = channel();
        let mut reader = ChanReader::new(rx);
        let stdout = ChanWriter::new(tx);

        let r = TaskBuilder::new().stdout(box stdout as Box<Writer + Send>)
                                  .try(proc() {
                print!("Hello, world!");
            });
        assert!(r.is_ok());

        let output = reader.read_to_string().unwrap();
        assert_eq!(output, "Hello, world!".to_string());
    }

    // NOTE: the corresponding test for stderr is in run-pass/task-stderr, due
    // to the test harness apparently interfering with stderr configuration.
}

#[test]
fn task_abort_no_kill_runtime() {
    use std::io::timer;
    use time::Duration;
    use mem;

    let tb = TaskBuilder::new();
    let rx = tb.try_future(proc() {});
    mem::drop(rx);
    timer::sleep(Duration::milliseconds(1000));
}
