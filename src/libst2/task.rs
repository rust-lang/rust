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
//! with their own stack and local state.
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
//! Fatal logic errors in Rust cause *task panic*, during which
//! a task will unwind the stack, running destructors and freeing
//! owned resources. Task panic is unrecoverable from within
//! the panicking task (i.e. there is no 'try/catch' in Rust), but
//! panic may optionally be detected from a different task. If
//! the main task panics the application will exit with a non-zero
//! exit code.
//!
//! ## Example
//!
//! ```rust
//! spawn(proc() {
//!     println!("Hello, World!");
//! })
//! ```

#![unstable = "The task spawning model will be changed as part of runtime reform, and the module \
               will likely be renamed from `task` to `thread`."]

use any::Any;
use comm::channel;
use io::{Writer, stdio};
use kinds::{Send, marker};
use option::{None, Some, Option};
use boxed::Box;
use result::Result;
use rustrt::local::Local;
use rustrt::task;
use rustrt::task::Task;
use str::{Str, SendStr, IntoMaybeOwned};
use string::{String, ToString};
use sync::Future;

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
pub struct TaskBuilder {
    // A name for the task-to-be, for identification in panic messages
    name: Option<SendStr>,
    // The size of the stack for the spawned task
    stack_size: Option<uint>,
    // Task-local stdout
    stdout: Option<Box<Writer + Send>>,
    // Task-local stderr
    stderr: Option<Box<Writer + Send>>,
    // Optionally wrap the eventual task body
    gen_body: Option<proc(v: proc():Send):Send -> proc():Send>,
    nocopy: marker::NoCopy,
}

impl TaskBuilder {
    /// Generate the base configuration for spawning a task, off of which more
    /// configuration methods can be chained.
    pub fn new() -> TaskBuilder { unimplemented!() }
}

impl TaskBuilder {
    /// Name the task-to-be. Currently the name is used for identification
    /// only in panic messages.
    #[unstable = "IntoMaybeOwned will probably change."]
    pub fn named<T: IntoMaybeOwned<'static>>(mut self, name: T) -> TaskBuilder { unimplemented!() }

    /// Set the size of the stack for the new task.
    pub fn stack_size(mut self, size: uint) -> TaskBuilder { unimplemented!() }

    /// Redirect task-local stdout.
    #[experimental = "May not want to make stdio overridable here."]
    pub fn stdout(mut self, stdout: Box<Writer + Send>) -> TaskBuilder { unimplemented!() }

    /// Redirect task-local stderr.
    #[experimental = "May not want to make stdio overridable here."]
    pub fn stderr(mut self, stderr: Box<Writer + Send>) -> TaskBuilder { unimplemented!() }

    // Where spawning actually happens (whether yielding a future or not)
    fn spawn_internal(self, f: proc():Send,
                      on_exit: Option<proc(Result<(), Box<Any + Send>>):Send>) { unimplemented!() }

    /// Creates and executes a new child task.
    ///
    /// Sets up a new task with its own call stack and schedules it to run
    /// the provided proc. The task has the properties and behavior
    /// specified by the `TaskBuilder`.
    pub fn spawn(self, f: proc():Send) { unimplemented!() }

    /// Execute a proc in a newly-spawned task and return a future representing
    /// the task's result. The task has the properties and behavior
    /// specified by the `TaskBuilder`.
    ///
    /// Taking the value of the future will block until the child task
    /// terminates.
    ///
    /// # Return value
    ///
    /// If the child task executes successfully (without panicking) then the
    /// future returns `result::Ok` containing the value returned by the
    /// function. If the child task panics then the future returns `result::Err`
    /// containing the argument to `panic!(...)` as an `Any` trait object.
    #[experimental = "Futures are experimental."]
    pub fn try_future<T:Send>(self, f: proc():Send -> T)
                              -> Future<Result<T, Box<Any + Send>>> { unimplemented!() }

    /// Execute a function in a newly-spawnedtask and block until the task
    /// completes or panics. Equivalent to `.try_future(f).unwrap()`.
    #[unstable = "Error type may change."]
    pub fn try<T:Send>(self, f: proc():Send -> T) -> Result<T, Box<Any + Send>> { unimplemented!() }
}

/* Convenience functions */

/// Creates and executes a new child task
///
/// Sets up a new task with its own call stack and schedules it to run
/// the provided unique closure.
///
/// This function is equivalent to `TaskBuilder::new().spawn(f)`.
pub fn spawn(f: proc(): Send) { unimplemented!() }

/// Execute a function in a newly-spawned task and return either the return
/// value of the function or an error if the task panicked.
///
/// This is equivalent to `TaskBuilder::new().try`.
#[unstable = "Error type may change."]
pub fn try<T: Send>(f: proc(): Send -> T) -> Result<T, Box<Any + Send>> { unimplemented!() }

/// Execute a function in another task and return a future representing the
/// task's result.
///
/// This is equivalent to `TaskBuilder::new().try_future`.
#[experimental = "Futures are experimental."]
pub fn try_future<T:Send>(f: proc():Send -> T) -> Future<Result<T, Box<Any + Send>>> { unimplemented!() }


/* Lifecycle functions */

/// Read the name of the current task.
#[stable]
pub fn name() -> Option<String> { unimplemented!() }

/// Yield control to the task scheduler.
#[unstable = "Name will change."]
pub fn deschedule() { unimplemented!() }

/// True if the running task is currently panicking (e.g. will return `true` inside a
/// destructor that is run while unwinding the stack after a call to `panic!()`).
#[unstable = "May move to a different module."]
pub fn failing() -> bool { unimplemented!() }

#[test]
fn task_abort_no_kill_runtime() { unimplemented!() }
