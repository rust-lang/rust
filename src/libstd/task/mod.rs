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
 * Task management.
 *
 * An executing Rust program consists of a tree of tasks, each with their own
 * stack, and sole ownership of their allocated heap data. Tasks communicate
 * with each other using ports and channels.
 *
 * When a task fails, that failure will propagate to its parent (the task
 * that spawned it) and the parent will fail as well. The reverse is not
 * true: when a parent task fails its children will continue executing. When
 * the root (main) task fails, all tasks fail, and then so does the entire
 * process.
 *
 * Tasks may execute in parallel and are scheduled automatically by the
 * runtime.
 *
 * # Example
 *
 * ~~~
 * do spawn {
 *     log(error, "Hello, World!");
 * }
 * ~~~
 */

#[allow(missing_doc)];

use prelude::*;

use cell::Cell;
use cmp::Eq;
use comm::{stream, Chan, GenericChan, GenericPort, Port};
use result::Result;
use result;
use rt::{context, OldTaskContext};
use task::rt::{task_id, sched_id};
use unstable::finally::Finally;
use util::replace;
use util;

#[cfg(test)] use cast;
#[cfg(test)] use comm::SharedChan;
#[cfg(test)] use comm;
#[cfg(test)] use ptr;
#[cfg(test)] use task;

mod local_data_priv;
pub mod rt;
pub mod spawn;

/// A handle to a scheduler
#[deriving(Eq)]
pub enum Scheduler {
    SchedulerHandle(sched_id)
}

/// A handle to a task
#[deriving(Eq)]
pub enum Task {
    TaskHandle(task_id)
}

/**
 * Indicates the manner in which a task exited.
 *
 * A task that completes without failing is considered to exit successfully.
 * Supervised ancestors and linked siblings may yet fail after this task
 * succeeds. Also note that in such a case, it may be nondeterministic whether
 * linked failure or successful exit happen first.
 *
 * If you wish for this result's delivery to block until all linked and/or
 * children tasks complete, recommend using a result future.
 */
#[deriving(Eq)]
pub enum TaskResult {
    Success,
    Failure,
}

/// Scheduler modes
#[deriving(Eq)]
pub enum SchedMode {
    /// Run task on the default scheduler
    DefaultScheduler,
    /// Run task on the current scheduler
    CurrentScheduler,
    /// Run task on a specific scheduler
    ExistingScheduler(Scheduler),
    /**
     * Tasks are scheduled on the main OS thread
     *
     * The main OS thread is the thread used to launch the runtime which,
     * in most cases, is the process's initial thread as created by the OS.
     */
    PlatformThread,
    /// All tasks run in the same OS thread
    SingleThreaded,
    /// Tasks are distributed among available CPUs
    ThreadPerCore,
    /// Each task runs in its own OS thread
    ThreadPerTask,
    /// Tasks are distributed among a fixed number of OS threads
    ManualThreads(uint),
}

/**
 * Scheduler configuration options
 *
 * # Fields
 *
 * * sched_mode - The operating mode of the scheduler
 *
 * * foreign_stack_size - The size of the foreign stack, in bytes
 *
 *     Rust code runs on Rust-specific stacks. When Rust code calls foreign
 *     code (via functions in foreign modules) it switches to a typical, large
 *     stack appropriate for running code written in languages like C. By
 *     default these foreign stacks have unspecified size, but with this
 *     option their size can be precisely specified.
 */
pub struct SchedOpts {
    mode: SchedMode,
    foreign_stack_size: Option<uint>,
}

/**
 * Task configuration options
 *
 * # Fields
 *
 * * linked - Propagate failure bidirectionally between child and parent.
 *            True by default. If both this and 'supervised' are false, then
 *            either task's failure will not affect the other ("unlinked").
 *
 * * supervised - Propagate failure unidirectionally from parent to child,
 *                but not from child to parent. False by default.
 *
 * * notify_chan - Enable lifecycle notifications on the given channel
 *
 * * sched - Specify the configuration of a new scheduler to create the task
 *           in
 *
 *     By default, every task is created in the same scheduler as its
 *     parent, where it is scheduled cooperatively with all other tasks
 *     in that scheduler. Some specialized applications may want more
 *     control over their scheduling, in which case they can be spawned
 *     into a new scheduler with the specific properties required.
 *
 *     This is of particular importance for libraries which want to call
 *     into foreign code that blocks. Without doing so in a different
 *     scheduler other tasks will be impeded or even blocked indefinitely.
 */
pub struct TaskOpts {
    linked: bool,
    supervised: bool,
    notify_chan: Option<Chan<TaskResult>>,
    sched: SchedOpts
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

// FIXME (#3724): Replace the 'consumed' bit with move mode on self
pub struct TaskBuilder {
    opts: TaskOpts,
    gen_body: Option<~fn(v: ~fn()) -> ~fn()>,
    can_not_copy: Option<util::NonCopyable>,
    consumed: bool,
}

/**
 * Generate the base configuration for spawning a task, off of which more
 * configuration methods can be chained.
 * For example, task().unlinked().spawn is equivalent to spawn_unlinked.
 */
pub fn task() -> TaskBuilder {
    TaskBuilder {
        opts: default_task_opts(),
        gen_body: None,
        can_not_copy: None,
        consumed: false,
    }
}

impl TaskBuilder {
    fn consume(&mut self) -> TaskBuilder {
        if self.consumed {
            fail!("Cannot copy a task_builder"); // Fake move mode on self
        }
        self.consumed = true;
        let gen_body = replace(&mut self.gen_body, None);
        let notify_chan = replace(&mut self.opts.notify_chan, None);
        TaskBuilder {
            opts: TaskOpts {
                linked: self.opts.linked,
                supervised: self.opts.supervised,
                notify_chan: notify_chan,
                sched: self.opts.sched
            },
            gen_body: gen_body,
            can_not_copy: None,
            consumed: false
        }
    }
}

impl TaskBuilder {
    /// Decouple the child task's failure from the parent's. If either fails,
    /// the other will not be killed.
    pub fn unlinked(&mut self) {
        self.opts.linked = false;
    }

    /// Unidirectionally link the child task's failure with the parent's. The
    /// child's failure will not kill the parent, but the parent's will kill
    /// the child.
    pub fn supervised(&mut self) {
        self.opts.supervised = true;
        self.opts.linked = false;
    }

    /// Link the child task's and parent task's failures. If either fails, the
    /// other will be killed.
    pub fn linked(&mut self) {
        self.opts.linked = true;
        self.opts.supervised = false;
    }

    /**
     * Get a future representing the exit status of the task.
     *
     * Taking the value of the future will block until the child task
     * terminates. The future-receiving callback specified will be called
     * *before* the task is spawned; as such, do not invoke .get() within the
     * closure; rather, store it in an outer variable/list for later use.
     *
     * Note that the future returning by this function is only useful for
     * obtaining the value of the next task to be spawning with the
     * builder. If additional tasks are spawned with the same builder
     * then a new result future must be obtained prior to spawning each
     * task.
     *
     * # Failure
     * Fails if a future_result was already set for this task.
     */
    pub fn future_result(&mut self, blk: &fn(v: Port<TaskResult>)) {
        // FIXME (#3725): Once linked failure and notification are
        // handled in the library, I can imagine implementing this by just
        // registering an arbitrary number of task::on_exit handlers and
        // sending out messages.

        if self.opts.notify_chan.is_some() {
            fail!("Can't set multiple future_results for one task!");
        }

        // Construct the future and give it to the caller.
        let (notify_pipe_po, notify_pipe_ch) = stream::<TaskResult>();

        blk(notify_pipe_po);

        // Reconfigure self to use a notify channel.
        self.opts.notify_chan = Some(notify_pipe_ch);
    }

    /// Configure a custom scheduler mode for the task.
    pub fn sched_mode(&mut self, mode: SchedMode) {
        self.opts.sched.mode = mode;
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
    pub fn add_wrapper(&mut self, wrapper: ~fn(v: ~fn()) -> ~fn()) {
        let prev_gen_body = replace(&mut self.gen_body, None);
        let prev_gen_body = match prev_gen_body {
            Some(gen) => gen,
            None => {
                let f: ~fn(~fn()) -> ~fn() = |body| body;
                f
            }
        };
        let prev_gen_body = Cell::new(prev_gen_body);
        let next_gen_body = {
            let f: ~fn(~fn()) -> ~fn() = |body| {
                let prev_gen_body = prev_gen_body.take();
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
    pub fn spawn(&mut self, f: ~fn()) {
        let gen_body = replace(&mut self.gen_body, None);
        let notify_chan = replace(&mut self.opts.notify_chan, None);
        let x = self.consume();
        let opts = TaskOpts {
            linked: x.opts.linked,
            supervised: x.opts.supervised,
            notify_chan: notify_chan,
            sched: x.opts.sched
        };
        let f = match gen_body {
            Some(gen) => {
                gen(f)
            }
            None => {
                f
            }
        };
        spawn::spawn_raw(opts, f);
    }

    /// Runs a task, while transfering ownership of one argument to the child.
    pub fn spawn_with<A:Owned>(&mut self, arg: A, f: ~fn(v: A)) {
        let arg = Cell::new(arg);
        do self.spawn {
            f(arg.take());
        }
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
    pub fn try<T:Owned>(&mut self, f: ~fn() -> T) -> Result<T,()> {
        let (po, ch) = stream::<T>();
        let mut result = None;

        self.future_result(|r| { result = Some(r); });

        do self.spawn {
            ch.send(f());
        }

        match result.unwrap().recv() {
            Success => result::Ok(po.recv()),
            Failure => result::Err(())
        }
    }
}


/* Task construction */

pub fn default_task_opts() -> TaskOpts {
    /*!
     * The default task options
     *
     * By default all tasks are supervised by their parent, are spawned
     * into the same scheduler, and do not post lifecycle notifications.
     */

    TaskOpts {
        linked: true,
        supervised: false,
        notify_chan: None,
        sched: SchedOpts {
            mode: DefaultScheduler,
            foreign_stack_size: None
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
pub fn spawn(f: ~fn()) {
    let mut task = task();
    task.spawn(f)
}

/// Creates a child task unlinked from the current one. If either this
/// task or the child task fails, the other will not be killed.
pub fn spawn_unlinked(f: ~fn()) {
    let mut task = task();
    task.unlinked();
    task.spawn(f)
}

pub fn spawn_supervised(f: ~fn()) {
    /*!
     * Creates a child task supervised by the current one. If the child
     * task fails, the parent will not be killed, but if the parent fails,
     * the child will be killed.
     */

    let mut task = task();
    task.supervised();
    task.spawn(f)
}

pub fn spawn_with<A:Owned>(arg: A, f: ~fn(v: A)) {
    /*!
     * Runs a task, while transfering ownership of one argument to the
     * child.
     *
     * This is useful for transfering ownership of noncopyables to
     * another task.
     *
     * This function is equivalent to `task().spawn_with(arg, f)`.
     */

    let mut task = task();
    task.spawn_with(arg, f)
}

pub fn spawn_sched(mode: SchedMode, f: ~fn()) {
    /*!
     * Creates a new task on a new or existing scheduler

     * When there are no more tasks to execute the
     * scheduler terminates.
     *
     * # Failure
     *
     * In manual threads mode the number of threads requested must be
     * greater than zero.
     */

    let mut task = task();
    task.sched_mode(mode);
    task.spawn(f)
}

pub fn try<T:Owned>(f: ~fn() -> T) -> Result<T,()> {
    /*!
     * Execute a function in another task and return either the return value
     * of the function or result::err.
     *
     * This is equivalent to task().supervised().try.
     */

    let mut task = task();
    task.supervised();
    task.try(f)
}


/* Lifecycle functions */

pub fn yield() {
    //! Yield control to the task scheduler

    unsafe {
        let task_ = rt::rust_get_task();
        let killed = rt::rust_task_yield(task_);
        if killed && !failing() {
            fail!("killed");
        }
    }
}

pub fn failing() -> bool {
    //! True if the running task has failed

    use rt::{context, OldTaskContext};
    use rt::local::Local;
    use rt::task::Task;

    match context() {
        OldTaskContext => {
            unsafe {
                rt::rust_task_is_unwinding(rt::rust_get_task())
            }
        }
        _ => {
            let mut unwinding = false;
            do Local::borrow::<Task> |local| {
                unwinding = match local.unwinder {
                    Some(unwinder) => {
                        unwinder.unwinding
                    }
                    None => {
                        // Because there is no unwinder we can't be unwinding.
                        // (The process will abort on failure)
                        false
                    }
                }
            }
            return unwinding;
        }
    }
}

pub fn get_task() -> Task {
    //! Get a handle to the running task

    unsafe {
        TaskHandle(rt::get_task_id())
    }
}

pub fn get_scheduler() -> Scheduler {
    SchedulerHandle(unsafe { rt::rust_get_sched_id() })
}

/**
 * Temporarily make the task unkillable
 *
 * # Example
 *
 * ~~~
 * do task::unkillable {
 *     // detach / yield / destroy must all be called together
 *     rustrt::rust_port_detach(po);
 *     // This must not result in the current task being killed
 *     task::yield();
 *     rustrt::rust_port_destroy(po);
 * }
 * ~~~
 */
pub unsafe fn unkillable<U>(f: &fn() -> U) -> U {
    if context() == OldTaskContext {
        let t = rt::rust_get_task();
        do (|| {
            rt::rust_task_inhibit_kill(t);
            f()
        }).finally {
            rt::rust_task_allow_kill(t);
        }
    } else {
        // FIXME #6377
        f()
    }
}

/// The inverse of unkillable. Only ever to be used nested in unkillable().
pub unsafe fn rekillable<U>(f: &fn() -> U) -> U {
    if context() == OldTaskContext {
        let t = rt::rust_get_task();
        do (|| {
            rt::rust_task_allow_kill(t);
            f()
        }).finally {
            rt::rust_task_inhibit_kill(t);
        }
    } else {
        // FIXME #6377
        f()
    }
}

/**
 * A stronger version of unkillable that also inhibits scheduling operations.
 * For use with exclusive ARCs, which use pthread mutexes directly.
 */
pub unsafe fn atomically<U>(f: &fn() -> U) -> U {
    if context() == OldTaskContext {
        let t = rt::rust_get_task();
        do (|| {
            rt::rust_task_inhibit_kill(t);
            rt::rust_task_inhibit_yield(t);
            f()
        }).finally {
            rt::rust_task_allow_yield(t);
            rt::rust_task_allow_kill(t);
        }
    } else {
        // FIXME #6377
        f()
    }
}

#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_cant_dup_task_builder() {
    let mut builder = task();
    builder.unlinked();
    do builder.spawn {}
    // FIXME(#3724): For now, this is a -runtime- failure, because we haven't
    // got move mode on self. When 3724 is fixed, this test should fail to
    // compile instead, and should go in tests/compile-fail.
    do builder.spawn {} // b should have been consumed by the previous call
}

// The following 8 tests test the following 2^3 combinations:
// {un,}linked {un,}supervised failure propagation {up,down}wards.

// !!! These tests are dangerous. If Something is buggy, they will hang, !!!
// !!! instead of exiting cleanly. This might wedge the buildbots.       !!!

#[test] #[ignore(cfg(windows))]
fn test_spawn_unlinked_unsup_no_fail_down() { // grandchild sends on a port
    let (po, ch) = stream();
    let ch = SharedChan::new(ch);
    do spawn_unlinked {
        let ch = ch.clone();
        do spawn_unlinked {
            // Give middle task a chance to fail-but-not-kill-us.
            for 16.times { task::yield(); }
            ch.send(()); // If killed first, grandparent hangs.
        }
        fail!(); // Shouldn't kill either (grand)parent or (grand)child.
    }
    po.recv();
}
#[test] #[ignore(cfg(windows))]
fn test_spawn_unlinked_unsup_no_fail_up() { // child unlinked fails
    do spawn_unlinked { fail!(); }
}
#[test] #[ignore(cfg(windows))]
fn test_spawn_unlinked_sup_no_fail_up() { // child unlinked fails
    do spawn_supervised { fail!(); }
    // Give child a chance to fail-but-not-kill-us.
    for 16.times { task::yield(); }
}
#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_unlinked_sup_fail_down() {
    do spawn_supervised { loop { task::yield(); } }
    fail!(); // Shouldn't leave a child hanging around.
}

#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_linked_sup_fail_up() { // child fails; parent fails
    let (po, _ch) = stream::<()>();

    // Unidirectional "parenting" shouldn't override bidirectional linked.
    // We have to cheat with opts - the interface doesn't support them because
    // they don't make sense (redundant with task().supervised()).
    let mut b0 = task();
    b0.opts.linked = true;
    b0.opts.supervised = true;

    do b0.spawn {
        fail!();
    }
    po.recv(); // We should get punted awake
}
#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_linked_sup_fail_down() { // parent fails; child fails
    // We have to cheat with opts - the interface doesn't support them because
    // they don't make sense (redundant with task().supervised()).
    let mut b0 = task();
    b0.opts.linked = true;
    b0.opts.supervised = true;
    do b0.spawn {
        loop {
            task::yield();
        }
    }
    fail!(); // *both* mechanisms would be wrong if this didn't kill the child
}
#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_linked_unsup_fail_up() { // child fails; parent fails
    let (po, _ch) = stream::<()>();
    // Default options are to spawn linked & unsupervised.
    do spawn { fail!(); }
    po.recv(); // We should get punted awake
}
#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_linked_unsup_fail_down() { // parent fails; child fails
    // Default options are to spawn linked & unsupervised.
    do spawn { loop { task::yield(); } }
    fail!();
}
#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_linked_unsup_default_opts() { // parent fails; child fails
    // Make sure the above test is the same as this one.
    let mut builder = task();
    builder.linked();
    do builder.spawn {
        loop {
            task::yield();
        }
    }
    fail!();
}

// A couple bonus linked failure tests - testing for failure propagation even
// when the middle task exits successfully early before kill signals are sent.

#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_failure_propagate_grandchild() {
    // Middle task exits; does grandparent's failure propagate across the gap?
    do spawn_supervised {
        do spawn_supervised {
            loop { task::yield(); }
        }
    }
    for 16.times { task::yield(); }
    fail!();
}

#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_failure_propagate_secondborn() {
    // First-born child exits; does parent's failure propagate to sibling?
    do spawn_supervised {
        do spawn { // linked
            loop { task::yield(); }
        }
    }
    for 16.times { task::yield(); }
    fail!();
}

#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_failure_propagate_nephew_or_niece() {
    // Our sibling exits; does our failure propagate to sibling's child?
    do spawn { // linked
        do spawn_supervised {
            loop { task::yield(); }
        }
    }
    for 16.times { task::yield(); }
    fail!();
}

#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_linked_sup_propagate_sibling() {
    // Middle sibling exits - does eldest's failure propagate to youngest?
    do spawn { // linked
        do spawn { // linked
            loop { task::yield(); }
        }
    }
    for 16.times { task::yield(); }
    fail!();
}

#[test]
fn test_run_basic() {
    let (po, ch) = stream::<()>();
    let mut builder = task();
    do builder.spawn {
        ch.send(());
    }
    po.recv();
}

#[cfg(test)]
struct Wrapper {
    f: Option<Chan<()>>
}

#[test]
fn test_add_wrapper() {
    let (po, ch) = stream::<()>();
    let mut b0 = task();
    let ch = Cell::new(ch);
    do b0.add_wrapper |body| {
        let ch = Cell::new(ch.take());
        let result: ~fn() = || {
            let ch = ch.take();
            body();
            ch.send(());
        };
        result
    };
    do b0.spawn { }
    po.recv();
}

#[test]
#[ignore(cfg(windows))]
fn test_future_result() {
    let mut result = None;
    let mut builder = task();
    builder.future_result(|r| result = Some(r));
    do builder.spawn {}
    assert_eq!(result.unwrap().recv(), Success);

    result = None;
    let mut builder = task();
    builder.future_result(|r| result = Some(r));
    builder.unlinked();
    do builder.spawn {
        fail!();
    }
    assert_eq!(result.unwrap().recv(), Failure);
}

#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_back_to_the_future_result() {
    let mut builder = task();
    builder.future_result(util::ignore);
    builder.future_result(util::ignore);
}

#[test]
fn test_try_success() {
    match do try {
        ~"Success!"
    } {
        result::Ok(~"Success!") => (),
        _ => fail!()
    }
}

#[test]
#[ignore(cfg(windows))]
fn test_try_fail() {
    match do try {
        fail!()
    } {
        result::Err(()) => (),
        result::Ok(()) => fail!()
    }
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn test_spawn_sched_no_threads() {
    do spawn_sched(ManualThreads(0u)) { }
}

#[test]
fn test_spawn_sched() {
    let (po, ch) = stream::<()>();
    let ch = SharedChan::new(ch);

    fn f(i: int, ch: SharedChan<()>) {
        let parent_sched_id = unsafe { rt::rust_get_sched_id() };

        do spawn_sched(SingleThreaded) {
            let child_sched_id = unsafe { rt::rust_get_sched_id() };
            assert!(parent_sched_id != child_sched_id);

            if (i == 0) {
                ch.send(());
            } else {
                f(i - 1, ch.clone());
            }
        };

    }
    f(10, ch);
    po.recv();
}

#[test]
fn test_spawn_sched_childs_on_default_sched() {
    let (po, ch) = stream();

    // Assuming tests run on the default scheduler
    let default_id = unsafe { rt::rust_get_sched_id() };

    let ch = Cell::new(ch);
    do spawn_sched(SingleThreaded) {
        let parent_sched_id = unsafe { rt::rust_get_sched_id() };
        let ch = Cell::new(ch.take());
        do spawn {
            let ch = ch.take();
            let child_sched_id = unsafe { rt::rust_get_sched_id() };
            assert!(parent_sched_id != child_sched_id);
            assert_eq!(child_sched_id, default_id);
            ch.send(());
        };
    };

    po.recv();
}

#[cfg(test)]
mod testrt {
    use libc;

    #[nolink]
    pub extern {
        unsafe fn rust_dbg_lock_create() -> *libc::c_void;
        unsafe fn rust_dbg_lock_destroy(lock: *libc::c_void);
        unsafe fn rust_dbg_lock_lock(lock: *libc::c_void);
        unsafe fn rust_dbg_lock_unlock(lock: *libc::c_void);
        unsafe fn rust_dbg_lock_wait(lock: *libc::c_void);
        unsafe fn rust_dbg_lock_signal(lock: *libc::c_void);
    }
}

#[test]
fn test_spawn_sched_blocking() {
    unsafe {

        // Testing that a task in one scheduler can block in foreign code
        // without affecting other schedulers
        for 20u.times {
            let (start_po, start_ch) = stream();
            let (fin_po, fin_ch) = stream();

            let lock = testrt::rust_dbg_lock_create();

            do spawn_sched(SingleThreaded) {
                testrt::rust_dbg_lock_lock(lock);

                start_ch.send(());

                // Block the scheduler thread
                testrt::rust_dbg_lock_wait(lock);
                testrt::rust_dbg_lock_unlock(lock);

                fin_ch.send(());
            };

            // Wait until the other task has its lock
            start_po.recv();

            fn pingpong(po: &Port<int>, ch: &Chan<int>) {
                let mut val = 20;
                while val > 0 {
                    val = po.recv();
                    ch.send(val - 1);
                }
            }

            let (setup_po, setup_ch) = stream();
            let (parent_po, parent_ch) = stream();
            do spawn {
                let (child_po, child_ch) = stream();
                setup_ch.send(child_ch);
                pingpong(&child_po, &parent_ch);
            };

            let child_ch = setup_po.recv();
            child_ch.send(20);
            pingpong(&parent_po, &child_ch);
            testrt::rust_dbg_lock_lock(lock);
            testrt::rust_dbg_lock_signal(lock);
            testrt::rust_dbg_lock_unlock(lock);
            fin_po.recv();
            testrt::rust_dbg_lock_destroy(lock);
        }
    }
}

#[cfg(test)]
fn avoid_copying_the_body(spawnfn: &fn(v: ~fn())) {
    let (p, ch) = stream::<uint>();

    let x = ~1;
    let x_in_parent = ptr::to_unsafe_ptr(&*x) as uint;

    do spawnfn || {
        let x_in_child = ptr::to_unsafe_ptr(&*x) as uint;
        ch.send(x_in_child);
    }

    let x_in_child = p.recv();
    assert_eq!(x_in_parent, x_in_child);
}

#[test]
fn test_avoid_copying_the_body_spawn() {
    avoid_copying_the_body(spawn);
}

#[test]
fn test_avoid_copying_the_body_task_spawn() {
    do avoid_copying_the_body |f| {
        let mut builder = task();
        do builder.spawn || {
            f();
        }
    }
}

#[test]
fn test_avoid_copying_the_body_try() {
    do avoid_copying_the_body |f| {
        do try || {
            f()
        };
    }
}

#[test]
fn test_avoid_copying_the_body_unlinked() {
    do avoid_copying_the_body |f| {
        do spawn_unlinked || {
            f();
        }
    }
}

#[test]
fn test_platform_thread() {
    let (po, ch) = stream();
    let mut builder = task();
    builder.sched_mode(PlatformThread);
    do builder.spawn {
        ch.send(());
    }
    po.recv();
}

#[test]
#[ignore(cfg(windows))]
#[should_fail]
fn test_unkillable() {
    let (po, ch) = stream();

    // We want to do this after failing
    do spawn_unlinked {
        for 10.times { yield() }
        ch.send(());
    }

    do spawn {
        yield();
        // We want to fail after the unkillable task
        // blocks on recv
        fail!();
    }

    unsafe {
        do unkillable {
            let p = ~0;
            let pp: *uint = cast::transmute(p);

            // If we are killed here then the box will leak
            po.recv();

            let _p: ~int = cast::transmute(pp);
        }
    }

    // Now we can be killed
    po.recv();
}

#[test]
#[ignore(cfg(windows))]
#[should_fail]
fn test_unkillable_nested() {
    let (po, ch) = comm::stream();

    // We want to do this after failing
    do spawn_unlinked || {
        for 10.times { yield() }
        ch.send(());
    }

    do spawn {
        yield();
        // We want to fail after the unkillable task
        // blocks on recv
        fail!();
    }

    unsafe {
        do unkillable {
            do unkillable {} // Here's the difference from the previous test.
            let p = ~0;
            let pp: *uint = cast::transmute(p);

            // If we are killed here then the box will leak
            po.recv();

            let _p: ~int = cast::transmute(pp);
        }
    }

    // Now we can be killed
    po.recv();
}

#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_atomically() {
    unsafe { do atomically { yield(); } }
}

#[test]
fn test_atomically2() {
    unsafe { do atomically { } } yield(); // shouldn't fail
}

#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_atomically_nested() {
    unsafe { do atomically { do atomically { } yield(); } }
}

#[test]
fn test_child_doesnt_ref_parent() {
    // If the child refcounts the parent task, this will stack overflow when
    // climbing the task tree to dereference each ancestor. (See #1789)
    // (well, it would if the constant were 8000+ - I lowered it to be more
    // valgrind-friendly. try this at home, instead..!)
    static generations: uint = 16;
    fn child_no(x: uint) -> ~fn() {
        return || {
            if x < generations {
                task::spawn(child_no(x+1));
            }
        }
    }
    task::spawn(child_no(0));
}

#[test]
fn test_sched_thread_per_core() {
    let (port, chan) = comm::stream();

    do spawn_sched(ThreadPerCore) || {
        unsafe {
            let cores = rt::rust_num_threads();
            let reported_threads = rt::rust_sched_threads();
            assert_eq!(cores as uint, reported_threads as uint);
            chan.send(());
        }
    }

    port.recv();
}

#[test]
fn test_spawn_thread_on_demand() {
    let (port, chan) = comm::stream();

    do spawn_sched(ManualThreads(2)) || {
        unsafe {
            let max_threads = rt::rust_sched_threads();
            assert_eq!(max_threads as int, 2);
            let running_threads = rt::rust_sched_current_nonlazy_threads();
            assert_eq!(running_threads as int, 1);

            let (port2, chan2) = comm::stream();

            do spawn_sched(CurrentScheduler) || {
                chan2.send(());
            }

            let running_threads2 = rt::rust_sched_current_nonlazy_threads();
            assert_eq!(running_threads2 as int, 2);

            port2.recv();
            chan.send(());
        }
    }

    port.recv();
}

#[test]
fn test_simple_newsched_spawn() {
    use rt::test::run_in_newsched_task;

    do run_in_newsched_task {
        spawn(||())
    }
}
