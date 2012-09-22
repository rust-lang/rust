// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

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

use cmp::Eq;
use result::Result;
use pipes::{stream, Chan, Port};
use local_data_priv::{local_get, local_set};

export Task;
export TaskResult;
export Notification;
export SchedMode;
export SchedOpts;
export TaskOpts;
export TaskBuilder;

export task;
export default_task_opts;
export get_opts;
export set_opts;
export set_sched_mode;
export add_wrapper;
export run;

export future_result;
export run_listener;
export run_with;

export spawn;
export spawn_unlinked;
export spawn_supervised;
export spawn_with;
export spawn_listener;
export spawn_conversation;
export spawn_sched;
export try;

export yield;
export failing;
export get_task;
export unkillable, rekillable;
export atomically;

export local_data;

export SingleThreaded;
export ThreadPerCore;
export ThreadPerTask;
export ManualThreads;
export PlatformThread;

use rt::task_id;
use rt::rust_task;

/// A handle to a task
enum Task {
    TaskHandle(task_id)
}

#[cfg(stage0)]
impl Task : cmp::Eq {
    pure fn eq(&&other: Task) -> bool { *self == *other }
    pure fn ne(&&other: Task) -> bool { !self.eq(other) }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl Task : cmp::Eq {
    pure fn eq(other: &Task) -> bool { *self == *(*other) }
    pure fn ne(other: &Task) -> bool { !self.eq(other) }
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
enum TaskResult {
    Success,
    Failure,
}

#[cfg(stage0)]
impl TaskResult: Eq {
    pure fn eq(&&other: TaskResult) -> bool {
        match (self, other) {
            (Success, Success) | (Failure, Failure) => true,
            (Success, _) | (Failure, _) => false
        }
    }
    pure fn ne(&&other: TaskResult) -> bool { !self.eq(other) }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl TaskResult : Eq {
    pure fn eq(other: &TaskResult) -> bool {
        match (self, (*other)) {
            (Success, Success) | (Failure, Failure) => true,
            (Success, _) | (Failure, _) => false
        }
    }
    pure fn ne(other: &TaskResult) -> bool { !self.eq(other) }
}

/// A message type for notifying of task lifecycle events
enum Notification {
    /// Sent when a task exits with the task handle and result
    Exit(Task, TaskResult)
}

#[cfg(stage0)]
impl Notification : cmp::Eq {
    pure fn eq(&&other: Notification) -> bool {
        match self {
            Exit(e0a, e1a) => {
                match other {
                    Exit(e0b, e1b) => e0a == e0b && e1a == e1b
                }
            }
        }
    }
    pure fn ne(&&other: Notification) -> bool { !self.eq(other) }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl Notification : cmp::Eq {
    pure fn eq(other: &Notification) -> bool {
        match self {
            Exit(e0a, e1a) => {
                match (*other) {
                    Exit(e0b, e1b) => e0a == e0b && e1a == e1b
                }
            }
        }
    }
    pure fn ne(other: &Notification) -> bool { !self.eq(other) }
}

/// Scheduler modes
enum SchedMode {
    /// All tasks run in the same OS thread
    SingleThreaded,
    /// Tasks are distributed among available CPUs
    ThreadPerCore,
    /// Each task runs in its own OS thread
    ThreadPerTask,
    /// Tasks are distributed among a fixed number of OS threads
    ManualThreads(uint),
    /**
     * Tasks are scheduled on the main OS thread
     *
     * The main OS thread is the thread used to launch the runtime which,
     * in most cases, is the process's initial thread as created by the OS.
     */
    PlatformThread
}

#[cfg(stage0)]
impl SchedMode : cmp::Eq {
    pure fn eq(&&other: SchedMode) -> bool {
        match self {
            SingleThreaded => {
                match other {
                    SingleThreaded => true,
                    _ => false
                }
            }
            ThreadPerCore => {
                match other {
                    ThreadPerCore => true,
                    _ => false
                }
            }
            ThreadPerTask => {
                match other {
                    ThreadPerTask => true,
                    _ => false
                }
            }
            ManualThreads(e0a) => {
                match other {
                    ManualThreads(e0b) => e0a == e0b,
                    _ => false
                }
            }
            PlatformThread => {
                match other {
                    PlatformThread => true,
                    _ => false
                }
            }
        }
    }
    pure fn ne(&&other: SchedMode) -> bool {
        !self.eq(other)
    }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl SchedMode : cmp::Eq {
    pure fn eq(other: &SchedMode) -> bool {
        match self {
            SingleThreaded => {
                match (*other) {
                    SingleThreaded => true,
                    _ => false
                }
            }
            ThreadPerCore => {
                match (*other) {
                    ThreadPerCore => true,
                    _ => false
                }
            }
            ThreadPerTask => {
                match (*other) {
                    ThreadPerTask => true,
                    _ => false
                }
            }
            ManualThreads(e0a) => {
                match (*other) {
                    ManualThreads(e0b) => e0a == e0b,
                    _ => false
                }
            }
            PlatformThread => {
                match (*other) {
                    PlatformThread => true,
                    _ => false
                }
            }
        }
    }
    pure fn ne(other: &SchedMode) -> bool {
        !self.eq(other)
    }
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
type SchedOpts = {
    mode: SchedMode,
    foreign_stack_size: Option<uint>
};

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
type TaskOpts = {
    linked: bool,
    supervised: bool,
    mut notify_chan: Option<Chan<Notification>>,
    sched: Option<SchedOpts>,
};

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

// FIXME (#2585): Replace the 'consumed' bit with move mode on self
enum TaskBuilder = {
    opts: TaskOpts,
    gen_body: fn@(+fn~()) -> fn~(),
    can_not_copy: Option<util::NonCopyable>,
    mut consumed: bool,
};

/**
 * Generate the base configuration for spawning a task, off of which more
 * configuration methods can be chained.
 * For example, task().unlinked().spawn is equivalent to spawn_unlinked.
 */
fn task() -> TaskBuilder {
    TaskBuilder({
        opts: default_task_opts(),
        gen_body: |body| move body, // Identity function
        can_not_copy: None,
        mut consumed: false,
    })
}

#[doc(hidden)] // FIXME #3538
priv impl TaskBuilder {
    fn consume() -> TaskBuilder {
        if self.consumed {
            fail ~"Cannot copy a task_builder"; // Fake move mode on self
        }
        self.consumed = true;
        let notify_chan = if self.opts.notify_chan.is_none() {
            None
        } else {
            Some(option::swap_unwrap(&mut self.opts.notify_chan))
        };
        TaskBuilder({
            opts: {
                linked: self.opts.linked,
                supervised: self.opts.supervised,
                mut notify_chan: move notify_chan,
                sched: self.opts.sched
            },
            gen_body: self.gen_body,
            can_not_copy: None,
            mut consumed: false
        })
    }
}

impl TaskBuilder {
    /**
     * Decouple the child task's failure from the parent's. If either fails,
     * the other will not be killed.
     */
    fn unlinked() -> TaskBuilder {
        let notify_chan = if self.opts.notify_chan.is_none() {
            None
        } else {
            Some(option::swap_unwrap(&mut self.opts.notify_chan))
        };
        TaskBuilder({
            opts: {
                linked: false,
                supervised: self.opts.supervised,
                mut notify_chan: move notify_chan,
                sched: self.opts.sched
            },
            can_not_copy: None,
            .. *self.consume()
        })
    }
    /**
     * Unidirectionally link the child task's failure with the parent's. The
     * child's failure will not kill the parent, but the parent's will kill
     * the child.
     */
    fn supervised() -> TaskBuilder {
        let notify_chan = if self.opts.notify_chan.is_none() {
            None
        } else {
            Some(option::swap_unwrap(&mut self.opts.notify_chan))
        };
        TaskBuilder({
            opts: {
                linked: false,
                supervised: true,
                mut notify_chan: move notify_chan,
                sched: self.opts.sched
            },
            can_not_copy: None,
            .. *self.consume()
        })
    }
    /**
     * Link the child task's and parent task's failures. If either fails, the
     * other will be killed.
     */
    fn linked() -> TaskBuilder {
        let notify_chan = if self.opts.notify_chan.is_none() {
            None
        } else {
            Some(option::swap_unwrap(&mut self.opts.notify_chan))
        };
        TaskBuilder({
            opts: {
                linked: true,
                supervised: false,
                mut notify_chan: move notify_chan,
                sched: self.opts.sched
            },
            can_not_copy: None,
            .. *self.consume()
        })
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
    fn future_result(blk: fn(+future::Future<TaskResult>)) -> TaskBuilder {
        // FIXME (#1087, #1857): Once linked failure and notification are
        // handled in the library, I can imagine implementing this by just
        // registering an arbitrary number of task::on_exit handlers and
        // sending out messages.

        if self.opts.notify_chan.is_some() {
            fail ~"Can't set multiple future_results for one task!";
        }

        // Construct the future and give it to the caller.
        let (notify_pipe_ch, notify_pipe_po) = stream::<Notification>();

        blk(do future::from_fn |move notify_pipe_po| {
            match notify_pipe_po.recv() {
              Exit(_, result) => result
            }
        });

        // Reconfigure self to use a notify channel.
        TaskBuilder({
            opts: {
                linked: self.opts.linked,
                supervised: self.opts.supervised,
                mut notify_chan: Some(move notify_pipe_ch),
                sched: self.opts.sched
            },
            can_not_copy: None,
            .. *self.consume()
        })
    }
    /// Configure a custom scheduler mode for the task.
    fn sched_mode(mode: SchedMode) -> TaskBuilder {
        let notify_chan = if self.opts.notify_chan.is_none() {
            None
        } else {
            Some(option::swap_unwrap(&mut self.opts.notify_chan))
        };
        TaskBuilder({
            opts: {
                linked: self.opts.linked,
                supervised: self.opts.supervised,
                mut notify_chan: move notify_chan,
                sched: Some({ mode: mode, foreign_stack_size: None})
            },
            can_not_copy: None,
            .. *self.consume()
        })
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
    fn add_wrapper(wrapper: fn@(+fn~()) -> fn~()) -> TaskBuilder {
        let prev_gen_body = self.gen_body;
        let notify_chan = if self.opts.notify_chan.is_none() {
            None
        } else {
            Some(option::swap_unwrap(&mut self.opts.notify_chan))
        };
        TaskBuilder({
            opts: {
                linked: self.opts.linked,
                supervised: self.opts.supervised,
                mut notify_chan: move notify_chan,
                sched: self.opts.sched
            },
            gen_body: |body| { wrapper(prev_gen_body(move body)) },
            can_not_copy: None,
            .. *self.consume()
        })
    }

    /**
     * Creates and exucutes a new child task
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
    fn spawn(+f: fn~()) {
        let notify_chan = if self.opts.notify_chan.is_none() {
            None
        } else {
            let swapped_notify_chan =
                option::swap_unwrap(&mut self.opts.notify_chan);
            Some(move swapped_notify_chan)
        };
        let x = self.consume();
        let opts = {
            linked: x.opts.linked,
            supervised: x.opts.supervised,
            mut notify_chan: move notify_chan,
            sched: x.opts.sched
        };
        spawn::spawn_raw(move opts, x.gen_body(move f));
    }
    /// Runs a task, while transfering ownership of one argument to the child.
    fn spawn_with<A: Send>(+arg: A, +f: fn~(+A)) {
        let arg = ~mut Some(move arg);
        do self.spawn |move arg, move f|{
            f(option::swap_unwrap(arg))
        }
    }

    /**
     * Runs a new task while providing a channel from the parent to the child
     *
     * Sets up a communication channel from the current task to the new
     * child task, passes the port to child's body, and returns a channel
     * linked to the port to the parent.
     *
     * This encapsulates Some boilerplate handshaking logic that would
     * otherwise be required to establish communication from the parent
     * to the child.
     */
    fn spawn_listener<A: Send>(+f: fn~(comm::Port<A>)) -> comm::Chan<A> {
        let setup_po = comm::Port();
        let setup_ch = comm::Chan(setup_po);
        do self.spawn |move f| {
            let po = comm::Port();
            let ch = comm::Chan(po);
            comm::send(setup_ch, ch);
            f(move po);
        }
        comm::recv(setup_po)
    }

    /**
     * Runs a new task, setting up communication in both directions
     */
    fn spawn_conversation<A: Send, B: Send>
        (+f: fn~(comm::Port<A>, comm::Chan<B>))
        -> (comm::Port<B>, comm::Chan<A>) {
        let from_child = comm::Port();
        let to_parent = comm::Chan(from_child);
        let to_child = do self.spawn_listener |move f, from_parent| {
            f(from_parent, to_parent)
        };
        (from_child, to_child)
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
    fn try<T: Send>(+f: fn~() -> T) -> Result<T,()> {
        let po = comm::Port();
        let ch = comm::Chan(po);
        let mut result = None;

        let fr_task_builder = self.future_result(|+r| {
            result = Some(move r);
        });
        do fr_task_builder.spawn |move f| {
            comm::send(ch, f());
        }
        match future::get(&option::unwrap(move result)) {
            Success => result::Ok(comm::recv(po)),
            Failure => result::Err(())
        }
    }
}


/* Task construction */

fn default_task_opts() -> TaskOpts {
    /*!
     * The default task options
     *
     * By default all tasks are supervised by their parent, are spawned
     * into the same scheduler, and do not post lifecycle notifications.
     */

    {
        linked: true,
        supervised: false,
        mut notify_chan: None,
        sched: None
    }
}

/* Spawn convenience functions */

fn spawn(+f: fn~()) {
    /*!
     * Creates and executes a new child task
     *
     * Sets up a new task with its own call stack and schedules it to run
     * the provided unique closure.
     *
     * This function is equivalent to `task().spawn(f)`.
     */

    task().spawn(move f)
}

fn spawn_unlinked(+f: fn~()) {
    /*!
     * Creates a child task unlinked from the current one. If either this
     * task or the child task fails, the other will not be killed.
     */

    task().unlinked().spawn(move f)
}

fn spawn_supervised(+f: fn~()) {
    /*!
     * Creates a child task unlinked from the current one. If either this
     * task or the child task fails, the other will not be killed.
     */

    task().supervised().spawn(move f)
}

fn spawn_with<A:Send>(+arg: A, +f: fn~(+A)) {
    /*!
     * Runs a task, while transfering ownership of one argument to the
     * child.
     *
     * This is useful for transfering ownership of noncopyables to
     * another task.
     *
     * This function is equivalent to `task().spawn_with(arg, f)`.
     */

    task().spawn_with(move arg, move f)
}

fn spawn_listener<A:Send>(+f: fn~(comm::Port<A>)) -> comm::Chan<A> {
    /*!
     * Runs a new task while providing a channel from the parent to the child
     *
     * This function is equivalent to `task().spawn_listener(f)`.
     */

    task().spawn_listener(move f)
}

fn spawn_conversation<A: Send, B: Send>
    (+f: fn~(comm::Port<A>, comm::Chan<B>))
    -> (comm::Port<B>, comm::Chan<A>) {
    /*!
     * Runs a new task, setting up communication in both directions
     *
     * This function is equivalent to `task().spawn_conversation(f)`.
     */

    task().spawn_conversation(move f)
}

fn spawn_sched(mode: SchedMode, +f: fn~()) {
    /*!
     * Creates a new scheduler and executes a task on it
     *
     * Tasks subsequently spawned by that task will also execute on
     * the new scheduler. When there are no more tasks to execute the
     * scheduler terminates.
     *
     * # Failure
     *
     * In manual threads mode the number of threads requested must be
     * greater than zero.
     */

    task().sched_mode(mode).spawn(move f)
}

fn try<T:Send>(+f: fn~() -> T) -> Result<T,()> {
    /*!
     * Execute a function in another task and return either the return value
     * of the function or result::err.
     *
     * This is equivalent to task().supervised().try.
     */

    task().supervised().try(move f)
}


/* Lifecycle functions */

fn yield() {
    //! Yield control to the task scheduler

    let task_ = rt::rust_get_task();
    let killed = rt::rust_task_yield(task_);
    if killed && !failing() {
        fail ~"killed";
    }
}

fn failing() -> bool {
    //! True if the running task has failed

    rt::rust_task_is_unwinding(rt::rust_get_task())
}

fn get_task() -> Task {
    //! Get a handle to the running task

    TaskHandle(rt::get_task_id())
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
unsafe fn unkillable<U>(f: fn() -> U) -> U {
    struct AllowFailure {
        t: *rust_task,
        drop { rt::rust_task_allow_kill(self.t); }
    }

    fn AllowFailure(t: *rust_task) -> AllowFailure{
        AllowFailure {
            t: t
        }
    }

    let t = rt::rust_get_task();
    let _allow_failure = AllowFailure(t);
    rt::rust_task_inhibit_kill(t);
    f()
}

/// The inverse of unkillable. Only ever to be used nested in unkillable().
unsafe fn rekillable<U>(f: fn() -> U) -> U {
    struct DisallowFailure {
        t: *rust_task,
        drop { rt::rust_task_inhibit_kill(self.t); }
    }

    fn DisallowFailure(t: *rust_task) -> DisallowFailure {
        DisallowFailure {
            t: t
        }
    }

    let t = rt::rust_get_task();
    let _allow_failure = DisallowFailure(t);
    rt::rust_task_allow_kill(t);
    f()
}

/**
 * A stronger version of unkillable that also inhibits scheduling operations.
 * For use with exclusive ARCs, which use pthread mutexes directly.
 */
unsafe fn atomically<U>(f: fn() -> U) -> U {
    struct DeferInterrupts {
        t: *rust_task,
        drop {
            rt::rust_task_allow_yield(self.t);
            rt::rust_task_allow_kill(self.t);
        }
    }

    fn DeferInterrupts(t: *rust_task) -> DeferInterrupts {
        DeferInterrupts {
            t: t
        }
    }

    let t = rt::rust_get_task();
    let _interrupts = DeferInterrupts(t);
    rt::rust_task_inhibit_kill(t);
    rt::rust_task_inhibit_yield(t);
    f()
}

#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_cant_dup_task_builder() {
    let b = task().unlinked();
    do b.spawn { }
    // FIXME(#2585): For now, this is a -runtime- failure, because we haven't
    // got modes on self. When 2585 is fixed, this test should fail to compile
    // instead, and should go in tests/compile-fail.
    do b.spawn { } // b should have been consumed by the previous call
}

// The following 8 tests test the following 2^3 combinations:
// {un,}linked {un,}supervised failure propagation {up,down}wards.

// !!! These tests are dangerous. If Something is buggy, they will hang, !!!
// !!! instead of exiting cleanly. This might wedge the buildbots.       !!!

#[test] #[ignore(cfg(windows))]
fn test_spawn_unlinked_unsup_no_fail_down() { // grandchild sends on a port
    let po = comm::Port();
    let ch = comm::Chan(po);
    do spawn_unlinked {
        do spawn_unlinked {
            // Give middle task a chance to fail-but-not-kill-us.
            for iter::repeat(16) { task::yield(); }
            comm::send(ch, ()); // If killed first, grandparent hangs.
        }
        fail; // Shouldn't kill either (grand)parent or (grand)child.
    }
    comm::recv(po);
}
#[test] #[ignore(cfg(windows))]
fn test_spawn_unlinked_unsup_no_fail_up() { // child unlinked fails
    do spawn_unlinked { fail; }
}
#[test] #[ignore(cfg(windows))]
fn test_spawn_unlinked_sup_no_fail_up() { // child unlinked fails
    do spawn_supervised { fail; }
    // Give child a chance to fail-but-not-kill-us.
    for iter::repeat(16) { task::yield(); }
}
#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_unlinked_sup_fail_down() {
    do spawn_supervised { loop { task::yield(); } }
    fail; // Shouldn't leave a child hanging around.
}

#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_linked_sup_fail_up() { // child fails; parent fails
    let po = comm::Port::<()>();
    let _ch = comm::Chan(po);
    // Unidirectional "parenting" shouldn't override bidirectional linked.
    // We have to cheat with opts - the interface doesn't support them because
    // they don't make sense (redundant with task().supervised()).
    let opts = {
        let mut opts = default_task_opts();
        opts.linked = true;
        opts.supervised = true;
        move opts
    };

    let b0 = task();
    let b1 = TaskBuilder({
        opts: move opts,
        can_not_copy: None,
        .. *b0
    });
    do b1.spawn { fail; }
    comm::recv(po); // We should get punted awake
}
#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_linked_sup_fail_down() { // parent fails; child fails
    // We have to cheat with opts - the interface doesn't support them because
    // they don't make sense (redundant with task().supervised()).
    let opts = {
        let mut opts = default_task_opts();
        opts.linked = true;
        opts.supervised = true;
        move opts
    };

    let b0 = task();
    let b1 = TaskBuilder({
        opts: move opts,
        can_not_copy: None,
        .. *b0
    });
    do b1.spawn { loop { task::yield(); } }
    fail; // *both* mechanisms would be wrong if this didn't kill the child...
}
#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_linked_unsup_fail_up() { // child fails; parent fails
    let po = comm::Port::<()>();
    let _ch = comm::Chan(po);
    // Default options are to spawn linked & unsupervised.
    do spawn { fail; }
    comm::recv(po); // We should get punted awake
}
#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_linked_unsup_fail_down() { // parent fails; child fails
    // Default options are to spawn linked & unsupervised.
    do spawn { loop { task::yield(); } }
    fail;
}
#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_linked_unsup_default_opts() { // parent fails; child fails
    // Make sure the above test is the same as this one.
    do task().linked().spawn { loop { task::yield(); } }
    fail;
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
    for iter::repeat(16) { task::yield(); }
    fail;
}

#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_failure_propagate_secondborn() {
    // First-born child exits; does parent's failure propagate to sibling?
    do spawn_supervised {
        do spawn { // linked
            loop { task::yield(); }
        }
    }
    for iter::repeat(16) { task::yield(); }
    fail;
}

#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_failure_propagate_nephew_or_niece() {
    // Our sibling exits; does our failure propagate to sibling's child?
    do spawn { // linked
        do spawn_supervised {
            loop { task::yield(); }
        }
    }
    for iter::repeat(16) { task::yield(); }
    fail;
}

#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_linked_sup_propagate_sibling() {
    // Middle sibling exits - does eldest's failure propagate to youngest?
    do spawn { // linked
        do spawn { // linked
            loop { task::yield(); }
        }
    }
    for iter::repeat(16) { task::yield(); }
    fail;
}

#[test]
fn test_run_basic() {
    let po = comm::Port();
    let ch = comm::Chan(po);
    do task().spawn {
        comm::send(ch, ());
    }
    comm::recv(po);
}

#[test]
fn test_add_wrapper() {
    let po = comm::Port();
    let ch = comm::Chan(po);
    let b0 = task();
    let b1 = do b0.add_wrapper |body| {
        fn~() {
            body();
            comm::send(ch, ());
        }
    };
    do b1.spawn { }
    comm::recv(po);
}

#[test]
#[ignore(cfg(windows))]
fn test_future_result() {
    let mut result = None;
    do task().future_result(|+r| { result = Some(r); }).spawn { }
    assert future::get(&option::unwrap(result)) == Success;

    result = None;
    do task().future_result(|+r| { result = Some(r); }).unlinked().spawn {
        fail;
    }
    assert future::get(&option::unwrap(result)) == Failure;
}

#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_back_to_the_future_result() {
    let _ = task().future_result(util::ignore).future_result(util::ignore);
}

#[test]
fn test_spawn_listiner_bidi() {
    let po = comm::Port();
    let ch = comm::Chan(po);
    let ch = do spawn_listener |po| {
        // Now the child has a port called 'po' to read from and
        // an environment-captured channel called 'ch'.
        let res: ~str = comm::recv(po);
        assert res == ~"ping";
        comm::send(ch, ~"pong");
    };
    // Likewise, the parent has both a 'po' and 'ch'
    comm::send(ch, ~"ping");
    let res: ~str = comm::recv(po);
    assert res == ~"pong";
}

#[test]
fn test_spawn_conversation() {
    let (recv_str, send_int) = do spawn_conversation |recv_int, send_str| {
        let input = comm::recv(recv_int);
        let output = int::str(input);
        comm::send(send_str, output);
    };
    comm::send(send_int, 1);
    assert comm::recv(recv_str) == ~"1";
}

#[test]
fn test_try_success() {
    match do try {
        ~"Success!"
    } {
        result::Ok(~"Success!") => (),
        _ => fail
    }
}

#[test]
#[ignore(cfg(windows))]
fn test_try_fail() {
    match do try {
        fail
    } {
        result::Err(()) => (),
        result::Ok(()) => fail
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
    let po = comm::Port();
    let ch = comm::Chan(po);

    fn f(i: int, ch: comm::Chan<()>) {
        let parent_sched_id = rt::rust_get_sched_id();

        do spawn_sched(SingleThreaded) {
            let child_sched_id = rt::rust_get_sched_id();
            assert parent_sched_id != child_sched_id;

            if (i == 0) {
                comm::send(ch, ());
            } else {
                f(i - 1, ch);
            }
        };

    }
    f(10, ch);
    comm::recv(po);
}

#[test]
fn test_spawn_sched_childs_on_same_sched() {
    let po = comm::Port();
    let ch = comm::Chan(po);

    do spawn_sched(SingleThreaded) {
        let parent_sched_id = rt::rust_get_sched_id();
        do spawn {
            let child_sched_id = rt::rust_get_sched_id();
            // This should be on the same scheduler
            assert parent_sched_id == child_sched_id;
            comm::send(ch, ());
        };
    };

    comm::recv(po);
}

#[nolink]
#[cfg(test)]
extern mod testrt {
    #[legacy_exports];
    fn rust_dbg_lock_create() -> *libc::c_void;
    fn rust_dbg_lock_destroy(lock: *libc::c_void);
    fn rust_dbg_lock_lock(lock: *libc::c_void);
    fn rust_dbg_lock_unlock(lock: *libc::c_void);
    fn rust_dbg_lock_wait(lock: *libc::c_void);
    fn rust_dbg_lock_signal(lock: *libc::c_void);
}

#[test]
fn test_spawn_sched_blocking() {

    // Testing that a task in one scheduler can block in foreign code
    // without affecting other schedulers
    for iter::repeat(20u) {

        let start_po = comm::Port();
        let start_ch = comm::Chan(start_po);
        let fin_po = comm::Port();
        let fin_ch = comm::Chan(fin_po);

        let lock = testrt::rust_dbg_lock_create();

        do spawn_sched(SingleThreaded) {
            testrt::rust_dbg_lock_lock(lock);

            comm::send(start_ch, ());

            // Block the scheduler thread
            testrt::rust_dbg_lock_wait(lock);
            testrt::rust_dbg_lock_unlock(lock);

            comm::send(fin_ch, ());
        };

        // Wait until the other task has its lock
        comm::recv(start_po);

        fn pingpong(po: comm::Port<int>, ch: comm::Chan<int>) {
            let mut val = 20;
            while val > 0 {
                val = comm::recv(po);
                comm::send(ch, val - 1);
            }
        }

        let setup_po = comm::Port();
        let setup_ch = comm::Chan(setup_po);
        let parent_po = comm::Port();
        let parent_ch = comm::Chan(parent_po);
        do spawn {
            let child_po = comm::Port();
            comm::send(setup_ch, comm::Chan(child_po));
            pingpong(child_po, parent_ch);
        };

        let child_ch = comm::recv(setup_po);
        comm::send(child_ch, 20);
        pingpong(parent_po, child_ch);
        testrt::rust_dbg_lock_lock(lock);
        testrt::rust_dbg_lock_signal(lock);
        testrt::rust_dbg_lock_unlock(lock);
        comm::recv(fin_po);
        testrt::rust_dbg_lock_destroy(lock);
    }
}

#[cfg(test)]
fn avoid_copying_the_body(spawnfn: fn(+fn~())) {
    let p = comm::Port::<uint>();
    let ch = comm::Chan(p);

    let x = ~1;
    let x_in_parent = ptr::addr_of(*x) as uint;

    do spawnfn {
        let x_in_child = ptr::addr_of(*x) as uint;
        comm::send(ch, x_in_child);
    }

    let x_in_child = comm::recv(p);
    assert x_in_parent == x_in_child;
}

#[test]
fn test_avoid_copying_the_body_spawn() {
    avoid_copying_the_body(spawn);
}

#[test]
fn test_avoid_copying_the_body_spawn_listener() {
    do avoid_copying_the_body |f| {
        spawn_listener(fn~(move f, _po: comm::Port<int>) {
            f();
        });
    }
}

#[test]
fn test_avoid_copying_the_body_task_spawn() {
    do avoid_copying_the_body |f| {
        do task().spawn {
            f();
        }
    }
}

#[test]
fn test_avoid_copying_the_body_spawn_listener_1() {
    do avoid_copying_the_body |f| {
        task().spawn_listener(fn~(move f, _po: comm::Port<int>) {
            f();
        });
    }
}

#[test]
fn test_avoid_copying_the_body_try() {
    do avoid_copying_the_body |f| {
        do try {
            f()
        };
    }
}

#[test]
fn test_avoid_copying_the_body_unlinked() {
    do avoid_copying_the_body |f| {
        do spawn_unlinked {
            f();
        }
    }
}

#[test]
fn test_platform_thread() {
    let po = comm::Port();
    let ch = comm::Chan(po);
    do task().sched_mode(PlatformThread).spawn {
        comm::send(ch, ());
    }
    comm::recv(po);
}

#[test]
#[ignore(cfg(windows))]
#[should_fail]
fn test_unkillable() {
    let po = comm::Port();
    let ch = po.chan();

    // We want to do this after failing
    do spawn_unlinked {
        for iter::repeat(10u) { yield() }
        ch.send(());
    }

    do spawn {
        yield();
        // We want to fail after the unkillable task
        // blocks on recv
        fail;
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
    let (ch, po) = pipes::stream();

    // We want to do this after failing
    do spawn_unlinked {
        for iter::repeat(10u) { yield() }
        ch.send(());
    }

    do spawn {
        yield();
        // We want to fail after the unkillable task
        // blocks on recv
        fail;
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
    const generations: uint = 16;
    fn child_no(x: uint) -> fn~() {
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
    let (chan, port) = pipes::stream();

    do spawn_sched(ThreadPerCore) {
        let cores = rt::rust_num_threads();
        let reported_threads = rt::rust_sched_threads();
        assert(cores as uint == reported_threads as uint);
        chan.send(());
    }

    port.recv();
}

#[test]
fn test_spawn_thread_on_demand() {
    let (chan, port) = pipes::stream();

    do spawn_sched(ManualThreads(2)) {
        let max_threads = rt::rust_sched_threads();
        assert(max_threads as int == 2);
        let running_threads = rt::rust_sched_current_nonlazy_threads();
        assert(running_threads as int == 1);

        let (chan2, port2) = pipes::stream();

        do spawn() {
            chan2.send(());
        }

        let running_threads2 = rt::rust_sched_current_nonlazy_threads();
        assert(running_threads2 as int == 2);

        port2.recv();
        chan.send(());
    }

    port.recv();
}
