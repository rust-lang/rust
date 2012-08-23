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

import result::result;

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

export local_data_key;
export local_data_pop;
export local_data_get;
export local_data_set;
export local_data_modify;

export SingleThreaded;
export ThreadPerCore;
export ThreadPerTask;
export ManualThreads;
export PlatformThread;

/* Data types */

/// A handle to a task
enum Task { TaskHandle(task_id) }

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

/// A message type for notifying of task lifecycle events
enum Notification {
    /// Sent when a task exits with the task handle and result
    Exit(Task, TaskResult)
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
    foreign_stack_size: option<uint>
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
    notify_chan: option<comm::Chan<Notification>>,
    sched: option<SchedOpts>,
};

/**
 * The task builder type.
 *
 * Provides detailed control over the properties and behavior of new tasks.
 */
// NB: Builders are designed to be single-use because they do stateful
// things that get weird when reusing - e.g. if you create a result future
// it only applies to a single task, so then you have to maintain some
// potentially tricky state to ensure that everything behaves correctly
// when you try to reuse the builder to spawn a new task. We'll just
// sidestep that whole issue by making builders uncopyable and making
// the run function move them in.

// FIXME (#2585): Replace the 'consumed' bit with move mode on self
enum TaskBuilder = {
    opts: TaskOpts,
    gen_body: fn@(+fn~()) -> fn~(),
    can_not_copy: option<util::NonCopyable>,
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
        gen_body: |body| body, // Identity function
        can_not_copy: none,
        mut consumed: false,
    })
}

priv impl TaskBuilder {
    fn consume() -> TaskBuilder {
        if self.consumed {
            fail ~"Cannot copy a task_builder"; // Fake move mode on self
        }
        self.consumed = true;
        TaskBuilder({ can_not_copy: none, mut consumed: false, with *self })
    }
}

impl TaskBuilder {
    /**
     * Decouple the child task's failure from the parent's. If either fails,
     * the other will not be killed.
     */
    fn unlinked() -> TaskBuilder {
        TaskBuilder({
            opts: { linked: false with self.opts },
            can_not_copy: none,
            with *self.consume()
        })
    }
    /**
     * Unidirectionally link the child task's failure with the parent's. The
     * child's failure will not kill the parent, but the parent's will kill
     * the child.
     */
    fn supervised() -> TaskBuilder {
        TaskBuilder({
            opts: { linked: false, supervised: true with self.opts },
            can_not_copy: none,
            with *self.consume()
        })
    }
    /**
     * Link the child task's and parent task's failures. If either fails, the
     * other will be killed.
     */
    fn linked() -> TaskBuilder {
        TaskBuilder({
            opts: { linked: true, supervised: false with self.opts },
            can_not_copy: none,
            with *self.consume()
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
        let po = comm::port::<Notification>();
        let ch = comm::chan(po);

        blk(do future::from_fn {
            match comm::recv(po) {
              Exit(_, result) => result
            }
        });

        // Reconfigure self to use a notify channel.
        TaskBuilder({
            opts: { notify_chan: some(ch) with self.opts },
            can_not_copy: none,
            with *self.consume()
        })
    }
    /// Configure a custom scheduler mode for the task.
    fn sched_mode(mode: SchedMode) -> TaskBuilder {
        TaskBuilder({
            opts: { sched: some({ mode: mode, foreign_stack_size: none})
                    with self.opts },
            can_not_copy: none,
            with *self.consume()
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
        TaskBuilder({
            gen_body: |body| { wrapper(prev_gen_body(body)) },
            can_not_copy: none,
            with *self.consume()
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
        let x = self.consume();
        spawn_raw(x.opts, x.gen_body(f));
    }
    /// Runs a task, while transfering ownership of one argument to the child.
    fn spawn_with<A: send>(+arg: A, +f: fn~(+A)) {
        let arg = ~mut some(arg);
        do self.spawn {
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
     * This encapsulates some boilerplate handshaking logic that would
     * otherwise be required to establish communication from the parent
     * to the child.
     */
    fn spawn_listener<A: send>(+f: fn~(comm::Port<A>)) -> comm::Chan<A> {
        let setup_po = comm::port();
        let setup_ch = comm::chan(setup_po);
        do self.spawn {
            let po = comm::port();
            let ch = comm::chan(po);
            comm::send(setup_ch, ch);
            f(po);
        }
        comm::recv(setup_po)
    }

    /**
     * Runs a new task, setting up communication in both directions
     */
    fn spawn_conversation<A: send, B: send>
        (+f: fn~(comm::Port<A>, comm::Chan<B>))
        -> (comm::Port<B>, comm::Chan<A>) {
        let from_child = comm::port();
        let to_parent = comm::chan(from_child);
        let to_child = do self.spawn_listener |from_parent| {
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
    fn try<T: send>(+f: fn~() -> T) -> result<T,()> {
        let po = comm::port();
        let ch = comm::chan(po);
        let mut result = none;

        do self.future_result(|+r| { result = some(r); }).spawn {
            comm::send(ch, f());
        }
        match future::get(&option::unwrap(result)) {
            Success => result::ok(comm::recv(po)),
            Failure => result::err(())
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
        notify_chan: none,
        sched: none
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

    task().spawn(f)
}

fn spawn_unlinked(+f: fn~()) {
    /*!
     * Creates a child task unlinked from the current one. If either this
     * task or the child task fails, the other will not be killed.
     */

    task().unlinked().spawn(f)
}

fn spawn_supervised(+f: fn~()) {
    /*!
     * Creates a child task unlinked from the current one. If either this
     * task or the child task fails, the other will not be killed.
     */

    task().supervised().spawn(f)
}

fn spawn_with<A:send>(+arg: A, +f: fn~(+A)) {
    /*!
     * Runs a task, while transfering ownership of one argument to the
     * child.
     *
     * This is useful for transfering ownership of noncopyables to
     * another task.
     *
     * This function is equivalent to `task().spawn_with(arg, f)`.
     */

    task().spawn_with(arg, f)
}

fn spawn_listener<A:send>(+f: fn~(comm::Port<A>)) -> comm::Chan<A> {
    /*!
     * Runs a new task while providing a channel from the parent to the child
     *
     * This function is equivalent to `task().spawn_listener(f)`.
     */

    task().spawn_listener(f)
}

fn spawn_conversation<A: send, B: send>
    (+f: fn~(comm::Port<A>, comm::Chan<B>))
    -> (comm::Port<B>, comm::Chan<A>) {
    /*!
     * Runs a new task, setting up communication in both directions
     *
     * This function is equivalent to `task().spawn_conversation(f)`.
     */

    task().spawn_conversation(f)
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

    task().sched_mode(mode).spawn(f)
}

fn try<T:send>(+f: fn~() -> T) -> result<T,()> {
    /*!
     * Execute a function in another task and return either the return value
     * of the function or result::err.
     *
     * This is equivalent to task().supervised().try.
     */

    task().supervised().try(f)
}


/* Lifecycle functions */

fn yield() {
    //! Yield control to the task scheduler

    let task_ = rustrt::rust_get_task();
    let killed = rustrt::rust_task_yield(task_);
    if killed && !failing() {
        fail ~"killed";
    }
}

fn failing() -> bool {
    //! True if the running task has failed

    rustrt::rust_task_is_unwinding(rustrt::rust_get_task())
}

fn get_task() -> Task {
    //! Get a handle to the running task

    TaskHandle(rustrt::get_task_id())
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
        let t: *rust_task;
        new(t: *rust_task) { self.t = t; }
        drop { rustrt::rust_task_allow_kill(self.t); }
    }

    let t = rustrt::rust_get_task();
    let _allow_failure = AllowFailure(t);
    rustrt::rust_task_inhibit_kill(t);
    f()
}

/// The inverse of unkillable. Only ever to be used nested in unkillable().
unsafe fn rekillable<U>(f: fn() -> U) -> U {
    struct DisallowFailure {
        let t: *rust_task;
        new(t: *rust_task) { self.t = t; }
        drop { rustrt::rust_task_inhibit_kill(self.t); }
    }

    let t = rustrt::rust_get_task();
    let _allow_failure = DisallowFailure(t);
    rustrt::rust_task_allow_kill(t);
    f()
}

/**
 * A stronger version of unkillable that also inhibits scheduling operations.
 * For use with exclusive ARCs, which use pthread mutexes directly.
 */
unsafe fn atomically<U>(f: fn() -> U) -> U {
    struct DeferInterrupts {
        let t: *rust_task;
        new(t: *rust_task) { self.t = t; }
        drop {
            rustrt::rust_task_allow_yield(self.t);
            rustrt::rust_task_allow_kill(self.t);
        }
    }
    let t = rustrt::rust_get_task();
    let _interrupts = DeferInterrupts(t);
    rustrt::rust_task_inhibit_kill(t);
    rustrt::rust_task_inhibit_yield(t);
    f()
}

/****************************************************************************
 * Spawning & linked failure
 *
 * Several data structures are involved in task management to allow properly
 * propagating failure across linked/supervised tasks.
 *
 * (1) The "taskgroup_arc" is an unsafe::exclusive which contains a hashset of
 *     all tasks that are part of the group. Some tasks are 'members', which
 *     means if they fail, they will kill everybody else in the taskgroup.
 *     Other tasks are 'descendants', which means they will not kill tasks
 *     from this group, but can be killed by failing members.
 *
 *     A new one of these is created each spawn_linked or spawn_supervised.
 *
 * (2) The "tcb" is a per-task control structure that tracks a task's spawn
 *     configuration. It contains a reference to its taskgroup_arc, a
 *     reference to its node in the ancestor list (below), a flag for
 *     whether it's part of the 'main'/'root' taskgroup, and an optionally
 *     configured notification port. These are stored in TLS.
 *
 * (3) The "ancestor_list" is a cons-style list of unsafe::exclusives which
 *     tracks 'generations' of taskgroups -- a group's ancestors are groups
 *     which (directly or transitively) spawn_supervised-ed them. Each task
 *     is recorded in the 'descendants' of each of its ancestor groups.
 *
 *     Spawning a supervised task is O(n) in the number of generations still
 *     alive, and exiting (by success or failure) that task is also O(n).
 *
 * This diagram depicts the references between these data structures:
 *
 *          linked_________________________________
 *        ___/                   _________         \___
 *       /   \                  | group X |        /   \
 *      (  A  ) - - - - - - - > | {A,B} {}|< - - -(  B  )
 *       \___/                  |_________|        \___/
 *      unlinked
 *         |      __ (nil)
 *         |      //|                         The following code causes this:
 *         |__   //   /\         _________
 *        /   \ //    ||        | group Y |     fn taskA() {
 *       (  C  )- - - ||- - - > |{C} {D,E}|         spawn(taskB);
 *        \___/      /  \=====> |_________|         spawn_unlinked(taskC);
 *      supervise   /gen \                          ...
 *         |    __  \ 00 /                      }
 *         |    //|  \__/                       fn taskB() { ... }
 *         |__ //     /\         _________      fn taskC() {
 *        /   \/      ||        | group Z |         spawn_supervised(taskD);
 *       (  D  )- - - ||- - - > | {D} {E} |         ...
 *        \___/      /  \=====> |_________|     }
 *      supervise   /gen \                      fn taskD() {
 *         |    __  \ 01 /                          spawn_supervised(taskE);
 *         |    //|  \__/                           ...
 *         |__ //                _________      }
 *        /   \/                | group W |     fn taskE() { ... }
 *       (  E  )- - - - - - - > | {E}  {} |
 *        \___/                 |_________|
 *
 *        "tcb"               "taskgroup_arc"
 *             "ancestor_list"
 *
 ****************************************************************************/

#[allow(non_camel_case_types)] // runtime type
type sched_id = int;
#[allow(non_camel_case_types)] // runtime type
type task_id = int;

// These are both opaque runtime/compiler types that we don't know the
// structure of and should only deal with via unsafe pointer
#[allow(non_camel_case_types)] // runtime type
type rust_task = libc::c_void;
#[allow(non_camel_case_types)] // runtime type
type rust_closure = libc::c_void;

type TaskSet = send_map::linear::LinearMap<*rust_task,()>;

fn new_taskset() -> TaskSet {
    pure fn task_hash(t: &*rust_task) -> uint {
        let task: *rust_task = *t;
        hash::hash_uint(task as uint) as uint
    }
    pure fn task_eq(t1: &*rust_task, t2: &*rust_task) -> bool {
        let task1: *rust_task = *t1;
        let task2: *rust_task = *t2;
        task1 == task2
    }

    send_map::linear::linear_map(task_hash, task_eq)
}
fn taskset_insert(tasks: &mut TaskSet, task: *rust_task) {
    let didnt_overwrite = tasks.insert(task, ());
    assert didnt_overwrite;
}
fn taskset_remove(tasks: &mut TaskSet, task: *rust_task) {
    let was_present = tasks.remove(&task);
    assert was_present;
}
fn taskset_each(tasks: &TaskSet, blk: fn(+*rust_task) -> bool) {
    tasks.each_key(blk)
}

// One of these per group of linked-failure tasks.
type TaskGroupData = {
    // All tasks which might kill this group. When this is empty, the group
    // can be "GC"ed (i.e., its link in the ancestor list can be removed).
    mut members:     TaskSet,
    // All tasks unidirectionally supervised by (directly or transitively)
    // tasks in this group.
    mut descendants: TaskSet,
};
type TaskGroupArc = unsafe::Exclusive<option<TaskGroupData>>;

type TaskGroupInner = &mut option<TaskGroupData>;

// A taskgroup is 'dead' when nothing can cause it to fail; only members can.
pure fn taskgroup_is_dead(tg: &TaskGroupData) -> bool {
    (&tg.members).is_empty()
}

// A list-like structure by which taskgroups keep track of all ancestor groups
// which may kill them. Needed for tasks to be able to remove themselves from
// ancestor groups upon exit. The list has a node for each "generation", and
// ends either at the root taskgroup (which has no ancestors) or at a
// taskgroup which was spawned-unlinked. Tasks from intermediate generations
// have references to the middle of the list; when intermediate generations
// die, their node in the list will be collected at a descendant's spawn-time.
type AncestorNode = {
    // Since the ancestor list is recursive, we end up with references to
    // exclusives within other exclusives. This is dangerous business (if
    // circular references arise, deadlock and memory leaks are imminent).
    // Hence we assert that this counter monotonically decreases as we
    // approach the tail of the list.
    // FIXME(#3068): Make the generation counter togglable with #[cfg(debug)].
    generation:       uint,
    // Should really be an immutable non-option. This way appeases borrowck.
    mut parent_group: option<TaskGroupArc>,
    // Recursive rest of the list.
    mut ancestors:    AncestorList,
};
enum AncestorList = option<unsafe::Exclusive<AncestorNode>>;

// Accessors for taskgroup arcs and ancestor arcs that wrap the unsafety.
#[inline(always)]
fn access_group<U>(x: &TaskGroupArc, blk: fn(TaskGroupInner) -> U) -> U {
    unsafe { x.with(blk) }
}

#[inline(always)]
fn access_ancestors<U>(x: &unsafe::Exclusive<AncestorNode>,
                       blk: fn(x: &mut AncestorNode) -> U) -> U {
    unsafe { x.with(blk) }
}

// Iterates over an ancestor list.
// (1) Runs forward_blk on each ancestral taskgroup in the list
// (2) If forward_blk "break"s, runs optional bail_blk on all ancestral
//     taskgroups that forward_blk already ran on successfully (Note: bail_blk
//     is NOT called on the block that forward_blk broke on!).
// (3) As a bonus, coalesces away all 'dead' taskgroup nodes in the list.
// FIXME(#2190): Change option<fn@(...)> to option<fn&(...)>, to save on
// allocations. Once that bug is fixed, changing the sigil should suffice.
fn each_ancestor(list:        &mut AncestorList,
                 bail_opt:    option<fn@(TaskGroupInner)>,
                 forward_blk: fn(TaskGroupInner) -> bool)
        -> bool {
    // "Kickoff" call - there was no last generation.
    return !coalesce(list, bail_opt, forward_blk, uint::max_value);

    // Recursively iterates, and coalesces afterwards if needed. Returns
    // whether or not unwinding is needed (i.e., !successful iteration).
    fn coalesce(list:            &mut AncestorList,
                bail_opt:        option<fn@(TaskGroupInner)>,
                forward_blk:     fn(TaskGroupInner) -> bool,
                last_generation: uint) -> bool {
        // Need to swap the list out to use it, to appease borrowck.
        let tmp_list = util::replace(list, AncestorList(none));
        let (coalesce_this, early_break) =
            iterate(&tmp_list, bail_opt, forward_blk, last_generation);
        // What should our next ancestor end up being?
        if coalesce_this.is_some() {
            // Needed coalesce. Our next ancestor becomes our old
            // ancestor's next ancestor. ("next = old_next->next;")
            *list <- option::unwrap(coalesce_this);
        } else {
            // No coalesce; restore from tmp. ("next = old_next;")
            *list <- tmp_list;
        }
        return early_break;
    }

    // Returns an optional list-to-coalesce and whether unwinding is needed.
    // option<ancestor_list>:
    //     Whether or not the ancestor taskgroup being iterated over is
    //     dead or not; i.e., it has no more tasks left in it, whether or not
    //     it has descendants. If dead, the caller shall coalesce it away.
    // bool:
    //     True if the supplied block did 'break', here or in any recursive
    //     calls. If so, must call the unwinder on all previous nodes.
    fn iterate(ancestors:       &AncestorList,
               bail_opt:        option<fn@(TaskGroupInner)>,
               forward_blk:     fn(TaskGroupInner) -> bool,
               last_generation: uint) -> (option<AncestorList>, bool) {
        // At each step of iteration, three booleans are at play which govern
        // how the iteration should behave.
        // 'nobe_is_dead' - Should the list should be coalesced at this point?
        //                  Largely unrelated to the other two.
        // 'need_unwind'  - Should we run the bail_blk at this point? (i.e.,
        //                  do_continue was false not here, but down the line)
        // 'do_continue'  - Did the forward_blk succeed at this point? (i.e.,
        //                  should we recurse? or should our callers unwind?)

        // The map defaults to none, because if ancestors is none, we're at
        // the end of the list, which doesn't make sense to coalesce.
        return do (**ancestors).map_default((none,false)) |ancestor_arc| {
            // NB: Takes a lock! (this ancestor node)
            do access_ancestors(&ancestor_arc) |nobe| {
                // Check monotonicity
                assert last_generation > nobe.generation;
                /*##########################################################*
                 * Step 1: Look at this ancestor group (call iterator block).
                 *##########################################################*/
                let mut nobe_is_dead = false;
                let do_continue =
                    // NB: Takes a lock! (this ancestor node's parent group)
                    do with_parent_tg(&mut nobe.parent_group) |tg_opt| {
                        // Decide whether this group is dead. Note that the
                        // group being *dead* is disjoint from it *failing*.
                        nobe_is_dead = match *tg_opt {
                            some(ref tg) => taskgroup_is_dead(tg),
                            none => nobe_is_dead
                        };
                        // Call iterator block. (If the group is dead, it's
                        // safe to skip it. This will leave our *rust_task
                        // hanging around in the group even after it's freed,
                        // but that's ok because, by virtue of the group being
                        // dead, nobody will ever kill-all (foreach) over it.)
                        if nobe_is_dead { true } else { forward_blk(tg_opt) }
                    };
                /*##########################################################*
                 * Step 2: Recurse on the rest of the list; maybe coalescing.
                 *##########################################################*/
                // 'need_unwind' is only set if blk returned true above, *and*
                // the recursive call early-broke.
                let mut need_unwind = false;
                if do_continue {
                    // NB: Takes many locks! (ancestor nodes & parent groups)
                    need_unwind = coalesce(&mut nobe.ancestors, bail_opt,
                                           forward_blk, nobe.generation);
                }
                /*##########################################################*
                 * Step 3: Maybe unwind; compute return info for our caller.
                 *##########################################################*/
                if need_unwind && !nobe_is_dead {
                    do bail_opt.iter |bail_blk| {
                        do with_parent_tg(&mut nobe.parent_group) |tg_opt| {
                            bail_blk(tg_opt)
                        }
                    }
                }
                // Decide whether our caller should unwind.
                need_unwind = need_unwind || !do_continue;
                // Tell caller whether or not to coalesce and/or unwind
                if nobe_is_dead {
                    // Swap the list out here; the caller replaces us with it.
                    let rest = util::replace(&mut nobe.ancestors,
                                             AncestorList(none));
                    (some(rest), need_unwind)
                } else {
                    (none, need_unwind)
                }
            }
        };

        // Wrapper around exclusive::with that appeases borrowck.
        fn with_parent_tg<U>(parent_group: &mut option<TaskGroupArc>,
                             blk: fn(TaskGroupInner) -> U) -> U {
            // If this trips, more likely the problem is 'blk' failed inside.
            let tmp_arc = option::swap_unwrap(parent_group);
            let result = do access_group(&tmp_arc) |tg_opt| { blk(tg_opt) };
            *parent_group <- some(tmp_arc);
            result
        }
    }
}

// One of these per task.
struct Tcb {
    let me:            *rust_task;
    // List of tasks with whose fates this one's is intertwined.
    let tasks:         TaskGroupArc; // 'none' means the group has failed.
    // Lists of tasks who will kill us if they fail, but whom we won't kill.
    let mut ancestors: AncestorList;
    let is_main:       bool;
    let notifier:      option<AutoNotify>;
    new(me: *rust_task, -tasks: TaskGroupArc, -ancestors: AncestorList,
        is_main: bool, -notifier: option<AutoNotify>) {
        self.me        = me;
        self.tasks     = tasks;
        self.ancestors = ancestors;
        self.is_main   = is_main;
        self.notifier  = notifier;
        self.notifier.iter(|x| { x.failed = false; });
    }
    // Runs on task exit.
    drop {
        // If we are failing, the whole taskgroup needs to die.
        if rustrt::rust_task_is_unwinding(self.me) {
            self.notifier.iter(|x| { x.failed = true; });
            // Take everybody down with us.
            do access_group(&self.tasks) |tg| {
                kill_taskgroup(tg, self.me, self.is_main);
            }
        } else {
            // Remove ourselves from the group(s).
            do access_group(&self.tasks) |tg| {
                leave_taskgroup(tg, self.me, true);
            }
        }
        // It doesn't matter whether this happens before or after dealing with
        // our own taskgroup, so long as both happen before we die. We need to
        // remove ourself from every ancestor we can, so no cleanup; no break.
        for each_ancestor(&mut self.ancestors, none) |ancestor_group| {
            leave_taskgroup(ancestor_group, self.me, false);
        };
    }
}

struct AutoNotify {
    let notify_chan: comm::Chan<Notification>;
    let mut failed:  bool;
    new(chan: comm::Chan<Notification>) {
        self.notify_chan = chan;
        self.failed = true; // Un-set above when taskgroup successfully made.
    }
    drop {
        let result = if self.failed { Failure } else { Success };
        comm::send(self.notify_chan, Exit(get_task(), result));
    }
}

fn enlist_in_taskgroup(state: TaskGroupInner, me: *rust_task,
                       is_member: bool) -> bool {
    let newstate = util::replace(state, none);
    // If 'none', the group was failing. Can't enlist.
    if newstate.is_some() {
        let group = option::unwrap(newstate);
        taskset_insert(if is_member { &mut group.members }
                       else         { &mut group.descendants }, me);
        *state = some(group);
        true
    } else {
        false
    }
}

// NB: Runs in destructor/post-exit context. Can't 'fail'.
fn leave_taskgroup(state: TaskGroupInner, me: *rust_task, is_member: bool) {
    let newstate = util::replace(state, none);
    // If 'none', already failing and we've already gotten a kill signal.
    if newstate.is_some() {
        let group = option::unwrap(newstate);
        taskset_remove(if is_member { &mut group.members }
                       else         { &mut group.descendants }, me);
        *state = some(group);
    }
}

// NB: Runs in destructor/post-exit context. Can't 'fail'.
fn kill_taskgroup(state: TaskGroupInner, me: *rust_task, is_main: bool) {
    // NB: We could do the killing iteration outside of the group arc, by
    // having "let mut newstate" here, swapping inside, and iterating after.
    // But that would let other exiting tasks fall-through and exit while we
    // were trying to kill them, causing potential use-after-free. A task's
    // presence in the arc guarantees it's alive only while we hold the lock,
    // so if we're failing, all concurrently exiting tasks must wait for us.
    // To do it differently, we'd have to use the runtime's task refcounting,
    // but that could leave task structs around long after their task exited.
    let newstate = util::replace(state, none);
    // Might already be none, if somebody is failing simultaneously.
    // That's ok; only one task needs to do the dirty work. (Might also
    // see 'none' if somebody already failed and we got a kill signal.)
    if newstate.is_some() {
        let group = option::unwrap(newstate);
        for taskset_each(&group.members) |+sibling| {
            // Skip self - killing ourself won't do much good.
            if sibling != me {
                rustrt::rust_task_kill_other(sibling);
            }
        }
        for taskset_each(&group.descendants) |+child| {
            assert child != me;
            rustrt::rust_task_kill_other(child);
        }
        // Only one task should ever do this.
        if is_main {
            rustrt::rust_task_kill_all(me);
        }
        // Do NOT restore state to some(..)! It stays none to indicate
        // that the whole taskgroup is failing, to forbid new spawns.
    }
    // (note: multiple tasks may reach this point)
}

// FIXME (#2912): Work around core-vs-coretest function duplication. Can't use
// a proper closure because the #[test]s won't understand. Have to fake it.
macro_rules! taskgroup_key (
    // Use a "code pointer" value that will never be a real code pointer.
    () => (unsafe::transmute((-2 as uint, 0u)))
)

fn gen_child_taskgroup(linked: bool, supervised: bool)
        -> (TaskGroupArc, AncestorList, bool) {
    let spawner = rustrt::rust_get_task();
    /*######################################################################*
     * Step 1. Get spawner's taskgroup info.
     *######################################################################*/
    let spawner_group = match unsafe { local_get(spawner,
                                                 taskgroup_key!()) } {
        none => {
            // Main task, doing first spawn ever. Lazily initialise here.
            let mut members = new_taskset();
            taskset_insert(&mut members, spawner);
            let tasks =
                unsafe::exclusive(some({ mut members:     members,
                                         mut descendants: new_taskset() }));
            // Main task/group has no ancestors, no notifier, etc.
            let group =
                @Tcb(spawner, tasks, AncestorList(none), true, none);
            unsafe { local_set(spawner, taskgroup_key!(), group); }
            group
        }
        some(group) => group
    };
    /*######################################################################*
     * Step 2. Process spawn options for child.
     *######################################################################*/
    return if linked {
        // Child is in the same group as spawner.
        let g = spawner_group.tasks.clone();
        // Child's ancestors are spawner's ancestors.
        let a = share_ancestors(&mut spawner_group.ancestors);
        // Propagate main-ness.
        (g, a, spawner_group.is_main)
    } else {
        // Child is in a separate group from spawner.
        let g = unsafe::exclusive(some({ mut members:     new_taskset(),
                                         mut descendants: new_taskset() }));
        let a = if supervised {
            // Child's ancestors start with the spawner.
            let old_ancestors = share_ancestors(&mut spawner_group.ancestors);
            // FIXME(#3068) - The generation counter is only used for a debug
            // assertion, but initialising it requires locking a mutex. Hence
            // it should be enabled only in debug builds.
            let new_generation =
                match *old_ancestors {
                    some(arc) => access_ancestors(&arc, |a| a.generation+1),
                    none      => 0 // the actual value doesn't really matter.
                };
            assert new_generation < uint::max_value;
            // Build a new node in the ancestor list.
            AncestorList(some(unsafe::exclusive(
                { generation:       new_generation,
                  mut parent_group: some(spawner_group.tasks.clone()),
                  mut ancestors:    old_ancestors })))
        } else {
            // Child has no ancestors.
            AncestorList(none)
        };
        (g,a, false)
    };

    fn share_ancestors(ancestors: &mut AncestorList) -> AncestorList {
        // Appease the borrow-checker. Really this wants to be written as:
        // match ancestors
        //    some(ancestor_arc) { ancestor_list(some(ancestor_arc.clone())) }
        //    none               { ancestor_list(none) }
        let tmp = util::replace(&mut **ancestors, none);
        if tmp.is_some() {
            let ancestor_arc = option::unwrap(tmp);
            let result = ancestor_arc.clone();
            **ancestors <- some(ancestor_arc);
            AncestorList(some(result))
        } else {
            AncestorList(none)
        }
    }
}

fn spawn_raw(+opts: TaskOpts, +f: fn~()) {
    let (child_tg, ancestors, is_main) =
        gen_child_taskgroup(opts.linked, opts.supervised);

    unsafe {
        let child_data = ~mut some((child_tg, ancestors, f));
        // Being killed with the unsafe task/closure pointers would leak them.
        do unkillable {
            // Agh. Get move-mode items into the closure. FIXME (#2829)
            let (child_tg, ancestors, f) = option::swap_unwrap(child_data);
            // Create child task.
            let new_task = match opts.sched {
              none             => rustrt::new_task(),
              some(sched_opts) => new_task_in_new_sched(sched_opts)
            };
            assert !new_task.is_null();
            // Getting killed after here would leak the task.

            let child_wrapper =
                make_child_wrapper(new_task, child_tg, ancestors, is_main,
                                   opts.notify_chan, f);
            let fptr = ptr::addr_of(child_wrapper);
            let closure: *rust_closure = unsafe::reinterpret_cast(fptr);

            // Getting killed between these two calls would free the child's
            // closure. (Reordering them wouldn't help - then getting killed
            // between them would leak.)
            rustrt::start_task(new_task, closure);
            unsafe::forget(child_wrapper);
        }
    }

    // This function returns a closure-wrapper that we pass to the child task.
    // (1) It sets up the notification channel.
    // (2) It attempts to enlist in the child's group and all ancestor groups.
    // (3a) If any of those fails, it leaves all groups, and does nothing.
    // (3b) Otherwise it builds a task control structure and puts it in TLS,
    // (4) ...and runs the provided body function.
    fn make_child_wrapper(child: *rust_task, +child_arc: TaskGroupArc,
                          +ancestors: AncestorList, is_main: bool,
                          notify_chan: option<comm::Chan<Notification>>,
                          +f: fn~()) -> fn~() {
        let child_data = ~mut some((child_arc, ancestors));
        return fn~() {
            // Agh. Get move-mode items into the closure. FIXME (#2829)
            let mut (child_arc, ancestors) = option::swap_unwrap(child_data);
            // Child task runs this code.

            // Even if the below code fails to kick the child off, we must
            // send something on the notify channel.
            let notifier = notify_chan.map(|c| AutoNotify(c));

            if enlist_many(child, &child_arc, &mut ancestors) {
                let group = @Tcb(child, child_arc, ancestors,
                                 is_main, notifier);
                unsafe { local_set(child, taskgroup_key!(), group); }
                // Run the child's body.
                f();
                // TLS cleanup code will exit the taskgroup.
            }
        };

        // Set up membership in taskgroup and descendantship in all ancestor
        // groups. If any enlistment fails, some task was already failing, so
        // don't let the child task run, and undo every successful enlistment.
        fn enlist_many(child: *rust_task, child_arc: &TaskGroupArc,
                       ancestors: &mut AncestorList) -> bool {
            // Join this taskgroup.
            let mut result =
                do access_group(child_arc) |child_tg| {
                    enlist_in_taskgroup(child_tg, child, true) // member
                };
            if result {
                // Unwinding function in case any ancestral enlisting fails
                let bail = |tg: TaskGroupInner| {
                    leave_taskgroup(tg, child, false)
                };
                // Attempt to join every ancestor group.
                result =
                    for each_ancestor(ancestors, some(bail)) |ancestor_tg| {
                        // Enlist as a descendant, not as an actual member.
                        // Descendants don't kill ancestor groups on failure.
                        if !enlist_in_taskgroup(ancestor_tg, child, false) {
                            break;
                        }
                    };
                // If any ancestor group fails, need to exit this group too.
                if !result {
                    do access_group(child_arc) |child_tg| {
                        leave_taskgroup(child_tg, child, true); // member
                    }
                }
            }
            result
        }
    }

    fn new_task_in_new_sched(opts: SchedOpts) -> *rust_task {
        if opts.foreign_stack_size != none {
            fail ~"foreign_stack_size scheduler option unimplemented";
        }

        let num_threads = match opts.mode {
          SingleThreaded => 1u,
          ThreadPerCore => {
            fail ~"thread_per_core scheduling mode unimplemented"
          }
          ThreadPerTask => {
            fail ~"thread_per_task scheduling mode unimplemented"
          }
          ManualThreads(threads) => {
            if threads == 0u {
                fail ~"can not create a scheduler with no threads";
            }
            threads
          }
          PlatformThread => 0u /* Won't be used */
        };

        let sched_id = if opts.mode != PlatformThread {
            rustrt::rust_new_sched(num_threads)
        } else {
            rustrt::rust_osmain_sched_id()
        };
        rustrt::rust_new_task_in_sched(sched_id)
    }
}

/****************************************************************************
 * Task local data management
 *
 * Allows storing boxes with arbitrary types inside, to be accessed anywhere
 * within a task, keyed by a pointer to a global finaliser function. Useful
 * for task-spawning metadata (tracking linked failure state), dynamic
 * variables, and interfacing with foreign code with bad callback interfaces.
 *
 * To use, declare a monomorphic global function at the type to store, and use
 * it as the 'key' when accessing. See the 'tls' tests below for examples.
 *
 * Casting 'Arcane Sight' reveals an overwhelming aura of Transmutation magic.
 ****************************************************************************/

/**
 * Indexes a task-local data slot. The function's code pointer is used for
 * comparison. Recommended use is to write an empty function for each desired
 * task-local data slot (and use class destructors, not code inside the
 * function, if specific teardown is needed). DO NOT use multiple
 * instantiations of a single polymorphic function to index data of different
 * types; arbitrary type coercion is possible this way.
 *
 * One other exception is that this global state can be used in a destructor
 * context to create a circular @-box reference, which will crash during task
 * failure (see issue #3039).
 *
 * These two cases aside, the interface is safe.
 */
type LocalDataKey<T: owned> = &fn(+@T);

trait LocalData { }
impl<T: owned> @T: LocalData { }

// We use dvec because it's the best data structure in core. If TLS is used
// heavily in future, this could be made more efficient with a proper map.
type TaskLocalElement = (*libc::c_void, *libc::c_void, LocalData);
// Has to be a pointer at outermost layer; the foreign call returns void *.
type TaskLocalMap = @dvec::DVec<option<TaskLocalElement>>;

extern fn cleanup_task_local_map(map_ptr: *libc::c_void) unsafe {
    assert !map_ptr.is_null();
    // Get and keep the single reference that was created at the beginning.
    let _map: TaskLocalMap = unsafe::reinterpret_cast(map_ptr);
    // All local_data will be destroyed along with the map.
}

// Gets the map from the runtime. Lazily initialises if not done so already.
unsafe fn get_task_local_map(task: *rust_task) -> TaskLocalMap {

    // Relies on the runtime initialising the pointer to null.
    // NOTE: The map's box lives in TLS invisibly referenced once. Each time
    // we retrieve it for get/set, we make another reference, which get/set
    // drop when they finish. No "re-storing after modifying" is needed.
    let map_ptr = rustrt::rust_get_task_local_data(task);
    if map_ptr.is_null() {
        let map: TaskLocalMap = @dvec::dvec();
        // Use reinterpret_cast -- transmute would take map away from us also.
        rustrt::rust_set_task_local_data(task, unsafe::reinterpret_cast(map));
        rustrt::rust_task_local_data_atexit(task, cleanup_task_local_map);
        // Also need to reference it an extra time to keep it for now.
        unsafe::bump_box_refcount(map);
        map
    } else {
        let map = unsafe::transmute(map_ptr);
        unsafe::bump_box_refcount(map);
        map
    }
}

unsafe fn key_to_key_value<T: owned>(
    key: LocalDataKey<T>) -> *libc::c_void {

    // Keys are closures, which are (fnptr,envptr) pairs. Use fnptr.
    // Use reintepret_cast -- transmute would leak (forget) the closure.
    let pair: (*libc::c_void, *libc::c_void) = unsafe::reinterpret_cast(key);
    pair.first()
}

// If returning some(..), returns with @T with the map's reference. Careful!
unsafe fn local_data_lookup<T: owned>(
    map: TaskLocalMap, key: LocalDataKey<T>)
    -> option<(uint, *libc::c_void)> {

    let key_value = key_to_key_value(key);
    let map_pos = (*map).position(|entry|
        match entry {
            some((k,_,_)) => k == key_value,
            none => false
        }
    );
    do map_pos.map |index| {
        // .get() is guaranteed because of "none { false }" above.
        let (_, data_ptr, _) = (*map)[index].get();
        (index, data_ptr)
    }
}

unsafe fn local_get_helper<T: owned>(
    task: *rust_task, key: LocalDataKey<T>,
    do_pop: bool) -> option<@T> {

    let map = get_task_local_map(task);
    // Interpreturn our findings from the map
    do local_data_lookup(map, key).map |result| {
        // A reference count magically appears on 'data' out of thin air. It
        // was referenced in the local_data box, though, not here, so before
        // overwriting the local_data_box we need to give an extra reference.
        // We must also give an extra reference when not removing.
        let (index, data_ptr) = result;
        let data: @T = unsafe::transmute(data_ptr);
        unsafe::bump_box_refcount(data);
        if do_pop {
            (*map).set_elt(index, none);
        }
        data
    }
}

unsafe fn local_pop<T: owned>(
    task: *rust_task,
    key: LocalDataKey<T>) -> option<@T> {

    local_get_helper(task, key, true)
}

unsafe fn local_get<T: owned>(
    task: *rust_task,
    key: LocalDataKey<T>) -> option<@T> {

    local_get_helper(task, key, false)
}

unsafe fn local_set<T: owned>(
    task: *rust_task, key: LocalDataKey<T>, +data: @T) {

    let map = get_task_local_map(task);
    // Store key+data as *voids. Data is invisibly referenced once; key isn't.
    let keyval = key_to_key_value(key);
    // We keep the data in two forms: one as an unsafe pointer, so we can get
    // it back by casting; another in an existential box, so the reference we
    // own on it can be dropped when the box is destroyed. The unsafe pointer
    // does not have a reference associated with it, so it may become invalid
    // when the box is destroyed.
    let data_ptr = unsafe::reinterpret_cast(data);
    let data_box = data as LocalData;
    // Construct new entry to store in the map.
    let new_entry = some((keyval, data_ptr, data_box));
    // Find a place to put it.
    match local_data_lookup(map, key) {
        some((index, _old_data_ptr)) => {
            // Key already had a value set, _old_data_ptr, whose reference
            // will get dropped when the local_data box is overwritten.
            (*map).set_elt(index, new_entry);
        }
        none => {
            // Find an empty slot. If not, grow the vector.
            match (*map).position(|x| x == none) {
                some(empty_index) => (*map).set_elt(empty_index, new_entry),
                none => (*map).push(new_entry)
            }
        }
    }
}

unsafe fn local_modify<T: owned>(
    task: *rust_task, key: LocalDataKey<T>,
    modify_fn: fn(option<@T>) -> option<@T>) {

    // Could be more efficient by doing the lookup work, but this is easy.
    let newdata = modify_fn(local_pop(task, key));
    if newdata.is_some() {
        local_set(task, key, option::unwrap(newdata));
    }
}

/* Exported interface for task-local data (plus local_data_key above). */
/**
 * Remove a task-local data value from the table, returning the
 * reference that was originally created to insert it.
 */
unsafe fn local_data_pop<T: owned>(
    key: LocalDataKey<T>) -> option<@T> {

    local_pop(rustrt::rust_get_task(), key)
}
/**
 * Retrieve a task-local data value. It will also be kept alive in the
 * table until explicitly removed.
 */
unsafe fn local_data_get<T: owned>(
    key: LocalDataKey<T>) -> option<@T> {

    local_get(rustrt::rust_get_task(), key)
}
/**
 * Store a value in task-local data. If this key already has a value,
 * that value is overwritten (and its destructor is run).
 */
unsafe fn local_data_set<T: owned>(
    key: LocalDataKey<T>, +data: @T) {

    local_set(rustrt::rust_get_task(), key, data)
}
/**
 * Modify a task-local data value. If the function returns 'none', the
 * data is removed (and its reference dropped).
 */
unsafe fn local_data_modify<T: owned>(
    key: LocalDataKey<T>,
    modify_fn: fn(option<@T>) -> option<@T>) {

    local_modify(rustrt::rust_get_task(), key, modify_fn)
}

extern mod rustrt {
    #[rust_stack]
    fn rust_task_yield(task: *rust_task) -> bool;

    fn rust_get_sched_id() -> sched_id;
    fn rust_new_sched(num_threads: libc::uintptr_t) -> sched_id;

    fn get_task_id() -> task_id;
    #[rust_stack]
    fn rust_get_task() -> *rust_task;

    fn new_task() -> *rust_task;
    fn rust_new_task_in_sched(id: sched_id) -> *rust_task;

    fn start_task(task: *rust_task, closure: *rust_closure);

    fn rust_task_is_unwinding(task: *rust_task) -> bool;
    fn rust_osmain_sched_id() -> sched_id;
    #[rust_stack]
    fn rust_task_inhibit_kill(t: *rust_task);
    #[rust_stack]
    fn rust_task_allow_kill(t: *rust_task);
    #[rust_stack]
    fn rust_task_inhibit_yield(t: *rust_task);
    #[rust_stack]
    fn rust_task_allow_yield(t: *rust_task);
    fn rust_task_kill_other(task: *rust_task);
    fn rust_task_kill_all(task: *rust_task);

    #[rust_stack]
    fn rust_get_task_local_data(task: *rust_task) -> *libc::c_void;
    #[rust_stack]
    fn rust_set_task_local_data(task: *rust_task, map: *libc::c_void);
    #[rust_stack]
    fn rust_task_local_data_atexit(task: *rust_task, cleanup_fn: *u8);
}


#[test]
fn test_spawn_raw_simple() {
    let po = comm::port();
    let ch = comm::chan(po);
    do spawn_raw(default_task_opts()) {
        comm::send(ch, ());
    }
    comm::recv(po);
}

#[test]
#[ignore(cfg(windows))]
fn test_spawn_raw_unsupervise() {
    let opts = {
        linked: false
        with default_task_opts()
    };
    do spawn_raw(opts) {
        fail;
    }
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

// !!! These tests are dangerous. If something is buggy, they will hang, !!!
// !!! instead of exiting cleanly. This might wedge the buildbots.       !!!

#[test] #[ignore(cfg(windows))]
fn test_spawn_unlinked_unsup_no_fail_down() { // grandchild sends on a port
    let po = comm::port();
    let ch = comm::chan(po);
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
    let po = comm::port::<()>();
    let _ch = comm::chan(po);
    // Unidirectional "parenting" shouldn't override bidirectional linked.
    // We have to cheat with opts - the interface doesn't support them because
    // they don't make sense (redundant with task().supervised()).
    let b0 = task();
    let b1 = TaskBuilder({
        opts: { linked: true, supervised: true with b0.opts },
        can_not_copy: none,
        with *b0
    });
    do b1.spawn { fail; }
    comm::recv(po); // We should get punted awake
}
#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_linked_sup_fail_down() { // parent fails; child fails
    // We have to cheat with opts - the interface doesn't support them because
    // they don't make sense (redundant with task().supervised()).
    let b0 = task();
    let b1 = TaskBuilder({
        opts: { linked: true, supervised: true with b0.opts },
        can_not_copy: none,
        with *b0
    });
    do b1.spawn { loop { task::yield(); } }
    fail; // *both* mechanisms would be wrong if this didn't kill the child...
}
#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_spawn_linked_unsup_fail_up() { // child fails; parent fails
    let po = comm::port::<()>();
    let _ch = comm::chan(po);
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
#[ignore(cfg(windows))]
fn test_spawn_raw_notify() {
    let task_po = comm::port();
    let task_ch = comm::chan(task_po);
    let notify_po = comm::port();
    let notify_ch = comm::chan(notify_po);

    let opts = {
        notify_chan: some(notify_ch)
        with default_task_opts()
    };
    do spawn_raw(opts) {
        comm::send(task_ch, get_task());
    }
    let task_ = comm::recv(task_po);
    assert comm::recv(notify_po) == Exit(task_, Success);

    let opts = {
        linked: false,
        notify_chan: some(notify_ch)
        with default_task_opts()
    };
    do spawn_raw(opts) {
        comm::send(task_ch, get_task());
        fail;
    }
    let task_ = comm::recv(task_po);
    assert comm::recv(notify_po) == Exit(task_, Failure);
}

#[test]
fn test_run_basic() {
    let po = comm::port();
    let ch = comm::chan(po);
    do task().spawn {
        comm::send(ch, ());
    }
    comm::recv(po);
}

#[test]
fn test_add_wrapper() {
    let po = comm::port();
    let ch = comm::chan(po);
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
    let mut result = none;
    do task().future_result(|+r| { result = some(r); }).spawn { }
    assert future::get(&option::unwrap(result)) == Success;

    result = none;
    do task().future_result(|+r| { result = some(r); }).unlinked().spawn {
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
    let po = comm::port();
    let ch = comm::chan(po);
    let ch = do spawn_listener |po| {
        // Now the child has a port called 'po' to read from and
        // an environment-captured channel called 'ch'.
        let res = comm::recv(po);
        assert res == ~"ping";
        comm::send(ch, ~"pong");
    };
    // Likewise, the parent has both a 'po' and 'ch'
    comm::send(ch, ~"ping");
    let res = comm::recv(po);
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
        result::ok(~"Success!") => (),
        _ => fail
    }
}

#[test]
#[ignore(cfg(windows))]
fn test_try_fail() {
    match do try {
        fail
    } {
        result::err(()) => (),
        result::ok(()) => fail
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
    let po = comm::port();
    let ch = comm::chan(po);

    fn f(i: int, ch: comm::Chan<()>) {
        let parent_sched_id = rustrt::rust_get_sched_id();

        do spawn_sched(SingleThreaded) {
            let child_sched_id = rustrt::rust_get_sched_id();
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
    let po = comm::port();
    let ch = comm::chan(po);

    do spawn_sched(SingleThreaded) {
        let parent_sched_id = rustrt::rust_get_sched_id();
        do spawn {
            let child_sched_id = rustrt::rust_get_sched_id();
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

        let start_po = comm::port();
        let start_ch = comm::chan(start_po);
        let fin_po = comm::port();
        let fin_ch = comm::chan(fin_po);

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

        let setup_po = comm::port();
        let setup_ch = comm::chan(setup_po);
        let parent_po = comm::port();
        let parent_ch = comm::chan(parent_po);
        do spawn {
            let child_po = comm::port();
            comm::send(setup_ch, comm::chan(child_po));
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
    let p = comm::port::<uint>();
    let ch = comm::chan(p);

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
    let po = comm::port();
    let ch = comm::chan(po);
    do task().sched_mode(PlatformThread).spawn {
        comm::send(ch, ());
    }
    comm::recv(po);
}

#[test]
#[ignore(cfg(windows))]
#[should_fail]
fn test_unkillable() {
    let po = comm::port();
    let ch = po.chan();

    // We want to do this after failing
    do spawn_raw({ linked: false with default_task_opts() }) {
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
            let pp: *uint = unsafe::transmute(p);

            // If we are killed here then the box will leak
            po.recv();

            let _p: ~int = unsafe::transmute(pp);
        }
    }

    // Now we can be killed
    po.recv();
}

#[test]
#[ignore(cfg(windows))]
#[should_fail]
fn test_unkillable_nested() {
    let po = comm::port();
    let ch = po.chan();

    // We want to do this after failing
    do spawn_raw({ linked: false with default_task_opts() }) {
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
            let pp: *uint = unsafe::transmute(p);

            // If we are killed here then the box will leak
            po.recv();

            let _p: ~int = unsafe::transmute(pp);
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
fn test_tls_multitask() unsafe {
    fn my_key(+_x: @~str) { }
    local_data_set(my_key, @~"parent data");
    do task::spawn {
        assert local_data_get(my_key) == none; // TLS shouldn't carry over.
        local_data_set(my_key, @~"child data");
        assert *(local_data_get(my_key).get()) == ~"child data";
        // should be cleaned up for us
    }
    // Must work multiple times
    assert *(local_data_get(my_key).get()) == ~"parent data";
    assert *(local_data_get(my_key).get()) == ~"parent data";
    assert *(local_data_get(my_key).get()) == ~"parent data";
}

#[test]
fn test_tls_overwrite() unsafe {
    fn my_key(+_x: @~str) { }
    local_data_set(my_key, @~"first data");
    local_data_set(my_key, @~"next data"); // Shouldn't leak.
    assert *(local_data_get(my_key).get()) == ~"next data";
}

#[test]
fn test_tls_pop() unsafe {
    fn my_key(+_x: @~str) { }
    local_data_set(my_key, @~"weasel");
    assert *(local_data_pop(my_key).get()) == ~"weasel";
    // Pop must remove the data from the map.
    assert local_data_pop(my_key) == none;
}

#[test]
fn test_tls_modify() unsafe {
    fn my_key(+_x: @~str) { }
    local_data_modify(my_key, |data| {
        match data {
            some(@val) => fail ~"unwelcome value: " + val,
            none       => some(@~"first data")
        }
    });
    local_data_modify(my_key, |data| {
        match data {
            some(@~"first data") => some(@~"next data"),
            some(@val)           => fail ~"wrong value: " + val,
            none                 => fail ~"missing value"
        }
    });
    assert *(local_data_pop(my_key).get()) == ~"next data";
}

#[test]
fn test_tls_crust_automorestack_memorial_bug() unsafe {
    // This might result in a stack-canary clobber if the runtime fails to set
    // sp_limit to 0 when calling the cleanup extern - it might automatically
    // jump over to the rust stack, which causes next_c_sp to get recorded as
    // something within a rust stack segment. Then a subsequent upcall (esp.
    // for logging, think vsnprintf) would run on a stack smaller than 1 MB.
    fn my_key(+_x: @~str) { }
    do task::spawn {
        unsafe { local_data_set(my_key, @~"hax"); }
    }
}

#[test]
fn test_tls_multiple_types() unsafe {
    fn str_key(+_x: @~str) { }
    fn box_key(+_x: @@()) { }
    fn int_key(+_x: @int) { }
    do task::spawn {
        local_data_set(str_key, @~"string data");
        local_data_set(box_key, @@());
        local_data_set(int_key, @42);
    }
}

#[test]
fn test_tls_overwrite_multiple_types() unsafe {
    fn str_key(+_x: @~str) { }
    fn box_key(+_x: @@()) { }
    fn int_key(+_x: @int) { }
    do task::spawn {
        local_data_set(str_key, @~"string data");
        local_data_set(int_key, @42);
        // This could cause a segfault if overwriting-destruction is done with
        // the crazy polymorphic transmute rather than the provided finaliser.
        local_data_set(int_key, @31337);
    }
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn test_tls_cleanup_on_failure() unsafe {
    fn str_key(+_x: @~str) { }
    fn box_key(+_x: @@()) { }
    fn int_key(+_x: @int) { }
    local_data_set(str_key, @~"parent data");
    local_data_set(box_key, @@());
    do task::spawn { // spawn_linked
        local_data_set(str_key, @~"string data");
        local_data_set(box_key, @@());
        local_data_set(int_key, @42);
        fail;
    }
    // Not quite nondeterministic.
    local_data_set(int_key, @31337);
    fail;
}
