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
import dvec::extensions;
import dvec_iter::extensions;
import arc::methods;

export task;
export task_result;
export notification;
export sched_mode;
export sched_opts;
export task_opts;
export builder;
export task_builder;

export default_task_opts;
export get_opts;
export set_opts;
export set_sched_mode;
export add_wrapper;
export run;

export future_result;
export unsupervise;
export run_listener;
export run_with;

export spawn;
export spawn_unlinked;
export spawn_with;
export spawn_listener;
export spawn_sched;
export try;

export yield;
export failing;
export get_task;
export unkillable;

export local_data_key;
export local_data_pop;
export local_data_get;
export local_data_set;
export local_data_modify;

export single_threaded;
export thread_per_core;
export thread_per_task;
export manual_threads;
export osmain;

/* Data types */

/// A handle to a task
enum task { task_handle(task_id) }

/**
 * Indicates the manner in which a task exited.
 *
 * A task that completes without failing and whose supervised children
 * complete without failing is considered to exit successfully.
 *
 * FIXME (See #1868): This description does not indicate the current behavior
 * for linked failure.
 */
enum task_result {
    success,
    failure,
}

/// A message type for notifying of task lifecycle events
enum notification {
    /// Sent when a task exits with the task handle and result
    exit(task, task_result)
}

/// Scheduler modes
enum sched_mode {
    /// All tasks run in the same OS thread
    single_threaded,
    /// Tasks are distributed among available CPUs
    thread_per_core,
    /// Each task runs in its own OS thread
    thread_per_task,
    /// Tasks are distributed among a fixed number of OS threads
    manual_threads(uint),
    /**
     * Tasks are scheduled on the main OS thread
     *
     * The main OS thread is the thread used to launch the runtime which,
     * in most cases, is the process's initial thread as created by the OS.
     */
    osmain
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
type sched_opts = {
    mode: sched_mode,
    foreign_stack_size: option<uint>
};

/**
 * Task configuration options
 *
 * # Fields
 *
 * * linked - Do not propagate failure to the parent task
 *
 *     All tasks are linked together via a tree, from parents to children. By
 *     default children are 'supervised' by their parent and when they fail
 *     so too will their parents. Settings this flag to false disables that
 *     behavior.
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
type task_opts = {
    linked: bool,
    parented: bool,
    notify_chan: option<comm::chan<notification>>,
    sched: option<sched_opts>,
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
enum builder {
  builder_({
        mut opts: task_opts,
        mut gen_body: fn@(+fn~()) -> fn~(),
        can_not_copy: option<comm::port<()>>
    })
}

class dummy { let x: (); new() { self.x = (); } drop { } }

// FIXME (#2585): Replace the 'consumed' bit with move mode on self
enum task_builder = {
    opts: task_opts,
    gen_body: fn@(+fn~()) -> fn~(),
    can_not_copy: option<dummy>,
    mut consumed: bool,
};

/**
 * Generate the base configuration for spawning a task, off of which more
 * configuration methods can be chained.
 * For example, task().unlinked().spawn is equivalent to spawn_unlinked.
 */
fn task() -> task_builder {
    task_builder({
        opts: default_task_opts(),
        gen_body: |body| body, // Identity function
        can_not_copy: none,
        mut consumed: false,
    })
}

impl private_methods for task_builder {
    fn consume() -> task_builder {
        if self.consumed {
            fail ~"Cannot copy a task_builder"; // Fake move mode on self
        }
        self.consumed = true;
        task_builder({ can_not_copy: none, mut consumed: false, with *self })
    }
}

impl task_builder for task_builder {
    /**
     * Decouple the child task's failure from the parent's. If either fails,
     * the other will not be killed.
     */
    fn unlinked() -> task_builder {
        task_builder({
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
    fn supervised() -> task_builder {
        task_builder({
            opts: { linked: false, parented: true with self.opts },
            can_not_copy: none,
            with *self.consume()
        })
    }
    /**
     * Link the child task's and parent task's failures. If either fails, the
     * other will be killed.
     */
    fn linked() -> task_builder {
        task_builder({
            opts: { linked: true, parented: false with self.opts },
            can_not_copy: none,
            with *self.consume()
        })
    }

    /// Configure a future result notification for this task.
    fn future_result(blk: fn(-future::future<task_result>)) -> task_builder {
        // Construct the future and give it to the caller.
        let po = comm::port::<notification>();
        let ch = comm::chan(po);

        blk(do future::from_fn {
            alt comm::recv(po) {
              exit(_, result) { result }
            }
        });

        // Reconfigure self to use a notify channel.
        task_builder({
            opts: { notify_chan: some(ch) with self.opts },
            can_not_copy: none,
            with *self.consume()
        })
    }
    /// Configure a custom scheduler mode for the task.
    fn sched_mode(mode: sched_mode) -> task_builder {
        task_builder({
            opts: { sched: some({ mode: mode, foreign_stack_size: none})
                    with self.opts },
            can_not_copy: none,
            with *self.consume()
        })
    }
    fn add_wrapper(wrapper: fn@(+fn~()) -> fn~()) -> task_builder {
        let prev_gen_body = self.gen_body;
        task_builder({
            gen_body: |body| { wrapper(prev_gen_body(body)) },
            can_not_copy: none,
            with *self.consume()
        })
    }

    /// Run the task.
    fn spawn(+f: fn~()) {
        let x = self.consume();
        spawn_raw(x.opts, x.gen_body(f));
    }
    /// Runs a task, while transfering ownership of one argument to the child.
    fn spawn_with<A: send>(+arg: A, +f: fn~(+A)) {
        let arg = ~mut some(arg);
        do self.spawn {
            let mut my_arg = none;
            my_arg <-> *arg;
            f(option::unwrap(my_arg))
        }
    }
    /// Runs a task with a listening port, returning the associated channel.
    fn spawn_listener<A: send>(+f: fn~(comm::port<A>)) -> comm::chan<A> {
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
}


/* Task construction */

fn default_task_opts() -> task_opts {
    /*!
     * The default task options
     *
     * By default all tasks are supervised by their parent, are spawned
     * into the same scheduler, and do not post lifecycle notifications.
     */

    {
        linked: true,
        parented: false,
        notify_chan: none,
        sched: none
    }
}

fn builder() -> builder {
    //! Construct a builder

    let body_identity = fn@(+body: fn~()) -> fn~() { body };

    builder_({
        mut opts: default_task_opts(),
        mut gen_body: body_identity,
        can_not_copy: none
    })
}

fn get_opts(builder: builder) -> task_opts {
    //! Get the task_opts associated with a builder

    builder.opts
}

fn set_opts(builder: builder, opts: task_opts) {
    /*!
     * Set the task_opts associated with a builder
     *
     * To update a single option use a pattern like the following:
     *
     *     set_opts(builder, {
     *         linked: false
     *         with get_opts(builder)
     *     });
     */

    builder.opts = opts;
}

fn set_sched_mode(builder: builder, mode: sched_mode) {
    set_opts(builder, {
        sched: some({
            mode: mode,
            foreign_stack_size: none
        })
        with get_opts(builder)
    });
}

fn add_wrapper(builder: builder, gen_body: fn@(+fn~()) -> fn~()) {
    /*!
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

    let prev_gen_body = builder.gen_body;
    builder.gen_body = fn@(+body: fn~()) -> fn~() {
        gen_body(prev_gen_body(body))
    };
}

fn run(-builder: builder, +f: fn~()) {
    /*!
     * Creates and exucutes a new child task
     *
     * Sets up a new task with its own call stack and schedules it to run
     * the provided unique closure. The task has the properties and behavior
     * specified by `builder`.
     *
     * # Failure
     *
     * When spawning into a new scheduler, the number of threads requested
     * must be greater than zero.
     */

    let body = builder.gen_body(f);
    spawn_raw(builder.opts, body);
}


/* Builder convenience functions */

fn future_result(builder: builder) -> future::future<task_result> {
    /*!
     * Get a future representing the exit status of the task.
     *
     * Taking the value of the future will block until the child task
     * terminates.
     *
     * Note that the future returning by this function is only useful for
     * obtaining the value of the next task to be spawning with the
     * builder. If additional tasks are spawned with the same builder
     * then a new result future must be obtained prior to spawning each
     * task.
     */

    // FIXME (#1087, #1857): Once linked failure and notification are
    // handled in the library, I can imagine implementing this by just
    // registering an arbitrary number of task::on_exit handlers and
    // sending out messages.

    let po = comm::port();
    let ch = comm::chan(po);

    set_opts(builder, {
        notify_chan: some(ch)
        with get_opts(builder)
    });

    do future::from_fn {
        alt comm::recv(po) {
          exit(_, result) { result }
        }
    }
}

fn unsupervise(builder: builder) {
    //! Configures the new task to not propagate failure to its parent

    set_opts(builder, {
        linked: false
        with get_opts(builder)
    });
}

fn run_with<A:send>(-builder: builder,
                    +arg: A,
                    +f: fn~(+A)) {

    /*!
     * Runs a task, while transfering ownership of one argument to the
     * child.
     *
     * This is useful for transfering ownership of noncopyables to
     * another task.
     *
     */

    let arg = ~mut some(arg);
    do run(builder) {
        let mut my_arg = none;
        my_arg <-> *arg;
        f(option::unwrap(my_arg))
    }
}

fn run_listener<A:send>(-builder: builder,
                        +f: fn~(comm::port<A>)) -> comm::chan<A> {
    /*!
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

    let setup_po = comm::port();
    let setup_ch = comm::chan(setup_po);

    do run(builder) {
        let po = comm::port();
        let mut ch = comm::chan(po);
        comm::send(setup_ch, ch);
        f(po);
    }

    comm::recv(setup_po)
}


/* Spawn convenience functions */

fn spawn(+f: fn~()) {
    /*!
     * Creates and executes a new child task
     *
     * Sets up a new task with its own call stack and schedules it to run
     * the provided unique closure.
     *
     * This function is equivalent to `run(new_builder(), f)`.
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
     * This function is equivalent to `run_with(builder(), arg, f)`.
     */

    task().spawn_with(arg, f)
}

fn spawn_listener<A:send>(+f: fn~(comm::port<A>)) -> comm::chan<A> {
    /*!
     * Runs a new task while providing a channel from the parent to the child
     *
     * Sets up a communication channel from the current task to the new
     * child task, passes the port to child's body, and returns a channel
     * linked to the port to the parent.
     *
     * This encapsulates some boilerplate handshaking logic that would
     * otherwise be required to establish communication from the parent
     * to the child.
     *
     * The simplest way to establish bidirectional communication between
     * a parent in child is as follows:
     *
     *     let po = comm::port();
     *     let ch = comm::chan(po);
     *     let ch = do spawn_listener |po| {
     *         // Now the child has a port called 'po' to read from and
     *         // an environment-captured channel called 'ch'.
     *     };
     *     // Likewise, the parent has both a 'po' and 'ch'
     *
     * This function is equivalent to `run_listener(builder(), f)`.
     */

    task().spawn_listener(f)
}

fn spawn_sched(mode: sched_mode, +f: fn~()) {
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
     * # Return value
     *
     * If the function executed successfully then try returns result::ok
     * containing the value returned by the function. If the function fails
     * then try returns result::err containing nil.
     */

    let po = comm::port();
    let ch = comm::chan(po);

    let mut result = none;

    do task().unlinked().future_result(|-r| { result = some(r); }).spawn {
        comm::send(ch, f());
    }
    alt future::get(option::unwrap(result)) {
      success { result::ok(comm::recv(po)) }
      failure { result::err(()) }
    }
}


/* Lifecycle functions */

fn yield() {
    //! Yield control to the task scheduler

    let task_ = rustrt::rust_get_task();
    let mut killed = false;
    rustrt::rust_task_yield(task_, killed);
    if killed && !failing() {
        fail ~"killed";
    }
}

fn failing() -> bool {
    //! True if the running task has failed

    rustrt::rust_task_is_unwinding(rustrt::rust_get_task())
}

fn get_task() -> task {
    //! Get a handle to the running task

    task_handle(rustrt::get_task_id())
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
unsafe fn unkillable(f: fn()) {
    class allow_failure {
      let i: (); // since a class must have at least one field
      new(_i: ()) { self.i = (); }
      drop { rustrt::rust_task_allow_kill(); }
    }

    let _allow_failure = allow_failure(());
    rustrt::rust_task_inhibit_kill();
    f();
}


/****************************************************************************
 * Internal
 ****************************************************************************/

/* spawning */

type sched_id = int;
type task_id = int;

// These are both opaque runtime/compiler types that we don't know the
// structure of and should only deal with via unsafe pointer
type rust_task = libc::c_void;
type rust_closure = libc::c_void;

/* linked failure */

type taskgroup_arc =
    arc::exclusive<option<(dvec::dvec<option<*rust_task>>,dvec::dvec<uint>)>>;

class taskgroup {
    // FIXME (#2816): Change dvec to an O(1) data structure (and change 'me'
    // to a node-handle or somesuch when so done (or remove the field entirely
    // if keyed by *rust_task)).
    let me:         *rust_task;
    // List of tasks with whose fates this one's is intertwined.
    let tasks:      taskgroup_arc; // 'none' means the group already failed.
    let my_pos:     uint;          // Index into above for this task's slot.
    // Lists of tasks who will kill us if they fail, but whom we won't kill.
    let parents:    option<(taskgroup_arc,uint)>;
    let is_main:    bool;
    new(me: *rust_task, -tasks: taskgroup_arc, my_pos: uint,
        -parents: option<(taskgroup_arc,uint)>, is_main: bool) {
        self.me      = me;
        self.tasks   = tasks;
        self.my_pos  = my_pos;
        self.parents = parents;
        self.is_main = is_main;
    }
    // Runs on task exit.
    drop {
        // If we are failing, the whole taskgroup needs to die.
        if rustrt::rust_task_is_unwinding(self.me) {
            // Take everybody down with us.
            kill_taskgroup(self.tasks, self.me, self.my_pos, self.is_main);
        } else {
            // Remove ourselves from the group(s).
            leave_taskgroup(self.tasks, self.me, self.my_pos);
        }
        // It doesn't matter whether this happens before or after dealing with
        // our own taskgroup, so long as both happen before we die.
        alt self.parents {
            some((parent_group,pos_in_group)) {
                leave_taskgroup(parent_group, self.me, pos_in_group);
            }
            none { }
        }
    }
}

fn enlist_in_taskgroup(group_arc: taskgroup_arc,
                       me: *rust_task) -> option<uint> {
    do group_arc.with |_c, state| {
        // If 'none', the group was failing. Can't enlist.
        let mut newstate = none;
        *state <-> newstate;
        if newstate.is_some() {
            let (tasks,empty_slots) = option::unwrap(newstate);
            // Try to find an empty slot.
            let slotno = if empty_slots.len() > 0 {
                let empty_index = empty_slots.pop();
                assert tasks[empty_index] == none;
                tasks.set_elt(empty_index, some(me));
                empty_index
            } else {
                tasks.push(some(me));
                tasks.len() - 1
            };
            *state = some((tasks,empty_slots));
            some(slotno)
        } else {
            none
        }
    }
}

// NB: Runs in destructor/post-exit context. Can't 'fail'.
fn leave_taskgroup(group_arc: taskgroup_arc, me: *rust_task, index: uint) {
    do group_arc.with |_c, state| {
        let mut newstate = none;
        *state <-> newstate;
        // If 'none', already failing and we've already gotten a kill signal.
        if newstate.is_some() {
            let (tasks,empty_slots) = option::unwrap(newstate);
            assert tasks[index] == some(me);
            tasks.set_elt(index, none);
            empty_slots.push(index);
            *state = some((tasks,empty_slots));
        };
    };
}

// NB: Runs in destructor/post-exit context. Can't 'fail'.
fn kill_taskgroup(group_arc: taskgroup_arc, me: *rust_task, index: uint,
                  is_main: bool) {
    // NB: We could do the killing iteration outside of the group arc, by
    // having "let mut newstate" here, swapping inside, and iterating after.
    // But that would let other exiting tasks fall-through and exit while we
    // were trying to kill them, causing potential use-after-free. A task's
    // presence in the arc guarantees it's alive only while we hold the lock,
    // so if we're failing, all concurrently exiting tasks must wait for us.
    // To do it differently, we'd have to use the runtime's task refcounting.
    do group_arc.with |_c, state| {
        let mut newstate = none;
        *state <-> newstate;
        // Might already be none, if somebody is failing simultaneously.
        // That's ok; only one task needs to do the dirty work. (Might also
        // see 'none' if somebody already failed and we got a kill signal.)
        if newstate.is_some() {
            let (tasks,_empty_slots) = option::unwrap(newstate);
            // First remove ourself (killing ourself won't do much good). This
            // is duplicated here to avoid having to lock twice.
            assert tasks[index] == some(me);
            tasks.set_elt(index, none);
            // Now send takedown signal.
            for tasks.each |entry| {
                do entry.map |task| {
                    rustrt::rust_task_kill_other(task);
                };
            }
            // Only one task should ever do this.
            if is_main {
                rustrt::rust_task_kill_all(me);
            }
            // Do NOT restore state to some(..)! It stays none to indicate
            // that the whole taskgroup is failing, to forbid new spawns.
        }
        // (note: multiple tasks may reach this point)
    };
}

// FIXME (#2912): Work around core-vs-coretest function duplication. Can't use
// a proper closure because the #[test]s won't understand. Have to fake it.
unsafe fn taskgroup_key() -> local_data_key<taskgroup> {
    // Use a "code pointer" value that will never be a real code pointer.
    unsafe::transmute((-2 as uint, 0u))
}

fn share_parent_taskgroup() -> (taskgroup_arc, bool) {
    let me = rustrt::rust_get_task();
    alt unsafe { local_get(me, taskgroup_key()) } {
        some(group) {
            // Clone the shared state for the child; propagate main-ness.
            (group.tasks.clone(), group.is_main)
        }
        none {
            // Main task, doing first spawn ever.
            let tasks = arc::exclusive(some((dvec::from_elem(some(me)),
                                             dvec::dvec())));
            // Main group has no parent group.
            let group = @taskgroup(me, tasks.clone(), 0, none, true);
            unsafe { local_set(me, taskgroup_key(), group); }
            // Tell child task it's also in the main group.
            (tasks, true)
        }
    }
}

fn spawn_raw(opts: task_opts, +f: fn~()) {
    // Decide whether the child needs to be in a new linked failure group.
    let ((child_tg, is_main), parent_tg) = if opts.linked {
        // It doesn't mean anything for a linked-spawned-task to have a parent
        // group. The spawning task is already bidirectionally linked to it.
        (share_parent_taskgroup(), none)
    } else {
        // Detached from the parent group; create a new (non-main) one.
        ((arc::exclusive(some((dvec::dvec(),dvec::dvec()))), false),
         // Allow the parent to unidirectionally fail the child?
         if opts.parented {
             let (pg,_) = share_parent_taskgroup(); some(pg)
         } else {
             none
         })
    };

    unsafe {
        let child_data_ptr = ~mut some((child_tg, parent_tg, f));
        // Being killed with the unsafe task/closure pointers would leak them.
        do unkillable {
            // Agh. Get move-mode items into the closure. FIXME (#2829)
            let mut child_data = none;
            *child_data_ptr <-> child_data;
            let (child_tg, parent_tg, f) = option::unwrap(child_data);
            // Create child task.
            let new_task = alt opts.sched {
              none             { rustrt::new_task() }
              some(sched_opts) { new_task_in_new_sched(sched_opts) }
            };
            assert !new_task.is_null();
            // Getting killed after here would leak the task.

            let child_wrapper =
                make_child_wrapper(new_task, child_tg, parent_tg, is_main, f);
            let fptr = ptr::addr_of(child_wrapper);
            let closure: *rust_closure = unsafe::reinterpret_cast(fptr);

            do option::iter(opts.notify_chan) |c| {
                // FIXME (#1087): Would like to do notification in Rust
                rustrt::rust_task_config_notify(new_task, c);
            }

            // Getting killed between these two calls would free the child's
            // closure. (Reordering them wouldn't help - then getting killed
            // between them would leak.)
            rustrt::start_task(new_task, closure);
            unsafe::forget(child_wrapper);
        }
    }

    // This function returns a closure-wrapper that we pass to the child task.
    // In brief, it does the following:
    //     if enlist_in_group(child_group) {
    //         if parent_group {
    //             if !enlist_in_group(parent_group) {
    //                 leave_group(child_group); // Roll back
    //                 ret; // Parent group failed. Don't run child's f().
    //             }
    //         }
    //         stash_taskgroup_data_in_TLS(child_group, parent_group);
    //         f();
    //     } else {
    //         // My group failed. Don't run chid's f().
    //     }
    fn make_child_wrapper(child: *rust_task, -child_tg: taskgroup_arc,
                          -parent_tg: option<taskgroup_arc>, is_main: bool,
                          -f: fn~()) -> fn~() {
        let child_tg_ptr = ~mut some((child_tg, parent_tg));
        fn~() {
            // Agh. Get move-mode items into the closure. FIXME (#2829)
            let mut tg_data_opt = none;
            *child_tg_ptr <-> tg_data_opt;
            let (child_tg, parent_tg) = option::unwrap(tg_data_opt);
            // Child task runs this code.
            // Set up membership in taskgroup. If this returns none, some
            // task was already failing, so don't bother doing anything.
            alt enlist_in_taskgroup(child_tg, child) {
                some(my_pos) {
                    // Enlist in parent group too. If enlist returns none, a
                    // parent was failing: don't spawn; leave this group too.
                    let (pg, enlist_ok) = if parent_tg.is_some() {
                        let parent_group = option::unwrap(parent_tg);
                        alt enlist_in_taskgroup(parent_group, child) {
                            some(my_p_index) {
                                // Successful enlist.
                                (some((parent_group, my_p_index)), true)
                            }
                            none {
                                // Couldn't enlist. Have to quit here too.
                                leave_taskgroup(child_tg, child, my_pos);
                                (none, false)
                            }
                        }
                    } else {
                        // No parent group to enlist in. No worry.
                        (none, true)
                    };
                    if enlist_ok {
                        let group = @taskgroup(child, child_tg, my_pos,
                                               pg, is_main);
                        unsafe { local_set(child, taskgroup_key(), group); }
                        // Run the child's body.
                        f();
                        // TLS cleanup code will exit the taskgroup.
                    }
                }
                none { }
            }
        }
    }

    fn new_task_in_new_sched(opts: sched_opts) -> *rust_task {
        if opts.foreign_stack_size != none {
            fail ~"foreign_stack_size scheduler option unimplemented";
        }

        let num_threads = alt opts.mode {
          single_threaded { 1u }
          thread_per_core {
            fail ~"thread_per_core scheduling mode unimplemented"
          }
          thread_per_task {
            fail ~"thread_per_task scheduling mode unimplemented"
          }
          manual_threads(threads) {
            if threads == 0u {
                fail ~"can not create a scheduler with no threads";
            }
            threads
          }
          osmain { 0u /* Won't be used */ }
        };

        let sched_id = if opts.mode != osmain {
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
 * types; arbitrary type coercion is possible this way. The interface is safe
 * as long as all key functions are monomorphic.
 */
type local_data_key<T: owned> = fn@(+@T);

iface local_data { }
impl<T: owned> of local_data for @T { }

// We use dvec because it's the best data structure in core. If TLS is used
// heavily in future, this could be made more efficient with a proper map.
type task_local_element = (*libc::c_void, *libc::c_void, local_data);
// Has to be a pointer at outermost layer; the foreign call returns void *.
type task_local_map = @dvec::dvec<option<task_local_element>>;

extern fn cleanup_task_local_map(map_ptr: *libc::c_void) unsafe {
    assert !map_ptr.is_null();
    // Get and keep the single reference that was created at the beginning.
    let _map: task_local_map = unsafe::reinterpret_cast(map_ptr);
    // All local_data will be destroyed along with the map.
}

// Gets the map from the runtime. Lazily initialises if not done so already.
unsafe fn get_task_local_map(task: *rust_task) -> task_local_map {

    // Relies on the runtime initialising the pointer to null.
    // NOTE: The map's box lives in TLS invisibly referenced once. Each time
    // we retrieve it for get/set, we make another reference, which get/set
    // drop when they finish. No "re-storing after modifying" is needed.
    let map_ptr = rustrt::rust_get_task_local_data(task);
    if map_ptr.is_null() {
        let map: task_local_map = @dvec::dvec();
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
    key: local_data_key<T>) -> *libc::c_void {

    // Keys are closures, which are (fnptr,envptr) pairs. Use fnptr.
    // Use reintepret_cast -- transmute would leak (forget) the closure.
    let pair: (*libc::c_void, *libc::c_void) = unsafe::reinterpret_cast(key);
    pair.first()
}

// If returning some(..), returns with @T with the map's reference. Careful!
unsafe fn local_data_lookup<T: owned>(
    map: task_local_map, key: local_data_key<T>)
    -> option<(uint, *libc::c_void)> {

    let key_value = key_to_key_value(key);
    let map_pos = (*map).position(|entry|
        alt entry { some((k,_,_)) { k == key_value } none { false } }
    );
    do map_pos.map |index| {
        // .get() is guaranteed because of "none { false }" above.
        let (_, data_ptr, _) = (*map)[index].get();
        (index, data_ptr)
    }
}

unsafe fn local_get_helper<T: owned>(
    task: *rust_task, key: local_data_key<T>,
    do_pop: bool) -> option<@T> {

    let map = get_task_local_map(task);
    // Interpret our findings from the map
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
    key: local_data_key<T>) -> option<@T> {

    local_get_helper(task, key, true)
}

unsafe fn local_get<T: owned>(
    task: *rust_task,
    key: local_data_key<T>) -> option<@T> {

    local_get_helper(task, key, false)
}

unsafe fn local_set<T: owned>(
    task: *rust_task, key: local_data_key<T>, +data: @T) {

    let map = get_task_local_map(task);
    // Store key+data as *voids. Data is invisibly referenced once; key isn't.
    let keyval = key_to_key_value(key);
    // We keep the data in two forms: one as an unsafe pointer, so we can get
    // it back by casting; another in an existential box, so the reference we
    // own on it can be dropped when the box is destroyed. The unsafe pointer
    // does not have a reference associated with it, so it may become invalid
    // when the box is destroyed.
    let data_ptr = unsafe::reinterpret_cast(data);
    let data_box = data as local_data;
    // Construct new entry to store in the map.
    let new_entry = some((keyval, data_ptr, data_box));
    // Find a place to put it.
    alt local_data_lookup(map, key) {
        some((index, _old_data_ptr)) {
            // Key already had a value set, _old_data_ptr, whose reference
            // will get dropped when the local_data box is overwritten.
            (*map).set_elt(index, new_entry);
        }
        none {
            // Find an empty slot. If not, grow the vector.
            alt (*map).position(|x| x == none) {
                some(empty_index) {
                    (*map).set_elt(empty_index, new_entry);
                }
                none {
                    (*map).push(new_entry);
                }
            }
        }
    }
}

unsafe fn local_modify<T: owned>(
    task: *rust_task, key: local_data_key<T>,
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
    key: local_data_key<T>) -> option<@T> {

    local_pop(rustrt::rust_get_task(), key)
}
/**
 * Retrieve a task-local data value. It will also be kept alive in the
 * table until explicitly removed.
 */
unsafe fn local_data_get<T: owned>(
    key: local_data_key<T>) -> option<@T> {

    local_get(rustrt::rust_get_task(), key)
}
/**
 * Store a value in task-local data. If this key already has a value,
 * that value is overwritten (and its destructor is run).
 */
unsafe fn local_data_set<T: owned>(
    key: local_data_key<T>, +data: @T) {

    local_set(rustrt::rust_get_task(), key, data)
}
/**
 * Modify a task-local data value. If the function returns 'none', the
 * data is removed (and its reference dropped).
 */
unsafe fn local_data_modify<T: owned>(
    key: local_data_key<T>,
    modify_fn: fn(option<@T>) -> option<@T>) {

    local_modify(rustrt::rust_get_task(), key, modify_fn)
}

extern mod rustrt {
    #[rust_stack]
    fn rust_task_yield(task: *rust_task, &killed: bool);

    fn rust_get_sched_id() -> sched_id;
    fn rust_new_sched(num_threads: libc::uintptr_t) -> sched_id;

    fn get_task_id() -> task_id;
    #[rust_stack]
    fn rust_get_task() -> *rust_task;

    fn new_task() -> *rust_task;
    fn rust_new_task_in_sched(id: sched_id) -> *rust_task;

    fn rust_task_config_notify(
        task: *rust_task, &&chan: comm::chan<notification>);

    fn start_task(task: *rust_task, closure: *rust_closure);

    fn rust_task_is_unwinding(task: *rust_task) -> bool;
    fn rust_osmain_sched_id() -> sched_id;
    fn rust_task_inhibit_kill();
    fn rust_task_allow_kill();
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
            for iter::repeat(8192) { task::yield(); }
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
    for iter::repeat(8192) { task::yield(); }
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
    let b1 = task_builder({
        opts: { linked: true, parented: true with b0.opts },
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
    let b1 = task_builder({
        opts: { linked: true, parented: true with b0.opts },
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

// A bonus linked failure test

#[test] #[should_fail] // #[ignore(cfg(windows))]
#[ignore] // FIXME (#1868) (bblum) make this work
fn test_spawn_unlinked_sup_propagate_grandchild() {
    do spawn_supervised {
        do spawn_supervised {
            loop { task::yield(); }
        }
    }
    for iter::repeat(8192) { task::yield(); }
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
    assert comm::recv(notify_po) == exit(task_, success);

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
    assert comm::recv(notify_po) == exit(task_, failure);
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
    do task().future_result(|-r| { result = some(r); }).spawn { }
    assert future::get(option::unwrap(result)) == success;

    result = none;
    do task().future_result(|-r| { result = some(r); }).unlinked().spawn {
        fail;
    }
    assert future::get(option::unwrap(result)) == failure;
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
fn test_try_success() {
    alt do try {
        ~"Success!"
    } {
        result::ok(~"Success!") { }
        _ { fail; }
    }
}

#[test]
#[ignore(cfg(windows))]
fn test_try_fail() {
    alt do try {
        fail
    } {
        result::err(()) { }
        result::ok(()) { fail; }
    }
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn test_spawn_sched_no_threads() {
    do spawn_sched(manual_threads(0u)) { }
}

#[test]
fn test_spawn_sched() {
    let po = comm::port();
    let ch = comm::chan(po);

    fn f(i: int, ch: comm::chan<()>) {
        let parent_sched_id = rustrt::rust_get_sched_id();

        do spawn_sched(single_threaded) {
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

    do spawn_sched(single_threaded) {
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

        do spawn_sched(single_threaded) {
            testrt::rust_dbg_lock_lock(lock);

            comm::send(start_ch, ());

            // Block the scheduler thread
            testrt::rust_dbg_lock_wait(lock);
            testrt::rust_dbg_lock_unlock(lock);

            comm::send(fin_ch, ());
        };

        // Wait until the other task has its lock
        comm::recv(start_po);

        fn pingpong(po: comm::port<int>, ch: comm::chan<int>) {
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
        spawn_listener(fn~(move f, _po: comm::port<int>) {
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
fn test_avoid_copying_the_body_spawn_listener() {
    do avoid_copying_the_body |f| {
        task().spawn_listener(fn~(move f, _po: comm::port<int>) {
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
fn test_osmain() {
    let po = comm::port();
    let ch = comm::chan(po);
    do task().sched_mode(osmain).spawn {
        comm::send(ch, ());
    }
    comm::recv(po);
}

#[test]
#[ignore(cfg(windows))]
#[should_fail]
fn test_unkillable() {
    import comm::methods;
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
    import comm::methods;
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

#[test]
fn test_child_doesnt_ref_parent() {
    // If the child refcounts the parent task, this will stack overflow when
    // climbing the task tree to dereference each ancestor. (See #1789)
    const generations: uint = 8192;
    fn child_no(x: uint) -> fn~() {
        ret || {
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
        alt data {
            some(@val) { fail ~"unwelcome value: " + val }
            none       { some(@~"first data") }
        }
    });
    local_data_modify(my_key, |data| {
        alt data {
            some(@~"first data") { some(@~"next data") }
            some(@val)          { fail ~"wrong value: " + val }
            none                { fail ~"missing value" }
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
