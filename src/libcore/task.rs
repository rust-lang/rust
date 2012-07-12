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

export default_task_opts;
export get_opts;
export set_opts;
export set_sched_mode;
export add_wrapper;
export run;

export future_result;
export future_task;
export unsupervise;
export run_listener;
export run_with;

export spawn;
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
enum task = task_id;

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
 * * supervise - Do not propagate failure to the parent task
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
    supervise: bool,
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


/* Task construction */

fn default_task_opts() -> task_opts {
    /*!
     * The default task options
     *
     * By default all tasks are supervised by their parent, are spawned
     * into the same scheduler, and do not post lifecycle notifications.
     */

    {
        supervise: true,
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
     *         supervise: false
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

fn future_task(builder: builder) -> future::future<task> {
    //! Get a future representing the handle to the new task

    import future::future_pipe;

    let (po, ch) = future_pipe::init();

    let ch = ~mut some(ch);

    do add_wrapper(builder) |body, move ch| {
        let ch = { let mut t = none;
                  t <-> *ch;
                  ~mut t};
        fn~(move ch) {
            let mut po = none;
            po <-> *ch;
            future_pipe::server::completed(option::unwrap(po),
                                           get_task());
            body();
        }
    }
    future::from_port(po)
}

fn unsupervise(builder: builder) {
    //! Configures the new task to not propagate failure to its parent

    set_opts(builder, {
        supervise: false
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

    run(builder(), f);
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

    run_with(builder(), arg, f)
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

    run_listener(builder(), f)
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

    let mut builder = builder();
    set_sched_mode(builder, mode);
    run(builder, f);
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
    let mut builder = builder();
    unsupervise(builder);
    let result = future_result(builder);
    do run(builder) {
        comm::send(ch, f());
    }
    alt future::get(result) {
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
        fail "killed";
    }
}

fn failing() -> bool {
    //! True if the running task has failed

    rustrt::rust_task_is_unwinding(rustrt::rust_get_task())
}

fn get_task() -> task {
    //! Get a handle to the running task

    task(rustrt::get_task_id())
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

type taskgroup_arc = arc::exclusive<option<dvec::dvec<option<*rust_task>>>>;

class taskgroup {
    // FIXME (#2816): Change dvec to an O(1) data structure (and change 'me'
    // to a node-handle or somesuch when so done (or remove the field entirely
    // if keyed by *rust_task)).
    let tasks:      taskgroup_arc; // 'none' means the group already failed.
    let me:         *rust_task;
    let my_pos:     uint;
    // let parent_group: taskgroup_arc; // TODO(bblum)
    // TODO XXX bblum: add a list of empty slots to get runtime back
    let mut failed: bool;
    new(-tasks: taskgroup_arc, me: *rust_task, my_pos: uint) {
        self.tasks = tasks; self.me = me; self.my_pos = my_pos;
        self.failed = true; // This will get un-set on successful exit.
    }
    // Runs on task exit.
    drop {
        if self.failed {
            // Take everybody down with us.
            kill_taskgroup(self.tasks, self.me, self.my_pos);
        } else {
            // Remove ourselves from the group.
            leave_taskgroup(self.tasks, self.me, self.my_pos);
        }
    }
}

fn taskgroup_key(+_group: @taskgroup) { } // For TLS

fn enlist_in_taskgroup(group_arc: taskgroup_arc,
                       me: *rust_task) -> option<uint> {
    do group_arc.with |_c, state| {
        // If 'none', the group was failing. Can't enlist.
        do state.map |tasks| {
            // Try to find an empty slot.
            alt tasks.position(|x| x == none) {
                some(empty_index) {
                    tasks.set_elt(empty_index, some(me));
                    empty_index
                }
                none {
                    tasks.push(some(me));
                    tasks.len() - 1
                }
            }
        }
    }
}

// NB: Runs in destructor/post-exit context. Can't 'fail'.
fn leave_taskgroup(group_arc: taskgroup_arc, me: *rust_task, index: uint) {
    do group_arc.with |_c, state| {
        // If 'none', already failing and we've already gotten a kill signal.
        do state.map |tasks| {
            assert tasks[index] == some(me);
            tasks.set_elt(index, none);
        };
    };
}

// NB: Runs in destructor/post-exit context. Can't 'fail'.
fn kill_taskgroup(group_arc: taskgroup_arc, me: *rust_task, index: uint) {
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
        do newstate.map |tasks| {
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
        };
    };
}

fn share_parent_taskgroup() -> taskgroup_arc {
    let me = rustrt::rust_get_task();
    alt unsafe { local_get(me, taskgroup_key) } {
        some(group) {
            group.tasks.clone()
        }
        none {
            /* Main task, doing first spawn ever. */
            let tasks = arc::exclusive(some(dvec::from_elem(some(me))));
            let group = @taskgroup(tasks.clone(), me, 0);
            unsafe { local_set(me, taskgroup_key, group); }
            tasks
        }
    }
}

fn spawn_raw(opts: task_opts, +f: fn~()) {
    // Decide whether the child needs to be in a new linked failure group.
    let child_tg: taskgroup_arc = if opts.supervise {
        share_parent_taskgroup()
    } else {
        arc::exclusive(some(dvec::from_elem(none)))
    };

    unsafe {
        let child_data_ptr = ~mut some((child_tg, f));
        // Being killed with the unsafe task/closure pointers would leak them.
        do unkillable {
            // Agh. Get move-mode items into the closure. FIXME (#2829)
            let mut child_data = none;
            *child_data_ptr <-> child_data;
            let (child_tg, f) = option::unwrap(child_data);
            // Create child task.
            let new_task = alt opts.sched {
              none             { rustrt::new_task() }
              some(sched_opts) { new_task_in_new_sched(sched_opts) }
            };
            assert !new_task.is_null();
            // Getting killed after here would leak the task.

            let child_wrapper =
                make_child_wrapper(new_task, child_tg, opts.supervise, f);
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

    fn make_child_wrapper(child_task: *rust_task, -child_tg: taskgroup_arc,
                          supervise: bool, -f: fn~()) -> fn~() {
        let child_tg_ptr = ~mut some(child_tg);
        fn~() {
            // Agh. Get move-mode items into the closure. FIXME (#2829)
            let mut child_tg_opt = none;
            *child_tg_ptr <-> child_tg_opt;
            let child_tg = option::unwrap(child_tg_opt);
            // Child task runs this code.
            if !supervise {
                // FIXME (#1868, #1789) take this out later
                rustrt::unsupervise();
            }
            // Set up membership in taskgroup. If this returns none, the
            // parent was already failing, so don't bother doing anything.
            alt enlist_in_taskgroup(child_tg, child_task) {
                some(my_index) {
                    let group = @taskgroup(child_tg, child_task, my_index);
                    unsafe { local_set(child_task, taskgroup_key, group); }
                    // Run the child's body.
                    f();
                    // Report successful exit. (TLS cleanup code will tear
                    // down the group.)
                    group.failed = false;
                }
                none { }
            }
        }
    }

    fn new_task_in_new_sched(opts: sched_opts) -> *rust_task {
        if opts.foreign_stack_size != none {
            fail "foreign_stack_size scheduler option unimplemented";
        }

        let num_threads = alt opts.mode {
          single_threaded { 1u }
          thread_per_core {
            fail "thread_per_core scheduling mode unimplemented"
          }
          thread_per_task {
            fail "thread_per_task scheduling mode unimplemented"
          }
          manual_threads(threads) {
            if threads == 0u {
                fail "can not create a scheduler with no threads";
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
 * Indexes a task-local data slot. The function itself is used to
 * automatically finalise stored values; also, its code pointer is used for
 * comparison. Recommended use is to write an empty function for each desired
 * task-local data slot (and use class destructors, instead of code inside the
 * finaliser, if specific teardown is needed). DO NOT use multiple
 * instantiations of a single polymorphic function to index data of different
 * types; arbitrary type coercion is possible this way. The interface is safe
 * as long as all key functions are monomorphic.
 */
type local_data_key<T> = fn@(+@T);

// We use dvec because it's the best data structure in core. If TLS is used
// heavily in future, this could be made more efficient with a proper map.
type task_local_element = (*libc::c_void, *libc::c_void, fn@(+*libc::c_void));
// Has to be a pointer at outermost layer; the foreign call returns void *.
type task_local_map = @dvec::dvec<option<task_local_element>>;

extern fn cleanup_task_local_map(map_ptr: *libc::c_void) unsafe {
    assert !map_ptr.is_null();
    // Get and keep the single reference that was created at the beginning.
    let map: task_local_map = unsafe::reinterpret_cast(map_ptr);
    for (*map).each |entry| {
        alt entry {
            // Finaliser drops data. We drop the finaliser implicitly here.
            some((_key, data, finalise_fn)) { finalise_fn(data); }
            none { }
        }
    }
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

unsafe fn key_to_key_value<T>(key: local_data_key<T>) -> *libc::c_void {
    // Keys are closures, which are (fnptr,envptr) pairs. Use fnptr.
    // Use reintepret_cast -- transmute would leak (forget) the closure.
    let pair: (*libc::c_void, *libc::c_void) = unsafe::reinterpret_cast(key);
    tuple::first(pair)
}

// If returning some(..), returns with @T with the map's reference. Careful!
unsafe fn local_data_lookup<T>(map: task_local_map, key: local_data_key<T>)
        -> option<(uint, *libc::c_void, fn@(+*libc::c_void))> {
    let key_value = key_to_key_value(key);
    let map_pos = (*map).position(|entry|
        alt entry { some((k,_,_)) { k == key_value } none { false } }
    );
    do map_pos.map |index| {
        // .get() is guaranteed because of "none { false }" above.
        let (_, data_ptr, finaliser) = (*map)[index].get();
        (index, data_ptr, finaliser)
    }
}

unsafe fn local_get_helper<T>(task: *rust_task, key: local_data_key<T>,
                              do_pop: bool) -> option<@T> {
    let map = get_task_local_map(task);
    // Interpret our findings from the map
    do local_data_lookup(map, key).map |result| {
        // A reference count magically appears on 'data' out of thin air.
        // 'data' has the reference we originally stored it with. We either
        // need to erase it from the map or artificially bump the count.
        let (index, data_ptr, _) = result;
        let data: @T = unsafe::transmute(data_ptr);
        if do_pop {
            (*map).set_elt(index, none);
        } else {
            unsafe::bump_box_refcount(data);
        }
        data
    }
}

unsafe fn local_pop<T>(task: *rust_task,
                       key: local_data_key<T>) -> option<@T> {
    local_get_helper(task, key, true)
}

unsafe fn local_get<T>(task: *rust_task,
                       key: local_data_key<T>) -> option<@T> {
    local_get_helper(task, key, false)
}

unsafe fn local_set<T>(task: *rust_task, key: local_data_key<T>, +data: @T) {
    let map = get_task_local_map(task);
    // Store key+data as *voids. Data is invisibly referenced once; key isn't.
    let keyval = key_to_key_value(key);
    let data_ptr = unsafe::transmute(data);
    // Finaliser is called at task exit to de-reference up remaining entries.
    let finaliser: fn@(+*libc::c_void) = unsafe::reinterpret_cast(key);
    // Construct new entry to store in the map.
    let new_entry = some((keyval, data_ptr, finaliser));
    // Find a place to put it.
    alt local_data_lookup(map, key) {
        some((index, old_data_ptr, old_finaliser)) {
            // Key already had a value set, old_data_ptr, whose reference we
            // need to drop. After that, overwriting its slot will be safe.
            // (The heap-allocated finaliser will be freed in the overwrite.)
            // FIXME(#2734): just transmuting old_data_ptr to @T doesn't work,
            // similarly to the sample there (but more our/unsafety's fault?).
            old_finaliser(old_data_ptr);
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

unsafe fn local_modify<T>(task: *rust_task, key: local_data_key<T>,
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
unsafe fn local_data_pop<T>(key: local_data_key<T>) -> option<@T> {
    local_pop(rustrt::rust_get_task(), key)
}
/**
 * Retrieve a task-local data value. It will also be kept alive in the
 * table until explicitly removed.
 */
unsafe fn local_data_get<T>(key: local_data_key<T>) -> option<@T> {
    local_get(rustrt::rust_get_task(), key)
}
/**
 * Store a value in task-local data. If this key already has a value,
 * that value is overwritten (and its destructor is run).
 */
unsafe fn local_data_set<T>(key: local_data_key<T>, +data: @T) {
    local_set(rustrt::rust_get_task(), key, data)
}
/**
 * Modify a task-local data value. If the function returns 'none', the
 * data is removed (and its reference dropped).
 */
unsafe fn local_data_modify<T>(key: local_data_key<T>,
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
    fn unsupervise();
    fn rust_osmain_sched_id() -> sched_id;
    fn rust_task_inhibit_kill();
    fn rust_task_allow_kill();
    fn rust_task_kill_other(task: *rust_task);

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
        supervise: false
        with default_task_opts()
    };
    do spawn_raw(opts) {
        fail;
    }
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
        supervise: false,
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
    let buildr = builder();
    do run(buildr) {
        comm::send(ch, ());
    }
    comm::recv(po);
}

#[test]
fn test_add_wrapper() {
    let po = comm::port();
    let ch = comm::chan(po);
    let buildr = builder();
    do add_wrapper(buildr) |body| {
        fn~() {
            body();
            comm::send(ch, ());
        }
    }
    do run(buildr) { }
    comm::recv(po);
}

#[test]
#[ignore(cfg(windows))]
fn test_future_result() {
    let buildr = builder();
    let result = future_result(buildr);
    do run(buildr) { }
    assert future::get(result) == success;

    let buildr = builder();
    let result = future_result(buildr);
    unsupervise(buildr);
    do run(buildr) { fail }
    assert future::get(result) == failure;
}

#[test]
fn test_future_task() {
    let po = comm::port();
    let ch = comm::chan(po);
    let buildr = builder();
    let task1 = future_task(buildr);
    do run(buildr) { comm::send(ch, get_task()) }
    assert future::get(task1) == comm::recv(po);
}

#[test]
fn test_spawn_listiner_bidi() {
    let po = comm::port();
    let ch = comm::chan(po);
    let ch = do spawn_listener |po| {
        // Now the child has a port called 'po' to read from and
        // an environment-captured channel called 'ch'.
        let res = comm::recv(po);
        assert res == "ping";
        comm::send(ch, "pong");
    };
    // Likewise, the parent has both a 'po' and 'ch'
    comm::send(ch, "ping");
    let res = comm::recv(po);
    assert res == "pong";
}

#[test]
fn test_try_success() {
    alt do try {
        "Success!"
    } {
        result::ok("Success!") { }
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
fn test_avoid_copying_the_body_run() {
    do avoid_copying_the_body |f| {
        let buildr = builder();
        do run(buildr) {
            f();
        }
    }
}

#[test]
fn test_avoid_copying_the_body_run_listener() {
    do avoid_copying_the_body |f| {
        let buildr = builder();
        run_listener(buildr, fn~(move f, _po: comm::port<int>) {
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
fn test_avoid_copying_the_body_future_task() {
    do avoid_copying_the_body |f| {
        let buildr = builder();
        future_task(buildr);
        do run(buildr) {
            f();
        }
    }
}

#[test]
fn test_avoid_copying_the_body_unsupervise() {
    do avoid_copying_the_body |f| {
        let buildr = builder();
        unsupervise(buildr);
        do run(buildr) {
            f();
        }
    }
}

#[test]
fn test_osmain() {
    let buildr = builder();
    set_sched_mode(buildr, osmain);

    let po = comm::port();
    let ch = comm::chan(po);
    do run(buildr) {
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
    do spawn_raw({ supervise: false with default_task_opts() }) {
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
    do spawn_raw({ supervise: false with default_task_opts() }) {
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
fn test_tls_multitask() unsafe {
    fn my_key(+_x: @str) { }
    local_data_set(my_key, @"parent data");
    do task::spawn {
        assert local_data_get(my_key) == none; // TLS shouldn't carry over.
        local_data_set(my_key, @"child data");
        assert *(local_data_get(my_key).get()) == "child data";
        // should be cleaned up for us
    }
    // Must work multiple times
    assert *(local_data_get(my_key).get()) == "parent data";
    assert *(local_data_get(my_key).get()) == "parent data";
    assert *(local_data_get(my_key).get()) == "parent data";
}

#[test]
fn test_tls_overwrite() unsafe {
    fn my_key(+_x: @str) { }
    local_data_set(my_key, @"first data");
    local_data_set(my_key, @"next data"); // Shouldn't leak.
    assert *(local_data_get(my_key).get()) == "next data";
}

#[test]
fn test_tls_pop() unsafe {
    fn my_key(+_x: @str) { }
    local_data_set(my_key, @"weasel");
    assert *(local_data_pop(my_key).get()) == "weasel";
    // Pop must remove the data from the map.
    assert local_data_pop(my_key) == none;
}

#[test]
fn test_tls_modify() unsafe {
    fn my_key(+_x: @str) { }
    local_data_modify(my_key, |data| {
        alt data {
            some(@val) { fail "unwelcome value: " + val }
            none       { some(@"first data") }
        }
    });
    local_data_modify(my_key, |data| {
        alt data {
            some(@"first data") { some(@"next data") }
            some(@val)          { fail "wrong value: " + val }
            none                { fail "missing value" }
        }
    });
    assert *(local_data_pop(my_key).get()) == "next data";
}

#[test]
fn test_tls_crust_automorestack_memorial_bug() unsafe {
    // This might result in a stack-canary clobber if the runtime fails to set
    // sp_limit to 0 when calling the cleanup extern - it might automatically
    // jump over to the rust stack, which causes next_c_sp to get recorded as
    // something within a rust stack segment. Then a subsequent upcall (esp.
    // for logging, think vsnprintf) would run on a stack smaller than 1 MB.
    fn my_key(+_x: @str) { }
    do task::spawn {
        unsafe { local_data_set(my_key, @"hax"); }
    }
}

#[test]
fn test_tls_multiple_types() unsafe {
    fn str_key(+_x: @str) { }
    fn box_key(+_x: @@()) { }
    fn int_key(+_x: @int) { }
    do task::spawn {
        local_data_set(str_key, @"string data");
        local_data_set(box_key, @@());
        local_data_set(int_key, @42);
    }
}

#[test]
fn test_tls_overwrite_multiple_types() unsafe {
    fn str_key(+_x: @str) { }
    fn box_key(+_x: @@()) { }
    fn int_key(+_x: @int) { }
    do task::spawn {
        local_data_set(str_key, @"string data");
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
    fn str_key(+_x: @str) { }
    fn box_key(+_x: @@()) { }
    fn int_key(+_x: @int) { }
    local_data_set(str_key, @"parent data");
    local_data_set(box_key, @@());
    do task::spawn { // spawn_linked
        local_data_set(str_key, @"string data");
        local_data_set(box_key, @@());
        local_data_set(int_key, @42);
        fail;
    }
    // Not quite nondeterministic.
    local_data_set(int_key, @31337);
    fail;
}
