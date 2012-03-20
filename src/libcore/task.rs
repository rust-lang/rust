#[doc = "
Task management.

An executing Rust program consists of a tree of tasks, each with their own
stack, and sole ownership of their allocated heap data. Tasks communicate
with each other using ports and channels.

When a task fails, that failure will propagate to its parent (the task
that spawned it) and the parent will fail as well. The reverse is not
true: when a parent task fails its children will continue executing. When
the root (main) task fails, all tasks fail, and then so does the entire
process.

Tasks may execute in parallel and are scheduled automatically by the runtime.

# Example

~~~
spawn {||
    log(error, \"Hello, World!\");
}
~~~
"];

import result::result;

export task;
export task_result;
export notification;
export sched_mode;
export sched_opts;
export task_opts;
export task_builder::{};

export default_task_opts;
export get_opts;
export set_opts;
export add_wrapper;
export run;

export future_result;
export future_task;
export unsupervise;
export run_listener;

export spawn;
export spawn_listener;
export spawn_sched;
export try;

export yield;
export failing;
export get_task;


/* Data types */

#[doc = "A handle to a task"]
enum task = task_id;

#[doc = "
Indicates the manner in which a task exited.

A task that completes without failing and whose supervised children complete
without failing is considered to exit successfully.

FIXME: This description does not indicate the current behavior for linked
failure.
"]
enum task_result {
    success,
    failure,
}

#[doc = "A message type for notifying of task lifecycle events"]
enum notification {
    #[doc = "Sent when a task exits with the task handle and result"]
    exit(task, task_result)
}

#[doc = "Scheduler modes"]
enum sched_mode {
    #[doc = "All tasks run in the same OS thread"]
    single_threaded,
    #[doc = "Tasks are distributed among available CPUs"]
    thread_per_core,
    #[doc = "Each task runs in its own OS thread"]
    thread_per_task,
    #[doc = "Tasks are distributed among a fixed number of OS threads"]
    manual_threads(uint),
}

#[doc = "
Scheduler configuration options

# Fields

* sched_mode - The operating mode of the scheduler

* native_stack_size - The size of the native stack, in bytes

    Rust code runs on Rust-specific stacks. When Rust code calls native code
    (via functions in native modules) it switches to a typical, large stack
    appropriate for running code written in languages like C. By default these
    native stacks have unspecified size, but with this option their size can
    be precisely specified.
"]
type sched_opts = {
    mode: sched_mode,
    native_stack_size: option<uint>,
};

#[doc = "
Task configuration options

# Fields

* supervise - Do not propagate failure to the parent task

    All tasks are linked together via a tree, from parents to children. By
    default children are 'supervised' by their parent and when they fail
    so too will their parents. Settings this flag to false disables that
    behavior.

* notify_chan - Enable lifecycle notifications on the given channel

* sched - Specify the configuration of a new scheduler to create the task in

    By default, every task is created in the same scheduler as its
    parent, where it is scheduled cooperatively with all other tasks
    in that scheduler. Some specialized applications may want more
    control over their scheduling, in which case they can be spawned
    into a new scheduler with the specific properties required.

    This is of particular importance for libraries which want to call
    into native code that blocks. Without doing so in a different
    scheduler other tasks will be impeded or even blocked indefinitely.

"]
type task_opts = {
    supervise: bool,
    notify_chan: option<comm::chan<notification>>,
    sched: option<sched_opts>,
};

#[doc = "
The task builder type.

Provides detailed control over the properties and behavior of new tasks.
"]
// NB: Builders are designed to be single-use because they do stateful
// things that get weird when reusing - e.g. if you create a result future
// it only applies to a single task, so then you have to maintain some
// potentially tricky state to ensure that everything behaves correctly
// when you try to reuse the builder to spawn a new task. We'll just
// sidestep that whole issue by making builder's uncopyable and making
// the run function move them in.
enum task_builder {
    task_builder_({
        mutable opts: task_opts,
        mutable gen_body: fn@(+fn~()) -> fn~(),
        can_not_copy: option<comm::port<()>>
    })
}


/* Task construction */

fn default_task_opts() -> task_opts {
    #[doc = "
    The default task options

    By default all tasks are supervised by their parent, are spawned
    into the same scheduler, and do not post lifecycle notifications.
    "];

    {
        supervise: true,
        notify_chan: none,
        sched: none
    }
}

fn task_builder() -> task_builder {
    #[doc = "Construct a task_builder"];

    let body_identity = fn@(+body: fn~()) -> fn~() { body };

    task_builder_({
        mutable opts: default_task_opts(),
        mutable gen_body: body_identity,
        can_not_copy: none
    })
}

fn get_opts(builder: task_builder) -> task_opts {
    #[doc = "Get the task_opts associated with a task_builder"];

    builder.opts
}

fn set_opts(builder: task_builder, opts: task_opts) {
    #[doc = "
    Set the task_opts associated with a task_builder

    To update a single option use a pattern like the following:

        set_opts(builder, {
            supervise: false
            with get_opts(builder)
        });
    "];

    builder.opts = opts;
}

fn add_wrapper(builder: task_builder, gen_body: fn@(+fn~()) -> fn~()) {
    #[doc = "
    Add a wrapper to the body of the spawned task.

    Before the task is spawned it is passed through a 'body generator'
    function that may perform local setup operations as well as wrap
    the task body in remote setup operations. With this the behavior
    of tasks can be extended in simple ways.

    This function augments the current body generator with a new body
    generator by applying the task body which results from the
    existing body generator to the new body generator.
    "];

    let prev_gen_body = builder.gen_body;
    builder.gen_body = fn@(+body: fn~()) -> fn~() {
        gen_body(prev_gen_body(body))
    };
}

fn run(-builder: task_builder, +f: fn~()) {
    #[doc = "
    Creates and exucutes a new child task

    Sets up a new task with its own call stack and schedules it to run
    the provided unique closure. The task has the properties and behavior
    specified by `builder`.

    # Failure

    When spawning into a new scheduler, the number of threads requested
    must be greater than zero.
    "];

    let body = builder.gen_body(f);
    spawn_raw(builder.opts, body);
}


/* Builder convenience functions */

fn future_result(builder: task_builder) -> future::future<task_result> {
    #[doc = "
    Get a future representing the exit status of the task.

    Taking the value of the future will block until the child task terminates.

    Note that the future returning by this function is only useful for
    obtaining the value of the next task to be spawning with the
    builder. If additional tasks are spawned with the same builder
    then a new result future must be obtained prior to spawning each
    task.
    "];

    // FIXME (1087, 1857): Once linked failure and notification are
    // handled in the library, I can imagine implementing this by just
    // registering an arbitrary number of task::on_exit handlers and
    // sending out messages.

    let po = comm::port();
    let ch = comm::chan(po);

    set_opts(builder, {
        notify_chan: some(ch)
        with get_opts(builder)
    });

    future::from_fn {||
        alt comm::recv(po) {
          exit(_, result) { result }
        }
    }
}

fn future_task(builder: task_builder) -> future::future<task> {
    #[doc = "Get a future representing the handle to the new task"];

    let mut po = comm::port();
    let ch = comm::chan(po);
    add_wrapper(builder) {|body|
        fn~() {
            comm::send(ch, get_task());
            body();
        }
    }
    future::from_port(po)
}

fn unsupervise(builder: task_builder) {
    #[doc = "Configures the new task to not propagate failure to its parent"];

    set_opts(builder, {
        supervise: false
        with get_opts(builder)
    });
}

fn run_listener<A:send>(-builder: task_builder,
                        +f: fn~(comm::port<A>)) -> comm::chan<A> {
    #[doc = "
    Runs a new task while providing a channel from the parent to the child

    Sets up a communication channel from the current task to the new
    child task, passes the port to child's body, and returns a channel
    linked to the port to the parent.

    This encapsulates some boilerplate handshaking logic that would
    otherwise be required to establish communication from the parent
    to the child.
    "];

    let setup_po = comm::port();
    let setup_ch = comm::chan(setup_po);

    run(builder) {||
        let po = comm::port();
        let mut ch = comm::chan(po);
        comm::send(setup_ch, ch);
        f(po);
    }

    comm::recv(setup_po)
}


/* Spawn convenience functions */

fn spawn(+f: fn~()) {
    #[doc = "
    Creates and exucutes a new child task

    Sets up a new task with its own call stack and schedules it to run
    the provided unique closure.

    This function is equivalent to `run(new_task_builder(), f)`.
    "];

    run(task_builder(), f);
}

fn spawn_listener<A:send>(+f: fn~(comm::port<A>)) -> comm::chan<A> {
    #[doc = "
    Runs a new task while providing a channel from the parent to the child

    Sets up a communication channel from the current task to the new
    child task, passes the port to child's body, and returns a channel
    linked to the port to the parent.

    This encapsulates some boilerplate handshaking logic that would
    otherwise be required to establish communication from the parent
    to the child.

    The simplest way to establish bidirectional communication between
    a parent in child is as follows:

        let po = comm::port();
        let ch = comm::chan(po);
        let ch = spawn_listener {|po|
            // Now the child has a port called 'po' to read from and
            // an environment-captured channel called 'ch'.
        };
        // Likewise, the parent has both a 'po' and 'ch'

    This function is equivalent to `run_listener(new_task_builder(), f)`.
    "];

    run_listener(task_builder(), f)
}

fn spawn_sched(mode: sched_mode, +f: fn~()) {
    #[doc = "
    Creates a new scheduler and executes a task on it

    Tasks subsequently spawned by that task will also execute on
    the new scheduler. When there are no more tasks to execute the
    scheduler terminates.

    # Failure

    In manual threads mode the number of threads requested must be
    greater than zero.
    "];

    let mut builder = task_builder();
    set_opts(builder, {
        sched: some({
            mode: mode,
            native_stack_size: none
        })
        with get_opts(builder)
    });
    run(builder, f);
}

fn try<T:send>(+f: fn~() -> T) -> result<T,()> {
    #[doc = "
    Execute a function in another task and return either the return value
    of the function or result::err.

    # Return value

    If the function executed successfully then try returns result::ok
    containing the value returned by the function. If the function fails
    then try returns result::err containing nil.
    "];

    let po = comm::port();
    let ch = comm::chan(po);
    let mut builder = task_builder();
    unsupervise(builder);
    let result = future_result(builder);
    run(builder) {||
        comm::send(ch, f());
    }
    alt future::get(result) {
      success { result::ok(comm::recv(po)) }
      failure { result::err(()) }
    }
}


/* Lifecycle functions */

fn yield() {
    #[doc = "Yield control to the task scheduler"];

    let task_ = rustrt::rust_get_task();
    let mut killed = false;
    rusti::task_yield(task_, killed);
    if killed && !failing() {
        fail "killed";
    }
}

fn failing() -> bool {
    #[doc = "True if the running task has failed"];

    rustrt::rust_task_is_unwinding(rustrt::rust_get_task())
}

fn get_task() -> task {
    #[doc = "Get a handle to the running task"];

    task(rustrt::get_task_id())
}


/* Internal */

type sched_id = int;
type task_id = int;

// These are both opaque runtime/compiler types that we don't know the
// structure of and should only deal with via unsafe pointer
type rust_task = libc::c_void;
type rust_closure = libc::c_void;

fn spawn_raw(opts: task_opts, +f: fn~()) unsafe {

    let mut f = if opts.supervise {
        f
    } else {
        // FIXME: The runtime supervision API is weird here because it
        // was designed to let the child unsupervise itself, when what
        // we actually want is for parents to unsupervise new
        // children.
        fn~() {
            rustrt::unsupervise();
            f();
        }
    };

    let fptr = ptr::addr_of(f);
    let closure: *rust_closure = unsafe::reinterpret_cast(fptr);

    let new_task = alt opts.sched {
      none {
        rustrt::new_task()
      }
      some(sched_opts) {
        new_task_in_new_sched(sched_opts)
      }
    };

    option::may(opts.notify_chan) {|c|
        // FIXME (1087): Would like to do notification in Rust
        rustrt::rust_task_config_notify(new_task, c);
    }

    rustrt::start_task(new_task, closure);
    unsafe::forget(f);

    fn new_task_in_new_sched(opts: sched_opts) -> *rust_task {
        if opts.native_stack_size != none {
            fail "native_stack_size scheduler option unimplemented";
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
        };

        let sched_id = rustrt::rust_new_sched(num_threads);
        rustrt::rust_new_task_in_sched(sched_id)
    }

}

#[abi = "rust-intrinsic"]
native mod rusti {
    fn task_yield(task: *rust_task, &killed: bool);
}

native mod rustrt {
    fn rust_get_sched_id() -> sched_id;
    fn rust_new_sched(num_threads: libc::uintptr_t) -> sched_id;

    fn get_task_id() -> task_id;
    fn rust_get_task() -> *rust_task;

    fn new_task() -> *rust_task;
    fn rust_new_task_in_sched(id: sched_id) -> *rust_task;

    fn rust_task_config_notify(
        task: *rust_task, &&chan: comm::chan<notification>);

    fn start_task(task: *rust_task, closure: *rust_closure);

    fn rust_task_is_unwinding(rt: *rust_task) -> bool;
    fn unsupervise();
}


#[test]
fn test_spawn_raw_simple() {
    let po = comm::port();
    let ch = comm::chan(po);
    spawn_raw(default_task_opts()) {||
        comm::send(ch, ());
    }
    comm::recv(po);
}

#[test]
#[ignore(cfg(target_os = "win32"))]
fn test_spawn_raw_unsupervise() {
    let opts = {
        supervise: false
        with default_task_opts()
    };
    spawn_raw(opts) {||
        fail;
    }
}

#[test]
#[ignore(cfg(target_os = "win32"))]
fn test_spawn_raw_notify() {
    let task_po = comm::port();
    let task_ch = comm::chan(task_po);
    let notify_po = comm::port();
    let notify_ch = comm::chan(notify_po);

    let opts = {
        notify_chan: some(notify_ch)
        with default_task_opts()
    };
    spawn_raw(opts) {||
        comm::send(task_ch, get_task());
    }
    let task_ = comm::recv(task_po);
    assert comm::recv(notify_po) == exit(task_, success);

    let opts = {
        supervise: false,
        notify_chan: some(notify_ch)
        with default_task_opts()
    };
    spawn_raw(opts) {||
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
    let builder = task_builder();
    run(builder) {||
        comm::send(ch, ());
    }
    comm::recv(po);
}

#[test]
fn test_add_wrapper() {
    let po = comm::port();
    let ch = comm::chan(po);
    let builder = task_builder();
    add_wrapper(builder) {|body|
        fn~() {
            body();
            comm::send(ch, ());
        }
    }
    run(builder) {||}
    comm::recv(po);
}

#[test]
#[ignore(cfg(target_os = "win32"))]
fn test_future_result() {
    let builder = task_builder();
    let result = future_result(builder);
    run(builder) {||}
    assert future::get(result) == success;

    let builder = task_builder();
    let result = future_result(builder);
    unsupervise(builder);
    run(builder) {|| fail }
    assert future::get(result) == failure;
}

#[test]
fn test_future_task() {
    let po = comm::port();
    let ch = comm::chan(po);
    let builder = task_builder();
    let task1 = future_task(builder);
    run(builder) {|| comm::send(ch, get_task()) }
    assert future::get(task1) == comm::recv(po);
}

#[test]
fn test_spawn_listiner_bidi() {
    let po = comm::port();
    let ch = comm::chan(po);
    let ch = spawn_listener {|po|
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
    alt try {||
        "Success!"
    } {
        result::ok("Success!") { }
        _ { fail; }
    }
}

#[test]
#[ignore(cfg(target_os = "win32"))]
fn test_try_fail() {
    alt try {||
        fail
    } {
        result::err(()) { }
        _ { fail; }
    }
}

#[test]
#[should_fail]
#[ignore(cfg(target_os = "win32"))]
fn test_spawn_sched_no_threads() {
    spawn_sched(manual_threads(0u)) {|| };
}

#[test]
fn test_spawn_sched() {
    let po = comm::port();
    let ch = comm::chan(po);

    fn f(i: int, ch: comm::chan<()>) {
        let parent_sched_id = rustrt::rust_get_sched_id();

        spawn_sched(single_threaded) {||
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

    spawn_sched(single_threaded) {||
        let parent_sched_id = rustrt::rust_get_sched_id();
        spawn {||
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
native mod testrt {
    fn rust_dbg_lock_create() -> *libc::c_void;
    fn rust_dbg_lock_destroy(lock: *libc::c_void);
    fn rust_dbg_lock_lock(lock: *libc::c_void);
    fn rust_dbg_lock_unlock(lock: *libc::c_void);
    fn rust_dbg_lock_wait(lock: *libc::c_void);
    fn rust_dbg_lock_signal(lock: *libc::c_void);
}

#[test]
fn test_spawn_sched_blocking() {

    // Testing that a task in one scheduler can block natively
    // without affecting other schedulers
    iter::repeat(20u) {||

        let start_po = comm::port();
        let start_ch = comm::chan(start_po);
        let fin_po = comm::port();
        let fin_ch = comm::chan(fin_po);

        let lock = testrt::rust_dbg_lock_create();

        spawn_sched(single_threaded) {||
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
            let val = 20;
            while val > 0 {
                val = comm::recv(po);
                comm::send(ch, val - 1);
            }
        }

        let setup_po = comm::port();
        let setup_ch = comm::chan(setup_po);
        let parent_po = comm::port();
        let parent_ch = comm::chan(parent_po);
        spawn {||
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

    spawnfn {||
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
    avoid_copying_the_body {|f|
        spawn_listener(fn~[move f](_po: comm::port<int>) {
            f();
        });
    }
}

#[test]
fn test_avoid_copying_the_body_run() {
    avoid_copying_the_body {|f|
        let builder = task_builder();
        run(builder) {||
            f();
        }
    }
}

#[test]
fn test_avoid_copying_the_body_run_listener() {
    avoid_copying_the_body {|f|
        let builder = task_builder();
        run_listener(builder, fn~[move f](_po: comm::port<int>) {
            f();
        });
    }
}

#[test]
fn test_avoid_copying_the_body_try() {
    avoid_copying_the_body {|f|
        try {||
            f()
        };
    }
}

#[test]
fn test_avoid_copying_the_body_future_task() {
    avoid_copying_the_body {|f|
        let builder = task_builder();
        future_task(builder);
        run(builder) {||
            f();
        }
    }
}

#[test]
fn test_avoid_copying_the_body_unsupervise() {
    avoid_copying_the_body {|f|
        let builder = task_builder();
        unsupervise(builder);
        run(builder) {||
            f();
        }
    }
}
