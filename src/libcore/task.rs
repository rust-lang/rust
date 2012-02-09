/*
Module: task

Task management.

An executing Rust program consists of a tree of tasks, each with their own
stack, and sole ownership of their allocated heap data. Tasks communicate
with each other using ports and channels.

When a task fails, that failure will propagate to its parent (the task
that spawned it) and the parent will fail as well. The reverse is not
true: when a parent task fails its children will continue executing. When
the root (main) task fails, all tasks fail, and then so does the entire
process.

A task may remove itself from this failure propagation mechanism by
calling the <unsupervise> function, after which failure will only
result in the termination of that task.

Tasks may execute in parallel and are scheduled automatically by the runtime.

Example:

> spawn {||
>   log(debug, "Hello, World!");
> };

*/
import cast = unsafe::reinterpret_cast;
import comm;
import ptr;
import c = ctypes;

export task;
export joinable_task;
export yield;
export task_notification;
export join;
export unsupervise;
export task_result;
export tr_success;
export tr_failure;
export get_task;
export spawn;
export spawn_joinable;
export spawn_connected;
export spawn_sched;
export connected_fn;
export connected_task;
export currently_unwinding;
export try;

#[abi = "rust-intrinsic"]
native mod rusti {
    // these must run on the Rust stack so that they can swap stacks etc:
    fn task_yield(task: *rust_task, &killed: bool);
}

type rust_closure = {
    fnptr: c::intptr_t, envptr: c::intptr_t
};

#[link_name = "rustrt"]
#[abi = "cdecl"]
native mod rustrt {
    fn rust_get_sched_id() -> sched_id;
    fn rust_new_sched(num_threads: c::uintptr_t) -> sched_id;

    fn get_task_id() -> task_id;
    fn rust_get_task() -> *rust_task;

    fn new_task() -> task_id;
    fn rust_new_task_in_sched(id: sched_id) -> task_id;

    fn rust_task_config_notify(
        id: task_id, &&chan: comm::chan<task_notification>);

    fn start_task(id: task, closure: *rust_closure);

    fn rust_task_is_unwinding(rt: *rust_task) -> bool;
    fn unsupervise();
}

/* Section: Types */

type rust_task = *ctypes::void;

type sched_id = int;
type task_id = int;

/*
Type: task

A handle to a task
*/
type task = task_id;

/*
Function: spawn

Creates and executes a new child task

Sets up a new task with its own call stack and schedules it to be
executed.  Upon execution, the closure `f()` will be invoked.

Parameters:

f - A function to execute in the new task

Returns:

A handle to the new task
*/
fn spawn(+f: fn~()) -> task {
    spawn_inner(f, none, new_task_in_this_sched)
}

fn spawn_inner(
    -f: fn~(),
    notify: option<comm::chan<task_notification>>,
    new_task: fn() -> task_id
) -> task unsafe {
    let closure: *rust_closure = unsafe::reinterpret_cast(ptr::addr_of(f));
    #debug("spawn: closure={%x,%x}", (*closure).fnptr, (*closure).envptr);
    let id = new_task();

    // set up notifications if they are enabled.
    option::may(notify) {|c|
        rustrt::rust_task_config_notify(id, c);
    }

    rustrt::start_task(id, closure);
    unsafe::leak(f);
    ret id;
}

fn new_task_in_this_sched() -> task_id {
    rustrt::new_task()
}

fn new_task_in_new_sched(num_threads: uint) -> task_id {
    let sched_id = rustrt::rust_new_sched(num_threads);
    rustrt::rust_new_task_in_sched(sched_id)
}

/*
Function: spawn_sched

Creates a new scheduler and executes a task on it. Tasks subsequently
spawned by that task will also execute on the new scheduler. When
there are no more tasks to execute the scheduler terminates.

Arguments:

num_threads - The number of OS threads to dedicate schedule tasks on
f - A unique closure to execute as a task on the new scheduler

Failure:

The number of threads must be greater than 0

*/
fn spawn_sched(num_threads: uint, +f: fn~()) -> task {
    if num_threads < 1u {
        fail "Can not create a scheduler with no threads";
    }
    spawn_inner(f, none, bind new_task_in_new_sched(num_threads))
}

/*
Type: joinable_task

A task that sends notification upon termination
*/
type joinable_task = (task, comm::port<task_notification>);

fn spawn_joinable(+f: fn~()) -> joinable_task {
    let notify_port = comm::port();
    let notify_chan = comm::chan(notify_port);
    let task = spawn_inner(f, some(notify_chan), new_task_in_this_sched);
    ret (task, notify_port);
    /*
    resource notify_rsrc(data: (comm::chan<task_notification>,
                                task,
                                @mutable task_result)) {
        let (chan, task, tr) = data;
        let msg = exit(task, *tr);
        comm::send(chan, msg);
    }

    let notify_port = comm::port();
    let notify_chan = comm::chan(notify_port);
    let g = fn~[copy notify_chan; move f]() {
        let this_task = rustrt::get_task_id();
        let result = @mutable tr_failure;
        let _rsrc = notify_rsrc((notify_chan, this_task, result));
        f();
        *result = tr_success; // rsrc will fire msg when fn returns
    };
    let task = spawn(g);
    ret (task, notify_port);
    */
}

/*
Tag: task_result

Indicates the manner in which a task exited
*/
enum task_result {
    /* Variant: tr_success */
    tr_success,
    /* Variant: tr_failure */
    tr_failure,
}

/*
Tag: task_notification

Message sent upon task exit to indicate normal or abnormal termination
*/
enum task_notification {
    /* Variant: exit */
    exit(task, task_result),
}

/*
Type: connected_fn

The prototype for a connected child task function.  Such a function will be
supplied with a channel to send messages to the parent and a port to receive
messages from the parent. The type parameter `ToCh` is the type for messages
sent from the parent to the child and `FrCh` is the type for messages sent
from the child to the parent. */
type connected_fn<ToCh, FrCh> = fn~(comm::port<ToCh>, comm::chan<FrCh>);

/*
Type: connected_fn

The result type of <spawn_connected>
*/
type connected_task<ToCh, FrCh> = {
    from_child: comm::port<FrCh>,
    to_child: comm::chan<ToCh>,
    task: task
};

/*
Function: spawn_connected

Spawns a child task along with a port/channel for exchanging messages
with the parent task.  The type `ToCh` represents messages sent to the child
and `FrCh` messages received from the child.

Parameters:

f - the child function to execute

Returns:

The new child task along with the port to receive messages and the channel
to send messages.
*/
fn spawn_connected<ToCh:send, FrCh:send>(+f: connected_fn<ToCh, FrCh>)
    -> connected_task<ToCh,FrCh> {
    let from_child_port = comm::port::<FrCh>();
    let from_child_chan = comm::chan(from_child_port);
    let get_to_child_port = comm::port::<comm::chan<ToCh>>();
    let get_to_child_chan = comm::chan(get_to_child_port);
    let child_task = spawn(fn~[move f]() {
        let to_child_port = comm::port::<ToCh>();
        comm::send(get_to_child_chan, comm::chan(to_child_port));
        f(to_child_port, from_child_chan);
    });
    let to_child_chan = comm::recv(get_to_child_port);
    ret {from_child: from_child_port,
         to_child: to_child_chan,
         task: child_task};
}

/* Section: Operations */

/*
Type: get_task

Retreives a handle to the currently executing task
*/
fn get_task() -> task { rustrt::get_task_id() }

/*
Function: yield

Yield control to the task scheduler

The scheduler may schedule another task to execute.
*/
fn yield() {
    let task = rustrt::rust_get_task();
    let killed = false;
    rusti::task_yield(task, killed);
    if killed && !currently_unwinding() {
        fail "killed";
    }
}

/*
Function: join

Wait for a child task to exit

The child task must have been spawned with <spawn_joinable>, which
produces a notification port that the child uses to communicate its
exit status.

Returns:

A task_result indicating whether the task terminated normally or failed
*/
fn join(task_port: joinable_task) -> task_result {
    let (id, port) = task_port;
    alt comm::recv::<task_notification>(port) {
      exit(_id, res) {
        if _id == id {
            ret res
        } else {
            fail #fmt["join received id %d, expected %d", _id, id]
        }
      }
    }
}

/*
Function: unsupervise

Detaches this task from its parent in the task tree

An unsupervised task will not propagate its failure up the task tree
*/
fn unsupervise() {
    rustrt::unsupervise();
}

/*
Function: currently_unwinding()

True if we are currently unwinding after a failure.
*/
fn currently_unwinding() -> bool {
    rustrt::rust_task_is_unwinding(rustrt::rust_get_task())
}

/*
Function: try

Execute a function in another task and return either the return value
of the function or result::err.

Returns:

If the function executed successfully then try returns result::ok
containing the value returned by the function. If the function fails
then try returns result::err containing nil.
*/
fn try<T:send>(+f: fn~() -> T) -> result::t<T,()> {
    let p = comm::port();
    let ch = comm::chan(p);
    alt join(spawn_joinable {||
        unsupervise();
        comm::send(ch, f());
    }) {
      tr_success { result::ok(comm::recv(p)) }
      tr_failure { result::err(()) }
    }
}

#[cfg(test)]
mod tests {
    // FIXME: Leaks on windows
    #[test]
    #[ignore(cfg(target_os = "win32"))]
    fn test_unsupervise() {
        fn f() { unsupervise(); fail; }
        spawn {|| f();};
    }

    #[test]
    fn test_lib_spawn() {
        fn foo() { #error("Hello, World!"); }
        spawn {|| foo();};
    }

    #[test]
    fn test_lib_spawn2() {
        fn foo(x: int) { assert (x == 42); }
        spawn {|| foo(42);};
    }

    #[test]
    fn test_join_chan() {
        fn winner() { }

        let t = spawn_joinable {|| winner();};
        alt join(t) {
          tr_success {/* yay! */ }
          _ { fail "invalid task status received" }
        }
    }

    // FIXME: Leaks on windows
    #[test]
    #[ignore(cfg(target_os = "win32"))]
    fn test_join_chan_fail() {
        fn failer() { unsupervise(); fail }

        let t = spawn_joinable {|| failer();};
        alt join(t) {
          tr_failure {/* yay! */ }
          _ { fail "invalid task status received" }
        }
    }

    #[test]
    fn spawn_polymorphic() {
        fn foo<T:send>(x: T) { log(error, x); }
        spawn {|| foo(true);};
        spawn {|| foo(42);};
    }

    #[test]
    fn try_success() {
        alt try {||
            "Success!"
        } {
            result::ok("Success!") { }
            _ { fail; }
        }
    }

    #[test]
    #[ignore(cfg(target_os = "win32"))]
    fn try_fail() {
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
    fn spawn_sched_no_threads() {
        spawn_sched(0u) {|| };
    }

    #[test]
    fn spawn_sched_1() {
        let po = comm::port();
        let ch = comm::chan(po);

        fn f(i: int, ch: comm::chan<()>) {
            let parent_sched_id = rustrt::rust_get_sched_id();

            spawn_sched(1u) {||
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
    fn spawn_sched_childs_on_same_sched() {
        let po = comm::port();
        let ch = comm::chan(po);

        spawn_sched(1u) {||
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
    native mod rt {
        fn rust_dbg_lock_create() -> *ctypes::void;
        fn rust_dbg_lock_destroy(lock: *ctypes::void);
        fn rust_dbg_lock_lock(lock: *ctypes::void);
        fn rust_dbg_lock_unlock(lock: *ctypes::void);
        fn rust_dbg_lock_wait(lock: *ctypes::void);
        fn rust_dbg_lock_signal(lock: *ctypes::void);
    }

    #[test]
    fn spawn_sched_blocking() {

        // Testing that a task in one scheduler can block natively
        // without affecting other schedulers
        iter::repeat(20u) {||

            let start_po = comm::port();
            let start_ch = comm::chan(start_po);
            let fin_po = comm::port();
            let fin_ch = comm::chan(fin_po);

            let lock = rt::rust_dbg_lock_create();

            spawn_sched(1u) {||
                rt::rust_dbg_lock_lock(lock);

                comm::send(start_ch, ());

                // Block the scheduler thread
                rt::rust_dbg_lock_wait(lock);
                rt::rust_dbg_lock_unlock(lock);

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
            rt::rust_dbg_lock_lock(lock);
            rt::rust_dbg_lock_signal(lock);
            rt::rust_dbg_lock_unlock(lock);
            comm::recv(fin_po);
            rt::rust_dbg_lock_destroy(lock);
        }
    }

}


// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
