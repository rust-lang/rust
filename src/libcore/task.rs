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
export sleep;
export yield;
export task_notification;
export join;
export unsupervise;
export pin;
export unpin;
export task_result;
export tr_success;
export tr_failure;
export get_task;
export spawn;
export spawn_joinable;
export spawn_connected;
export connected_fn;
export connected_task;
export currently_unwinding;
export try;

#[abi = "rust-intrinsic"]
native mod rusti {
    // these must run on the Rust stack so that they can swap stacks etc:
    fn task_sleep(task: *rust_task, time_in_us: c::size_t, &killed: bool);
}

type rust_closure = {
    fnptr: c::intptr_t, envptr: c::intptr_t
};

#[link_name = "rustrt"]
#[abi = "cdecl"]
native mod rustrt {
    // these can run on the C stack:
    fn pin_task();
    fn unpin_task();
    fn get_task_id() -> task_id;
    fn rust_get_task() -> *rust_task;

    fn new_task() -> task_id;
    fn drop_task(task_id: *rust_task);
    fn get_task_pointer(id: task_id) -> *rust_task;

    fn migrate_alloc(alloc: *u8, target: task_id);

    fn start_task(id: task, closure: *rust_closure);

    fn rust_task_is_unwinding(rt: *rust_task) -> bool;
}

/* Section: Types */

type rust_task =
    {id: task,
     mutable notify_enabled: int,
     mutable notify_chan: comm::chan<task_notification>,
     mutable stack_ptr: *u8};

resource rust_task_ptr(task: *rust_task) { rustrt::drop_task(task); }

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
    spawn_inner(f, none)
}

fn spawn_inner(-f: fn~(),
               notify: option<comm::chan<task_notification>>) -> task unsafe {
    let closure: *rust_closure = unsafe::reinterpret_cast(ptr::addr_of(f));
    #debug("spawn: closure={%x,%x}", (*closure).fnptr, (*closure).envptr);
    let id = rustrt::new_task();

    // set up notifications if they are enabled.
    option::may(notify) {|c|
        let task_ptr <- rust_task_ptr(rustrt::get_task_pointer(id));
        (**task_ptr).notify_enabled = 1;
        (**task_ptr).notify_chan = c;
    }

    rustrt::start_task(id, closure);
    unsafe::leak(f);
    ret id;
}

/*
Type: joinable_task

A task that sends notification upon termination
*/
type joinable_task = (task, comm::port<task_notification>);

fn spawn_joinable(+f: fn~()) -> joinable_task {
    let notify_port = comm::port();
    let notify_chan = comm::chan(notify_port);
    let task = spawn_inner(f, some(notify_chan));
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
tag task_result {
    /* Variant: tr_success */
    tr_success;
    /* Variant: tr_failure */
    tr_failure;
}

/*
Tag: task_notification

Message sent upon task exit to indicate normal or abnormal termination
*/
tag task_notification {
    /* Variant: exit */
    exit(task, task_result);
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
Function: sleep

Hints the scheduler to yield this task for a specified ammount of time.

Parameters:

time_in_us - maximum number of microseconds to yield control for
*/
fn sleep(time_in_us: uint) {
    let task = rustrt::rust_get_task();
    let killed = false;
    // FIXME: uncomment this when extfmt is moved to core
    // in a snapshot.
    // #debug("yielding for %u us", time_in_us);
    rusti::task_sleep(task, time_in_us, killed);
    if killed && !currently_unwinding() {
        fail "killed";
    }
}

/*
Function: yield

Yield control to the task scheduler

The scheduler may schedule another task to execute.
*/
fn yield() { sleep(1u) }

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
            // FIXME: uncomment this when extfmt is moved to core
            // in a snapshot.
            // fail #fmt["join received id %d, expected %d", _id, id]
            fail;
        }
      }
    }
}

/*
Function: unsupervise

Detaches this task from its parent in the task tree

An unsupervised task will not propagate its failure up the task tree
*/
fn unsupervise() { ret sys::unsupervise(); }

/*
Function: pin

Pins the current task and future child tasks to a single scheduler thread
*/
fn pin() { rustrt::pin_task(); }

/*
Function: unpin

Unpin the current task and future child tasks
*/
fn unpin() { rustrt::unpin_task(); }

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
      tr_success. { result::ok(comm::recv(p)) }
      tr_failure. { result::err(()) }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_sleep() { sleep(1000000u); }

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
          tr_success. {/* yay! */ }
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
          tr_failure. {/* yay! */ }
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
}


// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
