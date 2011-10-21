import cast = unsafe::reinterpret_cast;
import comm;
import option::{some, none};
import option = option::t;
import ptr;

export task;
export joinable_task;
export sleep;
export yield;
export task_notification;
export join;
export unsupervise;
export pin;
export unpin;
export set_min_stack;
export task_result;
export tr_success;
export tr_failure;
export get_task_id;
export spawn;
export spawn_notify;
export spawn_joinable;

native "rust" mod rustrt {                           // C Stack?
    fn task_sleep(time_in_us: uint);                 // No
    fn task_yield();                                 // No
    fn start_task(id: task_id, closure: *u8);        // No
    fn task_join(t: task_id) -> int;                 // Refactor
}

native "c-stack-cdecl" mod rustrt2 = "rustrt" {
    fn pin_task();                                   // Yes
    fn unpin_task();                                 // Yes
    fn get_task_id() -> task_id;                     // Yes

    fn set_min_stack(stack_size: uint);              // Yes

    fn new_task() -> task_id;
    fn drop_task(task: *rust_task);
    fn get_task_pointer(id: task_id) -> *rust_task;

    fn migrate_alloc(alloc: *u8, target: task_id);   // Yes
}

type rust_task =
    {id: task,
     mutable notify_enabled: u32,
     mutable notify_chan: comm::chan<task_notification>,
     mutable stack_ptr: *u8};

resource rust_task_ptr(task: *rust_task) { rustrt2::drop_task(task); }

type task = int;
type task_id = task;
type joinable_task = (task_id, comm::port<task_notification>);

fn get_task_id() -> task_id { rustrt2::get_task_id() }

/**
 * Hints the scheduler to yield this task for a specified ammount of time.
 *
 * arg: time_in_us maximum number of microseconds to yield control for
 */
fn sleep(time_in_us: uint) { ret rustrt::task_sleep(time_in_us); }

fn yield() { ret rustrt::task_yield(); }

tag task_result { tr_success; tr_failure; }

tag task_notification { exit(task, task_result); }

fn join(task_port: (task_id, comm::port<task_notification>)) -> task_result {
    let (id, port) = task_port;
    alt comm::recv::<task_notification>(port) {
      exit(_id, res) {
        if _id == id {
            ret res
        } else { fail #fmt["join received id %d, expected %d", _id, id] }
      }
    }
}

fn join_id(t: task_id) -> task_result {
    alt rustrt::task_join(t) { 0 { tr_success } _ { tr_failure } }
}

fn unsupervise() { ret sys::unsupervise(); }

fn pin() { rustrt2::pin_task(); }

fn unpin() { rustrt2::unpin_task(); }

fn set_min_stack(stack_size: uint) { rustrt2::set_min_stack(stack_size); }

fn spawn<~T>(-data: T, f: fn(T)) -> task {
    spawn_inner2(data, f, none)
}

fn spawn_notify<~T>(-data: T, f: fn(T),
                         notify: comm::chan<task_notification>) -> task {
    spawn_inner2(data, f, some(notify))
}

fn spawn_joinable<~T>(-data: T, f: fn(T)) -> joinable_task {
    let p = comm::port::<task_notification>();
    let id = spawn_notify(data, f, comm::chan::<task_notification>(p));
    ret (id, p);
}

// FIXME: To transition from the unsafe spawn that spawns a shared closure to
// the safe spawn that spawns a bare function we're going to write
// barefunc-spawn on top of unsafe-spawn.  Sadly, bind does not work reliably
// enough to suite our needs (#1034, probably others yet to be discovered), so
// we're going to copy the bootstrap data into a unique pointer, cast it to an
// unsafe pointer then wrap up the bare function and the unsafe pointer in a
// shared closure to spawn.
//
// After the transition this should all be rewritten.

fn spawn_inner2<~T>(-data: T, f: fn(T),
                    notify: option<comm::chan<task_notification>>)
    -> task_id {

    fn wrapper<~T>(-data: *u8, f: fn(T)) {
        let data: ~T = unsafe::reinterpret_cast(data);
        f(*data);
    }

    let data = ~data;
    let dataptr: *u8 = unsafe::reinterpret_cast(data);
    unsafe::leak(data);
    let wrapped = bind wrapper(dataptr, f);
    ret unsafe_spawn_inner(wrapped, notify);
}

// FIXME: This is the old spawn function that spawns a shared closure.
// It is a hack and needs to be rewritten.
fn unsafe_spawn_inner(-thunk: fn@(),
                      notify: option<comm::chan<task_notification>>) ->
   task_id unsafe {
    let id = rustrt2::new_task();

    let raw_thunk: {code: u32, env: u32} = cast(thunk);

    // set up the task pointer
    let task_ptr <- rust_task_ptr(rustrt2::get_task_pointer(id));

    assert (ptr::null() != (**task_ptr).stack_ptr);

    // copy the thunk from our stack to the new stack
    let sp: uint = cast((**task_ptr).stack_ptr);
    let ptrsize = sys::size_of::<*u8>();
    let thunkfn: *mutable uint = cast(sp - ptrsize * 2u);
    let thunkenv: *mutable uint = cast(sp - ptrsize);
    *thunkfn = cast(raw_thunk.code);;
    *thunkenv = cast(raw_thunk.env);;
    // align the stack to 16 bytes
    (**task_ptr).stack_ptr = cast(sp - ptrsize * 4u);

    // set up notifications if they are enabled.
    alt notify {
      some(c) {
        (**task_ptr).notify_enabled = 1u32;;
        (**task_ptr).notify_chan = c;
      }
      none { }
    }

    // give the thunk environment's allocation to the new task
    rustrt2::migrate_alloc(cast(raw_thunk.env), id);
    rustrt::start_task(id, cast(thunkfn));
    // don't cleanup the thunk in this task
    unsafe::leak(thunk);
    ret id;
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
