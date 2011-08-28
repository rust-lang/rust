import cast = unsafe::reinterpret_cast;
import comm;
import option::some;
import option::none;
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
export spawn;
export spawn_notify;
export spawn_joinable;
export task_result;
export tr_success;
export tr_failure;
export get_task_id;

native "rust" mod rustrt {
    fn task_sleep(time_in_us: uint);
    fn task_yield();
    fn task_join(t: task_id) -> int;
    fn unsupervise();
    fn pin_task();
    fn unpin_task();
    fn get_task_id() -> task_id;

    type rust_chan;

    fn set_min_stack(stack_size: uint);

    fn new_task() -> task_id;
    fn drop_task(task: *rust_task);
    fn get_task_pointer(id: task_id) -> *rust_task;
    fn start_task(id: task_id);
    fn get_task_trampoline() -> u32;

    fn migrate_alloc(alloc: *u8, target: task_id);
}

type rust_task =
    {id: task,
     mutable notify_enabled: u32,
     mutable notify_chan: comm::chan<task_notification>,
     ctx: task_context,
     stack_ptr: *u8};

type task_context = {regs: x86_registers, next: *u8};

resource rust_task_ptr(task: *rust_task) { rustrt::drop_task(task); }

type task = int;
type task_id = task;
type joinable_task = (task_id, comm::port<task_notification>);

fn get_task_id() -> task_id { rustrt::get_task_id() }

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
        } else { fail #ifmt["join received id %d, expected %d", _id, id] }
      }
    }
}

fn join_id(t: task_id) -> task_result {
    alt rustrt::task_join(t) { 0 { tr_success } _ { tr_failure } }
}

fn unsupervise() { ret rustrt::unsupervise(); }

fn pin() { rustrt::pin_task(); }

fn unpin() { rustrt::unpin_task(); }

fn set_min_stack(stack_size: uint) { rustrt::set_min_stack(stack_size); }

fn spawn(thunk: -fn()) -> task { spawn_inner(thunk, none) }

fn spawn_notify(thunk: -fn(), notify: comm::chan<task_notification>) -> task {
    spawn_inner(thunk, some(notify))
}

fn spawn_joinable(thunk: -fn()) -> joinable_task {
    let p = comm::port::<task_notification>();
    let id = spawn_notify(thunk, comm::chan::<task_notification>(p));
    ret (id, p);
}

// FIXME: make this a fn~ once those are supported.
fn spawn_inner(thunk: -fn(), notify: option<comm::chan<task_notification>>) ->
   task_id {
    let id = rustrt::new_task();

    // the order of arguments are outptr, taskptr, envptr.
    // LLVM fastcall puts the first two in ecx, edx, and the rest on the
    // stack.

    // set up the task pointer
    let task_ptr = rust_task_ptr(rustrt::get_task_pointer(id));
    let regs = ptr::addr_of((**task_ptr).ctx.regs);
    (*regs).edx = cast(*task_ptr);
    (*regs).esp = cast((**task_ptr).stack_ptr);

    assert (ptr::null() != (**task_ptr).stack_ptr);

    let raw_thunk: {code: u32, env: u32} = cast(thunk);
    (*regs).eip = raw_thunk.code;

    // set up notifications if they are enabled.
    alt notify {
      some(c) {
        (**task_ptr).notify_enabled = 1u32;;
        (**task_ptr).notify_chan = c;
      }
      none { }
    };

    // okay, now we align the stack and add the environment pointer and a fake
    // return address.

    // -12 for the taskm output location, the env pointer
    // -4 for the return address.
    (*regs).esp = align_down((*regs).esp - 12u32) - 4u32;

    let ra: *mutable u32 = cast((*regs).esp);
    let env: *mutable u32 = cast((*regs).esp + 4u32);
    let tptr: *mutable u32 = cast((*regs).esp + 12u32);

    // put the return pointer in ecx.
    (*regs).ecx = (*regs).esp + 8u32;;

    *tptr = cast(*task_ptr);;
    *env = raw_thunk.env;;
    *ra = rustrt::get_task_trampoline();

    rustrt::migrate_alloc(cast(raw_thunk.env), id);
    rustrt::start_task(id);

    unsafe::leak(thunk);

    ret id;
}

// Who says we can't write an operating system in Rust?
type x86_registers =
    // This needs to match the structure in context.h


    {mutable eax: u32,
     mutable ebx: u32,
     mutable ecx: u32,
     mutable edx: u32,
     mutable ebp: u32,
     mutable esi: u32,
     mutable edi: u32,
     mutable esp: u32,
     mutable cs: u16,
     mutable ds: u16,
     mutable ss: u16,
     mutable es: u16,
     mutable fs: u16,
     mutable gs: u16,
     mutable eflags: u32,
     mutable eip: u32};

fn align_down(x: u32) -> u32 {

    // Aligns x down to 16 bytes
    x & !15u32
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
