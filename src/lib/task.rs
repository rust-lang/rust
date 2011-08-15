import cast = unsafe::reinterpret_cast;

native "rust" mod rustrt {
    fn task_sleep(time_in_us: uint);
    fn task_yield();
    fn task_join(t: task_id) -> int;
    fn unsupervise();
    fn pin_task();
    fn unpin_task();
    fn get_task_id() -> task_id;

    type rust_chan;
    type rust_task;

    fn set_min_stack(stack_size: uint);

    fn new_task() -> task_id;
    fn drop_task(id : task_id);
    fn get_task_pointer(id : task_id) -> *rust_task;
    fn get_task_context(id : task_id) -> *x86_registers;
    fn start_task(id : task_id);
    fn get_task_trampoline() -> u32;

    fn migrate_alloc(alloc : *u8, target : task_id);

    fn leak[@T](thing : -T);
}

type task_id = int;

fn get_task_id() -> task_id {
    rustrt::get_task_id()
}

/**
 * Hints the scheduler to yield this task for a specified ammount of time.
 *
 * arg: time_in_us maximum number of microseconds to yield control for
 */
fn sleep(time_in_us: uint) { ret rustrt::task_sleep(time_in_us); }

fn yield() { ret rustrt::task_yield(); }

tag task_result { tr_success; tr_failure; }

// FIXME: Re-enable this once the task type is removed from the compiler.
/*
fn join(t: task) -> task_result {
    join_id(cast(t))
}
*/

fn join_id(t : task_id) -> task_result {
    alt rustrt::task_join(t) { 0 { tr_success } _ { tr_failure } }
}

fn unsupervise() { ret rustrt::unsupervise(); }

fn pin() { rustrt::pin_task(); }

fn unpin() { rustrt::unpin_task(); }

fn set_min_stack(stack_size : uint) {
    rustrt::set_min_stack(stack_size);
}

// FIXME: make this a fn~ once those are supported.
fn _spawn(thunk : fn() -> ()) -> task_id {
    let id = rustrt::new_task();

    // the order of arguments are outptr, taskptr, envptr.

    // In LLVM fastcall puts the first two in ecx, edx, and the rest on the
    // stack.
    let regs = rustrt::get_task_context(id);

    // set up the task pointer
    let task_ptr : u32 = cast(rustrt::get_task_pointer(id));
    (*regs).edx = task_ptr;

    let raw_thunk : { code: u32, env: u32 } = cast(thunk);
    (*regs).eip = raw_thunk.code;

    // okay, now we align the stack and add the environment pointer and a fake
    // return address.

    // -12 for the taskm output location, the env pointer
    // -4 for the return address.
    (*regs).esp = align_down((*regs).esp - 12u32) - 4u32;

    let ra : *mutable u32 = cast((*regs).esp);
    let env : *mutable u32 = cast((*regs).esp+4u32);
    let tptr : *mutable u32 = cast((*regs).esp+12u32);

    // put the return pointer in ecx.
    (*regs).ecx = (*regs).esp + 8u32;

    *tptr = task_ptr;
    *env = raw_thunk.env;
    *ra = rustrt::get_task_trampoline();

    rustrt::migrate_alloc(cast(raw_thunk.env), id);
    rustrt::start_task(id);

    rustrt::leak(thunk);

    // Drop twice because get_task_context and get_task_pounter both bump the
    // ref count and expect us to free it.
    rustrt::drop_task(id);
    rustrt::drop_task(id);

    ret id;
}

// Who says we can't write an operating system in Rust?
type x86_registers = {
    // This needs to match the structure in context.h
    mutable eax : u32,
    mutable ebx : u32,
    mutable ecx : u32,
    mutable edx : u32,
    mutable ebp : u32,
    mutable esi : u32,
    mutable edi : u32,
    mutable esp : u32,

    mutable cs : u16,
    mutable ds : u16,
    mutable ss : u16,
    mutable es : u16,
    mutable fs : u16,
    mutable gs : u16,

    mutable eflags : u32,
    mutable eip : u32
};

fn align_down(x : u32) -> u32 {
    // Aligns x down to 16 bytes
    x & !(15u32)
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
