use libc::{c_char, c_void, intptr_t, uintptr_t};
use ptr::{mut_null, null, to_unsafe_ptr};
use repr::BoxRepr;
use sys::TypeDesc;
use unsafe::transmute;

export annihilate;

/**
 * Runtime structures
 *
 * NB: These must match the representation in the C++ runtime.
 */

type DropGlue = fn(**TypeDesc, *c_void);
type FreeGlue = fn(**TypeDesc, *c_void);

type TaskID = uintptr_t;

struct StackSegment { priv opaque: () }
struct Scheduler { priv opaque: () }
struct SchedulerLoop { priv opaque: () }
struct Kernel { priv opaque: () }
struct Env { priv opaque: () }
struct AllocHeader { priv opaque: () }
struct MemoryRegion { priv opaque: () }

// XXX: i386
struct Registers {
    data: [u64 * 22]
}

struct Context {
    regs: Registers,
    next: *Context,
    pad: u64
}

struct BoxedRegion {
    env: *Env,
    backing_region: *MemoryRegion,
    live_allocs: *BoxRepr
}

struct Task {
    // Public fields
    refcount: intptr_t,
    id: TaskID,
    ctx: Context,
    stack_segment: *StackSegment,
    runtime_sp: uintptr_t,
    scheduler: *Scheduler,
    scheduler_loop: *SchedulerLoop,

    // Fields known only to the runtime
    kernel: *Kernel,
    name: *c_char,
    list_index: *i32,
    rendezvous_ptr: *uintptr_t,
    boxed_region: BoxedRegion
}

/*
 * Box annihilation
 *
 * This runs at task death to free all boxes.
 */

/// Destroys all managed memory (i.e. @ boxes) held by the current task.
#[cfg(notest)]
#[lang="annihilate"]
pub unsafe fn annihilate() {
    use rt::rt_free;

    let task: *Task = transmute(rustrt::rust_get_task());

    // Pass 1: Make all boxes immortal.
    let box = (*task).boxed_region.live_allocs;
    let mut box: *mut BoxRepr = transmute(copy box);
    assert (*box).prev == null();
    while box != mut_null() {
        debug!("making box immortal: %x", box as uint);
        (*box).ref_count = 0x77777777;
        box = transmute(copy (*box).next);
    }

    // Pass 2: Drop all boxes.
    let box = (*task).boxed_region.live_allocs;
    let mut box: *mut BoxRepr = transmute(copy box);
    assert (*box).prev == null();
    while box != mut_null() {
        debug!("calling drop glue for box: %x", box as uint);
        let tydesc: *TypeDesc = transmute(copy (*box).type_desc);
        let drop_glue: DropGlue = transmute(((*tydesc).drop_glue, 0));
        drop_glue(to_unsafe_ptr(&tydesc), transmute(&(*box).data));

        box = transmute(copy (*box).next);
    }

    // Pass 3: Free all boxes.
    loop {
        let box = (*task).boxed_region.live_allocs;
        if box == null() { break; }
        let mut box: *mut BoxRepr = transmute(copy box);
        assert (*box).prev == null();

        debug!("freeing box: %x", box as uint);
        rt_free(transmute(box));
    }
}

/// Bindings to the runtime
extern mod rustrt {
    #[rust_stack]
    /*priv*/ fn rust_get_task() -> *c_void;
}

