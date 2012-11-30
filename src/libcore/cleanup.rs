// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[doc(hidden)];

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use libc::{c_char, c_void, intptr_t, uintptr_t};
use ptr::{mut_null, null, to_unsafe_ptr};
use repr::BoxRepr;
use sys::TypeDesc;
use cast::transmute;

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

#[cfg(target_arch="x86")]
#[cfg(target_arch="arm")]
struct Registers {
    data: [u32 * 16]
}

#[cfg(target_arch="x86")]
#[cfg(target_arch="arm")]
struct Context {
    regs: Registers,
    next: *Context,
    pad: [u32 * 3]
}

#[cfg(target_arch="x86_64")]
struct Registers {
    data: [u64 * 22]
}

#[cfg(target_arch="x86_64")]
struct Context {
    regs: Registers,
    next: *Context,
    pad: uintptr_t
}

struct BoxedRegion {
    env: *Env,
    backing_region: *MemoryRegion,
    live_allocs: *BoxRepr
}

#[cfg(target_arch="x86")]
#[cfg(target_arch="arm")]
struct Task {
    // Public fields
    refcount: intptr_t,                 // 0
    id: TaskID,                         // 4
    pad: [u32 * 2],                     // 8
    ctx: Context,                       // 16
    stack_segment: *StackSegment,       // 96
    runtime_sp: uintptr_t,              // 100
    scheduler: *Scheduler,              // 104
    scheduler_loop: *SchedulerLoop,     // 108

    // Fields known only to the runtime
    kernel: *Kernel,                    // 112
    name: *c_char,                      // 116
    list_index: i32,                    // 120
    rendezvous_ptr: *uintptr_t,         // 124
    boxed_region: BoxedRegion           // 128
}

#[cfg(target_arch="x86_64")]
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
    list_index: i32,
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
    use io::WriterUtil;

    let task: *Task = transmute(rustrt::rust_get_task());

    // Pass 1: Make all boxes immortal.
    let box = (*task).boxed_region.live_allocs;
    let mut box: *mut BoxRepr = transmute(copy box);
    while box != mut_null() {
        debug!("making box immortal: %x", box as uint);
        (*box).header.ref_count = 0x77777777;
        box = transmute(copy (*box).header.next);
    }

    // Pass 2: Drop all boxes.
    let box = (*task).boxed_region.live_allocs;
    let mut box: *mut BoxRepr = transmute(copy box);
    while box != mut_null() {
        debug!("calling drop glue for box: %x", box as uint);
        let tydesc: *TypeDesc = transmute(copy (*box).header.type_desc);
        let drop_glue: DropGlue = transmute(((*tydesc).drop_glue, 0));
        drop_glue(to_unsafe_ptr(&tydesc), transmute(&(*box).data));

        box = transmute(copy (*box).header.next);
    }

    // Pass 3: Free all boxes.
    loop {
        let box = (*task).boxed_region.live_allocs;
        if box == null() { break; }
        let mut box: *mut BoxRepr = transmute(copy box);
        assert (*box).header.prev == null();

        debug!("freeing box: %x", box as uint);
        rt_free(transmute(box));
    }
}

/// Bindings to the runtime
extern mod rustrt {
    #[legacy_exports];
    #[rust_stack]
    /*priv*/ fn rust_get_task() -> *c_void;
}

