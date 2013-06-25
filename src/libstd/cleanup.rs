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

use libc::{c_char, intptr_t, uintptr_t};
use ptr::{mut_null};
use repr::BoxRepr;
use cast::transmute;
use unstable::intrinsics::TyDesc;
#[cfg(not(test))] use unstable::lang::clear_task_borrow_list;

/**
 * Runtime structures
 *
 * NB: These must match the representation in the C++ runtime.
 */

type TaskID = uintptr_t;

struct StackSegment { priv opaque: () }
struct Scheduler { priv opaque: () }
struct SchedulerLoop { priv opaque: () }
struct Kernel { priv opaque: () }
struct Env { priv opaque: () }
struct AllocHeader { priv opaque: () }
struct MemoryRegion { priv opaque: () }

#[cfg(target_arch="x86")]
struct Registers {
    data: [u32, ..16]
}

#[cfg(target_arch="arm")]
#[cfg(target_arch="mips")]
struct Registers {
    data: [u32, ..32]
}

#[cfg(target_arch="x86")]
#[cfg(target_arch="arm")]
#[cfg(target_arch="mips")]
struct Context {
    regs: Registers,
    next: *Context,
    pad: [u32, ..3]
}

#[cfg(target_arch="x86_64")]
struct Registers {
    data: [u64, ..22]
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
#[cfg(target_arch="mips")]
struct Task {
    // Public fields
    refcount: intptr_t,                 // 0
    id: TaskID,                         // 4
    pad: [u32, ..2],                    // 8
    ctx: Context,                       // 16
    stack_segment: *StackSegment,       // 96
    runtime_sp: uintptr_t,              // 100
    scheduler: *Scheduler,              // 104
    scheduler_loop: *SchedulerLoop,     // 108

    // Fields known only to the runtime
    kernel: *Kernel,                    // 112
    name: *c_char,                      // 116
    list_index: i32,                    // 120
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
    boxed_region: BoxedRegion
}

/*
 * Box annihilation
 *
 * This runs at task death to free all boxes.
 */

struct AnnihilateStats {
    n_total_boxes: uint,
    n_unique_boxes: uint,
    n_bytes_freed: uint
}

unsafe fn each_live_alloc(read_next_before: bool,
                          f: &fn(box: *mut BoxRepr, uniq: bool) -> bool) -> bool {
    //! Walks the internal list of allocations

    use managed;

    let task: *Task = transmute(rustrt::rust_get_task());
    let box = (*task).boxed_region.live_allocs;
    let mut box: *mut BoxRepr = transmute(copy box);
    while box != mut_null() {
        let next_before = transmute(copy (*box).header.next);
        let uniq =
            (*box).header.ref_count == managed::raw::RC_MANAGED_UNIQUE;

        if !f(box, uniq) {
            return false;
        }

        if read_next_before {
            box = next_before;
        } else {
            box = transmute(copy (*box).header.next);
        }
    }
    return true;
}

#[cfg(unix)]
fn debug_mem() -> bool {
    ::rt::env::get().debug_mem
}

#[cfg(windows)]
fn debug_mem() -> bool {
    false
}

#[inline]
#[cfg(not(stage0))]
unsafe fn call_drop_glue(tydesc: *TyDesc, data: *i8) {
    // This function should be inlined when stage0 is gone
    ((*tydesc).drop_glue)(data);
}

#[inline]
#[cfg(stage0)]
unsafe fn call_drop_glue(tydesc: *TyDesc, data: *i8) {
    ((*tydesc).drop_glue)(0 as **TyDesc, data);
}

/// Destroys all managed memory (i.e. @ boxes) held by the current task.
#[cfg(not(test))]
#[lang="annihilate"]
pub unsafe fn annihilate() {
    use unstable::lang::local_free;
    use io::WriterUtil;
    use io;
    use libc;
    use sys;
    use managed;

    let mut stats = AnnihilateStats {
        n_total_boxes: 0,
        n_unique_boxes: 0,
        n_bytes_freed: 0
    };

    // Quick hack: we need to free this list upon task exit, and this
    // is a convenient place to do it.
    clear_task_borrow_list();

    // Pass 1: Make all boxes immortal.
    //
    // In this pass, nothing gets freed, so it does not matter whether
    // we read the next field before or after the callback.
    for each_live_alloc(true) |box, uniq| {
        stats.n_total_boxes += 1;
        if uniq {
            stats.n_unique_boxes += 1;
        } else {
            (*box).header.ref_count = managed::raw::RC_IMMORTAL;
        }
    }

    // Pass 2: Drop all boxes.
    //
    // In this pass, unique-managed boxes may get freed, but not
    // managed boxes, so we must read the `next` field *after* the
    // callback, as the original value may have been freed.
    for each_live_alloc(false) |box, uniq| {
        if !uniq {
            let tydesc = (*box).header.type_desc;
            let data = transmute(&(*box).data);
            call_drop_glue(tydesc, data);
        }
    }

    // Pass 3: Free all boxes.
    //
    // In this pass, managed boxes may get freed (but not
    // unique-managed boxes, though I think that none of those are
    // left), so we must read the `next` field before, since it will
    // not be valid after.
    for each_live_alloc(true) |box, uniq| {
        if !uniq {
            stats.n_bytes_freed +=
                (*((*box).header.type_desc)).size
                + sys::size_of::<BoxRepr>();
            local_free(transmute(box));
        }
    }

    if debug_mem() {
        // We do logging here w/o allocation.
        let dbg = libc::STDERR_FILENO as io::fd_t;
        dbg.write_str("annihilator stats:");
        dbg.write_str("\n  total_boxes: ");
        dbg.write_uint(stats.n_total_boxes);
        dbg.write_str("\n  unique_boxes: ");
        dbg.write_uint(stats.n_unique_boxes);
        dbg.write_str("\n  bytes_freed: ");
        dbg.write_uint(stats.n_bytes_freed);
        dbg.write_str("\n");
    }
}

/// Bindings to the runtime
pub mod rustrt {
    use libc::c_void;

    #[link_name = "rustrt"]
    pub extern {
        #[rust_stack]
        // FIXME (#4386): Unable to make following method private.
        pub unsafe fn rust_get_task() -> *c_void;
    }
}
