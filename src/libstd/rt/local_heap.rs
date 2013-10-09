// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The local, garbage collected heap

use libc;
use libc::{c_void, uintptr_t, size_t};
use ops::Drop;
use option::{Option, None, Some};
use rt::local::Local;
use rt::task::Task;
use unstable::raw;

type MemoryRegion = c_void;

struct Env { priv opaque: () }

struct BoxedRegion {
    env: *Env,
    backing_region: *MemoryRegion,
    live_allocs: *raw::Box<()>,
}

pub type OpaqueBox = c_void;
pub type TypeDesc = c_void;

pub struct LocalHeap {
    memory_region: *MemoryRegion,
    boxed_region: *BoxedRegion
}

impl LocalHeap {
    #[fixed_stack_segment] #[inline(never)]
    pub fn new() -> LocalHeap {
        unsafe {
            // XXX: These usually come from the environment
            let detailed_leaks = false as uintptr_t;
            let poison_on_free = false as uintptr_t;
            let region = rust_new_memory_region(detailed_leaks, poison_on_free);
            assert!(region.is_not_null());
            let boxed = rust_new_boxed_region(region, poison_on_free);
            assert!(boxed.is_not_null());
            LocalHeap {
                memory_region: region,
                boxed_region: boxed
            }
        }
    }

    #[fixed_stack_segment] #[inline(never)]
    pub fn alloc(&mut self, td: *TypeDesc, size: uint) -> *OpaqueBox {
        unsafe {
            return rust_boxed_region_malloc(self.boxed_region, td, size as size_t);
        }
    }

    #[fixed_stack_segment] #[inline(never)]
    pub fn realloc(&mut self, ptr: *OpaqueBox, size: uint) -> *OpaqueBox {
        unsafe {
            return rust_boxed_region_realloc(self.boxed_region, ptr, size as size_t);
        }
    }

    #[fixed_stack_segment] #[inline(never)]
    pub fn free(&mut self, box: *OpaqueBox) {
        unsafe {
            return rust_boxed_region_free(self.boxed_region, box);
        }
    }
}

impl Drop for LocalHeap {
    #[fixed_stack_segment] #[inline(never)]
    fn drop(&mut self) {
        unsafe {
            rust_delete_boxed_region(self.boxed_region);
            rust_delete_memory_region(self.memory_region);
        }
    }
}

pub unsafe fn local_malloc(td: *libc::c_char, size: libc::uintptr_t) -> *libc::c_char {
    // XXX: Unsafe borrow for speed. Lame.
    let task: Option<*mut Task> = Local::try_unsafe_borrow();
    match task {
        Some(task) => {
            (*task).heap.alloc(td as *libc::c_void, size as uint) as *libc::c_char
        }
        None => rtabort!("local malloc outside of task")
    }
}

// A little compatibility function
pub unsafe fn local_free(ptr: *libc::c_char) {
    // XXX: Unsafe borrow for speed. Lame.
    let task_ptr: Option<*mut Task> = Local::try_unsafe_borrow();
    match task_ptr {
        Some(task) => {
            (*task).heap.free(ptr as *libc::c_void);
        }
        None => rtabort!("local free outside of task")
    }
}

pub fn live_allocs() -> *raw::Box<()> {
    let region = do Local::borrow |task: &mut Task| {
        task.heap.boxed_region
    };

    return unsafe { (*region).live_allocs };
}

extern {
    fn rust_new_memory_region(detailed_leaks: uintptr_t,
                               poison_on_free: uintptr_t) -> *MemoryRegion;
    fn rust_delete_memory_region(region: *MemoryRegion);
    fn rust_new_boxed_region(region: *MemoryRegion,
                             poison_on_free: uintptr_t) -> *BoxedRegion;
    fn rust_delete_boxed_region(region: *BoxedRegion);
    fn rust_boxed_region_malloc(region: *BoxedRegion,
                                td: *TypeDesc,
                                size: size_t) -> *OpaqueBox;
    fn rust_boxed_region_realloc(region: *BoxedRegion,
                                 ptr: *OpaqueBox,
                                 size: size_t) -> *OpaqueBox;
    fn rust_boxed_region_free(region: *BoxedRegion, box: *OpaqueBox);
}

#[cfg(test)]
mod bench {
    use extra::test::BenchHarness;

    #[bench]
    fn alloc_managed_small(bh: &mut BenchHarness) {
        do bh.iter {
            @10;
        }
    }

    #[bench]
    fn alloc_managed_big(bh: &mut BenchHarness) {
        do bh.iter {
            @[10, ..1000];
        }
    }
}
