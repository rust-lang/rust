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

use libc::{c_void, uintptr_t, size_t};
use ops::Drop;

type MemoryRegion = c_void;
type BoxedRegion = c_void;

pub type OpaqueBox = c_void;
pub type TypeDesc = c_void;

pub struct LocalHeap {
    memory_region: *MemoryRegion,
    boxed_region: *BoxedRegion
}

impl LocalHeap {
    pub fn new() -> LocalHeap {
        unsafe {
            // Don't need synchronization for the single-threaded local heap
            let synchronized = false as uintptr_t;
            // XXX: These usually come from the environment
            let detailed_leaks = false as uintptr_t;
            let poison_on_free = false as uintptr_t;
            let region = rust_new_memory_region(synchronized, detailed_leaks, poison_on_free);
            assert!(region.is_not_null());
            let boxed = rust_new_boxed_region(region, poison_on_free);
            assert!(boxed.is_not_null());
            LocalHeap {
                memory_region: region,
                boxed_region: boxed
            }
        }
    }

    pub fn alloc(&mut self, td: *TypeDesc, size: uint) -> *OpaqueBox {
        unsafe {
            return rust_boxed_region_malloc(self.boxed_region, td, size as size_t);
        }
    }

    pub fn free(&mut self, box: *OpaqueBox) {
        unsafe {
            return rust_boxed_region_free(self.boxed_region, box);
        }
    }
}

impl Drop for LocalHeap {
    fn finalize(&self) {
        unsafe {
            rust_delete_boxed_region(self.boxed_region);
            rust_delete_memory_region(self.memory_region);
        }
    }
}

extern {
    fn rust_new_memory_region(synchronized: uintptr_t,
                               detailed_leaks: uintptr_t,
                               poison_on_free: uintptr_t) -> *MemoryRegion;
    fn rust_delete_memory_region(region: *MemoryRegion);
    fn rust_new_boxed_region(region: *MemoryRegion,
                             poison_on_free: uintptr_t) -> *BoxedRegion;
    fn rust_delete_boxed_region(region: *BoxedRegion);
    fn rust_boxed_region_malloc(region: *BoxedRegion,
                                td: *TypeDesc,
                                size: size_t) -> *OpaqueBox;
    fn rust_boxed_region_free(region: *BoxedRegion, box: *OpaqueBox);
}
