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

use core::prelude::*;

use alloc::libc_heap;
use alloc::util;
use libc::{c_void, free};

use core::mem;
use core::ptr;
use core::raw;
use local::Local;
use task::Task;

static RC_IMMORTAL : uint = 0x77777777;

pub type Box = raw::Box<()>;

pub struct MemoryRegion {
    live_allocations: uint,
}

pub struct LocalHeap {
    memory_region: MemoryRegion,
    live_allocs: *mut raw::Box<()>,
}

impl LocalHeap {
    pub fn new() -> LocalHeap {
        LocalHeap {
            memory_region: MemoryRegion { live_allocations: 0 },
            live_allocs: ptr::null_mut(),
        }
    }

    #[inline]
    #[allow(deprecated)]
    pub fn alloc(&mut self,
                 drop_glue: fn(*mut u8),
                 size: uint,
                 align: uint) -> *mut Box {
        let total_size = util::get_box_size(size, align);
        let alloc = self.memory_region.malloc(total_size);
        {
            // Make sure that we can't use `mybox` outside of this scope
            let mybox: &mut Box = unsafe { mem::transmute(alloc) };
            // Clear out this box, and move it to the front of the live
            // allocations list
            mybox.drop_glue = drop_glue;
            mybox.ref_count = 1;
            mybox.prev = ptr::null_mut();
            mybox.next = self.live_allocs;
            if !self.live_allocs.is_null() {
                unsafe { (*self.live_allocs).prev = alloc; }
            }
            self.live_allocs = alloc;
        }
        return alloc;
    }

    #[inline]
    pub fn realloc(&mut self, ptr: *mut Box, size: uint) -> *mut Box {
        // Make sure that we can't use `mybox` outside of this scope
        let total_size = size + mem::size_of::<Box>();
        let new_box = self.memory_region.realloc(ptr, total_size);
        {
            // Fix links because we could have moved around
            let mybox: &mut Box = unsafe { mem::transmute(new_box) };
            if !mybox.prev.is_null() {
                unsafe { (*mybox.prev).next = new_box; }
            }
            if !mybox.next.is_null() {
                unsafe { (*mybox.next).prev = new_box; }
            }
        }
        if self.live_allocs == ptr {
            self.live_allocs = new_box;
        }
        return new_box;
    }

    #[inline]
    pub fn free(&mut self, alloc: *mut Box) {
        {
            // Make sure that we can't use `mybox` outside of this scope
            let mybox: &mut Box = unsafe { mem::transmute(alloc) };

            // Unlink it from the linked list
            if !mybox.prev.is_null() {
                unsafe { (*mybox.prev).next = mybox.next; }
            }
            if !mybox.next.is_null() {
                unsafe { (*mybox.next).prev = mybox.prev; }
            }
            if self.live_allocs == alloc {
                self.live_allocs = mybox.next;
            }
        }

        self.memory_region.free(alloc);
    }

    /// Immortalize all pending allocations, forcing them to live forever.
    ///
    /// This function will freeze all allocations to prevent all pending
    /// allocations from being deallocated. This is used in preparation for when
    /// a task is about to destroy TLD.
    pub unsafe fn immortalize(&mut self) {
        let mut n_total_boxes = 0u;

        // Pass 1: Make all boxes immortal.
        //
        // In this pass, nothing gets freed, so it does not matter whether
        // we read the next field before or after the callback.
        self.each_live_alloc(true, |_, alloc| {
            n_total_boxes += 1;
            (*alloc).ref_count = RC_IMMORTAL;
        });

        if debug_mem() {
            // We do logging here w/o allocation.
            rterrln!("total boxes annihilated: {}", n_total_boxes);
        }
    }

    /// Continues deallocation of the all pending allocations in this arena.
    ///
    /// This is invoked from the destructor, and requires that `immortalize` has
    /// been called previously.
    unsafe fn annihilate(&mut self) {
        // Pass 2: Drop all boxes.
        //
        // In this pass, unique-managed boxes may get freed, but not
        // managed boxes, so we must read the `next` field *after* the
        // callback, as the original value may have been freed.
        self.each_live_alloc(false, |_, alloc| {
            let drop_glue = (*alloc).drop_glue;
            let data = &mut (*alloc).data as *mut ();
            drop_glue(data as *mut u8);
        });

        // Pass 3: Free all boxes.
        //
        // In this pass, managed boxes may get freed (but not
        // unique-managed boxes, though I think that none of those are
        // left), so we must read the `next` field before, since it will
        // not be valid after.
        self.each_live_alloc(true, |me, alloc| {
            me.free(alloc);
        });
    }

    unsafe fn each_live_alloc(&mut self, read_next_before: bool,
                              f: |&mut LocalHeap, alloc: *mut raw::Box<()>|) {
        //! Walks the internal list of allocations

        let mut alloc = self.live_allocs;
        while alloc != ptr::null_mut() {
            let next_before = (*alloc).next;

            f(self, alloc);

            if read_next_before {
                alloc = next_before;
            } else {
                alloc = (*alloc).next;
            }
        }
    }
}

impl Drop for LocalHeap {
    fn drop(&mut self) {
        unsafe { self.annihilate() }
        assert!(self.live_allocs.is_null());
    }
}

struct AllocHeader;

impl AllocHeader {
    fn init(&mut self, _size: u32) {}
    fn assert_sane(&self) {}
    fn update_size(&mut self, _size: u32) {}

    fn as_box(&mut self) -> *mut Box {
        let myaddr: uint = unsafe { mem::transmute(self) };
        (myaddr + AllocHeader::size()) as *mut Box
    }

    fn size() -> uint {
        // For some platforms, 16 byte alignment is required.
        let ptr_size = 16;
        let header_size = mem::size_of::<AllocHeader>();
        return (header_size + ptr_size - 1) / ptr_size * ptr_size;
    }

    fn from(a_box: *mut Box) -> *mut AllocHeader {
        (a_box as uint - AllocHeader::size()) as *mut AllocHeader
    }
}

#[cfg(unix)]
fn debug_mem() -> bool {
    // FIXME: Need to port the environment struct to newsched
    false
}

#[cfg(windows)]
fn debug_mem() -> bool {
    false
}

impl MemoryRegion {
    #[inline]
    fn malloc(&mut self, size: uint) -> *mut Box {
        let total_size = size + AllocHeader::size();
        let alloc: *mut AllocHeader = unsafe {
            libc_heap::malloc_raw(total_size) as *mut AllocHeader
        };

        let alloc: &mut AllocHeader = unsafe { mem::transmute(alloc) };
        alloc.init(size as u32);
        self.claim(alloc);
        self.live_allocations += 1;

        return alloc.as_box();
    }

    #[inline]
    fn realloc(&mut self, alloc: *mut Box, size: uint) -> *mut Box {
        rtassert!(!alloc.is_null());
        let orig_alloc = AllocHeader::from(alloc);
        unsafe { (*orig_alloc).assert_sane(); }

        let total_size = size + AllocHeader::size();
        let alloc: *mut AllocHeader = unsafe {
            libc_heap::realloc_raw(orig_alloc as *mut u8, total_size) as *mut AllocHeader
        };

        let alloc: &mut AllocHeader = unsafe { mem::transmute(alloc) };
        alloc.assert_sane();
        alloc.update_size(size as u32);
        self.update(alloc, orig_alloc as *mut AllocHeader);
        return alloc.as_box();
    }

    #[inline]
    fn free(&mut self, alloc: *mut Box) {
        rtassert!(!alloc.is_null());
        let alloc = AllocHeader::from(alloc);
        unsafe {
            (*alloc).assert_sane();
            self.release(mem::transmute(alloc));
            rtassert!(self.live_allocations > 0);
            self.live_allocations -= 1;
            free(alloc as *mut c_void)
        }
    }

    #[inline]
    fn claim(&mut self, _alloc: &mut AllocHeader) {}
    #[inline]
    fn release(&mut self, _alloc: &AllocHeader) {}
    #[inline]
    fn update(&mut self, _alloc: &mut AllocHeader, _orig: *mut AllocHeader) {}
}

impl Drop for MemoryRegion {
    fn drop(&mut self) {
        if self.live_allocations != 0 {
            rtabort!("leaked managed memory ({} objects)", self.live_allocations);
        }
    }
}

#[cfg(not(test))]
#[lang="malloc"]
#[inline]
pub unsafe fn local_malloc_(drop_glue: fn(*mut u8), size: uint,
                            align: uint) -> *mut u8 {
    local_malloc(drop_glue, size, align)
}

#[inline]
pub unsafe fn local_malloc(drop_glue: fn(*mut u8), size: uint,
                           align: uint) -> *mut u8 {
    // FIXME: Unsafe borrow for speed. Lame.
    let task: Option<*mut Task> = Local::try_unsafe_borrow();
    match task {
        Some(task) => {
            (*task).heap.alloc(drop_glue, size, align) as *mut u8
        }
        None => rtabort!("local malloc outside of task")
    }
}

#[cfg(not(test))]
#[lang="free"]
#[inline]
pub unsafe fn local_free_(ptr: *mut u8) {
    local_free(ptr)
}

// NB: Calls to free CANNOT be allowed to fail, as throwing an exception from
// inside a landing pad may corrupt the state of the exception handler. If a
// problem occurs, call exit instead.
#[inline]
pub unsafe fn local_free(ptr: *mut u8) {
    // FIXME: Unsafe borrow for speed. Lame.
    let task_ptr: Option<*mut Task> = Local::try_unsafe_borrow();
    match task_ptr {
        Some(task) => {
            (*task).heap.free(ptr as *mut Box)
        }
        None => rtabort!("local free outside of task")
    }
}

#[cfg(test)]
mod bench {
    extern crate test;
    use self::test::Bencher;
    use std::gc::GC;

    #[bench]
    fn alloc_managed_small(b: &mut Bencher) {
        b.iter(|| { box(GC) 10i });
    }

    #[bench]
    fn alloc_managed_big(b: &mut Bencher) {
        b.iter(|| { box(GC) ([10i, ..1000]) });
    }
}
