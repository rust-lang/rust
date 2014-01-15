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

use cast;
use iter::Iterator;
use libc::{c_void, uintptr_t};
use libc;
use mem;
use ops::Drop;
use option::{Option, None, Some};
use ptr;
use ptr::RawPtr;
use rt::env;
use rt::global_heap;
use rt::local::Local;
use rt::task::Task;
use unstable::intrinsics::TyDesc;
use unstable::raw;
use vec::ImmutableVector;

// This has no meaning with out rtdebug also turned on.
#[cfg(rtdebug)]
static TRACK_ALLOCATIONS: int = 0;
#[cfg(rtdebug)]
static MAGIC: u32 = 0xbadc0ffe;

pub type Box = raw::Box<()>;

pub struct MemoryRegion {
    priv allocations: ~[*AllocHeader],
    priv live_allocations: uint,
}

pub struct LocalHeap {
    priv memory_region: MemoryRegion,

    priv poison_on_free: bool,
    priv live_allocs: *mut raw::Box<()>,
}

impl LocalHeap {
    #[inline]
    pub fn new() -> LocalHeap {
        let region = MemoryRegion {
            allocations: ~[],
            live_allocations: 0,
        };
        LocalHeap {
            memory_region: region,
            poison_on_free: env::poison_on_free(),
            live_allocs: ptr::mut_null(),
        }
    }

    #[inline]
    pub fn alloc(&mut self, td: *TyDesc, size: uint) -> *mut Box {
        let total_size = global_heap::get_box_size(size, unsafe { (*td).align });
        let alloc = self.memory_region.malloc(total_size);
        {
            // Make sure that we can't use `mybox` outside of this scope
            let mybox: &mut Box = unsafe { cast::transmute(alloc) };
            // Clear out this box, and move it to the front of the live
            // allocations list
            mybox.type_desc = td;
            mybox.ref_count = 1;
            mybox.prev = ptr::mut_null();
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
            let mybox: &mut Box = unsafe { cast::transmute(new_box) };
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
            let mybox: &mut Box = unsafe { cast::transmute(alloc) };
            assert!(!mybox.type_desc.is_null());

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

            // Destroy the box memory-wise
            if self.poison_on_free {
                unsafe {
                    let ptr: *mut u8 = cast::transmute(&mybox.data);
                    ptr::set_memory(ptr, 0xab, (*mybox.type_desc).size);
                }
            }
            mybox.prev = ptr::mut_null();
            mybox.next = ptr::mut_null();
            mybox.type_desc = ptr::null();
        }

        self.memory_region.free(alloc);
    }
}

impl Drop for LocalHeap {
    fn drop(&mut self) {
        assert!(self.live_allocs.is_null());
    }
}

#[cfg(rtdebug)]
struct AllocHeader {
    magic: u32,
    index: i32,
    size: u32,
}
#[cfg(not(rtdebug))]
struct AllocHeader;

impl AllocHeader {
    #[cfg(rtdebug)]
    fn init(&mut self, size: u32) {
        if TRACK_ALLOCATIONS > 0 {
            self.magic = MAGIC;
            self.index = -1;
            self.size = size;
        }
    }
    #[cfg(not(rtdebug))]
    fn init(&mut self, _size: u32) {}

    #[cfg(rtdebug)]
    fn assert_sane(&self) {
        if TRACK_ALLOCATIONS > 0 {
            rtassert!(self.magic == MAGIC);
        }
    }
    #[cfg(not(rtdebug))]
    fn assert_sane(&self) {}

    #[cfg(rtdebug)]
    fn update_size(&mut self, size: u32) {
        if TRACK_ALLOCATIONS > 0 {
            self.size = size;
        }
    }
    #[cfg(not(rtdebug))]
    fn update_size(&mut self, _size: u32) {}

    fn as_box(&mut self) -> *mut Box {
        let myaddr: uint = unsafe { cast::transmute(self) };
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

impl MemoryRegion {
    #[inline]
    fn malloc(&mut self, size: uint) -> *mut Box {
        let total_size = size + AllocHeader::size();
        let alloc: *AllocHeader = unsafe {
            global_heap::malloc_raw(total_size) as *AllocHeader
        };

        let alloc: &mut AllocHeader = unsafe { cast::transmute(alloc) };
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
        let alloc: *AllocHeader = unsafe {
            global_heap::realloc_raw(orig_alloc as *mut libc::c_void,
                                     total_size) as *AllocHeader
        };

        let alloc: &mut AllocHeader = unsafe { cast::transmute(alloc) };
        alloc.assert_sane();
        alloc.update_size(size as u32);
        self.update(alloc, orig_alloc as *AllocHeader);
        return alloc.as_box();
    }

    #[inline]
    fn free(&mut self, alloc: *mut Box) {
        rtassert!(!alloc.is_null());
        let alloc = AllocHeader::from(alloc);
        unsafe {
            (*alloc).assert_sane();
            self.release(cast::transmute(alloc));
            rtassert!(self.live_allocations > 0);
            self.live_allocations -= 1;
            global_heap::exchange_free(alloc as *libc::c_char)
        }
    }

    #[cfg(rtdebug)]
    fn claim(&mut self, alloc: &mut AllocHeader) {
        alloc.assert_sane();
        if TRACK_ALLOCATIONS > 1 {
            alloc.index = self.allocations.len() as i32;
            self.allocations.push(&*alloc as *AllocHeader);
        }
    }
    #[cfg(not(rtdebug))]
    #[inline]
    fn claim(&mut self, _alloc: &mut AllocHeader) {}

    #[cfg(rtdebug)]
    fn release(&mut self, alloc: &AllocHeader) {
        alloc.assert_sane();
        if TRACK_ALLOCATIONS > 1 {
            rtassert!(self.allocations[alloc.index] == alloc as *AllocHeader);
            self.allocations[alloc.index] = ptr::null();
        }
    }
    #[cfg(not(rtdebug))]
    #[inline]
    fn release(&mut self, _alloc: &AllocHeader) {}

    #[cfg(rtdebug)]
    fn update(&mut self, alloc: &mut AllocHeader, orig: *AllocHeader) {
        alloc.assert_sane();
        if TRACK_ALLOCATIONS > 1 {
            rtassert!(self.allocations[alloc.index] == orig);
            self.allocations[alloc.index] = &*alloc as *AllocHeader;
        }
    }
    #[cfg(not(rtdebug))]
    #[inline]
    fn update(&mut self, _alloc: &mut AllocHeader, _orig: *AllocHeader) {}
}

impl Drop for MemoryRegion {
    fn drop(&mut self) {
        if self.live_allocations != 0 {
            rtabort!("leaked managed memory ({} objects)", self.live_allocations);
        }
        rtassert!(self.allocations.iter().all(|s| s.is_null()));
    }
}

#[inline]
pub unsafe fn local_malloc(td: *libc::c_char, size: libc::uintptr_t) -> *libc::c_char {
    // XXX: Unsafe borrow for speed. Lame.
    let task: Option<*mut Task> = Local::try_unsafe_borrow();
    match task {
        Some(task) => {
            (*task).heap.alloc(td as *TyDesc, size as uint) as *libc::c_char
        }
        None => rtabort!("local malloc outside of task")
    }
}

// A little compatibility function
#[inline]
pub unsafe fn local_free(ptr: *libc::c_char) {
    // XXX: Unsafe borrow for speed. Lame.
    let task_ptr: Option<*mut Task> = Local::try_unsafe_borrow();
    match task_ptr {
        Some(task) => {
            (*task).heap.free(ptr as *mut Box)
        }
        None => rtabort!("local free outside of task")
    }
}

pub fn live_allocs() -> *mut Box {
    let mut task = Local::borrow(None::<Task>);
    task.get().heap.live_allocs
}

#[cfg(test)]
mod bench {
    use extra::test::BenchHarness;

    #[bench]
    fn alloc_managed_small(bh: &mut BenchHarness) {
        bh.iter(|| { @10; });
    }

    #[bench]
    fn alloc_managed_big(bh: &mut BenchHarness) {
        bh.iter(|| { @[10, ..1000]; });
    }
}
