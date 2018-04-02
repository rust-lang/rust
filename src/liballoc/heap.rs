// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "allocator_api",
            reason = "the precise API and guarantees it provides may be tweaked \
                      slightly, especially to possibly take into account the \
                      types being stored to make room for a future \
                      tracing garbage collector",
            issue = "32838")]

use core::intrinsics::{min_align_of_val, size_of_val};
use core::mem::{self, ManuallyDrop};
use core::ptr::NonNull;
use core::usize;

use core::heap;
pub use core::heap::{AllocErr, CannotReallocInPlace, CollectionAllocErr, Layout};
#[cfg(not(stage0))]
pub use core::heap::{Alloc, Excess};
#[doc(hidden)]
#[cfg(stage0)]
pub mod __core {
    pub use core::*;
}

extern "Rust" {
    #[allocator]
    #[rustc_allocator_nounwind]
    fn __rust_alloc(size: usize, align: usize, err: *mut u8) -> *mut u8;
    #[cold]
    #[rustc_allocator_nounwind]
    fn __rust_oom(err: *const u8) -> !;
    #[rustc_allocator_nounwind]
    fn __rust_dealloc(ptr: *mut u8, size: usize, align: usize);
    #[rustc_allocator_nounwind]
    fn __rust_usable_size(layout: *const u8,
                          min: *mut usize,
                          max: *mut usize);
    #[rustc_allocator_nounwind]
    fn __rust_realloc(ptr: *mut u8,
                      old_size: usize,
                      old_align: usize,
                      new_size: usize,
                      new_align: usize,
                      err: *mut u8) -> *mut u8;
    #[rustc_allocator_nounwind]
    fn __rust_alloc_zeroed(size: usize, align: usize, err: *mut u8) -> *mut u8;
    #[rustc_allocator_nounwind]
    fn __rust_alloc_excess(size: usize,
                           align: usize,
                           excess: *mut usize,
                           err: *mut u8) -> *mut u8;
    #[rustc_allocator_nounwind]
    fn __rust_realloc_excess(ptr: *mut u8,
                             old_size: usize,
                             old_align: usize,
                             new_size: usize,
                             new_align: usize,
                             excess: *mut usize,
                             err: *mut u8) -> *mut u8;
    #[rustc_allocator_nounwind]
    fn __rust_grow_in_place(ptr: *mut u8,
                            old_size: usize,
                            old_align: usize,
                            new_size: usize,
                            new_align: usize) -> u8;
    #[rustc_allocator_nounwind]
    fn __rust_shrink_in_place(ptr: *mut u8,
                              old_size: usize,
                              old_align: usize,
                              new_size: usize,
                              new_align: usize) -> u8;
}

#[derive(Copy, Clone, Default, Debug)]
pub struct Heap;

unsafe impl heap::Alloc for Heap {
    #[inline]
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        let mut err = ManuallyDrop::new(mem::uninitialized::<AllocErr>());
        let ptr = __rust_alloc(layout.size(),
                               layout.align(),
                               &mut *err as *mut AllocErr as *mut u8);
        NonNull::new(ptr).ok_or_else(|| ManuallyDrop::into_inner(err))
    }

    #[inline]
    #[cold]
    fn oom(&mut self, err: AllocErr) -> ! {
        unsafe {
            __rust_oom(&err as *const AllocErr as *const u8)
        }
    }

    #[inline]
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        __rust_dealloc(ptr.as_ptr(), layout.size(), layout.align())
    }

    #[inline]
    fn usable_size(&self, layout: &Layout) -> (usize, usize) {
        let mut min = 0;
        let mut max = 0;
        unsafe {
            __rust_usable_size(layout as *const Layout as *const u8,
                               &mut min,
                               &mut max);
        }
        (min, max)
    }

    #[inline]
    unsafe fn realloc(&mut self,
                      ptr: NonNull<u8>,
                      layout: Layout,
                      new_layout: Layout)
                      -> Result<NonNull<u8>, AllocErr>
    {
        let mut err = ManuallyDrop::new(mem::uninitialized::<AllocErr>());
        let ptr = __rust_realloc(ptr.as_ptr(),
                                 layout.size(),
                                 layout.align(),
                                 new_layout.size(),
                                 new_layout.align(),
                                 &mut *err as *mut AllocErr as *mut u8);
        NonNull::new(ptr).ok_or_else(|| ManuallyDrop::into_inner(err))
    }

    #[inline]
    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        let mut err = ManuallyDrop::new(mem::uninitialized::<AllocErr>());
        let ptr = __rust_alloc_zeroed(layout.size(),
                                      layout.align(),
                                      &mut *err as *mut AllocErr as *mut u8);
        NonNull::new(ptr).ok_or_else(|| ManuallyDrop::into_inner(err))
    }

    #[inline]
    unsafe fn alloc_excess(&mut self, layout: Layout) -> Result<heap::Excess, AllocErr> {
        let mut err = ManuallyDrop::new(mem::uninitialized::<AllocErr>());
        let mut size = 0;
        let ptr = __rust_alloc_excess(layout.size(),
                                      layout.align(),
                                      &mut size,
                                      &mut *err as *mut AllocErr as *mut u8);
        NonNull::new(ptr).map(|p| heap::Excess(p, size))
            .ok_or_else(|| ManuallyDrop::into_inner(err))
    }

    #[inline]
    unsafe fn realloc_excess(&mut self,
                             ptr: NonNull<u8>,
                             layout: Layout,
                             new_layout: Layout) -> Result<heap::Excess, AllocErr> {
        let mut err = ManuallyDrop::new(mem::uninitialized::<AllocErr>());
        let mut size = 0;
        let ptr = __rust_realloc_excess(ptr.as_ptr(),
                                        layout.size(),
                                        layout.align(),
                                        new_layout.size(),
                                        new_layout.align(),
                                        &mut size,
                                        &mut *err as *mut AllocErr as *mut u8);
        NonNull::new(ptr).map(|p| heap::Excess(p, size))
            .ok_or_else(|| ManuallyDrop::into_inner(err))
    }

    #[inline]
    unsafe fn grow_in_place(&mut self,
                            ptr: NonNull<u8>,
                            layout: Layout,
                            new_layout: Layout)
                            -> Result<(), CannotReallocInPlace>
    {
        debug_assert!(new_layout.size() >= layout.size());
        debug_assert!(new_layout.align() == layout.align());
        let ret = __rust_grow_in_place(ptr.as_ptr(),
                                       layout.size(),
                                       layout.align(),
                                       new_layout.size(),
                                       new_layout.align());
        if ret != 0 {
            Ok(())
        } else {
            Err(CannotReallocInPlace)
        }
    }

    #[inline]
    unsafe fn shrink_in_place(&mut self,
                              ptr: NonNull<u8>,
                              layout: Layout,
                              new_layout: Layout) -> Result<(), CannotReallocInPlace> {
        debug_assert!(new_layout.size() <= layout.size());
        debug_assert!(new_layout.align() == layout.align());
        let ret = __rust_shrink_in_place(ptr.as_ptr(),
                                         layout.size(),
                                         layout.align(),
                                         new_layout.size(),
                                         new_layout.align());
        if ret != 0 {
            Ok(())
        } else {
            Err(CannotReallocInPlace)
        }
    }
}

// When building stage0 with an older rustc, #[global_allocator] will
// expand to code using alloc::heap::Alloc and expecting the old API.
// A newer rustc's #[global_allocator] expansion uses core::heap::Alloc
// and the new API. For stage0, we thus expose the old API from this
// module.
#[cfg(stage0)]
#[derive(Debug)]
pub struct Excess(pub *mut u8, pub usize);

#[cfg(stage0)]
pub unsafe trait Alloc: heap::Alloc {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<*mut u8, AllocErr> {
        heap::Alloc::alloc(self, layout).map(|p| p.as_ptr())
    }

    unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
        heap::Alloc::dealloc(self, NonNull::new_unchecked(ptr), layout)
    }

    fn oom(&mut self, e: AllocErr) -> ! {
        heap::Alloc::oom(self, e)
    }

    fn usable_size(&self, layout: &Layout) -> (usize, usize) {
        heap::Alloc::usable_size(self, layout)
    }

    unsafe fn realloc(&mut self,
                      ptr: *mut u8,
                      layout: Layout,
                      new_layout: Layout) -> Result<*mut u8, AllocErr> {
        heap::Alloc::realloc(self, NonNull::new_unchecked(ptr), layout, new_layout)
            .map(|p| p.as_ptr())
    }

    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> Result<*mut u8, AllocErr> {
        heap::Alloc::alloc_zeroed(self, layout).map(|p| p.as_ptr())
    }

    unsafe fn alloc_excess(&mut self, layout: Layout) -> Result<Excess, AllocErr> {
        heap::Alloc::alloc_excess(self, layout)
            .map(|heap::Excess(p, s)| Excess(p.as_ptr(), s))
    }

    unsafe fn realloc_excess(&mut self,
                             ptr: *mut u8,
                             layout: Layout,
                             new_layout: Layout) -> Result<Excess, AllocErr> {
        heap::Alloc::realloc_excess(self, NonNull::new_unchecked(ptr), layout, new_layout)
            .map(|heap::Excess(p, s)| Excess(p.as_ptr(), s))
    }

    unsafe fn grow_in_place(&mut self,
                            ptr: *mut u8,
                            layout: Layout,
                            new_layout: Layout) -> Result<(), CannotReallocInPlace> {
        heap::Alloc::grow_in_place(self, NonNull::new_unchecked(ptr), layout, new_layout)
    }

    unsafe fn shrink_in_place(&mut self,
                              ptr: *mut u8,
                              layout: Layout,
                              new_layout: Layout) -> Result<(), CannotReallocInPlace> {
        heap::Alloc::shrink_in_place(self, NonNull::new_unchecked(ptr), layout, new_layout)
    }
}

#[cfg(stage0)]
unsafe impl<T: heap::Alloc> Alloc for T {}

/// The allocator for unique pointers.
// This function must not unwind. If it does, MIR trans will fail.
#[cfg(not(test))]
#[lang = "exchange_malloc"]
#[inline]
unsafe fn exchange_malloc(size: usize, align: usize) -> *mut u8 {
    if size == 0 {
        align as *mut u8
    } else {
        let layout = Layout::from_size_align_unchecked(size, align);
        heap::Alloc::alloc(&mut Heap, layout).unwrap_or_else(|err| {
            heap::Alloc::oom(&mut Heap, err)
        }).as_ptr()
    }
}

#[cfg_attr(not(test), lang = "box_free")]
#[inline]
pub(crate) unsafe fn box_free<T: ?Sized>(ptr: *mut T) {
    let size = size_of_val(&*ptr);
    let align = min_align_of_val(&*ptr);
    // We do not allocate for Box<T> when T is ZST, so deallocation is also not necessary.
    if size != 0 {
        let layout = Layout::from_size_align_unchecked(size, align);
        heap::Alloc::dealloc(&mut Heap, NonNull::new_unchecked(ptr).cast(), layout);
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use self::test::Bencher;
    use boxed::Box;
    use heap::Heap;
    use core::heap::{Alloc, Layout};

    #[test]
    fn allocate_zeroed() {
        unsafe {
            let layout = Layout::from_size_align(1024, 1).unwrap();
            let ptr = Heap.alloc_zeroed(layout.clone())
                .unwrap_or_else(|e| Heap.oom(e));

            let end = ptr.as_ptr().offset(layout.size() as isize);
            let mut i = ptr.as_ptr();
            while i < end {
                assert_eq!(*i, 0);
                i = i.offset(1);
            }
            Heap.dealloc(ptr, layout);
        }
    }

    #[bench]
    fn alloc_owned_small(b: &mut Bencher) {
        b.iter(|| {
            let _: Box<_> = box 10;
        })
    }
}
