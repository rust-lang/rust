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
use core::usize;

pub use allocator::*;
#[doc(hidden)]
pub mod __core {
    pub use core::*;
}

extern "Rust" {
    #[allocator]
    fn __rust_alloc(size: usize, align: usize, err: *mut u8) -> *mut u8;
    fn __rust_oom(err: *const u8) -> !;
    fn __rust_dealloc(ptr: *mut u8, size: usize, align: usize);
    fn __rust_usable_size(layout: *const u8,
                          min: *mut usize,
                          max: *mut usize);
    fn __rust_realloc(ptr: *mut u8,
                      old_size: usize,
                      old_align: usize,
                      new_size: usize,
                      new_align: usize,
                      err: *mut u8) -> *mut u8;
    fn __rust_alloc_zeroed(size: usize, align: usize, err: *mut u8) -> *mut u8;
    fn __rust_alloc_excess(size: usize,
                           align: usize,
                           excess: *mut usize,
                           err: *mut u8) -> *mut u8;
    fn __rust_realloc_excess(ptr: *mut u8,
                             old_size: usize,
                             old_align: usize,
                             new_size: usize,
                             new_align: usize,
                             excess: *mut usize,
                             err: *mut u8) -> *mut u8;
    fn __rust_grow_in_place(ptr: *mut u8,
                            old_size: usize,
                            old_align: usize,
                            new_size: usize,
                            new_align: usize) -> u8;
    fn __rust_shrink_in_place(ptr: *mut u8,
                              old_size: usize,
                              old_align: usize,
                              new_size: usize,
                              new_align: usize) -> u8;
}

#[derive(Copy, Clone, Default, Debug)]
pub struct Heap;

unsafe impl Alloc for Heap {
    #[inline]
    unsafe fn alloc(&mut self, layout: Layout) -> Result<*mut u8, AllocErr> {
        let mut err = ManuallyDrop::new(mem::uninitialized::<AllocErr>());
        let ptr = __rust_alloc(layout.size(),
                               layout.align(),
                               &mut *err as *mut AllocErr as *mut u8);
        if ptr.is_null() {
            Err(ManuallyDrop::into_inner(err))
        } else {
            Ok(ptr)
        }
    }

    #[inline]
    fn oom(&mut self, err: AllocErr) -> ! {
        unsafe {
            __rust_oom(&err as *const AllocErr as *const u8)
        }
    }

    #[inline]
    unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
        __rust_dealloc(ptr, layout.size(), layout.align())
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
                      ptr: *mut u8,
                      layout: Layout,
                      new_layout: Layout)
                      -> Result<*mut u8, AllocErr>
    {
        let mut err = ManuallyDrop::new(mem::uninitialized::<AllocErr>());
        let ptr = __rust_realloc(ptr,
                                 layout.size(),
                                 layout.align(),
                                 new_layout.size(),
                                 new_layout.align(),
                                 &mut *err as *mut AllocErr as *mut u8);
        if ptr.is_null() {
            Err(ManuallyDrop::into_inner(err))
        } else {
            mem::forget(err);
            Ok(ptr)
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> Result<*mut u8, AllocErr> {
        let mut err = ManuallyDrop::new(mem::uninitialized::<AllocErr>());
        let ptr = __rust_alloc_zeroed(layout.size(),
                                      layout.align(),
                                      &mut *err as *mut AllocErr as *mut u8);
        if ptr.is_null() {
            Err(ManuallyDrop::into_inner(err))
        } else {
            Ok(ptr)
        }
    }

    #[inline]
    unsafe fn alloc_excess(&mut self, layout: Layout) -> Result<Excess, AllocErr> {
        let mut err = ManuallyDrop::new(mem::uninitialized::<AllocErr>());
        let mut size = 0;
        let ptr = __rust_alloc_excess(layout.size(),
                                      layout.align(),
                                      &mut size,
                                      &mut *err as *mut AllocErr as *mut u8);
        if ptr.is_null() {
            Err(ManuallyDrop::into_inner(err))
        } else {
            Ok(Excess(ptr, size))
        }
    }

    #[inline]
    unsafe fn realloc_excess(&mut self,
                             ptr: *mut u8,
                             layout: Layout,
                             new_layout: Layout) -> Result<Excess, AllocErr> {
        let mut err = ManuallyDrop::new(mem::uninitialized::<AllocErr>());
        let mut size = 0;
        let ptr = __rust_realloc_excess(ptr,
                                        layout.size(),
                                        layout.align(),
                                        new_layout.size(),
                                        new_layout.align(),
                                        &mut size,
                                        &mut *err as *mut AllocErr as *mut u8);
        if ptr.is_null() {
            Err(ManuallyDrop::into_inner(err))
        } else {
            Ok(Excess(ptr, size))
        }
    }

    #[inline]
    unsafe fn grow_in_place(&mut self,
                            ptr: *mut u8,
                            layout: Layout,
                            new_layout: Layout)
                            -> Result<(), CannotReallocInPlace>
    {
        debug_assert!(new_layout.size() >= layout.size());
        debug_assert!(new_layout.align() == layout.align());
        let ret = __rust_grow_in_place(ptr,
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
                              ptr: *mut u8,
                              layout: Layout,
                              new_layout: Layout) -> Result<(), CannotReallocInPlace> {
        debug_assert!(new_layout.size() <= layout.size());
        debug_assert!(new_layout.align() == layout.align());
        let ret = __rust_shrink_in_place(ptr,
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

/// An arbitrary non-null address to represent zero-size allocations.
///
/// This preserves the non-null invariant for types like `Box<T>`. The address
/// may overlap with non-zero-size memory allocations.
#[rustc_deprecated(since = "1.19", reason = "Use Unique/Shared::empty() instead")]
#[unstable(feature = "heap_api", issue = "27700")]
pub const EMPTY: *mut () = 1 as *mut ();

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
        Heap.alloc(layout).unwrap_or_else(|err| {
            Heap.oom(err)
        })
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
        Heap.dealloc(ptr as *mut u8, layout);
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use self::test::Bencher;
    use boxed::Box;
    use heap::{Heap, Alloc, Layout};

    #[test]
    fn allocate_zeroed() {
        unsafe {
            let layout = Layout::from_size_align(1024, 1).unwrap();
            let ptr = Heap.alloc_zeroed(layout.clone())
                .unwrap_or_else(|e| Heap.oom(e));

            let end = ptr.offset(layout.size() as isize);
            let mut i = ptr;
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
