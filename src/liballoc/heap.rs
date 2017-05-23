// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "heap_api",
            reason = "the precise API and guarantees it provides may be tweaked \
                      slightly, especially to possibly take into account the \
                      types being stored to make room for a future \
                      tracing garbage collector",
            issue = "27700")]

use allocator::{Alloc, AllocErr, CannotReallocInPlace, Layout};
use core::{isize, usize, cmp, ptr};
use core::intrinsics::{min_align_of_val, size_of_val};

#[allow(improper_ctypes)]
extern "C" {
    #[allocator]
    fn __rust_allocate(size: usize, align: usize) -> *mut u8;
    fn __rust_allocate_zeroed(size: usize, align: usize) -> *mut u8;
    fn __rust_deallocate(ptr: *mut u8, old_size: usize, align: usize);
    fn __rust_reallocate(ptr: *mut u8, old_size: usize, size: usize, align: usize) -> *mut u8;
    fn __rust_reallocate_inplace(ptr: *mut u8,
                                 old_size: usize,
                                 size: usize,
                                 align: usize)
                                 -> usize;
    fn __rust_usable_size(size: usize, align: usize) -> usize;
}

#[inline(always)]
fn check_size_and_alignment(size: usize, align: usize) {
    debug_assert!(size != 0);
    debug_assert!(size <= isize::MAX as usize,
                  "Tried to allocate too much: {} bytes",
                  size);
    debug_assert!(usize::is_power_of_two(align),
                  "Invalid alignment of allocation: {}",
                  align);
}

#[derive(Copy, Clone, Default, Debug)]
pub struct HeapAlloc;

unsafe impl Alloc for HeapAlloc {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<*mut u8, AllocErr> {
        let addr = allocate(layout.size(), layout.align());
        if addr.is_null() {
            Err(AllocErr::Exhausted { request: layout })
        } else {
            Ok(addr)
        }
    }

    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> Result<*mut u8, AllocErr> {
        let addr = allocate_zeroed(layout.size(), layout.align());
        if addr.is_null() {
            Err(AllocErr::Exhausted { request: layout })
        } else {
            Ok(addr)
        }
    }

    unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
        deallocate(ptr, layout.size(), layout.align());
    }

    fn usable_size(&self, layout: &Layout) -> (usize, usize) {
        (layout.size(), usable_size(layout.size(), layout.align()))
    }

    unsafe fn realloc(&mut self,
                      ptr: *mut u8,
                      layout: Layout,
                      new_layout: Layout)
                      -> Result<*mut u8, AllocErr>
    {
        let old_size = layout.size();
        let new_size = new_layout.size();
        if layout.align() == new_layout.align() {
            let new_ptr = reallocate(ptr, old_size, new_size, layout.align());
            if new_ptr.is_null() {
                // We assume `reallocate` already tried alloc + copy +
                // dealloc fallback; thus pointless to repeat effort
                Err(AllocErr::Exhausted { request: new_layout })
            } else {
                Ok(new_ptr)
            }
        } else {
            // if alignments don't match, fall back on alloc + copy + dealloc
            let result = self.alloc(new_layout);
            if let Ok(new_ptr) = result {
                ptr::copy_nonoverlapping(ptr as *const u8, new_ptr, cmp::min(old_size, new_size));
                self.dealloc(ptr, layout);
            }
            result
        }
    }

    unsafe fn grow_in_place(&mut self,
                            ptr: *mut u8,
                            layout: Layout,
                            new_layout: Layout)
                            -> Result<(), CannotReallocInPlace>
    {
        // grow_in_place spec requires this, and the spec for reallocate_inplace
        // makes it hard to detect failure if it does not hold.
        debug_assert!(new_layout.size() >= layout.size());

        if layout.align() != new_layout.align() { // reallocate_inplace requires this.
            return Err(CannotReallocInPlace);
        }
        let usable = reallocate_inplace(ptr, layout.size(), new_layout.size(), layout.align());
        if usable >= new_layout.size() { Ok(()) } else { Err(CannotReallocInPlace) }
    }
}

// FIXME: #13996: mark the `allocate` and `reallocate` return value as `noalias`

/// Return a pointer to `size` bytes of memory aligned to `align`.
///
/// On failure, return a null pointer.
///
/// Behavior is undefined if the requested size is 0 or the alignment is not a
/// power of 2. The alignment must be no larger than the largest supported page
/// size on the platform.
#[inline]
pub unsafe fn allocate(size: usize, align: usize) -> *mut u8 {
    check_size_and_alignment(size, align);
    __rust_allocate(size, align)
}

/// Return a pointer to `size` bytes of memory aligned to `align` and
/// initialized to zeroes.
///
/// On failure, return a null pointer.
///
/// Behavior is undefined if the requested size is 0 or the alignment is not a
/// power of 2. The alignment must be no larger than the largest supported page
/// size on the platform.
#[inline]
pub unsafe fn allocate_zeroed(size: usize, align: usize) -> *mut u8 {
    check_size_and_alignment(size, align);
    __rust_allocate_zeroed(size, align)
}

/// Resize the allocation referenced by `ptr` to `size` bytes.
///
/// On failure, return a null pointer and leave the original allocation intact.
///
/// If the allocation was relocated, the memory at the passed-in pointer is
/// undefined after the call.
///
/// Behavior is undefined if the requested size is 0 or the alignment is not a
/// power of 2. The alignment must be no larger than the largest supported page
/// size on the platform.
///
/// The `old_size` and `align` parameters are the parameters that were used to
/// create the allocation referenced by `ptr`. The `old_size` parameter may be
/// any value in range_inclusive(requested_size, usable_size).
#[inline]
pub unsafe fn reallocate(ptr: *mut u8, old_size: usize, size: usize, align: usize) -> *mut u8 {
    check_size_and_alignment(size, align);
    __rust_reallocate(ptr, old_size, size, align)
}

/// Resize the allocation referenced by `ptr` to `size` bytes.
///
/// If the operation succeeds, it returns `usable_size(size, align)` and if it
/// fails (or is a no-op) it returns `usable_size(old_size, align)`.
///
/// Behavior is undefined if the requested size is 0 or the alignment is not a
/// power of 2. The alignment must be no larger than the largest supported page
/// size on the platform.
///
/// The `old_size` and `align` parameters are the parameters that were used to
/// create the allocation referenced by `ptr`. The `old_size` parameter may be
/// any value in range_inclusive(requested_size, usable_size).
#[inline]
pub unsafe fn reallocate_inplace(ptr: *mut u8,
                                 old_size: usize,
                                 size: usize,
                                 align: usize)
                                 -> usize {
    check_size_and_alignment(size, align);
    __rust_reallocate_inplace(ptr, old_size, size, align)
}

/// Deallocates the memory referenced by `ptr`.
///
/// The `ptr` parameter must not be null.
///
/// The `old_size` and `align` parameters are the parameters that were used to
/// create the allocation referenced by `ptr`. The `old_size` parameter may be
/// any value in range_inclusive(requested_size, usable_size).
#[inline]
pub unsafe fn deallocate(ptr: *mut u8, old_size: usize, align: usize) {
    __rust_deallocate(ptr, old_size, align)
}

/// Returns the usable size of an allocation created with the specified the
/// `size` and `align`.
#[inline]
pub fn usable_size(size: usize, align: usize) -> usize {
    unsafe { __rust_usable_size(size, align) }
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
        let ptr = allocate(size, align);
        if ptr.is_null() {
            ::oom()
        }
        ptr
    }
}

#[cfg_attr(not(test), lang = "box_free")]
#[inline]
pub(crate) unsafe fn box_free<T: ?Sized>(ptr: *mut T) {
    let size = size_of_val(&*ptr);
    let align = min_align_of_val(&*ptr);
    // We do not allocate for Box<T> when T is ZST, so deallocation is also not necessary.
    if size != 0 {
        deallocate(ptr as *mut u8, size, align);
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use self::test::Bencher;
    use boxed::Box;
    use heap;

    #[test]
    fn allocate_zeroed() {
        unsafe {
            let size = 1024;
            let ptr = heap::allocate_zeroed(size, 1);
            if ptr.is_null() {
                ::oom()
            }

            let end = ptr.offset(size as isize);
            let mut i = ptr;
            while i < end {
                assert_eq!(*i, 0);
                i = i.offset(1);
            }
            heap::deallocate(ptr, size, 1);
        }
    }

    #[test]
    fn basic_reallocate_inplace_noop() {
        unsafe {
            let size = 4000;
            let ptr = heap::allocate(size, 8);
            if ptr.is_null() {
                ::oom()
            }
            let ret = heap::reallocate_inplace(ptr, size, size, 8);
            heap::deallocate(ptr, size, 8);
            assert_eq!(ret, heap::usable_size(size, 8));
        }
    }

    #[bench]
    fn alloc_owned_small(b: &mut Bencher) {
        b.iter(|| {
            let _: Box<_> = box 10;
        })
    }
}
