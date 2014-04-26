// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use intrinsics::{abort, cttz32};
use libc::{c_int, c_void, size_t};
use ptr::RawPtr;

#[link(name = "jemalloc", kind = "static")]
#[link(name = "pthread")]
extern {
    fn jemallocx(size: size_t, flags: c_int) -> *mut c_void;
    fn jerallocx(ptr: *mut c_void, size: size_t, flags: c_int) -> *mut c_void;
    fn jexallocx(ptr: *mut c_void, size: size_t, extra: size_t, flags: c_int) -> size_t;
    fn jedallocx(ptr: *mut c_void, flags: c_int);
    fn jenallocx(size: size_t, flags: c_int) -> size_t;
}

// MALLOCX_ALIGN(a) macro
#[inline(always)]
fn mallocx_align(a: uint) -> c_int { unsafe { cttz32(a as u32) as c_int } }

/// Return a pointer to `size` bytes of memory.
///
/// Behavior is undefined if the requested size is 0 or the alignment is not a power of 2. The
/// alignment must be no larger than the largest supported page size on the platform.
#[inline]
pub unsafe fn allocate(size: uint, align: uint) -> *mut u8 {
    let ptr = jemallocx(size as size_t, mallocx_align(align)) as *mut u8;
    if ptr.is_null() {
        abort()
    }
    ptr
}

/// Extend or shrink the allocation referenced by `ptr` to `size` bytes of memory.
///
/// Behavior is undefined if the requested size is 0 or the alignment is not a power of 2. The
/// alignment must be no larger than the largest supported page size on the platform.
///
/// The `old_size` and `align` parameters are the parameters that were used to create the
/// allocation referenced by `ptr`. The `old_size` parameter may also be the value returned by
/// `usable_size` for the requested size.
#[inline]
#[allow(unused_variable)] // for the parameter names in the documentation
pub unsafe fn reallocate(ptr: *mut u8, size: uint, align: uint, old_size: uint) -> *mut u8 {
    let ptr = jerallocx(ptr as *mut c_void, size as size_t, mallocx_align(align)) as *mut u8;
    if ptr.is_null() {
        abort()
    }
    ptr
}

/// Extend or shrink the allocation referenced by `ptr` to `size` bytes of memory in-place.
///
/// Return true if successful, otherwise false if the allocation was not altered.
///
/// Behavior is undefined if the requested size is 0 or the alignment is not a power of 2. The
/// alignment must be no larger than the largest supported page size on the platform.
///
/// The `old_size` and `align` parameters are the parameters that were used to
/// create the allocation referenced by `ptr`. The `old_size` parameter may be
/// any value in range_inclusive(requested_size, usable_size).
#[inline]
#[allow(unused_variable)] // for the parameter names in the documentation
pub unsafe fn reallocate_inplace(ptr: *mut u8, size: uint, align: uint, old_size: uint) -> bool {
    jexallocx(ptr as *mut c_void, size as size_t, 0, mallocx_align(align)) == size as size_t
}

/// Deallocate the memory referenced by `ptr`.
///
/// The `ptr` parameter must not be null.
///
/// The `size` and `align` parameters are the parameters that were used to create the
/// allocation referenced by `ptr`. The `size` parameter may also be the value returned by
/// `usable_size` for the requested size.
#[inline]
#[allow(unused_variable)] // for the parameter names in the documentation
pub unsafe fn deallocate(ptr: *mut u8, size: uint, align: uint) {
    jedallocx(ptr as *mut c_void, mallocx_align(align))
}

/// Return the usable size of an allocation created with the specified the `size` and `align`.
#[inline]
pub fn usable_size(size: uint, align: uint) -> uint {
    unsafe { jenallocx(size as size_t, mallocx_align(align)) as uint }
}
