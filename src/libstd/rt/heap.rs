// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: #13994: port to the sized deallocation API when available
// FIXME: #13996: need a way to mark the `allocate` and `reallocate` return values as `noalias`

use intrinsics::{abort, cttz32};
use libc::{c_int, c_void, size_t};
use ptr::RawPtr;

#[link(name = "jemalloc", kind = "static")]
extern {
    fn je_mallocx(size: size_t, flags: c_int) -> *mut c_void;
    fn je_rallocx(ptr: *mut c_void, size: size_t, flags: c_int) -> *mut c_void;
    fn je_xallocx(ptr: *mut c_void, size: size_t, extra: size_t, flags: c_int) -> size_t;
    fn je_dallocx(ptr: *mut c_void, flags: c_int);
    fn je_nallocx(size: size_t, flags: c_int) -> size_t;
}

// -lpthread needs to occur after -ljemalloc, the earlier argument isn't enough
#[cfg(not(windows), not(target_os = "android"))]
#[link(name = "pthread")]
extern {}

// MALLOCX_ALIGN(a) macro
#[inline(always)]
fn mallocx_align(a: uint) -> c_int { unsafe { cttz32(a as u32) as c_int } }

/// Return a pointer to `size` bytes of memory.
///
/// Behavior is undefined if the requested size is 0 or the alignment is not a power of 2. The
/// alignment must be no larger than the largest supported page size on the platform.
#[inline]
pub unsafe fn allocate(size: uint, align: uint) -> *mut u8 {
    let ptr = je_mallocx(size as size_t, mallocx_align(align)) as *mut u8;
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
    let ptr = je_rallocx(ptr as *mut c_void, size as size_t, mallocx_align(align)) as *mut u8;
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
    je_xallocx(ptr as *mut c_void, size as size_t, 0, mallocx_align(align)) == size as size_t
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
    je_dallocx(ptr as *mut c_void, mallocx_align(align))
}

/// Return the usable size of an allocation created with the specified the `size` and `align`.
#[inline]
pub fn usable_size(size: uint, align: uint) -> uint {
    unsafe { je_nallocx(size as size_t, mallocx_align(align)) as uint }
}

/// The allocator for unique pointers.
#[cfg(stage0)]
#[lang="exchange_malloc"]
#[inline(always)]
pub unsafe fn exchange_malloc_(size: uint) -> *mut u8 {
    exchange_malloc(size)
}

/// The allocator for unique pointers.
#[cfg(not(test), not(stage0))]
#[lang="exchange_malloc"]
#[inline(always)]
pub unsafe fn exchange_malloc_(size: uint, align: uint) -> *mut u8 {
    exchange_malloc(size, align)
}

/// The allocator for unique pointers.
#[cfg(stage0)]
#[inline]
pub unsafe fn exchange_malloc(size: uint) -> *mut u8 {
    // The compiler never calls `exchange_free` on ~ZeroSizeType, so zero-size
    // allocations can point to this `static`. It would be incorrect to use a null
    // pointer, due to enums assuming types like unique pointers are never null.
    static EMPTY: () = ();

    if size == 0 {
        &EMPTY as *() as *mut u8
    } else {
        allocate(size, 8)
    }
}

/// The allocator for unique pointers.
#[cfg(not(stage0))]
#[inline]
pub unsafe fn exchange_malloc(size: uint, align: uint) -> *mut u8 {
    // The compiler never calls `exchange_free` on ~ZeroSizeType, so zero-size
    // allocations can point to this `static`. It would be incorrect to use a null
    // pointer, due to enums assuming types like unique pointers are never null.
    static EMPTY: () = ();

    if size == 0 {
        &EMPTY as *() as *mut u8
    } else {
        allocate(size, align)
    }
}

#[cfg(not(test))]
#[lang="exchange_free"]
#[inline]
// FIXME: #13994 (rustc should pass align and size here)
pub unsafe fn exchange_free_(ptr: *mut u8) {
    exchange_free(ptr, 0, 8)
}

#[inline]
pub unsafe fn exchange_free(ptr: *mut u8, size: uint, align: uint) {
    deallocate(ptr, size, align);
}

// FIXME: #7496
#[cfg(not(test))]
#[lang="closure_exchange_malloc"]
#[inline]
unsafe fn closure_exchange_malloc(drop_glue: fn(*mut u8), size: uint, align: uint) -> *mut u8 {
    let total_size = ::rt::util::get_box_size(size, align);
    let p = allocate(total_size, 8);

    let alloc = p as *mut ::raw::Box<()>;
    (*alloc).drop_glue = drop_glue;

    alloc as *mut u8
}

// hack for libcore
#[no_mangle]
#[doc(hidden)]
#[deprecated]
#[cfg(stage0, not(test))]
pub extern "C" fn rust_malloc(size: uint) -> *mut u8 {
    unsafe { exchange_malloc(size) }
}

// hack for libcore
#[no_mangle]
#[doc(hidden)]
#[deprecated]
#[cfg(not(stage0), not(test))]
pub extern "C" fn rust_malloc(size: uint, align: uint) -> *mut u8 {
    unsafe { exchange_malloc(size, align) }
}

// hack for libcore
#[no_mangle]
#[doc(hidden)]
#[deprecated]
#[cfg(not(test))]
pub extern "C" fn rust_free(ptr: *mut u8, size: uint, align: uint) {
    unsafe { exchange_free(ptr, size, align) }
}

#[cfg(test)]
mod bench {
    extern crate test;
    use self::test::Bencher;

    #[bench]
    fn alloc_owned_small(b: &mut Bencher) {
        b.iter(|| {
            box 10
        })
    }

    #[bench]
    fn alloc_owned_big(b: &mut Bencher) {
        b.iter(|| {
            box [10, ..1000]
        })
    }
}
