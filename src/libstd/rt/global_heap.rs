// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


//! The global (exchange) heap.

use libc::{c_void, size_t, free, malloc, realloc};
use ptr::{RawPtr, mut_null};
use intrinsics::abort;
use raw;
use mem::size_of;

#[inline]
pub fn get_box_size(body_size: uint, body_align: uint) -> uint {
    let header_size = size_of::<raw::Box<()>>();
    let total_size = align_to(header_size, body_align) + body_size;
    total_size
}

// Rounds |size| to the nearest |alignment|. Invariant: |alignment| is a power
// of two.
#[inline]
fn align_to(size: uint, align: uint) -> uint {
    assert!(align != 0);
    (size + align - 1) & !(align - 1)
}

/// A wrapper around libc::malloc, aborting on out-of-memory
#[inline]
pub unsafe fn malloc_raw(size: uint) -> *mut u8 {
    // `malloc(0)` may allocate, but it may also return a null pointer
    // http://pubs.opengroup.org/onlinepubs/9699919799/functions/malloc.html
    if size == 0 {
        mut_null()
    } else {
        let p = malloc(size as size_t);
        if p.is_null() {
            // we need a non-allocating way to print an error here
            abort();
        }
        p as *mut u8
    }
}

/// A wrapper around libc::realloc, aborting on out-of-memory
#[inline]
pub unsafe fn realloc_raw(ptr: *mut u8, size: uint) -> *mut u8 {
    // `realloc(ptr, 0)` may allocate, but it may also return a null pointer
    // http://pubs.opengroup.org/onlinepubs/9699919799/functions/realloc.html
    if size == 0 {
        free(ptr as *mut c_void);
        mut_null()
    } else {
        let p = realloc(ptr as *mut c_void, size as size_t);
        if p.is_null() {
            // we need a non-allocating way to print an error here
            abort();
        }
        p as *mut u8
    }
}

/// The allocator for unique pointers without contained managed pointers.
#[cfg(not(test))]
#[lang="exchange_malloc"]
#[inline]
pub unsafe fn exchange_malloc(size: uint) -> *mut u8 {
    // The compiler never calls `exchange_free` on ~ZeroSizeType, so zero-size
    // allocations can point to this `static`. It would be incorrect to use a null
    // pointer, due to enums assuming types like unique pointers are never null.
    static EMPTY: () = ();

    if size == 0 {
        &EMPTY as *() as *mut u8
    } else {
        malloc_raw(size)
    }
}

// FIXME: #7496
#[cfg(not(test))]
#[lang="closure_exchange_malloc"]
#[inline]
pub unsafe fn closure_exchange_malloc_(drop_glue: fn(*mut u8), size: uint, align: uint) -> *u8 {
    closure_exchange_malloc(drop_glue, size, align)
}

#[inline]
pub unsafe fn closure_exchange_malloc(drop_glue: fn(*mut u8), size: uint, align: uint) -> *u8 {
    let total_size = get_box_size(size, align);
    let p = malloc_raw(total_size);

    let alloc = p as *mut raw::Box<()>;
    (*alloc).drop_glue = drop_glue;

    alloc as *u8
}

// NB: Calls to free CANNOT be allowed to fail, as throwing an exception from
// inside a landing pad may corrupt the state of the exception handler.
#[cfg(not(test))]
#[lang="exchange_free"]
#[inline]
pub unsafe fn exchange_free_(ptr: *u8) {
    exchange_free(ptr)
}

#[inline]
pub unsafe fn exchange_free(ptr: *u8) {
    free(ptr as *mut c_void);
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
