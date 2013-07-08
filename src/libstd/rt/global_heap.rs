// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc::{c_char, c_void, size_t, uintptr_t, free, malloc, realloc};
use managed::raw::{BoxHeaderRepr, BoxRepr};
use unstable::intrinsics::TyDesc;
use sys::size_of;

extern {
    #[rust_stack]
    fn abort();
}

#[inline]
fn get_box_size(body_size: uint, body_align: uint) -> uint {
    let header_size = size_of::<BoxHeaderRepr>();
    // FIXME (#2699): This alignment calculation is suspicious. Is it right?
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
pub unsafe fn malloc_raw(size: uint) -> *c_void {
    let p = malloc(size as size_t);
    if p.is_null() {
        // we need a non-allocating way to print an error here
        abort();
    }
    p
}

/// A wrapper around libc::realloc, aborting on out-of-memory
#[inline]
pub unsafe fn realloc_raw(ptr: *mut c_void, size: uint) -> *mut c_void {
    let p = realloc(ptr, size as size_t);
    if p.is_null() {
        // we need a non-allocating way to print an error here
        abort();
    }
    p
}

// FIXME #4942: Make these signatures agree with exchange_alloc's signatures
#[cfg(stage0, not(test))]
#[lang="exchange_malloc"]
#[inline]
pub unsafe fn exchange_malloc(td: *c_char, size: uintptr_t) -> *c_char {
    let td = td as *TyDesc;
    let size = size as uint;

    assert!(td.is_not_null());

    let total_size = get_box_size(size, (*td).align);
    let p = malloc_raw(total_size as uint);

    let box: *mut BoxRepr = p as *mut BoxRepr;
    (*box).header.ref_count = -1;
    (*box).header.type_desc = td;

    box as *c_char
}

// FIXME #4942: Make these signatures agree with exchange_alloc's signatures
#[cfg(not(stage0), not(test))]
#[lang="exchange_malloc"]
#[inline]
pub unsafe fn exchange_malloc(_align: u32, size: uintptr_t) -> *c_char {
    malloc_raw(size as uint) as *c_char
}

#[cfg(not(test))]
#[lang="vector_exchange_malloc"]
#[inline]
pub unsafe fn vector_exchange_malloc(align: u32, size: uintptr_t) -> *c_char {
    let total_size = get_box_size(size as uint, align as uint);
    malloc_raw(total_size as uint) as *c_char
}

// FIXME: #7496
#[cfg(not(test))]
#[lang="closure_exchange_malloc"]
#[inline]
pub unsafe fn closure_exchange_malloc(td: *c_char, size: uintptr_t) -> *c_char {
    let td = td as *TyDesc;
    let size = size as uint;

    assert!(td.is_not_null());

    let total_size = get_box_size(size, (*td).align);
    let p = malloc_raw(total_size as uint);

    let box: *mut BoxRepr = p as *mut BoxRepr;
    (*box).header.type_desc = td;

    box as *c_char
}

// NB: Calls to free CANNOT be allowed to fail, as throwing an exception from
// inside a landing pad may corrupt the state of the exception handler.
#[cfg(not(test))]
#[lang="exchange_free"]
#[inline]
pub unsafe fn exchange_free(ptr: *c_char) {
    free(ptr as *c_void);
}
