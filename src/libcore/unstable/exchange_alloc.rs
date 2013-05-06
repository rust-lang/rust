// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use sys::{TypeDesc, size_of};
use libc::{c_void, size_t};
use c_malloc = libc::malloc;
use c_free = libc::free;
use managed::raw::{BoxHeaderRepr, BoxRepr};
use cast::transmute;
use unstable::intrinsics::{atomic_xadd,atomic_xsub};
use ptr::null;
use intrinsic::TyDesc;

pub unsafe fn malloc(td: *TypeDesc, size: uint) -> *c_void {
    assert!(td.is_not_null());

    let total_size = get_box_size(size, (*td).align);
    let p = c_malloc(total_size as size_t);
    assert!(p.is_not_null());

    // FIXME #3475: Converting between our two different tydesc types
    let td: *TyDesc = transmute(td);

    let box: &mut BoxRepr = transmute(p);
    box.header.ref_count = -1; // Exchange values not ref counted
    box.header.type_desc = td;
    box.header.prev = null();
    box.header.next = null();

    let exchange_count = &mut *rust_get_exchange_count_ptr();
    atomic_xadd(exchange_count, 1);

    return transmute(box);
}
/**
Thin wrapper around libc::malloc, none of the box header
stuff in exchange_alloc::malloc
*/
pub unsafe fn malloc_raw(size: uint) -> *c_void {
    let p = c_malloc(size as size_t);
    if p.is_null() {
        fail!(~"Failure in malloc_raw: result ptr is null");
    }
    p
}

pub unsafe fn free(ptr: *c_void) {
    let exchange_count = &mut *rust_get_exchange_count_ptr();
    atomic_xsub(exchange_count, 1);

    assert!(ptr.is_not_null());
    c_free(ptr);
}
///Thin wrapper around libc::free, as with exchange_alloc::malloc_raw
pub unsafe fn free_raw(ptr: *c_void) {
    c_free(ptr);
}

fn get_box_size(body_size: uint, body_align: uint) -> uint {
    let header_size = size_of::<BoxHeaderRepr>();
    // FIXME (#2699): This alignment calculation is suspicious. Is it right?
    let total_size = align_to(header_size, body_align) + body_size;
    return total_size;
}

// Rounds |size| to the nearest |alignment|. Invariant: |alignment| is a power
// of two.
fn align_to(size: uint, align: uint) -> uint {
    assert!(align != 0);
    (size + align - 1) & !(align - 1)
}

extern {
    #[rust_stack]
    fn rust_get_exchange_count_ptr() -> *mut int;
}
