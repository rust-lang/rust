// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// test that ordinary fat pointer operations work.

#![feature(rustc_attrs)]

struct Wrapper<T: ?Sized>(u32, T);

struct FatPtrContainer<'a> {
    ptr: &'a [u8]
}

#[rustc_mir]
fn fat_ptr_project(a: &Wrapper<[u8]>) -> &[u8] {
    &a.1
}

#[rustc_mir]
fn fat_ptr_simple(a: &[u8]) -> &[u8] {
    a
}

#[rustc_mir]
fn fat_ptr_via_local(a: &[u8]) -> &[u8] {
    let x = a;
    x
}

#[rustc_mir]
fn fat_ptr_from_struct(s: FatPtrContainer) -> &[u8] {
    s.ptr
}

#[rustc_mir]
fn fat_ptr_to_struct(a: &[u8]) -> FatPtrContainer {
    FatPtrContainer { ptr: a }
}

#[rustc_mir]
fn fat_ptr_store_to<'a>(a: &'a [u8], b: &mut &'a [u8]) {
    *b = a;
}

#[rustc_mir]
fn fat_ptr_constant() -> &'static str {
    "HELLO"
}

fn main() {
    let a = Wrapper(4, [7,6,5]);

    let p = fat_ptr_project(&a);
    let p = fat_ptr_simple(p);
    let p = fat_ptr_via_local(p);
    let p = fat_ptr_from_struct(fat_ptr_to_struct(p));

    let mut target : &[u8] = &[42];
    fat_ptr_store_to(p, &mut target);
    assert_eq!(target, &a.1);

    assert_eq!(fat_ptr_constant(), "HELLO");
}
