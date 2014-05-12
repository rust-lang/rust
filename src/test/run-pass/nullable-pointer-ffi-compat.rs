// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// #11303, #11040:
// This would previously crash on i686 linux due to abi differences
// between returning an Option<T> and T, where T is a non nullable
// pointer.
// If we have an enum with two variants such that one is zero sized
// and the other contains a nonnullable pointer, we don't use a
// separate discriminant. Instead we use that pointer field to differentiate
// between the 2 cases.
// Also, if the variant with the nonnullable pointer has no other fields
// then we simply express the enum as just a pointer and not wrap it
// in a struct.

use std::mem;

#[inline(never)]
extern "C" fn foo<'a>(x: &'a int) -> Option<&'a int> { Some(x) }

static FOO: int = 0xDEADBEE;

pub fn main() {
    unsafe {
        let f: extern "C" fn<'a>(&'a int) -> &'a int = mem::transmute(foo);
        assert_eq!(*f(&FOO), FOO);
    }
}
