// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![forbid(improper_ctypes)]
#![allow(dead_code)]

#[repr(C)]
pub struct Foo {
    size: u8,
    __value: ::std::marker::PhantomData<i32>,
}

#[repr(C)]
pub struct ZeroSizeWithPhantomData<T>(::std::marker::PhantomData<T>);

#[repr(C)]
pub struct Bar {
    size: u8,
    baz: ZeroSizeWithPhantomData<i32>,
}

extern "C" {
    pub fn bar(_: *mut Foo, _: *mut Bar);
}

fn main() {
}
