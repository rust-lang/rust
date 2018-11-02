// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass
#![allow(dead_code)]

// Test that extern types can be used as fields within a transparent struct. See issue #55541.

#![feature(const_transmute, extern_types)]

extern {
    type A;
}
unsafe impl Sync for A {}

#[repr(transparent)]
struct Foo(A);

#[repr(transparent)]
struct Bar(std::marker::PhantomData<u64>, A);

static FOO: &'static Foo = {
    static VALUE: usize = b'F' as usize;
    unsafe { std::mem::transmute(&VALUE) }
};

static BAR: &'static Bar = {
    static VALUE: usize = b'B' as usize;
    unsafe { std::mem::transmute(&VALUE) }
};

fn main() {}
