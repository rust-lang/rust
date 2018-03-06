// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that pointers to extern types can be cast from/to usize,
// despite being !Sized.

#![feature(extern_types)]

extern {
    type A;
}

struct Foo {
    x: u8,
    tail: A,
}

struct Bar<T: ?Sized> {
    x: u8,
    tail: T,
}

#[cfg(target_pointer_width = "32")]
const MAGIC: usize = 0xdeadbeef;
#[cfg(target_pointer_width = "64")]
const MAGIC: usize = 0x12345678deadbeef;

fn main() {
    assert_eq!((MAGIC as *const A) as usize, MAGIC);
    assert_eq!((MAGIC as *const Foo) as usize, MAGIC);
    assert_eq!((MAGIC as *const Bar<A>) as usize, MAGIC);
    assert_eq!((MAGIC as *const Bar<Bar<A>>) as usize, MAGIC);
}
