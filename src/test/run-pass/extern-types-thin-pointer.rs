// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that pointers and references to extern types are thin, ie they have the same size and
// alignment as a pointer to ().

#![feature(extern_types)]

use std::mem::{align_of, size_of};

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

fn assert_thin<T: ?Sized>() {
    assert_eq!(size_of::<*const T>(), size_of::<*const ()>());
    assert_eq!(align_of::<*const T>(), align_of::<*const ()>());

    assert_eq!(size_of::<*mut T>(), size_of::<*mut ()>());
    assert_eq!(align_of::<*mut T>(), align_of::<*mut ()>());

    assert_eq!(size_of::<&T>(), size_of::<&()>());
    assert_eq!(align_of::<&T>(), align_of::<&()>());

    assert_eq!(size_of::<&mut T>(), size_of::<&mut ()>());
    assert_eq!(align_of::<&mut T>(), align_of::<&mut ()>());
}

fn main() {
    assert_thin::<A>();
    assert_thin::<Foo>();
    assert_thin::<Bar<A>>();
    assert_thin::<Bar<Bar<A>>>();
}
