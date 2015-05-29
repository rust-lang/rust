// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that explicit discriminant sizing inhibits the non-nullable pointer
// optimization in enum layout.

use std::mem::size_of;

#[repr(i8)]
enum Ei8<T> {
    _None,
    _Some(T),
}

#[repr(u8)]
enum Eu8<T> {
    _None,
    _Some(T),
}

#[repr(i16)]
enum Ei16<T> {
    _None,
    _Some(T),
}

#[repr(u16)]
enum Eu16<T> {
    _None,
    _Some(T),
}

#[repr(i32)]
enum Ei32<T> {
    _None,
    _Some(T),
}

#[repr(u32)]
enum Eu32<T> {
    _None,
    _Some(T),
}

#[repr(i64)]
enum Ei64<T> {
    _None,
    _Some(T),
}

#[repr(u64)]
enum Eu64<T> {
    _None,
    _Some(T),
}

#[repr(isize)]
enum Eint<T> {
    _None,
    _Some(T),
}

#[repr(usize)]
enum Euint<T> {
    _None,
    _Some(T),
}

#[repr(C)]
enum EC<T> {
    _None,
    _Some(T),
}

pub fn main() {
    assert_eq!(size_of::<Ei8<()>>(), 1);
    assert_eq!(size_of::<Eu8<()>>(), 1);
    assert_eq!(size_of::<Ei16<()>>(), 2);
    assert_eq!(size_of::<Eu16<()>>(), 2);
    assert_eq!(size_of::<Ei32<()>>(), 4);
    assert_eq!(size_of::<Eu32<()>>(), 4);
    assert_eq!(size_of::<Ei64<()>>(), 8);
    assert_eq!(size_of::<Eu64<()>>(), 8);
    assert_eq!(size_of::<Eint<()>>(), size_of::<isize>());
    assert_eq!(size_of::<Euint<()>>(), size_of::<usize>());

    let ptrsize = size_of::<&i32>();
    assert!(size_of::<Ei8<&i32>>() > ptrsize);
    assert!(size_of::<Eu8<&i32>>() > ptrsize);
    assert!(size_of::<Ei16<&i32>>() > ptrsize);
    assert!(size_of::<Eu16<&i32>>() > ptrsize);
    assert!(size_of::<Ei32<&i32>>() > ptrsize);
    assert!(size_of::<Eu32<&i32>>() > ptrsize);
    assert!(size_of::<Ei64<&i32>>() > ptrsize);
    assert!(size_of::<Eu64<&i32>>() > ptrsize);
    assert!(size_of::<Eint<&i32>>() > ptrsize);
    assert!(size_of::<Euint<&i32>>() > ptrsize);

    assert!(size_of::<EC<&i32>>() > ptrsize);
}
