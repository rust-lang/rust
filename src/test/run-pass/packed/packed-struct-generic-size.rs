// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
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
#![allow(unused_comparisons)]

use std::mem;

#[repr(packed)]
struct P1<T, S> {
    a: T,
    b: u8,
    c: S
}

#[repr(packed(2))]
struct P2<T, S> {
    a: T,
    b: u8,
    c: S
}

#[repr(C, packed(4))]
struct P4C<T, S> {
    a: T,
    b: u8,
    c: S
}

macro_rules! check {
    ($t:ty, $align:expr, $size:expr) => ({
        assert_eq!(mem::align_of::<$t>(), $align);
        assert_eq!(mem::size_of::<$t>(), $size);
    });
}

pub fn main() {
    check!(P1::<u8, u8>, 1, 3);
    check!(P1::<u64, u16>, 1, 11);

    check!(P2::<u8, u8>, 1, 3);
    check!(P2::<u64, u16>, 2, 12);

    check!(P4C::<u8, u8>, 1, 3);
    check!(P4C::<u16, u64>, 4, 12);
}
