// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z print-type-sizes
// compile-pass

// This test makes sure that the tag is not grown for `repr(C)` or `repr(u8)`
// variants (see https://github.com/rust-lang/rust/issues/50098 for the original bug).

#![feature(start)]
#![allow(dead_code)]

#[repr(C, u8)]
enum ReprCu8 {
    A(u16),
    B,
}

#[repr(u8)]
enum Repru8 {
    A(u16),
    B,
}

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    0
}
