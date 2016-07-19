// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -O
// Regression test for https://github.com/rust-lang/rust/issues/33868

#[repr(C)]
pub struct S {
    a: u32,
    b: f32,
    c: u32
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn test(s: S) -> u32 {
    s.c
}

fn main() {
    assert_eq!(test(S{a: 0, b: 0.0, c: 42}), 42);
}
