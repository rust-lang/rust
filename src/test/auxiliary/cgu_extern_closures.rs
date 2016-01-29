// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "lib"]

#[inline]
pub fn inlined_fn(x: i32, y: i32) -> i32 {

    let closure = |a, b| { a + b };

    closure(x, y)
}

pub fn inlined_fn_generic<T>(x: i32, y: i32, z: T) -> (i32, T) {

    let closure = |a, b| { a + b };

    (closure(x, y), z)
}

pub fn non_inlined_fn(x: i32, y: i32) -> i32 {

    let closure = |a, b| { a + b };

    closure(x, y)
}
