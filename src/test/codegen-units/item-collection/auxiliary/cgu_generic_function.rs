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

struct Struct(u32);

#[inline(never)]
pub fn foo<T>(x: T) -> (T, u32, i8) {
    let (x, Struct(y)) = bar(x);
    (x, y, 2)
}

#[inline(never)]
fn bar<T>(x: T) -> (T, Struct) {
    let _ = not_exported_and_not_generic(0);
    (x, Struct(1))
}

// These should not contribute to the codegen items of other crates.
#[inline(never)]
pub fn exported_but_not_generic(x: i32) -> i64 {
    x as i64
}

#[inline(never)]
fn not_exported_and_not_generic(x: u32) -> u64 {
    x as u64
}

