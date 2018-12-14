// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// should not warn on lossy casting in constant types
// because not supported yet
const C: i32 = 42;
const C_I64: i64 = C as i64;

fn main() {
    // should suggest i64::from(c)
    let c: i32 = 42;
    let c_i64: i64 = c as i64;
}
