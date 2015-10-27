// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
fn ret() -> u32 {
    static x: u32 = 10;
    x & if true { 10u32 } else { 20u32 } & x
}

fn ret2() -> &'static u32 {
    static x: u32 = 10;
    if true { 10u32; } else { 20u32; }
    &x
}

fn main() {}
