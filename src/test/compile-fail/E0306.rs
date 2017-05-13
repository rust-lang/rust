// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const A: [u32; "hello"] = [];
//~^ ERROR expected `usize` for array length, found string literal [E0306]
//~| NOTE expected `usize`

const B: [u32; true] = [];
//~^ ERROR expected `usize` for array length, found boolean [E0306]
//~| NOTE expected `usize`

const C: [u32; 0.0] = [];
//~^ ERROR expected `usize` for array length, found float [E0306]
//~| NOTE expected `usize`

fn main() {
}
