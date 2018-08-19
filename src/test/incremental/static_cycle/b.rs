// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions:rpass1 rpass2

#![cfg_attr(rpass2, warn(dead_code))]

pub static mut BAA: *const i8 = unsafe { &BOO as *const _ as *const i8 };

pub static mut BOO: *const i8 = unsafe { &BAA as *const _ as *const i8 };

fn main() {}
