// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod _common;

use std::mem::transmute;
use _common::validate;

fn main() {
    for bits in 0u32..(1 << 21) {
        let single: f32 = unsafe { transmute(bits) };
        validate(format!("{:e}", single));
        let double: f64 = unsafe { transmute(bits as u64) };
        validate(format!("{:e}", double));
    }
}
