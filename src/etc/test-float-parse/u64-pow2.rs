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

use _common::validate;
use std::u64;

fn main() {
    for exp in 19..64 {
        let power: u64 = 1 << exp;
        validate(power.to_string());
        for offset in 1..123 {
            validate((power + offset).to_string());
            validate((power - offset).to_string());
        }
    }
    for offset in 0..123 {
        validate((u64::MAX - offset).to_string());
    }
}
