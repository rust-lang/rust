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

fn main() {
    let mut pow = vec![];
    for i in 0..63 {
        pow.push(1u64 << i);
    }
    for a in &pow {
        for b in &pow {
            for c in &pow {
                validate((a | b | c).to_string());
            }
        }
    }
}
