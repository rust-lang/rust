// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_patterns, pattern_parentheses)]

const VALUE: usize = 21;

pub fn main() {
    match &18 {
        &(18..=18) => {}
        _ => { unreachable!(); }
    }
    match &21 {
        &(VALUE..=VALUE) => {}
        _ => { unreachable!(); }
    }
    match Box::new(18) {
        box (18..=18) => {}
        _ => { unreachable!(); }
    }
    match Box::new(21) {
        box (VALUE..=VALUE) => {}
        _ => { unreachable!(); }
    }
}
