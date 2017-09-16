// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(advanced_slice_patterns, slice_patterns)]

fn check(list: &[u8]) {
    match list {
        &[] => {},
        &[_u1, _u2, ref _next..] => {},
        &[_u1] => {},
    }
}

fn main() {}
