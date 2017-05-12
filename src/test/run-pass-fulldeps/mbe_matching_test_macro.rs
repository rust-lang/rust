// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:procedural_mbe_matching.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(procedural_mbe_matching)]

pub fn main() {
    assert_eq!(matches!(Some(123), None | Some(0)), false);
    assert_eq!(matches!(Some(123), None | Some(123)), true);
    assert_eq!(matches!(true, true), true);
}
