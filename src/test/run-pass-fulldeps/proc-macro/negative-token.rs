// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:negative-token.rs
// ignore-stage1

#![feature(proc_macro_non_items)]

extern crate negative_token;

use negative_token::*;

fn main() {
    assert_eq!(-1, neg_one!());
    assert_eq!(-1.0, neg_one_float!());
}
