// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::duplicate_underscore_argument)]
#[allow(dead_code, unused)]

fn join_the_dark_side(darth: i32, _darth: i32) {}
fn join_the_light_side(knight: i32, _master: i32) {} // the Force is strong with this one

fn main() {
    join_the_dark_side(0, 0);
    join_the_light_side(0, 0);
}
