// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::if_let_some_result)]

fn str_to_int(x: &str) -> i32 {
    if let Some(y) = x.parse().ok() {
        y
    } else {
        0
    }
}

fn str_to_int_ok(x: &str) -> i32 {
    if let Ok(y) = x.parse() {
        y
    } else {
        0
    }
}

fn main() {
    let y = str_to_int("1");
    let z = str_to_int_ok("2");
    println!("{}{}", y, z);
}
