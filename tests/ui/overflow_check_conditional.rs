// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(clippy::many_single_char_names)]
#![warn(clippy::overflow_check_conditional)]

fn main() {
    let a: u32 = 1;
    let b: u32 = 2;
    let c: u32 = 3;
    if a + b < a {}
    if a > a + b {}
    if a + b < b {}
    if b > a + b {}
    if a - b > b {}
    if b < a - b {}
    if a - b > a {}
    if a < a - b {}
    if a + b < c {}
    if c > a + b {}
    if a - b < c {}
    if c > a - b {}
    let i = 1.1;
    let j = 2.2;
    if i + j < i {}
    if i - j < i {}
    if i > i + j {}
    if i - j < i {}
}
