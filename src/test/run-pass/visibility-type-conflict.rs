// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// `pub ( path_start` parses as visibility, check that it doesn't
// affect macros and workaround with parens works.

#![feature(type_macros)]

struct S(pub ((u8, u8)));
struct K(pub ((u8)));
struct U(pub (*const u8, usize));

macro_rules! m {
    ($t: ty) => ($t)
}

macro_rules! n {
    ($t: ty) => (struct L(pub $t);)
}

struct Z(pub m!((u8, u8)));
n! { (u8, u8) }

fn main() {}
