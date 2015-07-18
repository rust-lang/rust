// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! https://github.com/rust-lang/rust/pull/27115

fn takes_str_argument(s: &str) -> &str { s }

macro_rules! string_or_ident {
    ($x: ident) => {
        takes_str_argument(stringify!($x))
    };
    ($x: expr) => {
        takes_str_argument($x)
    };
}

fn main() {
    assert_eq!(string_or_ident!(foo), "foo");
    assert_eq!(string_or_ident!("foo"), "foo");
}
