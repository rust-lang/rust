// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]
#![allow(warnings)]

// Check that you are allowed to implement using elision but write
// trait without elision (a bug in this cropped up during
// bootstrapping, so this is a regression test).

pub struct SplitWhitespace<'a> {
    x: &'a u8
}

pub trait UnicodeStr {
    fn split_whitespace<'a>(&'a self) -> SplitWhitespace<'a>;
}

impl UnicodeStr for str {
    #[inline]
    fn split_whitespace(&self) -> SplitWhitespace {
        unimplemented!()
    }
}

#[rustc_error]
fn main() { } //~ ERROR compilation successful
