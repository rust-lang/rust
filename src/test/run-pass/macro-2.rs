// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty - token trees can't pretty print

#![feature(macro_rules)]

pub fn main() {

    macro_rules! mylambda_tt(
        ($x:ident, $body:expr) => ({
            fn f($x: int) -> int { return $body; };
            f
        })
    )

    assert!(mylambda_tt!(y, y * 2)(8) == 16)
}
