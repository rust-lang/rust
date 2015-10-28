// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test: issue had to do with "givens" in region inference,
// which were not being considered during the contraction phase.

// error-pattern:explicit panic

struct Parser<'i: 't, 't>(&'i u8, &'t u8);

impl<'i, 't> Parser<'i, 't> {
    fn parse_nested_block<F, T>(&mut self, parse: F) -> Result<T, ()>
        where for<'tt> F: FnOnce(&mut Parser<'i, 'tt>) -> T { panic!() }

    fn expect_exhausted(&mut self) -> Result<(), ()> { Ok(()) }
}

fn main() {
    let x = 0u8;
    Parser(&x, &x).parse_nested_block(|input| input.expect_exhausted()).unwrap();
}
