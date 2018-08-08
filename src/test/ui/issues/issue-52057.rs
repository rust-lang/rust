// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #52057. There is an implied bound
// that `I: 'a` where `'a` is the lifetime of `self` in `parse_first`;
// but to observe that, one must normalize first.
//
// run-pass

#![feature(nll)]

pub trait Parser {
    type Input;

    #[inline(always)]
    fn parse_first(input: &mut Self::Input);
}

impl<'a, I, P: ?Sized> Parser for &'a mut P
where
    P: Parser<Input = I>,
{
    type Input = I;

    fn parse_first(_: &mut Self::Input) {}
}

fn main() {}
