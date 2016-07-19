// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for Issue #20971.

// error-pattern:Hello, world!

pub trait Parser {
    type Input;
    fn parse(&mut self, input: <Self as Parser>::Input);
}

impl Parser for () {
    type Input = ();
    fn parse(&mut self, input: ()) {}
}

pub fn many() -> Box<Parser<Input = <() as Parser>::Input> + 'static> {
    panic!("Hello, world!")
}

fn main() {
    many().parse(());
}
