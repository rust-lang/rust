// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #21212: an overflow occurred during trait
// checking where normalizing `Self::Input` led to normalizing the
// where clauses in the environment which in turn required normalizing
// `Self::Input`.

pub trait Parser {
    type Input;

    fn parse(input: <Self as Parser>::Input) {
        panic!()
    }
}

impl <P> Parser for P {
    type Input = ();
}

fn main() {
}
