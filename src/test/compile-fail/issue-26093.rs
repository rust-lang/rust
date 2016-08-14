// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! not_an_lvalue {
    ($thing:expr) => {
        $thing = 42;
        //~^ ERROR invalid left-hand side expression
        //~^^ NOTE left-hand of expression not valid
    }
}

fn main() {
    not_an_lvalue!(99);
    //~^ NOTE in this expansion of not_an_lvalue!
}
