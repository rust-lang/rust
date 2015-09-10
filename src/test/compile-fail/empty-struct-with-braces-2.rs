// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Empty struct defined with braces shouldn't add names into value namespace

#![deny(warnings)]

struct Empty {}

fn main() {
    let e = Empty {};

    match e {
        Empty => () //~ ERROR unused variable: `Empty`
        //~^ ERROR variable `Empty` should have a snake case name such as `empty`
    }
}
