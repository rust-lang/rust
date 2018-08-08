// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-rustfix

fn main() {
    match &Some(3) {
        &None => 1
        &Some(2) => { 3 }
        //~^ ERROR expected one of `,`, `.`, `?`, `}`, or an operator, found `=>`
        //~| NOTE expected one of `,`, `.`, `?`, `}`, or an operator here
        _ => 2
    };
}
