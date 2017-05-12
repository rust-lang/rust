// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let x = (0, 2);

    match x {
        (0, ref y) | (y, 0) => {} //~ ERROR E0409
                                  //~^ NOTE bound in different ways
                                  //~| NOTE first binding
                                  //~| ERROR E0308
                                  //~| NOTE expected &{integer}, found integral variable
                                  //~| NOTE expected type `&{integer}`
                                  //~| NOTE    found type `{integer}`
        _ => ()
    }
}
