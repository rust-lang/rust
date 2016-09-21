// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(untagged_unions)]

union U {
    a: u8,
    b: u16,
}

fn main() {
    let u = U {}; //~ ERROR union expressions should have exactly one field
    let u = U { a: 0 }; // OK
    let u = U { a: 0, b: 1 }; //~ ERROR union expressions should have exactly one field
    let u = U { a: 0, b: 1, c: 2 }; //~ ERROR union expressions should have exactly one field
                                    //~^ ERROR union `U` has no field named `c`
                                    //~| NOTE `U` does not have this field
    let u = U { ..u }; //~ ERROR union expressions should have exactly one field
                       //~^ ERROR functional record update syntax requires a struct

    let U {} = u; //~ ERROR union patterns should have exactly one field
    let U { a } = u; // OK
    let U { a, b } = u; //~ ERROR union patterns should have exactly one field
    let U { a, b, c } = u; //~ ERROR union patterns should have exactly one field
                           //~^ ERROR union `U` does not have a field named `c`
                           //~| NOTE union `U` does not have field `c`
    let U { .. } = u; //~ ERROR union patterns should have exactly one field
                      //~^ ERROR `..` cannot be used in union patterns
    let U { a, .. } = u; //~ ERROR `..` cannot be used in union patterns
}
