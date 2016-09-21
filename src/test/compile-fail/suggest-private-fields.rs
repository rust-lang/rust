// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:struct_field_privacy.rs

extern crate struct_field_privacy as xc;

use xc::B;

struct A {
    pub a: u32,
    b: u32,
}

fn main () {
    // external crate struct
    let k = B {
        aa: 20,
        //~^ ERROR struct `xc::B` has no field named `aa`
        //~| NOTE field does not exist - did you mean `a`?
        bb: 20,
        //~^ ERROR struct `xc::B` has no field named `bb`
        //~| NOTE field does not exist - did you mean `a`?
    };
    // local crate struct
    let l = A {
        aa: 20,
        //~^ ERROR struct `A` has no field named `aa`
        //~| NOTE field does not exist - did you mean `a`?
        bb: 20,
        //~^ ERROR struct `A` has no field named `bb`
        //~| NOTE field does not exist - did you mean `b`?
    };
}
