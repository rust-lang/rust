// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #3645

fn main() {
    let n = 1;
    let a = [0; n];
    //~^ ERROR attempt to use a non-constant value in a constant [E0435]
    let b = [0; ()];
    //~^ ERROR mismatched types
    //~| expected type `usize`
    //~| found type `()`
    //~| expected usize, found ()
    let c = [0; true];
    //~^ ERROR mismatched types
    //~| expected usize, found bool
    let d = [0; 0.5];
    //~^ ERROR mismatched types
    //~| expected type `usize`
    //~| found type `{float}`
    //~| expected usize, found floating-point variable
    let e = [0; "foo"];
    //~^ ERROR mismatched types
    //~| expected type `usize`
    //~| found type `&'static str`
    //~| expected usize, found reference
    let f = [0; -4_isize];
    //~^ ERROR mismatched types
    //~| expected usize, found isize
    let f = [0_usize; -1_isize];
    //~^ ERROR mismatched types
    //~| expected usize, found isize
    struct G {
        g: (),
    }
    let g = [0; G { g: () }];
    //~^ ERROR mismatched types
    //~| expected type `usize`
    //~| found type `main::G`
    //~| expected usize, found struct `main::G`
}
