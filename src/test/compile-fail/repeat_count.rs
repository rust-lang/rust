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
    //~^ ERROR expected constant integer for repeat count, found variable [E0307]
    let b = [0; ()];
    //~^ ERROR mismatched types
    //~| expected `usize`
    //~| found `()`
    //~| expected usize
    //~| found ()) [E0308]
    //~| ERROR expected positive integer for repeat count, found tuple [E0306]
    let c = [0; true];
    //~^ ERROR mismatched types
    //~| expected `usize`
    //~| found `bool`
    //~| expected usize
    //~| found bool) [E0308]
    //~| ERROR expected positive integer for repeat count, found boolean [E0306]
    let d = [0; 0.5];
    //~^ ERROR mismatched types
    //~| expected `usize`
    //~| found `_`
    //~| expected usize
    //~| found floating-point variable) [E0308]
    //~| ERROR expected positive integer for repeat count, found float [E0306]
    let e = [0; "foo"];
    //~^ ERROR mismatched types
    //~| expected `usize`
    //~| found `&'static str`
    //~| expected usize
    //~| found &-ptr) [E0308]
    //~| ERROR expected positive integer for repeat count, found string literal [E0306]
    let f = [0; -4_isize];
    //~^ ERROR mismatched types
    //~| expected `usize`
    //~| found `isize`
    //~| expected usize
    //~| found isize) [E0308]
    //~| ERROR expected positive integer for repeat count, found negative integer [E0306]
    let f = [0_usize; -1_isize];
    //~^ ERROR mismatched types
    //~| expected `usize`
    //~| found `isize`
    //~| expected usize
    //~| found isize) [E0308]
    //~| ERROR expected positive integer for repeat count, found negative integer [E0306]
    struct G {
        g: (),
    }
    let g = [0; G { g: () }];
    //~^ ERROR mismatched types
    //~| expected `usize`
    //~| found `main::G`
    //~| expected usize
    //~| found struct `main::G`) [E0308]
    //~| ERROR expected positive integer for repeat count, found struct [E0306]
}
