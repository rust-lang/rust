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
    let a = [0; n]; //~ ERROR expected constant integer for repeat count, found variable
    let b = [0; ()];
//~^ ERROR mismatched types
//~| expected `usize`
//~| found `()`
//~| expected usize
//~| found ()
//~| ERROR expected constant integer for repeat count, found non-constant expression
    let c = [0; true];
    //~^ ERROR mismatched types
    //~| expected `usize`
    //~| found `bool`
    //~| expected usize
    //~| found bool
    //~| ERROR expected positive integer for repeat count, found boolean
    let d = [0; 0.5];
    //~^ ERROR mismatched types
    //~| expected `usize`
    //~| found `_`
    //~| expected usize
    //~| found floating-point variable
    //~| ERROR expected positive integer for repeat count, found float
    let e = [0; "foo"];
    //~^ ERROR mismatched types
    //~| expected `usize`
    //~| found `&'static str`
    //~| expected usize
    //~| found &-ptr
    //~| ERROR expected positive integer for repeat count, found string
    let f = [0; -4_isize];
    //~^ ERROR mismatched types
    //~| expected `usize`
    //~| found `isize`
    //~| expected usize
    //~| found isize
    //~| ERROR expected positive integer for repeat count, found negative integer
    let f = [0_usize; -1_isize];
    //~^ ERROR mismatched types
    //~| expected `usize`
    //~| found `isize`
    //~| expected usize
    //~| found isize
    //~| ERROR expected positive integer for repeat count, found negative integer
}
