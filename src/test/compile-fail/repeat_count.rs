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
    let a = [0, ..n]; //~ ERROR expected constant integer for repeat count, found variable
    let b = [0, ..()]; //~ ERROR expected positive integer for repeat count, found ()
    //~^ ERROR: expected `uint`, found `()`
    let c = [0, ..true]; //~ ERROR expected positive integer for repeat count, found boolean
    //~^ ERROR: expected `uint`, found `bool`
    let d = [0, ..0.5]; //~ ERROR expected positive integer for repeat count, found float
    //~^ ERROR: expected `uint`, found `<generic float #0>`
    let e = [0, .."foo"]; //~ ERROR expected positive integer for repeat count, found string
    //~^ ERROR: expected `uint`, found `&'static str`
    let f = [0, ..-4];
    //~^ ERROR expected positive integer for repeat count, found negative integer
    let f = [0u, ..-1];
    //~^ ERROR expected positive integer for repeat count, found negative integer
}
