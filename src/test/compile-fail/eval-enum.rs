// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Test {
    DivZero = 1/0,
    //~^ attempt to divide by zero
    //~| ERROR could not evaluate enum discriminant
    //~| ERROR this expression will panic at runtime
    RemZero = 1%0,
    //~^ attempt to calculate the remainder with a divisor of zero
    //~| ERROR could not evaluate enum discriminant
    //~| ERROR this expression will panic at runtime
}

fn main() {}
