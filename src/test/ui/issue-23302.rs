// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that an enum with recursion in the discriminant throws
// the appropriate error (rather than, say, blowing the stack).
enum X {
    A = X::A as isize, //~ ERROR E0265
                       //~^ NOTE recursion not allowed in constant
}

// Since `Y::B` here defaults to `Y::A+1`, this is also a
// recursive definition.
enum Y {
    A = Y::B as isize, //~ ERROR E0265
                       //~^ NOTE recursion not allowed in constant
    B,
}

const A: i32 = B; //~ ERROR E0265
                  //~^ NOTE recursion not allowed in constant

const B: i32 = A; //~ ERROR E0265
                  //~^ NOTE recursion not allowed in constant

fn main() { }
