// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that explicit region bounds are allowed on the various
// nominal types (but not on other types) and that they are type
// checked.

struct an_enum<'a>(&'a isize);
struct a_class<'a> { x:&'a isize }

fn a_fn1<'a,'b>(e: an_enum<'a>) -> an_enum<'b> {
    return e; //~  ERROR mismatched types
              //~| expected type `an_enum<'b>`
              //~| found type `an_enum<'a>`
              //~| lifetime mismatch
}

fn a_fn3<'a,'b>(e: a_class<'a>) -> a_class<'b> {
    return e; //~  ERROR mismatched types
              //~| expected type `a_class<'b>`
              //~| found type `a_class<'a>`
              //~| lifetime mismatch
}

fn main() { }
