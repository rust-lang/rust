// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that overloaded call parameter checking does not ICE
// when a type error or unconstrained type variable propagates
// into it.

#![feature(unboxed_closures)]

fn main() {
    (return)((),());
    //~^ ERROR the type of this value must be known
    //~^^ ERROR the type of this value must be known
    //~^^^ ERROR cannot use call notation
}
