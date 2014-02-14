// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:noexporttypelib.rs

extern crate noexporttypelib;

fn main() {
    // Here, the type returned by foo() is not exported.
    // This used to cause internal errors when serializing
    // because the def_id associated with the type was
    // not convertible to a path.
  let x: int = noexporttypelib::foo();
    //~^ ERROR expected `int` but found `std::option::Option<int>`
}
