// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! foo {
    ($a:expr) => $a; //~ ERROR macro rhs must be delimited
}

fn main() {
    foo!(0); // Check that we report errors at macro definition, not expansion.

    let _: cfg!(foo) = (); //~ ERROR non-type macro in type position
    derive!(); //~ ERROR `derive` can only be used in attributes
}
