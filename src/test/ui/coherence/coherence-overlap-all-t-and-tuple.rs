// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we detect an overlap here in the case where:
//
//    for some type X:
//      T = (X,)
//      T11 = X, U11 = X
//
// Seems pretty basic, but then there was issue #24241. :)

trait From<U> {
    fn foo() {}
}

impl <T> From<T> for T {
}

impl <T11, U11> From<(U11,)> for (T11,) { //~ ERROR E0119
}

fn main() { }
