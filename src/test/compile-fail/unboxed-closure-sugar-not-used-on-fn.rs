// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// Test that the `Fn` traits require `()` form without a feature gate.

fn bar1(x: &Fn<()>) {
    //~^ ERROR angle-bracket notation is not stable when used with the `Fn` family
}

fn bar2<T>(x: &T) where T: Fn<()> {
    //~^ ERROR angle-bracket notation is not stable when used with the `Fn` family
}

fn main() { }

