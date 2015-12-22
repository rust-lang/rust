// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that two unrelated functions have no trans dependency.

#![feature(rustc_attrs)]
#![allow(dead_code)]

fn main() { }

mod x {
    #[rustc_if_this_changed]
    pub fn x() { }
}

mod y {
    use x;

    // These dependencies SHOULD exist:
    #[rustc_then_this_would_need(TypeckItemBody)] //~ ERROR OK
    #[rustc_then_this_would_need(TransCrateItem)] //~ ERROR OK
    pub fn y() {
        x::x();
    }
}

mod z {
    use y;

    // These are expected to yield errors, because changes to `x`
    // affect the BODY of `y`, but not its signature.
    #[rustc_then_this_would_need(TypeckItemBody)] //~ ERROR no path
    #[rustc_then_this_would_need(TransCrateItem)] //~ ERROR no path
    pub fn z() {
        y::y();
    }
}
