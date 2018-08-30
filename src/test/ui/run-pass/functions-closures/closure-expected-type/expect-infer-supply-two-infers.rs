// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn with_closure<A, F>(_: F)
    where F: FnOnce(Vec<A>, A)
{
}

fn expect_free_supply_free<'x>(x: &'x u32) {
    with_closure(|mut x: Vec<_>, y| {
        // Shows that the call to `x.push()` is influencing type of `y`...
        x.push(22_u32);

        // ...since we now know the type of `y` and can resolve the method call.
        y.wrapping_add(1);
    });
}

fn main() { }
