// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    let vr = v.iter().filter(|x| {
        x % 2 == 0
        //~^ ERROR binary operation `%` cannot be applied to type `&&{integer}`
        //~| NOTE this is a reference to a type that `%` can be applied to
        //~| NOTE an implementation of `std::ops::Rem` might be missing for `&&{integer}`
    });
    println!("{:?}", vr);
}
