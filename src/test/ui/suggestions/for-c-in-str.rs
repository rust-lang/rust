// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// E0277 should point exclusively at line 14, not the entire for loop span

fn main() {
    for c in "asdf" {
    //~^ ERROR the trait bound `&str: std::iter::Iterator` is not satisfied
    //~| NOTE `&str` is not an iterator
    //~| HELP the trait `std::iter::Iterator` is not implemented for `&str`
    //~| NOTE required by `std::iter::IntoIterator::into_iter`
        println!("");
    }
}
