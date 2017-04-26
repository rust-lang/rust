// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn foo<F: Fn(usize)>(x: F) { }

fn main() {
    foo(|y: String| { });
    //~^ ERROR E0281
    //~| ERROR E0281
    //~| NOTE implements
    //~| NOTE implements
    //~| NOTE requires
    //~| NOTE requires
    //~| NOTE expected usize, found struct `std::string::String`
    //~| NOTE expected usize, found struct `std::string::String`
    //~| NOTE required by `foo`
    //~| NOTE required by `foo`
}
