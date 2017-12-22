// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct S;

impl Iterator for S {
    type Item = i32;
    fn next(&mut self) -> Result<i32, i32> { Ok(7) }
    //~^ ERROR method `next` has an incompatible type for trait
    //~| expected enum `std::option::Option`, found enum `std::result::Result`
}

fn main() {}
