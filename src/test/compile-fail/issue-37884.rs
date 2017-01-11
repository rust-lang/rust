// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct RepeatMut<'a, T>(T, &'a ());

impl<'a, T: 'a> Iterator for RepeatMut<'a, T> {
    type Item = &'a mut T;
    fn next(&'a mut self) -> Option<Self::Item>
    //~^ ERROR method not compatible with trait
    //~| lifetime mismatch
    //~| NOTE expected type `fn(&mut RepeatMut<'a, T>) -> std::option::Option<&mut T>`
    {
    //~^ NOTE the anonymous lifetime #1 defined on the body
    //~| NOTE ...does not necessarily outlive the lifetime 'a as defined on the body
        Some(&mut self.0)
    }
}

fn main() {}
