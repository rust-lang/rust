// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;

trait Misc {}

fn size_of_copy<T: Copy+?Sized>() -> usize { mem::size_of::<T>() }

fn main() {
    size_of_copy::<Misc+Copy>();
    //~^ ERROR `Misc + Copy: std::marker::Copy` is not satisfied
    //~| ERROR the trait `std::marker::Copy` cannot be made into an object
}
