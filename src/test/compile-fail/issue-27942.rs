// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Resources<'a> {}

pub trait Buffer<'a, R: Resources<'a>> {
    fn select(&self) -> BufferViewHandle<R>;
    //~^ ERROR mismatched types
    //~| lifetime mismatch
    //~| NOTE expected type `Resources<'_>`
    //~| NOTE the lifetime 'a as defined on the method body at 14:4...
    //~| NOTE ...does not necessarily outlive the anonymous lifetime #1 defined on the method body
    //~| ERROR mismatched types
    //~| lifetime mismatch
    //~| NOTE expected type `Resources<'_>`
    //~| NOTE the anonymous lifetime #1 defined on the method body at 14:4...
    //~| NOTE ...does not necessarily outlive the lifetime 'a as defined on the method body
}

pub struct BufferViewHandle<'a, R: 'a+Resources<'a>>(&'a R);

fn main() {}
