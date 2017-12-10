// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Array: Sized {}

fn f<T: Array>(x: &T) {
    let _ = x
    //~^ ERROR `Array` cannot be made into an object
    //~| NOTE the trait cannot require that `Self : Sized`
    //~| NOTE requirements on the impl of `std::ops::CoerceUnsized<&Array>`
    //~| NOTE the trait `Array` cannot be made into an object
    as
    &Array;
    //~^ ERROR `Array` cannot be made into an object
    //~| NOTE the trait cannot require that `Self : Sized`
    //~| NOTE the trait `Array` cannot be made into an object
}

fn main() {}
