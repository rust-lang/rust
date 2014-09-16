// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that when you implement a trait that has a sized type
// parameter, the corresponding value must be sized. Also that the
// self type must be sized if appropriate.

trait Foo<T> { fn take(self, x: &T) { } } // Note: T is sized

impl Foo<[int]> for uint { }
//~^ ERROR the trait `core::kinds::Sized` is not implemented for the type `[int]`

impl Foo<int> for [uint] { }
//~^ ERROR the trait `core::kinds::Sized` is not implemented for the type `[uint]`

pub fn main() { }
