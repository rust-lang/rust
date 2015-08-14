// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we get an error when you use `<Self as Get>::Value` in
// the trait definition even if there is no default method.

trait Get {
    type Value;
}

trait Other {
    fn okay<U:Get>(&self, foo: U, bar: <Self as Get>::Value);
    //~^ ERROR E0277
}

impl Get for () {
    type Value = f32;
}

impl Get for f64 {
    type Value = u32;
}

impl Other for () {
    fn okay<U:Get>(&self, _foo: U, _bar: <Self as Get>::Value) { }
}

impl Other for f64 {
    fn okay<U:Get>(&self, _foo: U, _bar: <Self as Get>::Value) { }
}

fn main() { }
