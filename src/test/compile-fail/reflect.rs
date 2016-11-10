// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(reflect_marker)]

// Test that there is no way to get a generic type `T` to be
// considered as `Reflect` (or accessible via something that is
// considered `Reflect`) without a reflect bound, but that any
// concrete type works fine. Note that object types are tested
// separately.

use std::marker::Reflect;
use std::io::Write;

struct Struct<T>(T);

fn is_reflect<T:Reflect>() { }

fn c<T>() {
    is_reflect::<Struct<T>>(); //~ ERROR E0277
}

fn ok_c<T: Reflect>() {
    is_reflect::<Struct<T>>(); // OK
}

fn d<T>() {
    is_reflect::<(i32, T)>(); //~ ERROR E0277
}

fn main() {
    is_reflect::<&i32>(); // OK
    is_reflect::<Box<Write>>(); // OK
}
