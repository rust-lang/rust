// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that types that appear in input types in an object type are
// subject to the reflect check.

use std::marker::Reflect;
use std::io::Write;

trait Get<T> {
    fn get(self) -> T;
}

struct Struct<T>(T);

fn is_reflect<T:Reflect>() { }

fn a<T>() {
    is_reflect::<T>(); //~ ERROR not implemented
}

fn ok_a<T: Reflect>() {
    is_reflect::<T>(); // OK
}

fn b<T>() {
    is_reflect::<Box<Get<T>>>(); //~ ERROR not implemented
}

fn ok_b<T: Reflect>() {
    is_reflect::<Box<Get<T>>>(); // OK
}

fn c<T>() {
    is_reflect::<Box<Get<Struct<T>>>>(); //~ ERROR not implemented
}

fn main() {
    is_reflect::<Box<Get<Struct<()>>>>(); // OK
}
