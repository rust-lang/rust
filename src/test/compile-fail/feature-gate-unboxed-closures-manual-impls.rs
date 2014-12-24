// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

struct Foo;
impl Fn<(), ()> for Foo { //~ ERROR manual implementations of `Fn` are experimental
    extern "rust-call" fn call(&self, args: ()) -> () {}
}
struct Bar;
impl FnMut<(), ()> for Bar { //~ ERROR manual implementations of `FnMut` are experimental
    extern "rust-call" fn call_mut(&self, args: ()) -> () {}
}
struct Baz;
impl FnOnce<(), ()> for Baz { //~ ERROR manual implementations of `FnOnce` are experimental
    extern "rust-call" fn call_once(&self, args: ()) -> () {}
}

fn main() {}
