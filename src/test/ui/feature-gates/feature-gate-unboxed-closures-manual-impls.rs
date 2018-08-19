// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that manual impls of the `Fn` traits are not possible without
// a feature gate. In fact, the specialized check for these cases
// never triggers (yet), because they encounter other problems around
// angle bracket vs parentheses notation.

#![allow(dead_code)]

struct Foo;
impl Fn<()> for Foo {
    extern "rust-call" fn call(self, args: ()) -> () {}
    //~^ ERROR rust-call ABI is subject to change
}
struct Foo1;
impl FnOnce() for Foo1 {
    extern "rust-call" fn call_once(self, args: ()) -> () {}
    //~^ ERROR rust-call ABI is subject to change
}
struct Bar;
impl FnMut<()> for Bar {
    extern "rust-call" fn call_mut(&self, args: ()) -> () {}
    //~^ ERROR rust-call ABI is subject to change
}
struct Baz;
impl FnOnce<()> for Baz {
    extern "rust-call" fn call_once(&self, args: ()) -> () {}
    //~^ ERROR rust-call ABI is subject to change
}

fn main() {}
