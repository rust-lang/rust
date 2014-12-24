// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unboxed_closures)]

// Test that unboxing shim for calling rust-call ABI methods through a
// trait box works and does not cause an ICE.

struct Foo { foo: uint }

impl FnMut<(), uint> for Foo {
    extern "rust-call" fn call_mut(&mut self, _: ()) -> uint { self.foo }
}

impl FnMut<(uint,), uint> for Foo {
    extern "rust-call" fn call_mut(&mut self, (x,): (uint,)) -> uint { self.foo + x }
}

impl FnMut<(uint, uint), uint> for Foo {
    extern "rust-call" fn call_mut(&mut self, (x, y): (uint, uint)) -> uint { self.foo + x + y }
}

fn main() {
    let mut f = box Foo { foo: 42 } as Box<FnMut<(), uint>>;
    assert_eq!(f.call_mut(()), 42);

    let mut f = box Foo { foo: 40 } as Box<FnMut<(uint,), uint>>;
    assert_eq!(f.call_mut((2,)), 42);

    let mut f = box Foo { foo: 40 } as Box<FnMut<(uint, uint), uint>>;
    assert_eq!(f.call_mut((1, 1)), 42);
}
