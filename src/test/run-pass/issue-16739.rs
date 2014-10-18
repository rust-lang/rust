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
// trait box works and does not cause an ICE

struct Foo { foo: uint }

impl FnOnce<(), uint> for Foo {
    #[rust_call_abi_hack]
    fn call_once(self, _: ()) -> uint { self.foo }
}

impl FnOnce<(uint,), uint> for Foo {
    #[rust_call_abi_hack]
    fn call_once(self, (x,): (uint,)) -> uint { self.foo + x }
}

impl FnOnce<(uint, uint), uint> for Foo {
    #[rust_call_abi_hack]
    fn call_once(self, (x, y): (uint, uint)) -> uint { self.foo + x + y }
}

fn main() {
    let f = box Foo { foo: 42 } as Box<FnOnce<(), uint>>;
    assert_eq!(f.call_once(()), 42);

    let f = box Foo { foo: 40 } as Box<FnOnce<(uint,), uint>>;
    assert_eq!(f.call_once((2,)), 42);

    let f = box Foo { foo: 40 } as Box<FnOnce<(uint, uint), uint>>;
    assert_eq!(f.call_once((1, 1)), 42);
}
