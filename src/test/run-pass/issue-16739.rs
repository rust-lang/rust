// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax)]
#![feature(unboxed_closures)]

// Test that unboxing shim for calling rust-call ABI methods through a
// trait box works and does not cause an ICE.

struct Foo { foo: u32 }

impl FnMut<()> for Foo {
    type Output = u32;
    extern "rust-call" fn call_mut(&mut self, _: ()) -> u32 { self.foo }
}

impl FnMut<(u32,)> for Foo {
    type Output = u32;
    extern "rust-call" fn call_mut(&mut self, (x,): (u32,)) -> u32 { self.foo + x }
}

impl FnMut<(u32,u32)> for Foo {
    type Output = u32;
    extern "rust-call" fn call_mut(&mut self, (x, y): (u32, u32)) -> u32 { self.foo + x + y }
}

fn main() {
    let mut f = box Foo { foo: 42 } as Box<FnMut() -> u32>;
    assert_eq!(f.call_mut(()), 42);

    let mut f = box Foo { foo: 40 } as Box<FnMut(u32) -> u32>;
    assert_eq!(f.call_mut((2,)), 42);

    let mut f = box Foo { foo: 40 } as Box<FnMut(u32, u32) -> u32>;
    assert_eq!(f.call_mut((1, 1)), 42);
}
