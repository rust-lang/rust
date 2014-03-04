// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Tests contravariant type parameters in implementations of traits, see #2687

trait A {
    fn a(&self) -> int;
}

impl A for u8 {
    fn a(&self) -> int {
        8
    }
}

impl A for u32 {
    fn a(&self) -> int {
        32
    }
}

trait B: A {
    fn b(&self) -> int;
}

impl B for u32 {
    fn b(&self) -> int {
        -1
    }
}

trait Foo {
    fn test_fn<T: B>(&self, x: T) -> int;
    fn test_duplicated_bounds1_fn<T: B+B>(&self) -> int;
    fn test_duplicated_bounds2_fn<T: B>(&self) -> int;
}

impl Foo for int {
    fn test_fn<T: A>(&self, x: T) -> int {
        x.a()
    }

    fn test_duplicated_bounds1_fn<T: B+B>(&self) -> int {
        99
    }

    fn test_duplicated_bounds2_fn<T: B+B>(&self) -> int {
        199
    }
}

fn main() {
    let x: int = 0;
    assert!(8 == x.test_fn(5u8));
    assert!(32 == x.test_fn(5u32));
}
