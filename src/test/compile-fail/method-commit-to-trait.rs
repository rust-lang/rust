// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we pick `Foo`, and also pick the `impl`, even though in
// this case the vector type `T` is not copyable. This is because
// there is no other reasonable choice. The error you see is thus
// about `T` being non-copyable, not about `Foo` being
// unimplemented. This is better for user too, since it suggests minimal
// diff requird to fix program.

trait Object { }

trait Foo {
    fn foo(self) -> int;
}

impl<T:Copy> Foo for Vec<T> {
    fn foo(self) -> int {1}
}

fn test1<T>(x: Vec<T>) {
    x.foo();
    //~^ ERROR `core::kinds::Copy` is not implemented for the type `T`
}

fn main() { }
