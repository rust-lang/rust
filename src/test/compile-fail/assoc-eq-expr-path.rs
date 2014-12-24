// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that an associated type cannot be bound in an expression path.

#![feature(associated_types)]

trait Foo {
    type A;
    fn bar() -> int;
}

impl Foo for int {
    type A = uint;
    fn bar() -> int { 42 }
}

pub fn main() {
    let x: int = Foo::<A=uint>::bar();
    //~^ERROR unexpected binding of associated item in expression path
}
