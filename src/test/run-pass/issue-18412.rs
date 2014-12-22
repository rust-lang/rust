// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that non-static methods can be assigned to local variables as
// function pointers.

trait Foo {
    fn foo(&self) -> uint;
}

struct A(uint);

impl A {
    fn bar(&self) -> uint { self.0 }
}

impl Foo for A {
    fn foo(&self) -> uint { self.bar() }
}

fn main() {
    let f = A::bar;
    let g = Foo::foo;
    let a = A(42);

    assert_eq!(f(&a), g(&a));
}
