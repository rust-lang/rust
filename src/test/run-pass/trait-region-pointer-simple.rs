// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo {
    fn f(&self) -> int;
}

struct A {
    x: int
}

impl Foo for A {
    fn f(&self) -> int {
        io::println(~"Today's number is " + self.x.to_str());
        return self.x;
    }
}

pub fn main() {
    let a = A { x: 3 };
    let b = (&a) as &Foo;
    assert!(b.f() == 3);
}

