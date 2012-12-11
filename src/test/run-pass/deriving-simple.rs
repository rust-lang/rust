// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait MyEq {
    #[derivable]
    pure fn eq(&self, other: &self) -> bool;
}

struct A {
    x: int
}

struct B {
    x: A,
    y: A,
    z: A
}

impl A : MyEq {
    pure fn eq(&self, other: &A) -> bool {
        unsafe { io::println(fmt!("eq %d %d", self.x, other.x)); }
        self.x == other.x
    }
}

impl B : MyEq;

fn main() {
    let b = B { x: A { x: 1 }, y: A { x: 2 }, z: A { x: 3 } };
    let c = B { x: A { x: 1 }, y: A { x: 3 }, z: A { x: 4 } };
    assert b.eq(&b);
    assert c.eq(&c);
    assert !b.eq(&c);
    assert !c.eq(&b);
}

