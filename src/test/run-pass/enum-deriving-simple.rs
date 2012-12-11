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

enum B {
    C(A),
    D(A),
    E(A)
}

impl A : MyEq {
    pure fn eq(&self, other: &A) -> bool {
        unsafe { io::println("in eq"); }
        self.x == other.x
    }
}

impl B : MyEq;

fn main() {
    let c = C(A { x: 15 });
    let d = D(A { x: 30 });
    let e = C(A { x: 30 });
    assert c.eq(&c);
    assert !c.eq(&d);
    assert !c.eq(&e);
}

