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
    #[derivable]
    pure fn ne(&self, other: &self) -> bool;
}

struct A {
    x: int
}

impl int : MyEq {
    pure fn eq(&self, other: &int) -> bool { *self == *other }
    pure fn ne(&self, other: &int) -> bool { *self != *other }
}

impl A : MyEq {
    pure fn ne(&self, other: &A) -> bool { !self.eq(other) }
}

fn main() {
    let a = A { x: 1 };
    assert a.eq(&a);
    assert !a.ne(&a);
}

