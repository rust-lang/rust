// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait TraitA {
    fn method_a(&self) -> int;
}

trait TraitB {
    fn gimme_an_a<A:TraitA>(&self, a: A) -> int;
}

impl TraitB for int {
    fn gimme_an_a<A:TraitA>(&self, a: A) -> int {
        a.method_a() + *self
    }
}

fn call_it<B:TraitB>(b: B)  -> int {
    let y = 4u;
    b.gimme_an_a(y) //~ ERROR the trait `TraitA` is not implemented
}

fn main() {
    let x = 3i;
    assert_eq!(call_it(x), 22);
}
