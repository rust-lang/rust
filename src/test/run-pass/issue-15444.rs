// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait MyTrait {
    fn foo(&self);
}

impl<A, B, C> MyTrait for fn(A, B) -> C {
    fn foo(&self) {}
}

fn bar<T: MyTrait>(t: &T) {
    t.foo()
}

fn thing(a: int, b: int) -> int {
    a + b
}

fn main() {
    let thing: fn(int, int) -> int = thing; // coerce to fn type
    bar(&thing);
}
