// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


trait Repeat<A> { fn get(&self) -> A; }

impl<A:Clone> Repeat<A> for A {
    fn get(&self) -> A { self.clone() }
}

fn repeater<A:Clone>(v: A) -> Box<Repeat<A>:> {
    box v as Box<Repeat<A>:> // No
}

fn main() {
    // Error results because the type of is inferred to be
    // ~Repeat<&'blk int> where blk is the lifetime of the block below.

    let y = {
        let tmp0 = 3;
        let tmp1 = &tmp0; //~ ERROR `tmp0` does not live long enough
        repeater(tmp1)
    };
    assert!(3 == *(y.get()));
}
