// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait repeat<A> { fn get(&self) -> A; }

impl<A:Copy> repeat<A> for @A {
    fn get(&self) -> A { **self }
}

fn repeater<A:Copy>(v: @A) -> @repeat<A> {
    // Note: owned kind is not necessary as A appears in the trait type
    @v as @repeat<A> // No
}

fn main() {
    // Error results because the type of is inferred to be
    // @repeat<&'blk int> where blk is the lifetime of the block below.

    let y = { //~ ERROR reference is not valid
        let x: &'blk int = &3;
        repeater(@x)
    };
    assert!(3 == *(y.get())); //~ ERROR dereference of reference outside its lifetime
    //~^ ERROR reference is not valid outside of its lifetime
    //~^^ ERROR reference is not valid outside of its lifetime
    //~^^^ ERROR reference is not valid outside of its lifetime
    //~^^^^ ERROR cannot infer an appropriate lifetime
}
