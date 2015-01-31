// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

// Test for what happens when a type parameter `A` is closed over into
// an object. This should yield errors unless `A` (and the object)
// both have suitable bounds.

trait Foo { fn get(&self); }

impl<A> Foo for A {
    fn get(&self) { }
}

fn repeater3<'a,A:'a>(v: A) -> Box<Foo+'a> {
    box v as Box<Foo+'a>
}

fn main() {
    // Error results because the type of is inferred to be
    // ~Repeat<&'blk isize> where blk is the lifetime of the block below.

    let _ = {
        let tmp0 = 3;
        let tmp1 = &tmp0; //~ ERROR `tmp0` does not live long enough
        repeater3(tmp1)
    };
}
