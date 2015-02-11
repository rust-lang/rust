// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we are able to successfully compile a setup where a trait
// (`Trait1`) references a struct (`SomeType<u32>`) which in turn
// carries a predicate that references the trait (`u32 : Trait1`,
// substituted).

#![allow(dead_code)]

trait Trait1 : Trait2<SomeType<u32>> {
    fn dumb(&self) { }
}

trait Trait2<A> {
    fn dumber(&self, _: A) { }
}

struct SomeType<A>
    where A : Trait1
{
    a: A
}

impl Trait1 for u32 { }

impl Trait2<SomeType<u32>> for u32 { }

fn main() { }
