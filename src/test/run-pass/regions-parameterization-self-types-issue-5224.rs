// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test how region-parameterization inference
// interacts with explicit self types.
//
// Issue #5224.

trait Getter {
    // This trait does not need to be
    // region-parameterized, because 'self
    // is bound in the self type:
    fn get(&self) -> &'self int;
}

struct Foo {
    field: int
}

impl Getter for Foo {
    fn get(&self) -> &'self int { &self.field }
}

fn get_int<G: Getter>(g: &G) -> int {
    *g.get()
}

pub fn main() {
    let foo = Foo { field: 22 };
    assert!(get_int(&foo) == 22);
}
