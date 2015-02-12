// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test associated type references in a struct literal. Issue #20535.

pub trait Foo {
    type Bar;

    fn dummy(&self) { }
}

impl Foo for int {
    type Bar = int;
}

struct Thing<F: Foo> {
    a: F,
    b: F::Bar,
}

fn main() {
    let thing = Thing{a: 1, b: 2};
    assert_eq!(thing.a + 1, thing.b);
}
