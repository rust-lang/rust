// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Trait {
    #[derivable]
    fn f(&self, x: int, y: &str);
}

impl int : Trait {
    fn f(&self, x: int, y: &str) {
        assert x == 42;
        assert y == "hello";
    }
}

impl float : Trait {
    fn f(&self, x: int, y: &str) {
        assert x == 42;
        assert y == "hello";
    }
}

struct Foo {
    x: int,
    y: float
}

impl Foo : Trait;

fn main() {
    let a: Foo = Foo { x: 1, y: 2.0 };
    a.f(42, "hello");
}

