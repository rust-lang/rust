// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:trait-safety-lib.rs

// Simple smoke test that unsafe traits can be compiled across crates.

extern crate "trait-safety-lib" as lib;

use lib::Foo;

struct Bar { x: int }
unsafe impl Foo for Bar {
    fn foo(&self) -> int { self.x }
}

fn take_foo<F:Foo>(f: &F) -> int { f.foo() }

fn main() {
    let x: int = 22;
    assert_eq!(22, take_foo(&x));

    let x: Bar = Bar { x: 23 };
    assert_eq!(23, take_foo(&x));
}
