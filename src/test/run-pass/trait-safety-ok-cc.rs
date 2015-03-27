// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:trait_safety_lib.rs

// Simple smoke test that unsafe traits can be compiled across crates.

// pretty-expanded FIXME #23616

extern crate trait_safety_lib as lib;

use lib::Foo;

struct Bar { x: isize }
unsafe impl Foo for Bar {
    fn foo(&self) -> isize { self.x }
}

fn take_foo<F:Foo>(f: &F) -> isize { f.foo() }

fn main() {
    let x: isize = 22;
    assert_eq!(22, take_foo(&x));

    let x: Bar = Bar { x: 23 };
    assert_eq!(23, take_foo(&x));
}
