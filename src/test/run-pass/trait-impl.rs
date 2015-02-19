// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test calling methods on an impl for a bare trait.

// aux-build:traitimpl.rs
extern crate traitimpl;
use traitimpl::Bar;

static mut COUNT: uint = 1;

trait T {
    fn t(&self) {}
}

impl<'a> T+'a {
    fn foo(&self) {
        unsafe { COUNT *= 2; }
    }
    fn bar() {
        unsafe { COUNT *= 3; }
    }
}

impl T for int {}

struct Foo;
impl<'a> Bar<'a> for Foo {}

fn main() {
    let x: &T = &42;

    x.foo();
    T::foo(x);
    T::bar();

    unsafe { assert!(COUNT == 12); }

    // Cross-crait case
    let x: &Bar = &Foo;
    x.bar();
}
