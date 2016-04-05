// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can quantify lifetimes outside a constraint (i.e., including
// the self type) in a where clause. Specifically, test that implementing for a
// specific lifetime is not enough to satisify the `for<'a> ...` constraint, which
// should require *all* lifetimes.

static X: &'static u32 = &42;

trait Bar {
    fn bar(&self);
}

impl Bar for &'static u32 {
    fn bar(&self) {}
}

fn foo<T>(x: &T)
    where for<'a> &'a T: Bar
{}

fn main() {
    foo(&X);
    //~^ error: `for<'a> &'a _: Bar` is not satisfied
}
