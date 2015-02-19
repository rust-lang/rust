// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub struct Foo;

pub trait Bar {
    fn bar(&self);
}

pub trait Baz {
    fn baz(&self) { }
}

impl<T: Baz> Bar for T {
    fn bar(&self) {}
}

impl Baz for Foo {}

pub fn foo(t: Box<Foo>) {
    t.bar(); // ~Foo doesn't implement Baz
    (*t).bar(); // ok b/c Foo implements Baz
}

fn main() {}
