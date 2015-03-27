// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test equality constraints on associated types in a where clause.

// pretty-expanded FIXME #23616

pub trait Foo {
    type A;
    fn boo(&self) -> <Self as Foo>::A;
}

#[derive(PartialEq)]
pub struct Bar;

impl Foo for isize {
    type A = usize;
    fn boo(&self) -> usize { 42 }
}

impl Foo for char {
    type A = Bar;
    fn boo(&self) -> Bar { Bar }
}

fn foo_bar<I: Foo<A=Bar>>(x: I) -> Bar {
    x.boo()
}

fn foo_uint<I: Foo<A=usize>>(x: I) -> usize {
    x.boo()
}

pub fn main() {
    let a = 42;
    foo_uint(a);

    let a = 'a';
    foo_bar(a);
}
