// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Foo { fn foo<T>(&self, ext_thing: &T); }
pub trait Bar: Foo { }
impl<T: Foo> Bar for T { }

pub struct Thing;
impl Foo for Thing {
    fn foo<T>(&self, _: &T) {}
}

#[inline(never)] fn foo(b: &Bar) { b.foo(&0_usize) }

fn main() {
    let mut thing = Thing;
    let test: &Bar = &mut thing; //~ ERROR cannot convert to a trait object
    foo(test);
}
