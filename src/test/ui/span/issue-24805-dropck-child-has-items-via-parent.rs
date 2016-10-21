// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that child trait who only has items via its *parent* trait
// does cause dropck to inject extra region constraints.

#![allow(non_camel_case_types)]

trait Parent { fn foo(&self); }
trait Child: Parent { }

impl Parent for i32 { fn foo(&self) { } }
impl<'a> Parent for &'a D_Child<i32> {
    fn foo(&self) {
        println!("accessing child value: {}", self.0);
    }
}

impl Child for i32 { }
impl<'a> Child for &'a D_Child<i32> { }

struct D_Child<T:Child>(T);
impl <T:Child> Drop for D_Child<T> { fn drop(&mut self) { self.0.foo() } }

fn f_child() {
    // `_d` and `d1` are assigned the *same* lifetime by region inference ...
    let (_d, d1);

    d1 = D_Child(1);
    // ... we store a reference to `d1` within `_d` ...
    _d = D_Child(&d1);

    // ... dropck *should* complain, because Drop of _d could (and
    // does) access the already dropped `d1` via the `foo` method.
}
//~^ ERROR `d1` does not live long enough

fn main() {
    f_child();
}
