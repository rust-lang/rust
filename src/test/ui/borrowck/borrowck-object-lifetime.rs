// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that borrows that occur due to calls to object methods
// properly "claim" the object path.



trait Foo {
    fn borrowed(&self) -> &();
    fn mut_borrowed(&mut self) -> &();
}

fn borrowed_receiver(x: &Foo) {
    let y = x.borrowed();
    let z = x.borrowed();
    z.use_ref();
    y.use_ref();
}

fn mut_borrowed_receiver(x: &mut Foo) {
    let y = x.borrowed();
    let z = x.mut_borrowed(); //~ ERROR cannot borrow
    y.use_ref();
}

fn mut_owned_receiver(mut x: Box<Foo>) {
    let y = x.borrowed();
    let z = &mut x; //~ ERROR cannot borrow
    y.use_ref();
}

fn imm_owned_receiver(mut x: Box<Foo>) {
    let y = x.borrowed();
    let z = &x;
    z.use_ref();
    y.use_ref();
}

fn main() {}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
