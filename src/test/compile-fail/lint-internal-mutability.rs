// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(internal_mutability)]

use std::cell::{Cell,RefCell};
use std::ty::Unsafe;
use std::rc::Rc;

struct Foo {
    x: Unsafe<uint>
}
impl Foo {
    fn amp_call(&self) {}
    fn amp_mut_call(&mut self) {}
    fn call(self) {}
}

struct Bar;

impl Bar {
    fn call(&self, _: Cell<uint>, _: &RefCell<uint>, _: &Rc<uint>) {}
    fn call2(&self, _: &Cell<uint>, _: RefCell<uint>, _: &uint) {}
}

fn call(_: Cell<uint>, _: &RefCell<uint>, _: &Rc<uint>) {}
fn call2(_: &Cell<uint>, _: RefCell<uint>, _: &uint) {}


fn main() {
    let a = Cell::new(1);
    let b1 = RefCell::new(2u);
    let b2 = b1.clone(); //~ ERROR internally mutable type used in method call
    let c = Rc::new(3);

    call(a, //~ ERROR internally mutable type used in function call
         &b1, //~ ERROR internally mutable type used in function call
         &c); //~ ERROR internally mutable type used in function call

    call2(&a, //~ ERROR internally mutable type used in function call
          b1, //~ ERROR internally mutable type used in function call
          &*c);

    let mut foo = Foo { x: Unsafe::new(4) };

    foo.amp_call();//~ ERROR internally mutable type used in method call
    foo.amp_mut_call();//~ ERROR internally mutable type used in method call
    foo.call();//~ ERROR internally mutable type used in method call

    Bar.call(a, //~ ERROR internally mutable type used in method call
             &b2, //~ ERROR internally mutable type used in method call
             &c); //~ ERROR internally mutable type used in method call

    Bar.call2(&a, //~ ERROR internally mutable type used in method call
              b2, //~ ERROR internally mutable type used in method call
              &*c);

}
