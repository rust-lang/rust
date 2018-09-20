// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(arbitrary_self_types)]

use std::rc::Rc;

trait Foo {
    fn foo(self: &Rc<Self>) -> usize;
}

trait Bar {
    fn foo(self: &Rc<Self>) -> usize where Self: Sized;
    fn bar(self: Rc<Self>) -> usize;
}

impl Foo for usize {
    fn foo(self: &Rc<Self>) -> usize {
        **self
    }
}

impl Bar for usize {
    fn foo(self: &Rc<Self>) -> usize {
        **self
    }

    fn bar(self: Rc<Self>) -> usize {
        *self
    }
}

fn make_foo() {
    let x = Rc::new(5usize) as Rc<Foo>;
    //~^ ERROR E0038
    //~| ERROR E0038
}

fn make_bar() {
    let x = Rc::new(5usize) as Rc<Bar>;
    x.bar();
}

fn main() {}
