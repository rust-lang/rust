// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo {
    x: int,
}

pub impl Foo {
    fn f(&self) {}
    fn g(&const self) {}
    fn h(&mut self) {}
}

fn a(x: &mut Foo) {
    x.f();
    x.g();
    x.h();
}

fn b(x: &Foo) {
    x.f();
    x.g();
    x.h(); //~ ERROR cannot borrow
}

fn c(x: &const Foo) {
    x.f(); //~ ERROR cannot borrow
    //~^ ERROR unsafe borrow
    x.g();
    x.h(); //~ ERROR cannot borrow
    //~^ ERROR unsafe borrow
}

fn main() {
}
