// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
# ICE when returning struct with borrowed pointer to trait

A function which takes a borrowed pointer to a trait and returns a
struct with that borrowed pointer results in an ICE.

This does not occur with concrete types, only with borrowed pointers
to traits.
*/

// original
trait Inner {
    fn print(&self);
}

impl Inner for int {
    fn print(&self) { print(format!("Inner: {}\n", *self)); }
}

struct Outer<'self> {
    inner: &'self Inner
}

impl<'self> Outer<'self> {
    fn new<'r>(inner: &'r Inner) -> Outer<'r> {
        Outer {
            inner: inner
        }
    }
}

pub fn main() {
    let inner = 5;
    let outer = Outer::new(&inner as &Inner);
    outer.inner.print();
}


// minimal
trait MyTrait<T> { }

pub struct MyContainer<'self, T> {
    foos: ~[&'self MyTrait<T>],
}

impl<'self, T> MyContainer<'self, T> {
    pub fn add (&mut self, foo: &'self MyTrait<T>) {
        self.foos.push(foo);
    }
}
