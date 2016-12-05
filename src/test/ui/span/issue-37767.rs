// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait A {
    fn foo(&mut self) {}
}

trait B : A {
    fn foo(&mut self) {}
}

fn bar<T: B>(a: &T) {
    a.foo()
}

trait C {
    fn foo(&self) {}
}

trait D : C {
    fn foo(&self) {}
}

fn quz<T: D>(a: &T) {
    a.foo()
}

trait E : Sized {
    fn foo(self) {}
}

trait F : E {
    fn foo(self) {}
}

fn foo<T: F>(a: T) {
    a.foo()
}

fn pass<T: C>(a: &T) {
    a.foo()
}

fn main() {}
