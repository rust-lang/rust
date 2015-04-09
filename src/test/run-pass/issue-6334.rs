// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that everything still compiles and runs fine even when
// we reorder the bounds.


trait A {
    fn a(&self) -> usize;
}

trait B {
    fn b(&self) -> usize;
}

trait C {
    fn combine<T:A+B>(&self, t: &T) -> usize;
}

struct Foo;

impl A for Foo {
    fn a(&self) -> usize { 1 }
}

impl B for Foo {
    fn b(&self) -> usize { 2 }
}

struct Bar;

impl C for Bar {
    // Note below: bounds in impl decl are in reverse order.
    fn combine<T:B+A>(&self, t: &T) -> usize {
        (t.a() * 100) + t.b()
    }
}

fn use_c<S:C, T:B+A>(s: &S, t: &T) -> usize {
    s.combine(t)
}

pub fn main() {
    let foo = Foo;
    let bar = Bar;
    let r = use_c(&bar, &foo);
    assert_eq!(r, 102);
}
