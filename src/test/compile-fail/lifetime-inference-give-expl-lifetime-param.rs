// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

struct Foo<'x> { bar: int }
fn foo1<'a>(x: &Foo) -> &'a int {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn foo1<'a>(x: &'a Foo) -> &'a int
    &x.bar //~ ERROR: cannot infer
}

fn foo2<'a, 'b>(x: &'a Foo) -> &'b int {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn foo2<'a>(x: &'a Foo) -> &'a int
    &x.bar //~ ERROR: cannot infer
}

fn foo3<'a>(x: &Foo) -> (&'a int, &'a int) {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn foo3<'a>(x: &'a Foo) -> (&'a int, &'a int)
    (&x.bar, &x.bar) //~ ERROR: cannot infer
    //~^ ERROR: cannot infer
}

fn foo4<'a, 'b>(x: &'a Foo) -> (&'b int, &'a int, &'b int) {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn foo4<'a>(x: &'a Foo) -> (&'a int, &'a int, &'a int)
    (&x.bar, &x.bar, &x.bar) //~ ERROR: cannot infer
    //~^ ERROR: cannot infer
}

struct Bar<'x, 'y, 'z> { bar: &'y int, baz: int }
fn bar1<'a>(x: &Bar) -> (&'a int, &'a int, &'a int) {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn bar1<'b, 'c, 'a>(x: &'a Bar<'b, 'a, 'c>) -> (&'a int, &'a int, &'a int)
    (x.bar, &x.baz, &x.baz) //~ ERROR: cannot infer
    //~^ ERROR: cannot infer
    //~^^ ERROR: cannot infer
}

fn bar2<'a, 'b, 'c>(x: &Bar<'a, 'b, 'c>) -> (&'a int, &'a int, &'a int) {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn bar2<'a, 'c>(x: &'a Bar<'a, 'a, 'c>) -> (&'a int, &'a int, &'a int)
    (x.bar, &x.baz, &x.baz) //~ ERROR: cannot infer
    //~^ ERROR: cannot infer
    //~^^ ERROR: cannot infer
}

struct Cat<'x, T> { cat: &'x int, t: T }
struct Dog<'y> { dog: &'y int }

fn cat2<'x, 'y>(x: Cat<'x, Dog<'y>>) -> &'x int {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn cat2<'x>(x: Cat<'x, Dog<'x>>) -> &'x int
    x.t.dog //~ ERROR: cannot infer
}

struct Baz<'x> {
    bar: &'x int
}

impl<'a> Baz<'a> {
    fn baz2<'b>(&self, x: &int) -> (&'b int, &'b int) {
        // The lifetime that gets assigned to `x` seems somewhat random.
        // I have disabled this test for the time being. --pcwalton
        (self.bar, x) //~ ERROR: cannot infer
        //~^ ERROR: cannot infer
    }
}

fn main() {}
