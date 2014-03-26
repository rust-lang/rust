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
fn foo1(x: &Foo) -> &int {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn foo1<'a>(x: &'a Foo) -> &'a int
    &x.bar //~ ERROR: cannot infer
}

fn foo2<'a, 'b>(x: &'a Foo) -> &'b int {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn foo2<'a>(x: &'a Foo) -> &'a int
    &x.bar //~ ERROR: cannot infer
}

fn foo3(x: &Foo) -> (&int, &int) {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn foo3<'a>(x: &'a Foo) -> (&'a int, &'a int)
    (&x.bar, &x.bar) //~ ERROR: cannot infer
    //~^ ERROR: cannot infer
}

fn foo4<'a, 'b>(x: &'a Foo) -> (&'b int, &'a int, &int) {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn foo4<'a>(x: &'a Foo) -> (&'a int, &'a int, &'a int)
    (&x.bar, &x.bar, &x.bar) //~ ERROR: cannot infer
    //~^ ERROR: cannot infer
}

fn foo5(x: &int) -> &int {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn foo5<'a>(x: &'a int) -> &'a int
    x //~ ERROR: mismatched types
    //~^ ERROR: cannot infer
}

struct Bar<'x, 'y, 'z> { bar: &'y int, baz: int }
fn bar1(x: &Bar) -> (&int, &int, &int) {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn bar1<'a, 'b, 'c, 'd>(x: &'d Bar<'b, 'a, 'c>) -> (&'a int, &'d int, &'d int)
    (x.bar, &x.baz, &x.baz) //~ ERROR: mismatched types
    //~^ ERROR: cannot infer
    //~^^ ERROR: cannot infer
}

fn bar2<'a, 'b, 'c>(x: &Bar<'a, 'b, 'c>) -> (&int, &int, &int) {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn bar2<'d, 'a, 'b, 'c>(x: &'d Bar<'a, 'b, 'c>) -> (&'b int, &'d int, &'d int)
    (x.bar, &x.baz, &x.baz) //~ ERROR: mismatched types
    //~^ ERROR: cannot infer
    //~^^ ERROR: cannot infer
}

struct Cat<'x, T> { cat: &'x int, t: T }
struct Dog<'y> { dog: &'y int }
fn cat<'x>(x: Cat<'x, Dog>) -> &int {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn cat<'a, 'x>(x: Cat<'x, Dog<'a>>) -> &'a int
    x.t.dog //~ ERROR: mismatched types
}

fn cat2<'x, 'y>(x: Cat<'x, Dog<'y>>) -> &int {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn cat2<'x, 'y>(x: Cat<'x, Dog<'y>>) -> &'y int
    x.t.dog //~ ERROR: mismatched types
}

struct Baz<'x> {
    bar: &'x int
}

impl<'x> Baz<'x> {
    fn baz1(&self) -> &int {
    //~^ NOTE: consider using an explicit lifetime parameter as shown: fn baz1(&self) -> &'x int
        self.bar //~ ERROR: mismatched types
    }
}

impl<'a> Baz<'a> {
    fn baz2(&self, x: &int) -> (&int, &int) {
    //~^ NOTE: consider using an explicit lifetime parameter as shown: fn baz2<'b>(&self, x: &'b int) -> (&'a int, &'b int)
        (self.bar, x) //~ ERROR: cannot infer
        //~^ ERROR: mismatched types
        //~^^ ERROR: mismatched types
    }
}

fn main() {}
