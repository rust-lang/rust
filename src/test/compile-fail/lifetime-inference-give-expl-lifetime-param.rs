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

use std::marker::PhantomData;

struct Foo<'x> { bar: isize, marker: PhantomData<&'x ()> }
fn foo1<'a>(x: &Foo) -> &'a isize {
//~^ HELP: consider using an explicit lifetime parameter as shown: fn foo1<'a>(x: &'a Foo) -> &'a isize
    &x.bar //~ ERROR: cannot infer
}

fn foo2<'a, 'b>(x: &'a Foo) -> &'b isize {
//~^ HELP: consider using an explicit lifetime parameter as shown: fn foo2<'a>(x: &'a Foo) -> &'a isize
    &x.bar //~ ERROR: cannot infer
}

fn foo3<'a>(x: &Foo) -> (&'a isize, &'a isize) {
//~^ HELP: consider using an explicit lifetime parameter as shown: fn foo3<'a>(x: &'a Foo) -> (&'a isize, &'a isize)
    (&x.bar, &x.bar) //~ ERROR: cannot infer
    //~^ ERROR: cannot infer
}

fn foo4<'a, 'b>(x: &'a Foo) -> (&'b isize, &'a isize, &'b isize) {
//~^ HELP: consider using an explicit lifetime parameter as shown: fn foo4<'a>(x: &'a Foo) -> (&'a isize, &'a isize, &'a isize)
    (&x.bar, &x.bar, &x.bar) //~ ERROR: cannot infer
    //~^ ERROR: cannot infer
}

struct Cat<'x, T> { cat: &'x isize, t: T }
struct Dog<'y> { dog: &'y isize }

fn cat2<'x, 'y>(x: Cat<'x, Dog<'y>>) -> &'x isize {
//~^ HELP: consider using an explicit lifetime parameter as shown: fn cat2<'x>(x: Cat<'x, Dog<'x>>) -> &'x isize
    x.t.dog //~ ERROR: cannot infer
}

struct Baz<'x> {
    bar: &'x isize
}

impl<'a> Baz<'a> {
    fn baz2<'b>(&self, x: &isize) -> (&'b isize, &'b isize) {
        // The lifetime that gets assigned to `x` seems somewhat random.
        // I have disabled this test for the time being. --pcwalton
        (self.bar, x) //~ ERROR: cannot infer
        //~^ ERROR: cannot infer
    }
}

fn main() {}
