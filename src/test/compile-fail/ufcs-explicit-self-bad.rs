// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

struct Foo {
    f: isize,
}

impl Foo {
    fn foo(self: isize, x: isize) -> isize {  //~ ERROR mismatched method receiver
        self.f + x
    }
}

struct Bar<T> {
    f: T,
}

impl<T> Bar<T> {
    fn foo(self: Bar<isize>, x: isize) -> isize { //~ ERROR mismatched method receiver
        x
    }
    fn bar(self: &Bar<usize>, x: isize) -> isize {   //~ ERROR mismatched method receiver
        x
    }
}

trait SomeTrait {
    fn dummy1(&self);
    fn dummy2(&self);
    fn dummy3(&self);
}

impl<'a, T> SomeTrait for &'a Bar<T> {
    fn dummy1(self: &&'a Bar<T>) { }
    fn dummy2(self: &Bar<T>) {} //~ ERROR mismatched method receiver
    //~^ ERROR mismatched method receiver
    fn dummy3(self: &&Bar<T>) {}
    //~^ ERROR mismatched method receiver
    //~| expected type `&&'a Bar<T>`
    //~| found type `&&Bar<T>`
    //~| lifetime mismatch
    //~| ERROR mismatched method receiver
    //~| expected type `&&'a Bar<T>`
    //~| found type `&&Bar<T>`
    //~| lifetime mismatch
}

fn main() {
    let foo = box Foo {
        f: 1,
    };
    println!("{}", foo.foo(2));
    let bar = box Bar {
        f: 1,
    };
    println!("{} {}", bar.foo(2), bar.bar(2));
}
