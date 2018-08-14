// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
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
    foo: isize,
}

struct Bar {
    bar: isize,
}

impl Bar {
    fn make_foo (&self, i: isize) -> Box<Foo> {
        return box Foo { nonexistent: self, foo: i }; //~ ERROR: no field named
    }
}

fn main () {
    let bar = Bar { bar: 1 };
    let foo = bar.make_foo(2);
    println!("{}", foo.foo);
}
