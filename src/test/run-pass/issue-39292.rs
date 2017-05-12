// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #39292. The object vtable was being
// incorrectly left with a null pointer.

trait Foo<T> {
    fn print<'a>(&'a self) where T: 'a { println!("foo"); }
}

impl<'a> Foo<&'a ()> for () { }

trait Bar: for<'a> Foo<&'a ()> { }

impl Bar for () {}

fn main() {
    (&() as &Bar).print(); // Segfault
}
