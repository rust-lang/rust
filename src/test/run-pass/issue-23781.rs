// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;

struct Foo;
impl fmt::Debug for Foo {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        println!("<Foo as Debug>::fmt()");

        write!(fmt, "")
    }
}

fn test1() {
    let foo_str = format!("{:?}", Foo);

    println!("{}", foo_str);
}

fn test2() {
    println!("{:?}", Foo);
}

fn main() {
    // This works fine
    test1();

    // This fails
    test2();
}
