// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the struct update syntax only exists once

#[derive(Debug)]
struct Foo {
    one: i32,
    two: i32,
    three: i32,
}

impl Default for Foo {
    fn default() -> Self {
        Foo { one : 1, two: 2, three: 3 }
    }
}

impl Foo {
    fn new() -> Self {
        Foo { one: 4, two: 5, three: 6 }
    }
}

fn main() {

    let _foo = Foo {
        one: 110,
        ..Foo::default(),
        ..Foo::new(),
    };

    println!("{:?}", _foo);
}

