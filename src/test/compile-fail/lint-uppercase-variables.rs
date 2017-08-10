// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(unused)]
#![allow(dead_code)]
#![deny(non_snake_case)]

mod foo {
    pub enum Foo { Foo }
}

struct Something {
    X: usize //~ ERROR structure field `X` should have a snake case name such as `x`
}

fn test(Xx: usize) { //~ ERROR variable `Xx` should have a snake case name such as `xx`
    println!("{}", Xx);
}

fn main() {
    let Test: usize = 0; //~ ERROR variable `Test` should have a snake case name such as `test`
    println!("{}", Test);

    match foo::Foo::Foo {
        Foo => {}
//~^ ERROR variable `Foo` should have a snake case name such as `foo`
//~^^ WARN `Foo` is named the same as one of the variants of the type `foo::Foo`
//~^^^ WARN unused variable: `Foo`
    }

    test(1);

    let _ = Something { X: 0 };
}
