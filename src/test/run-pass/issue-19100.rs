// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]

#[derive(Copy, Clone)]
enum Foo {
    Bar,
    Baz
}

impl Foo {
    fn foo(&self) {
        match self {
            &
Bar if true
//~^ WARN pattern binding `Bar` is named the same as one of the variants of the type `Foo`
//~^^ HELP to match on a variant, consider making the path in the pattern qualified: `Foo::Bar`
=> println!("bar"),
            &
Baz if false
//~^ WARN pattern binding `Baz` is named the same as one of the variants of the type `Foo`
//~^^ HELP to match on a variant, consider making the path in the pattern qualified: `Foo::Baz`
=> println!("baz"),
_ => ()
        }
    }
}

fn main() {}
