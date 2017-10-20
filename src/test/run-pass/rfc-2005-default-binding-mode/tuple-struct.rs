// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(match_default_bindings)]

enum Foo {
    Bar(Option<i8>, (), (), Vec<i32>),
    Baz,
}

pub fn main() {
    let foo = Foo::Bar(Some(1), (), (), vec![2, 3]);

    match &foo {
        Foo::Baz => panic!(),
        Foo::Bar(None, ..) => panic!(),
        Foo::Bar(Some(n), .., v) => {
            assert_eq!((*v).len(), 2);
            assert_eq!(*n, 1);
        }
    }
}
