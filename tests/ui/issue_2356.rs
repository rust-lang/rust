// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(clippy::while_let_on_iterator)]

use std::iter::Iterator;

struct Foo;

impl Foo {
    fn foo1<I: Iterator<Item = usize>>(mut it: I) {
        while let Some(_) = it.next() {
            println!("{:?}", it.size_hint());
        }
    }

    fn foo2<I: Iterator<Item = usize>>(mut it: I) {
        while let Some(e) = it.next() {
            println!("{:?}", e);
        }
    }
}

fn main() {
    Foo::foo1(vec![].into_iter());
    Foo::foo2(vec![].into_iter());
}
