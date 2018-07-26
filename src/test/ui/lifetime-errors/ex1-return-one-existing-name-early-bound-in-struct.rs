// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Clone)]
enum Foo<'a> {
    Bar(&'a str),
}

impl<'a> Foo<'a> {
    fn bar(&self, other: Foo) -> Foo<'a> {
        match *self {
            Foo::Bar(s) => {
                if s == "test" {
                    other //~ ERROR explicit lifetime
                } else {
                    self.clone()
                }
            }
        }
    }
}

fn main() { }
