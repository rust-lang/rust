// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue_41053.rs

pub trait Trait { fn foo(&self) {} }

pub struct Foo;

impl Iterator for Foo {
    type Item = Box<Trait>;
    fn next(&mut self) -> Option<Box<Trait>> {
        extern crate issue_41053;
        impl ::Trait for issue_41053::Test {
            fn foo(&self) {}
        }
        Some(Box::new(issue_41053::Test))
    }
}

fn main() {
    Foo.next().unwrap().foo();
}
