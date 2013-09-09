// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// original problem
fn foo<T>() -> int {
    {
        static foo: int = 2;
        foo
    }
}

// issue 8134
struct Foo;
impl<T> Foo {
    pub fn foo(&self) {
        static X: uint = 1;
    }
}

// issue 8134
pub struct Parser<T>;
impl<T: std::iter::Iterator<char>> Parser<T> {
    fn in_doctype(&mut self) {
        static DOCTYPEPattern: [char, ..6] = ['O', 'C', 'T', 'Y', 'P', 'E'];
    }
}

struct Bar;
impl<T> Foo {
    pub fn bar(&self) {
        static X: uint = 1;
    }
}
