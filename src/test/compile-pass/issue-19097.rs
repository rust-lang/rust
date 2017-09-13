// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// regression test for #19097

struct Foo<T>(T);

impl<'a, T> Foo<&'a T> {
    fn foo(&self) {}
}
impl<'a, T> Foo<&'a mut T> {
    fn foo(&self) {}
}

fn main() {}
