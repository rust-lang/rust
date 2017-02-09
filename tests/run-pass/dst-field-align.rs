// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

struct Foo<T: ?Sized> {
    a: u16,
    b: T
}

trait Bar {
    fn get(&self) -> usize;
}

impl Bar for usize {
    fn get(&self) -> usize { *self }
}

fn main() {
    let f : Foo<usize> = Foo { a: 0, b: 11 };
    let f : &Foo<Bar> = &f;
    assert_eq!(f.b.get(), 11);
}
