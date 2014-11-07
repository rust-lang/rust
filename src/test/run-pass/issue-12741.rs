// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Clone)]
pub struct Foo {
    f: fn(char, |char| -> char) -> char
}

impl Foo {
    fn bar(&self) -> char {
        ((*self).f)('a', |c: char| c)
    }
}

fn bla(c: char, cb: |char| -> char) -> char {
    cb(c)
}

pub fn make_foo() -> Foo {
    Foo {
        f: bla
    }
}

fn main() {
    let a = make_foo();
    assert_eq!(a.bar(), 'a');
}
