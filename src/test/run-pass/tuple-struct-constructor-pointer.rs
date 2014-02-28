// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Eq, Show)]
struct Foo(int);
#[deriving(Eq, Show)]
struct Bar(int, int);

pub fn main() {
    let f: fn(int) -> Foo = Foo;
    let g: fn(int, int) -> Bar = Bar;
    assert_eq!(f(42), Foo(42));
    assert_eq!(g(4, 7), Bar(4, 7));
}
