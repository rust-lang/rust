// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#[derive(Copy, Clone)]
enum Foo {
    Bar = 0xDEADBEE
}

static X: Foo = Foo::Bar;

pub fn main() {
    assert_eq!((X as usize), 0xDEADBEE);
    assert_eq!((Y as usize), 0xDEADBEE);
}

static Y: Foo = Foo::Bar;
