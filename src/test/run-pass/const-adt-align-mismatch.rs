// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;

#[derive(PartialEq, Debug)]
enum Foo {
    A(u32),
    Bar([u16; 4]),
    C
}

// NOTE(eddyb) Don't make this a const, needs to be a static
// so it is always instantiated as a LLVM constant value.
static FOO: Foo = Foo::C;

fn main() {
    assert_eq!(FOO, Foo::C);
    assert_eq!(mem::size_of::<Foo>(), 12);
    assert_eq!(mem::min_align_of::<Foo>(), 4);
}
