// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo {
    foo: Vec<u32>,
}

impl Copy for Foo { }

#[derive(Copy)]
struct Foo2<'a> {
    ty: &'a mut bool,
}

enum EFoo {
    Bar { x: Vec<u32> },
    Baz,
}

impl Copy for EFoo { }

#[derive(Copy)]
enum EFoo2<'a> {
    Bar(&'a mut bool),
    Baz,
}

fn main() {
}
