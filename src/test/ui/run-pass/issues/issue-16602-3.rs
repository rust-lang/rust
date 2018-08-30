// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Debug)]
enum Foo {
    Bar(u32, u32),
    Baz(&'static u32, &'static u32)
}

static NUM: u32 = 100;

fn main () {
    let mut b = Foo::Baz(&NUM, &NUM);
    b = Foo::Bar(f(&b), g(&b));
}

static FNUM: u32 = 1;

fn f (b: &Foo) -> u32 {
    FNUM
}

static GNUM: u32 = 2;

fn g (b: &Foo) -> u32 {
    GNUM
}
