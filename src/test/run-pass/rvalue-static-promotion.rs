// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::Cell;

const NONE_CELL_STRING: Option<Cell<String>> = None;

struct Foo<T>(T);
impl<T> Foo<T> {
    const FOO: Option<Box<T>> = None;
}

fn main() {
    let _: &'static u32 = &42;
    let _: &'static Option<u32> = &None;

    // We should be able to peek at consts and see they're None.
    let _: &'static Option<Cell<String>> = &NONE_CELL_STRING;
    let _: &'static Option<Box<()>> = &Foo::FOO;
}
