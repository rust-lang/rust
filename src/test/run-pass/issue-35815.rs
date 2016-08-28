// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;

struct Foo<T: ?Sized> {
    a: i64,
    b: bool,
    c: T,
}

fn main() {
    let foo: &Foo<i32> = &Foo { a: 1, b: false, c: 2i32 };
    let foo_unsized: &Foo<Send> = foo;
    assert_eq!(mem::size_of_val(foo), mem::size_of_val(foo_unsized));
}
