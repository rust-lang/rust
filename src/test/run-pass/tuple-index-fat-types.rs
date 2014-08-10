// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(tuple_indexing)]

struct Foo<'a>(&'a [int]);

fn main() {
    let x: &[int] = &[1i, 2, 3];
    let y = (x,);
    assert_eq!(y.0, x);

    let x: &[int] = &[1i, 2, 3];
    let y = Foo(x);
    assert_eq!(y.0, x);
}
