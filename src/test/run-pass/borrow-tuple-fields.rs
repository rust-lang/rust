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

struct Foo(int, int);

fn main() {
    let x = (1i, 2i);
    let a = &x.0;
    let b = &x.0;
    assert_eq!(*a, 1);
    assert_eq!(*b, 1);

    let mut x = (1i, 2i);
    {
        let a = &x.0;
        let b = &mut x.1;
        *b = 5;
        assert_eq!(*a, 1);
    }
    assert_eq!(x.0, 1);
    assert_eq!(x.1, 5);


    let x = Foo(1i, 2i);
    let a = &x.0;
    let b = &x.0;
    assert_eq!(*a, 1);
    assert_eq!(*b, 1);

    let mut x = Foo(1i, 2i);
    {
        let a = &x.0;
        let b = &mut x.1;
        *b = 5;
        assert_eq!(*a, 1);
    }
    assert_eq!(x.0, 1);
    assert_eq!(x.1, 5);
}
