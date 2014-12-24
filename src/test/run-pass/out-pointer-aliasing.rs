// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub struct Foo {
    f1: int,
    _f2: int,
}

impl Copy for Foo {}

#[inline(never)]
pub fn foo(f: &mut Foo) -> Foo {
    let ret = *f;
    f.f1 = 0;
    ret
}

pub fn main() {
    let mut f = Foo {
        f1: 8,
        _f2: 9,
    };
    f = foo(&mut f);
    assert_eq!(f.f1, 8);
}
