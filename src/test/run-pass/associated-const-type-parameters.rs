// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_consts)]

trait Foo {
    const X: i32;
    fn get_x() -> i32 {
       Self::X
    }
}

struct Abc;
impl Foo for Abc {
    const X: i32 = 11;
}

struct Def;
impl Foo for Def {
    const X: i32 = 97;
}

fn sub<A: Foo, B: Foo>() -> i32 {
    A::X - B::X
}

fn main() {
    assert_eq!(11, Abc::X);
    assert_eq!(97, Def::X);
    assert_eq!(11, Abc::get_x());
    assert_eq!(97, Def::get_x());
    assert_eq!(-86, sub::<Abc, Def>());
    assert_eq!(86, sub::<Def, Abc>());
}
