// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]

enum Abc {
    A(u8),
    B(i8),
    C,
    D,
}

#[rustc_mir]
fn foo(x: Abc) -> i32 {
    match x {
        Abc::C => 3,
        Abc::D => 4,
        Abc::B(_) => 2,
        Abc::A(_) => 1,
    }
}

#[rustc_mir]
fn foo2(x: Abc) -> bool {
    match x {
        Abc::D => true,
        _ => false
    }
}

fn main() {
    assert_eq!(1, foo(Abc::A(42)));
    assert_eq!(2, foo(Abc::B(-100)));
    assert_eq!(3, foo(Abc::C));
    assert_eq!(4, foo(Abc::D));

    assert_eq!(false, foo2(Abc::A(1)));
    assert_eq!(false, foo2(Abc::B(2)));
    assert_eq!(false, foo2(Abc::C));
    assert_eq!(true, foo2(Abc::D));
}
