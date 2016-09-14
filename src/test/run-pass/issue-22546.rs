// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Parsing patterns with paths with type parameters (issue #22544)

use std::default::Default;

#[derive(Default)]
pub struct Foo<T>(T, T);

impl<T: ::std::fmt::Display> Foo<T> {
    fn foo(&self) {
        match *self {
            Foo::<T>(ref x, ref y) => println!("Goodbye, World! {} {}", x, y)
        }
    }
}

trait Tr {
    type U;
}

impl<T> Tr for Foo<T> {
    type U = T;
}

struct Wrapper<T> {
    value: T
}

fn main() {
    let Foo::<i32>(a, b) = Default::default();

    let f = Foo(2,3);
    f.foo();

    let w = Wrapper { value: Foo(10u8, 11u8) };
    match w {
        Wrapper::<Foo<u8>> { value: Foo(10, 11) } => {},
        ::Wrapper::<<Foo<_> as Tr>::U> { value: Foo::<u8>(11, 16) } => { panic!() },
        _ => { panic!() }
    }

    if let None::<u8> = Some(8) {
        panic!();
    }
    if let None::<u8> { .. } = Some(8) {
        panic!();
    }
    if let Option::None::<u8> { .. } = Some(8) {
        panic!();
    }
}
