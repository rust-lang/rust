// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_consts)]

use std::str::FromStr;

struct A {}

trait X<T> {
    type Foo;
    const BAR: u32 = 128;

    fn foo() -> T;
    fn bar();
    fn    bay<
        'lifetime,    TypeParameterA
            >(  a   : usize,
                b: u8  );
}

impl std::fmt::Display for A {
//~^ ERROR not all trait items implemented, missing: `fmt`
//~| NOTE missing `fmt` in implementation
//~| NOTE fn fmt(&Self, &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error>;

}
impl FromStr for A{}
//~^ ERROR not all trait items implemented, missing: `Err`, `from_str`
//~| NOTE missing `Err`, `from_str` in implementation
//~| NOTE type Err;
//~| NOTE fn from_str(&str) -> std::result::Result<Self, <Self as std::str::FromStr>::Err>;

impl X<usize> for A {
//~^ ERROR not all trait items implemented, missing: `Foo`, `foo`, `bar`, `bay`
//~| NOTE missing `Foo`, `foo`, `bar`, `bay` in implementation
//~| NOTE type Foo;
//~| NOTE fn foo() -> T;
//~| NOTE fn bar();
//~| NOTE fn bay<'lifetime, TypeParameterA>(a: usize, b: u8);
}
