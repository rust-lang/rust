// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_rules)]

struct Nested<T> {
    _f: T
}

macro_rules! int_type(
    () => (
        int
    )
)

macro_rules! nested_type(
    ($x:ty) => (
        Nested<$x>
    )
)

macro_rules! indirect(
    () => (
        nested_type!(int_type!())
    )
)

macro_rules! ident_type(
    ($x:ident) => (
        $x
    )
)

pub fn main() {
    let a: int_type!();
    let _: int = a;

    let b: nested_type!(bool);
    let _: Nested<bool> = b;

    let c: indirect!();
    let _: Nested<int> = c;

    let d: ident_type!(u32);
    let _: u32 = d;
}
