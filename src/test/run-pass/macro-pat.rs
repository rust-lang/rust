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

macro_rules! mypat(
    () => (
        Some('y')
    )
)

macro_rules! char_x(
    () => (
        'x'
    )
)

macro_rules! some(
    ($x:pat) => (
        Some($x)
    )
)

macro_rules! indirect(
    () => (
        some!(char_x!())
    )
)

macro_rules! ident_pat(
    ($x:ident) => (
        $x
    )
)

fn f(c: Option<char>) -> uint {
    match c {
        Some('x') => 1,
        mypat!() => 2,
        _ => 3,
    }
}

pub fn main() {
    assert_eq!(1, f(Some('x')));
    assert_eq!(2, f(Some('y')));
    assert_eq!(3, f(None));

    assert_eq!(1, match Some('x') {
        Some(char_x!()) => 1,
        _ => 2,
    });

    assert_eq!(1, match Some('x') {
        some!(char_x!()) => 1,
        _ => 2,
    });

    assert_eq!(1, match Some('x') {
        indirect!() => 1,
        _ => 2,
    });

    assert_eq!(3, {
        let ident_pat!(x) = 2;
        x+1
    });
}
