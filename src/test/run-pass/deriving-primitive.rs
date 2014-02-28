// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::num::FromPrimitive;
use std::int;

#[deriving(Eq, FromPrimitive, Show)]
enum A {
    Foo = int::MAX,
    Bar = 1,
    Baz = 3,
    Qux,
}

pub fn main() {
    let x: Option<A> = FromPrimitive::from_int(int::MAX);
    assert_eq!(x, Some(Foo));

    let x: Option<A> = FromPrimitive::from_int(1);
    assert_eq!(x, Some(Bar));

    let x: Option<A> = FromPrimitive::from_int(3);
    assert_eq!(x, Some(Baz));

    let x: Option<A> = FromPrimitive::from_int(4);
    assert_eq!(x, Some(Qux));

    let x: Option<A> = FromPrimitive::from_int(5);
    assert_eq!(x, None);
}
