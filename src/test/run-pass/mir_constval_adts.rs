// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(PartialEq, Debug)]
struct Point {
    _x: i32,
    _y: i32,
}

#[derive(PartialEq, Eq, Debug)]
struct Newtype<T>(T);

const STRUCT: Point = Point { _x: 42, _y: 42 };
const TUPLE1: (i32, i32) = (42, 42);
const TUPLE2: (&'static str, &'static str) = ("hello","world");
const PAIR_NEWTYPE: (Newtype<i32>, Newtype<i32>) = (Newtype(42), Newtype(42));

fn mir() -> (Point, (i32, i32), (&'static str, &'static str), (Newtype<i32>, Newtype<i32>)) {
    let struct1 = STRUCT;
    let tuple1 = TUPLE1;
    let tuple2 = TUPLE2;
    let pair_newtype = PAIR_NEWTYPE;
    (struct1, tuple1, tuple2, pair_newtype)
}

const NEWTYPE: Newtype<&'static str> = Newtype("foobar");

fn test_promoted_newtype_str_ref() {
    let x = &NEWTYPE;
    assert_eq!(x, &Newtype("foobar"));
}

fn main(){
    assert_eq!(mir(), (STRUCT, TUPLE1, TUPLE2, PAIR_NEWTYPE));
    test_promoted_newtype_str_ref();
}

