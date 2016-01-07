// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(rustc_attrs)]

#[derive(PartialEq, Debug)]
struct Point {
    _x: i32,
    _y: i32,
}
const STRUCT: Point = Point { _x: 42, _y: 42 };
const TUPLE1: (i32, i32) = (42, 42);
const TUPLE2: (&'static str, &'static str) = ("hello","world");

#[rustc_mir]
fn mir() -> (Point, (i32, i32), (&'static str, &'static str)){
    let struct1 = STRUCT;
    let tuple1 = TUPLE1;
    let tuple2 = TUPLE2;
    (struct1, tuple1, tuple2)
}

fn main(){
    assert_eq!(mir(), (STRUCT, TUPLE1, TUPLE2));
}

