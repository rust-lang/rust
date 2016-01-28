// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "foo"]

// @has foo/fn.tuple0.html //pre 'pub fn tuple0(x: ())'
pub fn tuple0(x: ()) -> () { x }
// @has foo/fn.tuple1.html //pre 'pub fn tuple1(x: (i32,)) -> (i32,)'
pub fn tuple1(x: (i32,)) -> (i32,) { x }
// @has foo/fn.tuple2.html //pre 'pub fn tuple2(x: (i32, i32)) -> (i32, i32)'
pub fn tuple2(x: (i32, i32)) -> (i32, i32) { x }
