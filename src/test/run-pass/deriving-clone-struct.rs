// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Clone)]
struct S {
    _int: int,
    _i8: i8,
    _i16: i16,
    _i32: i32,
    _i64: i64,

    _uint: uint,
    _u8: u8,
    _u16: u16,
    _u32: u32,
    _u64: u64,

    _f32: f32,
    _f64: f64,

    _bool: bool,
    _char: char,
    _nil: ()
}

pub fn main() {}
