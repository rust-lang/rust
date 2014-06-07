// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::{int, i8, i16, i32, i64};
use std::task;

fn main() {
    assert!(task::try(proc() int::MIN / -1).is_err());
    assert!(task::try(proc() i8::MIN / -1).is_err());
    assert!(task::try(proc() i16::MIN / -1).is_err());
    assert!(task::try(proc() i32::MIN / -1).is_err());
    assert!(task::try(proc() i64::MIN / -1).is_err());
    assert!(task::try(proc() 1i / 0).is_err());
    assert!(task::try(proc() 1i8 / 0).is_err());
    assert!(task::try(proc() 1i16 / 0).is_err());
    assert!(task::try(proc() 1i32 / 0).is_err());
    assert!(task::try(proc() 1i64 / 0).is_err());
    assert!(task::try(proc() int::MIN % -1).is_err());
    assert!(task::try(proc() i8::MIN % -1).is_err());
    assert!(task::try(proc() i16::MIN % -1).is_err());
    assert!(task::try(proc() i32::MIN % -1).is_err());
    assert!(task::try(proc() i64::MIN % -1).is_err());
    assert!(task::try(proc() 1i % 0).is_err());
    assert!(task::try(proc() 1i8 % 0).is_err());
    assert!(task::try(proc() 1i16 % 0).is_err());
    assert!(task::try(proc() 1i32 % 0).is_err());
    assert!(task::try(proc() 1i64 % 0).is_err());
}
