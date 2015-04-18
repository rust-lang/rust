// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(zero_one)]

use std::num::Zero;
use std::thread;

fn main() {
    assert!(thread::spawn(move|| { isize::min_value() / -1; }).join().is_err());
    assert!(thread::spawn(move|| { i8::min_value() / -1; }).join().is_err());
    assert!(thread::spawn(move|| { i16::min_value() / -1; }).join().is_err());
    assert!(thread::spawn(move|| { i32::min_value() / -1; }).join().is_err());
    assert!(thread::spawn(move|| { i64::min_value() / -1; }).join().is_err());
    assert!(thread::spawn(move|| { 1isize / isize::zero(); }).join().is_err());
    assert!(thread::spawn(move|| { 1i8 / i8::zero(); }).join().is_err());
    assert!(thread::spawn(move|| { 1i16 / i16::zero(); }).join().is_err());
    assert!(thread::spawn(move|| { 1i32 / i32::zero(); }).join().is_err());
    assert!(thread::spawn(move|| { 1i64 / i64::zero(); }).join().is_err());
    assert!(thread::spawn(move|| { isize::min_value() % -1; }).join().is_err());
    assert!(thread::spawn(move|| { i8::min_value() % -1; }).join().is_err());
    assert!(thread::spawn(move|| { i16::min_value() % -1; }).join().is_err());
    assert!(thread::spawn(move|| { i32::min_value() % -1; }).join().is_err());
    assert!(thread::spawn(move|| { i64::min_value() % -1; }).join().is_err());
    assert!(thread::spawn(move|| { 1isize % isize::zero(); }).join().is_err());
    assert!(thread::spawn(move|| { 1i8 % i8::zero(); }).join().is_err());
    assert!(thread::spawn(move|| { 1i16 % i16::zero(); }).join().is_err());
    assert!(thread::spawn(move|| { 1i32 % i32::zero(); }).join().is_err());
    assert!(thread::spawn(move|| { 1i64 % i64::zero(); }).join().is_err());
}
