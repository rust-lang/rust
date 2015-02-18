// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::num::Int;
use std::thread;

// Avoid using constants, which would trigger compile-time errors.
fn min_val<T: Int>() -> T { Int::min_value() }
fn zero<T: Int>() -> T { Int::zero() }

fn main() {
    assert!(thread::spawn(move|| { min_val::<isize>() / -1; }).join().is_err());
    assert!(thread::spawn(move|| { min_val::<i8>() / -1; }).join().is_err());
    assert!(thread::spawn(move|| { min_val::<i16>() / -1; }).join().is_err());
    assert!(thread::spawn(move|| { min_val::<i32>() / -1; }).join().is_err());
    assert!(thread::spawn(move|| { min_val::<i64>() / -1; }).join().is_err());
    assert!(thread::spawn(move|| { 1isize / zero(); }).join().is_err());
    assert!(thread::spawn(move|| { 1i8 / zero(); }).join().is_err());
    assert!(thread::spawn(move|| { 1i16 / zero(); }).join().is_err());
    assert!(thread::spawn(move|| { 1i32 / zero(); }).join().is_err());
    assert!(thread::spawn(move|| { 1i64 / zero(); }).join().is_err());
    assert!(thread::spawn(move|| { min_val::<isize>() % -1; }).join().is_err());
    assert!(thread::spawn(move|| { min_val::<i8>() % -1; }).join().is_err());
    assert!(thread::spawn(move|| { min_val::<i16>() % -1; }).join().is_err());
    assert!(thread::spawn(move|| { min_val::<i32>() % -1; }).join().is_err());
    assert!(thread::spawn(move|| { min_val::<i64>() % -1; }).join().is_err());
    assert!(thread::spawn(move|| { 1isize % zero(); }).join().is_err());
    assert!(thread::spawn(move|| { 1i8 % zero(); }).join().is_err());
    assert!(thread::spawn(move|| { 1i16 % zero(); }).join().is_err());
    assert!(thread::spawn(move|| { 1i32 % zero(); }).join().is_err());
    assert!(thread::spawn(move|| { 1i64 % zero(); }).join().is_err());
}
