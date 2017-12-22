// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(const_err)]

use std::{isize, i8, i16, i32, i64};
use std::thread;

fn main() {
    assert!(thread::spawn(move|| { isize::MIN / -1; }).join().is_err());
    //~^ ERROR attempt to divide with overflow
    assert!(thread::spawn(move|| { i8::MIN / -1; }).join().is_err());
    //~^ ERROR attempt to divide with overflow
    assert!(thread::spawn(move|| { i16::MIN / -1; }).join().is_err());
    //~^ ERROR attempt to divide with overflow
    assert!(thread::spawn(move|| { i32::MIN / -1; }).join().is_err());
    //~^ ERROR attempt to divide with overflow
    assert!(thread::spawn(move|| { i64::MIN / -1; }).join().is_err());
    //~^ ERROR attempt to divide with overflow
    assert!(thread::spawn(move|| { 1isize / 0; }).join().is_err());
    //~^ ERROR attempt to divide by zero
    assert!(thread::spawn(move|| { 1i8 / 0; }).join().is_err());
    //~^ ERROR attempt to divide by zero
    assert!(thread::spawn(move|| { 1i16 / 0; }).join().is_err());
    //~^ ERROR attempt to divide by zero
    assert!(thread::spawn(move|| { 1i32 / 0; }).join().is_err());
    //~^ ERROR attempt to divide by zero
    assert!(thread::spawn(move|| { 1i64 / 0; }).join().is_err());
    //~^ ERROR attempt to divide by zero
    assert!(thread::spawn(move|| { isize::MIN % -1; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with overflow
    assert!(thread::spawn(move|| { i8::MIN % -1; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with overflow
    assert!(thread::spawn(move|| { i16::MIN % -1; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with overflow
    assert!(thread::spawn(move|| { i32::MIN % -1; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with overflow
    assert!(thread::spawn(move|| { i64::MIN % -1; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with overflow
    assert!(thread::spawn(move|| { 1isize % 0; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with a divisor of zero
    assert!(thread::spawn(move|| { 1i8 % 0; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with a divisor of zero
    assert!(thread::spawn(move|| { 1i16 % 0; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with a divisor of zero
    assert!(thread::spawn(move|| { 1i32 % 0; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with a divisor of zero
    assert!(thread::spawn(move|| { 1i64 % 0; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with a divisor of zero
}
