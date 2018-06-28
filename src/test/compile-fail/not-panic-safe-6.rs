// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

use std::panic::UnwindSafe;
use std::cell::RefCell;

fn assert<T: UnwindSafe + ?Sized>() {}

fn main() {
    assert::<*mut RefCell<i32>>();
    //~^ ERROR the type `std::cell::UnsafeCell<i32>` may contain interior mutability and a
    //~| ERROR the type `std::cell::UnsafeCell<isize>` may contain interior mutability and a
}
