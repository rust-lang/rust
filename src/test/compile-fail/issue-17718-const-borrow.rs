// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::UnsafeCell;

const A: UnsafeCell<uint> = UnsafeCell { value: 1 };
const B: &'static UnsafeCell<uint> = &A;
//~^ ERROR: cannot borrow a constant which contains interior mutability

struct C { a: UnsafeCell<uint> }
const D: C = C { a: UnsafeCell { value: 1 } };
const E: &'static UnsafeCell<uint> = &D.a;
//~^ ERROR: cannot borrow a constant which contains interior mutability
const F: &'static C = &D;
//~^ ERROR: cannot borrow a constant which contains interior mutability

fn main() {}
