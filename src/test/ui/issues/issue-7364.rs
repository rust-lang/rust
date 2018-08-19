// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

use std::cell::RefCell;

// Regression test for issue 7364
static boxed: Box<RefCell<isize>> = box RefCell::new(0);
//~^ ERROR allocations are not allowed in statics
//~| ERROR `std::cell::RefCell<isize>` cannot be shared between threads safely [E0277]

fn main() { }
