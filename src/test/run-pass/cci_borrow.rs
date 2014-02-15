// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast - check-fast doesn't understand aux-build
// aux-build:cci_borrow_lib.rs

#[feature(managed_boxes)];

extern crate cci_borrow_lib;
use cci_borrow_lib::foo;

pub fn main() {
    let p = @22u;
    let r = foo(p);
    info!("r={}", r);
    assert_eq!(r, 22u);
}
