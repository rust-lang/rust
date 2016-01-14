// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:const_fn_lib.rs

// A very basic test of const fn functionality.

#![feature(const_fn)]

extern crate const_fn_lib;

use const_fn_lib::foo;

const FOO: usize = foo();

fn main() {
    assert_eq!(FOO, 22);
    let _: [i32; foo()] = [42; 22];
}
