// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![feature(core)]

use std::ops::Add;

extern "C" fn foo<T: Add>(a: T, b: T) -> T::Output { a + b }

fn main() {
    assert_eq!(100u8, foo(0u8, 100u8));
    assert_eq!(100u16, foo(0u16, 100u16));
}
