// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_fn)]

const fn foo() -> *const i8 {
    b"foo" as *const _ as *const i8
}

const fn bar() -> i32 {
    *&{(1, 2, 3).1}
}

fn main() {
    assert_eq!(foo(), b"foo" as *const _ as *const i8);
    assert_eq!(bar(), 2);
}
