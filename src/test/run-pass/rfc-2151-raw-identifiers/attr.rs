// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(raw_identifiers)]

use std::mem;

#[r#repr(r#C, r#packed)]
struct Test {
    a: bool, b: u64
}

#[r#derive(r#Debug)]
struct Test2(u32);

pub fn main() {
    assert_eq!(mem::size_of::<Test>(), 9);
    assert_eq!("Test2(123)", format!("{:?}", Test2(123)));
}
