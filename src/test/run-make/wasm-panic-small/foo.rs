// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "cdylib"]

#[no_mangle]
#[cfg(a)]
pub fn foo() {
    panic!("test");
}

#[no_mangle]
#[cfg(b)]
pub fn foo() {
    panic!("{}", 1);
}

#[no_mangle]
#[cfg(c)]
pub fn foo() {
    panic!("{}", "a");
}

#[no_mangle]
#[cfg(d)]
pub fn foo() -> usize {
    use std::cell::Cell;
    thread_local!(static A: Cell<Vec<u32>> = Cell::new(Vec::new()));
    A.try_with(|x| x.replace(Vec::new()).len()).unwrap_or(0)
}
