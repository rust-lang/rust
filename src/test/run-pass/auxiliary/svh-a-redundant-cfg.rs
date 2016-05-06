// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The `svh-a-*.rs` files are all deviations from the base file
//! svh-a-base.rs with some difference (usually in `fn foo`) that
//! should not affect the strict version hash (SVH) computation
//! (#14132).

#![crate_name = "a"]

macro_rules! three {
    () => { 3 }
}

pub trait U {}
pub trait V {}
impl U for () {}
impl V for () {}

static A_CONSTANT : isize = 2;

// cfg attribute does not affect the svh, as long as it yields the same code.
#[cfg(not(an_unused_name))]
pub fn foo<T:U>(_: isize) -> isize {
    3
}

pub fn an_unused_name() -> isize {
    4
}
