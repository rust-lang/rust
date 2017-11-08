// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:attr-cfg.rs
// ignore-stage1
// revisions: foo bar

#![feature(proc_macro)]

extern crate attr_cfg;
use attr_cfg::attr_cfg;

#[attr_cfg]
fn outer() -> u8 {
    #[cfg(foo)]
    fn inner() -> u8 { 1 }

    #[cfg(bar)]
    fn inner() -> u8 { 2 }

    inner()
}

#[cfg(foo)]
fn main() {
    assert_eq!(outer(), 1);
}

#[cfg(bar)]
fn main() {
    assert_eq!(outer(), 2);
}
