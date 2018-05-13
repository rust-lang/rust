// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:derive-attr-cfg.rs
// ignore-stage1

#![feature(use_extern_macros)]

extern crate derive_attr_cfg;
use derive_attr_cfg::Foo;

#[derive(Foo)]
#[foo]
struct S {
    #[cfg(any())]
    x: i32
}

fn main() {
}
