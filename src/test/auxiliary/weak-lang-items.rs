// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic

// This aux-file will require the eh_personality function to be codegen'd, but
// it hasn't been defined just yet. Make sure we don't explode.

#![no_std]
#![feature(phase)]
#![crate_type = "rlib"]

#[phase(plugin, link)]
extern crate core;

struct A;

impl core::ops::Drop for A {
    fn drop(&mut self) {}
}

pub fn foo() {
    let _a = A;
    fail!("wut");
}

mod std {
    pub use core::{option, fmt};
}

