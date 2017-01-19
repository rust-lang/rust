// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:attr_proc_macro.rs

#![feature(proc_macro, custom_attribute)]
//~^ ERROR Cannot use `#![feature(proc_macro)]` and `#![feature(custom_attribute)] at the same time

extern crate attr_proc_macro;
use attr_proc_macro::attr_proc_macro;

#[attr_proc_macro]
fn foo() {}

fn main() {
    foo();
}
