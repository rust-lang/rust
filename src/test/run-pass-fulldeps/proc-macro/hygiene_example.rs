// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:hygiene_example_codegen.rs
// aux-build:hygiene_example.rs
// ignore-stage1

#![feature(use_extern_macros, proc_macro_non_items)]

extern crate hygiene_example;
use hygiene_example::hello;

fn main() {
    mod hygiene_example {} // no conflict with `extern crate hygiene_example;` from the proc macro
    macro_rules! format { () => {} } // does not interfere with `format!` from the proc macro
    macro_rules! hello_helper { () => {} } // similarly does not intefere with the proc macro

    let string = "world"; // no conflict with `string` from the proc macro
    hello!(string);
    hello!(string);
}
