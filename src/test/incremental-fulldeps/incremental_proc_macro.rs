// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:incremental_proc_macro_aux.rs
// ignore-stage1
// revisions: cfail1 cfail2
// compile-pass

// This test makes sure that we still find the proc-macro registrar function
// when we compile proc-macros incrementally (see #47292).

#![crate_type = "rlib"]

#[macro_use]
extern crate incremental_proc_macro_aux;

#[derive(IncrementalMacro)]
pub struct Foo {
    x: u32
}
