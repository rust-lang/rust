// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:proc-macro-gates.rs

#![feature(use_extern_macros, stmt_expr_attributes)]

extern crate proc_macro_gates as foo;

use foo::*;

// NB. these errors aren't the best errors right now, but they're definitely
// intended to be errors. Somehow using a custom attribute in these positions
// should either require a feature gate or not be allowed on stable.

fn _test6<#[a] T>() {}
//~^ ERROR: unknown to the compiler

fn _test7() {
    match 1 {
        #[a] //~ ERROR: unknown to the compiler
        0 => {}
        _ => {}
    }
}

fn main() {
}
