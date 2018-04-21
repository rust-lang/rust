// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:attr-stmt-expr.rs
// ignore-stage1

#![feature(proc_macro, proc_macro_expr)]

extern crate attr_stmt_expr;
use attr_stmt_expr::{expect_let, expect_print_stmt, expect_expr, expect_print_expr};

fn print_str(string: &'static str) {
    // macros are handled a bit differently
    #[expect_print_expr]
    //~^ ERROR attributes on expressions are experimental
    //~| HELP add #![feature(stmt_expr_attributes)] to the crate attributes to enable
    println!("{}", string)
}

fn main() {
    #[expect_let]
    let string = "Hello, world!";

    #[expect_print_stmt]
    println!("{}", string);

    #[expect_expr]
    //~^ ERROR attributes on expressions are experimental
    //~| HELP add #![feature(stmt_expr_attributes)] to the crate attributes to enable
    print_str("string")
}
