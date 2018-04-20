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

//! Attributes producing expressions in invalid locations

#![feature(proc_macro, stmt_expr_attributes, proc_macro_expr)]

extern crate attr_stmt_expr;
use attr_stmt_expr::{duplicate, no_output};

fn main() {
    let _ = #[no_output] "Hello, world!";
    //~^ ERROR expected expression, found `<eof>`

    let _ = #[duplicate] "Hello, world!";
    //~^ ERROR macro expansion ignores token `,` and any following

    let _ = {
        #[no_output]
        "Hello, world!"
    };

    let _ = {
        #[duplicate]
        //~^ ERROR macro expansion ignores token `,` and any following
        "Hello, world!"
    };
}
