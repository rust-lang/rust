// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![feature(stmt_expr_attributes)]
#![warn(unused_attributes)] //~ NOTE lint level defined here

fn foo<#[derive(Debug)] T>() { //~ WARN unused attribute
    match 0 {
        #[derive(Debug)] //~ WARN unused attribute
        _ => (),
    }
}

fn main() {
    // fold_stmt (Item)
    #[allow(dead_code)]
    #[derive(Debug)] // should not warn
    struct Foo;

    // fold_stmt (Mac)
    #[derive(Debug)]
    //~^ WARN `#[derive]` does nothing on macro invocations
    //~| NOTE this may become a hard error in a future release
    println!("Hello, world!");

    // fold_stmt (Semi)
    #[derive(Debug)] //~ WARN unused attribute
    "Hello, world!";

    // fold_stmt (Local)
    #[derive(Debug)] //~ WARN unused attribute
    let _ = "Hello, world!";

    // fold_expr
    let _ = #[derive(Debug)] "Hello, world!";
    //~^ WARN unused attribute

    let _ = [
        // fold_opt_expr
        #[derive(Debug)] //~ WARN unused attribute
        "Hello, world!"
    ];
}
