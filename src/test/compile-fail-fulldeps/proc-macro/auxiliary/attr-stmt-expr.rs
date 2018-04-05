// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic

#![feature(proc_macro)]
#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn expect_let(attr: TokenStream, item: TokenStream) -> TokenStream {
    assert!(attr.to_string().is_empty());
    assert_eq!(item.to_string(), "let string = \"Hello, world!\";");
    item
}

#[proc_macro_attribute]
pub fn expect_print_stmt(attr: TokenStream, item: TokenStream) -> TokenStream {
    assert!(attr.to_string().is_empty());
    assert_eq!(item.to_string(), "println!(\"{}\" , string);");
    item
}

#[proc_macro_attribute]
pub fn expect_expr(attr: TokenStream, item: TokenStream) -> TokenStream {
    assert!(attr.to_string().is_empty());
    assert_eq!(item.to_string(), "print_str(\"string\")");
    item
}

#[proc_macro_attribute]
pub fn expect_print_expr(attr: TokenStream, item: TokenStream) -> TokenStream {
    assert!(attr.to_string().is_empty());
    assert_eq!(item.to_string(), "println!(\"{}\" , string)");
    item
}

#[proc_macro_attribute]
pub fn duplicate(attr: TokenStream, item: TokenStream) -> TokenStream {
    assert!(attr.to_string().is_empty());
    format!("{}, {}", item, item).parse().unwrap()
}

#[proc_macro_attribute]
pub fn no_output(attr: TokenStream, item: TokenStream) -> TokenStream {
    assert!(attr.to_string().is_empty());
    assert!(!item.to_string().is_empty());
    "".parse().unwrap()
}
