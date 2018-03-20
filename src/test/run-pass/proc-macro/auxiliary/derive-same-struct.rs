// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic
// compile-flags:--crate-type proc-macro

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(AToB)]
pub fn derive1(input: TokenStream) -> TokenStream {
    println!("input1: {:?}", input.to_string());
    assert_eq!(input.to_string(), "struct A;");
    "#[derive(BToC)] struct B;".parse().unwrap()
}

#[proc_macro_derive(BToC)]
pub fn derive2(input: TokenStream) -> TokenStream {
    assert_eq!(input.to_string(), "struct B;");
    "struct C;".parse().unwrap()
}
