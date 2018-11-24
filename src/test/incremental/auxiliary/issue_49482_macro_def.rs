// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host
// no-prefer-dynamic

#![crate_type="proc-macro"]
#![allow(non_snake_case)]

extern crate proc_macro;

macro_rules! proc_macro_expr_impl {
    ($(
        $( #[$attr:meta] )*
        pub fn $func:ident($input:ident: &str) -> String;
    )+) => {
        $(
            $( #[$attr] )*
            #[proc_macro_derive($func)]
            pub fn $func(_input: ::proc_macro::TokenStream) -> ::proc_macro::TokenStream {
                panic!()
            }
        )+
    };
}

proc_macro_expr_impl! {
    pub fn f1(input: &str) -> String;
    pub fn f2(input: &str) -> String;
    pub fn f3(input: &str) -> String;
    pub fn f4(input: &str) -> String;
    pub fn f5(input: &str) -> String;
    pub fn f6(input: &str) -> String;
    pub fn f7(input: &str) -> String;
    pub fn f8(input: &str) -> String;
    pub fn f9(input: &str) -> String;
    pub fn fA(input: &str) -> String;
    pub fn fB(input: &str) -> String;
    pub fn fC(input: &str) -> String;
    pub fn fD(input: &str) -> String;
    pub fn fE(input: &str) -> String;
    pub fn fF(input: &str) -> String;
}
