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

#![crate_type = "proc-macro"]
#![feature(proc_macro)]

extern crate proc_macro;

use proc_macro::*;

#[proc_macro]
pub fn tokens(input: TokenStream) -> TokenStream {
    assert_nothing_joint(input);
    TokenStream::new()
}

#[proc_macro_attribute]
pub fn nothing(_: TokenStream, input: TokenStream) -> TokenStream {
    assert_nothing_joint(input);
    TokenStream::new()
}

fn assert_nothing_joint(s: TokenStream) {
    for tt in s {
        match tt {
            TokenTree::Group(g) => assert_nothing_joint(g.stream()),
            TokenTree::Punct(p) => assert_eq!(p.spacing(), Spacing::Alone),
            _ => {}
        }
    }
}
