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

use proc_macro::*;

#[proc_macro]
pub fn single_quote_alone(_: TokenStream) -> TokenStream {
    // `&'a u8`, but the `'` token is not joint
    let trees: Vec<TokenTree> = vec![
        Punct::new('&', Spacing::Alone).into(),
        Punct::new('\'', Spacing::Alone).into(),
        Ident::new("a", Span::call_site()).into(),
        Ident::new("u8", Span::call_site()).into(),
    ];
    trees.into_iter().collect()
}
