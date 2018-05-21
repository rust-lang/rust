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
pub fn bar(_input: TokenStream) -> TokenStream {
    let mut ret = Vec::<TokenTree>::new();
    ret.push(Ident::new("static", Span::call_site()).into());
    ret.push(Ident::new("FOO", Span::call_site()).into());
    ret.push(Punct::new(':', Spacing::Alone).into());
    ret.push(Punct::new('&', Spacing::Alone).into());
    ret.push(Punct::new('\'', Spacing::Joint).into());
    ret.push(Ident::new("static", Span::call_site()).into());
    ret.push(Ident::new("i32", Span::call_site()).into());
    ret.push(Punct::new('=', Spacing::Alone).into());
    ret.push(Punct::new('&', Spacing::Alone).into());
    ret.push(Literal::i32_unsuffixed(1).into());
    ret.push(Punct::new(';', Spacing::Alone).into());
    ret.into_iter().collect()
}
