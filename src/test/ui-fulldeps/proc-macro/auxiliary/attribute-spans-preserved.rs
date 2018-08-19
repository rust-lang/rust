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

extern crate proc_macro;

use proc_macro::*;

#[proc_macro_attribute]
pub fn foo(attr: TokenStream, f: TokenStream) -> TokenStream {
    let mut tokens = f.into_iter();
    assert_eq!(tokens.next().unwrap().to_string(), "#");
    let next_attr = match tokens.next().unwrap() {
        TokenTree::Group(g) => g,
        _ => panic!(),
    };

    let fn_tok = tokens.next().unwrap();
    let ident_tok = tokens.next().unwrap();
    let args_tok = tokens.next().unwrap();
    let body = tokens.next().unwrap();

    let new_body = attr.into_iter()
        .chain(next_attr.stream().into_iter().skip(1));

    let tokens = vec![
        fn_tok,
        ident_tok,
        args_tok,
        Group::new(Delimiter::Brace, new_body.collect()).into(),
    ].into_iter().collect::<TokenStream>();
    println!("{}", tokens);
    return tokens
}
