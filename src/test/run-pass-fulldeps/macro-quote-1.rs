// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-stage1

#![feature(plugin)]
#![feature(rustc_private)]
#![plugin(proc_macro_plugin)]

extern crate syntax;
extern crate syntax_pos;

use syntax::ast::Ident;
use syntax::parse::token;
use syntax::tokenstream::TokenTree;

fn main() {
    let true_tok = TokenTree::Token(syntax_pos::DUMMY_SP, token::Ident(Ident::from_str("true")));
    assert!(qquote!(true).eq_unspanned(&true_tok.into()));
}
