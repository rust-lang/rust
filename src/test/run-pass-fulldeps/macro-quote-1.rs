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

use syntax::ast::{Ident, Name};
use syntax::parse::token::{self, Token, Lit};
use syntax::tokenstream::TokenTree;

fn main() {
    let true_tok = token::Ident(Ident::from_str("true"));
    assert!(quote!(true).eq_unspanned(&true_tok.into()));

    // issue #35829, extended check to proc_macro.
    let triple_dot_tok = Token::DotDotDot;
    assert!(quote!(...).eq_unspanned(&triple_dot_tok.into()));

    let byte_str_tok = Token::Literal(Lit::ByteStr(Name::intern("one")), None);
    assert!(quote!(b"one").eq_unspanned(&byte_str_tok.into()));

    let byte_str_raw_tok = Token::Literal(Lit::ByteStrRaw(Name::intern("#\"two\"#"), 3), None);
    assert!(quote!(br###"#"two"#"###).eq_unspanned(&byte_str_raw_tok.into()));

    let str_raw_tok = Token::Literal(Lit::StrRaw(Name::intern("#\"three\"#"), 2), None);
    assert!(quote!(r##"#"three"#"##).eq_unspanned(&str_raw_tok.into()));
}
