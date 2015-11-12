// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that `quote`-related macro are gated by `quote` feature gate.

// (To sanity-check the code, uncomment this.)
// #![feature(quote)]

// FIXME the error message that is current emitted seems pretty bad.

#![feature(rustc_private)]
#![allow(dead_code, unused_imports, unused_variables)]

#[macro_use]
extern crate syntax;

use syntax::ast;
use syntax::codemap::Span;
use syntax::parse;

struct ParseSess;

impl ParseSess {
    fn cfg(&self) -> ast::CrateConfig { loop { } }
    fn parse_sess<'a>(&'a self) -> &'a parse::ParseSess { loop { } }
    fn call_site(&self) -> Span { loop { } }
    fn ident_of(&self, st: &str) -> ast::Ident { loop { } }
    fn name_of(&self, st: &str) -> ast::Name { loop { } }
}

pub fn main() {
    let ecx = &ParseSess;
    let x = quote_tokens!(ecx, 3);    //~ ERROR macro undefined: 'quote_tokens!'
    let x = quote_expr!(ecx, 3);      //~ ERROR macro undefined: 'quote_expr!'
    let x = quote_ty!(ecx, 3);        //~ ERROR macro undefined: 'quote_ty!'
    let x = quote_method!(ecx, 3);    //~ ERROR macro undefined: 'quote_method!'
    let x = quote_item!(ecx, 3);      //~ ERROR macro undefined: 'quote_item!'
    let x = quote_pat!(ecx, 3);       //~ ERROR macro undefined: 'quote_pat!'
    let x = quote_arm!(ecx, 3);       //~ ERROR macro undefined: 'quote_arm!'
    let x = quote_stmt!(ecx, 3);      //~ ERROR macro undefined: 'quote_stmt!'
    let x = quote_matcher!(ecx, 3);   //~ ERROR macro undefined: 'quote_matcher!'
    let x = quote_attr!(ecx, 3);      //~ ERROR macro undefined: 'quote_attr!'
    let x = quote_arg!(ecx, 3);       //~ ERROR macro undefined: 'quote_arg!'
    let x = quote_block!(ecx, 3);     //~ ERROR macro undefined: 'quote_block!'
    let x = quote_meta_item!(ecx, 3); //~ ERROR macro undefined: 'quote_meta_item!'
    let x = quote_path!(ecx, 3);      //~ ERROR macro undefined: 'quote_path!'
}
