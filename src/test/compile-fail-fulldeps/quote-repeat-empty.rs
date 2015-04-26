// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test repeating in a `quote`-related macro.

#![feature(quote)]
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
    let x = quote_tokens!(ecx, $()*);   //~ ERROR attempted to repeat an expression
}
