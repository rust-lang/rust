// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
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

#![crate_type = "dylib"]
#![feature(macro_registrar, quote, globs)]

extern crate other = "syntax-extension-with-dll-deps-1";
extern crate syntax;

use syntax::ast::{Name, TokenTree, Item, MetaItem};
use syntax::codemap::Span;
use syntax::ext::base::*;
use syntax::parse::token;

#[macro_registrar]
pub fn macro_registrar(register: |Name, SyntaxExtension|) {
    register(token::intern("foo"),
        NormalTT(~BasicMacroExpander {
            expander: expand_foo,
            span: None,
        },
        None));
}

fn expand_foo(cx: &mut ExtCtxt, sp: Span, tts: &[TokenTree]) -> ~MacResult {
    let answer = other::the_answer();
    MacExpr::new(quote_expr!(cx, $answer))
}
