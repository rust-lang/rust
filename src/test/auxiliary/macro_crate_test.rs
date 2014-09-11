// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host

#![feature(globs, plugin_registrar, macro_rules, quote)]

extern crate syntax;
extern crate rustc;

use syntax::ast::{TokenTree, Item, MetaItem};
use syntax::codemap::Span;
use syntax::ext::base::*;
use syntax::parse::token;
use syntax::parse;
use syntax::ptr::P;
use rustc::plugin::Registry;

#[macro_export]
macro_rules! exported_macro (() => (2i))

macro_rules! unexported_macro (() => (3i))

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("make_a_1", expand_make_a_1);
    reg.register_macro("forged_ident", expand_forged_ident);
    reg.register_macro("identity", expand_identity);
    reg.register_syntax_extension(
        token::intern("into_foo"),
        ItemModifier(box expand_into_foo));
}

fn expand_make_a_1(cx: &mut ExtCtxt, sp: Span, tts: &[TokenTree])
                   -> Box<MacResult+'static> {
    if !tts.is_empty() {
        cx.span_fatal(sp, "make_a_1 takes no arguments");
    }
    MacExpr::new(quote_expr!(cx, 1i))
}

// See Issue #15750
fn expand_identity(cx: &mut ExtCtxt, _span: Span, tts: &[TokenTree])
                   -> Box<MacResult+'static> {
    // Parse an expression and emit it unchanged.
    let mut parser = parse::new_parser_from_tts(cx.parse_sess(),
        cx.cfg(), Vec::from_slice(tts));
    let expr = parser.parse_expr();
    MacExpr::new(quote_expr!(&mut *cx, $expr))
}

fn expand_into_foo(cx: &mut ExtCtxt, sp: Span, attr: &MetaItem, it: P<Item>)
                   -> P<Item> {
    P(Item {
        attrs: it.attrs.clone(),
        ..(*quote_item!(cx, enum Foo { Bar, Baz }).unwrap()).clone()
    })
}

fn expand_forged_ident(cx: &mut ExtCtxt, sp: Span, tts: &[TokenTree]) -> Box<MacResult+'static> {
    use syntax::ext::quote::rt::*;

    if !tts.is_empty() {
        cx.span_fatal(sp, "forged_ident takes no arguments");
    }

    // Most of this is modelled after the expansion of the `quote_expr!`
    // macro ...
    let parse_sess = cx.parse_sess();
    let cfg = cx.cfg();

    // ... except this is where we inject a forged identifier,
    // and deliberately do not call `cx.parse_tts_with_hygiene`
    // (because we are testing that this will be *rejected*
    //  by the default parser).

    let expr = {
        let tt = cx.parse_tts("\x00name_2,ctxt_0\x00".to_string());
        let mut parser = new_parser_from_tts(parse_sess, cfg, tt);
        parser.parse_expr()
    };
    MacExpr::new(expr)
}

pub fn foo() {}
