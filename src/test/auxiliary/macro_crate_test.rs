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

#![feature(globs, plugin_registrar, macro_rules, quote, managed_boxes)]

extern crate syntax;
extern crate rustc;

use syntax::ast::{TokenTree, Item, MetaItem};
use syntax::codemap::Span;
use syntax::ext::base::*;
use syntax::parse::token;
use rustc::plugin::Registry;

use std::gc::{Gc, GC};

#[macro_export]
macro_rules! exported_macro (() => (2i))

macro_rules! unexported_macro (() => (3i))

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("make_a_1", expand_make_a_1);
    reg.register_syntax_extension(
        token::intern("into_foo"),
        ItemModifier(expand_into_foo));
}

fn expand_make_a_1(cx: &mut ExtCtxt, sp: Span, tts: &[TokenTree])
                   -> Box<MacResult> {
    if !tts.is_empty() {
        cx.span_fatal(sp, "make_a_1 takes no arguments");
    }
    MacExpr::new(quote_expr!(cx, 1i))
}

fn expand_into_foo(cx: &mut ExtCtxt, sp: Span, attr: Gc<MetaItem>, it: Gc<Item>)
                   -> Gc<Item> {
    box(GC) Item {
        attrs: it.attrs.clone(),
        ..(*quote_item!(cx, enum Foo { Bar, Baz }).unwrap()).clone()
    }
}

pub fn foo() {}
