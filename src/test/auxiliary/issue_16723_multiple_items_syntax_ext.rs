// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-stage1
// force-host

#![feature(plugin_registrar, managed_boxes, quote)]
#![crate_type = "dylib"]

extern crate syntax;
extern crate rustc;

use syntax::ast;
use syntax::codemap;
use syntax::ext::base::{ExtCtxt, MacResult, MacItems};
use rustc::plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("multiple_items", expand)
}

fn expand(cx: &mut ExtCtxt, _: codemap::Span, _: &[ast::TokenTree]) -> Box<MacResult+'static> {
    MacItems::new(vec![
        quote_item!(cx, struct Struct1;).unwrap(),
        quote_item!(cx, struct Struct2;).unwrap()
    ].into_iter())
}
