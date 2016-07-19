// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host

#![feature(plugin_registrar, quote, rustc_private)]
#![crate_type = "dylib"]

extern crate syntax;
extern crate rustc;
extern crate rustc_plugin;
extern crate syntax_pos;

use syntax::ast;
use syntax::ext::base::{ExtCtxt, MacResult, MacEager};
use syntax::util::small_vector::SmallVector;
use syntax::tokenstream;
use rustc_plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("multiple_items", expand)
}

fn expand(cx: &mut ExtCtxt, _: syntax_pos::Span, _: &[tokenstream::TokenTree])
          -> Box<MacResult+'static> {
    MacEager::items(SmallVector::many(vec![
        quote_item!(cx, struct Struct1;).unwrap(),
        quote_item!(cx, struct Struct2;).unwrap()
    ]))
}
