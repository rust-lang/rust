// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host

#![feature(plugin_registrar)]
#![feature(box_syntax, rustc_private)]

extern crate syntax;
extern crate syntax_pos;
extern crate rustc;
extern crate rustc_plugin;

use std::borrow::ToOwned;
use syntax::ast;
use syntax::ext::build::AstBuilder;
use syntax::ext::base::{TTMacroExpander, ExtCtxt, MacResult, MacEager, NormalTT};
use syntax::print::pprust;
use syntax::ptr::P;
use syntax::symbol::Symbol;
use syntax_pos::Span;
use syntax::tokenstream;
use rustc_plugin::Registry;

struct Expander {
    args: Vec<ast::NestedMetaItem>,
}

impl TTMacroExpander for Expander {
    fn expand<'cx>(&self,
                   ecx: &'cx mut ExtCtxt,
                   sp: Span,
                   _: &[tokenstream::TokenTree]) -> Box<MacResult+'cx> {
        let args = self.args.iter().map(|i| pprust::meta_list_item_to_string(i))
            .collect::<Vec<_>>().join(", ");
        MacEager::expr(ecx.expr_str(sp, Symbol::intern(&args)))
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    let args = reg.args().to_owned();
    reg.register_syntax_extension(Symbol::intern("plugin_args"),
        // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
        NormalTT(Box::new(Expander { args: args, }), None, false));
}
