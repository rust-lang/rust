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
extern crate rustc;
extern crate rustc_plugin;

use std::borrow::ToOwned;
use syntax::ast;
use syntax::codemap::Span;
use syntax::ext::build::AstBuilder;
use syntax::ext::base::{TTMacroExpander, ExtCtxt, MacResult, MacEager, NormalTT};
use syntax::parse::token;
use syntax::print::pprust;
use syntax::ptr::P;
use rustc_plugin::Registry;

struct Expander {
    args: Vec<P<ast::MetaItem>>,
}

impl TTMacroExpander for Expander {
    fn expand<'cx>(&self,
                   ecx: &'cx mut ExtCtxt,
                   sp: Span,
                   _: &[ast::TokenTree]) -> Box<MacResult+'cx> {
        let args = self.args.iter().map(|i| pprust::meta_item_to_string(&*i))
            .collect::<Vec<_>>().join(", ");
        let interned = token::intern_and_get_ident(&args[..]);
        MacEager::expr(ecx.expr_str(sp, interned))
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    let args = reg.args().clone();
    reg.register_syntax_extension(token::intern("plugin_args"),
        // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
        NormalTT(Box::new(Expander { args: args, }), None, false));
}
