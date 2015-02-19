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
#![feature(box_syntax)]

extern crate syntax;
extern crate rustc;

use std::borrow::ToOwned;
use syntax::ast;
use syntax::codemap::Span;
use syntax::ext::build::AstBuilder;
use syntax::ext::base::{TTMacroExpander, ExtCtxt, MacResult, MacExpr, NormalTT};
use syntax::parse::token;
use syntax::print::pprust;
use syntax::ptr::P;
use rustc::plugin::Registry;

struct Expander {
    args: Vec<P<ast::MetaItem>>,
}

impl TTMacroExpander for Expander {
    fn expand<'cx>(&self,
                   ecx: &'cx mut ExtCtxt,
                   sp: Span,
                   _: &[ast::TokenTree]) -> Box<MacResult+'cx> {
        let args = self.args.iter().map(|i| pprust::meta_item_to_string(&*i))
            .collect::<Vec<_>>().connect(", ");
        let interned = token::intern_and_get_ident(&args[..]);
        MacExpr::new(ecx.expr_str(sp, interned))
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    let args = reg.args().clone();
    reg.register_syntax_extension(token::intern("plugin_args"),
        NormalTT(box Expander { args: args, }, None));
}
