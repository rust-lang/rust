// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;
use syntax::codemap::{self, DUMMY_SP, Span, respan};
use syntax::ext::base::ExtCtxt;
use syntax::ext::expand;
use syntax::ext::quote::rt::ToTokens;
use syntax::feature_gate::GatedCfg;
use syntax::parse::ParseSess;
use syntax::ptr::P;

use expr::ExprBuilder;
use invoke::{Invoke, Identity};

/// A Builder for macro invocations.
///
/// Note that there are no commas added between args, as otherwise
/// that macro invocations that could be expressed would be limited.
/// You will need to add all required symbols with `with_arg` or
/// `with_argss`.
pub struct MacBuilder<F=Identity> {
    callback: F,
    span: Span,
    tokens: Vec<ast::TokenTree>,
    path: Option<ast::Path>,
}

impl MacBuilder {
    pub fn new() -> Self {
        MacBuilder::new_with_callback(Identity)
    }
}

impl<F> MacBuilder<F>
    where F: Invoke<ast::Mac>
{
    pub fn new_with_callback(callback: F) -> Self {
        MacBuilder {
            callback: callback,
            span: DUMMY_SP,
            tokens: vec![],
            path: None,
        }
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn path(mut self, path: ast::Path) -> Self {
        self.path = Some(path);
        self
    }

    pub fn build(self) -> F::Result {
        let mac = ast::Mac_::MacInvocTT(
            self.path.expect("No path set for macro"), self.tokens, 0);
        self.callback.invoke(respan(self.span, mac))
    }

    pub fn with_args<I, T>(self, iter: I) -> Self
        where I: IntoIterator<Item=T>, T: ToTokens
    {
        iter.into_iter().fold(self, |self_, expr| self_.with_arg(expr))
    }

    pub fn with_arg<T>(mut self, expr: T) -> Self
        where T: ToTokens
    {
        let parse_sess = ParseSess::new();
        let mut feature_gated_cfgs = Vec::new();
        let cx = make_ext_ctxt(&parse_sess, &mut feature_gated_cfgs);
        let tokens = expr.to_tokens(&cx);
        assert!(tokens.len() == 1);
        self.tokens.push(tokens[0].clone());
        self
    }

    pub fn expr(self) -> ExprBuilder<Self> {
        ExprBuilder::new_with_callback(self)
    }

}

impl<F> Invoke<P<ast::Expr>> for MacBuilder<F>
    where F: Invoke<ast::Mac>,
{
    type Result = Self;

    fn invoke(self, expr: P<ast::Expr>) -> Self {
        self.with_arg(expr)
    }
}

fn make_ext_ctxt<'a>(sess: &'a ParseSess,
                     feature_gated_cfgs: &'a mut Vec<GatedCfg>) -> ExtCtxt<'a> {
    let info = codemap::ExpnInfo {
        call_site: codemap::DUMMY_SP,
        callee: codemap::NameAndSpan {
            name: "test".to_string(),
            format: codemap::MacroAttribute,
            allow_internal_unstable: false,
            span: None
        }
    };

    let cfg = vec![];
    let ecfg = expand::ExpansionConfig::default(String::new());

    let mut cx = ExtCtxt::new(&sess, cfg, ecfg, feature_gated_cfgs);
    cx.bt_push(info);

    cx
}
