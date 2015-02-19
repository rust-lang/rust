// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use codemap::Span;
use ext::base::*;
use ext::base;
use feature_gate;
use parse::token;
use parse::token::{str_to_ident};
use ptr::P;

pub fn expand_syntax_ext<'cx>(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
                              -> Box<base::MacResult+'cx> {
    if !cx.ecfg.enable_concat_idents() {
        feature_gate::emit_feature_err(&cx.parse_sess.span_diagnostic,
                                       "concat_idents",
                                       sp,
                                       feature_gate::EXPLAIN_CONCAT_IDENTS);
        return base::DummyResult::expr(sp);
    }

    let mut res_str = String::new();
    for (i, e) in tts.iter().enumerate() {
        if i & 1 == 1 {
            match *e {
                ast::TtToken(_, token::Comma) => {},
                _ => {
                    cx.span_err(sp, "concat_idents! expecting comma.");
                    return DummyResult::expr(sp);
                },
            }
        } else {
            match *e {
                ast::TtToken(_, token::Ident(ident, _)) => {
                    res_str.push_str(&token::get_ident(ident))
                },
                _ => {
                    cx.span_err(sp, "concat_idents! requires ident args.");
                    return DummyResult::expr(sp);
                },
            }
        }
    }
    let res = str_to_ident(&res_str[..]);

    let e = P(ast::Expr {
        id: ast::DUMMY_NODE_ID,
        node: ast::ExprPath(
            ast::Path {
                 span: sp,
                 global: false,
                 segments: vec!(
                    ast::PathSegment {
                        identifier: res,
                        parameters: ast::PathParameters::none(),
                    }
                )
            }
        ),
        span: sp,
    });
    MacExpr::new(e)
}
