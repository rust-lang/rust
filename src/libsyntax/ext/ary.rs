// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of the ary![] macro

use ast;
use codemap::Span;
use ext::base;
use ext::base::{ExtCtxt, MacExpr, MacResult, DummyResult};
use ext::build::AstBuilder;
use parse;
use parse::token;

pub fn expand_ary(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
                 -> Box<base::MacResult> {
    let mut p = parse::new_parser_from_tts(cx.parse_sess(), cx.cfg(), Vec::from_slice(tts));
    let val_expr = p.parse_expr();
    p.expect(&token::COMMA);
    p.expect(&token::DOTDOT);
    // negative literals should not be a fatal error
    if p.eat(&token::BINOP(token::MINUS)) {
        p.span_err(p.last_span, "expected positive integral literal");
        return DummyResult::expr(sp);
    }
    let lit = p.parse_lit();
    let count = match lit.node {
        ast::LitInt(i, _) | ast::LitIntUnsuffixed(i) if i > 0 => i as u64,
        ast::LitUint(u, _) if u > 0 => u,
        _ => {
            p.span_err(lit.span, "expected positive integral literal");
            return DummyResult::expr(sp);
        }
    };
    let count = match count.to_uint() {
        None => {
            p.span_err(lit.span, "integral literal out of range");
            return DummyResult::expr(sp);
        }
        Some(x) => x
    };
    let exprs = Vec::from_fn(count, |_| val_expr.clone());
    MacExpr::new(cx.expr_vec(sp, exprs))
}
