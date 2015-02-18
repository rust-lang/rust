// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use codemap;
use ext::base;
use ext::build::AstBuilder;
use parse::token;

use std::string::String;

pub fn expand_syntax_ext(cx: &mut base::ExtCtxt,
                         sp: codemap::Span,
                         tts: &[ast::TokenTree])
                         -> Box<base::MacResult+'static> {
    let es = match base::get_exprs_from_tts(cx, sp, tts) {
        Some(e) => e,
        None => return base::DummyResult::expr(sp)
    };
    let mut accumulator = String::new();
    for e in es {
        match e.node {
            ast::ExprLit(ref lit) => {
                match lit.node {
                    ast::LitStr(ref s, _) |
                    ast::LitFloat(ref s, _) |
                    ast::LitFloatUnsuffixed(ref s) => {
                        accumulator.push_str(&s);
                    }
                    ast::LitChar(c) => {
                        accumulator.push(c);
                    }
                    ast::LitInt(i, ast::UnsignedIntLit(_)) |
                    ast::LitInt(i, ast::SignedIntLit(_, ast::Plus)) |
                    ast::LitInt(i, ast::UnsuffixedIntLit(ast::Plus)) => {
                        accumulator.push_str(&format!("{}", i)[]);
                    }
                    ast::LitInt(i, ast::SignedIntLit(_, ast::Minus)) |
                    ast::LitInt(i, ast::UnsuffixedIntLit(ast::Minus)) => {
                        accumulator.push_str(&format!("-{}", i)[]);
                    }
                    ast::LitBool(b) => {
                        accumulator.push_str(&format!("{}", b)[]);
                    }
                    ast::LitByte(..) |
                    ast::LitBinary(..) => {
                        cx.span_err(e.span, "cannot concatenate a binary literal");
                    }
                }
            }
            _ => {
                cx.span_err(e.span, "expected a literal");
            }
        }
    }
    base::MacExpr::new(cx.expr_str(
            sp,
            token::intern_and_get_ident(&accumulator[..])))
}
