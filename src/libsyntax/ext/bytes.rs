// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/* The compiler code necessary to support the bytes! extension. */

use ast;
use codemap::Span;
use ext::base::*;
use ext::base;
use ext::build::AstBuilder;

use std::char;

pub fn expand_syntax_ext(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree]) -> base::MacResult {
    // Gather all argument expressions
    let exprs = match get_exprs_from_tts(cx, sp, tts) {
        None => return MacResult::dummy_expr(),
        Some(e) => e,
    };
    let mut bytes = ~[];

    for expr in exprs.iter() {
        match expr.node {
            // expression is a literal
            ast::ExprLit(lit) => match lit.node {
                // string literal, push each byte to vector expression
                ast::LitStr(s, _) => {
                    for byte in s.bytes() {
                        bytes.push(cx.expr_u8(expr.span, byte));
                    }
                }

                // u8 literal, push to vector expression
                ast::LitUint(v, ast::TyU8) => {
                    if v > 0xFF {
                        cx.span_err(expr.span, "Too large u8 literal in bytes!")
                    } else {
                        bytes.push(cx.expr_u8(expr.span, v as u8));
                    }
                }

                // integer literal, push to vector expression
                ast::LitIntUnsuffixed(v) => {
                    if v > 0xFF {
                        cx.span_err(expr.span, "Too large integer literal in bytes!")
                    } else if v < 0 {
                        cx.span_err(expr.span, "Negative integer literal in bytes!")
                    } else {
                        bytes.push(cx.expr_u8(expr.span, v as u8));
                    }
                }

                // char literal, push to vector expression
                ast::LitChar(v) => {
                    if char::from_u32(v).unwrap().is_ascii() {
                        bytes.push(cx.expr_u8(expr.span, v as u8));
                    } else {
                        cx.span_err(expr.span, "Non-ascii char literal in bytes!")
                    }
                }

                _ => cx.span_err(expr.span, "Unsupported literal in bytes!")
            },

            _ => cx.span_err(expr.span, "Non-literal in bytes!")
        }
    }

    let e = cx.expr_vec_slice(sp, bytes);
    MRExpr(e)
}
