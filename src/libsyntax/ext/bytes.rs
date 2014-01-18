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


pub fn expand_syntax_ext(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
                         -> Box<base::MacResult> {
    // Gather all argument expressions
    let exprs = match get_exprs_from_tts(cx, sp, tts) {
        None => return DummyResult::expr(sp),
        Some(e) => e,
    };
    let mut bytes = Vec::new();
    let mut err = false;

    for expr in exprs.iter() {
        match expr.node {
            // expression is a literal
            ast::ExprLit(lit) => match lit.node {
                // string literal, push each byte to vector expression
                ast::LitStr(ref s, _) => {
                    for byte in s.get().bytes() {
                        bytes.push(cx.expr_u8(expr.span, byte));
                    }
                }

                // u8 literal, push to vector expression
                ast::LitUint(v, ast::TyU8) => {
                    if v > 0xFF {
                        cx.span_err(expr.span, "too large u8 literal in bytes!");
                        err = true;
                    } else {
                        bytes.push(cx.expr_u8(expr.span, v as u8));
                    }
                }

                // integer literal, push to vector expression
                ast::LitIntUnsuffixed(v) => {
                    if v > 0xFF {
                        cx.span_err(expr.span, "too large integer literal in bytes!");
                        err = true;
                    } else if v < 0 {
                        cx.span_err(expr.span, "negative integer literal in bytes!");
                        err = true;
                    } else {
                        bytes.push(cx.expr_u8(expr.span, v as u8));
                    }
                }

                // char literal, push to vector expression
                ast::LitChar(v) => {
                    if v.is_ascii() {
                        bytes.push(cx.expr_u8(expr.span, v as u8));
                    } else {
                        cx.span_err(expr.span, "non-ascii char literal in bytes!");
                        err = true;
                    }
                }

                _ => {
                    cx.span_err(expr.span, "unsupported literal in bytes!");
                    err = true;
                }
            },

            _ => {
                cx.span_err(expr.span, "non-literal in bytes!");
                err = true;
            }
        }
    }

    // For some reason using quote_expr!() here aborts if we threw an error.
    // I'm assuming that the end of the recursive parse tricks the compiler
    // into thinking this is a good time to stop. But we'd rather keep going.
    if err {
        // Since the compiler will stop after the macro expansion phase anyway, we
        // don't need type info, so we can just return a DummyResult
        return DummyResult::expr(sp);
    }

    let e = cx.expr_vec_slice(sp, bytes);
    let e = quote_expr!(cx, { static BYTES: &'static [u8] = $e; BYTES});
    MacExpr::new(e)
}
