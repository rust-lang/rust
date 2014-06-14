// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/* The compiler code necessary to support the hex_bytes! extension. */

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
                // Only string literals are supported, it is not clear what to do in other
                // cases. Probably what the user wanted was to use bytes! so recommend
                // that instead
                ast::LitStr(ref s, _) => {
                    let mut iter = s.get().chars();
                    loop {
                        let first = iter.next();
                        // If no first character, we're done
                        if first.is_none() {
                            break;
                        }
                        // If no second, there was an odd read
                        let second = iter.next();
                        if second.is_none() {
                            cx.span_err(expr.span, "uneven number of hex digits in hex_bytes!");
                            err = true;
                            break;
                        }
                        // If we have both, we are golden
                        let (hi, lo) = (first.unwrap().to_digit(16), second.unwrap().to_digit(16));
                        if hi.is_none() || lo.is_none() {
                            cx.span_err(expr.span, "invalid character in hex_bytes!");
                            err = true;
                            break;
                        }
                        // If we have both, -and- they successfully decoded
                        bytes.push(cx.expr_u8(expr.span, (hi.unwrap() * 16 + lo.unwrap()) as u8));
                    }
                }
                _ => {
                    cx.span_err(expr.span, "non-string in hex_bytes!. Perhaps you meant bytes!?");
                    err = true;
                }
            },
            _ => {
                cx.span_err(expr.span, "non-literal in hex_bytes!");
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
