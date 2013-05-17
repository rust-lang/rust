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
use codemap::span;
use ext::base::*;
use ext::base;
use ext::build::{mk_u8, mk_slice_vec_e};

pub fn expand_syntax_ext(cx: @ext_ctxt, sp: span, tts: &[ast::token_tree]) -> base::MacResult {
    // Gather all argument expressions
    let exprs = get_exprs_from_tts(cx, tts);
    let mut bytes = ~[];

    for exprs.each |expr| {
        match expr.node {
            // expression is a literal
            ast::expr_lit(lit) => match lit.node {
                // string literal, push each byte to vector expression
                ast::lit_str(s) => {
                    for s.each |byte| {
                        bytes.push(mk_u8(cx, sp, byte));
                    }
                }

                // u8 literal, push to vector expression
                ast::lit_uint(v, ast::ty_u8) => {
                    if v > 0xFF {
                        cx.span_err(sp, "Too large u8 literal in bytes!")
                    } else {
                        bytes.push(mk_u8(cx, sp, v as u8));
                    }
                }

                // integer literal, push to vector expression
                ast::lit_int_unsuffixed(v) => {
                    if v > 0xFF {
                        cx.span_err(sp, "Too large integer literal in bytes!")
                    } else if v < 0 {
                        cx.span_err(sp, "Negative integer literal in bytes!")
                    } else {
                        bytes.push(mk_u8(cx, sp, v as u8));
                    }
                }

                // char literal, push to vector expression
                ast::lit_int(v, ast::ty_char) => {
                    if (v as char).is_ascii() {
                        bytes.push(mk_u8(cx, sp, v as u8));
                    } else {
                        cx.span_err(sp, "Non-ascii char literal in bytes!")
                    }
                }

                _ => cx.span_err(sp, "Unsupported literal in bytes!")
            },

            _ => cx.span_err(sp, "Non-literal in bytes!")
        }
    }

    let e = mk_slice_vec_e(cx, sp, bytes);
    MRExpr(e)
}
