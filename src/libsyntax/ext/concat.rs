// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::char;

use ast;
use codemap;
use ext::base;
use ext::build::AstBuilder;

pub fn expand_syntax_ext(cx: @base::ExtCtxt,
                         sp: codemap::Span,
                         tts: &[ast::token_tree]) -> base::MacResult {
    let es = base::get_exprs_from_tts(cx, sp, tts);
    let mut accumulator = ~"";
    for e in es.move_iter() {
        let e = cx.expand_expr(e);
        match e.node {
            ast::ExprLit(lit) => {
                match lit.node {
                    ast::lit_str(s, _) |
                    ast::lit_float(s, _) |
                    ast::lit_float_unsuffixed(s) => {
                        accumulator.push_str(s);
                    }
                    ast::lit_char(c) => {
                        accumulator.push_char(char::from_u32(c).unwrap());
                    }
                    ast::lit_int(i, _) |
                    ast::lit_int_unsuffixed(i) => {
                        accumulator.push_str(format!("{}", i));
                    }
                    ast::lit_uint(u, _) => {
                        accumulator.push_str(format!("{}", u));
                    }
                    ast::lit_nil => {}
                    ast::lit_bool(b) => {
                        accumulator.push_str(format!("{}", b));
                    }
                    ast::lit_binary(..) => {
                        cx.span_err(e.span, "cannot concatenate a binary literal");
                    }
                }
            }
            _ => {
                cx.span_err(e.span, "expected a literal");
            }
        }
    }
    return base::MRExpr(cx.expr_str(sp, accumulator.to_managed()));
}
