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
use opt_vec;
use parse::token;
use parse::token::{str_to_ident};
use std::vec_ng::Vec;

pub fn expand_syntax_ext(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
    -> base::MacResult {
    let mut res_str = ~"";
    for (i, e) in tts.iter().enumerate() {
        if i & 1 == 1 {
            match *e {
                ast::TTTok(_, token::COMMA) => (),
                _ => {
                    cx.span_err(sp, "concat_idents! expecting comma.");
                    return MacResult::dummy_expr(sp);
                }
            }
        } else {
            match *e {
                ast::TTTok(_, token::IDENT(ident,_)) => {
                    res_str.push_str(token::get_ident(ident).get())
                }
                _ => {
                    cx.span_err(sp, "concat_idents! requires ident args.");
                    return MacResult::dummy_expr(sp);
                }
            }
        }
    }
    let res = str_to_ident(res_str);

    let e = @ast::Expr {
        id: ast::DUMMY_NODE_ID,
        node: ast::ExprPath(
            ast::Path {
                 span: sp,
                 global: false,
                 segments: vec!(
                    ast::PathSegment {
                        identifier: res,
                        lifetimes: Vec::new(),
                        types: opt_vec::Empty,
                    }
                )
            }
        ),
        span: sp,
    };
    MRExpr(e)
}
