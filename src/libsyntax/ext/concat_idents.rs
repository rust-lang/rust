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

pub fn expand_syntax_ext(cx: @ExtCtxt, sp: Span, tts: &[ast::token_tree])
    -> base::MacResult {
    let mut res_str = ~"";
    for (i, e) in tts.iter().enumerate() {
        if i & 1 == 1 {
            match *e {
                ast::tt_tok(_, token::COMMA) => (),
                _ => cx.span_fatal(sp, "concat_idents! expecting comma.")
            }
        } else {
            match *e {
                ast::tt_tok(_, token::IDENT(ident,_)) => res_str.push_str(cx.str_of(ident)),
                _ => cx.span_fatal(sp, "concat_idents! requires ident args.")
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
                 segments: ~[
                    ast::PathSegment {
                        identifier: res,
                        lifetimes: opt_vec::Empty,
                        types: opt_vec::Empty,
                    }
                ]
            }
        ),
        span: sp,
    };
    MRExpr(e)
}
