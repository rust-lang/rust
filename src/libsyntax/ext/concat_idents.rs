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
use codemap::span;
use ext::base::*;
use ext::base;
use parse::token;
use parse::token::{str_to_ident};

pub fn expand_syntax_ext(cx: @ExtCtxt, sp: span, tts: &[ast::token_tree])
    -> base::MacResult {
    let mut res_str = ~"";
    for tts.iter().enumerate().advance |(i, e)| {
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

    let e = @ast::expr {
        id: cx.next_id(),
        node: ast::expr_path(
            @ast::Path {
                 span: sp,
                 global: false,
                 idents: ~[res],
                 rp: None,
                 types: ~[],
            }
        ),
        span: sp,
    };
    MRExpr(e)
}
