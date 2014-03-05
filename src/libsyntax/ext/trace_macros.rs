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
use ext::base::ExtCtxt;
use ext::base;
use parse::lexer::{new_tt_reader, Reader};
use parse::parser::Parser;
use parse::token::keywords;

pub fn expand_trace_macros(cx: &mut ExtCtxt,
                           sp: Span,
                           tt: &[ast::TokenTree])
                        -> base::MacResult {
    let sess = cx.parse_sess();
    let cfg = cx.cfg();
    let tt_rdr = new_tt_reader(cx.parse_sess().span_diagnostic,
                               None,
                               tt.iter().map(|x| (*x).clone()).collect());
    let mut rust_parser = Parser(sess, cfg.clone(), tt_rdr.dup());

    if rust_parser.is_keyword(keywords::True) {
        cx.set_trace_macros(true);
    } else if rust_parser.is_keyword(keywords::False) {
        cx.set_trace_macros(false);
    } else {
        cx.span_err(sp, "trace_macros! only accepts `true` or `false`");
        return base::MacResult::dummy_expr(sp);
    }

    rust_parser.bump();

    let mut rust_parser = Parser(sess, cfg, tt_rdr.dup());
    let result = rust_parser.parse_expr();
    base::MRExpr(result)
}
