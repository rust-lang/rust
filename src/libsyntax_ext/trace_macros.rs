// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ext::base::ExtCtxt;
use syntax::ext::base;
use syntax::feature_gate;
use syntax::symbol::keywords;
use syntax_pos::Span;
use syntax::tokenstream::TokenTree;

pub fn expand_trace_macros(cx: &mut ExtCtxt,
                           sp: Span,
                           tt: &[TokenTree])
                           -> Box<base::MacResult + 'static> {
    if !cx.ecfg.enable_trace_macros() {
        feature_gate::emit_feature_err(&cx.parse_sess,
                                       "trace_macros",
                                       sp,
                                       feature_gate::GateIssue::Language,
                                       feature_gate::EXPLAIN_TRACE_MACROS);
        return base::DummyResult::any(sp);
    }

    match (tt.len(), tt.first()) {
        (1, Some(&TokenTree::Token(_, ref tok))) if tok.is_keyword(keywords::True) => {
            cx.set_trace_macros(true);
        }
        (1, Some(&TokenTree::Token(_, ref tok))) if tok.is_keyword(keywords::False) => {
            cx.set_trace_macros(false);
        }
        _ => cx.span_err(sp, "trace_macros! accepts only `true` or `false`"),
    }

    base::DummyResult::any(sp)
}
