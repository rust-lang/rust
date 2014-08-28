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
use parse::token::{keywords, is_keyword};


pub fn expand_trace_macros(cx: &mut ExtCtxt,
                           sp: Span,
                           tt: &[ast::TokenTree])
                           -> Box<base::MacResult+'static> {
    match tt {
        [ast::TTTok(_, ref tok)] if is_keyword(keywords::True, tok) => {
            cx.set_trace_macros(true);
        }
        [ast::TTTok(_, ref tok)] if is_keyword(keywords::False, tok) => {
            cx.set_trace_macros(false);
        }
        _ => cx.span_err(sp, "trace_macros! accepts only `true` or `false`"),
    }

    base::DummyResult::any(sp)
}
