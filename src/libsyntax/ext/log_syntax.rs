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
use codemap;
use ext::base::*;
use ext::base;
use print;

pub fn expand_syntax_ext(cx: @ext_ctxt,
                         sp: codemap::span,
                         tt: &[ast::token_tree])
                      -> base::MacResult {

    cx.print_backtrace();
    io::stdout().write_line(
        print::pprust::tt_to_str(
            ast::tt_delim(vec::from_slice(tt)),
            cx.parse_sess().interner));

    //trivial expression
    MRExpr(@ast::expr {
        id: cx.next_id(),
        callee_id: cx.next_id(),
        node: ast::expr_lit(@codemap::spanned {
            node: ast::lit_nil,
            span: sp
        }),
        span: sp,
    })
}
