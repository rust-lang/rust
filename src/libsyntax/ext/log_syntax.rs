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
use parse::token::{get_ident_interner};

pub fn expand_syntax_ext(cx: @ExtCtxt,
                         sp: codemap::Span,
                         tt: &[ast::token_tree])
                      -> base::MacResult {

    cx.print_backtrace();
    println(
        print::pprust::tt_to_str(
            &ast::tt_delim(@tt.to_owned()),
            get_ident_interner()));

    //trivial expression
    MRExpr(@ast::Expr {
        id: ast::DUMMY_NODE_ID,
        node: ast::ExprLit(@codemap::Spanned {
            node: ast::lit_nil,
            span: sp
        }),
        span: sp,
    })
}
