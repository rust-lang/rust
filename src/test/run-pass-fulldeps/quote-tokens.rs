// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
#[feature(quote)];
#[feature(managed_boxes)];

extern mod syntax;

use syntax::ext::base::ExtCtxt;

fn syntax_extension(cx: &ExtCtxt) {
    let e_toks : ~[syntax::ast::token_tree] = quote_tokens!(cx, 1 + 2);
    let p_toks : ~[syntax::ast::token_tree] = quote_tokens!(cx, (x, 1 .. 4, *));

    let a: @syntax::ast::Expr = quote_expr!(cx, 1 + 2);
    let _b: Option<@syntax::ast::item> = quote_item!(cx, static foo : int = $e_toks; );
    let _c: @syntax::ast::Pat = quote_pat!(cx, (x, 1 .. 4, *) );
    let _d: @syntax::ast::Stmt = quote_stmt!(cx, let x = $a; );
    let _e: @syntax::ast::Expr = quote_expr!(cx, match foo { $p_toks => 10 } );
}

fn main() {
}
