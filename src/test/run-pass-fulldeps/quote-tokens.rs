// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-cross-compile
#![feature(quote, rustc_private)]

extern crate syntax;

use syntax::ext::base::ExtCtxt;
use syntax::ptr::P;
use syntax::parse::PResult;

fn syntax_extension(cx: &ExtCtxt) {
    let e_toks : Vec<syntax::tokenstream::TokenTree> = quote_tokens!(cx, 1 + 2);
    let p_toks : Vec<syntax::tokenstream::TokenTree> = quote_tokens!(cx, (x, 1 .. 4, *));

    let a: P<syntax::ast::Expr> = quote_expr!(cx, 1 + 2);
    let _b: Option<P<syntax::ast::Item>> = quote_item!(cx, static foo : isize = $e_toks; );
    let _c: P<syntax::ast::Pat> = quote_pat!(cx, (x, 1 .. 4, *) );
    let _d: Option<syntax::ast::Stmt> = quote_stmt!(cx, let x = $a; );
    let _d: syntax::ast::Arm = quote_arm!(cx, (ref x, ref y) = (x, y) );
    let _e: P<syntax::ast::Expr> = quote_expr!(cx, match foo { $p_toks => 10 } );

    let _f: P<syntax::ast::Expr> = quote_expr!(cx, ());
    let _g: P<syntax::ast::Expr> = quote_expr!(cx, true);
    let _h: P<syntax::ast::Expr> = quote_expr!(cx, 'a');

    let i: Option<P<syntax::ast::Item>> = quote_item!(cx, #[derive(Eq)] struct Foo; );
    assert!(i.is_some());

    let _l: P<syntax::ast::Ty> = quote_ty!(cx, &isize);

    let _m: Vec<syntax::tokenstream::TokenTree> = quote_matcher!(cx, $($foo:tt,)* bar);
    let _n: syntax::ast::Attribute = quote_attr!(cx, #![cfg(foo, bar = "baz")]);

    let _o: Option<P<syntax::ast::Item>> = quote_item!(cx, fn foo<T: ?Sized>() {});

    let stmts = vec![
        quote_stmt!(cx, let x = 1;).unwrap(),
        quote_stmt!(cx, let y = 2;).unwrap(),
    ];
    let expr: P<syntax::ast::Expr> = quote_expr!(cx, x + y);
}

fn main() {
}
