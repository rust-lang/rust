// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
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

use syntax::codemap::DUMMY_SP;
use syntax::print::pprust::*;

fn main() {
    let ps = syntax::parse::ParseSess::new();
    let mut cx = syntax::ext::base::ExtCtxt::new(
        &ps, vec![],
        syntax::ext::expand::ExpansionConfig::default("qquote".to_string()));
    cx.bt_push(syntax::codemap::ExpnInfo {
        call_site: DUMMY_SP,
        callee: syntax::codemap::NameAndSpan {
            name: "".to_string(),
            format: syntax::codemap::MacroBang,
            allow_internal_unstable: false,
            span: None,
        }
    });
    let cx = &mut cx;

    macro_rules! check {
        ($f: ident, $($e: expr),+; $expect: expr) => ({
            $(assert_eq!($f(&$e), $expect);)+
        });
    }

    let abc = quote_expr!(cx, 23);
    check!(expr_to_string, abc, *quote_expr!(cx, $abc); "23");

    let ty = quote_ty!(cx, isize);
    check!(ty_to_string, ty, *quote_ty!(cx, $ty); "isize");

    let item = quote_item!(cx, static x: $ty = 10;).unwrap();
    check!(item_to_string, item, quote_item!(cx, $item).unwrap(); "static x: isize = 10;");

    let twenty: u16 = 20;
    let stmt = quote_stmt!(cx, let x = $twenty;).unwrap();
    check!(stmt_to_string, stmt, *quote_stmt!(cx, $stmt).unwrap(); "let x = 20u16;");

    let pat = quote_pat!(cx, Some(_));
    check!(pat_to_string, pat, *quote_pat!(cx, $pat); "Some(_)");

    let expr = quote_expr!(cx, (x, y));
    let arm = quote_arm!(cx, (ref x, ref y) => $expr,);
    check!(arm_to_string, arm, quote_arm!(cx, $arm); " (ref x, ref y) => (x, y),");

    let attr = quote_attr!(cx, #![cfg(foo = "bar")]);
    check!(attribute_to_string, attr, quote_attr!(cx, $attr); r#"#![cfg(foo = "bar")]"#);
}
