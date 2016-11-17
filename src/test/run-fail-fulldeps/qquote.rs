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

// error-pattern:expected expression, found statement (`let`)

#![feature(quote, rustc_private)]

extern crate syntax;
extern crate syntax_pos;

use syntax::ast;
use syntax::codemap;
use syntax::print::pprust;
use syntax::symbol::Symbol;
use syntax_pos::DUMMY_SP;

fn main() {
    let ps = syntax::parse::ParseSess::new();
    let mut resolver = syntax::ext::base::DummyResolver;
    let mut cx = syntax::ext::base::ExtCtxt::new(
        &ps,
        syntax::ext::expand::ExpansionConfig::default("qquote".to_string()),
        &mut resolver);
    cx.bt_push(syntax::codemap::ExpnInfo {
        call_site: DUMMY_SP,
        callee: syntax::codemap::NameAndSpan {
            format: syntax::codemap::MacroBang(Symbol::intern("")),
            allow_internal_unstable: false,
            span: None,
        }
    });
    let cx = &mut cx;

    println!("{}", pprust::expr_to_string(&*quote_expr!(&cx, 23)));
    assert_eq!(pprust::expr_to_string(&*quote_expr!(&cx, 23)), "23");

    let expr = quote_expr!(&cx, let x isize = 20;);
    println!("{}", pprust::expr_to_string(&*expr));
    assert_eq!(pprust::expr_to_string(&*expr), "let x isize = 20;");
}
