// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
 * The compiler code necessary to support the env! extension.  Eventually this
 * should all get sucked into either the compiler syntax extension plugin
 * interface.
 */

use syntax::ast;
use syntax::codemap::Span;
use syntax::ext::base::*;
use syntax::ext::base;
use syntax::ext::build::AstBuilder;
use syntax::parse::token;

use std::env;

pub fn expand_option_env<'cx>(cx: &'cx mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
                              -> Box<base::MacResult+'cx> {
    let var = match get_single_str_from_tts(cx, sp, tts, "option_env!") {
        None => return DummyResult::expr(sp),
        Some(v) => v
    };

    let e = match env::var(&var[..]) {
      Err(..) => {
          cx.expr_path(cx.path_all(sp,
                                   true,
                                   cx.std_path(&["option", "Option", "None"]),
                                   Vec::new(),
                                   vec!(cx.ty_rptr(sp,
                                                   cx.ty_ident(sp,
                                                        cx.ident_of("str")),
                                                   Some(cx.lifetime(sp,
                                                        cx.ident_of(
                                                            "'static").name)),
                                                   ast::MutImmutable)),
                                   Vec::new()))
      }
      Ok(s) => {
          cx.expr_call_global(sp,
                              cx.std_path(&["option", "Option", "Some"]),
                              vec!(cx.expr_str(sp,
                                               token::intern_and_get_ident(
                                          &s[..]))))
      }
    };
    MacEager::expr(e)
}

pub fn expand_env<'cx>(cx: &'cx mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
                       -> Box<base::MacResult+'cx> {
    let mut exprs = match get_exprs_from_tts(cx, sp, tts) {
        Some(ref exprs) if exprs.is_empty() => {
            cx.span_err(sp, "env! takes 1 or 2 arguments");
            return DummyResult::expr(sp);
        }
        None => return DummyResult::expr(sp),
        Some(exprs) => exprs.into_iter()
    };

    let var = match expr_to_string(cx,
                                exprs.next().unwrap(),
                                "expected string literal") {
        None => return DummyResult::expr(sp),
        Some((v, _style)) => v
    };
    let msg = match exprs.next() {
        None => {
            token::intern_and_get_ident(&format!("environment variable `{}` \
                                                 not defined",
                                                var))
        }
        Some(second) => {
            match expr_to_string(cx, second, "expected string literal") {
                None => return DummyResult::expr(sp),
                Some((s, _style)) => s
            }
        }
    };

    match exprs.next() {
        None => {}
        Some(_) => {
            cx.span_err(sp, "env! takes 1 or 2 arguments");
            return DummyResult::expr(sp);
        }
    }

    let e = match env::var(&var[..]) {
        Err(_) => {
            cx.span_err(sp, &msg);
            cx.expr_usize(sp, 0)
        }
        Ok(s) => cx.expr_str(sp, token::intern_and_get_ident(&s))
    };
    MacEager::expr(e)
}
