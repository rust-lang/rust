// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
 * The compiler code necessary to support the `push_unsafe!` and
 * `pop_unsafe!` macros.
 *
 * This is a hack to allow a kind of "safety hygiene", where a macro
 * can generate code with an interior expression that inherits the
 * safety of some outer context.
 *
 * For example, in:
 *
 * ```rust
 * fn foo() { push_unsafe!( { EXPR_1; pop_unsafe!( EXPR_2 ) } ) }
 * ```
 *
 * the `EXPR_1` is considered to be in an `unsafe` context,
 * but `EXPR_2` is considered to be in a "safe" (i.e. checked) context.
 *
 * For comparison, in:
 *
 * ```rust
 * fn foo() { unsafe { push_unsafe!( { EXPR_1; pop_unsafe!( EXPR_2 ) } ) } }
 * ```
 *
 * both `EXPR_1` and `EXPR_2` are considered to be in `unsafe`
 * contexts.
 *
 */

use ast;
use codemap::Span;
use ext::base::*;
use ext::base;
use ext::build::AstBuilder;
use feature_gate;
use ptr::P;

enum PushPop { Push, Pop }

pub fn expand_push_unsafe<'cx>(cx: &'cx mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
                               -> Box<base::MacResult+'cx> {
    expand_pushpop_unsafe(cx, sp, tts, PushPop::Push)
}

pub fn expand_pop_unsafe<'cx>(cx: &'cx mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
                               -> Box<base::MacResult+'cx> {
    expand_pushpop_unsafe(cx, sp, tts, PushPop::Pop)
}

fn expand_pushpop_unsafe<'cx>(cx: &'cx mut ExtCtxt, sp: Span, tts: &[ast::TokenTree],
                                  pp: PushPop) -> Box<base::MacResult+'cx> {
    feature_gate::check_for_pushpop_syntax(
        cx.ecfg.features, &cx.parse_sess.span_diagnostic, sp);

    let mut exprs = match get_exprs_from_tts(cx, sp, tts) {
        Some(exprs) => exprs.into_iter(),
        None => return DummyResult::expr(sp),
    };

    let expr = match (exprs.next(), exprs.next()) {
        (Some(expr), None) => expr,
        _ => {
            let msg = match pp {
                PushPop::Push => "push_unsafe! takes 1 arguments",
                PushPop::Pop => "pop_unsafe! takes 1 arguments",
            };
            cx.span_err(sp, msg);
            return DummyResult::expr(sp);
        }
    };

    let source = ast::UnsafeSource::CompilerGenerated;
    let check_mode = match pp {
        PushPop::Push => ast::BlockCheckMode::PushUnsafeBlock(source),
        PushPop::Pop => ast::BlockCheckMode::PopUnsafeBlock(source),
    };

    MacEager::expr(cx.expr_block(P(ast::Block {
        stmts: vec![],
        expr: Some(expr),
        id: ast::DUMMY_NODE_ID,
        rules: check_mode,
        span: sp
    })))
}
