// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;
use syntax::ext::base;
use syntax::ext::build::AstBuilder;
use syntax::symbol::Symbol;
use syntax::tokenstream;
use syntax_pos::Span;

use std::string::String;

pub fn expand_syntax_ext(
    cx: &mut base::ExtCtxt,
    sp: Span,
    tts: &[tokenstream::TokenTree],
) -> Box<dyn base::MacResult + 'static> {
    let es = match base::get_exprs_from_tts(cx, sp, tts) {
        Some(e) => e,
        None => return base::DummyResult::expr(sp),
    };
    let mut string_accumulator = String::new();
    let mut string_pos = vec![];
    let mut b_accumulator: Vec<u8> = vec![];
    let mut b_pos: Vec<Span> = vec![];
    let mut missing_literal = vec![];
    for e in es {
        match e.node {
            ast::ExprKind::Lit(ref lit) => match lit.node {
                ast::LitKind::Str(ref s, _)
                | ast::LitKind::Float(ref s, _)
                | ast::LitKind::FloatUnsuffixed(ref s) => {
                    string_accumulator.push_str(&s.as_str());
                    string_pos.push(e.span);
                }
                ast::LitKind::Char(c) => {
                    string_accumulator.push(c);
                    string_pos.push(e.span);
                }
                ast::LitKind::Int(i, ast::LitIntType::Unsigned(_))
                | ast::LitKind::Int(i, ast::LitIntType::Signed(_))
                | ast::LitKind::Int(i, ast::LitIntType::Unsuffixed) => {
                    string_accumulator.push_str(&i.to_string());
                    string_pos.push(e.span);
                }
                ast::LitKind::Bool(b) => {
                    string_accumulator.push_str(&b.to_string());
                    string_pos.push(e.span);
                }
                ast::LitKind::Byte(byte) => {
                    b_accumulator.push(byte);
                    b_pos.push(e.span);
                }
                ast::LitKind::ByteStr(ref b_str) => {
                    b_accumulator.extend(b_str.iter());
                    b_pos.push(e.span);
                }
            },
            _ => {
                missing_literal.push(e.span);
            }
        }
    }
    if missing_literal.len() > 0 {
        let mut err = cx.struct_span_err(missing_literal, "expected a literal");
        err.note("only literals (like `\"foo\"`, `42` and `3.14`) can be passed to `concat!()`");
        err.emit();
    }
    // Do not allow mixing "" and b""
    if string_accumulator.len() > 0 && b_accumulator.len() > 0 {
        let mut err = cx.struct_span_err(
            b_pos.clone(),
            "cannot concatenate a byte string literal with string literals",
        );
        for pos in &b_pos {
            err.span_label(*pos, "byte string literal");
        }
        for pos in &string_pos {
            err.span_label(*pos, "string literal");

        }
        err.help("do not mix byte string literals and string literals");
        err.multipart_suggestion(
            "you can use byte string literals",
            string_pos
                .iter()
                .map(|pos| (pos.shrink_to_lo(), "b".to_string()))
                .collect(),
        );
        err.emit();
    }
    let sp = sp.apply_mark(cx.current_expansion.mark);
    if b_accumulator.len() > 0 {
        base::MacEager::expr(cx.expr_lit(
            sp,
            ast::LitKind::new_byte_str(b_accumulator),
        ))
    } else {
        base::MacEager::expr(cx.expr_str(sp, Symbol::intern(&string_accumulator)))
    }
}
