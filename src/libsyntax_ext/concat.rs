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
    let mut int_pos = vec![];
    let mut bool_pos = vec![];
    let mut b_accumulator: Vec<u8> = vec![];
    let mut b_pos: Vec<Span> = vec![];
    // We don't support mixing things with byte str literals, but do a best effort to fill in a
    // reasonable byte str output to avoid further errors down the line.
    let mut unified_accumulator: Vec<u8> = vec![];
    let mut missing_literal = vec![];
    for e in es {
        match e.node {
            ast::ExprKind::Lit(ref lit) => match lit.node {
                ast::LitKind::Str(ref s, _)
                | ast::LitKind::Float(ref s, _)
                | ast::LitKind::FloatUnsuffixed(ref s) => {
                    string_accumulator.push_str(&s.as_str());
                    string_pos.push(e.span);
                    // If we ever allow `concat!("", b"")`, we should probably add a warn by default
                    // lint to this code.
                    unified_accumulator.extend(s.to_string().into_bytes());
                }
                ast::LitKind::Char(c) => {
                    string_accumulator.push(c);
                    string_pos.push(e.span);
                    unified_accumulator.extend(c.to_string().into_bytes());
                }
                ast::LitKind::Int(i, ast::LitIntType::Unsigned(_))
                | ast::LitKind::Int(i, ast::LitIntType::Signed(_))
                | ast::LitKind::Int(i, ast::LitIntType::Unsuffixed) => {
                    string_accumulator.push_str(&i.to_string());
                    int_pos.push(e.span);
                    // If we ever allow `concat!()` mixing byte literals with integers, we need to
                    // define the appropriate behavior for it. Consistently considering them as
                    // "machine width" would be bug-prone. Taking the smallest possible size for the
                    // literal is probably what people _that don't think about it_ would expect, but
                    // would be inconsistent. Another option is only to accept the literals if they
                    // would fit in a `u8`.
                    unified_accumulator.extend(i.to_bytes().iter());
                }
                ast::LitKind::Bool(b) => {
                    string_accumulator.push_str(&b.to_string());
                    bool_pos.push(e.span);
                    // would `concat!(true, b"asdf")` ever make sense?
                }
                ast::LitKind::Byte(byte) => {
                    b_accumulator.push(byte);
                    b_pos.push(e.span);
                    unified_accumulator.push(byte);
                }
                ast::LitKind::ByteStr(ref b_str) => {
                    b_accumulator.extend(b_str.iter());
                    b_pos.push(e.span);
                    unified_accumulator.extend(b_str.iter());
                }
            }
            _ => {
                // Consider the possibility of allowing `concat!(b"asdf", [1, 2, 3, 4])`, given
                // that every single element of the array is a valid `u8`.
                missing_literal.push(e.span);
            }
        }
    }
    if missing_literal.len() > 0 {
        let mut err = cx.struct_span_err(missing_literal, "expected a literal");
        err.note("only literals (like `\"foo\"`, `42` and `3.14`) can be passed to `concat!()`");
        err.emit();
    }
    let sp = sp.apply_mark(cx.current_expansion.mark);
    // Do not allow mixing "" and b"", but return the joint b"" to avoid further errors
    if b_pos.len() > 0 && (string_pos.len() > 0 || int_pos.len() > 0 || bool_pos.len() > 0) {
        let mut mixings = vec![];
        if string_pos.len() > 0 {
            mixings.push("string");
        }
        if int_pos.len() > 0 {
            mixings.push("numeric");
        }
        if bool_pos.len() > 0 {
            mixings.push("boolean");
        }
        let mut err = cx.struct_span_err(
            b_pos.clone(),
            "cannot concatenate a byte string literal with other literals",
        );
        if mixings.len() > 0 && (int_pos.len() > 0 || bool_pos.len() > 0) {
            let msg = if mixings.len() >= 2 {
                format!(
                    "{} or {}",
                    mixings[0..mixings.len() - 1].join(", "),
                    mixings.last().unwrap(),
                )
            } else {
                mixings[0].to_string()
            };
            err.note(&format!("we don't support mixing {} literals and byte strings", msg));
        }
        if string_pos.len() > 0 && int_pos.len() == 0 && bool_pos.len() == 0 {
            err.multipart_suggestion(
                "we don't support mixing string and byte string literals, use only byte strings",
                string_pos
                    .iter()
                    .map(|pos| (pos.shrink_to_lo(), "b".to_string()))
                    .collect(),
            );
        }
        for pos in &b_pos {
            err.span_label(*pos, "byte string literal");
        }
        for pos in &string_pos {
            err.span_label(*pos, "string literal");
        }
        for pos in &int_pos {
            err.span_label(*pos, "numeric literal");
        }
        for pos in &bool_pos {
            err.span_label(*pos, "boolean literal");
        }
        err.emit();
        base::MacEager::expr(cx.expr_byte_str(sp, unified_accumulator))
    } else if b_accumulator.len() > 0 {
        base::MacEager::expr(cx.expr_byte_str(sp, b_accumulator))
    } else {
        base::MacEager::expr(cx.expr_str(sp, Symbol::intern(&string_accumulator)))
    }
}
