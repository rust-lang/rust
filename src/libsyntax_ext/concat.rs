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
use syntax_pos;
use syntax::tokenstream;

use std::string::String;

pub fn expand_syntax_ext(cx: &mut base::ExtCtxt,
                         sp: syntax_pos::Span,
                         tts: &[tokenstream::TokenTree])
                         -> Box<base::MacResult + 'static> {
    let es = match base::get_exprs_from_tts(cx, sp, tts) {
        Some(e) => e,
        None => return base::DummyResult::expr(sp),
    };
    let mut accumulator = String::new();
    for e in es {
        match e.node {
            ast::ExprKind::Lit(ref lit) => {
                match lit.node {
                    ast::LitKind::Str(ref s, _) |
                    ast::LitKind::Float(ref s, _) |
                    ast::LitKind::FloatUnsuffixed(ref s) => {
                        accumulator.push_str(&s.as_str());
                    }
                    ast::LitKind::Char(c) => {
                        accumulator.push(c);
                    }
                    ast::LitKind::Int(i, ast::LitIntType::Unsigned(_)) |
                    ast::LitKind::Int(i, ast::LitIntType::Signed(_)) |
                    ast::LitKind::Int(i, ast::LitIntType::Unsuffixed) => {
                        accumulator.push_str(&format!("{}", i));
                    }
                    ast::LitKind::Bool(b) => {
                        accumulator.push_str(&format!("{}", b));
                    }
                    ast::LitKind::Byte(..) |
                    ast::LitKind::ByteStr(..) => {
                        cx.span_err(e.span, "cannot concatenate a byte string literal");
                    }
                }
            }
            _ => {
                cx.span_err(e.span, "expected a literal");
            }
        }
    }
    base::MacEager::expr(cx.expr_str(sp, Symbol::intern(&accumulator)))
}
