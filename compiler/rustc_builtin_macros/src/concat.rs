use rustc_ast as ast;
use rustc_ast::tokenstream::TokenStream;
use rustc_expand::base::{self, DummyResult};
use rustc_span::symbol::Symbol;

use std::string::String;

pub fn expand_concat(
    cx: &mut base::ExtCtxt<'_>,
    sp: rustc_span::Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'static> {
    let Some(es) = base::get_exprs_from_tts(cx, sp, tts) else {
        return DummyResult::any(sp);
    };
    let mut accumulator = String::new();
    let mut missing_literal = vec![];
    let mut has_errors = false;
    for e in es {
        match e.kind {
            ast::ExprKind::Lit(ref lit) => match lit.kind {
                ast::LitKind::Str(ref s, _) | ast::LitKind::Float(ref s, _) => {
                    accumulator.push_str(s.as_str());
                }
                ast::LitKind::Char(c) => {
                    accumulator.push(c);
                }
                ast::LitKind::Int(
                    i,
                    ast::LitIntType::Unsigned(_)
                    | ast::LitIntType::Signed(_)
                    | ast::LitIntType::Unsuffixed,
                ) => {
                    accumulator.push_str(&i.to_string());
                }
                ast::LitKind::Bool(b) => {
                    accumulator.push_str(&b.to_string());
                }
                ast::LitKind::Byte(..) | ast::LitKind::ByteStr(..) => {
                    cx.span_err(e.span, "cannot concatenate a byte string literal");
                }
                ast::LitKind::Err => {
                    has_errors = true;
                }
            },
            ast::ExprKind::Err => {
                has_errors = true;
            }
            _ => {
                missing_literal.push(e.span);
            }
        }
    }
    if !missing_literal.is_empty() {
        let mut err = cx.struct_span_err(missing_literal, "expected a literal");
        err.note("only literals (like `\"foo\"`, `42` and `3.14`) can be passed to `concat!()`");
        err.emit();
        return DummyResult::any(sp);
    } else if has_errors {
        return DummyResult::any(sp);
    }
    let sp = cx.with_def_site_ctxt(sp);
    base::MacEager::expr(cx.expr_str(sp, Symbol::intern(&accumulator)))
}
