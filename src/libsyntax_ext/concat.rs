use syntax::ast;
use syntax::ext::base;
use syntax::ext::build::AstBuilder;
use syntax::symbol::Symbol;
use syntax::tokenstream;

use std::string::String;

pub fn expand_syntax_ext(
    cx: &mut base::ExtCtxt<'_>,
    sp: syntax_pos::Span,
    tts: &[tokenstream::TokenTree],
) -> Box<dyn base::MacResult + 'static> {
    let es = match base::get_exprs_from_tts(cx, sp, tts) {
        Some(e) => e,
        None => return base::DummyResult::expr(sp),
    };
    let mut accumulator = String::new();
    let mut missing_literal = vec![];
    let mut has_errors = false;
    for e in es {
        match e.node {
            ast::ExprKind::Lit(ref lit) => match lit.node {
                ast::LitKind::Str(ref s, _)
                | ast::LitKind::Float(ref s, _)
                | ast::LitKind::FloatUnsuffixed(ref s) => {
                    accumulator.push_str(&s.as_str());
                }
                ast::LitKind::Char(c) => {
                    accumulator.push(c);
                }
                ast::LitKind::Int(i, ast::LitIntType::Unsigned(_))
                | ast::LitKind::Int(i, ast::LitIntType::Signed(_))
                | ast::LitKind::Int(i, ast::LitIntType::Unsuffixed) => {
                    accumulator.push_str(&i.to_string());
                }
                ast::LitKind::Bool(b) => {
                    accumulator.push_str(&b.to_string());
                }
                ast::LitKind::Byte(..) | ast::LitKind::ByteStr(..) => {
                    cx.span_err(e.span, "cannot concatenate a byte string literal");
                }
                ast::LitKind::Err(_) => {
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
    if missing_literal.len() > 0 {
        let mut err = cx.struct_span_err(missing_literal, "expected a literal");
        err.note("only literals (like `\"foo\"`, `42` and `3.14`) can be passed to `concat!()`");
        err.emit();
        return base::DummyResult::expr(sp);
    } else if has_errors {
        return base::DummyResult::expr(sp);
    }
    let sp = sp.apply_mark(cx.current_expansion.mark);
    base::MacEager::expr(cx.expr_str(sp, Symbol::intern(&accumulator)))
}
