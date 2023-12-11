use rustc_ast as ast;
use rustc_ast::tokenstream::TokenStream;
use rustc_expand::base::{self, DummyResult};
use rustc_parse::parser::token_lit_to_lit_kind_and_report_errs;
use rustc_span::symbol::Symbol;

use crate::errors;

pub fn expand_concat(
    cx: &mut base::ExtCtxt<'_>,
    sp: rustc_span::Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'static> {
    let Some(es) = base::get_exprs_from_tts(cx, tts) else {
        return DummyResult::any(sp);
    };
    let mut accumulator = String::new();
    let mut missing_literal = vec![];
    let mut has_errors = false;
    for e in es {
        match e.kind {
            ast::ExprKind::Lit(token_lit) => {
                match token_lit_to_lit_kind_and_report_errs(&cx.sess.parse_sess, token_lit, e.span)
                {
                    Ok(ast::LitKind::Str(s, _) | ast::LitKind::Float(s, _)) => {
                        accumulator.push_str(s.as_str());
                    }
                    Ok(ast::LitKind::Char(c)) => {
                        accumulator.push(c);
                    }
                    Ok(ast::LitKind::Int(i, _)) => {
                        accumulator.push_str(&i.to_string());
                    }
                    Ok(ast::LitKind::Bool(b)) => {
                        accumulator.push_str(&b.to_string());
                    }
                    Ok(ast::LitKind::CStr(..)) => {
                        cx.emit_err(errors::ConcatCStrLit { span: e.span });
                        has_errors = true;
                    }
                    Ok(ast::LitKind::Byte(..) | ast::LitKind::ByteStr(..)) => {
                        cx.emit_err(errors::ConcatBytestr { span: e.span });
                        has_errors = true;
                    }
                    Ok(ast::LitKind::Err) | Err(()) => {
                        has_errors = true;
                    }
                }
            }
            // We also want to allow negative numeric literals.
            ast::ExprKind::Unary(ast::UnOp::Neg, ref expr)
                if let ast::ExprKind::Lit(token_lit) = expr.kind =>
            {
                match token_lit_to_lit_kind_and_report_errs(&cx.sess.parse_sess, token_lit, e.span)
                {
                    Ok(ast::LitKind::Int(i, _)) => accumulator.push_str(&format!("-{i}")),
                    Ok(ast::LitKind::Float(f, _)) => accumulator.push_str(&format!("-{f}")),
                    Err(()) => {
                        has_errors = true;
                    }
                    _ => missing_literal.push(e.span),
                }
            }
            ast::ExprKind::IncludedBytes(..) => {
                cx.emit_err(errors::ConcatBytestr { span: e.span });
            }
            ast::ExprKind::Err => {
                has_errors = true;
            }
            _ => {
                missing_literal.push(e.span);
            }
        }
    }

    if !missing_literal.is_empty() {
        cx.emit_err(errors::ConcatMissingLiteral { spans: missing_literal });
        return DummyResult::any(sp);
    } else if has_errors {
        return DummyResult::any(sp);
    }
    let sp = cx.with_def_site_ctxt(sp);
    base::MacEager::expr(cx.expr_str(sp, Symbol::intern(&accumulator)))
}
