use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{ExprKind, LitKind, UnOp};
use rustc_expand::base::{DummyResult, ExpandResult, ExtCtxt, MacEager, MacroExpanderResult};
use rustc_session::errors::report_lit_error;
use rustc_span::Symbol;

use crate::errors;
use crate::util::get_exprs_from_tts;

pub(crate) fn expand_concat(
    cx: &mut ExtCtxt<'_>,
    sp: rustc_span::Span,
    tts: TokenStream,
) -> MacroExpanderResult<'static> {
    let ExpandResult::Ready(mac) = get_exprs_from_tts(cx, tts) else {
        return ExpandResult::Retry(());
    };
    let es = match mac {
        Ok(es) => es,
        Err(guar) => return ExpandResult::Ready(DummyResult::any(sp, guar)),
    };
    let mut accumulator = String::new();
    let mut missing_literal = vec![];
    let mut guar = None;
    for e in es {
        match e.kind {
            ExprKind::Lit(token_lit) => match LitKind::from_token_lit(token_lit) {
                Ok(LitKind::Str(s, _) | LitKind::Float(s, _)) => {
                    accumulator.push_str(s.as_str());
                }
                Ok(LitKind::Char(c)) => {
                    accumulator.push(c);
                }
                Ok(LitKind::Int(i, _)) => {
                    accumulator.push_str(&i.to_string());
                }
                Ok(LitKind::Bool(b)) => {
                    accumulator.push_str(&b.to_string());
                }
                Ok(LitKind::CStr(..)) => {
                    guar = Some(cx.dcx().emit_err(errors::ConcatCStrLit { span: e.span }));
                }
                Ok(LitKind::Byte(..) | LitKind::ByteStr(..)) => {
                    guar = Some(cx.dcx().emit_err(errors::ConcatBytestr { span: e.span }));
                }
                Ok(LitKind::Err(guarantee)) => {
                    guar = Some(guarantee);
                }
                Err(err) => {
                    guar = Some(report_lit_error(&cx.sess.psess, err, token_lit, e.span));
                }
            },
            // We also want to allow negative numeric literals.
            ExprKind::Unary(UnOp::Neg, ref expr) if let ExprKind::Lit(token_lit) = expr.kind => {
                match LitKind::from_token_lit(token_lit) {
                    Ok(LitKind::Int(i, _)) => accumulator.push_str(&format!("-{i}")),
                    Ok(LitKind::Float(f, _)) => accumulator.push_str(&format!("-{f}")),
                    Err(err) => {
                        guar = Some(report_lit_error(&cx.sess.psess, err, token_lit, e.span));
                    }
                    _ => missing_literal.push(e.span),
                }
            }
            ExprKind::IncludedBytes(..) => {
                cx.dcx().emit_err(errors::ConcatBytestr { span: e.span });
            }
            ExprKind::Err(guarantee) => {
                guar = Some(guarantee);
            }
            ExprKind::Dummy => cx.dcx().span_bug(e.span, "concatenating `ExprKind::Dummy`"),
            _ => {
                missing_literal.push(e.span);
            }
        }
    }

    ExpandResult::Ready(if !missing_literal.is_empty() {
        let guar = cx.dcx().emit_err(errors::ConcatMissingLiteral { spans: missing_literal });
        DummyResult::any(sp, guar)
    } else if let Some(guar) = guar {
        DummyResult::any(sp, guar)
    } else {
        let sp = cx.with_def_site_ctxt(sp);
        MacEager::expr(cx.expr_str(sp, Symbol::intern(&accumulator)))
    })
}
