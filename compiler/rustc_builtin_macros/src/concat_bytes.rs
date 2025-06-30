use rustc_ast::ptr::P;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{ExprKind, LitIntType, LitKind, StrStyle, UintTy, token};
use rustc_expand::base::{DummyResult, ExpandResult, ExtCtxt, MacEager, MacroExpanderResult};
use rustc_session::errors::report_lit_error;
use rustc_span::{ErrorGuaranteed, Span};

use crate::errors;
use crate::util::get_exprs_from_tts;

/// Emits errors for literal expressions that are invalid inside and outside of an array.
fn invalid_type_err(
    cx: &ExtCtxt<'_>,
    token_lit: token::Lit,
    span: Span,
    is_nested: bool,
) -> ErrorGuaranteed {
    use errors::{
        ConcatBytesInvalid, ConcatBytesInvalidSuggestion, ConcatBytesNonU8, ConcatBytesOob,
    };
    let snippet = cx.sess.source_map().span_to_snippet(span).ok();
    let dcx = cx.dcx();
    match LitKind::from_token_lit(token_lit) {
        Ok(LitKind::CStr(_, style)) => {
            // Avoid ambiguity in handling of terminal `NUL` by refusing to
            // concatenate C string literals as bytes.
            let sugg = if let Some(mut as_bstr) = snippet
                && style == StrStyle::Cooked
                && as_bstr.starts_with('c')
                && as_bstr.ends_with('"')
            {
                // Suggest`c"foo"` -> `b"foo\0"` if we can
                as_bstr.replace_range(0..1, "b");
                as_bstr.pop();
                as_bstr.push_str(r#"\0""#);
                Some(ConcatBytesInvalidSuggestion::CStrLit { span, as_bstr })
            } else {
                // No suggestion for a missing snippet, raw strings, or if for some reason we have
                // a span that doesn't match `c"foo"` (possible if a proc macro assigns a span
                // that doesn't actually point to a C string).
                None
            };
            // We can only provide a suggestion if we have a snip and it is not a raw string
            dcx.emit_err(ConcatBytesInvalid { span, lit_kind: "C string", sugg, cs_note: Some(()) })
        }
        Ok(LitKind::Char(_)) => {
            let sugg =
                snippet.map(|snippet| ConcatBytesInvalidSuggestion::CharLit { span, snippet });
            dcx.emit_err(ConcatBytesInvalid { span, lit_kind: "character", sugg, cs_note: None })
        }
        Ok(LitKind::Str(_, _)) => {
            // suggestion would be invalid if we are nested
            let sugg = if !is_nested {
                snippet.map(|snippet| ConcatBytesInvalidSuggestion::StrLit { span, snippet })
            } else {
                None
            };
            dcx.emit_err(ConcatBytesInvalid { span, lit_kind: "string", sugg, cs_note: None })
        }
        Ok(LitKind::Float(_, _)) => {
            dcx.emit_err(ConcatBytesInvalid { span, lit_kind: "float", sugg: None, cs_note: None })
        }
        Ok(LitKind::Bool(_)) => dcx.emit_err(ConcatBytesInvalid {
            span,
            lit_kind: "boolean",
            sugg: None,
            cs_note: None,
        }),
        Ok(LitKind::Int(_, _)) if !is_nested => {
            let sugg =
                snippet.map(|snippet| ConcatBytesInvalidSuggestion::IntLit { span, snippet });
            dcx.emit_err(ConcatBytesInvalid { span, lit_kind: "numeric", sugg, cs_note: None })
        }
        Ok(LitKind::Int(val, LitIntType::Unsuffixed | LitIntType::Unsigned(UintTy::U8))) => {
            assert!(val.get() > u8::MAX.into()); // must be an error
            dcx.emit_err(ConcatBytesOob { span })
        }
        Ok(LitKind::Int(_, _)) => dcx.emit_err(ConcatBytesNonU8 { span }),
        Ok(LitKind::ByteStr(..) | LitKind::Byte(_)) => unreachable!(),
        Ok(LitKind::Err(guar)) => guar,
        Err(err) => report_lit_error(&cx.sess.psess, err, token_lit, span),
    }
}

/// Returns `expr` as a *single* byte literal if applicable.
///
/// Otherwise, returns `None`, and either pushes the `expr`'s span to `missing_literals` or
/// updates `guar` accordingly.
fn handle_array_element(
    cx: &ExtCtxt<'_>,
    guar: &mut Option<ErrorGuaranteed>,
    missing_literals: &mut Vec<rustc_span::Span>,
    expr: &P<rustc_ast::Expr>,
) -> Option<u8> {
    let dcx = cx.dcx();

    match expr.kind {
        ExprKind::Lit(token_lit) => {
            match LitKind::from_token_lit(token_lit) {
                Ok(LitKind::Int(
                    val,
                    LitIntType::Unsuffixed | LitIntType::Unsigned(UintTy::U8),
                )) if let Ok(val) = u8::try_from(val.get()) => {
                    return Some(val);
                }
                Ok(LitKind::Byte(val)) => return Some(val),
                Ok(LitKind::ByteStr(..)) => {
                    guar.get_or_insert_with(|| {
                        dcx.emit_err(errors::ConcatBytesArray { span: expr.span, bytestr: true })
                    });
                }
                _ => {
                    guar.get_or_insert_with(|| invalid_type_err(cx, token_lit, expr.span, true));
                }
            };
        }
        ExprKind::Array(_) | ExprKind::Repeat(_, _) => {
            guar.get_or_insert_with(|| {
                dcx.emit_err(errors::ConcatBytesArray { span: expr.span, bytestr: false })
            });
        }
        ExprKind::IncludedBytes(..) => {
            guar.get_or_insert_with(|| {
                dcx.emit_err(errors::ConcatBytesArray { span: expr.span, bytestr: false })
            });
        }
        _ => missing_literals.push(expr.span),
    }

    None
}

pub(crate) fn expand_concat_bytes(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'static> {
    let ExpandResult::Ready(mac) = get_exprs_from_tts(cx, tts) else {
        return ExpandResult::Retry(());
    };
    let es = match mac {
        Ok(es) => es,
        Err(guar) => return ExpandResult::Ready(DummyResult::any(sp, guar)),
    };
    let mut accumulator = Vec::new();
    let mut missing_literals = vec![];
    let mut guar = None;
    for e in es {
        match &e.kind {
            ExprKind::Array(exprs) => {
                for expr in exprs {
                    if let Some(elem) =
                        handle_array_element(cx, &mut guar, &mut missing_literals, expr)
                    {
                        accumulator.push(elem);
                    }
                }
            }
            ExprKind::Repeat(expr, count) => {
                if let ExprKind::Lit(token_lit) = count.value.kind
                    && let Ok(LitKind::Int(count_val, _)) = LitKind::from_token_lit(token_lit)
                {
                    if let Some(elem) =
                        handle_array_element(cx, &mut guar, &mut missing_literals, expr)
                    {
                        for _ in 0..count_val.get() {
                            accumulator.push(elem);
                        }
                    }
                } else {
                    guar = Some(
                        cx.dcx().emit_err(errors::ConcatBytesBadRepeat { span: count.value.span }),
                    );
                }
            }
            &ExprKind::Lit(token_lit) => match LitKind::from_token_lit(token_lit) {
                Ok(LitKind::Byte(val)) => {
                    accumulator.push(val);
                }
                Ok(LitKind::ByteStr(ref byte_sym, _)) => {
                    accumulator.extend_from_slice(byte_sym.as_byte_str());
                }
                _ => {
                    guar.get_or_insert_with(|| invalid_type_err(cx, token_lit, e.span, false));
                }
            },
            ExprKind::IncludedBytes(byte_sym) => {
                accumulator.extend_from_slice(byte_sym.as_byte_str());
            }
            ExprKind::Err(guarantee) => {
                guar = Some(*guarantee);
            }
            ExprKind::Dummy => cx.dcx().span_bug(e.span, "concatenating `ExprKind::Dummy`"),
            _ => {
                missing_literals.push(e.span);
            }
        }
    }
    ExpandResult::Ready(if !missing_literals.is_empty() {
        let guar = cx.dcx().emit_err(errors::ConcatBytesMissingLiteral { spans: missing_literals });
        MacEager::expr(DummyResult::raw_expr(sp, Some(guar)))
    } else if let Some(guar) = guar {
        MacEager::expr(DummyResult::raw_expr(sp, Some(guar)))
    } else {
        let sp = cx.with_def_site_ctxt(sp);
        MacEager::expr(cx.expr_byte_str(sp, accumulator))
    })
}
