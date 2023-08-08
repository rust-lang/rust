use rustc_ast as ast;
use rustc_ast::{ptr::P, tokenstream::TokenStream};
use rustc_expand::base::{self, DummyResult};
use rustc_session::errors::report_lit_error;
use rustc_span::Span;

use crate::errors;

/// Emits errors for literal expressions that are invalid inside and outside of an array.
fn invalid_type_err(
    cx: &mut base::ExtCtxt<'_>,
    token_lit: ast::token::Lit,
    span: Span,
    is_nested: bool,
) {
    use errors::{
        ConcatBytesInvalid, ConcatBytesInvalidSuggestion, ConcatBytesNonU8, ConcatBytesOob,
    };
    let snippet = cx.sess.source_map().span_to_snippet(span).ok();
    match ast::LitKind::from_token_lit(token_lit) {
        Ok(ast::LitKind::CStr(_, _)) => {
            // FIXME(c_str_literals): should concatenation of C string literals
            // include the null bytes in the end?
            cx.emit_err(errors::ConcatCStrLit { span: span });
        }
        Ok(ast::LitKind::Char(_)) => {
            let sugg =
                snippet.map(|snippet| ConcatBytesInvalidSuggestion::CharLit { span, snippet });
            cx.sess.emit_err(ConcatBytesInvalid { span, lit_kind: "character", sugg });
        }
        Ok(ast::LitKind::Str(_, _)) => {
            // suggestion would be invalid if we are nested
            let sugg = if !is_nested {
                snippet.map(|snippet| ConcatBytesInvalidSuggestion::StrLit { span, snippet })
            } else {
                None
            };
            cx.emit_err(ConcatBytesInvalid { span, lit_kind: "string", sugg });
        }
        Ok(ast::LitKind::Float(_, _)) => {
            cx.emit_err(ConcatBytesInvalid { span, lit_kind: "float", sugg: None });
        }
        Ok(ast::LitKind::Bool(_)) => {
            cx.emit_err(ConcatBytesInvalid { span, lit_kind: "boolean", sugg: None });
        }
        Ok(ast::LitKind::Err) => {}
        Ok(ast::LitKind::Int(_, _)) if !is_nested => {
            let sugg =
                snippet.map(|snippet| ConcatBytesInvalidSuggestion::IntLit { span: span, snippet });
            cx.emit_err(ConcatBytesInvalid { span, lit_kind: "numeric", sugg });
        }
        Ok(ast::LitKind::Int(
            val,
            ast::LitIntType::Unsuffixed | ast::LitIntType::Unsigned(ast::UintTy::U8),
        )) => {
            assert!(val > u8::MAX.into()); // must be an error
            cx.emit_err(ConcatBytesOob { span });
        }
        Ok(ast::LitKind::Int(_, _)) => {
            cx.emit_err(ConcatBytesNonU8 { span });
        }
        Ok(ast::LitKind::ByteStr(..) | ast::LitKind::Byte(_)) => unreachable!(),
        Err(err) => {
            report_lit_error(&cx.sess.parse_sess, err, token_lit, span);
        }
    }
}

fn handle_array_element(
    cx: &mut base::ExtCtxt<'_>,
    has_errors: &mut bool,
    missing_literals: &mut Vec<rustc_span::Span>,
    expr: &P<rustc_ast::Expr>,
) -> Option<u8> {
    match expr.kind {
        ast::ExprKind::Array(_) | ast::ExprKind::Repeat(_, _) => {
            if !*has_errors {
                cx.emit_err(errors::ConcatBytesArray { span: expr.span, bytestr: false });
            }
            *has_errors = true;
            None
        }
        ast::ExprKind::Lit(token_lit) => match ast::LitKind::from_token_lit(token_lit) {
            Ok(ast::LitKind::Int(
                val,
                ast::LitIntType::Unsuffixed | ast::LitIntType::Unsigned(ast::UintTy::U8),
            )) if val <= u8::MAX.into() => Some(val as u8),

            Ok(ast::LitKind::Byte(val)) => Some(val),
            Ok(ast::LitKind::ByteStr(..)) => {
                if !*has_errors {
                    cx.emit_err(errors::ConcatBytesArray { span: expr.span, bytestr: true });
                }
                *has_errors = true;
                None
            }
            _ => {
                if !*has_errors {
                    invalid_type_err(cx, token_lit, expr.span, true);
                }
                *has_errors = true;
                None
            }
        },
        ast::ExprKind::IncludedBytes(..) => {
            if !*has_errors {
                cx.emit_err(errors::ConcatBytesArray { span: expr.span, bytestr: false });
            }
            *has_errors = true;
            None
        }
        _ => {
            missing_literals.push(expr.span);
            None
        }
    }
}

pub fn expand_concat_bytes(
    cx: &mut base::ExtCtxt<'_>,
    sp: rustc_span::Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'static> {
    let Some(es) = base::get_exprs_from_tts(cx, tts) else {
        return DummyResult::any(sp);
    };
    let mut accumulator = Vec::new();
    let mut missing_literals = vec![];
    let mut has_errors = false;
    for e in es {
        match &e.kind {
            ast::ExprKind::Array(exprs) => {
                for expr in exprs {
                    if let Some(elem) =
                        handle_array_element(cx, &mut has_errors, &mut missing_literals, expr)
                    {
                        accumulator.push(elem);
                    }
                }
            }
            ast::ExprKind::Repeat(expr, count) => {
                if let ast::ExprKind::Lit(token_lit) = count.value.kind
                && let Ok(ast::LitKind::Int(count_val, _)) =
                    ast::LitKind::from_token_lit(token_lit)
                {
                    if let Some(elem) =
                        handle_array_element(cx, &mut has_errors, &mut missing_literals, expr)
                    {
                        for _ in 0..count_val {
                            accumulator.push(elem);
                        }
                    }
                } else {
                    cx.emit_err(errors::ConcatBytesBadRepeat {span: count.value.span });
                }
            }
            &ast::ExprKind::Lit(token_lit) => match ast::LitKind::from_token_lit(token_lit) {
                Ok(ast::LitKind::Byte(val)) => {
                    accumulator.push(val);
                }
                Ok(ast::LitKind::ByteStr(ref bytes, _)) => {
                    accumulator.extend_from_slice(&bytes);
                }
                _ => {
                    if !has_errors {
                        invalid_type_err(cx, token_lit, e.span, false);
                    }
                    has_errors = true;
                }
            },
            ast::ExprKind::IncludedBytes(bytes) => {
                accumulator.extend_from_slice(bytes);
            }
            ast::ExprKind::Err => {
                has_errors = true;
            }
            _ => {
                missing_literals.push(e.span);
            }
        }
    }
    if !missing_literals.is_empty() {
        cx.emit_err(errors::ConcatBytesMissingLiteral { spans: missing_literals });
        return base::MacEager::expr(DummyResult::raw_expr(sp, true));
    } else if has_errors {
        return base::MacEager::expr(DummyResult::raw_expr(sp, true));
    }
    let sp = cx.with_def_site_ctxt(sp);
    base::MacEager::expr(cx.expr_byte_str(sp, accumulator))
}
