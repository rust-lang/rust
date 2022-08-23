use rustc_ast as ast;
use rustc_ast::{ptr::P, tokenstream::TokenStream};
use rustc_expand::base::{self, DummyResult};

use crate::errors::{
    BooleanLiteralsConcatenate, ByteLiteralExpected, CharacterLiteralsConcatenate,
    DoublyNestedArrayConcatenate, FloatLiteralsConcatenate, InvalidNumericLiteral,
    InvalidRepeatCount, NumericLiteralsConcatenate, OutOfBoundNumericLiteral, Snippet,
    StringLiteralsConcatenate,
};

/// Emits errors for literal expressions that are invalid inside and outside of an array.
fn invalid_type_err(cx: &mut base::ExtCtxt<'_>, expr: &P<rustc_ast::Expr>, is_nested: bool) {
    let ast::ExprKind::Lit(lit) = &expr.kind else {
        unreachable!();
    };
    match lit.kind {
        ast::LitKind::Char(_) => {
            let sub = cx
                .sess
                .source_map()
                .span_to_snippet(expr.span)
                .ok()
                .map(|s| Snippet::ByteCharacter { span: expr.span, snippet: s });
            cx.emit_err(CharacterLiteralsConcatenate { span: expr.span, sub });
        }
        ast::LitKind::Str(_, _) => {
            // suggestion would be invalid if we are nested
            let sub = if !is_nested {
                cx.sess
                    .source_map()
                    .span_to_snippet(expr.span)
                    .ok()
                    .map(|s| Snippet::ByteString { span: expr.span, snippet: s })
            } else {
                None
            };
            cx.emit_err(StringLiteralsConcatenate { span: expr.span, sub });
        }
        ast::LitKind::Float(_, _) => {
            cx.emit_err(FloatLiteralsConcatenate { span: expr.span });
        }
        ast::LitKind::Bool(_) => {
            cx.emit_err(BooleanLiteralsConcatenate { span: expr.span });
        }
        ast::LitKind::Err => {}
        ast::LitKind::Int(_, _) if !is_nested => {
            let sub = cx.sess.source_map().span_to_snippet(expr.span).ok().map(|s| {
                Snippet::WrappingNumberInArray { span: expr.span, snippet: format!("[{}]", s) }
            });
            cx.emit_err(NumericLiteralsConcatenate { span: expr.span, sub });
        }
        ast::LitKind::Int(
            val,
            ast::LitIntType::Unsuffixed | ast::LitIntType::Unsigned(ast::UintTy::U8),
        ) => {
            assert!(val > u8::MAX.into()); // must be an error
            cx.emit_err(OutOfBoundNumericLiteral { span: expr.span });
        }
        ast::LitKind::Int(_, _) => {
            cx.emit_err(InvalidNumericLiteral { span: expr.span });
        }
        _ => unreachable!(),
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
                cx.emit_err(DoublyNestedArrayConcatenate {
                    span: expr.span,
                    note: None,
                    help: None,
                });
            }
            *has_errors = true;
            None
        }
        ast::ExprKind::Lit(ref lit) => match lit.kind {
            ast::LitKind::Int(
                val,
                ast::LitIntType::Unsuffixed | ast::LitIntType::Unsigned(ast::UintTy::U8),
            ) if val <= u8::MAX.into() => Some(val as u8),

            ast::LitKind::Byte(val) => Some(val),
            ast::LitKind::ByteStr(_) => {
                if !*has_errors {
                    cx.emit_err(DoublyNestedArrayConcatenate {
                        span: expr.span,
                        note: Some(()),
                        help: Some(()),
                    });
                }
                *has_errors = true;
                None
            }
            _ => {
                if !*has_errors {
                    invalid_type_err(cx, expr, true);
                }
                *has_errors = true;
                None
            }
        },
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
    let Some(es) = base::get_exprs_from_tts(cx, sp, tts) else {
        return DummyResult::any(sp);
    };
    let mut accumulator = Vec::new();
    let mut missing_literals = vec![];
    let mut has_errors = false;
    for e in es {
        match e.kind {
            ast::ExprKind::Array(ref exprs) => {
                for expr in exprs {
                    if let Some(elem) =
                        handle_array_element(cx, &mut has_errors, &mut missing_literals, expr)
                    {
                        accumulator.push(elem);
                    }
                }
            }
            ast::ExprKind::Repeat(ref expr, ref count) => {
                if let ast::ExprKind::Lit(ast::Lit {
                    kind: ast::LitKind::Int(count_val, _), ..
                }) = count.value.kind
                {
                    if let Some(elem) =
                        handle_array_element(cx, &mut has_errors, &mut missing_literals, expr)
                    {
                        for _ in 0..count_val {
                            accumulator.push(elem);
                        }
                    }
                } else {
                    cx.emit_err(InvalidRepeatCount { span: count.value.span });
                }
            }
            ast::ExprKind::Lit(ref lit) => match lit.kind {
                ast::LitKind::Byte(val) => {
                    accumulator.push(val);
                }
                ast::LitKind::ByteStr(ref bytes) => {
                    accumulator.extend_from_slice(&bytes);
                }
                _ => {
                    if !has_errors {
                        invalid_type_err(cx, &e, false);
                    }
                    has_errors = true;
                }
            },
            ast::ExprKind::Err => {
                has_errors = true;
            }
            _ => {
                missing_literals.push(e.span);
            }
        }
    }
    if !missing_literals.is_empty() {
        cx.emit_err(ByteLiteralExpected { spans: missing_literals.clone() });
        return base::MacEager::expr(DummyResult::raw_expr(sp, true));
    } else if has_errors {
        return base::MacEager::expr(DummyResult::raw_expr(sp, true));
    }
    let sp = cx.with_def_site_ctxt(sp);
    base::MacEager::expr(cx.expr_byte_str(sp, accumulator))
}
