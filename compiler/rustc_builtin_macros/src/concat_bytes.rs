use rustc_ast as ast;
use rustc_ast::{ptr::P, tokenstream::TokenStream};
use rustc_data_structures::sync::Lrc;
use rustc_errors::Applicability;
use rustc_expand::base::{self, DummyResult};

/// Emits errors for literal expressions that are invalid inside and outside of an array.
fn invalid_type_err(cx: &mut base::ExtCtxt<'_>, expr: &P<rustc_ast::Expr>, is_nested: bool) {
    let lit = if let ast::ExprKind::Lit(lit) = &expr.kind {
        lit
    } else {
        unreachable!();
    };
    match lit.kind {
        ast::LitKind::Char(_) => {
            let mut err = cx.struct_span_err(expr.span, "cannot concatenate character literals");
            if let Ok(snippet) = cx.sess.source_map().span_to_snippet(expr.span) {
                err.span_suggestion(
                    expr.span,
                    "try using a byte character",
                    format!("b{}", snippet),
                    Applicability::MachineApplicable,
                )
                .emit();
            }
        }
        ast::LitKind::Str(_, _) => {
            let mut err = cx.struct_span_err(expr.span, "cannot concatenate string literals");
            // suggestion would be invalid if we are nested
            if !is_nested {
                if let Ok(snippet) = cx.sess.source_map().span_to_snippet(expr.span) {
                    err.span_suggestion(
                        expr.span,
                        "try using a byte string",
                        format!("b{}", snippet),
                        Applicability::MachineApplicable,
                    );
                }
            }
            err.emit();
        }
        ast::LitKind::Float(_, _) => {
            cx.span_err(expr.span, "cannot concatenate float literals");
        }
        ast::LitKind::Bool(_) => {
            cx.span_err(expr.span, "cannot concatenate boolean literals");
        }
        ast::LitKind::Err(_) => {}
        ast::LitKind::Int(_, _) if !is_nested => {
            let mut err = cx.struct_span_err(expr.span, "cannot concatenate numeric literals");
            if let Ok(snippet) = cx.sess.source_map().span_to_snippet(expr.span) {
                err.span_suggestion(
                    expr.span,
                    "try wrapping the number in an array",
                    format!("[{}]", snippet),
                    Applicability::MachineApplicable,
                );
            }
            err.emit();
        }
        ast::LitKind::Int(
            val,
            ast::LitIntType::Unsuffixed | ast::LitIntType::Unsigned(ast::UintTy::U8),
        ) => {
            assert!(val > u8::MAX.into()); // must be an error
            cx.span_err(expr.span, "numeric literal is out of bounds");
        }
        ast::LitKind::Int(_, _) => {
            cx.span_err(expr.span, "numeric literal is not a `u8`");
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
                cx.span_err(expr.span, "cannot concatenate doubly nested array");
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
                    cx.struct_span_err(expr.span, "cannot concatenate doubly nested array")
                        .note("byte strings are treated as arrays of bytes")
                        .help("try flattening the array")
                        .emit();
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
    let es = match base::get_exprs_from_tts(cx, sp, tts) {
        Some(e) => e,
        None => return DummyResult::any(sp),
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
                    cx.span_err(count.value.span, "repeat count is not a positive number");
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
        let mut err = cx.struct_span_err(missing_literals.clone(), "expected a byte literal");
        err.note("only byte literals (like `b\"foo\"`, `b's'`, and `[3, 4, 5]`) can be passed to `concat_bytes!()`");
        err.emit();
        return base::MacEager::expr(DummyResult::raw_expr(sp, true));
    } else if has_errors {
        return base::MacEager::expr(DummyResult::raw_expr(sp, true));
    }
    let sp = cx.with_def_site_ctxt(sp);
    base::MacEager::expr(cx.expr_lit(sp, ast::LitKind::ByteStr(Lrc::from(accumulator))))
}
