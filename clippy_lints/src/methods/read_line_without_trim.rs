use std::ops::ControlFlow;

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::visitors::for_each_local_use_after_expr;
use clippy_utils::{get_parent_expr, sym};
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{BinOpKind, Expr, ExprKind, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

use super::READ_LINE_WITHOUT_TRIM;

fn expr_is_string_literal_without_trailing_newline(expr: &Expr<'_>) -> bool {
    if let ExprKind::Lit(lit) = expr.kind
        && let LitKind::Str(sym, _) = lit.node
    {
        !sym.as_str().ends_with('\n')
    } else {
        false
    }
}

/// Will a `.parse::<ty>()` call fail if the input has a trailing newline?
fn parse_fails_on_trailing_newline(ty: Ty<'_>) -> bool {
    // only allow a very limited set of types for now, for which we 100% know parsing will fail
    matches!(ty.kind(), ty::Float(_) | ty::Bool | ty::Int(_) | ty::Uint(_))
}

pub fn check(cx: &LateContext<'_>, call: &Expr<'_>, recv: &Expr<'_>, arg: &Expr<'_>) {
    let recv_ty = cx.typeck_results().expr_ty(recv);
    if is_type_diagnostic_item(cx, recv_ty, sym::Stdin)
        && let ExprKind::Path(QPath::Resolved(_, path)) = arg.peel_borrows().kind
        && let Res::Local(local_id) = path.res
    {
        // We've checked that `call` is a call to `Stdin::read_line()` with the right receiver,
        // now let's check if the first use of the string passed to `::read_line()`
        // is used for operations that will always fail (e.g. parsing "6\n" into a number)
        let _ = for_each_local_use_after_expr(cx, local_id, call.hir_id, |expr| {
            if let Some(parent) = get_parent_expr(cx, expr) {
                let data = if let ExprKind::MethodCall(segment, recv, args, span) = parent.kind {
                    if args.is_empty()
                        && segment.ident.name == sym::parse
                        && let parse_result_ty = cx.typeck_results().expr_ty(parent)
                        && is_type_diagnostic_item(cx, parse_result_ty, sym::Result)
                        && let ty::Adt(_, substs) = parse_result_ty.kind()
                        && let Some(ok_ty) = substs[0].as_type()
                        && parse_fails_on_trailing_newline(ok_ty)
                    {
                        // Called `s.parse::<T>()` where `T` is a type we know for certain will fail
                        // if the input has a trailing newline
                        Some((
                            span,
                            "calling `.parse()` on a string without trimming the trailing newline character",
                            "checking",
                        ))
                    } else if segment.ident.name == sym::ends_with
                        && recv.span == expr.span
                        && let [arg] = args
                        && expr_is_string_literal_without_trailing_newline(arg)
                    {
                        // Called `s.ends_with(<some string literal>)` where the argument is a string literal that does
                        // not end with a newline, thus always evaluating to false
                        Some((
                            parent.span,
                            "checking the end of a string without trimming the trailing newline character",
                            "parsing",
                        ))
                    } else {
                        None
                    }
                } else if let ExprKind::Binary(binop, left, right) = parent.kind
                    && let BinOpKind::Eq = binop.node
                    && (expr_is_string_literal_without_trailing_newline(left)
                        || expr_is_string_literal_without_trailing_newline(right))
                {
                    // `s == <some string literal>` where the string literal does not end with a newline
                    Some((
                        parent.span,
                        "comparing a string literal without trimming the trailing newline character",
                        "comparison",
                    ))
                } else {
                    None
                };

                if let Some((primary_span, lint_message, operation)) = data {
                    span_lint_and_then(cx, READ_LINE_WITHOUT_TRIM, primary_span, lint_message, |diag| {
                        let local_snippet = snippet(cx, expr.span, "<expr>");

                        diag.span_note(
                            call.span,
                            format!(
                                "call to `.read_line()` here, \
                                which leaves a trailing newline character in the buffer, \
                                which in turn will cause the {operation} to always fail"
                            ),
                        );

                        diag.span_suggestion(
                            expr.span,
                            "try",
                            format!("{local_snippet}.trim_end()"),
                            Applicability::MachineApplicable,
                        );
                    });
                }
            }

            // only consider the first use to prevent this scenario:
            // ```
            // let mut s = String::new();
            // std::io::stdin().read_line(&mut s);
            // s.pop();
            // let _x: i32 = s.parse().unwrap();
            // ```
            // this is actually fine, because the pop call removes the trailing newline.
            ControlFlow::<(), ()>::Break(())
        });
    }
}
