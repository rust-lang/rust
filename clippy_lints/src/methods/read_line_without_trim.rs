use std::ops::ControlFlow;

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::visitors::for_each_local_use_after_expr;
use clippy_utils::{get_parent_expr, match_def_path};
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::sym;

use super::READ_LINE_WITHOUT_TRIM;

/// Will a `.parse::<ty>()` call fail if the input has a trailing newline?
fn parse_fails_on_trailing_newline(ty: Ty<'_>) -> bool {
    // only allow a very limited set of types for now, for which we 100% know parsing will fail
    matches!(ty.kind(), ty::Float(_) | ty::Bool | ty::Int(_) | ty::Uint(_))
}

pub fn check(cx: &LateContext<'_>, call: &Expr<'_>, recv: &Expr<'_>, arg: &Expr<'_>) {
    if let Some(recv_adt) = cx.typeck_results().expr_ty(recv).ty_adt_def()
        && match_def_path(cx, recv_adt.did(), &["std", "io", "stdio", "Stdin"])
        && let ExprKind::Path(QPath::Resolved(_, path)) = arg.peel_borrows().kind
        && let Res::Local(local_id) = path.res
    {
        // We've checked that `call` is a call to `Stdin::read_line()` with the right receiver,
        // now let's check if the first use of the string passed to `::read_line()` is
        // parsed into a type that will always fail if it has a trailing newline.
        for_each_local_use_after_expr(cx, local_id, call.hir_id, |expr| {
            if let Some(parent) = get_parent_expr(cx, expr)
                && let ExprKind::MethodCall(segment, .., span) = parent.kind
                && segment.ident.name == sym!(parse)
                && let parse_result_ty = cx.typeck_results().expr_ty(parent)
                && is_type_diagnostic_item(cx, parse_result_ty, sym::Result)
                && let ty::Adt(_, substs) = parse_result_ty.kind()
                && let Some(ok_ty) = substs[0].as_type()
                && parse_fails_on_trailing_newline(ok_ty)
            {
                let local_snippet = snippet(cx, expr.span, "<expr>");
                span_lint_and_then(
                    cx,
                    READ_LINE_WITHOUT_TRIM,
                    span,
                    "calling `.parse()` without trimming the trailing newline character",
                    |diag| {
                        diag.span_note(call.span, "call to `.read_line()` here, \
                            which leaves a trailing newline character in the buffer, \
                            which in turn will cause `.parse()` to fail");

                        diag.span_suggestion(
                            expr.span,
                            "try",
                            format!("{local_snippet}.trim_end()"),
                            Applicability::MachineApplicable,
                        );
                    }
                );
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
