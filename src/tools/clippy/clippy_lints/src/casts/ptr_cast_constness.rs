use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::sugg::Sugg;
use clippy_utils::{std_or_core, sym};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Mutability, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty, TypeVisitableExt};

use super::PTR_CAST_CONSTNESS;

pub(super) fn check<'tcx>(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    cast_expr: &Expr<'_>,
    cast_from: Ty<'tcx>,
    cast_to: Ty<'tcx>,
    msrv: Msrv,
) {
    if let ty::RawPtr(from_ty, from_mutbl) = cast_from.kind()
        && let ty::RawPtr(to_ty, to_mutbl) = cast_to.kind()
        && matches!(
            (from_mutbl, to_mutbl),
            (Mutability::Not, Mutability::Mut) | (Mutability::Mut, Mutability::Not)
        )
        && from_ty == to_ty
        && !from_ty.has_erased_regions()
    {
        if let ExprKind::Call(func, []) = cast_expr.kind
            && let ExprKind::Path(QPath::Resolved(None, path)) = func.kind
            && let Some(defid) = path.res.opt_def_id()
            && let Some(prefix) = std_or_core(cx)
            && let mut app = Applicability::MachineApplicable
            && let sugg = format!("{}", Sugg::hir_with_applicability(cx, cast_expr, "_", &mut app))
            && let Some((_, after_lt)) = sugg.split_once("::<")
            && let Some((source, target, target_func)) = match cx.tcx.get_diagnostic_name(defid) {
                Some(sym::ptr_null) => Some(("const", "mutable", "null_mut")),
                Some(sym::ptr_null_mut) => Some(("mutable", "const", "null")),
                _ => None,
            }
        {
            span_lint_and_sugg(
                cx,
                PTR_CAST_CONSTNESS,
                expr.span,
                format!("`as` casting to make a {source} null pointer into a {target} null pointer"),
                format!("use `{target_func}()` directly instead"),
                format!("{prefix}::ptr::{target_func}::<{after_lt}"),
                app,
            );
            return;
        }

        if msrv.meets(cx, msrvs::POINTER_CAST_CONSTNESS) {
            let mut app = Applicability::MachineApplicable;
            let sugg = Sugg::hir_with_context(cx, cast_expr, expr.span.ctxt(), "_", &mut app);
            let constness = match *to_mutbl {
                Mutability::Not => "const",
                Mutability::Mut => "mut",
            };

            span_lint_and_sugg(
                cx,
                PTR_CAST_CONSTNESS,
                expr.span,
                "`as` casting between raw pointers while changing only its constness",
                format!("try `pointer::cast_{constness}`, a safer alternative"),
                format!("{}.cast_{constness}()", sugg.maybe_paren()),
                app,
            );
        }
    }
}

pub(super) fn check_null_ptr_cast_method(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if let ExprKind::MethodCall(method, cast_expr, [], _) = expr.kind
        && let ExprKind::Call(func, []) = cast_expr.kind
        && let ExprKind::Path(QPath::Resolved(None, path)) = func.kind
        && let Some(defid) = path.res.opt_def_id()
        && let method = match (cx.tcx.get_diagnostic_name(defid), method.ident.name) {
            (Some(sym::ptr_null), sym::cast_mut) => "null_mut",
            (Some(sym::ptr_null_mut), sym::cast_const) => "null",
            _ => return,
        }
        && let Some(prefix) = std_or_core(cx)
        && let mut app = Applicability::MachineApplicable
        && let sugg = format!("{}", Sugg::hir_with_applicability(cx, cast_expr, "_", &mut app))
        && let Some((_, after_lt)) = sugg.split_once("::<")
    {
        span_lint_and_sugg(
            cx,
            PTR_CAST_CONSTNESS,
            expr.span,
            "changing constness of a null pointer",
            format!("use `{method}()` directly instead"),
            format!("{prefix}::ptr::{method}::<{after_lt}"),
            app,
        );
    }
}
