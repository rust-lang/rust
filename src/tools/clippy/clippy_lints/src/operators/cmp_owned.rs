use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::MaybeQPath;
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::{implements_trait, is_copy};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, UnOp};
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;
use rustc_span::symbol::sym;

use super::CMP_OWNED;

pub(super) fn check(cx: &LateContext<'_>, op: BinOpKind, lhs: &Expr<'_>, rhs: &Expr<'_>) {
    if op.is_comparison() {
        check_op(cx, lhs, rhs, true);
        check_op(cx, rhs, lhs, false);
    }
}

#[derive(Default)]
struct EqImpl {
    ty_eq_other: bool,
    other_eq_ty: bool,
}
impl EqImpl {
    fn is_implemented(&self) -> bool {
        self.ty_eq_other || self.other_eq_ty
    }
}

fn symmetric_partial_eq<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>, other: Ty<'tcx>) -> Option<EqImpl> {
    cx.tcx.lang_items().eq_trait().map(|def_id| EqImpl {
        ty_eq_other: implements_trait(cx, ty, def_id, &[other.into()]),
        other_eq_ty: implements_trait(cx, other, def_id, &[ty.into()]),
    })
}

fn check_op(cx: &LateContext<'_>, expr: &Expr<'_>, other: &Expr<'_>, left: bool) {
    let typeck = cx.typeck_results();
    let (arg, arg_span) = match expr.kind {
        ExprKind::MethodCall(_, arg, [], _)
            if typeck
                .type_dependent_def_id(expr.hir_id)
                .and_then(|id| cx.tcx.trait_of_assoc(id))
                .is_some_and(|id| matches!(cx.tcx.get_diagnostic_name(id), Some(sym::ToString | sym::ToOwned))) =>
        {
            (arg, arg.span)
        },
        ExprKind::Call(path, [arg])
            if path
                .res(cx)
                .opt_def_id()
                .is_some_and(|did| match cx.tcx.get_diagnostic_name(did) {
                    Some(sym::from_str_method) => true,
                    Some(sym::from_fn) => !is_copy(cx, typeck.expr_ty(expr)),
                    _ => false,
                }) =>
        {
            (arg, arg.span)
        },
        _ => return,
    };

    let arg_ty = typeck.expr_ty(arg);
    let other_ty = typeck.expr_ty(other);

    let without_deref = symmetric_partial_eq(cx, arg_ty, other_ty).unwrap_or_default();
    let with_deref = arg_ty
        .builtin_deref(true)
        .and_then(|ty| symmetric_partial_eq(cx, ty, other_ty))
        .unwrap_or_default();

    if !with_deref.is_implemented() && !without_deref.is_implemented() {
        return;
    }

    let other_gets_derefed = matches!(other.kind, ExprKind::Unary(UnOp::Deref, _));

    let lint_span = if other_gets_derefed {
        expr.span.to(other.span)
    } else {
        expr.span
    };

    span_lint_and_then(
        cx,
        CMP_OWNED,
        lint_span,
        "this creates an owned instance just for comparison",
        |diag| {
            // This also catches `PartialEq` implementations that call `to_owned`.
            if other_gets_derefed {
                diag.span_label(lint_span, "try implementing the comparison without allocating");
                return;
            }

            let mut applicability = Applicability::MachineApplicable;
            let (arg_snip, _) = snippet_with_context(cx, arg_span, expr.span.ctxt(), "..", &mut applicability);
            let (expr_snip, eq_impl) = if with_deref.is_implemented() && !arg_ty.peel_refs().is_str() {
                (format!("*{arg_snip}"), with_deref)
            } else {
                (arg_snip.to_string(), without_deref)
            };

            let (span, hint) = if (eq_impl.ty_eq_other && left) || (eq_impl.other_eq_ty && !left) {
                (expr.span, expr_snip)
            } else {
                let span = expr.span.to(other.span);

                let cmp_span = if other.span < expr.span {
                    other.span.between(expr.span)
                } else {
                    expr.span.between(other.span)
                };

                let (cmp_snippet, _) = snippet_with_context(cx, cmp_span, expr.span.ctxt(), "..", &mut applicability);
                let (other_snippet, _) =
                    snippet_with_context(cx, other.span, expr.span.ctxt(), "..", &mut applicability);

                if eq_impl.ty_eq_other {
                    (span, format!("{expr_snip}{cmp_snippet}{other_snippet}"))
                } else {
                    (span, format!("{other_snippet}{cmp_snippet}{expr_snip}"))
                }
            };

            diag.span_suggestion(span, "try", hint, applicability);
        },
    );
}
