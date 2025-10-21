use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::{MaybeDef, MaybeQPath};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::option_arg_ty;
use clippy_utils::{is_none_arm, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir::{Arm, BindingMode, ByRef, Expr, ExprKind, LangItem, Mutability, PatKind, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty;

use super::MATCH_AS_REF;

pub(crate) fn check(cx: &LateContext<'_>, ex: &Expr<'_>, arms: &[Arm<'_>], expr: &Expr<'_>) {
    if let [arm1, arm2] = arms
        && arm1.guard.is_none()
        && arm2.guard.is_none()
        && let Some(arm_ref_mutbl) = if is_none_arm(cx, arm1) {
            as_ref_some_arm(cx, arm2)
        } else if is_none_arm(cx, arm2) {
            as_ref_some_arm(cx, arm1)
        } else {
            None
        }
    {
        let method = match arm_ref_mutbl {
            Mutability::Not => "as_ref",
            Mutability::Mut => "as_mut",
        };

        let output_ty = cx.typeck_results().expr_ty(expr);
        let input_ty = cx.typeck_results().expr_ty(ex);

        let cast = if let Some(input_ty) = option_arg_ty(cx, input_ty)
            && let Some(output_ty) = option_arg_ty(cx, output_ty)
            && let ty::Ref(_, output_ty, _) = *output_ty.kind()
            && input_ty != output_ty
        {
            ".map(|x| x as _)"
        } else {
            ""
        };

        let mut applicability = Applicability::MachineApplicable;
        span_lint_and_then(
            cx,
            MATCH_AS_REF,
            expr.span,
            format!("manual implementation of `Option::{method}`"),
            |diag| {
                diag.span_suggestion_verbose(
                    expr.span,
                    format!("use `Option::{method}()` directly"),
                    format!(
                        "{}.{method}(){cast}",
                        Sugg::hir_with_applicability(cx, ex, "_", &mut applicability).maybe_paren(),
                    ),
                    applicability,
                );
            },
        );
    }
}

// Checks if arm has the form `Some(ref v) => Some(v)` (checks for `ref` and `ref mut`)
fn as_ref_some_arm(cx: &LateContext<'_>, arm: &Arm<'_>) -> Option<Mutability> {
    if let PatKind::TupleStruct(ref qpath, [first_pat, ..], _) = arm.pat.kind
        && cx
            .qpath_res(qpath, arm.pat.hir_id)
            .ctor_parent(cx)
            .is_lang_item(cx, LangItem::OptionSome)
        && let PatKind::Binding(BindingMode(ByRef::Yes(mutabl), _), .., ident, _) = first_pat.kind
        && let ExprKind::Call(e, [arg]) = peel_blocks(arm.body).kind
        && e.res(cx).ctor_parent(cx).is_lang_item(cx, LangItem::OptionSome)
        && let ExprKind::Path(QPath::Resolved(_, path2)) = arg.kind
        && path2.segments.len() == 1
        && ident.name == path2.segments[0].ident.name
    {
        return Some(mutabl);
    }
    None
}
