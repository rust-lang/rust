use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{is_none_arm, is_res_lang_ctor, path_res, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir::{Arm, BindingMode, ByRef, Expr, ExprKind, LangItem, Mutability, PatKind, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty;

use super::MATCH_AS_REF;

pub(crate) fn check(cx: &LateContext<'_>, ex: &Expr<'_>, arms: &[Arm<'_>], expr: &Expr<'_>) {
    if arms.len() == 2 && arms[0].guard.is_none() && arms[1].guard.is_none() {
        let arm_ref_mut = if is_none_arm(cx, &arms[0]) {
            is_ref_some_arm(cx, &arms[1])
        } else if is_none_arm(cx, &arms[1]) {
            is_ref_some_arm(cx, &arms[0])
        } else {
            None
        };
        if let Some(rb) = arm_ref_mut {
            let suggestion = match rb {
                Mutability::Not => "as_ref",
                Mutability::Mut => "as_mut",
            };

            let output_ty = cx.typeck_results().expr_ty(expr);
            let input_ty = cx.typeck_results().expr_ty(ex);

            let cast = if let ty::Adt(_, args) = input_ty.kind()
                && let input_ty = args.type_at(0)
                && let ty::Adt(_, args) = output_ty.kind()
                && let output_ty = args.type_at(0)
                && let ty::Ref(_, output_ty, _) = *output_ty.kind()
                && input_ty != output_ty
            {
                ".map(|x| x as _)"
            } else {
                ""
            };

            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                MATCH_AS_REF,
                expr.span,
                format!("use `{suggestion}()` instead"),
                "try",
                format!(
                    "{}.{suggestion}(){cast}",
                    snippet_with_applicability(cx, ex.span, "_", &mut applicability),
                ),
                applicability,
            );
        }
    }
}

// Checks if arm has the form `Some(ref v) => Some(v)` (checks for `ref` and `ref mut`)
fn is_ref_some_arm(cx: &LateContext<'_>, arm: &Arm<'_>) -> Option<Mutability> {
    if let PatKind::TupleStruct(ref qpath, [first_pat, ..], _) = arm.pat.kind
        && is_res_lang_ctor(cx, cx.qpath_res(qpath, arm.pat.hir_id), LangItem::OptionSome)
        && let PatKind::Binding(BindingMode(ByRef::Yes(mutabl), _), .., ident, _) = first_pat.kind
        && let ExprKind::Call(e, [arg]) = peel_blocks(arm.body).kind
        && is_res_lang_ctor(cx, path_res(cx, e), LangItem::OptionSome)
        && let ExprKind::Path(QPath::Resolved(_, path2)) = arg.kind
        && path2.segments.len() == 1
        && ident.name == path2.segments[0].ident.name
    {
        return Some(mutabl);
    }
    None
}
