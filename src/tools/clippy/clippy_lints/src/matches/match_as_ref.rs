use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{is_res_lang_ctor, path_res, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir::{Arm, BindingAnnotation, ByRef, Expr, ExprKind, LangItem, Mutability, PatKind, QPath};
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

            let cast = if_chain! {
                if let ty::Adt(_, args) = input_ty.kind();
                let input_ty = args.type_at(0);
                if let ty::Adt(_, args) = output_ty.kind();
                let output_ty = args.type_at(0);
                if let ty::Ref(_, output_ty, _) = *output_ty.kind();
                if input_ty != output_ty;
                then {
                    ".map(|x| x as _)"
                } else {
                    ""
                }
            };

            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                MATCH_AS_REF,
                expr.span,
                &format!("use `{suggestion}()` instead"),
                "try this",
                format!(
                    "{}.{suggestion}(){cast}",
                    snippet_with_applicability(cx, ex.span, "_", &mut applicability),
                ),
                applicability,
            );
        }
    }
}

// Checks if arm has the form `None => None`
fn is_none_arm(cx: &LateContext<'_>, arm: &Arm<'_>) -> bool {
    matches!(
        arm.pat.kind,
        PatKind::Path(ref qpath) if is_res_lang_ctor(cx, cx.qpath_res(qpath, arm.pat.hir_id), LangItem::OptionNone)
    )
}

// Checks if arm has the form `Some(ref v) => Some(v)` (checks for `ref` and `ref mut`)
fn is_ref_some_arm(cx: &LateContext<'_>, arm: &Arm<'_>) -> Option<Mutability> {
    if_chain! {
        if let PatKind::TupleStruct(ref qpath, [first_pat, ..], _) = arm.pat.kind;
        if is_res_lang_ctor(cx, cx.qpath_res(qpath, arm.pat.hir_id), LangItem::OptionSome);
        if let PatKind::Binding(BindingAnnotation(ByRef::Yes, mutabl), .., ident, _) = first_pat.kind;
        if let ExprKind::Call(e, [arg]) = peel_blocks(arm.body).kind;
        if is_res_lang_ctor(cx, path_res(cx, e), LangItem::OptionSome);
        if let ExprKind::Path(QPath::Resolved(_, path2)) = arg.kind;
        if path2.segments.len() == 1 && ident.name == path2.segments[0].ident.name;
        then {
            return Some(mutabl)
        }
    }
    None
}
