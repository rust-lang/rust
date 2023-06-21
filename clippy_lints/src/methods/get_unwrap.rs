use super::utils::derefs_to_slice;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::get_parent_expr;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::GET_UNWRAP;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &hir::Expr<'_>,
    recv: &'tcx hir::Expr<'tcx>,
    get_arg: &'tcx hir::Expr<'_>,
    is_mut: bool,
) {
    // Note: we don't want to lint `get_mut().unwrap` for `HashMap` or `BTreeMap`,
    // because they do not implement `IndexMut`
    let mut applicability = Applicability::MachineApplicable;
    let expr_ty = cx.typeck_results().expr_ty(recv);
    let get_args_str = snippet_with_applicability(cx, get_arg.span, "..", &mut applicability);
    let caller_type = if derefs_to_slice(cx, recv, expr_ty).is_some() {
        "slice"
    } else if is_type_diagnostic_item(cx, expr_ty, sym::Vec) {
        "Vec"
    } else if is_type_diagnostic_item(cx, expr_ty, sym::VecDeque) {
        "VecDeque"
    } else if !is_mut && is_type_diagnostic_item(cx, expr_ty, sym::HashMap) {
        "HashMap"
    } else if !is_mut && is_type_diagnostic_item(cx, expr_ty, sym::BTreeMap) {
        "BTreeMap"
    } else {
        return; // caller is not a type that we want to lint
    };

    let mut span = expr.span;

    // Handle the case where the result is immediately dereferenced,
    // either directly be the user, or as a result of a method call or the like
    // by not requiring an explicit reference
    let needs_ref = if let Some(parent) = get_parent_expr(cx, expr)
        && let hir::ExprKind::Unary(hir::UnOp::Deref, _)
            | hir::ExprKind::MethodCall(..)
            | hir::ExprKind::Field(..)
            | hir::ExprKind::Index(..) = parent.kind
    {
        if let hir::ExprKind::Unary(hir::UnOp::Deref, _) = parent.kind {
            // if the user explicitly dereferences the result, we can adjust
            // the span to also include the deref part
            span = parent.span;
        }
        false
    } else {
        true
    };

    let mut_str = if is_mut { "_mut" } else { "" };
    let borrow_str = if !needs_ref {
        ""
    } else if is_mut {
        "&mut "
    } else {
        "&"
    };

    span_lint_and_sugg(
        cx,
        GET_UNWRAP,
        span,
        &format!("called `.get{mut_str}().unwrap()` on a {caller_type}. Using `[]` is more clear and more concise"),
        "try this",
        format!(
            "{borrow_str}{}[{get_args_str}]",
            snippet_with_applicability(cx, recv.span, "..", &mut applicability)
        ),
        applicability,
    );
}
