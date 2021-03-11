use crate::methods::derefs_to_slice;
use crate::utils::{
    get_parent_expr, is_type_diagnostic_item, match_type, paths, snippet_with_applicability, span_lint_and_sugg,
};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::GET_UNWRAP;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &hir::Expr<'_>, get_args: &'tcx [hir::Expr<'_>], is_mut: bool) {
    // Note: we don't want to lint `get_mut().unwrap` for `HashMap` or `BTreeMap`,
    // because they do not implement `IndexMut`
    let mut applicability = Applicability::MachineApplicable;
    let expr_ty = cx.typeck_results().expr_ty(&get_args[0]);
    let get_args_str = if get_args.len() > 1 {
        snippet_with_applicability(cx, get_args[1].span, "..", &mut applicability)
    } else {
        return; // not linting on a .get().unwrap() chain or variant
    };
    let mut needs_ref;
    let caller_type = if derefs_to_slice(cx, &get_args[0], expr_ty).is_some() {
        needs_ref = get_args_str.parse::<usize>().is_ok();
        "slice"
    } else if is_type_diagnostic_item(cx, expr_ty, sym::vec_type) {
        needs_ref = get_args_str.parse::<usize>().is_ok();
        "Vec"
    } else if is_type_diagnostic_item(cx, expr_ty, sym::vecdeque_type) {
        needs_ref = get_args_str.parse::<usize>().is_ok();
        "VecDeque"
    } else if !is_mut && is_type_diagnostic_item(cx, expr_ty, sym::hashmap_type) {
        needs_ref = true;
        "HashMap"
    } else if !is_mut && match_type(cx, expr_ty, &paths::BTREEMAP) {
        needs_ref = true;
        "BTreeMap"
    } else {
        return; // caller is not a type that we want to lint
    };

    let mut span = expr.span;

    // Handle the case where the result is immediately dereferenced
    // by not requiring ref and pulling the dereference into the
    // suggestion.
    if_chain! {
        if needs_ref;
        if let Some(parent) = get_parent_expr(cx, expr);
        if let hir::ExprKind::Unary(hir::UnOp::Deref, _) = parent.kind;
        then {
            needs_ref = false;
            span = parent.span;
        }
    }

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
        &format!(
            "called `.get{0}().unwrap()` on a {1}. Using `[]` is more clear and more concise",
            mut_str, caller_type
        ),
        "try this",
        format!(
            "{}{}[{}]",
            borrow_str,
            snippet_with_applicability(cx, get_args[0].span, "..", &mut applicability),
            get_args_str
        ),
        applicability,
    );
}
