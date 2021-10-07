use super::utils::derefs_to_slice;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::ITER_COUNT;

pub(crate) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'_>, recv: &'tcx Expr<'tcx>, iter_method: &str) {
    let ty = cx.typeck_results().expr_ty(recv);
    let caller_type = if derefs_to_slice(cx, recv, ty).is_some() {
        "slice"
    } else if is_type_diagnostic_item(cx, ty, sym::Vec) {
        "Vec"
    } else if is_type_diagnostic_item(cx, ty, sym::VecDeque) {
        "VecDeque"
    } else if is_type_diagnostic_item(cx, ty, sym::HashSet) {
        "HashSet"
    } else if is_type_diagnostic_item(cx, ty, sym::HashMap) {
        "HashMap"
    } else if is_type_diagnostic_item(cx, ty, sym::BTreeMap) {
        "BTreeMap"
    } else if is_type_diagnostic_item(cx, ty, sym::BTreeSet) {
        "BTreeSet"
    } else if is_type_diagnostic_item(cx, ty, sym::LinkedList) {
        "LinkedList"
    } else if is_type_diagnostic_item(cx, ty, sym::BinaryHeap) {
        "BinaryHeap"
    } else {
        return;
    };
    let mut applicability = Applicability::MachineApplicable;
    span_lint_and_sugg(
        cx,
        ITER_COUNT,
        expr.span,
        &format!("called `.{}().count()` on a `{}`", iter_method, caller_type),
        "try",
        format!(
            "{}.len()",
            snippet_with_applicability(cx, recv.span, "..", &mut applicability),
        ),
        applicability,
    );
}
