use super::utils::derefs_to_slice;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::res::MaybeDef;
use clippy_utils::source::snippet_with_applicability;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::{Symbol, sym};

use super::ITER_COUNT;

pub(crate) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'_>, recv: &'tcx Expr<'tcx>, iter_method: Symbol) {
    let ty = cx.typeck_results().expr_ty(recv);
    let caller_type = match ty.opt_diag_name(cx) {
        _ if derefs_to_slice(cx, recv, ty).is_some() => "slice",
        Some(sym::Vec) => "Vec",
        Some(sym::VecDeque) => "VecDeque",
        Some(sym::HashSet) => "HashSet",
        Some(sym::HashMap) => "HashMap",
        Some(sym::BTreeMap) => "BTreeMap",
        Some(sym::BTreeSet) => "BTreeSet",
        Some(sym::LinkedList) => "LinkedList",
        Some(sym::BinaryHeap) => "BinaryHeap",
        _ => return,
    };
    let mut applicability = Applicability::MachineApplicable;
    span_lint_and_sugg(
        cx,
        ITER_COUNT,
        expr.span,
        format!("called `.{iter_method}().count()` on a `{caller_type}`"),
        "try",
        format!(
            "{}.len()",
            snippet_with_applicability(cx, recv.span, "..", &mut applicability),
        ),
        applicability,
    );
}
