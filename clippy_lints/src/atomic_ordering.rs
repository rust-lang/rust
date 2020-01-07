use crate::utils::{match_def_path, span_help_and_lint};
use if_chain::if_chain;
use rustc::declare_lint_pass;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::ty;
use rustc_hir::def_id::DefId;
use rustc_hir::*;
use rustc_session::declare_tool_lint;

declare_clippy_lint! {
    /// **What it does:** Checks for usage of invalid atomic
    /// ordering in Atomic*::{load, store} calls.
    ///
    /// **Why is this bad?** Using an invalid atomic ordering
    /// will cause a panic at run-time.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,no_run
    /// # use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let x = AtomicBool::new(true);
    ///
    /// let _ = x.load(Ordering::Release);
    /// let _ = x.load(Ordering::AcqRel);
    ///
    /// x.store(false, Ordering::Acquire);
    /// x.store(false, Ordering::AcqRel);
    /// ```
    pub INVALID_ATOMIC_ORDERING,
    correctness,
    "usage of invalid atomic ordering in atomic load/store calls"
}

declare_lint_pass!(AtomicOrdering => [INVALID_ATOMIC_ORDERING]);

const ATOMIC_TYPES: [&str; 12] = [
    "AtomicBool",
    "AtomicI8",
    "AtomicI16",
    "AtomicI32",
    "AtomicI64",
    "AtomicIsize",
    "AtomicPtr",
    "AtomicU8",
    "AtomicU16",
    "AtomicU32",
    "AtomicU64",
    "AtomicUsize",
];

fn type_is_atomic(cx: &LateContext<'_, '_>, expr: &Expr<'_>) -> bool {
    if let ty::Adt(&ty::AdtDef { did, .. }, _) = cx.tables.expr_ty(expr).kind {
        ATOMIC_TYPES
            .iter()
            .any(|ty| match_def_path(cx, did, &["core", "sync", "atomic", ty]))
    } else {
        false
    }
}

fn match_ordering_def_path(cx: &LateContext<'_, '_>, did: DefId, orderings: &[&str]) -> bool {
    orderings
        .iter()
        .any(|ordering| match_def_path(cx, did, &["core", "sync", "atomic", "Ordering", ordering]))
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for AtomicOrdering {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::MethodCall(ref method_path, _, args) = &expr.kind;
            let method = method_path.ident.name.as_str();
            if type_is_atomic(cx, &args[0]);
            if method == "load" || method == "store";
            let ordering_arg = if method == "load" { &args[1] } else { &args[2] };
            if let ExprKind::Path(ref ordering_qpath) = ordering_arg.kind;
            if let Some(ordering_def_id) = cx.tables.qpath_res(ordering_qpath, ordering_arg.hir_id).opt_def_id();
            then {
                if method == "load" &&
                    match_ordering_def_path(cx, ordering_def_id, &["Release", "AcqRel"]) {
                    span_help_and_lint(
                        cx,
                        INVALID_ATOMIC_ORDERING,
                        ordering_arg.span,
                        "atomic loads cannot have `Release` and `AcqRel` ordering",
                        "consider using ordering modes `Acquire`, `SeqCst` or `Relaxed`"
                    );
                } else if method == "store" &&
                    match_ordering_def_path(cx, ordering_def_id, &["Acquire", "AcqRel"]) {
                    span_help_and_lint(
                        cx,
                        INVALID_ATOMIC_ORDERING,
                        ordering_arg.span,
                        "atomic stores cannot have `Acquire` and `AcqRel` ordering",
                        "consider using ordering modes `Release`, `SeqCst` or `Relaxed`"
                    );
                }
            }
        }
    }
}
