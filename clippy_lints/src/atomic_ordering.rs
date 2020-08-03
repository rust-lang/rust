use crate::utils::{match_def_path, span_lint_and_help};
use if_chain::if_chain;
use rustc_hir::def_id::DefId;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for usage of invalid atomic
    /// ordering in atomic loads/stores and memory fences.
    ///
    /// **Why is this bad?** Using an invalid atomic ordering
    /// will cause a panic at run-time.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,no_run
    /// # use std::sync::atomic::{self, AtomicBool, Ordering};
    ///
    /// let x = AtomicBool::new(true);
    ///
    /// let _ = x.load(Ordering::Release);
    /// let _ = x.load(Ordering::AcqRel);
    ///
    /// x.store(false, Ordering::Acquire);
    /// x.store(false, Ordering::AcqRel);
    ///
    /// atomic::fence(Ordering::Relaxed);
    /// atomic::compiler_fence(Ordering::Relaxed);
    /// ```
    pub INVALID_ATOMIC_ORDERING,
    correctness,
    "usage of invalid atomic ordering in atomic loads/stores and memory fences"
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

fn type_is_atomic(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let ty::Adt(&ty::AdtDef { did, .. }, _) = cx.typeck_results().expr_ty(expr).kind() {
        ATOMIC_TYPES
            .iter()
            .any(|ty| match_def_path(cx, did, &["core", "sync", "atomic", ty]))
    } else {
        false
    }
}

fn match_ordering_def_path(cx: &LateContext<'_>, did: DefId, orderings: &[&str]) -> bool {
    orderings
        .iter()
        .any(|ordering| match_def_path(cx, did, &["core", "sync", "atomic", "Ordering", ordering]))
}

fn check_atomic_load_store(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if_chain! {
        if let ExprKind::MethodCall(ref method_path, _, args, _) = &expr.kind;
        let method = method_path.ident.name.as_str();
        if type_is_atomic(cx, &args[0]);
        if method == "load" || method == "store";
        let ordering_arg = if method == "load" { &args[1] } else { &args[2] };
        if let ExprKind::Path(ref ordering_qpath) = ordering_arg.kind;
        if let Some(ordering_def_id) = cx.qpath_res(ordering_qpath, ordering_arg.hir_id).opt_def_id();
        then {
            if method == "load" &&
                match_ordering_def_path(cx, ordering_def_id, &["Release", "AcqRel"]) {
                span_lint_and_help(
                    cx,
                    INVALID_ATOMIC_ORDERING,
                    ordering_arg.span,
                    "atomic loads cannot have `Release` and `AcqRel` ordering",
                    None,
                    "consider using ordering modes `Acquire`, `SeqCst` or `Relaxed`"
                );
            } else if method == "store" &&
                match_ordering_def_path(cx, ordering_def_id, &["Acquire", "AcqRel"]) {
                span_lint_and_help(
                    cx,
                    INVALID_ATOMIC_ORDERING,
                    ordering_arg.span,
                    "atomic stores cannot have `Acquire` and `AcqRel` ordering",
                    None,
                    "consider using ordering modes `Release`, `SeqCst` or `Relaxed`"
                );
            }
        }
    }
}

fn check_memory_fence(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if_chain! {
        if let ExprKind::Call(ref func, ref args) = expr.kind;
        if let ExprKind::Path(ref func_qpath) = func.kind;
        if let Some(def_id) = cx.qpath_res(func_qpath, func.hir_id).opt_def_id();
        if ["fence", "compiler_fence"]
            .iter()
            .any(|func| match_def_path(cx, def_id, &["core", "sync", "atomic", func]));
        if let ExprKind::Path(ref ordering_qpath) = &args[0].kind;
        if let Some(ordering_def_id) = cx.qpath_res(ordering_qpath, args[0].hir_id).opt_def_id();
        if match_ordering_def_path(cx, ordering_def_id, &["Relaxed"]);
        then {
            span_lint_and_help(
                cx,
                INVALID_ATOMIC_ORDERING,
                args[0].span,
                "memory fences cannot have `Relaxed` ordering",
                None,
                "consider using ordering modes `Acquire`, `Release`, `AcqRel` or `SeqCst`"
            );
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for AtomicOrdering {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        check_atomic_load_store(cx, expr);
        check_memory_fence(cx, expr);
    }
}
