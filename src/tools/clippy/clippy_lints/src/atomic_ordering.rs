use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::match_def_path;
use if_chain::if_chain;
use rustc_hir::def_id::DefId;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for usage of invalid atomic
    /// ordering in atomic loads/stores/exchanges/updates and
    /// memory fences.
    ///
    /// **Why is this bad?** Using an invalid atomic ordering
    /// will cause a panic at run-time.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,no_run
    /// # use std::sync::atomic::{self, AtomicU8, Ordering};
    ///
    /// let x = AtomicU8::new(0);
    ///
    /// // Bad: `Release` and `AcqRel` cannot be used for `load`.
    /// let _ = x.load(Ordering::Release);
    /// let _ = x.load(Ordering::AcqRel);
    ///
    /// // Bad: `Acquire` and `AcqRel` cannot be used for `store`.
    /// x.store(1, Ordering::Acquire);
    /// x.store(2, Ordering::AcqRel);
    ///
    /// // Bad: `Relaxed` cannot be used as a fence's ordering.
    /// atomic::fence(Ordering::Relaxed);
    /// atomic::compiler_fence(Ordering::Relaxed);
    ///
    /// // Bad: `Release` and `AcqRel` are both always invalid
    /// // for the failure ordering (the last arg).
    /// let _ = x.compare_exchange(1, 2, Ordering::SeqCst, Ordering::Release);
    /// let _ = x.compare_exchange_weak(2, 3, Ordering::AcqRel, Ordering::AcqRel);
    ///
    /// // Bad: The failure ordering is not allowed to be
    /// // stronger than the success order, and `SeqCst` is
    /// // stronger than `Relaxed`.
    /// let _ = x.fetch_update(Ordering::Relaxed, Ordering::SeqCst, |val| Some(val + val));
    /// ```
    pub INVALID_ATOMIC_ORDERING,
    correctness,
    "usage of invalid atomic ordering in atomic operations and memory fences"
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
        if let ExprKind::MethodCall(method_path, _, args, _) = &expr.kind;
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
        if let ExprKind::Call(func, args) = expr.kind;
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

fn opt_ordering_defid(cx: &LateContext<'_>, ord_arg: &Expr<'_>) -> Option<DefId> {
    if let ExprKind::Path(ref ord_qpath) = ord_arg.kind {
        cx.qpath_res(ord_qpath, ord_arg.hir_id).opt_def_id()
    } else {
        None
    }
}

fn check_atomic_compare_exchange(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if_chain! {
        if let ExprKind::MethodCall(method_path, _, args, _) = &expr.kind;
        let method = method_path.ident.name.as_str();
        if type_is_atomic(cx, &args[0]);
        if method == "compare_exchange" || method == "compare_exchange_weak" || method == "fetch_update";
        let (success_order_arg, failure_order_arg) = if method == "fetch_update" {
            (&args[1], &args[2])
        } else {
            (&args[3], &args[4])
        };
        if let Some(fail_ordering_def_id) = opt_ordering_defid(cx, failure_order_arg);
        then {
            // Helper type holding on to some checking and error reporting data. Has
            // - (success ordering name,
            // - list of failure orderings forbidden by the success order,
            // - suggestion message)
            type OrdLintInfo = (&'static str, &'static [&'static str], &'static str);
            let relaxed: OrdLintInfo = ("Relaxed", &["SeqCst", "Acquire"], "ordering mode `Relaxed`");
            let acquire: OrdLintInfo = ("Acquire", &["SeqCst"], "ordering modes `Acquire` or `Relaxed`");
            let seq_cst: OrdLintInfo = ("SeqCst", &[], "ordering modes `Acquire`, `SeqCst` or `Relaxed`");
            let release = ("Release", relaxed.1, relaxed.2);
            let acqrel = ("AcqRel", acquire.1, acquire.2);
            let search = [relaxed, acquire, seq_cst, release, acqrel];

            let success_lint_info = opt_ordering_defid(cx, success_order_arg)
                .and_then(|success_ord_def_id| -> Option<OrdLintInfo> {
                    search
                        .iter()
                        .find(|(ordering, ..)| {
                            match_def_path(cx, success_ord_def_id,
                                &["core", "sync", "atomic", "Ordering", ordering])
                        })
                        .copied()
                });

            if match_ordering_def_path(cx, fail_ordering_def_id, &["Release", "AcqRel"]) {
                // If we don't know the success order is, use what we'd suggest
                // if it were maximally permissive.
                let suggested = success_lint_info.unwrap_or(seq_cst).2;
                span_lint_and_help(
                    cx,
                    INVALID_ATOMIC_ORDERING,
                    failure_order_arg.span,
                    &format!(
                        "{}'s failure ordering may not be `Release` or `AcqRel`",
                        method,
                    ),
                    None,
                    &format!("consider using {} instead", suggested),
                );
            } else if let Some((success_ord_name, bad_ords_given_success, suggested)) = success_lint_info {
                if match_ordering_def_path(cx, fail_ordering_def_id, bad_ords_given_success) {
                    span_lint_and_help(
                        cx,
                        INVALID_ATOMIC_ORDERING,
                        failure_order_arg.span,
                        &format!(
                            "{}'s failure ordering may not be stronger than the success ordering of `{}`",
                            method,
                            success_ord_name,
                        ),
                        None,
                        &format!("consider using {} instead", suggested),
                    );
                }
            }
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for AtomicOrdering {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        check_atomic_load_store(cx, expr);
        check_memory_fence(cx, expr);
        check_atomic_compare_exchange(cx, expr);
    }
}
