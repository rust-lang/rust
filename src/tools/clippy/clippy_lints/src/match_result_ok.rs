use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{higher, is_res_lang_ctor, sym};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unnecessary `ok()` in `while let`.
    ///
    /// ### Why is this bad?
    /// Calling `ok()` in `while let` is unnecessary, instead match
    /// on `Ok(pat)`
    ///
    /// ### Example
    /// ```ignore
    /// while let Some(value) = iter.next().ok() {
    ///     vec.push(value)
    /// }
    ///
    /// if let Some(value) = iter.next().ok() {
    ///     vec.push(value)
    /// }
    /// ```
    /// Use instead:
    /// ```ignore
    /// while let Ok(value) = iter.next() {
    ///     vec.push(value)
    /// }
    ///
    /// if let Ok(value) = iter.next() {
    ///        vec.push(value)
    /// }
    /// ```
    #[clippy::version = "1.57.0"]
    pub MATCH_RESULT_OK,
    style,
    "usage of `ok()` in `let Some(pat)` statements is unnecessary, match on `Ok(pat)` instead"
}

declare_lint_pass!(MatchResultOk => [MATCH_RESULT_OK]);

impl<'tcx> LateLintPass<'tcx> for MatchResultOk {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let (let_pat, let_expr, ifwhile) =
            if let Some(higher::IfLet { let_pat, let_expr, .. }) = higher::IfLet::hir(cx, expr) {
                (let_pat, let_expr, "if")
            } else if let Some(higher::WhileLet { let_pat, let_expr, .. }) = higher::WhileLet::hir(expr) {
                (let_pat, let_expr, "while")
            } else {
                return;
            };

        if let ExprKind::MethodCall(ok_path, recv, [], ..) = let_expr.kind //check is expr.ok() has type Result<T,E>.ok(, _)
            && let PatKind::TupleStruct(ref pat_path, [ok_pat], _)  = let_pat.kind //get operation
            && ok_path.ident.name == sym::ok
            && is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(recv), sym::Result)
            && is_res_lang_ctor(cx, cx.qpath_res(pat_path, let_pat.hir_id), LangItem::OptionSome)
            && let ctxt = expr.span.ctxt()
            && let_expr.span.ctxt() == ctxt
            && let_pat.span.ctxt() == ctxt
        {
            let mut applicability = Applicability::MachineApplicable;
            let some_expr_string = snippet_with_context(cx, ok_pat.span, ctxt, "", &mut applicability).0;
            let trimmed_ok = snippet_with_context(cx, recv.span, ctxt, "", &mut applicability).0;
            let sugg = format!(
                "{ifwhile} let Ok({some_expr_string}) = {}",
                trimmed_ok.trim().trim_end_matches('.'),
            );
            span_lint_and_sugg(
                cx,
                MATCH_RESULT_OK,
                expr.span.with_hi(let_expr.span.hi()),
                "matching on `Some` with `ok()` is redundant",
                format!("consider matching on `Ok({some_expr_string})` and removing the call to `ok` instead"),
                sugg,
                applicability,
            );
        }
    }
}
