use crate::{LateContext, LateLintPass, LintContext};

use hir::{Expr, Pat};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_middle::ty;
use rustc_span::sym;

declare_lint! {
    /// ### What it does
    ///
    /// Checks for `for` loops over `Option` or `Result` values.
    ///
    /// ### Why is this bad?
    /// Readability. This is more clearly expressed as an `if
    /// let`.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # let opt = Some(1);
    /// # let res: Result<i32, std::io::Error> = Ok(1);
    /// for x in opt {
    ///     // ..
    /// }
    ///
    /// for x in &res {
    ///     // ..
    /// }
    ///
    /// for x in res.iter() {
    ///     // ..
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let opt = Some(1);
    /// # let res: Result<i32, std::io::Error> = Ok(1);
    /// if let Some(x) = opt {
    ///     // ..
    /// }
    ///
    /// if let Ok(x) = res {
    ///     // ..
    /// }
    /// ```
    pub FOR_LOOP_OVER_FALLIBLES,
    Warn,
    "for-looping over an `Option` or a `Result`, which is more clearly expressed as an `if let`"
}

declare_lint_pass!(ForLoopOverFallibles => [FOR_LOOP_OVER_FALLIBLES]);

impl<'tcx> LateLintPass<'tcx> for ForLoopOverFallibles {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let Some((pat, arg)) = extract_for_loop(expr) else { return };

        let ty = cx.typeck_results().expr_ty(arg);

        let ty::Adt(adt, _) = ty.kind() else { return };

        let (article, ty, var) = match adt.did() {
            did if cx.tcx.is_diagnostic_item(sym::Option, did) => ("an", "Option", "Some"),
            did if cx.tcx.is_diagnostic_item(sym::Result, did) => ("a", "Result", "Ok"),
            _ => return,
        };

        let Ok(arg_snip) = cx.sess().source_map().span_to_snippet(arg.span) else { return };

        let msg = format!(
            "for loop over `{arg_snip}`, which is {article} `{ty}`. This is more readably written as an `if let` statement",
        );

        cx.struct_span_lint(FOR_LOOP_OVER_FALLIBLES, arg.span, |diag| {
            let mut warn = diag.build(msg);

            if let Some(recv) = extract_iterator_next_call(cx, arg)
            && let Ok(recv_snip) = cx.sess().source_map().span_to_snippet(recv.span)
            {
                warn.span_suggestion(
                    recv.span.between(arg.span.shrink_to_hi()),
                    format!("to iterate over `{recv_snip}` remove the call to `next`"),
                    "",
                    Applicability::MaybeIncorrect
                );
            }

            warn.multipart_suggestion_verbose(
                "consider using `if let` to clear intent",
                vec![
                    // NB can't use `until` here because `expr.span` and `pat.span` have different syntax contexts
                    (expr.span.with_hi(pat.span.lo()), format!("if let {var}(")),
                    (pat.span.between(arg.span), format!(") = ")),
                ],
                Applicability::MachineApplicable,
            );

            warn.emit()
        })
    }
}

fn extract_for_loop<'tcx>(expr: &Expr<'tcx>) -> Option<(&'tcx Pat<'tcx>, &'tcx Expr<'tcx>)> {
    if let hir::ExprKind::DropTemps(e) = expr.kind
    && let hir::ExprKind::Match(iterexpr, [arm], hir::MatchSource::ForLoopDesugar) = e.kind
    && let hir::ExprKind::Call(_, [arg]) = iterexpr.kind
    && let hir::ExprKind::Loop(block, ..) = arm.body.kind
    && let [stmt] = block.stmts
    && let hir::StmtKind::Expr(e) = stmt.kind
    && let hir::ExprKind::Match(_, [_, some_arm], _) = e.kind
    && let hir::PatKind::Struct(_, [field], _) = some_arm.pat.kind
    {
        Some((field.pat, arg))
    } else {
        None
    }
}

fn extract_iterator_next_call<'tcx>(
    cx: &LateContext<'_>,
    expr: &Expr<'tcx>,
) -> Option<&'tcx Expr<'tcx>> {
    // This won't work for `Iterator::next(iter)`, is this an issue?
    if let hir::ExprKind::MethodCall(_, [recv], _) = expr.kind
    && cx.typeck_results().type_dependent_def_id(expr.hir_id) == cx.tcx.lang_items().next_fn()
    {
        Some(recv)
    } else {
        return None
    }
}
