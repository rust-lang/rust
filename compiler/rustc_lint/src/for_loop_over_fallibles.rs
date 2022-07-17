use crate::{LateContext, LateLintPass, LintContext};

use hir::{Expr, Pat};
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

        let Ok(pat_snip) = cx.sess().source_map().span_to_snippet(pat.span) else { return };
        let Ok(arg_snip) = cx.sess().source_map().span_to_snippet(arg.span) else { return };

        let help_string = format!(
            "consider replacing `for {pat_snip} in {arg_snip}` with `if let {var}({pat_snip}) = {arg_snip}`"
        );
        let msg = format!(
            "for loop over `{arg_snip}`, which is {article} `{ty}`. This is more readably written as an `if let` statement",
        );

        cx.struct_span_lint(FOR_LOOP_OVER_FALLIBLES, arg.span, |diag| {
            diag.build(msg).help(help_string).emit()
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