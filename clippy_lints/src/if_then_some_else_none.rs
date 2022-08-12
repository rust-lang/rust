use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::eager_or_lazy::switch_to_eager_eval;
use clippy_utils::source::snippet_with_macro_callsite;
use clippy_utils::{contains_return, higher, is_else_clause, is_lang_ctor, meets_msrv, msrvs, peel_blocks};
use rustc_hir::LangItem::{OptionNone, OptionSome};
use rustc_hir::{Expr, ExprKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for if-else that could be written using either `bool::then` or `bool::then_some`.
    ///
    /// ### Why is this bad?
    /// Looks a little redundant. Using `bool::then` is more concise and incurs no loss of clarity.
    /// For simple calculations and known values, use `bool::then_some`, which is eagerly evaluated
    /// in comparison to `bool::then`.
    ///
    /// ### Example
    /// ```rust
    /// # let v = vec![0];
    /// let a = if v.is_empty() {
    ///     println!("true!");
    ///     Some(42)
    /// } else {
    ///     None
    /// };
    /// ```
    ///
    /// Could be written:
    ///
    /// ```rust
    /// # let v = vec![0];
    /// let a = v.is_empty().then(|| {
    ///     println!("true!");
    ///     42
    /// });
    /// ```
    #[clippy::version = "1.53.0"]
    pub IF_THEN_SOME_ELSE_NONE,
    restriction,
    "Finds if-else that could be written using either `bool::then` or `bool::then_some`"
}

pub struct IfThenSomeElseNone {
    msrv: Option<RustcVersion>,
}

impl IfThenSomeElseNone {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(IfThenSomeElseNone => [IF_THEN_SOME_ELSE_NONE]);

impl<'tcx> LateLintPass<'tcx> for IfThenSomeElseNone {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if !meets_msrv(self.msrv, msrvs::BOOL_THEN) {
            return;
        }

        if in_external_macro(cx.sess(), expr.span) {
            return;
        }

        // We only care about the top-most `if` in the chain
        if is_else_clause(cx.tcx, expr) {
            return;
        }

        if let Some(higher::If { cond, then, r#else: Some(els) }) = higher::If::hir(expr)
            && let ExprKind::Block(then_block, _) = then.kind
            && let Some(then_expr) = then_block.expr
            && let ExprKind::Call(then_call, [then_arg]) = then_expr.kind
            && let ExprKind::Path(ref then_call_qpath) = then_call.kind
            && is_lang_ctor(cx, then_call_qpath, OptionSome)
            && let ExprKind::Path(ref qpath) = peel_blocks(els).kind
            && is_lang_ctor(cx, qpath, OptionNone)
            && !stmts_contains_early_return(then_block.stmts)
        {
            let cond_snip = snippet_with_macro_callsite(cx, cond.span, "[condition]");
            let cond_snip = if matches!(cond.kind, ExprKind::Unary(_, _) | ExprKind::Binary(_, _, _)) {
                format!("({})", cond_snip)
            } else {
                cond_snip.into_owned()
            };
            let arg_snip = snippet_with_macro_callsite(cx, then_arg.span, "");
            let mut method_body = if then_block.stmts.is_empty() {
                arg_snip.into_owned()
            } else {
                format!("{{ /* snippet */ {} }}", arg_snip)
            };
            let method_name = if switch_to_eager_eval(cx, expr) && meets_msrv(self.msrv, msrvs::BOOL_THEN_SOME) {
                "then_some"
            } else {
                method_body.insert_str(0, "|| ");
                "then"
            };

            let help = format!(
                "consider using `bool::{}` like: `{}.{}({})`",
                method_name, cond_snip, method_name, method_body,
            );
            span_lint_and_help(
                cx,
                IF_THEN_SOME_ELSE_NONE,
                expr.span,
                &format!("this could be simplified with `bool::{}`", method_name),
                None,
                &help,
            );
        }
    }

    extract_msrv_attr!(LateContext);
}

fn stmts_contains_early_return(stmts: &[Stmt<'_>]) -> bool {
    stmts.iter().any(|stmt| {
        let Stmt { kind: StmtKind::Semi(e), .. } = stmt else { return false };

        contains_return(e)
    })
}
