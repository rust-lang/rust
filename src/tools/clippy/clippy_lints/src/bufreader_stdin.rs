use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::res::{MaybeDef as _, MaybeResPath as _};
use clippy_utils::source::snippet_opt;
use clippy_utils::sym;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass, impl_lint_pass};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `BufReader::new` with `Stdin` or `StdinLock`.
    ///
    /// ### Why is this bad?
    /// `Stdin` is already buffered. Re-buffering it only increases the memcpy calls.
    ///
    /// ### Example
    ///
    /// ```ignore
    /// let reader = std::io::BufReader::new(std::io::stdin());
    /// ```
    ///
    /// Use instead:
    ///
    /// ```ignore
    /// let stdin = std::io::stdin();
    /// let reader = stdin.lock();
    /// ```

    #[clippy::version = "1.97.0"]
    pub BUFREADER_STDIN,
    pedantic,
    "using `BufReader::new` with `Stdin` or `StdinLock` is unnecessary and less efficient"
}

impl_lint_pass!(BufreaderStdin => [BUFREADER_STDIN]);

pub struct BufreaderStdin {}

impl BufreaderStdin {
    pub fn new() -> Self {
        Self {}
    }
}

impl<'tcx> LateLintPass<'tcx> for BufreaderStdin {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Call(func, [arg]) = expr.kind
            && let ExprKind::Path(QPath::TypeRelative(ty, segment)) = func.kind
            && segment.ident.name == sym::new
            && ty.basic_res().is_diag_item(cx, sym::IoBufReader)
            && let Some(arg_did) = cx.typeck_results().expr_ty(arg).opt_def_id()
        {
            let arg_did_name = cx.tcx.get_diagnostic_name(arg_did);
            let (msg, help) = match arg_did_name {
                Some(sym::Stdin) => (
                    "wrapping already buffered `Stdin` into a `BufReader`",
                    "instead of wrapping `Stdin` in `BufReader`, use the `lock` method directly",
                ),
                Some(sym::StdinLock) => (
                    "wrapping already buffered `StdinLock` into a `BufReader`",
                    "instead of wrapping `StdinLock` in `BufReader`, use it directly",
                ),
                _ => return,
            };

            if arg.span.from_expansion() {
                span_lint(cx, BUFREADER_STDIN, expr.span, msg);
                return;
            }

            span_lint_and_then(cx, BUFREADER_STDIN, expr.span, msg, |diag| {
                let Some(snip) = snippet_opt(cx, arg.span) else {
                    return;
                };
                let sugg = match arg_did_name {
                    Some(sym::Stdin) => format!("{snip}.lock()"),
                    Some(sym::StdinLock) => snip,
                    _ => return,
                };

                diag.span_suggestion(expr.span, help, sugg, Applicability::MaybeIncorrect);
            });
        }
    }
}
