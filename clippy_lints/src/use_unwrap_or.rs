use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::is_type_diagnostic_item;
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `.or(…).unwrap()` calls to Options and Results.
    ///
    /// ### Why is this bad?
    /// You should use `.unwrap_or(…)` instead for clarity.
    ///
    /// ### Example
    /// ```rust
    /// # let fallback = "fallback";
    /// // Result
    /// # type Error = &'static str;
    /// # let result: Result<&str, Error> = Err("error");
    /// let port = result.or::<Error>(Ok(fallback)).unwrap();
    ///
    /// // Option
    /// # let option: Option<&str> = None;
    /// let value = option.or(Some(fallback)).unwrap();
    /// ```
    /// Use instead:
    /// ```rust
    /// # let fallback = "fallback";
    /// // Result
    /// # let result: Result<&str, &str> = Err("error");
    /// let port = result.unwrap_or(fallback);
    ///
    /// // Option
    /// # let option: Option<&str> = None;
    /// let value = option.unwrap_or(fallback);
    /// ```
    #[clippy::version = "1.61.0"]
    pub USE_UNWRAP_OR,
    complexity,
    "checks for `.or(…).unwrap()` calls to Options and Results."
}
declare_lint_pass!(UseUnwrapOr => [USE_UNWRAP_OR]);

impl<'tcx> LateLintPass<'tcx> for UseUnwrapOr {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // look for x.or().unwrap()
        if_chain! {
            if let ExprKind::MethodCall(path, args, unwrap_span) = expr.kind;
            if path.ident.name == sym::unwrap;
            if let Some(caller) = args.first();
            if let ExprKind::MethodCall(caller_path, caller_args, or_span) = caller.kind;
            if caller_path.ident.name == sym::or;
            then {
                let ty = cx.typeck_results().expr_ty(&caller_args[0]); // get type of x (we later check if it's Option or Result)
                let title;
                let arg = &caller_args[1]; // the argument or(xyz) is called with

                if is_type_diagnostic_item(cx, ty, sym::Option) {
                    title = ".or(Some(…)).unwrap() found";
                    if !is(arg, "Some") {
                        return;
                    }

                } else if is_type_diagnostic_item(cx, ty, sym::Result) {
                    title = ".or(Ok(…)).unwrap() found";
                    if !is(arg, "Ok") {
                        return;
                    }
                } else {
                    // Someone has implemented a struct with .or(...).unwrap() chaining,
                    // but it's not an Option or a Result, so bail
                    return;
                }

                // span = or_span + unwrap_span
                let span = Span::new(or_span.lo(), unwrap_span.hi(), or_span.ctxt(), or_span.parent());

                span_lint_and_help(
                    cx,
                    USE_UNWRAP_OR,
                    span,
                    title,
                    None,
                    "use `unwrap_or()` instead"
                );
            }
        }
    }
}

/// is expr a Call to name?
/// name might be "Some", "Ok", "Err", etc.
fn is<'a>(expr: &Expr<'a>, name: &str) -> bool {
    if_chain! {
        if let ExprKind::Call(some_expr, _some_args) = expr.kind;
        if let ExprKind::Path(QPath::Resolved(_, path)) = &some_expr.kind;
        if let Some(path_segment) = path.segments.first();
        if path_segment.ident.name.as_str() == name;
        then {
            true
        }
        else {
            false
        }
    }
}
