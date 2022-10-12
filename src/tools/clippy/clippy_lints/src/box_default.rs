use clippy_utils::{diagnostics::span_lint_and_help, is_default_equivalent, path_def_id};
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// checks for `Box::new(T::default())`, which is better written as
    /// `Box::<T>::default()`.
    ///
    /// ### Why is this bad?
    /// First, it's more complex, involving two calls instead of one.
    /// Second, `Box::default()` can be faster
    /// [in certain cases](https://nnethercote.github.io/perf-book/standard-library-types.html#box).
    ///
    /// ### Known problems
    /// The lint may miss some cases (e.g. Box::new(String::from(""))).
    /// On the other hand, it will trigger on cases where the `default`
    /// code comes from a macro that does something different based on
    /// e.g. target operating system.
    ///
    /// ### Example
    /// ```rust
    /// let x: Box<String> = Box::new(Default::default());
    /// ```
    /// Use instead:
    /// ```rust
    /// let x: Box<String> = Box::default();
    /// ```
    #[clippy::version = "1.65.0"]
    pub BOX_DEFAULT,
    perf,
    "Using Box::new(T::default()) instead of Box::default()"
}

declare_lint_pass!(BoxDefault => [BOX_DEFAULT]);

impl LateLintPass<'_> for BoxDefault {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let ExprKind::Call(box_new, [arg]) = expr.kind
            && let ExprKind::Path(QPath::TypeRelative(ty, seg)) = box_new.kind
            && let ExprKind::Call(..) = arg.kind
            && !in_external_macro(cx.sess(), expr.span)
            && expr.span.eq_ctxt(arg.span)
            && seg.ident.name == sym::new
            && path_def_id(cx, ty) == cx.tcx.lang_items().owned_box()
            && is_default_equivalent(cx, arg)
        {
            span_lint_and_help(
                cx,
                BOX_DEFAULT,
                expr.span,
                "`Box::new(_)` of default value",
                None,
                "use `Box::default()` instead",
            );
        }
    }
}
