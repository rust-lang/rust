use clippy_utils::diagnostics::span_lint_and_help;
use rustc_hir::{Impl, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for direct implementations of `ToString`.
    /// ### Why is this bad?
    /// This trait is automatically implemented for any type which implements the `Display` trait.
    /// As such, `ToString` shouldn’t be implemented directly: `Display` should be implemented instead,
    /// and you get the `ToString` implementation for free.
    /// ### Example
    /// ```no_run
    /// struct Point {
    ///   x: usize,
    ///   y: usize,
    /// }
    ///
    /// impl ToString for Point {
    ///   fn to_string(&self) -> String {
    ///     format!("({}, {})", self.x, self.y)
    ///   }
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// struct Point {
    ///   x: usize,
    ///   y: usize,
    /// }
    ///
    /// impl std::fmt::Display for Point {
    ///   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    ///     write!(f, "({}, {})", self.x, self.y)
    ///   }
    /// }
    /// ```
    #[clippy::version = "1.78.0"]
    pub TO_STRING_TRAIT_IMPL,
    style,
    "check for direct implementations of `ToString`"
}

declare_lint_pass!(ToStringTraitImpl => [TO_STRING_TRAIT_IMPL]);

impl<'tcx> LateLintPass<'tcx> for ToStringTraitImpl {
    fn check_item(&mut self, cx: &LateContext<'tcx>, it: &'tcx Item<'tcx>) {
        if let ItemKind::Impl(Impl {
            of_trait: Some(of_trait),
            ..
        }) = it.kind
            && let Some(trait_did) = of_trait.trait_ref.trait_def_id()
            && cx.tcx.is_diagnostic_item(sym::ToString, trait_did)
        {
            span_lint_and_help(
                cx,
                TO_STRING_TRAIT_IMPL,
                it.span,
                "direct implementation of `ToString`",
                None,
                "prefer implementing `Display` instead",
            );
        }
    }
}
