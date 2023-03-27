use clippy_utils::diagnostics::span_lint_and_help;
use rustc_hir::{Local, TyKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Detects when a variable is declared with an explicit type of `_`.
    /// ### Why is this bad?
    /// It adds noise, `: _` provides zero clarity or utility.
    /// ### Example
    /// ```rust,ignore
    /// let my_number: _ = 1;
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// let my_number = 1;
    /// ```
    #[clippy::version = "1.69.0"]
    pub LET_WITH_TYPE_UNDERSCORE,
    complexity,
    "unneeded underscore type (`_`) in a variable declaration"
}
declare_lint_pass!(UnderscoreTyped => [LET_WITH_TYPE_UNDERSCORE]);

impl LateLintPass<'_> for UnderscoreTyped {
    fn check_local<'tcx>(&mut self, cx: &LateContext<'tcx>, local: &'tcx Local<'tcx>) {
        if_chain! {
            if !in_external_macro(cx.tcx.sess, local.span);
            if let Some(ty) = local.ty; // Ensure that it has a type defined
            if let TyKind::Infer = &ty.kind; // that type is '_'
            if local.span.ctxt() == ty.span.ctxt();
            then {
                span_lint_and_help(cx,
                    LET_WITH_TYPE_UNDERSCORE,
                    local.span,
                    "variable declared with type underscore",
                    Some(ty.span.with_lo(local.pat.span.hi())),
                    "remove the explicit type `_` declaration"
                )
            }
        };
    }
}
