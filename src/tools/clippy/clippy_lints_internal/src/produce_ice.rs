use rustc_ast::ast::NodeId;
use rustc_ast::visit::FnKind;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_lint_defs::declare_tool_lint;
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_tool_lint! {
    /// ### What it does
    /// Not an actual lint. This lint is only meant for testing our customized internal compiler
    /// error message by calling `panic`.
    ///
    /// ### Why is this bad?
    /// ICE in large quantities can damage your teeth
    ///
    /// ### Example
    /// ```rust,ignore
    /// 🍦🍦🍦🍦🍦
    /// ```
    pub clippy::PRODUCE_ICE,
    Warn,
    "this message should not appear anywhere as we ICE before and don't emit the lint",
    report_in_external_macro: true
}

declare_lint_pass!(ProduceIce => [PRODUCE_ICE]);

impl EarlyLintPass for ProduceIce {
    fn check_fn(&mut self, ctx: &EarlyContext<'_>, fn_kind: FnKind<'_>, span: Span, _: NodeId) {
        if is_trigger_fn(fn_kind) {
            ctx.sess()
                .dcx()
                .span_delayed_bug(span, "Would you like some help with that?");
        }
    }
}

fn is_trigger_fn(fn_kind: FnKind<'_>) -> bool {
    match fn_kind {
        FnKind::Fn(_, _, func) => func.ident.name.as_str() == "it_looks_like_you_are_trying_to_kill_clippy",
        FnKind::Closure(..) => false,
    }
}
