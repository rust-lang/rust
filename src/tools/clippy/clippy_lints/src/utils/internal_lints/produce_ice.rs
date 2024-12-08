use rustc_ast::ast::NodeId;
use rustc_ast::visit::FnKind;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
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
    pub PRODUCE_ICE,
    internal,
    "this message should not appear anywhere as we ICE before and don't emit the lint"
}

declare_lint_pass!(ProduceIce => [PRODUCE_ICE]);

impl EarlyLintPass for ProduceIce {
    fn check_fn(&mut self, _: &EarlyContext<'_>, fn_kind: FnKind<'_>, _: Span, _: NodeId) {
        assert!(!is_trigger_fn(fn_kind), "Would you like some help with that?");
    }
}

fn is_trigger_fn(fn_kind: FnKind<'_>) -> bool {
    match fn_kind {
        FnKind::Fn(_, ident, ..) => ident.name.as_str() == "it_looks_like_you_are_trying_to_kill_clippy",
        FnKind::Closure(..) => false,
    }
}
