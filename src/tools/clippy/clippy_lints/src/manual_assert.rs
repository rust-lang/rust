use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::macros::{root_macro_call, FormatArgsExpn};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{peel_blocks_with_stmt, sugg};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Detects `if`-then-`panic!` that can be replaced with `assert!`.
    ///
    /// ### Why is this bad?
    /// `assert!` is simpler than `if`-then-`panic!`.
    ///
    /// ### Example
    /// ```rust
    /// let sad_people: Vec<&str> = vec![];
    /// if !sad_people.is_empty() {
    ///     panic!("there are sad people: {:?}", sad_people);
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// let sad_people: Vec<&str> = vec![];
    /// assert!(sad_people.is_empty(), "there are sad people: {:?}", sad_people);
    /// ```
    #[clippy::version = "1.57.0"]
    pub MANUAL_ASSERT,
    pedantic,
    "`panic!` and only a `panic!` in `if`-then statement"
}

declare_lint_pass!(ManualAssert => [MANUAL_ASSERT]);

impl<'tcx> LateLintPass<'tcx> for ManualAssert {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if_chain! {
            if let ExprKind::If(cond, then, None) = expr.kind;
            if !matches!(cond.kind, ExprKind::Let(_));
            if !expr.span.from_expansion();
            let then = peel_blocks_with_stmt(then);
            if let Some(macro_call) = root_macro_call(then.span);
            if cx.tcx.item_name(macro_call.def_id) == sym::panic;
            if !cx.tcx.sess.source_map().is_multiline(cond.span);
            if let Some(format_args) = FormatArgsExpn::find_nested(cx, then, macro_call.expn);
            then {
                let mut applicability = Applicability::MachineApplicable;
                let format_args_snip = snippet_with_applicability(cx, format_args.inputs_span(), "..", &mut applicability);
                let cond = cond.peel_drop_temps();
                let (cond, not) = match cond.kind {
                    ExprKind::Unary(UnOp::Not, e) => (e, ""),
                    _ => (cond, "!"),
                };
                let cond_sugg = sugg::Sugg::hir_with_applicability(cx, cond, "..", &mut applicability).maybe_par();
                let sugg = format!("assert!({not}{cond_sugg}, {format_args_snip});");
                span_lint_and_sugg(
                    cx,
                    MANUAL_ASSERT,
                    expr.span,
                    "only a `panic!` in `if`-then statement",
                    "try",
                    sugg,
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}
