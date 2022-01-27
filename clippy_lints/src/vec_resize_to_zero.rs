use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{match_def_path, paths};
use if_chain::if_chain;
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Spanned;

declare_clippy_lint! {
    /// ### What it does
    /// Finds occurrences of `Vec::resize(0, an_int)`
    ///
    /// ### Why is this bad?
    /// This is probably an argument inversion mistake.
    ///
    /// ### Example
    /// ```rust
    /// vec!(1, 2, 3, 4, 5).resize(0, 5)
    /// ```
    #[clippy::version = "1.46.0"]
    pub VEC_RESIZE_TO_ZERO,
    correctness,
    "emptying a vector with `resize(0, an_int)` instead of `clear()` is probably an argument inversion mistake"
}

declare_lint_pass!(VecResizeToZero => [VEC_RESIZE_TO_ZERO]);

impl<'tcx> LateLintPass<'tcx> for VecResizeToZero {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let hir::ExprKind::MethodCall(path_segment, args, _) = expr.kind;
            if let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
            if match_def_path(cx, method_def_id, &paths::VEC_RESIZE) && args.len() == 3;
            if let ExprKind::Lit(Spanned { node: LitKind::Int(0, _), .. }) = args[1].kind;
            if let ExprKind::Lit(Spanned { node: LitKind::Int(..), .. }) = args[2].kind;
            then {
                let method_call_span = expr.span.with_lo(path_segment.ident.span.lo());
                span_lint_and_then(
                    cx,
                    VEC_RESIZE_TO_ZERO,
                    expr.span,
                    "emptying a vector with `resize`",
                    |db| {
                        db.help("the arguments may be inverted...");
                        db.span_suggestion(
                            method_call_span,
                            "...or you can empty the vector with",
                            "clear()".to_string(),
                            Applicability::MaybeIncorrect,
                        );
                    },
                );
            }
        }
    }
}
