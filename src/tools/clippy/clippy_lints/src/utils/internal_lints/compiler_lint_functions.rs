use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::match_type;
use clippy_utils::{is_lint_allowed, paths};
use if_chain::if_chain;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `cx.span_lint*` and suggests to use the `utils::*`
    /// variant of the function.
    ///
    /// ### Why is this bad?
    /// The `utils::*` variants also add a link to the Clippy documentation to the
    /// warning/error messages.
    ///
    /// ### Example
    /// ```rust,ignore
    /// cx.span_lint(LINT_NAME, "message");
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// utils::span_lint(cx, LINT_NAME, "message");
    /// ```
    pub COMPILER_LINT_FUNCTIONS,
    internal,
    "usage of the lint functions of the compiler instead of the utils::* variant"
}

impl_lint_pass!(CompilerLintFunctions => [COMPILER_LINT_FUNCTIONS]);

#[derive(Clone, Default)]
pub struct CompilerLintFunctions {
    map: FxHashMap<&'static str, &'static str>,
}

impl CompilerLintFunctions {
    #[must_use]
    pub fn new() -> Self {
        let mut map = FxHashMap::default();
        map.insert("span_lint", "utils::span_lint");
        map.insert("struct_span_lint", "utils::span_lint");
        map.insert("lint", "utils::span_lint");
        map.insert("span_lint_note", "utils::span_lint_and_note");
        map.insert("span_lint_help", "utils::span_lint_and_help");
        Self { map }
    }
}

impl<'tcx> LateLintPass<'tcx> for CompilerLintFunctions {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if is_lint_allowed(cx, COMPILER_LINT_FUNCTIONS, expr.hir_id) {
            return;
        }

        if_chain! {
            if let ExprKind::MethodCall(path, self_arg, _, _) = &expr.kind;
            let fn_name = path.ident;
            if let Some(sugg) = self.map.get(fn_name.as_str());
            let ty = cx.typeck_results().expr_ty(self_arg).peel_refs();
            if match_type(cx, ty, &paths::EARLY_CONTEXT) || match_type(cx, ty, &paths::LATE_CONTEXT);
            then {
                span_lint_and_help(
                    cx,
                    COMPILER_LINT_FUNCTIONS,
                    path.ident.span,
                    "usage of a compiler lint function",
                    None,
                    &format!("please use the Clippy variant of this function: `{sugg}`"),
                );
            }
        }
    }
}
