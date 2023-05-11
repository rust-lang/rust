use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::macros::root_macro_call_first_node;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{is_in_cfg_test, is_in_test_function};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of the [`dbg!`](https://doc.rust-lang.org/std/macro.dbg.html) macro.
    ///
    /// ### Why is this bad?
    /// The `dbg!` macro is intended as a debugging tool. It should not be present in released
    /// software or committed to a version control system.
    ///
    /// ### Example
    /// ```rust,ignore
    /// dbg!(true)
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// true
    /// ```
    #[clippy::version = "1.34.0"]
    pub DBG_MACRO,
    restriction,
    "`dbg!` macro is intended as a debugging tool"
}

#[derive(Copy, Clone)]
pub struct DbgMacro {
    allow_dbg_in_tests: bool,
}

impl_lint_pass!(DbgMacro => [DBG_MACRO]);

impl DbgMacro {
    pub fn new(allow_dbg_in_tests: bool) -> Self {
        DbgMacro { allow_dbg_in_tests }
    }
}

impl LateLintPass<'_> for DbgMacro {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        let Some(macro_call) = root_macro_call_first_node(cx, expr) else { return };
        if cx.tcx.is_diagnostic_item(sym::dbg_macro, macro_call.def_id) {
            // allows `dbg!` in test code if allow-dbg-in-test is set to true in clippy.toml
            if self.allow_dbg_in_tests
                && (is_in_test_function(cx.tcx, expr.hir_id) || is_in_cfg_test(cx.tcx, expr.hir_id))
            {
                return;
            }
            let mut applicability = Applicability::MachineApplicable;
            let suggestion = match expr.peel_drop_temps().kind {
                // dbg!()
                ExprKind::Block(_, _) => String::new(),
                // dbg!(1)
                ExprKind::Match(val, ..) => {
                    snippet_with_applicability(cx, val.span.source_callsite(), "..", &mut applicability).to_string()
                },
                // dbg!(2, 3)
                ExprKind::Tup(
                    [
                        Expr {
                            kind: ExprKind::Match(first, ..),
                            ..
                        },
                        ..,
                        Expr {
                            kind: ExprKind::Match(last, ..),
                            ..
                        },
                    ],
                ) => {
                    let snippet = snippet_with_applicability(
                        cx,
                        first.span.source_callsite().to(last.span.source_callsite()),
                        "..",
                        &mut applicability,
                    );
                    format!("({snippet})")
                },
                _ => return,
            };

            span_lint_and_sugg(
                cx,
                DBG_MACRO,
                macro_call.span,
                "the `dbg!` macro is intended as a debugging tool",
                "remove the invocation before committing it to a version control system",
                suggestion,
                applicability,
            );
        }
    }
}
