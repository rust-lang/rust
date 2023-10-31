use clippy_utils::diagnostics::span_lint;
use clippy_utils::macros::{is_panic, root_macro_call_first_node};
use rustc_hir::Expr;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `panic!`.
    ///
    /// ### Why is this bad?
    /// `panic!` will stop the execution of the executable
    ///
    /// ### Example
    /// ```no_run
    /// panic!("even with a good reason");
    /// ```
    #[clippy::version = "1.40.0"]
    pub PANIC,
    restriction,
    "usage of the `panic!` macro"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `unimplemented!`.
    ///
    /// ### Why is this bad?
    /// This macro should not be present in production code
    ///
    /// ### Example
    /// ```no_run
    /// unimplemented!();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub UNIMPLEMENTED,
    restriction,
    "`unimplemented!` should not be present in production code"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `todo!`.
    ///
    /// ### Why is this bad?
    /// This macro should not be present in production code
    ///
    /// ### Example
    /// ```no_run
    /// todo!();
    /// ```
    #[clippy::version = "1.40.0"]
    pub TODO,
    restriction,
    "`todo!` should not be present in production code"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `unreachable!`.
    ///
    /// ### Why is this bad?
    /// This macro can cause code to panic
    ///
    /// ### Example
    /// ```no_run
    /// unreachable!();
    /// ```
    #[clippy::version = "1.40.0"]
    pub UNREACHABLE,
    restriction,
    "usage of the `unreachable!` macro"
}

declare_lint_pass!(PanicUnimplemented => [UNIMPLEMENTED, UNREACHABLE, TODO, PANIC]);

impl<'tcx> LateLintPass<'tcx> for PanicUnimplemented {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let Some(macro_call) = root_macro_call_first_node(cx, expr) else {
            return;
        };
        if is_panic(cx, macro_call.def_id) {
            if cx.tcx.hir().is_inside_const_context(expr.hir_id) {
                return;
            }

            span_lint(
                cx,
                PANIC,
                macro_call.span,
                "`panic` should not be present in production code",
            );
            return;
        }
        match cx.tcx.item_name(macro_call.def_id).as_str() {
            "todo" => {
                span_lint(
                    cx,
                    TODO,
                    macro_call.span,
                    "`todo` should not be present in production code",
                );
            },
            "unimplemented" => {
                span_lint(
                    cx,
                    UNIMPLEMENTED,
                    macro_call.span,
                    "`unimplemented` should not be present in production code",
                );
            },
            "unreachable" => {
                span_lint(cx, UNREACHABLE, macro_call.span, "usage of the `unreachable!` macro");
            },
            _ => {},
        }
    }
}
