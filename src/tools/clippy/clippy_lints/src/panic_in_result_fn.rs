use crate::utils::{find_macro_calls, is_type_diagnostic_item, return_ty, span_lint_and_then};
use rustc_hir as hir;
use rustc_hir::intravisit::FnKind;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `panic!`, `unimplemented!`, `todo!`, `unreachable!` or assertions in a function of type result.
    ///
    /// **Why is this bad?** For some codebases, it is desirable for functions of type result to return an error instead of crashing. Hence panicking macros should be avoided.
    ///
    /// **Known problems:** Functions called from a function returning a `Result` may invoke a panicking macro. This is not checked.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// fn result_with_panic() -> Result<bool, String>
    /// {
    ///     panic!("error");
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// fn result_without_panic() -> Result<bool, String> {
    ///     Err(String::from("error"))
    /// }
    /// ```
    pub PANIC_IN_RESULT_FN,
    restriction,
    "functions of type `Result<..>` that contain `panic!()`, `todo!()`, `unreachable()`, `unimplemented()` or assertion"
}

declare_lint_pass!(PanicInResultFn  => [PANIC_IN_RESULT_FN]);

impl<'tcx> LateLintPass<'tcx> for PanicInResultFn {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_kind: FnKind<'tcx>,
        _: &'tcx hir::FnDecl<'tcx>,
        body: &'tcx hir::Body<'tcx>,
        span: Span,
        hir_id: hir::HirId,
    ) {
        if !matches!(fn_kind, FnKind::Closure(_))
            && is_type_diagnostic_item(cx, return_ty(cx, hir_id), sym::result_type)
        {
            lint_impl_body(cx, span, body);
        }
    }
}

fn lint_impl_body<'tcx>(cx: &LateContext<'tcx>, impl_span: Span, body: &'tcx hir::Body<'tcx>) {
    let panics = find_macro_calls(
        &[
            "unimplemented",
            "unreachable",
            "panic",
            "todo",
            "assert",
            "assert_eq",
            "assert_ne",
            "debug_assert",
            "debug_assert_eq",
            "debug_assert_ne",
        ],
        body,
    );
    if !panics.is_empty() {
        span_lint_and_then(
            cx,
            PANIC_IN_RESULT_FN,
            impl_span,
            "used `unimplemented!()`, `unreachable!()`, `todo!()`, `panic!()` or assertion in a function that returns `Result`",
            move |diag| {
                diag.help(
                    "`unimplemented!()`, `unreachable!()`, `todo!()`, `panic!()` or assertions should not be used in a function that returns `Result` as `Result` is expected to return an error instead of crashing",
                );
                diag.span_note(panics, "return Err() instead of panicking");
            },
        );
    }
}
