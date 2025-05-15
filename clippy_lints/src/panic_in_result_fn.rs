use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::{is_panic, root_macro_call_first_node};
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::visitors::{Descend, for_each_expr};
use clippy_utils::{is_inside_always_const_context, return_ty};
use core::ops::ControlFlow;
use rustc_hir as hir;
use rustc_hir::intravisit::FnKind;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `panic!` or assertions in a function whose return type is `Result`.
    ///
    /// ### Why restrict this?
    /// For some codebases, it is desirable for functions of type result to return an error instead of crashing. Hence panicking macros should be avoided.
    ///
    /// ### Known problems
    /// Functions called from a function returning a `Result` may invoke a panicking macro. This is not checked.
    ///
    /// ### Example
    /// ```no_run
    /// fn result_with_panic() -> Result<bool, String>
    /// {
    ///     panic!("error");
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn result_without_panic() -> Result<bool, String> {
    ///     Err(String::from("error"))
    /// }
    /// ```
    #[clippy::version = "1.48.0"]
    pub PANIC_IN_RESULT_FN,
    restriction,
    "functions of type `Result<..>` that contain `panic!()` or assertion"
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
        def_id: LocalDefId,
    ) {
        if matches!(fn_kind, FnKind::Closure) {
            return;
        }
        let owner = cx.tcx.local_def_id_to_hir_id(def_id).expect_owner();
        if is_type_diagnostic_item(cx, return_ty(cx, owner), sym::Result) {
            lint_impl_body(cx, span, body);
        }
    }
}

fn lint_impl_body<'tcx>(cx: &LateContext<'tcx>, impl_span: Span, body: &'tcx hir::Body<'tcx>) {
    let mut panics = Vec::new();
    let _: Option<!> = for_each_expr(cx, body.value, |e| {
        let Some(macro_call) = root_macro_call_first_node(cx, e) else {
            return ControlFlow::Continue(Descend::Yes);
        };
        if !is_inside_always_const_context(cx.tcx, e.hir_id)
            && (is_panic(cx, macro_call.def_id)
                || matches!(
                    cx.tcx.get_diagnostic_name(macro_call.def_id),
                    Some(sym::assert_macro | sym::assert_eq_macro | sym::assert_ne_macro)
                ))
        {
            panics.push(macro_call.span);
            ControlFlow::Continue(Descend::No)
        } else {
            ControlFlow::Continue(Descend::Yes)
        }
    });
    if !panics.is_empty() {
        span_lint_and_then(
            cx,
            PANIC_IN_RESULT_FN,
            impl_span,
            "used `panic!()` or assertion in a function that returns `Result`",
            move |diag| {
                diag.help(
                    "`panic!()` or assertions should not be used in a function that returns `Result` as `Result` is expected to return an error instead of crashing",
                );
                diag.span_note(panics, "return Err() instead of panicking");
            },
        );
    }
}
