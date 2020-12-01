use crate::utils::{is_expn_of, is_type_diagnostic_item, return_ty, span_lint_and_then};
use rustc_hir as hir;
use rustc_hir::intravisit::{self, FnKind, NestedVisitorMap, Visitor};
use rustc_hir::Expr;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `panic!`, `unimplemented!`, `todo!` or `unreachable!` in a function of type result.
    ///
    /// **Why is this bad?** For some codebases, it is desirable for functions of type result to return an error instead of crashing. Hence unimplemented, panic and unreachable should be avoided.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// fn result_with_panic() -> Result<bool, String>
    /// {
    ///     panic!("error");
    /// }
    /// ```
    pub PANIC_IN_RESULT_FN,
    restriction,
    "functions of type `Result<..>` that contain `panic!()`, `todo!()` or `unreachable()` or `unimplemented()` "
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

struct FindPanicUnimplementedUnreachable {
    result: Vec<Span>,
}

impl<'tcx> Visitor<'tcx> for FindPanicUnimplementedUnreachable {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if ["unimplemented", "unreachable", "panic", "todo"]
            .iter()
            .any(|fun| is_expn_of(expr.span, fun).is_some())
        {
            self.result.push(expr.span);
        }
        // and check sub-expressions
        intravisit::walk_expr(self, expr);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

fn lint_impl_body<'tcx>(cx: &LateContext<'tcx>, impl_span: Span, body: &'tcx hir::Body<'tcx>) {
    let mut panics = FindPanicUnimplementedUnreachable { result: Vec::new() };
    panics.visit_expr(&body.value);
    if !panics.result.is_empty() {
        span_lint_and_then(
            cx,
            PANIC_IN_RESULT_FN,
            impl_span,
            "used `unimplemented!()`, `unreachable!()`, `todo!()` or `panic!()` in a function that returns `Result`",
            move |diag| {
                diag.help(
                    "`unimplemented!()`, `unreachable!()`, `todo!()` or `panic!()` should not be used in a function that returns `Result` as `Result` is expected to return an error instead of crashing",
                );
                diag.span_note(panics.result, "return Err() instead of panicking");
            },
        );
    }
}
