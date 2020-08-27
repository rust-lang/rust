use crate::utils::{is_expn_of, is_type_diagnostic_item, return_ty, span_lint_and_then};
use if_chain::if_chain;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `panic!`, `unimplemented!` or `unreachable!` in a function of type result/option.
    ///
    /// **Why is this bad?** For some codebases,
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// fn option_with_panic() -> Option<bool> // should emit lint
    /// {
    ///     panic!("error");
    /// }
    /// ```

    pub PANIC_IN_RESULT,
    restriction,
    "functions of type `Result<..>` / `Option`<...> that contain `panic!()` or `unreachable()` or `unimplemented()` "
}

declare_lint_pass!(PanicInResult => [PANIC_IN_RESULT]);

impl<'tcx> LateLintPass<'tcx> for PanicInResult {
    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &'tcx hir::ImplItem<'_>) {
        if_chain! {
            // first check if it's a method or function
            if let hir::ImplItemKind::Fn(ref _signature, _) = impl_item.kind;
            // checking if its return type is `result` or `option`
            if is_type_diagnostic_item(cx, return_ty(cx, impl_item.hir_id), sym!(result_type))
                || is_type_diagnostic_item(cx, return_ty(cx, impl_item.hir_id), sym!(option_type));
            then {
                lint_impl_body(cx, impl_item.span, impl_item);
            }
        }
    }
}

use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{Expr, ImplItemKind};

struct FindPanicUnimplementedUnreachable {
    result: Vec<Span>,
}

impl<'tcx> Visitor<'tcx> for FindPanicUnimplementedUnreachable {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if is_expn_of(expr.span, "unimplemented").is_some() {
            self.result.push(expr.span);
        } else if is_expn_of(expr.span, "unreachable").is_some() {
            self.result.push(expr.span);
        } else if is_expn_of(expr.span, "panic").is_some() {
            self.result.push(expr.span);
        }

        // and check sub-expressions
        intravisit::walk_expr(self, expr);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

fn lint_impl_body<'tcx>(cx: &LateContext<'tcx>, impl_span: Span, impl_item: &'tcx hir::ImplItem<'_>) {
    if_chain! {
        if let ImplItemKind::Fn(_, body_id) = impl_item.kind;
        then {
            let body = cx.tcx.hir().body(body_id);
            let mut fpu = FindPanicUnimplementedUnreachable {
                result: Vec::new(),
            };
            fpu.visit_expr(&body.value);

            // if we've found one, lint
            if  !fpu.result.is_empty()  {
                span_lint_and_then(
                    cx,
                    PANIC_IN_RESULT,
                    impl_span,
                    "used unimplemented, unreachable or panic in a function that returns result or option",
                    move |diag| {
                        diag.help(
                            "unimplemented, unreachable or panic should not be used in a function that returns result or option" );
                        diag.span_note(fpu.result, "will cause the application to crash.");
                    });
            }
        }
    }
}
