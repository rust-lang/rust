use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::{is_panic, root_macro_call_first_node};
use clippy_utils::method_chain_args;
use clippy_utils::ty::is_type_diagnostic_item;
use if_chain::if_chain;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for impls of `From<..>` that contain `panic!()` or `unwrap()`
    ///
    /// ### Why is this bad?
    /// `TryFrom` should be used if there's a possibility of failure.
    ///
    /// ### Example
    /// ```rust
    /// struct Foo(i32);
    ///
    /// impl From<String> for Foo {
    ///     fn from(s: String) -> Self {
    ///         Foo(s.parse().unwrap())
    ///     }
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// struct Foo(i32);
    ///
    /// impl TryFrom<String> for Foo {
    ///     type Error = ();
    ///     fn try_from(s: String) -> Result<Self, Self::Error> {
    ///         if let Ok(parsed) = s.parse() {
    ///             Ok(Foo(parsed))
    ///         } else {
    ///             Err(())
    ///         }
    ///     }
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub FALLIBLE_IMPL_FROM,
    nursery,
    "Warn on impls of `From<..>` that contain `panic!()` or `unwrap()`"
}

declare_lint_pass!(FallibleImplFrom => [FALLIBLE_IMPL_FROM]);

impl<'tcx> LateLintPass<'tcx> for FallibleImplFrom {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        // check for `impl From<???> for ..`
        if_chain! {
            if let hir::ItemKind::Impl(impl_) = &item.kind;
            if let Some(impl_trait_ref) = cx.tcx.impl_trait_ref(item.def_id);
            if cx.tcx.is_diagnostic_item(sym::From, impl_trait_ref.def_id);
            then {
                lint_impl_body(cx, item.span, impl_.items);
            }
        }
    }
}

fn lint_impl_body<'tcx>(cx: &LateContext<'tcx>, impl_span: Span, impl_items: &[hir::ImplItemRef]) {
    use rustc_hir::intravisit::{self, Visitor};
    use rustc_hir::{Expr, ImplItemKind};

    struct FindPanicUnwrap<'a, 'tcx> {
        lcx: &'a LateContext<'tcx>,
        typeck_results: &'tcx ty::TypeckResults<'tcx>,
        result: Vec<Span>,
    }

    impl<'a, 'tcx> Visitor<'tcx> for FindPanicUnwrap<'a, 'tcx> {
        fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
            if let Some(macro_call) = root_macro_call_first_node(self.lcx, expr) {
                if is_panic(self.lcx, macro_call.def_id) {
                    self.result.push(expr.span);
                }
            }

            // check for `unwrap`
            if let Some(arglists) = method_chain_args(expr, &["unwrap"]) {
                let receiver_ty = self.typeck_results.expr_ty(arglists[0].0).peel_refs();
                if is_type_diagnostic_item(self.lcx, receiver_ty, sym::Option)
                    || is_type_diagnostic_item(self.lcx, receiver_ty, sym::Result)
                {
                    self.result.push(expr.span);
                }
            }

            // and check sub-expressions
            intravisit::walk_expr(self, expr);
        }
    }

    for impl_item in impl_items {
        if_chain! {
            if impl_item.ident.name == sym::from;
            if let ImplItemKind::Fn(_, body_id) =
                cx.tcx.hir().impl_item(impl_item.id).kind;
            then {
                // check the body for `begin_panic` or `unwrap`
                let body = cx.tcx.hir().body(body_id);
                let mut fpu = FindPanicUnwrap {
                    lcx: cx,
                    typeck_results: cx.tcx.typeck(impl_item.id.def_id),
                    result: Vec::new(),
                };
                fpu.visit_expr(body.value);

                // if we've found one, lint
                if !fpu.result.is_empty() {
                    span_lint_and_then(
                        cx,
                        FALLIBLE_IMPL_FROM,
                        impl_span,
                        "consider implementing `TryFrom` instead",
                        move |diag| {
                            diag.help(
                                "`From` is intended for infallible conversions only. \
                                Use `TryFrom` if there's a possibility for the conversion to fail");
                            diag.span_note(fpu.result, "potential failure(s)");
                        });
                }
            }
        }
    }
}
