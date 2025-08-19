use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::{is_panic, root_macro_call_first_node};
use clippy_utils::method_chain_args;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;
use rustc_span::{Span, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for impls of `From<..>` that contain `panic!()` or `unwrap()`
    ///
    /// ### Why is this bad?
    /// `TryFrom` should be used if there's a possibility of failure.
    ///
    /// ### Example
    /// ```no_run
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
    /// ```no_run
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
        if let hir::ItemKind::Impl(_) = &item.kind
            && let Some(impl_trait_ref) = cx.tcx.impl_trait_ref(item.owner_id)
            && cx
                .tcx
                .is_diagnostic_item(sym::From, impl_trait_ref.skip_binder().def_id)
        {
            lint_impl_body(cx, item.owner_id, item.span);
        }
    }
}

fn lint_impl_body(cx: &LateContext<'_>, item_def_id: hir::OwnerId, impl_span: Span) {
    use rustc_hir::Expr;
    use rustc_hir::intravisit::{self, Visitor};

    struct FindPanicUnwrap<'a, 'tcx> {
        lcx: &'a LateContext<'tcx>,
        typeck_results: &'tcx ty::TypeckResults<'tcx>,
        result: Vec<Span>,
    }

    impl<'tcx> Visitor<'tcx> for FindPanicUnwrap<'_, 'tcx> {
        fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
            if let Some(macro_call) = root_macro_call_first_node(self.lcx, expr)
                && is_panic(self.lcx, macro_call.def_id)
            {
                self.result.push(expr.span);
            }

            // check for `unwrap`
            if let Some(arglists) = method_chain_args(expr, &[sym::unwrap]) {
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

    for impl_item in cx
        .tcx
        .associated_items(item_def_id)
        .filter_by_name_unhygienic_and_kind(sym::from, ty::AssocTag::Fn)
    {
        let impl_item_def_id = impl_item.def_id.expect_local();

        // check the body for `begin_panic` or `unwrap`
        let body = cx.tcx.hir_body_owned_by(impl_item_def_id);
        let mut fpu = FindPanicUnwrap {
            lcx: cx,
            typeck_results: cx.tcx.typeck(impl_item_def_id),
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
                        Use `TryFrom` if there's a possibility for the conversion to fail",
                    );
                    diag.span_note(fpu.result, "potential failure(s)");
                },
            );
        }
    }
}
