use clippy_utils::{diagnostics::span_lint_and_sugg, get_parent_node, last_path_segment, ty::implements_trait};
use rustc_errors::Applicability;
use rustc_hir::{ExprKind, ImplItem, ImplItemKind, ItemKind, Node, UnOp};
use rustc_hir_analysis::hir_ty_to_ty;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::EarlyBinder;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual implementations of `Clone` when `Copy` is already implemented.
    ///
    /// ### Why is this bad?
    /// If both `Clone` and `Copy` are implemented, they must agree. This is done by dereferencing
    /// `self` in `Clone`'s implementation. Anything else is incorrect.
    ///
    /// ### Example
    /// ```rust,ignore
    /// #[derive(Eq, PartialEq)]
    /// struct A(u32);
    ///
    /// impl Clone for A {
    ///     fn clone(&self) -> Self {
    ///         Self(self.0)
    ///     }
    /// }
    ///
    /// impl Copy for A {}
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// #[derive(Eq, PartialEq)]
    /// struct A(u32);
    ///
    /// impl Clone for A {
    ///     fn clone(&self) -> Self {
    ///         *self
    ///     }
    /// }
    ///
    /// impl Copy for A {}
    /// ```
    #[clippy::version = "1.72.0"]
    pub INCORRECT_CLONE_IMPL_ON_COPY_TYPE,
    correctness,
    "manual implementation of `Clone` on a `Copy` type"
}
declare_lint_pass!(IncorrectImpls => [INCORRECT_CLONE_IMPL_ON_COPY_TYPE]);

impl LateLintPass<'_> for IncorrectImpls {
    #[expect(clippy::needless_return)]
    fn check_impl_item(&mut self, cx: &LateContext<'_>, impl_item: &ImplItem<'_>) {
        let node = get_parent_node(cx.tcx, impl_item.hir_id());
        let Some(Node::Item(item)) = node else {
            return;
        };
        let ItemKind::Impl(imp) = item.kind else {
            return;
        };
        let Some(trait_impl) = cx.tcx.impl_trait_ref(item.owner_id).map(EarlyBinder::skip_binder) else {
            return;
        };
        let trait_impl_def_id = trait_impl.def_id;
        if cx.tcx.is_automatically_derived(item.owner_id.to_def_id()) {
            return;
        }
        let ImplItemKind::Fn(_, impl_item_id) = cx.tcx.hir().impl_item(impl_item.impl_item_id()).kind else {
            return;
        };
        let body = cx.tcx.hir().body(impl_item_id);
        let ExprKind::Block(block, ..) = body.value.kind else {
            return;
        };
        // Above is duplicated from the `duplicate_manual_partial_ord_impl` branch.
        // Remove it while solving conflicts once that PR is merged.

        // Actual implementation; remove this comment once aforementioned PR is merged
        if cx.tcx.is_diagnostic_item(sym::Clone, trait_impl_def_id)
            && let Some(copy_def_id) = cx.tcx.get_diagnostic_item(sym::Copy)
            && implements_trait(
                    cx,
                    hir_ty_to_ty(cx.tcx, imp.self_ty),
                    copy_def_id,
                    trait_impl.substs,
                )
        {
            if impl_item.ident.name == sym::clone {
                if block.stmts.is_empty()
                    && let Some(expr) = block.expr
                    && let ExprKind::Unary(UnOp::Deref, inner) = expr.kind
                    && let ExprKind::Path(qpath) = inner.kind
                    && last_path_segment(&qpath).ident.name == symbol::kw::SelfLower
                {} else {
                    span_lint_and_sugg(
                        cx,
                        INCORRECT_CLONE_IMPL_ON_COPY_TYPE,
                        block.span,
                        "incorrect implementation of `clone` on a `Copy` type",
                        "change this to",
                        "{ *self }".to_owned(),
                        Applicability::MaybeIncorrect,
                    );

                    return;
                }
            }

            if impl_item.ident.name == sym::clone_from {
                span_lint_and_sugg(
                    cx,
                    INCORRECT_CLONE_IMPL_ON_COPY_TYPE,
                    impl_item.span,
                    "incorrect implementation of `clone_from` on a `Copy` type",
                    "remove this",
                    String::new(),
                    Applicability::MaybeIncorrect,
                );

                return;
            }
        }
    }
}
