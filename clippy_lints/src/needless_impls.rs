use clippy_utils::{
    diagnostics::span_lint_and_then, get_parent_node, is_res_lang_ctor, path_res, ty::implements_trait,
};
use rustc_errors::Applicability;
use rustc_hir::{def::Res, Expr, ExprKind, ImplItem, ImplItemKind, ItemKind, LangItem, Node, PatKind};
use rustc_hir_analysis::hir_ty_to_ty;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::EarlyBinder;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;
use std::borrow::Cow;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual implementations of both `PartialOrd` and `Ord` when only `Ord` is
    /// necessary.
    ///
    /// ### Why is this bad?
    /// If both `PartialOrd` and `Ord` are implemented, they must agree. This is commonly done by
    /// wrapping the result of `cmp` in `Some` for `partial_cmp`. Not doing this may silently
    /// introduce an error upon refactoring.
    ///
    /// ### Example
    /// ```rust,ignore
    /// #[derive(Eq, PartialEq)]
    /// struct A(u32);
    ///
    /// impl Ord for A {
    ///     fn cmp(&self, other: &Self) -> Ordering {
    ///         todo!();
    ///     }
    /// }
    ///
    /// impl PartialOrd for A {
    ///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    ///         todo!();
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// #[derive(Eq, PartialEq)]
    /// struct A(u32);
    ///
    /// impl Ord for A {
    ///     fn cmp(&self, other: &Self) -> Ordering {
    ///         todo!();
    ///     }
    /// }
    ///
    /// impl PartialOrd for A {
    ///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    ///         Some(self.cmp(other))
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.72.0"]
    pub NEEDLESS_PARTIAL_ORD_IMPL,
    correctness,
    "manual implementation of `PartialOrd` when `Ord` is already implemented"
}
declare_lint_pass!(NeedlessImpls => [NEEDLESS_PARTIAL_ORD_IMPL]);

impl LateLintPass<'_> for NeedlessImpls {
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

        if cx.tcx.is_diagnostic_item(sym::PartialOrd, trait_impl_def_id)
            && impl_item.ident.name == sym::partial_cmp
            && let Some(ord_def_id) = cx
                .tcx
                .diagnostic_items(trait_impl.def_id.krate)
                .name_to_id
                .get(&sym::Ord)
            && implements_trait(
                    cx,
                    hir_ty_to_ty(cx.tcx, imp.self_ty),
                    *ord_def_id,
                    trait_impl.substs,
                )
        {
            if block.stmts.is_empty()
                && let Some(expr) = block.expr
                && let ExprKind::Call(
                        Expr {
                            kind: ExprKind::Path(some_path),
                            hir_id: some_hir_id,
                            ..
                        },
                        [cmp_expr],
                    ) = expr.kind
                && is_res_lang_ctor(cx, cx.qpath_res(some_path, *some_hir_id), LangItem::OptionSome)
                && let ExprKind::MethodCall(cmp_path, _, [other_expr], ..) = cmp_expr.kind
                && cmp_path.ident.name == sym::cmp
                && let Res::Local(..) = path_res(cx, other_expr)
            {} else {
                span_lint_and_then(
                    cx,
                    NEEDLESS_PARTIAL_ORD_IMPL,
                    item.span,
                    "manual implementation of `PartialOrd` when `Ord` is already implemented",
                    |diag| {
                        let (help, app) = if let Some(other) = body.params.get(0)
                            && let PatKind::Binding(_, _, other_ident, ..) = other.pat.kind
                        {
                            (
                                Cow::Owned(format!("{{ Some(self.cmp({})) }}", other_ident.name)),
                                Applicability::Unspecified,
                            )
                        } else {
                            (Cow::Borrowed("{ Some(self.cmp(...)) }"), Applicability::HasPlaceholders)
                        };

                        diag.span_suggestion(
                            block.span,
                            "change this to",
                            help,
                            app,
                        );
                    }
                );
            }
        }
    }
}
