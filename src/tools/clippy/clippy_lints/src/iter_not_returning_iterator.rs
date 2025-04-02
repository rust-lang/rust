use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::implements_trait;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{FnSig, ImplItem, ImplItemKind, Item, ItemKind, Node, TraitItem, TraitItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::{Symbol, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Detects methods named `iter` or `iter_mut` that do not have a return type that implements `Iterator`.
    ///
    /// ### Why is this bad?
    /// Methods named `iter` or `iter_mut` conventionally return an `Iterator`.
    ///
    /// ### Example
    /// ```no_run
    /// // `String` does not implement `Iterator`
    /// struct Data {}
    /// impl Data {
    ///     fn iter(&self) -> String {
    ///         todo!()
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::str::Chars;
    /// struct Data {}
    /// impl Data {
    ///     fn iter(&self) -> Chars<'static> {
    ///         todo!()
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.57.0"]
    pub ITER_NOT_RETURNING_ITERATOR,
    pedantic,
    "methods named `iter` or `iter_mut` that do not return an `Iterator`"
}

declare_lint_pass!(IterNotReturningIterator => [ITER_NOT_RETURNING_ITERATOR]);

impl<'tcx> LateLintPass<'tcx> for IterNotReturningIterator {
    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'_>) {
        if let TraitItemKind::Fn(fn_sig, _) = &item.kind
            && matches!(item.ident.name, sym::iter | sym::iter_mut)
        {
            check_sig(cx, item.ident.name, fn_sig, item.owner_id.def_id);
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'tcx>) {
        if let ImplItemKind::Fn(fn_sig, _) = &item.kind
            && matches!(item.ident.name, sym::iter | sym::iter_mut)
            && !matches!(
                cx.tcx.parent_hir_node(item.hir_id()),
                Node::Item(Item { kind: ItemKind::Impl(i), .. }) if i.of_trait.is_some()
            )
        {
            check_sig(cx, item.ident.name, fn_sig, item.owner_id.def_id);
        }
    }
}

fn check_sig(cx: &LateContext<'_>, name: Symbol, sig: &FnSig<'_>, fn_id: LocalDefId) {
    if sig.decl.implicit_self.has_implicit_self() {
        let ret_ty = cx
            .tcx
            .instantiate_bound_regions_with_erased(cx.tcx.fn_sig(fn_id).instantiate_identity().output());
        let ret_ty = cx
            .tcx
            .try_normalize_erasing_regions(cx.typing_env(), ret_ty)
            .unwrap_or(ret_ty);
        if cx
            .tcx
            .get_diagnostic_item(sym::Iterator)
            .is_some_and(|iter_id| !implements_trait(cx, ret_ty, iter_id, &[]))
        {
            span_lint(
                cx,
                ITER_NOT_RETURNING_ITERATOR,
                sig.span,
                format!("this method is named `{name}` but its return type does not implement `Iterator`"),
            );
        }
    }
}
