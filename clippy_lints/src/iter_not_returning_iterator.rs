use clippy_utils::{diagnostics::span_lint, get_parent_node, ty::implements_trait};
use rustc_hir::{def_id::LocalDefId, FnSig, ImplItem, ImplItemKind, Item, ItemKind, Node, TraitItem, TraitItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Detects methods named `iter` or `iter_mut` that do not have a return type that implements `Iterator`.
    ///
    /// ### Why is this bad?
    /// Methods named `iter` or `iter_mut` conventionally return an `Iterator`.
    ///
    /// ### Example
    /// ```rust
    /// // `String` does not implement `Iterator`
    /// struct Data {}
    /// impl Data {
    ///     fn iter(&self) -> String {
    ///         todo!()
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::str::Chars;
    /// struct Data {}
    /// impl Data {
    ///    fn iter(&self) -> Chars<'static> {
    ///        todo!()
    ///    }
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
        let name = item.ident.name.as_str();
        if matches!(name, "iter" | "iter_mut") {
            if let TraitItemKind::Fn(fn_sig, _) = &item.kind {
                check_sig(cx, name, fn_sig, item.owner_id.def_id);
            }
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'tcx>) {
        let name = item.ident.name.as_str();
        if matches!(name, "iter" | "iter_mut")
            && !matches!(
                get_parent_node(cx.tcx, item.hir_id()),
                Some(Node::Item(Item { kind: ItemKind::Impl(i), .. })) if i.of_trait.is_some()
            )
        {
            if let ImplItemKind::Fn(fn_sig, _) = &item.kind {
                check_sig(cx, name, fn_sig, item.owner_id.def_id);
            }
        }
    }
}

fn check_sig(cx: &LateContext<'_>, name: &str, sig: &FnSig<'_>, fn_id: LocalDefId) {
    if sig.decl.implicit_self.has_implicit_self() {
        let ret_ty = cx.tcx.erase_late_bound_regions(cx.tcx.bound_fn_sig(fn_id.into()).subst_identity().output());
        let ret_ty = cx
            .tcx
            .try_normalize_erasing_regions(cx.param_env, ret_ty)
            .unwrap_or(ret_ty);
        if cx
            .tcx
            .get_diagnostic_item(sym::Iterator)
            .map_or(false, |iter_id| !implements_trait(cx, ret_ty, iter_id, &[]))
        {
            span_lint(
                cx,
                ITER_NOT_RETURNING_ITERATOR,
                sig.span,
                &format!("this method is named `{name}` but its return type does not implement `Iterator`"),
            );
        }
    }
}
