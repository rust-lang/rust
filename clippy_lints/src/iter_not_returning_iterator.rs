use clippy_utils::{diagnostics::span_lint, return_ty, ty::implements_trait};
use rustc_hir::{ImplItem, ImplItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::kw;
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
    pub ITER_NOT_RETURNING_ITERATOR,
    pedantic,
    "methods named `iter` or `iter_mut` that do not return an `Iterator`"
}

declare_lint_pass!(IterNotReturningIterator => [ITER_NOT_RETURNING_ITERATOR]);

impl LateLintPass<'_> for IterNotReturningIterator {
    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &'tcx ImplItem<'tcx>) {
        let name: &str = &impl_item.ident.name.as_str();
        if_chain! {
            if let ImplItemKind::Fn(fn_sig, _) = &impl_item.kind;
            let ret_ty = return_ty(cx, impl_item.hir_id());
            if matches!(name, "iter" | "iter_mut");
            if let [param] = cx.tcx.fn_arg_names(impl_item.def_id);
            if param.name == kw::SelfLower;
            if let Some(iter_trait_id) = cx.tcx.get_diagnostic_item(sym::Iterator);
            if !implements_trait(cx, ret_ty, iter_trait_id, &[]);

            then {
                span_lint(
                    cx,
                    ITER_NOT_RETURNING_ITERATOR,
                    fn_sig.span,
                    &format!("this method is named `{}` but its return type does not implement `Iterator`", name),
                );
            }
        }
    }
}
