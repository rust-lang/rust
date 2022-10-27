use clippy_utils::diagnostics::span_lint_and_note;
use clippy_utils::ty::is_copy;
use rustc_hir::{Impl, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

use if_chain::if_chain;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for types that implement `Copy` as well as
    /// `Iterator`.
    ///
    /// ### Why is this bad?
    /// Implicit copies can be confusing when working with
    /// iterator combinators.
    ///
    /// ### Example
    /// ```rust,ignore
    /// #[derive(Copy, Clone)]
    /// struct Countdown(u8);
    ///
    /// impl Iterator for Countdown {
    ///     // ...
    /// }
    ///
    /// let a: Vec<_> = my_iterator.take(1).collect();
    /// let b: Vec<_> = my_iterator.collect();
    /// ```
    #[clippy::version = "1.30.0"]
    pub COPY_ITERATOR,
    pedantic,
    "implementing `Iterator` on a `Copy` type"
}

declare_lint_pass!(CopyIterator => [COPY_ITERATOR]);

impl<'tcx> LateLintPass<'tcx> for CopyIterator {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if_chain! {
            if let ItemKind::Impl(Impl {
                of_trait: Some(ref trait_ref),
                ..
            }) = item.kind;
            let ty = cx.tcx.type_of(item.owner_id);
            if is_copy(cx, ty);
            if let Some(trait_id) = trait_ref.trait_def_id();
            if cx.tcx.is_diagnostic_item(sym::Iterator, trait_id);
            then {
                span_lint_and_note(
                    cx,
                    COPY_ITERATOR,
                    item.span,
                    "you are implementing `Iterator` on a `Copy` type",
                    None,
                    "consider implementing `IntoIterator` instead",
                );
            }
        }
    }
}
