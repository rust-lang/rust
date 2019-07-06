use crate::utils::{is_copy, match_path, paths, span_note_and_lint};
use rustc::hir::{Item, ItemKind};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for types that implement `Copy` as well as
    /// `Iterator`.
    ///
    /// **Why is this bad?** Implicit copies can be confusing when working with
    /// iterator combinators.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
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
    pub COPY_ITERATOR,
    pedantic,
    "implementing `Iterator` on a `Copy` type"
}

declare_lint_pass!(CopyIterator => [COPY_ITERATOR]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for CopyIterator {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if let ItemKind::Impl(_, _, _, _, Some(ref trait_ref), _, _) = item.node {
            let ty = cx.tcx.type_of(cx.tcx.hir().local_def_id(item.hir_id));

            if is_copy(cx, ty) && match_path(&trait_ref.path, &paths::ITERATOR) {
                span_note_and_lint(
                    cx,
                    COPY_ITERATOR,
                    item.span,
                    "you are implementing `Iterator` on a `Copy` type",
                    item.span,
                    "consider implementing `IntoIterator` instead",
                );
            }
        }
    }
}
