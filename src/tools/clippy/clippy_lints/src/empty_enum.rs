//! lint when there is an enum with no variants

use crate::utils::span_lint_and_help;
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for `enum`s with no variants.
    ///
    /// **Why is this bad?** If you want to introduce a type which
    /// can't be instantiated, you should use `!` (the never type),
    /// or a wrapper around it, because `!` has more extensive
    /// compiler support (type inference, etc...) and wrappers
    /// around it are the conventional way to define an uninhabited type.
    /// For further information visit [never type documentation](https://doc.rust-lang.org/std/primitive.never.html)
    ///
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// Bad:
    /// ```rust
    /// enum Test {}
    /// ```
    ///
    /// Good:
    /// ```rust
    /// #![feature(never_type)]
    ///
    /// struct Test(!);
    /// ```
    pub EMPTY_ENUM,
    pedantic,
    "enum with no variants"
}

declare_lint_pass!(EmptyEnum => [EMPTY_ENUM]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for EmptyEnum {
    fn check_item(&mut self, cx: &LateContext<'_, '_>, item: &Item<'_>) {
        let did = cx.tcx.hir().local_def_id(item.hir_id);
        if let ItemKind::Enum(..) = item.kind {
            let ty = cx.tcx.type_of(did);
            let adt = ty.ty_adt_def().expect("already checked whether this is an enum");
            if adt.variants.is_empty() {
                span_lint_and_help(
                    cx,
                    EMPTY_ENUM,
                    item.span,
                    "enum with no variants",
                    None,
                    "consider using the uninhabited type `!` (never type) or a wrapper \
                    around it to introduce a type which can't be instantiated",
                );
            }
        }
    }
}
