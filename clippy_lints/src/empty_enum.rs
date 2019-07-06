//! lint when there is an enum with no variants

use crate::utils::span_lint_and_then;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for `enum`s with no variants.
    ///
    /// **Why is this bad?** Enum's with no variants should be replaced with `!`,
    /// the uninhabited type,
    /// or a wrapper around it.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// enum Test {}
    /// ```
    pub EMPTY_ENUM,
    pedantic,
    "enum with no variants"
}

declare_lint_pass!(EmptyEnum => [EMPTY_ENUM]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for EmptyEnum {
    fn check_item(&mut self, cx: &LateContext<'_, '_>, item: &Item) {
        let did = cx.tcx.hir().local_def_id(item.hir_id);
        if let ItemKind::Enum(..) = item.node {
            let ty = cx.tcx.type_of(did);
            let adt = ty.ty_adt_def().expect("already checked whether this is an enum");
            if adt.variants.is_empty() {
                span_lint_and_then(cx, EMPTY_ENUM, item.span, "enum with no variants", |db| {
                    db.span_help(
                        item.span,
                        "consider using the uninhabited type `!` or a wrapper around it",
                    );
                });
            }
        }
    }
}
