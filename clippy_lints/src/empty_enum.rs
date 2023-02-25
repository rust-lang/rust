//! lint when there is an enum with no variants

use clippy_utils::diagnostics::span_lint_and_help;
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `enum`s with no variants.
    ///
    /// As of this writing, the `never_type` is still a
    /// nightly-only experimental API. Therefore, this lint is only triggered
    /// if the `never_type` is enabled.
    ///
    /// ### Why is this bad?
    /// If you want to introduce a type which
    /// can't be instantiated, you should use `!` (the primitive type "never"),
    /// or a wrapper around it, because `!` has more extensive
    /// compiler support (type inference, etc...) and wrappers
    /// around it are the conventional way to define an uninhabited type.
    /// For further information visit [never type documentation](https://doc.rust-lang.org/std/primitive.never.html)
    ///
    ///
    /// ### Example
    /// ```rust
    /// enum Test {}
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// #![feature(never_type)]
    ///
    /// struct Test(!);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub EMPTY_ENUM,
    pedantic,
    "enum with no variants"
}

declare_lint_pass!(EmptyEnum => [EMPTY_ENUM]);

impl<'tcx> LateLintPass<'tcx> for EmptyEnum {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        // Only suggest the `never_type` if the feature is enabled
        if !cx.tcx.features().never_type {
            return;
        }

        if let ItemKind::Enum(..) = item.kind {
            let ty = cx.tcx.type_of(item.owner_id).subst_identity();
            let adt = ty.ty_adt_def().expect("already checked whether this is an enum");
            if adt.variants().is_empty() {
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
