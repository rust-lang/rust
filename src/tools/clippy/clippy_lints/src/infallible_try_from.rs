use clippy_utils::diagnostics::span_lint;
use clippy_utils::sym;
use rustc_errors::MultiSpan;
use rustc_hir::{AssocItemKind, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Finds manual impls of `TryFrom` with infallible error types.
    ///
    /// ### Why is this bad?
    ///
    /// Infalliable conversions should be implemented via `From` with the blanket conversion.
    ///
    /// ### Example
    /// ```no_run
    /// use std::convert::Infallible;
    /// struct MyStruct(i16);
    /// impl TryFrom<i16> for MyStruct {
    ///     type Error = Infallible;
    ///     fn try_from(other: i16) -> Result<Self, Infallible> {
    ///         Ok(Self(other.into()))
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// struct MyStruct(i16);
    /// impl From<i16> for MyStruct {
    ///     fn from(other: i16) -> Self {
    ///         Self(other)
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.88.0"]
    pub INFALLIBLE_TRY_FROM,
    suspicious,
    "TryFrom with infallible Error type"
}
declare_lint_pass!(InfallibleTryFrom => [INFALLIBLE_TRY_FROM]);

impl<'tcx> LateLintPass<'tcx> for InfallibleTryFrom {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        let ItemKind::Impl(imp) = item.kind else { return };
        let Some(r#trait) = imp.of_trait else { return };
        let Some(trait_def_id) = r#trait.trait_def_id() else {
            return;
        };
        if !cx.tcx.is_diagnostic_item(sym::TryFrom, trait_def_id) {
            return;
        }
        for ii in imp.items {
            if ii.kind == AssocItemKind::Type {
                let ii = cx.tcx.hir_impl_item(ii.id);
                if ii.ident.name != sym::Error {
                    continue;
                }
                let ii_ty = ii.expect_type();
                let ii_ty_span = ii_ty.span;
                let ii_ty = clippy_utils::ty::ty_from_hir_ty(cx, ii_ty);
                if !ii_ty.is_inhabited_from(cx.tcx, ii.owner_id.to_def_id(), cx.typing_env()) {
                    let mut span = MultiSpan::from_span(cx.tcx.def_span(item.owner_id.to_def_id()));
                    span.push_span_label(ii_ty_span, "infallible error type");
                    span_lint(
                        cx,
                        INFALLIBLE_TRY_FROM,
                        span,
                        "infallible TryFrom impl; consider implementing From, instead",
                    );
                }
            }
        }
    }
}
