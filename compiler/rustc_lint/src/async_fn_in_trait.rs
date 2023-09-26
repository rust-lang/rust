use crate::lints::AsyncFnInTraitDiag;
use crate::LateContext;
use crate::LateLintPass;
use rustc_hir as hir;
use rustc_trait_selection::traits::error_reporting::suggestions::suggest_desugaring_async_fn_to_impl_future_in_trait;

declare_lint! {
    /// TODO
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn foo<T: Drop>() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// TODO
    pub ASYNC_FN_IN_TRAIT,
    Warn,
    "TODO"
}

declare_lint_pass!(
    // TODO:
    AsyncFnInTrait => [ASYNC_FN_IN_TRAIT]
);

impl<'tcx> LateLintPass<'tcx> for AsyncFnInTrait {
    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'tcx>) {
        if let hir::TraitItemKind::Fn(sig, body) = item.kind
            && let hir::IsAsync::Async(async_span) = sig.header.asyncness
        {
            if cx.tcx.features().return_type_notation {
                return;
            }

            let hir::FnRetTy::Return(hir::Ty { kind: hir::TyKind::OpaqueDef(def, ..), .. }) =
                sig.decl.output
            else {
                // This should never happen, but let's not ICE.
                return;
            };
            let sugg = suggest_desugaring_async_fn_to_impl_future_in_trait(
                cx.tcx,
                sig,
                body,
                def.owner_id.def_id,
                " + Send",
            );
            cx.tcx.emit_spanned_lint(ASYNC_FN_IN_TRAIT, item.hir_id(), async_span, AsyncFnInTraitDiag {
                sugg
            });
        }
    }
}
