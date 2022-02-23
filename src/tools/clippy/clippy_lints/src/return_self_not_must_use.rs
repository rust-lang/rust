use clippy_utils::ty::is_must_use_ty;
use clippy_utils::{diagnostics::span_lint, nth_arg, return_ty};
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl, HirId, TraitItem, TraitItemKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// This lint warns when a method returning `Self` doesn't have the `#[must_use]` attribute.
    ///
    /// ### Why is this bad?
    /// It prevents to "forget" to use the newly created value.
    ///
    /// ### Limitations
    /// This lint is only applied on methods taking a `self` argument. It would be mostly noise
    /// if it was added on constructors for example.
    ///
    /// ### Example
    /// ```rust
    /// pub struct Bar;
    ///
    /// impl Bar {
    ///     // Bad
    ///     pub fn bar(&self) -> Self {
    ///         Self
    ///     }
    ///
    ///     // Good
    ///     #[must_use]
    ///     pub fn foo(&self) -> Self {
    ///         Self
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.59.0"]
    pub RETURN_SELF_NOT_MUST_USE,
    pedantic,
    "missing `#[must_use]` annotation on a method returning `Self`"
}

declare_lint_pass!(ReturnSelfNotMustUse => [RETURN_SELF_NOT_MUST_USE]);

fn check_method(cx: &LateContext<'tcx>, decl: &'tcx FnDecl<'tcx>, fn_def: LocalDefId, span: Span, hir_id: HirId) {
    if_chain! {
        // If it comes from an external macro, better ignore it.
        if !in_external_macro(cx.sess(), span);
        if decl.implicit_self.has_implicit_self();
        // We only show this warning for public exported methods.
        if cx.access_levels.is_exported(fn_def);
        // We don't want to emit this lint if the `#[must_use]` attribute is already there.
        if !cx.tcx.hir().attrs(hir_id).iter().any(|attr| attr.has_name(sym::must_use));
        if cx.tcx.visibility(fn_def.to_def_id()).is_public();
        let ret_ty = return_ty(cx, hir_id);
        let self_arg = nth_arg(cx, hir_id, 0);
        // If `Self` has the same type as the returned type, then we want to warn.
        //
        // For this check, we don't want to remove the reference on the returned type because if
        // there is one, we shouldn't emit a warning!
        if self_arg.peel_refs() == ret_ty;
        // If `Self` is already marked as `#[must_use]`, no need for the attribute here.
        if !is_must_use_ty(cx, ret_ty);

        then {
            span_lint(
                cx,
                RETURN_SELF_NOT_MUST_USE,
                span,
                "missing `#[must_use]` attribute on a method returning `Self`",
            );
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for ReturnSelfNotMustUse {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'tcx>,
        _: &'tcx Body<'tcx>,
        span: Span,
        hir_id: HirId,
    ) {
        if_chain! {
            // We are only interested in methods, not in functions or associated functions.
            if matches!(kind, FnKind::Method(_, _, _));
            if let Some(fn_def) = cx.tcx.hir().opt_local_def_id(hir_id);
            if let Some(impl_def) = cx.tcx.impl_of_method(fn_def.to_def_id());
            // We don't want this method to be te implementation of a trait because the
            // `#[must_use]` should be put on the trait definition directly.
            if cx.tcx.trait_id_of_impl(impl_def).is_none();

            then {
                check_method(cx, decl, fn_def, span, hir_id);
            }
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'tcx>) {
        if let TraitItemKind::Fn(ref sig, _) = item.kind {
            check_method(cx, sig.decl, item.def_id, item.span, item.hir_id());
        }
    }
}
