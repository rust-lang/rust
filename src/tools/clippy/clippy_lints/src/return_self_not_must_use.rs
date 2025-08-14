use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::is_must_use_ty;
use clippy_utils::{nth_arg, return_ty};
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl, OwnerId, TraitItem, TraitItemKind, find_attr};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// This lint warns when a method returning `Self` doesn't have the `#[must_use]` attribute.
    ///
    /// ### Why is this bad?
    /// Methods returning `Self` often create new values, having the `#[must_use]` attribute
    /// prevents users from "forgetting" to use the newly created value.
    ///
    /// The `#[must_use]` attribute can be added to the type itself to ensure that instances
    /// are never forgotten. Functions returning a type marked with `#[must_use]` will not be
    /// linted, as the usage is already enforced by the type attribute.
    ///
    /// ### Limitations
    /// This lint is only applied on methods taking a `self` argument. It would be mostly noise
    /// if it was added on constructors for example.
    ///
    /// ### Example
    /// ```no_run
    /// pub struct Bar;
    /// impl Bar {
    ///     // Missing attribute
    ///     pub fn bar(&self) -> Self {
    ///         Self
    ///     }
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # {
    /// // It's better to have the `#[must_use]` attribute on the method like this:
    /// pub struct Bar;
    /// impl Bar {
    ///     #[must_use]
    ///     pub fn bar(&self) -> Self {
    ///         Self
    ///     }
    /// }
    /// # }
    ///
    /// # {
    /// // Or on the type definition like this:
    /// #[must_use]
    /// pub struct Bar;
    /// impl Bar {
    ///     pub fn bar(&self) -> Self {
    ///         Self
    ///     }
    /// }
    /// # }
    /// ```
    #[clippy::version = "1.59.0"]
    pub RETURN_SELF_NOT_MUST_USE,
    pedantic,
    "missing `#[must_use]` annotation on a method returning `Self`"
}

declare_lint_pass!(ReturnSelfNotMustUse => [RETURN_SELF_NOT_MUST_USE]);

fn check_method(cx: &LateContext<'_>, decl: &FnDecl<'_>, fn_def: LocalDefId, span: Span, owner_id: OwnerId) {
    if !span.in_external_macro(cx.sess().source_map())
        // If it comes from an external macro, better ignore it.
        && decl.implicit_self.has_implicit_self()
        // We only show this warning for public exported methods.
        && cx.effective_visibilities.is_exported(fn_def)
        // We don't want to emit this lint if the `#[must_use]` attribute is already there.
        && !find_attr!(
            cx.tcx.hir_attrs(owner_id.into()),
            AttributeKind::MustUse { .. }
        )
        && cx.tcx.visibility(fn_def.to_def_id()).is_public()
        && let ret_ty = return_ty(cx, owner_id)
        && let self_arg = nth_arg(cx, owner_id, 0)
        // If `Self` has the same type as the returned type, then we want to warn.
        //
        // For this check, we don't want to remove the reference on the returned type because if
        // there is one, we shouldn't emit a warning!
        && self_arg.peel_refs() == ret_ty
        // If `Self` is already marked as `#[must_use]`, no need for the attribute here.
        && !is_must_use_ty(cx, ret_ty)
    {
        span_lint_and_help(
            cx,
            RETURN_SELF_NOT_MUST_USE,
            span,
            "missing `#[must_use]` attribute on a method returning `Self`",
            None,
            "consider adding the `#[must_use]` attribute to the method or directly to the `Self` type",
        );
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
        fn_def: LocalDefId,
    ) {
        if matches!(kind, FnKind::Method(_, _))
            // We are only interested in methods, not in functions or associated functions.
            && let Some(impl_def) = cx.tcx.impl_of_assoc(fn_def.to_def_id())
            // We don't want this method to be te implementation of a trait because the
            // `#[must_use]` should be put on the trait definition directly.
            && cx.tcx.trait_id_of_impl(impl_def).is_none()
        {
            let hir_id = cx.tcx.local_def_id_to_hir_id(fn_def);
            check_method(cx, decl, fn_def, span, hir_id.expect_owner());
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'tcx>) {
        if let TraitItemKind::Fn(ref sig, _) = item.kind {
            check_method(cx, sig.decl, item.owner_id.def_id, item.span, item.owner_id);
        }
    }
}
