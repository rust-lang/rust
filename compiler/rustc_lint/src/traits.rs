use rustc_hir::{self as hir, AmbigArg, LangItem};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::sym;

use crate::lints::{DropGlue, DropTraitConstraintsDiag};
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `drop_bounds` lint checks for generics with `std::ops::Drop` as
    /// bounds.
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
    /// A generic trait bound of the form `T: Drop` is most likely misleading
    /// and not what the programmer intended (they probably should have used
    /// `std::mem::needs_drop` instead).
    ///
    /// `Drop` bounds do not actually indicate whether a type can be trivially
    /// dropped or not, because a composite type containing `Drop` types does
    /// not necessarily implement `Drop` itself. Naïvely, one might be tempted
    /// to write an implementation that assumes that a type can be trivially
    /// dropped while also supplying a specialization for `T: Drop` that
    /// actually calls the destructor. However, this breaks down e.g. when `T`
    /// is `String`, which does not implement `Drop` itself but contains a
    /// `Vec`, which does implement `Drop`, so assuming `T` can be trivially
    /// dropped would lead to a memory leak here.
    ///
    /// Furthermore, the `Drop` trait only contains one method, `Drop::drop`,
    /// which may not be called explicitly in user code (`E0040`), so there is
    /// really no use case for using `Drop` in trait bounds, save perhaps for
    /// some obscure corner cases, which can use `#[allow(drop_bounds)]`.
    pub DROP_BOUNDS,
    Warn,
    "bounds of the form `T: Drop` are most likely incorrect"
}

declare_lint! {
    /// The `dyn_drop` lint checks for trait objects with `std::ops::Drop`.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn foo(_x: Box<dyn Drop>) {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// A trait object bound of the form `dyn Drop` is most likely misleading
    /// and not what the programmer intended.
    ///
    /// `Drop` bounds do not actually indicate whether a type can be trivially
    /// dropped or not, because a composite type containing `Drop` types does
    /// not necessarily implement `Drop` itself. Naïvely, one might be tempted
    /// to write a deferred drop system, to pull cleaning up memory out of a
    /// latency-sensitive code path, using `dyn Drop` trait objects. However,
    /// this breaks down e.g. when `T` is `String`, which does not implement
    /// `Drop`, but should probably be accepted.
    ///
    /// To write a trait object bound that accepts anything, use a placeholder
    /// trait with a blanket implementation.
    ///
    /// ```rust
    /// trait Placeholder {}
    /// impl<T> Placeholder for T {}
    /// fn foo(_x: Box<dyn Placeholder>) {}
    /// ```
    pub DYN_DROP,
    Warn,
    "trait objects of the form `dyn Drop` are useless"
}

declare_lint_pass!(
    /// Lint for bounds of the form `T: Drop`, which usually
    /// indicate an attempt to emulate `std::mem::needs_drop`.
    DropTraitConstraints => [DROP_BOUNDS, DYN_DROP]
);

impl<'tcx> LateLintPass<'tcx> for DropTraitConstraints {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        use rustc_middle::ty::ClauseKind;

        let predicates = cx.tcx.explicit_predicates_of(item.owner_id);
        for &(predicate, span) in predicates.predicates {
            let ClauseKind::Trait(trait_predicate) = predicate.kind().skip_binder() else {
                continue;
            };
            let def_id = trait_predicate.trait_ref.def_id;
            if cx.tcx.is_lang_item(def_id, LangItem::Drop) {
                // Explicitly allow `impl Drop`, a drop-guards-as-unnameable-type pattern.
                if trait_predicate.trait_ref.self_ty().is_impl_trait() {
                    continue;
                }
                let Some(def_id) = cx.tcx.get_diagnostic_item(sym::needs_drop) else { return };
                cx.emit_span_lint(
                    DROP_BOUNDS,
                    span,
                    DropTraitConstraintsDiag { predicate, tcx: cx.tcx, def_id },
                );
            }
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'_>, ty: &'tcx hir::Ty<'tcx, AmbigArg>) {
        let hir::TyKind::TraitObject(bounds, _lifetime_and_syntax_pointer) = &ty.kind else {
            return;
        };
        for bound in &bounds[..] {
            let def_id = bound.trait_ref.trait_def_id();
            if def_id.is_some_and(|def_id| cx.tcx.is_lang_item(def_id, LangItem::Drop)) {
                let Some(def_id) = cx.tcx.get_diagnostic_item(sym::needs_drop) else { return };
                cx.emit_span_lint(DYN_DROP, bound.span, DropGlue { tcx: cx.tcx, def_id });
            }
        }
    }
}
