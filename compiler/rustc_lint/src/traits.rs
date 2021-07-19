use crate::LateContext;
use crate::LateLintPass;
use crate::LintContext;
use rustc_hir as hir;
use rustc_span::symbol::sym;

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
    /// `Drop` bounds do not really accomplish anything. A type may have
    /// compiler-generated drop glue without implementing the `Drop` trait
    /// itself. The `Drop` trait also only has one method, `Drop::drop`, and
    /// that function is by fiat not callable in user code. So there is really
    /// no use case for using `Drop` in trait bounds.
    ///
    /// The most likely use case of a drop bound is to distinguish between
    /// types that have destructors and types that don't. Combined with
    /// specialization, a naive coder would write an implementation that
    /// assumed a type could be trivially dropped, then write a specialization
    /// for `T: Drop` that actually calls the destructor. Except that doing so
    /// is not correct; String, for example, doesn't actually implement Drop,
    /// but because String contains a Vec, assuming it can be trivially dropped
    /// will leak memory.
    pub DROP_BOUNDS,
    Warn,
    "bounds of the form `T: Drop` are useless"
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
        use rustc_middle::ty::PredicateKind::*;

        let predicates = cx.tcx.explicit_predicates_of(item.def_id);
        for &(predicate, span) in predicates.predicates {
            let trait_predicate = match predicate.kind().skip_binder() {
                Trait(trait_predicate, _constness) => trait_predicate,
                _ => continue,
            };
            let def_id = trait_predicate.trait_ref.def_id;
            if cx.tcx.lang_items().drop_trait() == Some(def_id) {
                // Explicitly allow `impl Drop`, a drop-guards-as-Voldemort-type pattern.
                if trait_predicate.trait_ref.self_ty().is_impl_trait() {
                    continue;
                }
                cx.struct_span_lint(DROP_BOUNDS, span, |lint| {
                    let needs_drop = match cx.tcx.get_diagnostic_item(sym::needs_drop) {
                        Some(needs_drop) => needs_drop,
                        None => return,
                    };
                    let msg = format!(
                        "bounds on `{}` are useless, consider instead \
                         using `{}` to detect if a type has a destructor",
                        predicate,
                        cx.tcx.def_path_str(needs_drop)
                    );
                    lint.build(&msg).emit()
                });
            }
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'_>, ty: &'tcx hir::Ty<'tcx>) {
        let bounds = match &ty.kind {
            hir::TyKind::TraitObject(bounds, _lifetime, _syntax) => bounds,
            _ => return,
        };
        for bound in &bounds[..] {
            let def_id = bound.trait_ref.trait_def_id();
            if cx.tcx.lang_items().drop_trait() == def_id {
                cx.struct_span_lint(DYN_DROP, bound.span, |lint| {
                    let needs_drop = match cx.tcx.get_diagnostic_item(sym::needs_drop) {
                        Some(needs_drop) => needs_drop,
                        None => return,
                    };
                    let msg = format!(
                        "types that do not implement `Drop` can still have drop glue, consider \
                        instead using `{}` to detect whether a type is trivially dropped",
                        cx.tcx.def_path_str(needs_drop)
                    );
                    lint.build(&msg).emit()
                });
            }
        }
    }
}
