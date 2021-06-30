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

declare_lint_pass!(
    /// Lint for bounds of the form `T: Drop`, which usually
    /// indicate an attempt to emulate `std::mem::needs_drop`.
    DropTraitConstraints => [DROP_BOUNDS]
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
                        "bounds on `{}` are most likely incorrect, consider instead \
                         using `{}` to detect whether a type can be trivially dropped",
                        predicate,
                        cx.tcx.def_path_str(needs_drop)
                    );
                    lint.build(&msg).emit()
                });
            }
        }
    }
}
