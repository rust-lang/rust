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

declare_lint_pass!(
    /// Lint for bounds of the form `T: Drop`, which usually
    /// indicate an attempt to emulate `std::mem::needs_drop`.
    DropTraitConstraints => [DROP_BOUNDS]
);

impl<'tcx> LateLintPass<'tcx> for DropTraitConstraints {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        use rustc_middle::ty::PredicateKind::*;

        let def_id = cx.tcx.hir().local_def_id(item.hir_id);
        let predicates = cx.tcx.explicit_predicates_of(def_id);
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
}
