use crate::utils::{match_def_path, paths, span_lint};
use if_chain::if_chain;
use rustc_hir::{GenericBound, GenericParam, WhereBoundPredicate, WherePredicate};
use rustc_lint::LateLintPass;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for generics with `std::ops::Drop` as bounds.
    ///
    /// **Why is this bad?** `Drop` bounds do not really accomplish anything.
    /// A type may have compiler-generated drop glue without implementing the
    /// `Drop` trait itself. The `Drop` trait also only has one method,
    /// `Drop::drop`, and that function is by fiat not callable in user code.
    /// So there is really no use case for using `Drop` in trait bounds.
    ///
    /// The most likely use case of a drop bound is to distinguish between types
    /// that have destructors and types that don't. Combined with specialization,
    /// a naive coder would write an implementation that assumed a type could be
    /// trivially dropped, then write a specialization for `T: Drop` that actually
    /// calls the destructor. Except that doing so is not correct; String, for
    /// example, doesn't actually implement Drop, but because String contains a
    /// Vec, assuming it can be trivially dropped will leak memory.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// fn foo<T: Drop>() {}
    /// ```
    pub DROP_BOUNDS,
    correctness,
    "Bounds of the form `T: Drop` are useless"
}

const DROP_BOUNDS_SUMMARY: &str = "Bounds of the form `T: Drop` are useless. \
                                   Use `std::mem::needs_drop` to detect if a type has drop glue.";

declare_lint_pass!(DropBounds => [DROP_BOUNDS]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for DropBounds {
    fn check_generic_param(&mut self, cx: &rustc_lint::LateContext<'a, 'tcx>, p: &'tcx GenericParam<'_>) {
        for bound in p.bounds.iter() {
            lint_bound(cx, bound);
        }
    }
    fn check_where_predicate(&mut self, cx: &rustc_lint::LateContext<'a, 'tcx>, p: &'tcx WherePredicate<'_>) {
        if let WherePredicate::BoundPredicate(WhereBoundPredicate { bounds, .. }) = p {
            for bound in *bounds {
                lint_bound(cx, bound);
            }
        }
    }
}

fn lint_bound<'a, 'tcx>(cx: &rustc_lint::LateContext<'a, 'tcx>, bound: &'tcx GenericBound<'_>) {
    if_chain! {
        if let GenericBound::Trait(t, _) = bound;
        if let Some(def_id) = t.trait_ref.path.res.opt_def_id();
        if match_def_path(cx, def_id, &paths::DROP_TRAIT);
        then {
            span_lint(
                cx,
                DROP_BOUNDS,
                t.span,
                DROP_BOUNDS_SUMMARY
            );
        }
    }
}
