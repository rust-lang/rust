use crate::utils::{in_macro, snippet, snippet_with_applicability, span_lint_and_help, SpanlessHash};
use if_chain::if_chain;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::Applicability;
use rustc_hir::{GenericBound, Generics, WherePredicate};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// **What it does:** This lint warns about unnecessary type repetitions in trait bounds
    ///
    /// **Why is this bad?** Repeating the type for every bound makes the code
    /// less readable than combining the bounds
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// pub fn foo<T>(t: T) where T: Copy, T: Clone {}
    /// ```
    ///
    /// Could be written as:
    ///
    /// ```rust
    /// pub fn foo<T>(t: T) where T: Copy + Clone {}
    /// ```
    pub TYPE_REPETITION_IN_BOUNDS,
    pedantic,
    "Types are repeated unnecessary in trait bounds use `+` instead of using `T: _, T: _`"
}

#[derive(Copy, Clone)]
pub struct TraitBounds {
    max_trait_bounds: u64,
}

impl TraitBounds {
    #[must_use]
    pub fn new(max_trait_bounds: u64) -> Self {
        Self { max_trait_bounds }
    }
}

impl_lint_pass!(TraitBounds => [TYPE_REPETITION_IN_BOUNDS]);

impl<'tcx> LateLintPass<'tcx> for TraitBounds {
    fn check_generics(&mut self, cx: &LateContext<'tcx>, gen: &'tcx Generics<'_>) {
        if in_macro(gen.span) {
            return;
        }
        let hash = |ty| -> u64 {
            let mut hasher = SpanlessHash::new(cx);
            hasher.hash_ty(ty);
            hasher.finish()
        };
        let mut map = FxHashMap::default();
        let mut applicability = Applicability::MaybeIncorrect;
        for bound in gen.where_clause.predicates {
            if_chain! {
                if let WherePredicate::BoundPredicate(ref p) = bound;
                if p.bounds.len() as u64 <= self.max_trait_bounds;
                if !in_macro(p.span);
                let h = hash(&p.bounded_ty);
                if let Some(ref v) = map.insert(h, p.bounds.iter().collect::<Vec<_>>());

                then {
                    let mut hint_string = format!(
                        "consider combining the bounds: `{}:",
                        snippet(cx, p.bounded_ty.span, "_")
                    );
                    for b in v.iter() {
                        if let GenericBound::Trait(ref poly_trait_ref, _) = b {
                            let path = &poly_trait_ref.trait_ref.path;
                            hint_string.push_str(&format!(
                                " {} +",
                                snippet_with_applicability(cx, path.span, "..", &mut applicability)
                            ));
                        }
                    }
                    for b in p.bounds.iter() {
                        if let GenericBound::Trait(ref poly_trait_ref, _) = b {
                            let path = &poly_trait_ref.trait_ref.path;
                            hint_string.push_str(&format!(
                                " {} +",
                                snippet_with_applicability(cx, path.span, "..", &mut applicability)
                            ));
                        }
                    }
                    hint_string.truncate(hint_string.len() - 2);
                    hint_string.push('`');
                    span_lint_and_help(
                        cx,
                        TYPE_REPETITION_IN_BOUNDS,
                        p.span,
                        "this type has already been used as a bound predicate",
                        None,
                        &hint_string,
                    );
                }
            }
        }
    }
}
