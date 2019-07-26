use crate::utils::{in_macro, span_help_and_lint, SpanlessHash};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_tool_lint, impl_lint_pass};
use rustc_data_structures::fx::FxHashMap;
use rustc::hir::*;

#[derive(Copy, Clone)]
pub struct TraitBounds;

declare_clippy_lint! {
    pub TYPE_REPETITION_IN_BOUNDS,
    complexity,
    "Types are repeated unnecessary in trait bounds use `+` instead of using `T: _, T: _`"
}

impl_lint_pass!(TraitBounds => [TYPE_REPETITION_IN_BOUNDS]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for TraitBounds {
    fn check_generics(&mut self, cx: &LateContext<'a, 'tcx>, gen: &'tcx Generics) {
        if in_macro(gen.span) {
            return;
        }
        let hash = | ty | -> u64 {
            let mut hasher = SpanlessHash::new(cx, cx.tables);
            hasher.hash_ty(ty);
            hasher.finish()
        };
        let mut map = FxHashMap::default();
        for bound in &gen.where_clause.predicates {
            if let WherePredicate::BoundPredicate(ref p) = bound {
                let h = hash(&p.bounded_ty);
                if let Some(ref v) = map.insert(h, p.bounds.iter().collect::<Vec<_>>()) {
                    let mut hint_string = format!("consider combining the bounds: `{:?}: ", p.bounded_ty);
                    for b in v.iter() {
                        hint_string.push_str(&format!("{:?}, ", b));
                    }
                    for b in p.bounds.iter() {
                        hint_string.push_str(&format!("{:?}, ", b));
                    }
                    hint_string.truncate(hint_string.len() - 2);
                    hint_string.push('`');
                    span_help_and_lint(
                        cx,
                        TYPE_REPETITION_IN_BOUNDS,
                        p.span,
                        "this type has already been used as a bound predicate",
                        &hint_string,
                    );
                }
            }
        }
    }
}
