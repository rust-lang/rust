use rustc_ast::visit::FnKind;
use rustc_ast::{Fn, NodeId, WherePredicateKind};
use rustc_data_structures::fx::FxHashMap;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

use clippy_utils::diagnostics::span_lint;
use clippy_utils::source::SpanRangeExt;

declare_clippy_lint! {
    /// ### What it does
    /// Check if a generic is defined both in the bound predicate and in the `where` clause.
    ///
    /// ### Why is this bad?
    /// It can be confusing for developers when seeing bounds for a generic in multiple places.
    ///
    /// ### Example
    /// ```no_run
    /// fn ty<F: std::fmt::Debug>(a: F)
    /// where
    ///     F: Sized,
    /// {}
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn ty<F>(a: F)
    /// where
    ///     F: Sized + std::fmt::Debug,
    /// {}
    /// ```
    #[clippy::version = "1.78.0"]
    pub MULTIPLE_BOUND_LOCATIONS,
    suspicious,
    "defining generic bounds in multiple locations"
}

declare_lint_pass!(MultipleBoundLocations => [MULTIPLE_BOUND_LOCATIONS]);

impl EarlyLintPass for MultipleBoundLocations {
    fn check_fn(&mut self, cx: &EarlyContext<'_>, kind: FnKind<'_>, _: Span, _: NodeId) {
        if let FnKind::Fn(_, _, Fn { generics, .. }) = kind
            && !generics.params.is_empty()
            && !generics.where_clause.predicates.is_empty()
        {
            let mut generic_params_with_bounds = FxHashMap::default();

            for param in &generics.params {
                if !param.bounds.is_empty() {
                    generic_params_with_bounds.insert(param.ident.as_str(), param.ident.span);
                }
            }
            for clause in &generics.where_clause.predicates {
                match &clause.kind {
                    WherePredicateKind::BoundPredicate(pred) => {
                        if (!pred.bound_generic_params.is_empty() || !pred.bounds.is_empty())
                            && let Some(Some(bound_span)) = pred
                                .bounded_ty
                                .span
                                .with_source_text(cx, |src| generic_params_with_bounds.get(src))
                        {
                            emit_lint(cx, *bound_span, pred.bounded_ty.span);
                        }
                    },
                    WherePredicateKind::RegionPredicate(pred) => {
                        if !pred.bounds.is_empty()
                            && let Some(bound_span) = generic_params_with_bounds.get(&pred.lifetime.ident.as_str())
                        {
                            emit_lint(cx, *bound_span, pred.lifetime.ident.span);
                        }
                    },
                    WherePredicateKind::EqPredicate(_) => {},
                }
            }
        }
    }
}

fn emit_lint(cx: &EarlyContext<'_>, bound_span: Span, where_span: Span) {
    span_lint(
        cx,
        MULTIPLE_BOUND_LOCATIONS,
        vec![bound_span, where_span],
        "bound is defined in more than one place",
    );
}
