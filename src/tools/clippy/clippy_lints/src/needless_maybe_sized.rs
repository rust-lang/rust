use clippy_utils::diagnostics::span_lint_and_then;
use rustc_errors::Applicability;
use rustc_hir::def_id::{DefId, DefIdMap};
use rustc_hir::{BoundPolarity, GenericBound, Generics, PolyTraitRef, TraitBoundModifiers, WherePredicateKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{ClauseKind, PredicatePolarity};
use rustc_session::declare_lint_pass;
use rustc_span::symbol::Ident;

declare_clippy_lint! {
    /// ### What it does
    /// Lints `?Sized` bounds applied to type parameters that cannot be unsized
    ///
    /// ### Why is this bad?
    /// The `?Sized` bound is misleading because it cannot be satisfied by an
    /// unsized type
    ///
    /// ### Example
    /// ```rust
    /// // `T` cannot be unsized because `Clone` requires it to be `Sized`
    /// fn f<T: Clone + ?Sized>(t: &T) {}
    /// ```
    /// Use instead:
    /// ```rust
    /// fn f<T: Clone>(t: &T) {}
    ///
    /// // or choose alternative bounds for `T` so that it can be unsized
    /// ```
    #[clippy::version = "1.81.0"]
    pub NEEDLESS_MAYBE_SIZED,
    suspicious,
    "a `?Sized` bound that is unusable due to a `Sized` requirement"
}
declare_lint_pass!(NeedlessMaybeSized => [NEEDLESS_MAYBE_SIZED]);

#[allow(clippy::struct_field_names)]
struct Bound<'tcx> {
    /// The [`DefId`] of the type parameter the bound refers to
    param: DefId,
    ident: Ident,

    trait_bound: &'tcx PolyTraitRef<'tcx>,

    predicate_pos: usize,
    bound_pos: usize,
}

/// Finds all of the [`Bound`]s that refer to a type parameter and are not from a macro expansion
fn type_param_bounds<'tcx>(generics: &'tcx Generics<'tcx>) -> impl Iterator<Item = Bound<'tcx>> {
    generics
        .predicates
        .iter()
        .enumerate()
        .filter_map(|(predicate_pos, predicate)| {
            let WherePredicateKind::BoundPredicate(bound_predicate) = &predicate.kind else {
                return None;
            };

            let (param, ident) = bound_predicate.bounded_ty.as_generic_param()?;

            Some(
                bound_predicate
                    .bounds
                    .iter()
                    .enumerate()
                    .filter_map(move |(bound_pos, bound)| match bound {
                        GenericBound::Trait(trait_bound) => Some(Bound {
                            param,
                            ident,
                            trait_bound,
                            predicate_pos,
                            bound_pos,
                        }),
                        GenericBound::Outlives(_) | GenericBound::Use(..) => None,
                    })
                    .filter(|bound| !bound.trait_bound.span.from_expansion()),
            )
        })
        .flatten()
}

/// Searches the supertraits of the trait referred to by `trait_bound` recursively, returning the
/// path taken to find a `Sized` bound if one is found
fn path_to_sized_bound(cx: &LateContext<'_>, trait_bound: &PolyTraitRef<'_>) -> Option<Vec<DefId>> {
    fn search(cx: &LateContext<'_>, path: &mut Vec<DefId>) -> bool {
        let trait_def_id = *path.last().unwrap();

        if Some(trait_def_id) == cx.tcx.lang_items().sized_trait() {
            return true;
        }

        for (predicate, _) in cx.tcx.explicit_super_predicates_of(trait_def_id).iter_identity_copied() {
            if let ClauseKind::Trait(trait_predicate) = predicate.kind().skip_binder()
                && trait_predicate.polarity == PredicatePolarity::Positive
                && !path.contains(&trait_predicate.def_id())
            {
                path.push(trait_predicate.def_id());
                if search(cx, path) {
                    return true;
                }
                path.pop();
            }
        }

        false
    }

    let mut path = vec![trait_bound.trait_ref.trait_def_id()?];
    search(cx, &mut path).then_some(path)
}

impl LateLintPass<'_> for NeedlessMaybeSized {
    fn check_generics(&mut self, cx: &LateContext<'_>, generics: &Generics<'_>) {
        let Some(sized_trait) = cx.tcx.lang_items().sized_trait() else {
            return;
        };

        let maybe_sized_params: DefIdMap<_> = type_param_bounds(generics)
            .filter(|bound| {
                bound.trait_bound.trait_ref.trait_def_id() == Some(sized_trait)
                    && matches!(bound.trait_bound.modifiers.polarity, BoundPolarity::Maybe(_))
            })
            .map(|bound| (bound.param, bound))
            .collect();

        for bound in type_param_bounds(generics) {
            if bound.trait_bound.modifiers == TraitBoundModifiers::NONE
                && let Some(sized_bound) = maybe_sized_params.get(&bound.param)
                && let Some(path) = path_to_sized_bound(cx, bound.trait_bound)
            {
                span_lint_and_then(
                    cx,
                    NEEDLESS_MAYBE_SIZED,
                    sized_bound.trait_bound.span,
                    "`?Sized` bound is ignored because of a `Sized` requirement",
                    |diag| {
                        let ty_param = sized_bound.ident;
                        diag.span_note(
                            bound.trait_bound.span,
                            format!("`{ty_param}` cannot be unsized because of the bound"),
                        );

                        for &[current_id, next_id] in path.array_windows() {
                            let current = cx.tcx.item_name(current_id);
                            let next = cx.tcx.item_name(next_id);
                            diag.note(format!("...because `{current}` has the bound `{next}`"));
                        }

                        diag.span_suggestion_verbose(
                            generics.span_for_bound_removal(sized_bound.predicate_pos, sized_bound.bound_pos),
                            "change the bounds that require `Sized`, or remove the `?Sized` bound",
                            "",
                            Applicability::MaybeIncorrect,
                        );
                    },
                );

                return;
            }
        }
    }
}
