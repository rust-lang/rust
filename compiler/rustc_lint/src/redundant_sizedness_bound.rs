use rustc_errors::Applicability;
use rustc_hir::def_id::{DefId, DefIdMap};
use rustc_hir::{
    BoundPolarity, GenericBound, Generics, PolyTraitRef, TraitBoundModifiers, WherePredicateKind,
};
use rustc_middle::ty::{ClauseKind, PredicatePolarity};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::symbol::Ident;

use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `redundant_sizedness_bound` lint detects redundant sizedness bounds applied to type parameters that are already
    /// otherwise implied.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![feature(sized_hierarchy)]
    /// use std::marker::MetaSized;
    /// // `T` must be `Sized` due to the bound `Clone`, thus `?Sized` is redundant.
    /// fn f<T: Clone + ?Sized>(t: &T) {}
    /// // `T` is `Sized` due to `Clone` bound, thereby implying `MetaSized` and making the explicit `MetaSized` bound redundant.
    /// fn g<T: MetaSized + Clone>(t: &T) {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Sizedness bounds that have no effect as another bound implies a greater degree of sizedness are potentially misleading
    /// This lint notifies the user of such redundant bounds.
    pub REDUNDANT_SIZEDNESS_BOUND,
    Warn,
    "a sizedness bound that is redundant due to another bound"
}
declare_lint_pass!(RedundantSizednessBound => [REDUNDANT_SIZEDNESS_BOUND]);

struct Bound<'tcx> {
    /// The [`DefId`] of the type parameter the bound refers to
    param: DefId,
    /// Identifier of type parameter
    ident: Ident,
    /// A reference to the trait bound applied to the parameter
    trait_bound: &'tcx PolyTraitRef<'tcx>,
    /// The index of the predicate within the generics predicate list
    predicate_pos: usize,
    /// Position of the bound in the bounds list of a predicate
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
                        GenericBound::Trait(trait_bound) => {
                            Some(Bound { param, ident, trait_bound, predicate_pos, bound_pos })
                        }
                        GenericBound::Outlives(_) | GenericBound::Use(..) => None,
                    })
                    .filter(|bound| !bound.trait_bound.span.from_expansion()),
            )
        })
        .flatten()
}

/// Searches the supertraits of the trait referred to by `trait_bound` recursively, returning the
/// path taken to find the `target` bound if one is found
fn path_to_bound(
    cx: &LateContext<'_>,
    trait_bound: &PolyTraitRef<'_>,
    target: DefId,
) -> Option<Vec<DefId>> {
    fn search(cx: &LateContext<'_>, path: &mut Vec<DefId>, target: DefId) -> bool {
        let trait_def_id = *path.last().unwrap();

        if trait_def_id == target {
            return true;
        }

        for (predicate, _) in
            cx.tcx.explicit_super_predicates_of(trait_def_id).iter_identity_copied()
        {
            if let ClauseKind::Trait(trait_predicate) = predicate.kind().skip_binder()
                && trait_predicate.polarity == PredicatePolarity::Positive
                && !path.contains(&trait_predicate.def_id())
            {
                path.push(trait_predicate.def_id());
                if search(cx, path, target) {
                    return true;
                }
                path.pop();
            }
        }

        false
    }

    let mut path = vec![trait_bound.trait_ref.trait_def_id()?];
    search(cx, &mut path, target).then_some(path)
}

// Checks if there exists a bound `redundant_bound` that is already implied by `implicit_bound`
fn check_redundant_sizedness_bound(
    redundant_bound: DefId,
    redundant_bound_polarity: BoundPolarity,
    implicit_bound: DefId,
    cx: &LateContext<'_>,
    generics: &Generics<'_>,
) -> bool {
    let redundant_sized_params: DefIdMap<_> = type_param_bounds(generics)
        .filter(|bound| {
            bound.trait_bound.trait_ref.trait_def_id() == Some(redundant_bound)
                && std::mem::discriminant(&bound.trait_bound.modifiers.polarity)
                    == std::mem::discriminant(&redundant_bound_polarity)
        })
        .map(|bound| (bound.param, bound))
        .collect();

    for bound in type_param_bounds(generics) {
        if bound.trait_bound.modifiers == TraitBoundModifiers::NONE
            && let Some(redundant_sized_bound) = redundant_sized_params.get(&bound.param)
            && let Some(path) = path_to_bound(cx, bound.trait_bound, implicit_bound)
        {
            let redundant_bound_polarity_str = match redundant_bound_polarity {
                BoundPolarity::Maybe(_) => "?",
                _ => "",
            };
            cx.span_lint(
                REDUNDANT_SIZEDNESS_BOUND,
                redundant_sized_bound.trait_bound.span,
                |diag| {
                    let redundant_bound_str = cx.tcx.def_path_str(redundant_bound);
                    let implicit_bound_str = cx.tcx.def_path_str(implicit_bound);

                    diag.primary_message(format!(
                        "`{}{}` bound is redundant because of a `{}` requirement",
                        redundant_bound_polarity_str, redundant_bound_str, implicit_bound_str,
                    ));
                    let ty_param = redundant_sized_bound.ident;
                    diag.span_note(
                        bound.trait_bound.span,
                        format!(
                            "`{ty_param}` is implied to be `{}` because of the bound",
                            implicit_bound_str,
                        ),
                    );

                    for &[current_id, next_id] in path.array_windows() {
                        let current = cx.tcx.item_name(current_id);
                        let next = cx.tcx.item_name(next_id);
                        diag.note(format!("...because `{current}` has the bound `{next}`"));
                    }

                    diag.span_suggestion_verbose(
                        generics.span_for_bound_removal(
                            redundant_sized_bound.predicate_pos,
                            redundant_sized_bound.bound_pos,
                        ),
                        format!(
                            "change the bounds that require `{}`, or remove the `{}{}` bound",
                            implicit_bound_str, redundant_bound_polarity_str, redundant_bound_str,
                        ),
                        "",
                        Applicability::MaybeIncorrect,
                    );
                },
            );

            return true;
        }
    }
    false
}

impl LateLintPass<'_> for RedundantSizednessBound {
    fn check_generics(&mut self, cx: &LateContext<'_>, generics: &Generics<'_>) {
        let Some(sized_trait) = cx.tcx.lang_items().sized_trait() else {
            return;
        };
        let Some(meta_sized_trait) = cx.tcx.lang_items().meta_sized_trait() else {
            return;
        };
        let Some(pointee_sized_trait) = cx.tcx.lang_items().pointee_sized_trait() else {
            return;
        };

        if check_redundant_sizedness_bound(
            sized_trait,
            BoundPolarity::Maybe(Default::default()),
            sized_trait,
            cx,
            generics,
        ) {
            return;
        }
        if check_redundant_sizedness_bound(
            meta_sized_trait,
            BoundPolarity::Positive,
            sized_trait,
            cx,
            generics,
        ) {
            return;
        }
        if check_redundant_sizedness_bound(
            pointee_sized_trait,
            BoundPolarity::Positive,
            sized_trait,
            cx,
            generics,
        ) {
            return;
        }
        if check_redundant_sizedness_bound(
            pointee_sized_trait,
            BoundPolarity::Positive,
            meta_sized_trait,
            cx,
            generics,
        ) {
            return;
        }
    }
}
