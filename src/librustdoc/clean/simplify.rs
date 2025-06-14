//! Simplification of where-clauses and parameter bounds into a prettier and
//! more canonical form.
//!
//! Currently all cross-crate-inlined function use `rustc_middle::ty` to reconstruct
//! the AST (e.g., see all of `clean::inline`), but this is not always a
//! non-lossy transformation. The current format of storage for where-clauses
//! for functions and such is simply a list of predicates. One example of this
//! is that the AST predicate of: `where T: Trait<Foo = Bar>` is encoded as:
//! `where T: Trait, <T as Trait>::Foo = Bar`.
//!
//! This module attempts to reconstruct the original where and/or parameter
//! bounds by special casing scenarios such as these. Fun!

use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::unord::UnordSet;
use rustc_hir::def_id::DefId;
use thin_vec::ThinVec;

use crate::clean;
use crate::clean::{GenericArgs as PP, WherePredicate as WP};
use crate::core::DocContext;

pub(crate) fn where_clauses(cx: &DocContext<'_>, clauses: ThinVec<WP>) -> ThinVec<WP> {
    // First, partition the where clause into its separate components.
    //
    // We use `FxIndexMap` so that the insertion order is preserved to prevent messing up to
    // the order of the generated bounds.
    let mut tybounds = FxIndexMap::default();
    let mut lifetimes = Vec::new();
    let mut equalities = Vec::new();

    for clause in clauses {
        match clause {
            WP::BoundPredicate { ty, bounds, bound_params } => {
                let (b, p): &mut (Vec<_>, Vec<_>) = tybounds.entry(ty).or_default();
                b.extend(bounds);
                p.extend(bound_params);
            }
            WP::RegionPredicate { lifetime, bounds } => {
                lifetimes.push((lifetime, bounds));
            }
            WP::EqPredicate { lhs, rhs } => equalities.push((lhs, rhs)),
        }
    }

    // Look for equality predicates on associated types that can be merged into
    // general bound predicates.
    equalities.retain(|(lhs, rhs)| {
        let Some((bounds, _)) = tybounds.get_mut(&lhs.self_type) else { return true };
        merge_bounds(cx, bounds, lhs.trait_.as_ref().unwrap().def_id(), lhs.assoc.clone(), rhs)
    });

    // And finally, let's reassemble everything
    let mut clauses = ThinVec::with_capacity(lifetimes.len() + tybounds.len() + equalities.len());
    clauses.extend(
        lifetimes.into_iter().map(|(lt, bounds)| WP::RegionPredicate { lifetime: lt, bounds }),
    );
    clauses.extend(tybounds.into_iter().map(|(ty, (bounds, bound_params))| WP::BoundPredicate {
        ty,
        bounds,
        bound_params,
    }));
    clauses.extend(equalities.into_iter().map(|(lhs, rhs)| WP::EqPredicate { lhs, rhs }));
    clauses
}

pub(crate) fn merge_bounds(
    cx: &clean::DocContext<'_>,
    bounds: &mut [clean::GenericBound],
    trait_did: DefId,
    assoc: clean::PathSegment,
    rhs: &clean::Term,
) -> bool {
    !bounds.iter_mut().any(|b| {
        let trait_ref = match *b {
            clean::GenericBound::TraitBound(ref mut tr, _) => tr,
            clean::GenericBound::Outlives(..) | clean::GenericBound::Use(_) => return false,
        };
        // If this QPath's trait `trait_did` is the same as, or a supertrait
        // of, the bound's trait `did` then we can keep going, otherwise
        // this is just a plain old equality bound.
        if !trait_is_same_or_supertrait(cx, trait_ref.trait_.def_id(), trait_did) {
            return false;
        }
        let last = trait_ref.trait_.segments.last_mut().expect("segments were empty");

        match last.args {
            PP::AngleBracketed { ref mut constraints, .. } => {
                constraints.push(clean::AssocItemConstraint {
                    assoc: assoc.clone(),
                    kind: clean::AssocItemConstraintKind::Equality { term: rhs.clone() },
                });
            }
            PP::Parenthesized { ref mut output, .. } => match output {
                Some(o) => assert_eq!(&clean::Term::Type(o.as_ref().clone()), rhs),
                None => {
                    if *rhs != clean::Term::Type(clean::Type::Tuple(Vec::new())) {
                        *output = Some(Box::new(rhs.ty().unwrap().clone()));
                    }
                }
            },
            PP::ReturnTypeNotation => {
                // Cannot merge bounds with RTN.
                return false;
            }
        };
        true
    })
}

fn trait_is_same_or_supertrait(cx: &DocContext<'_>, child: DefId, trait_: DefId) -> bool {
    if child == trait_ {
        return true;
    }
    let predicates = cx.tcx.explicit_super_predicates_of(child);
    predicates
        .iter_identity_copied()
        .filter_map(|(pred, _)| Some(pred.as_trait_clause()?.def_id()))
        .any(|did| trait_is_same_or_supertrait(cx, did, trait_))
}

pub(crate) fn sized_bounds(cx: &mut DocContext<'_>, generics: &mut clean::Generics) {
    let mut sized_params = UnordSet::new();

    // In the surface language, all type parameters except `Self` have an
    // implicit `Sized` bound unless removed with `?Sized`.
    // However, in the list of where-predicates below, `Sized` appears like a
    // normal bound: It's either present (the type is sized) or
    // absent (the type might be unsized) but never *maybe* (i.e. `?Sized`).
    //
    // This is unsuitable for rendering.
    // Thus, as a first step remove all `Sized` bounds that should be implicit.
    //
    // Note that associated types also have an implicit `Sized` bound but we
    // don't actually know the set of associated types right here so that
    // should be handled when cleaning associated types.
    generics.where_predicates.retain(|pred| {
        if let WP::BoundPredicate { ty: clean::Generic(param), bounds, .. } = pred
            && bounds.iter().any(|b| b.is_sized_bound(cx))
        {
            sized_params.insert(*param);
            false
        } else {
            true
        }
    });

    // As a final step, go through the type parameters again and insert a
    // `?Sized` bound for each one we didn't find to be `Sized`.
    for param in &generics.params {
        if let clean::GenericParamDefKind::Type { .. } = param.kind
            && !sized_params.contains(&param.name)
        {
            generics.where_predicates.push(WP::BoundPredicate {
                ty: clean::Type::Generic(param.name),
                bounds: vec![clean::GenericBound::maybe_sized(cx)],
                bound_params: Vec::new(),
            })
        }
    }
}

/// Move bounds that are (likely) directly attached to generic parameters from the where-clause to
/// the respective parameter.
///
/// There is no guarantee that this is what the user actually wrote but we have no way of knowing.
// FIXME(fmease): It'd make a lot of sense to just incorporate this logic into `clean_ty_generics`
// making every of its users benefit from it.
pub(crate) fn move_bounds_to_generic_parameters(generics: &mut clean::Generics) {
    use clean::types::*;

    let mut where_predicates = ThinVec::new();
    for mut pred in generics.where_predicates.drain(..) {
        if let WherePredicate::BoundPredicate { ty: Generic(arg), bounds, .. } = &mut pred
            && let Some(GenericParamDef {
                kind: GenericParamDefKind::Type { bounds: param_bounds, .. },
                ..
            }) = generics.params.iter_mut().find(|param| &param.name == arg)
        {
            param_bounds.extend(bounds.drain(..));
        } else if let WherePredicate::RegionPredicate { lifetime: Lifetime(arg), bounds } =
            &mut pred
            && let Some(GenericParamDef {
                kind: GenericParamDefKind::Lifetime { outlives: param_bounds },
                ..
            }) = generics.params.iter_mut().find(|param| &param.name == arg)
        {
            param_bounds.extend(bounds.drain(..).map(|bound| match bound {
                GenericBound::Outlives(lifetime) => lifetime,
                _ => unreachable!(),
            }));
        } else {
            where_predicates.push(pred);
        }
    }
    generics.where_predicates = where_predicates;
}
