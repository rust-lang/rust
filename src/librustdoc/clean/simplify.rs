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
use rustc_data_structures::thin_vec::ThinVec;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{TyCtxt, Unnormalized};

use crate::clean;
use crate::clean::{GenericArgs as PP, WherePredicate as WP};
use crate::core::DocContext;

pub(crate) fn where_clauses(tcx: TyCtxt<'_>, clauses: ThinVec<WP>) -> ThinVec<WP> {
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
        merge_bounds(tcx, bounds, lhs.trait_.as_ref().unwrap().def_id(), lhs.assoc.clone(), rhs)
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
    tcx: TyCtxt<'_>,
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
        if !trait_is_same_or_supertrait(tcx, trait_ref.trait_.def_id(), trait_did) {
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

fn trait_is_same_or_supertrait(tcx: TyCtxt<'_>, child: DefId, trait_: DefId) -> bool {
    if child == trait_ {
        return true;
    }
    let predicates = tcx.explicit_super_predicates_of(child);
    predicates
        .iter_identity_copied()
        .map(Unnormalized::skip_norm_wip)
        .filter_map(|(pred, _)| Some(pred.as_trait_clause()?.def_id()))
        .any(|did| trait_is_same_or_supertrait(tcx, did, trait_))
}

/// Reconstruct all sizedness bounds on non-`Self` type parameters as they appear in the surface
/// language given generics that were cleaned from the middle::ty IR.
///
/// For example, assuming `T` is a type parameter of the owner of `generics`,
/// `T: Sized` gets dropped and `T: MetaSized` gets rewritten to `T: ?Sized`.
pub(crate) fn sizedness_bounds(cx: &mut DocContext<'_>, generics: &mut clean::Generics) {
    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    enum Sizedness {
        PointeeSized,
        MetaSized,
        Sized,
    }

    let mut type_params: FxIndexMap<_, _> = generics
        .params
        .iter()
        .filter(|param| matches!(param.kind, clean::GenericParamDefKind::Type { .. }))
        .map(|param| (param.name, Sizedness::PointeeSized))
        .collect();

    generics.where_predicates.retain(|pred| {
        let WP::BoundPredicate { ty: clean::Generic(param), bounds, .. } = pred else {
            return true;
        };

        // We require the caller to pass generics that were cleaned from the middle::ty IR.
        // We know that that cleaning process never generates more than one bound per predicate.
        let [bound] = &*bounds else { unreachable!() };

        let clean::GenericBound::TraitBound(trait_ref, hir::TraitBoundModifiers::NONE) = bound
        else {
            return true;
        };

        // This transformation is only valid on type parameters defined on the closest item.
        // If the parameter was defined by the parent item we know that the sizedness bound
        // *has* to be user-written in which case we have to preserve it as is.
        let Some(param_sizedness) = type_params.get_mut(param) else { return true };

        let sizedness = match cx.tcx.as_lang_item(trait_ref.trait_.def_id()) {
            Some(hir::LangItem::Sized) => Sizedness::Sized,
            Some(hir::LangItem::MetaSized) => Sizedness::MetaSized,
            _ => return true,
        };

        if sizedness > *param_sizedness {
            *param_sizedness = sizedness;
        }

        false
    });

    for (param, sizedness) in type_params {
        generics.where_predicates.push(WP::BoundPredicate {
            ty: clean::Type::Generic(param),
            bounds: vec![match sizedness {
                // FIXME(sized-hierarchy, #157247): Actually render `MetaSized` as `MetaSized` and
                // `PointeeSized` as `PointeeSized` instead of `?Sized` if the crate enables
                // `sized_hierarchy` and doesn't set `#![doc(dont_leak…)]`.
                Sizedness::MetaSized | Sizedness::PointeeSized => {
                    clean::GenericBound::maybe_sized(cx)
                }
                Sizedness::Sized => continue,
            }],
            bound_params: Vec::new(),
        });
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
