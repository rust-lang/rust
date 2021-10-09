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
use rustc_hir::def_id::DefId;
use rustc_middle::ty;
use rustc_span::Symbol;

use crate::clean;
use crate::clean::GenericArgs as PP;
use crate::clean::WherePredicate as WP;
use crate::core::DocContext;

crate fn where_clauses(cx: &DocContext<'_>, clauses: Vec<WP>) -> Vec<WP> {
    // First, partition the where clause into its separate components.
    //
    // We use `FxIndexMap` so that the insertion order is preserved to prevent messing up to
    // the order of the generated bounds.
    let mut params: FxIndexMap<Symbol, (Vec<_>, Vec<_>)> = FxIndexMap::default();
    let mut lifetimes = Vec::new();
    let mut equalities = Vec::new();
    let mut tybounds = Vec::new();

    for clause in clauses {
        match clause {
            WP::BoundPredicate { ty, bounds, bound_params } => match ty {
                clean::Generic(s) => {
                    let (b, p) = params.entry(s).or_default();
                    b.extend(bounds);
                    p.extend(bound_params);
                }
                t => tybounds.push((t, (bounds, bound_params))),
            },
            WP::RegionPredicate { lifetime, bounds } => {
                lifetimes.push((lifetime, bounds));
            }
            WP::EqPredicate { lhs, rhs } => equalities.push((lhs, rhs)),
        }
    }

    // Look for equality predicates on associated types that can be merged into
    // general bound predicates
    equalities.retain(|&(ref lhs, ref rhs)| {
        let (self_, trait_did, name) = if let Some(p) = lhs.projection() {
            p
        } else {
            return true;
        };
        let generic = match self_ {
            clean::Generic(s) => s,
            _ => return true,
        };
        let (bounds, _) = match params.get_mut(generic) {
            Some(bound) => bound,
            None => return true,
        };

        merge_bounds(cx, bounds, trait_did, name, rhs)
    });

    // And finally, let's reassemble everything
    let mut clauses = Vec::new();
    clauses.extend(
        lifetimes.into_iter().map(|(lt, bounds)| WP::RegionPredicate { lifetime: lt, bounds }),
    );
    clauses.extend(params.into_iter().map(|(k, (bounds, params))| WP::BoundPredicate {
        ty: clean::Generic(k),
        bounds,
        bound_params: params,
    }));
    clauses.extend(tybounds.into_iter().map(|(ty, (bounds, bound_params))| WP::BoundPredicate {
        ty,
        bounds,
        bound_params,
    }));
    clauses.extend(equalities.into_iter().map(|(lhs, rhs)| WP::EqPredicate { lhs, rhs }));
    clauses
}

crate fn merge_bounds(
    cx: &clean::DocContext<'_>,
    bounds: &mut Vec<clean::GenericBound>,
    trait_did: DefId,
    name: Symbol,
    rhs: &clean::Type,
) -> bool {
    !bounds.iter_mut().any(|b| {
        let trait_ref = match *b {
            clean::GenericBound::TraitBound(ref mut tr, _) => tr,
            clean::GenericBound::Outlives(..) => return false,
        };
        // If this QPath's trait `trait_did` is the same as, or a supertrait
        // of, the bound's trait `did` then we can keep going, otherwise
        // this is just a plain old equality bound.
        if !trait_is_same_or_supertrait(cx, trait_ref.trait_.def_id(), trait_did) {
            return false;
        }
        let last = trait_ref.trait_.segments.last_mut().expect("segments were empty");
        match last.args {
            PP::AngleBracketed { ref mut bindings, .. } => {
                bindings.push(clean::TypeBinding {
                    name,
                    kind: clean::TypeBindingKind::Equality { ty: rhs.clone() },
                });
            }
            PP::Parenthesized { ref mut output, .. } => match output {
                Some(o) => assert_eq!(o.as_ref(), rhs),
                None => {
                    if *rhs != clean::Type::Tuple(Vec::new()) {
                        *output = Some(Box::new(rhs.clone()));
                    }
                }
            },
        };
        true
    })
}

fn trait_is_same_or_supertrait(cx: &DocContext<'_>, child: DefId, trait_: DefId) -> bool {
    if child == trait_ {
        return true;
    }
    let predicates = cx.tcx.super_predicates_of(child);
    debug_assert!(cx.tcx.generics_of(child).has_self);
    let self_ty = cx.tcx.types.self_param;
    predicates
        .predicates
        .iter()
        .filter_map(|(pred, _)| {
            if let ty::PredicateKind::Trait(pred) = pred.kind().skip_binder() {
                if pred.trait_ref.self_ty() == self_ty { Some(pred.def_id()) } else { None }
            } else {
                None
            }
        })
        .any(|did| trait_is_same_or_supertrait(cx, did, trait_))
}
