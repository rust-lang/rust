//! Stuff that will probably mostly replaced by Chalk.
use std::collections::HashMap;

use crate::db::HirDatabase;
use super::{ TraitRef, Substs, infer::{ TypeVarId, InferTy}, Ty};

// Copied (and simplified) from Chalk

#[derive(Clone, Debug, PartialEq, Eq)]
/// A (possible) solution for a proposed goal. Usually packaged in a `Result`,
/// where `Err` represents definite *failure* to prove a goal.
pub enum Solution {
    /// The goal indeed holds, and there is a unique value for all existential
    /// variables.
    Unique(Substs),

    /// The goal may be provable in multiple ways, but regardless we may have some guidance
    /// for type inference.
    Ambig(Guidance),
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// When a goal holds ambiguously (e.g., because there are multiple possible
/// solutions), we issue a set of *guidance* back to type inference.
pub enum Guidance {
    /// The existential variables *must* have the given values if the goal is
    /// ever to hold, but that alone isn't enough to guarantee the goal will
    /// actually hold.
    Definite(Substs),

    /// There are multiple plausible values for the existentials, but the ones
    /// here are suggested as the preferred choice heuristically. These should
    /// be used for inference fallback only.
    Suggested(Substs),

    /// There's no useful information to feed back to type inference
    Unknown,
}

/// Something that needs to be proven (by Chalk) during type checking, e.g. that
/// a certain type implements a certain trait. Proving the Obligation might
/// result in additional information about inference variables.
///
/// This might be handled by Chalk when we integrate it?
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Obligation {
    /// Prove that a certain type implements a trait (the type is the `Self` type
    /// parameter to the `TraitRef`).
    Trait(TraitRef),
}

/// Rudimentary check whether an impl exists for a given type and trait; this
/// will actually be done by chalk.
pub(crate) fn implements(db: &impl HirDatabase, trait_ref: TraitRef) -> Option<Solution> {
    // FIXME use all trait impls in the whole crate graph
    let krate = trait_ref.trait_.module(db).krate(db);
    let krate = match krate {
        Some(krate) => krate,
        None => return None,
    };
    let crate_impl_blocks = db.impls_in_crate(krate);
    let mut impl_blocks = crate_impl_blocks.lookup_impl_blocks_for_trait(&trait_ref.trait_);
    impl_blocks
        .find_map(|impl_block| unify_trait_refs(&trait_ref, &impl_block.target_trait_ref(db)?))
}

pub(super) fn canonicalize(trait_ref: TraitRef) -> (TraitRef, Vec<TypeVarId>) {
    let mut canonical = HashMap::new(); // mapping uncanonical -> canonical
    let mut uncanonical = Vec::new(); // mapping canonical -> uncanonical (which is dense)
    let mut substs = trait_ref.substs.0.to_vec();
    for ty in &mut substs {
        ty.walk_mut(&mut |ty| match ty {
            Ty::Infer(InferTy::TypeVar(tv)) => {
                let tv: &mut TypeVarId = tv;
                *tv = *canonical.entry(*tv).or_insert_with(|| {
                    let i = uncanonical.len();
                    uncanonical.push(*tv);
                    TypeVarId(i as u32)
                });
            }
            _ => {}
        });
    }
    (TraitRef { substs: substs.into(), ..trait_ref }, uncanonical)
}

fn unify_trait_refs(tr1: &TraitRef, tr2: &TraitRef) -> Option<Solution> {
    if tr1.trait_ != tr2.trait_ {
        return None;
    }
    let mut solution_substs = Vec::new();
    for (t1, t2) in tr1.substs.0.iter().zip(tr2.substs.0.iter()) {
        // this is very bad / hacky 'unification' logic, just enough to make the simple tests pass
        match (t1, t2) {
            (_, Ty::Infer(InferTy::TypeVar(_))) | (_, Ty::Unknown) | (_, Ty::Param { .. }) => {
                // type variable (or similar) in the impl, we just assume it works
            }
            (Ty::Infer(InferTy::TypeVar(v1)), _) => {
                // type variable in the query and fixed type in the impl, record its value
                solution_substs.resize_with(v1.0 as usize + 1, || Ty::Unknown);
                solution_substs[v1.0 as usize] = t2.clone();
            }
            _ => {
                // check that they're equal (actually we'd have to recurse etc.)
                if t1 != t2 {
                    return None;
                }
            }
        }
    }
    Some(Solution::Unique(solution_substs.into()))
}
