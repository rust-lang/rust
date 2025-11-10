//! Impl specialization related things

use hir_def::{ImplId, nameres::crate_def_map};
use intern::sym;
use tracing::debug;

use crate::{
    db::HirDatabase,
    next_solver::{
        DbInterner, TypingMode,
        infer::{
            DbInternerInferExt,
            traits::{Obligation, ObligationCause},
        },
        obligation_ctxt::ObligationCtxt,
    },
};

// rustc does not have a cycle handling for the `specializes` query, meaning a cycle is a bug,
// and indeed I was unable to cause cycles even with erroneous code. However, in r-a we can
// create a cycle if there is an error in the impl's where clauses. I believe well formed code
// cannot create a cycle, but a cycle handler is required nevertheless.
fn specializes_query_cycle(
    _db: &dyn HirDatabase,
    _specializing_impl_def_id: ImplId,
    _parent_impl_def_id: ImplId,
) -> bool {
    false
}

/// Is `specializing_impl_def_id` a specialization of `parent_impl_def_id`?
///
/// For every type that could apply to `specializing_impl_def_id`, we prove that
/// the `parent_impl_def_id` also applies (i.e. it has a valid impl header and
/// its where-clauses hold).
///
/// For the purposes of const traits, we also check that the specializing
/// impl is not more restrictive than the parent impl. That is, if the
/// `parent_impl_def_id` is a const impl (conditionally based off of some `[const]`
/// bounds), then `specializing_impl_def_id` must also be const for the same
/// set of types.
#[salsa::tracked(cycle_result = specializes_query_cycle)]
fn specializes_query(
    db: &dyn HirDatabase,
    specializing_impl_def_id: ImplId,
    parent_impl_def_id: ImplId,
) -> bool {
    let trait_env = db.trait_environment(specializing_impl_def_id.into());
    let interner = DbInterner::new_with(db, Some(trait_env.krate), trait_env.block);

    let specializing_impl_signature = db.impl_signature(specializing_impl_def_id);
    let parent_impl_signature = db.impl_signature(parent_impl_def_id);

    // We determine whether there's a subset relationship by:
    //
    // - replacing bound vars with placeholders in impl1,
    // - assuming the where clauses for impl1,
    // - instantiating impl2 with fresh inference variables,
    // - unifying,
    // - attempting to prove the where clauses for impl2
    //
    // The last three steps are encapsulated in `fulfill_implication`.
    //
    // See RFC 1210 for more details and justification.

    // Currently we do not allow e.g., a negative impl to specialize a positive one
    if specializing_impl_signature.is_negative() != parent_impl_signature.is_negative() {
        return false;
    }

    // create a parameter environment corresponding to an identity instantiation of the specializing impl,
    // i.e. the most generic instantiation of the specializing impl.
    let param_env = trait_env.env;

    // Create an infcx, taking the predicates of the specializing impl as assumptions:
    let infcx = interner.infer_ctxt().build(TypingMode::non_body_analysis());

    let specializing_impl_trait_ref =
        db.impl_trait(specializing_impl_def_id).unwrap().instantiate_identity();
    let cause = &ObligationCause::dummy();
    debug!(
        "fulfill_implication({:?}, trait_ref={:?} |- {:?} applies)",
        param_env, specializing_impl_trait_ref, parent_impl_def_id
    );

    // Attempt to prove that the parent impl applies, given all of the above.

    let mut ocx = ObligationCtxt::new(&infcx);

    let parent_args = infcx.fresh_args_for_item(parent_impl_def_id.into());
    let parent_impl_trait_ref = db
        .impl_trait(parent_impl_def_id)
        .expect("expected source impl to be a trait impl")
        .instantiate(interner, parent_args);

    // do the impls unify? If not, no specialization.
    let Ok(()) = ocx.eq(cause, param_env, specializing_impl_trait_ref, parent_impl_trait_ref)
    else {
        return false;
    };

    // Now check that the source trait ref satisfies all the where clauses of the target impl.
    // This is not just for correctness; we also need this to constrain any params that may
    // only be referenced via projection predicates.
    if let Some(predicates) =
        db.generic_predicates(parent_impl_def_id.into()).instantiate(interner, parent_args)
    {
        ocx.register_obligations(
            predicates
                .map(|predicate| Obligation::new(interner, cause.clone(), param_env, predicate)),
        );
    }

    let errors = ocx.evaluate_obligations_error_on_ambiguity();
    if !errors.is_empty() {
        // no dice!
        debug!(
            "fulfill_implication: for impls on {:?} and {:?}, \
                 could not fulfill: {:?} given {:?}",
            specializing_impl_trait_ref, parent_impl_trait_ref, errors, param_env
        );
        return false;
    }

    // FIXME: Check impl constness (when we implement const impls).

    debug!(
        "fulfill_implication: an impl for {:?} specializes {:?}",
        specializing_impl_trait_ref, parent_impl_trait_ref
    );

    true
}

// This function is used to avoid creating the query for crates that does not define `#![feature(specialization)]`,
// as the solver is calling this a lot, and creating the query consumes a lot of memory.
pub(crate) fn specializes(
    db: &dyn HirDatabase,
    specializing_impl_def_id: ImplId,
    parent_impl_def_id: ImplId,
) -> bool {
    let module = specializing_impl_def_id.loc(db).container;

    // We check that the specializing impl comes from a crate that has specialization enabled.
    //
    // We don't really care if the specialized impl (the parent) is in a crate that has
    // specialization enabled, since it's not being specialized.
    //
    // rustc also checks whether the specializing impls comes from a macro marked
    // `#[allow_internal_unstable(specialization)]`, but `#[allow_internal_unstable]`
    // is an internal feature, std is not using it for specialization nor is likely to
    // ever use it, and we don't have the span information necessary to replicate that.
    let def_map = crate_def_map(db, module.krate());
    if !def_map.is_unstable_feature_enabled(&sym::specialization)
        && !def_map.is_unstable_feature_enabled(&sym::min_specialization)
    {
        return false;
    }

    specializes_query(db, specializing_impl_def_id, parent_impl_def_id)
}
