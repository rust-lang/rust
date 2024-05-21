use rustc_middle::ty::{self, TyCtxt};
use rustc_span::def_id::DefId;
/// We early return `NoSolution` when trying to normalize associated types if
/// we know them to be rigid. This is necessary if there are a huge amount of
/// rigid associated types in the `ParamEnv` as we would otherwise get hangs
/// when trying to normalize each associated type with all other associated types.
///
/// See trait-system-refactor-initiative#109 for an example.
///
/// ```plain
/// is_rigid_alias(alias) :-
///     is_placeholder(alias.self_ty),
///     no_applicable_blanket_impls(alias.trait_def_id),
///     not(may_normalize_via_env(alias)),
///
/// may_normalize_via_env(alias) :- exists<projection_clause> {
///     projection_clause.def_id == alias.def_id,
///     projection_clause.args may_unify alias.args,
/// }
/// ```
#[instrument(level = "debug", skip(tcx), ret)]
pub(super) fn is_rigid_alias<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    alias: ty::AliasTerm<'tcx>,
) -> bool {
    // FIXME: This could consider associated types as rigid as long
    // as it considers the *recursive* item bounds of the alias,
    // which is non-trivial. We may be forced to handle this case
    // in the future.
    alias.self_ty().is_placeholder()
        && no_applicable_blanket_impls(tcx, alias.trait_def_id(tcx))
        && !may_normalize_via_env(param_env, alias)
}

// FIXME: This could check whether the blanket impl has any where-bounds
// which definitely don't hold. Doing so is quite annoying, both just in
// general, but we also have to be careful about builtin blanket impls,
// e.g. `DiscriminantKind`.
#[instrument(level = "trace", skip(tcx), ret)]
fn no_applicable_blanket_impls<'tcx>(tcx: TyCtxt<'tcx>, trait_def_id: DefId) -> bool {
    // FIXME(ptr_metadata): There's currently a builtin impl for `Pointee` which
    // applies for all `T` as long as `T: Sized` holds. THis impl should
    // get removed in favor of `Pointee` being a super trait of `Sized`.
    tcx.trait_impls_of(trait_def_id).blanket_impls().is_empty()
        && !tcx.lang_items().pointee_trait().is_some_and(|def_id| trait_def_id == def_id)
}

#[instrument(level = "trace", ret)]
fn may_normalize_via_env<'tcx>(param_env: ty::ParamEnv<'tcx>, alias: ty::AliasTerm<'tcx>) -> bool {
    for clause in param_env.caller_bounds() {
        let Some(projection_pred) = clause.as_projection_clause() else {
            continue;
        };

        if projection_pred.projection_def_id() != alias.def_id {
            continue;
        };

        let drcx = ty::fast_reject::DeepRejectCtxt {
            treat_obligation_params: ty::fast_reject::TreatParams::ForLookup,
        };
        if drcx.args_may_unify(alias.args, projection_pred.skip_binder().projection_term.args) {
            return true;
        }
    }

    false
}
