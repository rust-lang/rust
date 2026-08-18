//! Check if it's even possible to satisfy the 'where' clauses
//! for this item.
//!
//! It's possible to `#!feature(trivial_bounds)]` to write
//! a function with impossible to satisfy clauses, e.g.:
//! `fn foo() where String: Copy {}`.
//!
//! We don't usually need to worry about this kind of case,
//! since we would get a compilation error if the user tried
//! to call it. However, since we optimize even without any
//! calls to the function, we need to make sure that it even
//! makes sense to try to evaluate the body.
//!
//! If there are unsatisfiable where clauses, then all bets are
//! off, and we just give up.
//!
//! We manually filter the predicates, skipping anything that's not
//! "global". We are in a potentially generic context
//! (e.g. we are evaluating a function without instantiating generic
//! parameters, so this filtering serves two purposes:
//!
//! 1. We skip evaluating any predicates that we would
//!    never be able prove are unsatisfiable (e.g. `<T as Foo>`
//! 2. We avoid trying to normalize predicates involving generic
//!    parameters (e.g. `<T as Foo>::MyItem`). This can confuse
//!    the normalization code (leading to cycle errors), since
//!    it's usually never invoked in this way.

use rustc_middle::mir::{Body, START_BLOCK, TerminatorKind};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFlags, TypeVisitableExt, Unnormalized};
use rustc_span::def_id::DefId;
use rustc_trait_selection::traits;
use tracing::trace;

use crate::PassPolicy;
use crate::pass_manager::MirPass;

fn is_structurally_unsized<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.kind() {
        ty::Str | ty::Slice(_) | ty::Dynamic(_, _) | ty::Foreign(_) => true,
        ty::Tuple(tys) => tys.last().is_some_and(|ty| is_structurally_unsized(tcx, *ty)),
        ty::Adt(def, args) => {
            def.sizedness_constraint(tcx, ty::SizedTraitKind::Sized).is_some_and(|ty| {
                is_structurally_unsized(tcx, ty.instantiate(tcx, args).skip_norm_wip())
            })
        }
        _ => false,
    }
}

fn has_structurally_impossible_sized_clause<'tcx>(
    tcx: TyCtxt<'tcx>,
    sized_trait: DefId,
    predicate: ty::Clause<'tcx>,
) -> bool {
    let Some(trait_predicate) = predicate.as_trait_clause() else {
        return false;
    };
    let trait_predicate = trait_predicate.skip_binder();

    trait_predicate.polarity == ty::ClausePolarity::Positive
        && trait_predicate.def_id() == sized_trait
        && is_structurally_unsized(tcx, trait_predicate.self_ty())
}

pub(crate) struct ImpossibleClauses;

pub(crate) fn has_impossible_clauses<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> bool {
    let clauses = tcx.clauses_of(def_id).instantiate_identity(tcx);
    tracing::trace!(?clauses);

    // Some `Sized` clauses that mention local generics are still impossible
    // for every instantiation, e.g. `dyn Trait<T>: Sized`.
    if let Some(sized_trait) = tcx.lang_items().sized_trait() {
        if clauses
            .clauses
            .iter()
            .copied()
            .map(Unnormalized::skip_norm_wip)
            .any(|clause| has_structurally_impossible_sized_clause(tcx, sized_trait, clause))
        {
            return true;
        }
    }

    let clauses = clauses.clauses.into_iter().map(Unnormalized::skip_norm_wip).filter(|c| {
        !c.has_type_flags(
            // Only consider global clauses to simplify.
            TypeFlags::HAS_FREE_LOCAL_NAMES
                // Clauses that refer to alias constants as they cause cycles.
                | TypeFlags::HAS_CONST_ALIAS,
        )
    });
    let clauses: Vec<_> = traits::elaborate(tcx, clauses).collect();
    tracing::trace!(?clauses);
    clauses.references_error() || traits::impossible_clauses(tcx, clauses)
}

impl<'tcx> MirPass<'tcx> for ImpossibleClauses {
    #[tracing::instrument(level = "trace", skip(self, tcx, body))]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        tracing::trace!(def_id = ?body.source.def_id());
        let impossible =
            body.tainted_by_errors.is_some() || has_impossible_clauses(tcx, body.source.def_id());
        if impossible {
            trace!("found unsatisfiable clauses");
            // Clear the body to only contain a single `unreachable` statement.
            let bbs = body.basic_blocks.as_mut();
            bbs.raw.truncate(1);
            bbs[START_BLOCK].statements.clear();
            bbs[START_BLOCK].terminator_mut().kind = TerminatorKind::Unreachable;
            body.var_debug_info.clear();
            body.local_decls.raw.truncate(body.arg_count + 1);
        }
    }

    fn policy(&self, _sess: &rustc_session::Session) -> PassPolicy {
        // This can only replace code proven unreachable with immediate UB, so it cannot remove UB.
        PassPolicy::optional_non_optimization(true)
    }
}
