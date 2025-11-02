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
//! never be able prove are unsatisfiable (e.g. `<T as Foo>`
//! 2. We avoid trying to normalize predicates involving generic
//! parameters (e.g. `<T as Foo>::MyItem`). This can confuse
//! the normalization code (leading to cycle errors), since
//! it's usually never invoked in this way.

use rustc_middle::mir::{Body, START_BLOCK, TerminatorKind};
use rustc_middle::ty::{TyCtxt, TypeFlags, TypeVisitableExt};
use rustc_span::def_id::DefId;
use rustc_trait_selection::traits;
use tracing::trace;

use crate::pass_manager::MirPass;

pub(crate) struct ImpossiblePredicates;

fn has_impossible_predicates(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let predicates = tcx.predicates_of(def_id).instantiate_identity(tcx);
    tracing::trace!(?predicates);
    let predicates = predicates.predicates.into_iter().filter(|p| {
        !p.has_type_flags(
            // Only consider global clauses to simplify.
            TypeFlags::HAS_FREE_LOCAL_NAMES
                // Clauses that refer to unevaluated constants as they cause cycles.
                | TypeFlags::HAS_CT_PROJECTION,
        )
    });
    let predicates: Vec<_> = traits::elaborate(tcx, predicates).collect();
    tracing::trace!(?predicates);
    predicates.references_error() || traits::impossible_predicates(tcx, predicates)
}

impl<'tcx> MirPass<'tcx> for ImpossiblePredicates {
    #[tracing::instrument(level = "trace", skip(self, tcx, body))]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        tracing::trace!(def_id = ?body.source.def_id());
        let impossible = body.tainted_by_errors.is_some()
            || has_impossible_predicates(tcx, body.source.def_id());
        if impossible {
            trace!("found unsatisfiable predicates");
            // Clear the body to only contain a single `unreachable` statement.
            let bbs = body.basic_blocks.as_mut();
            bbs.raw.truncate(1);
            bbs[START_BLOCK].statements.clear();
            bbs[START_BLOCK].terminator_mut().kind = TerminatorKind::Unreachable;
            body.var_debug_info.clear();
            body.local_decls.raw.truncate(body.arg_count + 1);
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}
