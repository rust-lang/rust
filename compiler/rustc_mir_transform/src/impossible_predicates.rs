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
use rustc_middle::ty::{TyCtxt, TypeVisitableExt};
use rustc_trait_selection::traits;
use tracing::trace;

use crate::pass_manager::MirPass;

pub(crate) struct ImpossiblePredicates;

impl<'tcx> MirPass<'tcx> for ImpossiblePredicates {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let predicates = tcx
            .predicates_of(body.source.def_id())
            .predicates
            .iter()
            .filter_map(|(p, _)| if p.is_global() { Some(*p) } else { None });
        if traits::impossible_predicates(tcx, traits::elaborate(tcx, predicates).collect()) {
            trace!("found unsatisfiable predicates for {:?}", body.source);
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
