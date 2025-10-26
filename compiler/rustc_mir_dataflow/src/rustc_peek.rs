use rustc_ast::MetaItem;
use rustc_middle::mir::{self, Body, Local, Location};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::def_id::DefId;
use rustc_span::{Span, Symbol, sym};
use tracing::{debug, info};

use crate::errors::{
    PeekArgumentNotALocal, PeekArgumentUntracked, PeekBitNotSet, PeekMustBeNotTemporary,
    PeekMustBePlaceOrRefPlace, StopAfterDataFlowEndedCompilation,
};
use crate::framework::BitSetExt;
use crate::impls::{MaybeInitializedPlaces, MaybeLiveLocals, MaybeUninitializedPlaces};
use crate::move_paths::{HasMoveData, LookupResult, MoveData, MovePathIndex};
use crate::{Analysis, JoinSemiLattice, ResultsCursor};

fn has_rustc_mir_with(tcx: TyCtxt<'_>, def_id: DefId, name: Symbol) -> Option<MetaItem> {
    for attr in tcx.get_attrs(def_id, sym::rustc_mir) {
        let items = attr.meta_item_list();
        for item in items.iter().flat_map(|l| l.iter()) {
            match item.meta_item() {
                Some(mi) if mi.has_name(name) => return Some(mi.clone()),
                _ => continue,
            }
        }
    }
    None
}

pub fn sanity_check<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
    let def_id = body.source.def_id();
    if !tcx.has_attr(def_id, sym::rustc_mir) {
        debug!("skipping rustc_peek::SanityCheck on {}", tcx.def_path_str(def_id));
        return;
    } else {
        debug!("running rustc_peek::SanityCheck on {}", tcx.def_path_str(def_id));
    }

    let move_data = MoveData::gather_moves(body, tcx, |_| true);

    if has_rustc_mir_with(tcx, def_id, sym::rustc_peek_maybe_init).is_some() {
        let flow_inits = MaybeInitializedPlaces::new(tcx, body, &move_data)
            .iterate_to_fixpoint(tcx, body, None)
            .into_results_cursor(body);
        sanity_check_via_rustc_peek(tcx, flow_inits);
    }

    if has_rustc_mir_with(tcx, def_id, sym::rustc_peek_maybe_uninit).is_some() {
        let flow_uninits = MaybeUninitializedPlaces::new(tcx, body, &move_data)
            .iterate_to_fixpoint(tcx, body, None)
            .into_results_cursor(body);
        sanity_check_via_rustc_peek(tcx, flow_uninits);
    }

    if has_rustc_mir_with(tcx, def_id, sym::rustc_peek_liveness).is_some() {
        let flow_liveness =
            MaybeLiveLocals.iterate_to_fixpoint(tcx, body, None).into_results_cursor(body);
        sanity_check_via_rustc_peek(tcx, flow_liveness);
    }

    if has_rustc_mir_with(tcx, def_id, sym::stop_after_dataflow).is_some() {
        tcx.dcx().emit_fatal(StopAfterDataFlowEndedCompilation);
    }
}

/// This function scans `mir` for all calls to the intrinsic
/// `rustc_peek` that have the expression form `rustc_peek(&expr)`.
///
/// For each such call, determines what the dataflow bit-state is for
/// the L-value corresponding to `expr`; if the bit-state is a 1, then
/// that call to `rustc_peek` is ignored by the sanity check. If the
/// bit-state is a 0, then this pass emits an error message saying
/// "rustc_peek: bit not set".
///
/// The intention is that one can write unit tests for dataflow by
/// putting code into a UI test and using `rustc_peek` to
/// make observations about the results of dataflow static analyses.
///
/// (If there are any calls to `rustc_peek` that do not match the
/// expression form above, then that emits an error as well, but those
/// errors are not intended to be used for unit tests.)
fn sanity_check_via_rustc_peek<'tcx, A>(tcx: TyCtxt<'tcx>, mut cursor: ResultsCursor<'_, 'tcx, A>)
where
    A: RustcPeekAt<'tcx>,
{
    let def_id = cursor.body().source.def_id();
    debug!("sanity_check_via_rustc_peek def_id: {:?}", def_id);

    let peek_calls = cursor.body().basic_blocks.iter_enumerated().filter_map(|(bb, block_data)| {
        PeekCall::from_terminator(tcx, block_data.terminator()).map(|call| (bb, block_data, call))
    });

    for (bb, block_data, call) in peek_calls {
        // Look for a sequence like the following to indicate that we should be peeking at `_1`:
        //    _2 = &_1;
        //    rustc_peek(_2);
        //
        //    /* or */
        //
        //    _2 = _1;
        //    rustc_peek(_2);
        let (statement_index, peek_rval) = block_data
            .statements
            .iter()
            .enumerate()
            .find_map(|(i, stmt)| value_assigned_to_local(stmt, call.arg).map(|rval| (i, rval)))
            .expect(
                "call to rustc_peek should be preceded by \
                    assignment to temporary holding its argument",
            );

        match (call.kind, peek_rval) {
            (PeekCallKind::ByRef, mir::Rvalue::Ref(_, _, place))
            | (
                PeekCallKind::ByVal,
                mir::Rvalue::Use(mir::Operand::Move(place) | mir::Operand::Copy(place)),
            ) => {
                let loc = Location { block: bb, statement_index };
                cursor.seek_before_primary_effect(loc);
                let state = cursor.get();
                let analysis = cursor.analysis();
                analysis.peek_at(tcx, *place, state, call);
            }

            _ => {
                tcx.dcx().emit_err(PeekMustBePlaceOrRefPlace { span: call.span });
            }
        }
    }
}

/// If `stmt` is an assignment where the LHS is the given local (with no projections), returns the
/// RHS of the assignment.
fn value_assigned_to_local<'a, 'tcx>(
    stmt: &'a mir::Statement<'tcx>,
    local: Local,
) -> Option<&'a mir::Rvalue<'tcx>> {
    if let mir::StatementKind::Assign(box (place, rvalue)) = &stmt.kind
        && let Some(l) = place.as_local()
        && local == l
    {
        return Some(&*rvalue);
    }

    None
}

#[derive(Clone, Copy, Debug)]
enum PeekCallKind {
    ByVal,
    ByRef,
}

impl PeekCallKind {
    fn from_arg_ty(arg: Ty<'_>) -> Self {
        match arg.kind() {
            ty::Ref(_, _, _) => PeekCallKind::ByRef,
            _ => PeekCallKind::ByVal,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PeekCall {
    arg: Local,
    kind: PeekCallKind,
    span: Span,
}

impl PeekCall {
    fn from_terminator<'tcx>(
        tcx: TyCtxt<'tcx>,
        terminator: &mir::Terminator<'tcx>,
    ) -> Option<Self> {
        use mir::Operand;

        let span = terminator.source_info.span;
        if let mir::TerminatorKind::Call { func: Operand::Constant(func), args, .. } =
            &terminator.kind
            && let ty::FnDef(def_id, fn_args) = *func.const_.ty().kind()
        {
            if tcx.intrinsic(def_id)?.name != sym::rustc_peek {
                return None;
            }

            assert_eq!(fn_args.len(), 1);
            let kind = PeekCallKind::from_arg_ty(fn_args.type_at(0));
            let arg = match &args[0].node {
                Operand::Copy(place) | Operand::Move(place) => {
                    if let Some(local) = place.as_local() {
                        local
                    } else {
                        tcx.dcx().emit_err(PeekMustBeNotTemporary { span });
                        return None;
                    }
                }
                _ => {
                    tcx.dcx().emit_err(PeekMustBeNotTemporary { span });
                    return None;
                }
            };

            return Some(PeekCall { arg, kind, span });
        }

        None
    }
}

trait RustcPeekAt<'tcx>: Analysis<'tcx> {
    fn peek_at(
        &self,
        tcx: TyCtxt<'tcx>,
        place: mir::Place<'tcx>,
        state: &Self::Domain,
        call: PeekCall,
    );
}

impl<'tcx, A, D> RustcPeekAt<'tcx> for A
where
    A: Analysis<'tcx, Domain = D> + HasMoveData<'tcx>,
    D: JoinSemiLattice + Clone + BitSetExt<MovePathIndex>,
{
    fn peek_at(
        &self,
        tcx: TyCtxt<'tcx>,
        place: mir::Place<'tcx>,
        state: &Self::Domain,
        call: PeekCall,
    ) {
        match self.move_data().rev_lookup.find(place.as_ref()) {
            LookupResult::Exact(peek_mpi) => {
                let bit_state = state.contains(peek_mpi);
                debug!("rustc_peek({:?} = &{:?}) bit_state: {}", call.arg, place, bit_state);
                if !bit_state {
                    tcx.dcx().emit_err(PeekBitNotSet { span: call.span });
                }
            }

            LookupResult::Parent(..) => {
                tcx.dcx().emit_err(PeekArgumentUntracked { span: call.span });
            }
        }
    }
}

impl<'tcx> RustcPeekAt<'tcx> for MaybeLiveLocals {
    fn peek_at(
        &self,
        tcx: TyCtxt<'tcx>,
        place: mir::Place<'tcx>,
        state: &Self::Domain,
        call: PeekCall,
    ) {
        info!(?place, "peek_at");
        let Some(local) = place.as_local() else {
            tcx.dcx().emit_err(PeekArgumentNotALocal { span: call.span });
            return;
        };

        if !state.contains(local) {
            tcx.dcx().emit_err(PeekBitNotSet { span: call.span });
        }
    }
}
