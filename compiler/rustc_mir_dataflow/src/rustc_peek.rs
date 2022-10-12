use rustc_span::symbol::sym;
use rustc_span::Span;

use rustc_index::bit_set::ChunkedBitSet;
use rustc_middle::mir::MirPass;
use rustc_middle::mir::{self, Body, Local, Location};
use rustc_middle::ty::{self, Ty, TyCtxt};

use crate::errors::{
    PeekArgumentNotALocal, PeekArgumentUntracked, PeekBitNotSet, PeekMustBeNotTemporary,
    PeekMustBePlaceOrRefPlace, StopAfterDataFlowEndedCompilation,
};
use crate::framework::BitSetExt;
use crate::impls::{
    DefinitelyInitializedPlaces, MaybeInitializedPlaces, MaybeLiveLocals, MaybeUninitializedPlaces,
};
use crate::move_paths::{HasMoveData, MoveData};
use crate::move_paths::{LookupResult, MovePathIndex};
use crate::MoveDataParamEnv;
use crate::{Analysis, JoinSemiLattice, Results, ResultsCursor};

pub struct SanityCheck;

// FIXME: This should be a `MirLint`, but it needs to be moved back to `rustc_mir_transform` first.
impl<'tcx> MirPass<'tcx> for SanityCheck {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        use crate::has_rustc_mir_with;
        let def_id = body.source.def_id();
        if !tcx.has_attr(def_id, sym::rustc_mir) {
            debug!("skipping rustc_peek::SanityCheck on {}", tcx.def_path_str(def_id));
            return;
        } else {
            debug!("running rustc_peek::SanityCheck on {}", tcx.def_path_str(def_id));
        }

        let param_env = tcx.param_env(def_id);
        let (_, move_data) = MoveData::gather_moves(body, tcx, param_env).unwrap();
        let mdpe = MoveDataParamEnv { move_data, param_env };

        if has_rustc_mir_with(tcx, def_id, sym::rustc_peek_maybe_init).is_some() {
            let flow_inits = MaybeInitializedPlaces::new(tcx, body, &mdpe)
                .into_engine(tcx, body)
                .iterate_to_fixpoint();

            sanity_check_via_rustc_peek(tcx, body, &flow_inits);
        }

        if has_rustc_mir_with(tcx, def_id, sym::rustc_peek_maybe_uninit).is_some() {
            let flow_uninits = MaybeUninitializedPlaces::new(tcx, body, &mdpe)
                .into_engine(tcx, body)
                .iterate_to_fixpoint();

            sanity_check_via_rustc_peek(tcx, body, &flow_uninits);
        }

        if has_rustc_mir_with(tcx, def_id, sym::rustc_peek_definite_init).is_some() {
            let flow_def_inits = DefinitelyInitializedPlaces::new(tcx, body, &mdpe)
                .into_engine(tcx, body)
                .iterate_to_fixpoint();

            sanity_check_via_rustc_peek(tcx, body, &flow_def_inits);
        }

        if has_rustc_mir_with(tcx, def_id, sym::rustc_peek_liveness).is_some() {
            let flow_liveness = MaybeLiveLocals.into_engine(tcx, body).iterate_to_fixpoint();

            sanity_check_via_rustc_peek(tcx, body, &flow_liveness);
        }

        if has_rustc_mir_with(tcx, def_id, sym::stop_after_dataflow).is_some() {
            tcx.sess.emit_fatal(StopAfterDataFlowEndedCompilation);
        }
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
pub fn sanity_check_via_rustc_peek<'tcx, A>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    results: &Results<'tcx, A>,
) where
    A: RustcPeekAt<'tcx>,
{
    let def_id = body.source.def_id();
    debug!("sanity_check_via_rustc_peek def_id: {:?}", def_id);

    let mut cursor = ResultsCursor::new(body, results);

    let peek_calls = body.basic_blocks.iter_enumerated().filter_map(|(bb, block_data)| {
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
                results.analysis.peek_at(tcx, *place, state, call);
            }

            _ => {
                tcx.sess.emit_err(PeekMustBePlaceOrRefPlace { span: call.span });
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
    if let mir::StatementKind::Assign(box (place, rvalue)) = &stmt.kind {
        if let Some(l) = place.as_local() {
            if local == l {
                return Some(&*rvalue);
            }
        }
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
pub struct PeekCall {
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
        {
            if let ty::FnDef(def_id, substs) = *func.literal.ty().kind() {
                let name = tcx.item_name(def_id);
                if !tcx.is_intrinsic(def_id) || name != sym::rustc_peek {
                    return None;
                }

                assert_eq!(args.len(), 1);
                let kind = PeekCallKind::from_arg_ty(substs.type_at(0));
                let arg = match &args[0] {
                    Operand::Copy(place) | Operand::Move(place) => {
                        if let Some(local) = place.as_local() {
                            local
                        } else {
                            tcx.sess.emit_err(PeekMustBeNotTemporary { span });
                            return None;
                        }
                    }
                    _ => {
                        tcx.sess.emit_err(PeekMustBeNotTemporary { span });
                        return None;
                    }
                };

                return Some(PeekCall { arg, kind, span });
            }
        }

        None
    }
}

pub trait RustcPeekAt<'tcx>: Analysis<'tcx> {
    fn peek_at(
        &self,
        tcx: TyCtxt<'tcx>,
        place: mir::Place<'tcx>,
        flow_state: &Self::Domain,
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
        flow_state: &Self::Domain,
        call: PeekCall,
    ) {
        match self.move_data().rev_lookup.find(place.as_ref()) {
            LookupResult::Exact(peek_mpi) => {
                let bit_state = flow_state.contains(peek_mpi);
                debug!("rustc_peek({:?} = &{:?}) bit_state: {}", call.arg, place, bit_state);
                if !bit_state {
                    tcx.sess.emit_err(PeekBitNotSet { span: call.span });
                }
            }

            LookupResult::Parent(..) => {
                tcx.sess.emit_err(PeekArgumentUntracked { span: call.span });
            }
        }
    }
}

impl<'tcx> RustcPeekAt<'tcx> for MaybeLiveLocals {
    fn peek_at(
        &self,
        tcx: TyCtxt<'tcx>,
        place: mir::Place<'tcx>,
        flow_state: &ChunkedBitSet<Local>,
        call: PeekCall,
    ) {
        info!(?place, "peek_at");
        let Some(local) = place.as_local() else {
            tcx.sess.emit_err(PeekArgumentNotALocal { span: call.span });
            return;
        };

        if !flow_state.contains(local) {
            tcx.sess.emit_err(PeekBitNotSet { span: call.span });
        }
    }
}
