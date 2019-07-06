use rustc_target::spec::abi::{Abi};
use syntax::ast;
use syntax::symbol::sym;
use syntax_pos::Span;

use rustc::ty::{self, TyCtxt};
use rustc::hir::def_id::DefId;
use rustc::mir::{self, Body, Location, Local};
use rustc_data_structures::bit_set::BitSet;
use crate::transform::{MirPass, MirSource};

use crate::dataflow::{self, do_dataflow, DebugFormatted};
use crate::dataflow::MoveDataParamEnv;
use crate::dataflow::BitDenotation;
use crate::dataflow::DataflowResults;
use crate::dataflow::{
    DefinitelyInitializedPlaces, MaybeInitializedPlaces, MaybeUninitializedPlaces
};
use crate::dataflow::move_paths::{MovePathIndex, LookupResult};
use crate::dataflow::move_paths::{HasMoveData, MoveData};

use crate::dataflow::has_rustc_mir_with;

pub struct SanityCheck;

impl MirPass for SanityCheck {
    fn run_pass<'tcx>(&self, tcx: TyCtxt<'tcx>, src: MirSource<'tcx>, body: &mut Body<'tcx>) {
        let def_id = src.def_id();
        if !tcx.has_attr(def_id, sym::rustc_mir) {
            debug!("skipping rustc_peek::SanityCheck on {}", tcx.def_path_str(def_id));
            return;
        } else {
            debug!("running rustc_peek::SanityCheck on {}", tcx.def_path_str(def_id));
        }

        let attributes = tcx.get_attrs(def_id);
        let param_env = tcx.param_env(def_id);
        let move_data = MoveData::gather_moves(body, tcx).unwrap();
        let mdpe = MoveDataParamEnv { move_data: move_data, param_env: param_env };
        let dead_unwinds = BitSet::new_empty(body.basic_blocks().len());
        let flow_inits =
            do_dataflow(tcx, body, def_id, &attributes, &dead_unwinds,
                        MaybeInitializedPlaces::new(tcx, body, &mdpe),
                        |bd, i| DebugFormatted::new(&bd.move_data().move_paths[i]));
        let flow_uninits =
            do_dataflow(tcx, body, def_id, &attributes, &dead_unwinds,
                        MaybeUninitializedPlaces::new(tcx, body, &mdpe),
                        |bd, i| DebugFormatted::new(&bd.move_data().move_paths[i]));
        let flow_def_inits =
            do_dataflow(tcx, body, def_id, &attributes, &dead_unwinds,
                        DefinitelyInitializedPlaces::new(tcx, body, &mdpe),
                        |bd, i| DebugFormatted::new(&bd.move_data().move_paths[i]));

        if has_rustc_mir_with(&attributes, sym::rustc_peek_maybe_init).is_some() {
            sanity_check_via_rustc_peek(tcx, body, def_id, &attributes, &flow_inits);
        }
        if has_rustc_mir_with(&attributes, sym::rustc_peek_maybe_uninit).is_some() {
            sanity_check_via_rustc_peek(tcx, body, def_id, &attributes, &flow_uninits);
        }
        if has_rustc_mir_with(&attributes, sym::rustc_peek_definite_init).is_some() {
            sanity_check_via_rustc_peek(tcx, body, def_id, &attributes, &flow_def_inits);
        }
        if has_rustc_mir_with(&attributes, sym::stop_after_dataflow).is_some() {
            tcx.sess.fatal("stop_after_dataflow ended compilation");
        }
    }
}

/// This function scans `mir` for all calls to the intrinsic
/// `rustc_peek` that have the expression form `rustc_peek(&expr)`.
///
/// For each such call, determines what the dataflow bit-state is for
/// the L-value corresponding to `expr`; if the bit-state is a 1, then
/// that call to `rustc_peek` is ignored by the sanity check. If the
/// bit-state is a 0, then this pass emits a error message saying
/// "rustc_peek: bit not set".
///
/// The intention is that one can write unit tests for dataflow by
/// putting code into a compile-fail test and using `rustc_peek` to
/// make observations about the results of dataflow static analyses.
///
/// (If there are any calls to `rustc_peek` that do not match the
/// expression form above, then that emits an error as well, but those
/// errors are not intended to be used for unit tests.)
pub fn sanity_check_via_rustc_peek<'tcx, O>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    def_id: DefId,
    _attributes: &[ast::Attribute],
    results: &DataflowResults<'tcx, O>,
) where O: RustcPeekAt<'tcx> {
    debug!("sanity_check_via_rustc_peek def_id: {:?}", def_id);

    let peek_calls = body
            .basic_blocks()
            .iter_enumerated()
            .filter_map(|(bb, data)| {
                data.terminator
                    .as_ref()
                    .and_then(|term| PeekCall::from_terminator(tcx, term))
                    .map(|call| (bb, data, call))
            });

    for (bb, data, call) in peek_calls {
        // Look for a sequence like the following to indicate that we should be peeking at `_1`:
        //    _2 = &_1
        //    rustc_peek(_2)
        let (statement_index, peek_rval) = data.statements
            .iter()
            .map(|stmt| value_assigned_to_local(stmt, call.arg))
            .enumerate()
            .filter_map(|(idx, rval)| rval.map(|r| (idx, r)))
            .next()
            .expect("call to rustc_peek should be preceded by \
                    assignment to temporary holding its argument");

        if let mir::Rvalue::Ref(_, mir::BorrowKind::Shared, peeking_at_place) = peek_rval {
            let loc = Location { block: bb, statement_index };
            let flow_state = dataflow::state_for_location(loc, results.operator(), results, body);

            results.operator().peek_at(tcx, peeking_at_place, &flow_state, call);
        } else {
            let msg = "rustc_peek: argument expression \
                       must be immediate borrow of form `&expr`";
            tcx.sess.span_err(call.span, msg);
        }
    }
}

/// If `stmt` is an assignment where the LHS is the given local (with no projections), returns the
/// RHS of the assignment.
fn value_assigned_to_local<'a, 'tcx>(
    stmt: &'a mir::Statement<'tcx>,
    local: Local,
) -> Option<&'a mir::Rvalue<'tcx>> {
    if let mir::StatementKind::Assign(place, rvalue) = &stmt.kind {
        if let mir::Place::Base(mir::PlaceBase::Local(l)) = place {
            if local == *l {
                return Some(&*rvalue);
            }
        }
    }

    None
}

#[derive(Clone, Copy, Debug)]
pub struct PeekCall {
    arg: Local,
    span: Span,
}

impl PeekCall {
    fn from_terminator<'tcx>(
        tcx: TyCtxt<'tcx>,
        terminator: &mir::Terminator<'tcx>,
    ) -> Option<Self> {
        let span = terminator.source_info.span;
        if let mir::TerminatorKind::Call { func: mir::Operand::Constant(func), args, .. } =
            &terminator.kind
        {
            if let ty::FnDef(def_id, _) = func.ty.sty {
                let abi = tcx.fn_sig(def_id).abi();
                let name = tcx.item_name(def_id);
                if abi != Abi::RustIntrinsic || name != sym::rustc_peek {
                    return None;
                }

                assert_eq!(args.len(), 1);
                let arg = match args[0] {
                    | mir::Operand::Copy(mir::Place::Base(mir::PlaceBase::Local(local)))
                    | mir::Operand::Move(mir::Place::Base(mir::PlaceBase::Local(local)))
                    => local,

                    _ => {
                        tcx.sess.diagnostic().span_err(
                            span, "dataflow::sanity_check cannot feed a non-temp to rustc_peek.");
                        return None;
                    }
                };

                return Some(PeekCall {
                    arg,
                    span,
                });
            }
        }

        None
    }
}

pub trait RustcPeekAt<'tcx>: BitDenotation<'tcx> {
    fn peek_at(
        &self,
        tcx: TyCtxt<'tcx>,
        place: &mir::Place<'tcx>,
        flow_state: &BitSet<Self::Idx>,
        call: PeekCall,
    );
}

impl<'tcx, O> RustcPeekAt<'tcx> for O
    where O: BitDenotation<'tcx, Idx = MovePathIndex> + HasMoveData<'tcx>,
{
    fn peek_at(
        &self,
        tcx: TyCtxt<'tcx>,
        place: &mir::Place<'tcx>,
        flow_state: &BitSet<Self::Idx>,
        call: PeekCall,
    ) {
        match self.move_data().rev_lookup.find(place) {
            LookupResult::Exact(peek_mpi) => {
                let bit_state = flow_state.contains(peek_mpi);
                debug!("rustc_peek({:?} = &{:?}) bit_state: {}",
                       call.arg, place, bit_state);
                if !bit_state {
                    tcx.sess.span_err(call.span, "rustc_peek: bit not set");
                }
            }
            LookupResult::Parent(..) => {
                tcx.sess.span_err(call.span, "rustc_peek: argument untracked");
            }
        }
    }
}
