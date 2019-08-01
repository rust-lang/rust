use rustc_target::spec::abi::{Abi};
use syntax::ast;
use syntax::symbol::sym;
use syntax_pos::Span;

use rustc::ty::{self, TyCtxt, Ty};
use rustc::hir::def_id::DefId;
use rustc::mir::{self, Body, Location, Local};
use rustc_data_structures::bit_set::BitSet;
use crate::transform::{MirPass, MirSource};

use crate::dataflow::{self, do_dataflow, DebugFormatted};
use crate::dataflow::MoveDataParamEnv;
use crate::dataflow::BitDenotation;
use crate::dataflow::DataflowResults;
use crate::dataflow::HaveBeenBorrowedLocals;
use crate::dataflow::{ReachingDefinitions, UseDefChain};
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

        let flow_borrowed_locals =
            do_dataflow(tcx, body, def_id, &attributes, &dead_unwinds,
                        HaveBeenBorrowedLocals::new(body),
                        |_bd, i| DebugFormatted::new(&i));
        let flow_reaching_defs =
            do_dataflow(tcx, body, def_id, &attributes, &dead_unwinds,
                        ReachingDefinitions::new(tcx, body, tcx.param_env(def_id)),
                        |bd, i| DebugFormatted::new(&bd.get(i).location));
        let flow_use_def_chain =
            UseDefChain::new(body, &flow_reaching_defs, &flow_borrowed_locals);

        if has_rustc_mir_with(&attributes, sym::rustc_peek_maybe_init).is_some() {
            sanity_check_via_rustc_peek(tcx, body, def_id, &attributes, &flow_inits);
        }
        if has_rustc_mir_with(&attributes, sym::rustc_peek_maybe_uninit).is_some() {
            sanity_check_via_rustc_peek(tcx, body, def_id, &attributes, &flow_uninits);
        }
        if has_rustc_mir_with(&attributes, sym::rustc_peek_definite_init).is_some() {
            sanity_check_via_rustc_peek(tcx, body, def_id, &attributes, &flow_def_inits);
        }
        if has_rustc_mir_with(&attributes, sym::rustc_peek_use_def_chain).is_some() {
            sanity_check_via_rustc_peek(tcx, body, def_id, &attributes, &flow_use_def_chain);
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
    results: &O,
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

        match (call.kind, peek_rval) {
            | (PeekCallKind::ByRef, mir::Rvalue::Ref(_, _, peek_at))
            | (PeekCallKind::ByVal, mir::Rvalue::Use(mir::Operand::Move(peek_at)))
            | (PeekCallKind::ByVal, mir::Rvalue::Use(mir::Operand::Copy(peek_at)))
            => {
                let loc = Location { block: bb, statement_index };
                results.peek_at(tcx, body, peek_at, loc, call);
            }

            _ => {
                let msg = "rustc_peek: argument expression \
                           must be either `place` or `&place`";
                tcx.sess.span_err(call.span, msg);
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
    if let mir::StatementKind::Assign(place, rvalue) = &stmt.kind {
        if let mir::Place { base: mir::PlaceBase::Local(l), projection: None } = place {
            if local == *l {
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
        if let ty::Ref(_, _, _) = arg.sty {
            PeekCallKind::ByRef
        } else {
            PeekCallKind::ByVal
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
        let span = terminator.source_info.span;
        if let mir::TerminatorKind::Call { func: mir::Operand::Constant(func), args, .. } =
            &terminator.kind
        {
            if let ty::FnDef(def_id, substs) = func.ty.sty {
                let sig = tcx.fn_sig(def_id);
                let name = tcx.item_name(def_id);
                if sig.abi() != Abi::RustIntrinsic || name != sym::rustc_peek {
                    return None;
                }

                assert_eq!(args.len(), 1);
                let kind = PeekCallKind::from_arg_ty(substs.type_at(0));
                if let mir::Operand::Copy(place) | mir::Operand::Move(place) = &args[0] {
                    if let mir::Place {
                        base: mir::PlaceBase::Local(local),
                        projection: None
                    } = *place {
                        return Some(PeekCall {
                            arg: local,
                            kind,
                            span,
                        });
                    }
                }

                tcx.sess.diagnostic().span_err(
                    span, "dataflow::sanity_check cannot feed a non-temp to rustc_peek.");
                return None;
            }
        }

        None
    }
}

pub trait RustcPeekAt<'tcx> {
    fn peek_at(
        &self,
        tcx: TyCtxt<'tcx>,
        body: &mir::Body<'tcx>,
        place: &mir::Place<'tcx>,
        location: Location,
        call: PeekCall,
    );
}

impl<'tcx, O> RustcPeekAt<'tcx> for DataflowResults<'tcx, O>
    where O: BitDenotation<'tcx, Idx = MovePathIndex> + HasMoveData<'tcx>,
{
    fn peek_at(
        &self,
        tcx: TyCtxt<'tcx>,
        body: &mir::Body<'tcx>,
        place: &mir::Place<'tcx>,
        location: Location,
        call: PeekCall,
    ) {
        let operator = self.operator();
        let flow_state = dataflow::state_for_location(location, operator, self, body);

        match operator.move_data().rev_lookup.find(place.as_ref()) {
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

impl<'tcx> RustcPeekAt<'tcx> for UseDefChain<'_, 'tcx> {
    fn peek_at(
        &self,
        tcx: TyCtxt<'tcx>,
        body: &mir::Body<'tcx>,
        place: &mir::Place<'tcx>,
        location: Location,
        call: PeekCall,
    ) {

        let base_local = place
            .base_direct()
            .expect("Deref in argument to `rustc_peek`")
            .local()
            .expect("Argument to `rustc_peek` must be a local variable");

        let mut defs: Vec<_> = self
            .defs_for_use(base_local, location)
            .map(|def| {
                let span = def
                    .location
                    .map(|loc| {
                        let block = &body.basic_blocks()[loc.block];
                        block.statements
                            .get(loc.statement_index)
                            .map(|stmt| stmt.source_info)
                            .unwrap_or(block.terminator().source_info)
                            .span
                    })
                    .unwrap_or_else(|| {
                        // `def` represents the value of a parameter on function entry.
                        let local = def.kind.direct().unwrap();
                        body.local_decls[local].source_info.span
                    });

                let src = tcx.sess.source_map();
                let snippet = src.span_to_snippet(span).unwrap();
                let line_index = src.span_to_lines(span).unwrap().lines[0].line_index;
                let line_no = line_index + 1;

                (line_no, snippet)
            })
            .collect();

        defs.sort_by_key(|(line_no, _)| *line_no);
        let defs: Vec<_> = defs.into_iter()
            .map(|(line_no, snippet)| format!("{}: \"{}\"", line_no, snippet))
            .collect();

        let msg = format!("rustc_peek: [{}]", defs.join(", "));
        tcx.sess.span_err(call.span, &msg);
    }
}
