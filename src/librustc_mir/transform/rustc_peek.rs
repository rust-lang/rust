// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_target::spec::abi::{Abi};
use syntax::ast;
use syntax_pos::Span;

use rustc::ty::{self, TyCtxt};
use rustc::mir::{self, Mir, Location};
use rustc_data_structures::bit_set::BitSet;
use transform::{MirPass, MirSource};

use dataflow::{do_dataflow, DebugFormatted};
use dataflow::MoveDataParamEnv;
use dataflow::BitDenotation;
use dataflow::DataflowResults;
use dataflow::{DefinitelyInitializedPlaces, MaybeInitializedPlaces, MaybeUninitializedPlaces};
use dataflow::move_paths::{MovePathIndex, LookupResult};
use dataflow::move_paths::{HasMoveData, MoveData};
use dataflow;

use dataflow::has_rustc_mir_with;

pub struct SanityCheck;

impl MirPass for SanityCheck {
    fn run_pass<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          src: MirSource, mir: &mut Mir<'tcx>) {
        let def_id = src.def_id;
        let id = tcx.hir.as_local_node_id(def_id).unwrap();
        if !tcx.has_attr(def_id, "rustc_mir") {
            debug!("skipping rustc_peek::SanityCheck on {}", tcx.item_path_str(def_id));
            return;
        } else {
            debug!("running rustc_peek::SanityCheck on {}", tcx.item_path_str(def_id));
        }

        let attributes = tcx.get_attrs(def_id);
        let param_env = tcx.param_env(def_id);
        let move_data = MoveData::gather_moves(mir, tcx).unwrap();
        let mdpe = MoveDataParamEnv { move_data: move_data, param_env: param_env };
        let dead_unwinds = BitSet::new_empty(mir.basic_blocks().len());
        let flow_inits =
            do_dataflow(tcx, mir, id, &attributes, &dead_unwinds,
                        MaybeInitializedPlaces::new(tcx, mir, &mdpe),
                        |bd, i| DebugFormatted::new(&bd.move_data().move_paths[i]));
        let flow_uninits =
            do_dataflow(tcx, mir, id, &attributes, &dead_unwinds,
                        MaybeUninitializedPlaces::new(tcx, mir, &mdpe),
                        |bd, i| DebugFormatted::new(&bd.move_data().move_paths[i]));
        let flow_def_inits =
            do_dataflow(tcx, mir, id, &attributes, &dead_unwinds,
                        DefinitelyInitializedPlaces::new(tcx, mir, &mdpe),
                        |bd, i| DebugFormatted::new(&bd.move_data().move_paths[i]));

        if has_rustc_mir_with(&attributes, "rustc_peek_maybe_init").is_some() {
            sanity_check_via_rustc_peek(tcx, mir, id, &attributes, &flow_inits);
        }
        if has_rustc_mir_with(&attributes, "rustc_peek_maybe_uninit").is_some() {
            sanity_check_via_rustc_peek(tcx, mir, id, &attributes, &flow_uninits);
        }
        if has_rustc_mir_with(&attributes, "rustc_peek_definite_init").is_some() {
            sanity_check_via_rustc_peek(tcx, mir, id, &attributes, &flow_def_inits);
        }
        if has_rustc_mir_with(&attributes, "stop_after_dataflow").is_some() {
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
pub fn sanity_check_via_rustc_peek<'a, 'tcx, O>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                                mir: &Mir<'tcx>,
                                                id: ast::NodeId,
                                                _attributes: &[ast::Attribute],
                                                results: &DataflowResults<O>)
    where O: BitDenotation<Idx=MovePathIndex> + HasMoveData<'tcx>
{
    debug!("sanity_check_via_rustc_peek id: {:?}", id);
    // FIXME: this is not DRY. Figure out way to abstract this and
    // `dataflow::build_sets`. (But note it is doing non-standard
    // stuff, so such generalization may not be realistic.)

    for bb in mir.basic_blocks().indices() {
        each_block(tcx, mir, results, bb);
    }
}

fn each_block<'a, 'tcx, O>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                           mir: &Mir<'tcx>,
                           results: &DataflowResults<O>,
                           bb: mir::BasicBlock) where
    O: BitDenotation<Idx=MovePathIndex> + HasMoveData<'tcx>
{
    let move_data = results.0.operator.move_data();
    let mir::BasicBlockData { ref statements, ref terminator, is_cleanup: _ } = mir[bb];

    let (args, span) = match is_rustc_peek(tcx, terminator) {
        Some(args_and_span) => args_and_span,
        None => return,
    };
    assert!(args.len() == 1);
    let peek_arg_place = match args[0] {
        mir::Operand::Copy(ref place @ mir::Place::Local(_)) |
        mir::Operand::Move(ref place @ mir::Place::Local(_)) => Some(place),
        _ => None,
    };

    let peek_arg_place = match peek_arg_place {
        Some(arg) => arg,
        None => {
            tcx.sess.diagnostic().span_err(
                span, "dataflow::sanity_check cannot feed a non-temp to rustc_peek.");
            return;
        }
    };

    let mut on_entry = results.0.sets.on_entry_set_for(bb.index()).to_owned();
    let mut gen_set = results.0.sets.gen_set_for(bb.index()).clone();
    let mut kill_set = results.0.sets.kill_set_for(bb.index()).clone();

    // Emulate effect of all statements in the block up to (but not
    // including) the borrow within `peek_arg_place`. Do *not* include
    // call to `peek_arg_place` itself (since we are peeking the state
    // of the argument at time immediate preceding Call to
    // `rustc_peek`).

    let mut sets = dataflow::BlockSets { on_entry: &mut on_entry,
                                         gen_set: &mut gen_set,
                                         kill_set: &mut kill_set };

    for (j, stmt) in statements.iter().enumerate() {
        debug!("rustc_peek: ({:?},{}) {:?}", bb, j, stmt);
        let (place, rvalue) = match stmt.kind {
            mir::StatementKind::Assign(ref place, ref rvalue) => {
                (place, rvalue)
            }
            mir::StatementKind::FakeRead(..) |
            mir::StatementKind::StorageLive(_) |
            mir::StatementKind::StorageDead(_) |
            mir::StatementKind::InlineAsm { .. } |
            mir::StatementKind::Retag { .. } |
            mir::StatementKind::EscapeToRaw { .. } |
            mir::StatementKind::AscribeUserType(..) |
            mir::StatementKind::Nop => continue,
            mir::StatementKind::SetDiscriminant{ .. } =>
                span_bug!(stmt.source_info.span,
                          "sanity_check should run before Deaggregator inserts SetDiscriminant"),
        };

        if place == peek_arg_place {
            if let mir::Rvalue::Ref(_, mir::BorrowKind::Shared, ref peeking_at_place) = **rvalue {
                // Okay, our search is over.
                match move_data.rev_lookup.find(peeking_at_place) {
                    LookupResult::Exact(peek_mpi) => {
                        let bit_state = sets.on_entry.contains(peek_mpi);
                        debug!("rustc_peek({:?} = &{:?}) bit_state: {}",
                               place, peeking_at_place, bit_state);
                        if !bit_state {
                            tcx.sess.span_err(span, "rustc_peek: bit not set");
                        }
                    }
                    LookupResult::Parent(..) => {
                        tcx.sess.span_err(span, "rustc_peek: argument untracked");
                    }
                }
                return;
            } else {
                // Our search should have been over, but the input
                // does not match expectations of `rustc_peek` for
                // this sanity_check.
                let msg = "rustc_peek: argument expression \
                           must be immediate borrow of form `&expr`";
                tcx.sess.span_err(span, msg);
            }
        }

        let lhs_mpi = move_data.rev_lookup.find(place);

        debug!("rustc_peek: computing effect on place: {:?} ({:?}) in stmt: {:?}",
               place, lhs_mpi, stmt);
        // reset GEN and KILL sets before emulating their effect.
        sets.gen_set.clear();
        sets.kill_set.clear();
        results.0.operator.before_statement_effect(
            &mut sets, Location { block: bb, statement_index: j });
        results.0.operator.statement_effect(
            &mut sets, Location { block: bb, statement_index: j });
        sets.on_entry.union(sets.gen_set);
        sets.on_entry.subtract(sets.kill_set);
    }

    results.0.operator.before_terminator_effect(
        &mut sets,
        Location { block: bb, statement_index: statements.len() });

    tcx.sess.span_err(span, &format!("rustc_peek: MIR did not match \
                                      anticipated pattern; note that \
                                      rustc_peek expects input of \
                                      form `&expr`"));
}

fn is_rustc_peek<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                           terminator: &'a Option<mir::Terminator<'tcx>>)
                           -> Option<(&'a [mir::Operand<'tcx>], Span)> {
    if let Some(mir::Terminator { ref kind, source_info, .. }) = *terminator {
        if let mir::TerminatorKind::Call { func: ref oper, ref args, .. } = *kind {
            if let mir::Operand::Constant(ref func) = *oper {
                if let ty::FnDef(def_id, _) = func.ty.sty {
                    let abi = tcx.fn_sig(def_id).abi();
                    let name = tcx.item_name(def_id);
                    if abi == Abi::RustIntrinsic &&  name == "rustc_peek" {
                        return Some((args, source_info.span));
                    }
                }
            }
        }
    }
    return None;
}
