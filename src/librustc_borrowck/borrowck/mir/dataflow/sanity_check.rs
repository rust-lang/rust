// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::abi::{Abi};
use syntax::ast;
use syntax_pos::Span;

use rustc::ty::{self, TyCtxt};
use rustc::mir::repr::{self, Mir};
use rustc_data_structures::indexed_vec::Idx;

use super::super::gather_moves::{MovePathIndex};
use super::super::MoveDataParamEnv;
use super::BitDenotation;
use super::DataflowResults;

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
                                                flow_ctxt: &O::Ctxt,
                                                results: &DataflowResults<O>)
    where O: BitDenotation<Ctxt=MoveDataParamEnv<'tcx>, Idx=MovePathIndex>
{
    debug!("sanity_check_via_rustc_peek id: {:?}", id);
    // FIXME: this is not DRY. Figure out way to abstract this and
    // `dataflow::build_sets`. (But note it is doing non-standard
    // stuff, so such generalization may not be realistic.)

    for bb in mir.basic_blocks().indices() {
        each_block(tcx, mir, flow_ctxt, results, bb);
    }
}

fn each_block<'a, 'tcx, O>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                           mir: &Mir<'tcx>,
                           ctxt: &O::Ctxt,
                           results: &DataflowResults<O>,
                           bb: repr::BasicBlock) where
    O: BitDenotation<Ctxt=MoveDataParamEnv<'tcx>, Idx=MovePathIndex>
{
    let move_data = &ctxt.move_data;
    let repr::BasicBlockData { ref statements,
                               ref terminator,
                               is_cleanup: _ } = mir[bb];

    let (args, span) = match is_rustc_peek(tcx, terminator) {
        Some(args_and_span) => args_and_span,
        None => return,
    };
    assert!(args.len() == 1);
    let peek_arg_lval = match args[0] {
        repr::Operand::Consume(ref lval @ repr::Lvalue::Temp(_)) => {
            lval
        }
        repr::Operand::Consume(_) |
        repr::Operand::Constant(_) => {
            tcx.sess.diagnostic().span_err(
                span, "dataflow::sanity_check cannot feed a non-temp to rustc_peek.");
            return;
        }
    };

    let mut entry = results.0.sets.on_entry_set_for(bb.index()).to_owned();
    let mut gen = results.0.sets.gen_set_for(bb.index()).to_owned();
    let mut kill = results.0.sets.kill_set_for(bb.index()).to_owned();

    // Emulate effect of all statements in the block up to (but not
    // including) the borrow within `peek_arg_lval`. Do *not* include
    // call to `peek_arg_lval` itself (since we are peeking the state
    // of the argument at time immediate preceding Call to
    // `rustc_peek`).

    let mut sets = super::BlockSets { on_entry: &mut entry,
                                      gen_set: &mut gen,
                                      kill_set: &mut kill };

    for (j, stmt) in statements.iter().enumerate() {
        debug!("rustc_peek: ({:?},{}) {:?}", bb, j, stmt);
        let (lvalue, rvalue) = match stmt.kind {
            repr::StatementKind::Assign(ref lvalue, ref rvalue) => {
                (lvalue, rvalue)
            }
        };

        if lvalue == peek_arg_lval {
            if let repr::Rvalue::Ref(_,
                                     repr::BorrowKind::Shared,
                                     ref peeking_at_lval) = *rvalue {
                // Okay, our search is over.
                let peek_mpi = move_data.rev_lookup.find(peeking_at_lval);
                let bit_state = sets.on_entry.contains(&peek_mpi);
                debug!("rustc_peek({:?} = &{:?}) bit_state: {}",
                       lvalue, peeking_at_lval, bit_state);
                if !bit_state {
                    tcx.sess.span_err(span, &format!("rustc_peek: bit not set"));
                }
                return;
            } else {
                // Our search should have been over, but the input
                // does not match expectations of `rustc_peek` for
                // this sanity_check.
                let msg = &format!("rustc_peek: argument expression \
                                    must be immediate borrow of form `&expr`");
                tcx.sess.span_err(span, msg);
            }
        }

        let lhs_mpi = move_data.rev_lookup.find(lvalue);

        debug!("rustc_peek: computing effect on lvalue: {:?} ({:?}) in stmt: {:?}",
               lvalue, lhs_mpi, stmt);
        // reset GEN and KILL sets before emulating their effect.
        for e in sets.gen_set.words_mut() { *e = 0; }
        for e in sets.kill_set.words_mut() { *e = 0; }
        results.0.operator.statement_effect(ctxt, &mut sets, bb, j);
        sets.on_entry.union(sets.gen_set);
        sets.on_entry.subtract(sets.kill_set);
    }

    tcx.sess.span_err(span, &format!("rustc_peek: MIR did not match \
                                      anticipated pattern; note that \
                                      rustc_peek expects input of \
                                      form `&expr`"));
}

fn is_rustc_peek<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                           terminator: &'a Option<repr::Terminator<'tcx>>)
                           -> Option<(&'a [repr::Operand<'tcx>], Span)> {
    if let Some(repr::Terminator { ref kind, source_info, .. }) = *terminator {
        if let repr::TerminatorKind::Call { func: ref oper, ref args, .. } = *kind
        {
            if let repr::Operand::Constant(ref func) = *oper
            {
                if let ty::TyFnDef(def_id, _, &ty::BareFnTy { abi, .. }) = func.ty.sty
                {
                    let name = tcx.item_name(def_id);
                    if abi == Abi::RustIntrinsic || abi == Abi::PlatformIntrinsic {
                        if name.as_str() == "rustc_peek" {
                            return Some((args, source_info.span));
                        }
                    }
                }
            }
        }
    }
    return None;
}
