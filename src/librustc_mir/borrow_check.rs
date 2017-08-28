// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This query borrow-checks the MIR to (further) ensure it is not broken.

use rustc::hir::def_id::{DefId};
use rustc::infer::{InferCtxt};
use rustc::ty::{self, TyCtxt, ParamEnv};
use rustc::ty::maps::Providers;
use rustc::mir::{AssertMessage, BasicBlock, BorrowKind, Location, Lvalue};
use rustc::mir::{Mir, Mutability, Operand, Projection, ProjectionElem, Rvalue};
use rustc::mir::{Statement, StatementKind, Terminator, TerminatorKind};
use rustc::mir::transform::{MirSource};

use rustc_data_structures::indexed_set::{self, IdxSetBuf};
use rustc_data_structures::indexed_vec::{Idx};

use syntax::ast::{self};
use syntax_pos::{DUMMY_SP, Span};

use dataflow::{do_dataflow};
use dataflow::{MoveDataParamEnv};
use dataflow::{BitDenotation, BlockSets, DataflowResults, DataflowResultsConsumer};
use dataflow::{MaybeInitializedLvals, MaybeUninitializedLvals};
use dataflow::{Borrows, BorrowData, BorrowIndex};
use dataflow::move_paths::{HasMoveData, MoveData, MovePathIndex, LookupResult};
use util::borrowck_errors::{BorrowckErrors, Origin};

use self::MutateMode::{JustWrite, WriteAndRead};
use self::ConsumeKind::{Consume};


pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        mir_borrowck,
        ..*providers
    };
}

fn mir_borrowck<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) {
    let mir = tcx.mir_validated(def_id);
    let src = MirSource::from_local_def_id(tcx, def_id);
    debug!("run query mir_borrowck: {}", tcx.node_path_str(src.item_id()));

    let mir: &Mir<'tcx> = &mir.borrow();
    if !tcx.has_attr(def_id, "rustc_mir_borrowck") && !tcx.sess.opts.debugging_opts.borrowck_mir {
        return;
    }

    let id = src.item_id();
    let attributes = tcx.get_attrs(def_id);
    let param_env = tcx.param_env(def_id);
    tcx.infer_ctxt().enter(|_infcx| {

        let move_data = MoveData::gather_moves(mir, tcx, param_env);
        let mdpe = MoveDataParamEnv { move_data: move_data, param_env: param_env };
        let dead_unwinds = IdxSetBuf::new_empty(mir.basic_blocks().len());
        let flow_borrows = do_dataflow(tcx, mir, id, &attributes, &dead_unwinds,
                                       Borrows::new(tcx, mir),
                                       |bd, i| bd.location(i));
        let flow_inits = do_dataflow(tcx, mir, id, &attributes, &dead_unwinds,
                                     MaybeInitializedLvals::new(tcx, mir, &mdpe),
                                     |bd, i| &bd.move_data().move_paths[i]);
        let flow_uninits = do_dataflow(tcx, mir, id, &attributes, &dead_unwinds,
                                       MaybeUninitializedLvals::new(tcx, mir, &mdpe),
                                       |bd, i| &bd.move_data().move_paths[i]);

        let mut mbcx = MirBorrowckCtxt {
            tcx: tcx,
            mir: mir,
            node_id: id,
            move_data: &mdpe.move_data,
            param_env: param_env,
            fake_infer_ctxt: &_infcx,
        };

        let mut state = InProgress::new(flow_borrows,
                                        flow_inits,
                                        flow_uninits);

        mbcx.analyze_results(&mut state); // entry point for DataflowResultsConsumer
    });

    debug!("mir_borrowck done");
}

#[allow(dead_code)]
pub struct MirBorrowckCtxt<'c, 'b, 'a: 'b+'c, 'gcx: 'a+'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'gcx>,
    mir: &'b Mir<'gcx>,
    node_id: ast::NodeId,
    move_data: &'b MoveData<'gcx>,
    param_env: ParamEnv<'tcx>,
    fake_infer_ctxt: &'c InferCtxt<'c, 'gcx, 'tcx>,
}

// (forced to be `pub` due to its use as an associated type below.)
pub struct InProgress<'b, 'tcx: 'b> {
    borrows: FlowInProgress<Borrows<'b, 'tcx>>,
    inits: FlowInProgress<MaybeInitializedLvals<'b, 'tcx>>,
    uninits: FlowInProgress<MaybeUninitializedLvals<'b, 'tcx>>,
}

struct FlowInProgress<BD> where BD: BitDenotation {
    base_results: DataflowResults<BD>,
    curr_state: IdxSetBuf<BD::Idx>,
    stmt_gen: IdxSetBuf<BD::Idx>,
    stmt_kill: IdxSetBuf<BD::Idx>,
}

// Check that:
// 1. assignments are always made to mutable locations (FIXME: does that still really go here?)
// 2. loans made in overlapping scopes do not conflict
// 3. assignments do not affect things loaned out as immutable
// 4. moves do not affect things loaned out in any way
impl<'c, 'b, 'a: 'b+'c, 'gcx, 'tcx: 'a> DataflowResultsConsumer<'b, 'gcx>
    for MirBorrowckCtxt<'c, 'b, 'a, 'gcx, 'tcx>
{
    type FlowState = InProgress<'b, 'gcx>;

    fn mir(&self) -> &'b Mir<'gcx> { self.mir }

    fn reset_to_entry_of(&mut self, bb: BasicBlock, flow_state: &mut Self::FlowState) {
        flow_state.each_flow(|b| b.reset_to_entry_of(bb),
                             |i| i.reset_to_entry_of(bb),
                             |u| u.reset_to_entry_of(bb));
    }

    fn reconstruct_statement_effect(&mut self,
                                    location: Location,
                                    flow_state: &mut Self::FlowState) {
        flow_state.each_flow(|b| b.reconstruct_statement_effect(location),
                             |i| i.reconstruct_statement_effect(location),
                             |u| u.reconstruct_statement_effect(location));
    }

    fn apply_local_effect(&mut self,
                          _location: Location,
                          flow_state: &mut Self::FlowState) {
        flow_state.each_flow(|b| b.apply_local_effect(),
                             |i| i.apply_local_effect(),
                             |u| u.apply_local_effect());
    }

    fn reconstruct_terminator_effect(&mut self,
                                     location: Location,
                                     flow_state: &mut Self::FlowState) {
        flow_state.each_flow(|b| b.reconstruct_terminator_effect(location),
                             |i| i.reconstruct_terminator_effect(location),
                             |u| u.reconstruct_terminator_effect(location));
    }

    fn visit_block_entry(&mut self,
                         bb: BasicBlock,
                         flow_state: &Self::FlowState) {
        let summary = flow_state.summary();
        debug!("MirBorrowckCtxt::process_block({:?}): {}", bb, summary);
    }

    fn visit_statement_entry(&mut self,
                             location: Location,
                             stmt: &Statement<'gcx>,
                             flow_state: &Self::FlowState) {
        let summary = flow_state.summary();
        debug!("MirBorrowckCtxt::process_statement({:?}, {:?}): {}", location, stmt, summary);
        let span = stmt.source_info.span;
        match stmt.kind {
            StatementKind::Assign(ref lhs, ref rhs) => {
                self.mutate_lvalue(ContextKind::AssignLhs.new(location),
                                   (lhs, span), JustWrite, flow_state);
                self.consume_rvalue(ContextKind::AssignRhs.new(location),
                                    (rhs, span), location, flow_state);
            }
            StatementKind::SetDiscriminant { ref lvalue, variant_index: _ } => {
                self.mutate_lvalue(ContextKind::SetDiscrim.new(location),
                                   (lvalue, span), JustWrite, flow_state);
            }
            StatementKind::InlineAsm { ref asm, ref outputs, ref inputs } => {
                for (o, output) in asm.outputs.iter().zip(outputs) {
                    if o.is_indirect {
                        self.consume_lvalue(ContextKind::InlineAsm.new(location),
                                            Consume,
                                            (output, span),
                                            flow_state);
                    } else {
                        self.mutate_lvalue(ContextKind::InlineAsm.new(location),
                                           (output, span),
                                           if o.is_rw { WriteAndRead } else { JustWrite },
                                           flow_state);
                    }
                }
                for input in inputs {
                    self.consume_operand(ContextKind::InlineAsm.new(location),
                                         Consume,
                                         (input, span), flow_state);
                }
            }
            StatementKind::EndRegion(ref _rgn) => {
                // ignored when consuming results (update to
                // flow_state already handled).
            }
            StatementKind::Nop |
            StatementKind::Validate(..) |
            StatementKind::StorageLive(..) => {
                // ignored by borrowck
            }

            StatementKind::StorageDead(ref lvalue) => {
                // causes non-drop values to be dropped.
                self.consume_lvalue(ContextKind::StorageDead.new(location),
                                    ConsumeKind::Consume,
                                    (lvalue, span),
                                    flow_state)
            }
        }
    }

    fn visit_terminator_entry(&mut self,
                              location: Location,
                              term: &Terminator<'gcx>,
                              flow_state: &Self::FlowState) {
        let loc = location;
        let summary = flow_state.summary();
        debug!("MirBorrowckCtxt::process_terminator({:?}, {:?}): {}", location, term, summary);
        let span = term.source_info.span;
        match term.kind {
            TerminatorKind::SwitchInt { ref discr, switch_ty: _, values: _, targets: _ } => {
                self.consume_operand(ContextKind::SwitchInt.new(loc),
                                     Consume,
                                     (discr, span), flow_state);
            }
            TerminatorKind::Drop { location: ref drop_lvalue, target: _, unwind: _ } => {
                self.consume_lvalue(ContextKind::Drop.new(loc),
                                    ConsumeKind::Drop,
                                    (drop_lvalue, span), flow_state);
            }
            TerminatorKind::DropAndReplace { location: ref drop_lvalue,
                                             value: ref new_value,
                                             target: _,
                                             unwind: _ } => {
                self.mutate_lvalue(ContextKind::DropAndReplace.new(loc),
                                   (drop_lvalue, span), JustWrite, flow_state);
                self.consume_operand(ContextKind::DropAndReplace.new(loc),
                                     ConsumeKind::Drop,
                                     (new_value, span), flow_state);
            }
            TerminatorKind::Call { ref func, ref args, ref destination, cleanup: _ } => {
                self.consume_operand(ContextKind::CallOperator.new(loc),
                                     Consume,
                                     (func, span), flow_state);
                for arg in args {
                    self.consume_operand(ContextKind::CallOperand.new(loc),
                                         Consume,
                                         (arg, span), flow_state);
                }
                if let Some((ref dest, _/*bb*/)) = *destination {
                    self.mutate_lvalue(ContextKind::CallDest.new(loc),
                                       (dest, span), JustWrite, flow_state);
                }
            }
            TerminatorKind::Assert { ref cond, expected: _, ref msg, target: _, cleanup: _ } => {
                self.consume_operand(ContextKind::Assert.new(loc),
                                     Consume,
                                     (cond, span), flow_state);
                match *msg {
                    AssertMessage::BoundsCheck { ref len, ref index } => {
                        self.consume_operand(ContextKind::Assert.new(loc),
                                             Consume,
                                             (len, span), flow_state);
                        self.consume_operand(ContextKind::Assert.new(loc),
                                             Consume,
                                             (index, span), flow_state);
                    }
                    AssertMessage::Math(_/*const_math_err*/) => {}
                }
            }

            TerminatorKind::Goto { target: _ } |
            TerminatorKind::Resume |
            TerminatorKind::Return |
            TerminatorKind::Unreachable => {
                // no data used, thus irrelevant to borrowck
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum MutateMode { JustWrite, WriteAndRead }

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum ConsumeKind { Drop, Consume }

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Control { Continue, Break }

impl<'c, 'b, 'a: 'b+'c, 'gcx, 'tcx: 'a> MirBorrowckCtxt<'c, 'b, 'a, 'gcx, 'tcx> {
    fn mutate_lvalue(&mut self,
                     context: Context,
                     lvalue_span: (&Lvalue<'gcx>, Span),
                     mode: MutateMode,
                     flow_state: &InProgress<'b, 'gcx>) {
        // Write of P[i] or *P, or WriteAndRead of any P, requires P init'd.
        match mode {
            MutateMode::WriteAndRead => {
                self.check_if_path_is_moved(context, lvalue_span, flow_state);
            }
            MutateMode::JustWrite => {
                self.check_if_assigned_path_is_moved(context, lvalue_span, flow_state);
            }
        }

        // check we don't invalidate any outstanding loans
        self.each_borrow_involving_path(context,
                                        lvalue_span.0, flow_state, |this, _index, _data| {
                                            this.report_illegal_mutation_of_borrowed(context,
                                                                                     lvalue_span);
                                            Control::Break
                                        });

        // check for reassignments to immutable local variables
        self.check_if_reassignment_to_immutable_state(context, lvalue_span, flow_state);
    }

    fn consume_rvalue(&mut self,
                      context: Context,
                      (rvalue, span): (&Rvalue<'gcx>, Span),
                      location: Location,
                      flow_state: &InProgress<'b, 'gcx>) {
        match *rvalue {
            Rvalue::Ref(_/*rgn*/, bk, ref lvalue) => {
                self.borrow(context, location, bk, (lvalue, span), flow_state)
            }

            Rvalue::Use(ref operand) |
            Rvalue::Repeat(ref operand, _) |
            Rvalue::UnaryOp(_/*un_op*/, ref operand) |
            Rvalue::Cast(_/*cast_kind*/, ref operand, _/*ty*/) => {
                self.consume_operand(context, Consume, (operand, span), flow_state)
            }

            Rvalue::Len(ref lvalue) |
            Rvalue::Discriminant(ref lvalue) => {
                // len(_)/discriminant(_) merely read, not consume.
                self.check_if_path_is_moved(context, (lvalue, span), flow_state);
            }

            Rvalue::BinaryOp(_bin_op, ref operand1, ref operand2) |
            Rvalue::CheckedBinaryOp(_bin_op, ref operand1, ref operand2) => {
                self.consume_operand(context, Consume, (operand1, span), flow_state);
                self.consume_operand(context, Consume, (operand2, span), flow_state);
            }

            Rvalue::NullaryOp(_op, _ty) => {
                // nullary ops take no dynamic input; no borrowck effect.
                //
                // FIXME: is above actually true? Do we want to track
                // the fact that uninitialized data can be created via
                // `NullOp::Box`?
            }

            Rvalue::Aggregate(ref _aggregate_kind, ref operands) => {
                for operand in operands {
                    self.consume_operand(context, Consume, (operand, span), flow_state);
                }
            }
        }
    }

    fn consume_operand(&mut self,
                       context: Context,
                       consume_via_drop: ConsumeKind,
                       (operand, span): (&Operand<'gcx>, Span),
                       flow_state: &InProgress<'b, 'gcx>) {
        match *operand {
            Operand::Consume(ref lvalue) =>
                self.consume_lvalue(context, consume_via_drop, (lvalue, span), flow_state),
            Operand::Constant(_) => {}
        }
    }

    fn consume_lvalue(&mut self,
                      context: Context,
                      consume_via_drop: ConsumeKind,
                      lvalue_span: (&Lvalue<'gcx>, Span),
                      flow_state: &InProgress<'b, 'gcx>) {
        let lvalue = lvalue_span.0;
        let ty = lvalue.ty(self.mir, self.tcx).to_ty(self.tcx);
        let moves_by_default =
            self.fake_infer_ctxt.type_moves_by_default(self.param_env, ty, DUMMY_SP);
        if moves_by_default {
            // move of lvalue: check if this is move of already borrowed path
            self.each_borrow_involving_path(
                context, lvalue_span.0, flow_state, |this, _idx, borrow| {
                    if !borrow.compatible_with(BorrowKind::Mut) {
                        this.report_move_out_while_borrowed(context, lvalue_span);
                        Control::Break
                    } else {
                        Control::Continue
                    }
                });
        } else {
            // copy of lvalue: check if this is "copy of frozen path" (FIXME: see check_loans.rs)
            self.each_borrow_involving_path(
                context, lvalue_span.0, flow_state, |this, _idx, borrow| {
                    if !borrow.compatible_with(BorrowKind::Shared) {
                        this.report_use_while_mutably_borrowed(context, lvalue_span);
                        Control::Break
                    } else {
                        Control::Continue
                    }
                });
        }

        // Finally, check if path was already moved.
        match consume_via_drop {
            ConsumeKind::Drop => {
                // If path is merely being dropped, then we'll already
                // check the drop flag to see if it is moved (thus we
                // skip this check in that case).
            }
            ConsumeKind::Consume => {
                self.check_if_path_is_moved(context, lvalue_span, flow_state);
            }
        }
    }

    fn borrow(&mut self,
              context: Context,
              location: Location,
              bk: BorrowKind,
              lvalue_span: (&Lvalue<'gcx>, Span),
              flow_state: &InProgress<'b, 'gcx>) {
        debug!("borrow location: {:?} lvalue: {:?} span: {:?}",
               location, lvalue_span.0, lvalue_span.1);
        self.check_if_path_is_moved(context, lvalue_span, flow_state);
        self.check_for_conflicting_loans(context, location, bk, lvalue_span, flow_state);
    }
}

impl<'c, 'b, 'a: 'b+'c, 'gcx, 'tcx: 'a> MirBorrowckCtxt<'c, 'b, 'a, 'gcx, 'tcx> {
    fn check_if_reassignment_to_immutable_state(&mut self,
                                                context: Context,
                                                (lvalue, span): (&Lvalue<'gcx>, Span),
                                                flow_state: &InProgress<'b, 'gcx>) {
        let move_data = flow_state.inits.base_results.operator().move_data();

        // determine if this path has a non-mut owner (and thus needs checking).
        let mut l = lvalue;
        loop {
            match *l {
                Lvalue::Projection(ref proj) => {
                    l = &proj.base;
                    continue;
                }
                Lvalue::Local(local) => {
                    match self.mir.local_decls[local].mutability {
                        Mutability::Not => break, // needs check
                        Mutability::Mut => return,
                    }
                }
                Lvalue::Static(_) => {
                    // mutation of non-mut static is always illegal,
                    // independent of dataflow.
                    self.report_assignment_to_static(context, (lvalue, span));
                    return;
                }
            }
        }

        if let Some(mpi) = self.move_path_for_lvalue(context, move_data, lvalue) {
            if flow_state.inits.curr_state.contains(&mpi) {
                // may already be assigned before reaching this statement;
                // report error.
                self.report_illegal_reassignment(context, (lvalue, span));
            }
        }
    }

    fn check_if_path_is_moved(&mut self,
                              context: Context,
                              lvalue_span: (&Lvalue<'gcx>, Span),
                              flow_state: &InProgress<'b, 'gcx>) {
        // FIXME: analogous code in check_loans first maps `lvalue` to
        // its base_path ... but is that what we want here?
        let lvalue = self.base_path(lvalue_span.0);

        let maybe_uninits = &flow_state.uninits;
        let move_data = maybe_uninits.base_results.operator().move_data();
        if let Some(mpi) = self.move_path_for_lvalue(context, move_data, lvalue) {
            if maybe_uninits.curr_state.contains(&mpi) {
                // find and report move(s) that could cause this to be uninitialized
                self.report_use_of_moved(context, lvalue_span);
            } else {
                // sanity check: initialized on *some* path, right?
                assert!(flow_state.inits.curr_state.contains(&mpi));
            }
        }
    }

    fn move_path_for_lvalue(&mut self,
                            _context: Context,
                            move_data: &MoveData<'gcx>,
                            lvalue: &Lvalue<'gcx>)
                            -> Option<MovePathIndex>
    {
        // If returns None, then there is no move path corresponding
        // to a direct owner of `lvalue` (which means there is nothing
        // that borrowck tracks for its analysis).

        match move_data.rev_lookup.find(lvalue) {
            LookupResult::Parent(_) => None,
            LookupResult::Exact(mpi) => Some(mpi),
        }
    }

    fn check_if_assigned_path_is_moved(&mut self,
                                       context: Context,
                                       (lvalue, span): (&Lvalue<'gcx>, Span),
                                       flow_state: &InProgress<'b, 'gcx>) {
        // recur down lvalue; dispatch to check_if_path_is_moved when necessary
        let mut lvalue = lvalue;
        loop {
            match *lvalue {
                Lvalue::Local(_) | Lvalue::Static(_) => {
                    // assigning to `x` does not require `x` be initialized.
                    break;
                }
                Lvalue::Projection(ref proj) => {
                    let Projection { ref base, ref elem } = **proj;
                    match *elem {
                        ProjectionElem::Deref |
                        // assigning to *P requires `P` initialized.
                        ProjectionElem::Index(_/*operand*/) |
                        ProjectionElem::ConstantIndex { .. } |
                        // assigning to P[i] requires `P` initialized.
                        ProjectionElem::Downcast(_/*adt_def*/, _/*variant_idx*/) =>
                        // assigning to (P->variant) is okay if assigning to `P` is okay
                        //
                        // FIXME: is this true even if P is a adt with a dtor?
                        { }

                        ProjectionElem::Subslice { .. } => {
                            panic!("we dont allow assignments to subslices, context: {:?}",
                                   context);
                        }

                        ProjectionElem::Field(..) => {
                            // if type of `P` has a dtor, then
                            // assigning to `P.f` requires `P` itself
                            // be already initialized
                            let tcx = self.tcx;
                            match base.ty(self.mir, tcx).to_ty(tcx).sty {
                                ty::TyAdt(def, _) if def.has_dtor(tcx) => {

                                    // FIXME: analogous code in
                                    // check_loans.rs first maps
                                    // `base` to its base_path.

                                    self.check_if_path_is_moved(context,
                                                                (base, span), flow_state);

                                    // (base initialized; no need to
                                    // recur further)
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }

                    lvalue = base;
                    continue;
                }
            }
        }
    }

    fn check_for_conflicting_loans(&mut self,
                                   context: Context,
                                   _location: Location,
                                   _bk: BorrowKind,
                                   lvalue_span: (&Lvalue<'gcx>, Span),
                                   flow_state: &InProgress<'b, 'gcx>) {
        // NOTE FIXME: The analogous code in old borrowck
        // check_loans.rs is careful to iterate over every *issued*
        // loan, as opposed to just the in scope ones.
        //
        // (Or if you prefer, all the *other* iterations over loans
        // only consider loans that are in scope of some given
        // CodeExtent)
        //
        // The (currently skeletal) code here does not encode such a
        // distinction, which means it is almost certainly over
        // looking something.
        //
        // (It is probably going to reject code that should be
        // accepted, I suspect, by treated issued-but-out-of-scope
        // loans as issued-and-in-scope, and thus causing them to
        // interfere with other loans.)
        //
        // However, I just want to get something running, especially
        // since I am trying to move into new territory with NLL, so
        // lets get this going first, and then address the issued vs
        // in-scope distinction later.

        let state = &flow_state.borrows;
        let data = &state.base_results.operator().borrows();

        debug!("check_for_conflicting_loans location: {:?}", _location);

        // does any loan generated here conflict with a previously issued loan?
        let mut loans_generated = 0;
        for (g, gen) in state.elems_generated().map(|g| (g, &data[g])) {
            loans_generated += 1;
            for (i, issued) in state.elems_incoming().map(|i| (i, &data[i])) {
                debug!("check_for_conflicting_loans gen: {:?} issued: {:?} conflicts: {}",
                       (g, gen, self.base_path(&gen.lvalue),
                        self.restrictions(&gen.lvalue).collect::<Vec<_>>()),
                       (i, issued, self.base_path(&issued.lvalue),
                        self.restrictions(&issued.lvalue).collect::<Vec<_>>()),
                       self.conflicts_with(gen, issued));
                if self.conflicts_with(gen, issued) {
                    self.report_conflicting_borrow(context, lvalue_span, gen, issued);
                }
            }
        }

        // MIR statically ensures each statement gens *at most one*
        // loan; mutual conflict (within a statement) can't arise.
        //
        // As safe-guard, assert that above property actually holds.
        assert!(loans_generated <= 1);
    } }

impl<'c, 'b, 'a: 'b+'c, 'gcx, 'tcx: 'a> MirBorrowckCtxt<'c, 'b, 'a, 'gcx, 'tcx> {
    fn each_borrow_involving_path<F>(&mut self,
                                     _context: Context,
                                     lvalue: &Lvalue<'gcx>,
                                     flow_state: &InProgress<'b, 'gcx>,
                                     mut op: F)
        where F: FnMut(&mut Self, BorrowIndex, &BorrowData<'gcx>) -> Control
    {
        // FIXME: analogous code in check_loans first maps `lvalue` to
        // its base_path.

        let domain = flow_state.borrows.base_results.operator();
        let data = domain.borrows();

        // check for loan restricting path P being used. Accounts for
        // borrows of P, P.a.b, etc.
        for i in flow_state.borrows.elems_incoming() {
            // FIXME: check_loans.rs filtered this to "in scope"
            // loans; i.e. it took a scope S and checked that each
            // restriction's kill_scope was a superscope of S.
            let borrowed = &data[i];
            for restricted in self.restrictions(&borrowed.lvalue) {
                if restricted == lvalue {
                    let ctrl = op(self, i, borrowed);
                    if ctrl == Control::Break { return; }
                }
            }
        }

        // check for loans (not restrictions) on any base path.
        // e.g. Rejects `{ let x = &mut a.b; let y = a.b.c; }`,
        // since that moves out of borrowed path `a.b`.
        //
        // Limiting to loans (not restrictions) keeps this one
        // working: `{ let x = &mut a.b; let y = a.c; }`
        let mut cursor = lvalue;
        loop {
            // FIXME: check_loans.rs invoked `op` *before* cursor
            // shift here.  Might just work (and even avoid redundant
            // errors?) given code above?  But for now, I want to try
            // doing what I think is more "natural" check.
            for i in flow_state.borrows.elems_incoming() {
                let borrowed = &data[i];
                if borrowed.lvalue == *cursor {
                    let ctrl = op(self, i, borrowed);
                    if ctrl == Control::Break { return; }
                }
            }

            match *cursor {
                Lvalue::Local(_) | Lvalue::Static(_) => break,
                Lvalue::Projection(ref proj) => cursor = &proj.base,
            }
        }
    }
}

mod restrictions {
    use super::MirBorrowckCtxt;

    use rustc::hir;
    use rustc::ty::{self, TyCtxt};
    use rustc::mir::{Lvalue, Mir, Operand, ProjectionElem};

    pub(super) struct Restrictions<'c, 'tcx: 'c> {
        mir: &'c Mir<'tcx>,
        tcx: TyCtxt<'c, 'tcx, 'tcx>,
        lvalue_stack: Vec<&'c Lvalue<'tcx>>,
    }

    impl<'c, 'b, 'a: 'b+'c, 'gcx, 'tcx: 'a> MirBorrowckCtxt<'c, 'b, 'a, 'gcx, 'tcx> {
        pub(super) fn restrictions<'d>(&self,
                                       lvalue: &'d Lvalue<'gcx>)
                                       -> Restrictions<'d, 'gcx> where 'b: 'd
        {
            let lvalue_stack = if self.has_restrictions(lvalue) { vec![lvalue] } else { vec![] };
            Restrictions { lvalue_stack: lvalue_stack, mir: self.mir, tcx: self.tcx }
        }

        fn has_restrictions(&self, lvalue: &Lvalue<'gcx>) -> bool {
            let mut cursor = lvalue;
            loop {
                let proj = match *cursor {
                    Lvalue::Local(_) => return true,
                    Lvalue::Static(_) => return false,
                    Lvalue::Projection(ref proj) => proj,
                };
                match proj.elem {
                    ProjectionElem::Index(..) |
                    ProjectionElem::ConstantIndex { .. } |
                    ProjectionElem::Downcast(..) |
                    ProjectionElem::Subslice { .. } |
                    ProjectionElem::Field(_/*field*/, _/*ty*/) => {
                        cursor = &proj.base;
                        continue;
                    }
                    ProjectionElem::Deref => {
                        let ty = proj.base.ty(self.mir, self.tcx).to_ty(self.tcx);
                        match ty.sty {
                            ty::TyRawPtr(_) => {
                                return false;
                            }
                            ty::TyRef(_, ty::TypeAndMut { ty: _, mutbl: hir::MutImmutable }) => {
                                // FIXME: do I need to check validity of
                                // region here though? (I think the original
                                // check_loans code did, like readme says)
                                return false;
                            }
                            ty::TyRef(_, ty::TypeAndMut { ty: _, mutbl: hir::MutMutable }) => {
                                cursor = &proj.base;
                                continue;
                            }
                            ty::TyAdt(..) if ty.is_box() => {
                                cursor = &proj.base;
                                continue;
                            }
                            _ => {
                                panic!("unknown type fed to Projection Deref.");
                            }
                        }
                    }
                }
            }
        }
    }

    impl<'c, 'tcx> Iterator for Restrictions<'c, 'tcx> {
        type Item = &'c Lvalue<'tcx>;
        fn next(&mut self) -> Option<Self::Item> {
            'pop: loop {
                let lvalue = match self.lvalue_stack.pop() {
                    None => return None,
                    Some(lvalue) => lvalue,
                };

                // `lvalue` may not be a restriction itself, but may
                // hold one further down (e.g. we never return
                // downcasts here, but may return a base of a
                // downcast).
                //
                // Also, we need to enqueue any additional
                // subrestrictions that it implies, since we can only
                // return from from this call alone.

                let mut cursor = lvalue;
                'cursor: loop {
                    let proj = match *cursor {
                        Lvalue::Local(_) => return Some(cursor), // search yielded this leaf
                        Lvalue::Static(_) => continue 'pop, // fruitless leaf; try next on stack
                        Lvalue::Projection(ref proj) => proj,
                    };

                    match proj.elem {
                        ProjectionElem::Field(_/*field*/, _/*ty*/) => {
                            // FIXME: add union handling
                            self.lvalue_stack.push(&proj.base);
                            return Some(cursor);
                        }
                        ProjectionElem::Downcast(..) |
                        ProjectionElem::Subslice { .. } |
                        ProjectionElem::ConstantIndex { .. } |
                        ProjectionElem::Index(Operand::Constant(..)) => {
                            cursor = &proj.base;
                            continue 'cursor;
                        }
                        ProjectionElem::Index(Operand::Consume(ref index)) => {
                            self.lvalue_stack.push(index); // FIXME: did old borrowck do this?
                            cursor = &proj.base;
                            continue 'cursor;
                        }
                        ProjectionElem::Deref => {
                            // (handled below)
                        }
                    }

                    assert_eq!(proj.elem, ProjectionElem::Deref);

                    let ty = proj.base.ty(self.mir, self.tcx).to_ty(self.tcx);
                    match ty.sty {
                        ty::TyRawPtr(_) => {
                            // borrowck ignores raw ptrs; treat analogous to imm borrow
                            continue 'pop;
                        }
                        // R-Deref-Imm-Borrowed
                        ty::TyRef(_/*rgn*/, ty::TypeAndMut { ty: _, mutbl: hir::MutImmutable }) => {
                            // immutably-borrowed referents do not
                            // have recursively-implied restrictions
                            // (because preventing actions on `*LV`
                            // does nothing about aliases like `*LV1`)

                            // FIXME: do I need to check validity of
                            // `_r` here though? (I think the original
                            // check_loans code did, like the readme
                            // says)

                            // (And do I *really* not have to
                            // recursively process the `base` as a
                            // further search here? Leaving this `if
                            // false` here as a hint to look at this
                            // again later.
                            //
                            // Ah, it might be because the
                            // restrictions are distinct from the path
                            // substructure. Note that there is a
                            // separate loop over the path
                            // substructure in fn
                            // each_borrow_involving_path, for better
                            // or for worse.

                            if false {
                                cursor = &proj.base;
                                continue 'cursor;
                            } else {
                                continue 'pop;
                            }
                        }

                        // R-Deref-Mut-Borrowed
                        ty::TyRef(_/*rgn*/, ty::TypeAndMut { ty: _, mutbl: hir::MutMutable }) => {
                            // mutably-borrowed referents are
                            // themselves restricted.

                            // FIXME: do I need to check validity of
                            // `_r` here though? (I think the original
                            // check_loans code did, like the readme
                            // says)

                            // schedule base for future iteration.
                            self.lvalue_stack.push(&proj.base);
                            return Some(cursor); // search yielded interior node
                        }

                        // R-Deref-Send-Pointer
                        ty::TyAdt(..) if ty.is_box() => {
                            // borrowing interior of a box implies that
                            // its base can no longer be mutated (o/w box
                            // storage would be freed)
                            self.lvalue_stack.push(&proj.base);
                            return Some(cursor); // search yielded interior node
                        }

                        _ => panic!("unknown type fed to Projection Deref."),
                    }
                }
            }
        }
    }
}

impl<'c, 'b, 'a: 'b+'c, 'gcx, 'tcx: 'a> MirBorrowckCtxt<'c, 'b, 'a, 'gcx, 'tcx> {
    fn report_use_of_moved(&mut self,
                           _context: Context,
                           (lvalue, span): (&Lvalue, Span)) {
        let mut err = self.tcx.cannot_act_on_uninitialized_variable(
            span, "use", &self.describe_lvalue(lvalue), Origin::Mir);
        // FIXME: add span_label for use of uninitialized variable
        err.emit();
    }

    fn report_move_out_while_borrowed(&mut self,
                                      _context: Context,
                                      (lvalue, span): (&Lvalue, Span)) {
        let mut err = self.tcx.cannot_move_when_borrowed(
            span, &self.describe_lvalue(lvalue), Origin::Mir);
        // FIXME 1: add span_label for "borrow of `()` occurs here"
        // FIXME 2: add span_label for "move out of `{}` occurs here"
        err.emit();
    }

    fn report_use_while_mutably_borrowed(&mut self,
                                         _context: Context,
                                         (lvalue, span): (&Lvalue, Span)) {
        let mut err = self.tcx.cannot_use_when_mutably_borrowed(
            span, &self.describe_lvalue(lvalue), Origin::Mir);
        // FIXME 1: add span_label for "borrow of `()` occurs here"
        // FIXME 2: add span_label for "use of `{}` occurs here"
        err.emit();
    }

    fn report_conflicting_borrow(&mut self,
                                 _context: Context,
                                 (lvalue, span): (&Lvalue, Span),
                                 loan1: &BorrowData,
                                 loan2: &BorrowData) {
        // FIXME: obviously falsifiable. Generalize for non-eq lvalues later.
        assert_eq!(loan1.lvalue, loan2.lvalue);

        // FIXME: supply non-"" `opt_via` when appropriate
        let mut err = match (loan1.kind, "immutable", "mutable",
                             loan2.kind, "immutable", "mutable") {
            (BorrowKind::Shared, lft, _, BorrowKind::Mut, _, rgt) |
            (BorrowKind::Mut, _, lft, BorrowKind::Shared, rgt, _) |
            (BorrowKind::Mut, _, lft, BorrowKind::Mut, _, rgt) =>
                self.tcx.cannot_reborrow_already_borrowed(
                    span, &self.describe_lvalue(lvalue),
                    "", lft, "it", rgt, "", Origin::Mir),

            _ =>  self.tcx.cannot_mutably_borrow_multiply(
                span, &self.describe_lvalue(lvalue), "", Origin::Mir),
            // FIXME: add span labels for first and second mutable borrows, as well as
            // end point for first.
        };
        err.emit();
    }

    fn report_illegal_mutation_of_borrowed(&mut self, _: Context, (lvalue, span): (&Lvalue, Span)) {
        let mut err = self.tcx.cannot_assign_to_borrowed(
            span, &self.describe_lvalue(lvalue), Origin::Mir);
        // FIXME: add span labels for borrow and assignment points
        err.emit();
    }

    fn report_illegal_reassignment(&mut self, _context: Context, (lvalue, span): (&Lvalue, Span)) {
        let mut err = self.tcx.cannot_reassign_immutable(
            span, &self.describe_lvalue(lvalue), Origin::Mir);
        // FIXME: add span labels for borrow and assignment points
        err.emit();
    }

    fn report_assignment_to_static(&mut self, _context: Context, (lvalue, span): (&Lvalue, Span)) {
        let mut err = self.tcx.cannot_assign_static(
            span, &self.describe_lvalue(lvalue), Origin::Mir);
        // FIXME: add span labels for borrow and assignment points
        err.emit();
    }
}

impl<'c, 'b, 'a: 'b+'c, 'gcx, 'tcx: 'a> MirBorrowckCtxt<'c, 'b, 'a, 'gcx, 'tcx> {
    // End-user visible description of `lvalue`
    fn describe_lvalue(&self, lvalue: &Lvalue) -> String {
        let mut buf = String::new();
        self.append_lvalue_to_string(lvalue, &mut buf);
        buf
    }

    // Appends end-user visible description of `lvalue` to `buf`.
    fn append_lvalue_to_string(&self, lvalue: &Lvalue, buf: &mut String) {
        match *lvalue {
            Lvalue::Local(local) => {
                let local = &self.mir.local_decls[local];
                match local.name {
                    Some(name) => buf.push_str(&format!("{}", name)),
                    None => buf.push_str("_"),
                }
            }
            Lvalue::Static(ref static_) => {
                buf.push_str(&format!("{}", &self.tcx.item_name(static_.def_id)));
            }
            Lvalue::Projection(ref proj) => {
                let (prefix, suffix, index_operand) = match proj.elem {
                    ProjectionElem::Deref =>
                        ("(*", format!(")"), None),
                    ProjectionElem::Downcast(..) =>
                        ("",   format!(""), None), // (dont emit downcast info)
                    ProjectionElem::Field(field, _ty) =>
                        ("",   format!(".{}", field.index()), None),
                    ProjectionElem::Index(ref index) =>
                        ("",   format!(""), Some(index)),
                    ProjectionElem::ConstantIndex { offset, min_length, from_end: true } =>
                        ("",   format!("[{} of {}]", offset, min_length), None),
                    ProjectionElem::ConstantIndex { offset, min_length, from_end: false } =>
                        ("",   format!("[-{} of {}]", offset, min_length), None),
                    ProjectionElem::Subslice { from, to: 0 } =>
                        ("",   format!("[{}:]", from), None),
                    ProjectionElem::Subslice { from: 0, to } =>
                        ("",   format!("[:-{}]", to), None),
                    ProjectionElem::Subslice { from, to } =>
                        ("",   format!("[{}:-{}]", from, to), None),
                };
                buf.push_str(prefix);
                self.append_lvalue_to_string(&proj.base, buf);
                if let Some(index) = index_operand {
                    buf.push_str("[");
                    self.append_operand_to_string(index, buf);
                    buf.push_str("]");
                } else {
                    buf.push_str(&suffix);
                }

            }
        }
    }

    fn append_operand_to_string(&self, operand: &Operand, buf: &mut String) {
        match *operand {
            Operand::Consume(ref lvalue) => {
                self.append_lvalue_to_string(lvalue, buf);
            }
            Operand::Constant(ref constant) => {
                buf.push_str(&format!("{:?}", constant));
            }
        }
    }
}

impl<'c, 'b, 'a: 'b+'c, 'gcx, 'tcx: 'a> MirBorrowckCtxt<'c, 'b, 'a, 'gcx, 'tcx> {
    // FIXME: needs to be able to express errors analogous to check_loans.rs
    fn conflicts_with(&self, loan1: &BorrowData<'gcx>, loan2: &BorrowData<'gcx>) -> bool {
        if loan1.compatible_with(loan2.kind) { return false; }

        let loan2_base_path = self.base_path(&loan2.lvalue);
        for restricted in self.restrictions(&loan1.lvalue) {
            if restricted != loan2_base_path { continue; }
            return true;
        }

        let loan1_base_path = self.base_path(&loan1.lvalue);
        for restricted in self.restrictions(&loan2.lvalue) {
            if restricted != loan1_base_path { continue; }
            return true;
        }

        return false;
    }

    // FIXME (#16118): function intended to allow the borrow checker
    // to be less precise in its handling of Box while still allowing
    // moves out of a Box. They should be removed when/if we stop
    // treating Box specially (e.g. when/if DerefMove is added...)

    fn base_path<'d>(&self, lvalue: &'d Lvalue<'gcx>) -> &'d Lvalue<'gcx> {
        //! Returns the base of the leftmost (deepest) dereference of an
        //! Box in `lvalue`. If there is no dereference of an Box
        //! in `lvalue`, then it just returns `lvalue` itself.

        let mut cursor = lvalue;
        let mut deepest = lvalue;
        loop {
            let proj = match *cursor {
                Lvalue::Local(..) | Lvalue::Static(..) => return deepest,
                Lvalue::Projection(ref proj) => proj,
            };
            if proj.elem == ProjectionElem::Deref &&
                lvalue.ty(self.mir, self.tcx).to_ty(self.tcx).is_box()
            {
                deepest = &proj.base;
            }
            cursor = &proj.base;
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Context {
    kind: ContextKind,
    loc: Location,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum ContextKind {
    AssignLhs,
    AssignRhs,
    SetDiscrim,
    InlineAsm,
    SwitchInt,
    Drop,
    DropAndReplace,
    CallOperator,
    CallOperand,
    CallDest,
    Assert,
    StorageDead,
}

impl ContextKind {
    fn new(self, loc: Location) -> Context { Context { kind: self, loc: loc } }
}

impl<'b, 'tcx: 'b> InProgress<'b, 'tcx> {
    pub(super) fn new(borrows: DataflowResults<Borrows<'b, 'tcx>>,
                      inits: DataflowResults<MaybeInitializedLvals<'b, 'tcx>>,
                      uninits: DataflowResults<MaybeUninitializedLvals<'b, 'tcx>>)
                      -> Self {
        InProgress {
            borrows: FlowInProgress::new(borrows),
            inits: FlowInProgress::new(inits),
            uninits: FlowInProgress::new(uninits),
        }
    }

    fn each_flow<XB, XI, XU>(&mut self,
                             mut xform_borrows: XB,
                             mut xform_inits: XI,
                             mut xform_uninits: XU) where
        XB: FnMut(&mut FlowInProgress<Borrows<'b, 'tcx>>),
        XI: FnMut(&mut FlowInProgress<MaybeInitializedLvals<'b, 'tcx>>),
        XU: FnMut(&mut FlowInProgress<MaybeUninitializedLvals<'b, 'tcx>>),
    {
        xform_borrows(&mut self.borrows);
        xform_inits(&mut self.inits);
        xform_uninits(&mut self.uninits);
    }

    fn summary(&self) -> String {
        let mut s = String::new();

        s.push_str("borrows in effect: [");
        let mut saw_one = false;
        self.borrows.each_state_bit(|borrow| {
            if saw_one { s.push_str(", "); };
            saw_one = true;
            let borrow_data = &self.borrows.base_results.operator().borrows()[borrow];
            s.push_str(&format!("{}", borrow_data));
        });
        s.push_str("] ");

        s.push_str("borrows generated: [");
        let mut saw_one = false;
        self.borrows.each_gen_bit(|borrow| {
            if saw_one { s.push_str(", "); };
            saw_one = true;
            let borrow_data = &self.borrows.base_results.operator().borrows()[borrow];
            s.push_str(&format!("{}", borrow_data));
        });
        s.push_str("] ");

        s.push_str("inits: [");
        let mut saw_one = false;
        self.inits.each_state_bit(|mpi_init| {
            if saw_one { s.push_str(", "); };
            saw_one = true;
            let move_path =
                &self.inits.base_results.operator().move_data().move_paths[mpi_init];
            s.push_str(&format!("{}", move_path));
        });
        s.push_str("] ");

        s.push_str("uninits: [");
        let mut saw_one = false;
        self.uninits.each_state_bit(|mpi_uninit| {
            if saw_one { s.push_str(", "); };
            saw_one = true;
            let move_path =
                &self.uninits.base_results.operator().move_data().move_paths[mpi_uninit];
            s.push_str(&format!("{}", move_path));
        });
        s.push_str("]");

        return s;
    }
}

impl<BD> FlowInProgress<BD> where BD: BitDenotation {
    fn each_state_bit<F>(&self, f: F) where F: FnMut(BD::Idx) {
        self.curr_state.each_bit(self.base_results.operator().bits_per_block(), f)
    }

    fn each_gen_bit<F>(&self, f: F) where F: FnMut(BD::Idx) {
        self.stmt_gen.each_bit(self.base_results.operator().bits_per_block(), f)
    }

    fn new(results: DataflowResults<BD>) -> Self {
        let bits_per_block = results.sets().bits_per_block();
        let curr_state = IdxSetBuf::new_empty(bits_per_block);
        let stmt_gen = IdxSetBuf::new_empty(bits_per_block);
        let stmt_kill = IdxSetBuf::new_empty(bits_per_block);
        FlowInProgress {
            base_results: results,
            curr_state: curr_state,
            stmt_gen: stmt_gen,
            stmt_kill: stmt_kill,
        }
    }

    fn reset_to_entry_of(&mut self, bb: BasicBlock) {
        (*self.curr_state).clone_from(self.base_results.sets().on_entry_set_for(bb.index()));
    }

    fn reconstruct_statement_effect(&mut self, loc: Location) {
        self.stmt_gen.reset_to_empty();
        self.stmt_kill.reset_to_empty();
        let mut ignored = IdxSetBuf::new_empty(0);
        let mut sets = BlockSets {
            on_entry: &mut ignored, gen_set: &mut self.stmt_gen, kill_set: &mut self.stmt_kill,
        };
        self.base_results.operator().statement_effect(&mut sets, loc);
    }

    fn reconstruct_terminator_effect(&mut self, loc: Location) {
        self.stmt_gen.reset_to_empty();
        self.stmt_kill.reset_to_empty();
        let mut ignored = IdxSetBuf::new_empty(0);
        let mut sets = BlockSets {
            on_entry: &mut ignored, gen_set: &mut self.stmt_gen, kill_set: &mut self.stmt_kill,
        };
        self.base_results.operator().terminator_effect(&mut sets, loc);
    }

    fn apply_local_effect(&mut self) {
        self.curr_state.union(&self.stmt_gen);
        self.curr_state.subtract(&self.stmt_kill);
    }

    fn elems_generated(&self) -> indexed_set::Elems<BD::Idx> {
        let univ = self.base_results.sets().bits_per_block();
        self.stmt_gen.elems(univ)
    }

    fn elems_incoming(&self) -> indexed_set::Elems<BD::Idx> {
        let univ = self.base_results.sets().bits_per_block();
        self.curr_state.elems(univ)
    }
}

impl<'tcx> BorrowData<'tcx> {
    fn compatible_with(&self, bk: BorrowKind) -> bool {
        match (self.kind, bk) {
            (BorrowKind::Shared, BorrowKind::Shared) => true,

            (BorrowKind::Mut, _) |
            (BorrowKind::Unique, _) |
            (_, BorrowKind::Mut) |
            (_, BorrowKind::Unique) => false,
        }
    }
}
