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
                // NOTE: NLL RFC calls for *shallow* write; using Deep
                // for short-term compat w/ AST-borrowck. Also, switch
                // to shallow requires to dataflow: "if this is an
                // assignment `lv = <rvalue>`, then any loan for some
                // path P of which `lv` is a prefix is killed."
                self.mutate_lvalue(ContextKind::AssignLhs.new(location),
                                   (lhs, span), Deep, JustWrite, flow_state);

                self.consume_rvalue(ContextKind::AssignRhs.new(location),
                                    (rhs, span), location, flow_state);
            }
            StatementKind::SetDiscriminant { ref lvalue, variant_index: _ } => {
                self.mutate_lvalue(ContextKind::SetDiscrim.new(location),
                                   (lvalue, span),
                                   Shallow(Some(ArtificialField::Discriminant)),
                                   JustWrite,
                                   flow_state);
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
                                           Deep,
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
                // `Nop`, `Validate`, and `StorageLive` are irrelevant
                // to borrow check.
            }

            StatementKind::StorageDead(local) => {
                self.access_lvalue(ContextKind::StorageDead.new(location),
                                   (&Lvalue::Local(local), span),
                                   (Shallow(None), Write(WriteKind::StorageDead)),
                                   flow_state);
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
                                   (drop_lvalue, span),
                                   Deep,
                                   JustWrite,
                                   flow_state);
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
                                       (dest, span),
                                       Deep,
                                       JustWrite,
                                       flow_state);
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
                    AssertMessage::GeneratorResumedAfterReturn => {}
                    AssertMessage::GeneratorResumedAfterPanic => {}
                }
            }

            TerminatorKind::Yield { ref value, resume: _, drop: _} => {
                self.consume_operand(ContextKind::Yield.new(loc),
                                     Consume, (value, span), flow_state);
            }

            TerminatorKind::Goto { target: _ } |
            TerminatorKind::Resume |
            TerminatorKind::Return |
            TerminatorKind::GeneratorDrop |
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

use self::ShallowOrDeep::{Shallow, Deep};
use self::ReadOrWrite::{Read, Write};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum ArtificialField {
    Discriminant,
    ArrayLength,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum ShallowOrDeep {
    /// From the RFC: "A *shallow* access means that the immediate
    /// fields reached at LV are accessed, but references or pointers
    /// found within are not dereferenced. Right now, the only access
    /// that is shallow is an assignment like `x = ...;`, which would
    /// be a *shallow write* of `x`."
    Shallow(Option<ArtificialField>),

    /// From the RFC: "A *deep* access means that all data reachable
    /// through the given lvalue may be invalidated or accesses by
    /// this action."
    Deep,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum ReadOrWrite {
    /// From the RFC: "A *read* means that the existing data may be
    /// read, but will not be changed."
    Read(ReadKind),

    /// From the RFC: "A *write* means that the data may be mutated to
    /// new values or otherwise invalidated (for example, it could be
    /// de-initialized, as in a move operation).
    Write(WriteKind),
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum ReadKind {
    Borrow(BorrowKind),
    Copy,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum WriteKind {
    StorageDead,
    MutableBorrow(BorrowKind),
    Mutate,
    Move,
}

impl<'c, 'b, 'a: 'b+'c, 'gcx, 'tcx: 'a> MirBorrowckCtxt<'c, 'b, 'a, 'gcx, 'tcx> {
    fn access_lvalue(&mut self,
                     context: Context,
                     lvalue_span: (&Lvalue<'gcx>, Span),
                     kind: (ShallowOrDeep, ReadOrWrite),
                     flow_state: &InProgress<'b, 'gcx>) {
        // FIXME: also need to check permissions (e.g. reject mut
        // borrow of immutable ref, moves through non-`Box`-ref)
        let (sd, rw) = kind;
        self.each_borrow_involving_path(
            context, (sd, lvalue_span.0), flow_state, |this, _index, borrow, common_prefix| {
                match (rw, borrow.kind) {
                    (Read(_), BorrowKind::Shared) => {
                        Control::Continue
                    }
                    (Read(kind), BorrowKind::Unique) |
                    (Read(kind), BorrowKind::Mut) => {
                        match kind {
                            ReadKind::Copy =>
                                this.report_use_while_mutably_borrowed(
                                    context, lvalue_span, borrow),
                            ReadKind::Borrow(bk) => {
                                let end_issued_loan_span =
                                    flow_state.borrows.base_results.operator().region_span(
                                        &borrow.region).end_point();
                                this.report_conflicting_borrow(
                                    context, common_prefix, lvalue_span, bk,
                                    &borrow, end_issued_loan_span)
                            }
                        }
                        Control::Break
                    }
                    (Write(kind), _) => {
                        match kind {
                            WriteKind::MutableBorrow(bk) => {
                                let end_issued_loan_span =
                                    flow_state.borrows.base_results.operator().region_span(
                                        &borrow.region).end_point();
                                this.report_conflicting_borrow(
                                    context, common_prefix, lvalue_span, bk,
                                    &borrow, end_issued_loan_span)
                            }
                            WriteKind::StorageDead |
                            WriteKind::Mutate =>
                                this.report_illegal_mutation_of_borrowed(
                                    context, lvalue_span, borrow),
                            WriteKind::Move =>
                                this.report_move_out_while_borrowed(
                                    context, lvalue_span, &borrow),
                        }
                        Control::Break
                    }
                }
            });
    }

    fn mutate_lvalue(&mut self,
                     context: Context,
                     lvalue_span: (&Lvalue<'gcx>, Span),
                     kind: ShallowOrDeep,
                     mode: MutateMode,
                     flow_state: &InProgress<'b, 'gcx>) {
        // Write of P[i] or *P, or WriteAndRead of any P, requires P init'd.
        match mode {
            MutateMode::WriteAndRead => {
                self.check_if_path_is_moved(context, "update", lvalue_span, flow_state);
            }
            MutateMode::JustWrite => {
                self.check_if_assigned_path_is_moved(context, lvalue_span, flow_state);
            }
        }

        self.access_lvalue(context, lvalue_span, (kind, Write(WriteKind::Mutate)), flow_state);

        // check for reassignments to immutable local variables
        self.check_if_reassignment_to_immutable_state(context, lvalue_span, flow_state);
    }

    fn consume_rvalue(&mut self,
                      context: Context,
                      (rvalue, span): (&Rvalue<'gcx>, Span),
                      _location: Location,
                      flow_state: &InProgress<'b, 'gcx>) {
        match *rvalue {
            Rvalue::Ref(_/*rgn*/, bk, ref lvalue) => {
                let access_kind = match bk {
                    BorrowKind::Shared => (Deep, Read(ReadKind::Borrow(bk))),
                    BorrowKind::Unique |
                    BorrowKind::Mut => (Deep, Write(WriteKind::MutableBorrow(bk))),
                };
                self.access_lvalue(context, (lvalue, span), access_kind, flow_state);
                self.check_if_path_is_moved(context, "borrow", (lvalue, span), flow_state);
            }

            Rvalue::Use(ref operand) |
            Rvalue::Repeat(ref operand, _) |
            Rvalue::UnaryOp(_/*un_op*/, ref operand) |
            Rvalue::Cast(_/*cast_kind*/, ref operand, _/*ty*/) => {
                self.consume_operand(context, Consume, (operand, span), flow_state)
            }

            Rvalue::Len(ref lvalue) |
            Rvalue::Discriminant(ref lvalue) => {
                let af = match *rvalue {
                    Rvalue::Len(..) => ArtificialField::ArrayLength,
                    Rvalue::Discriminant(..) => ArtificialField::Discriminant,
                    _ => unreachable!(),
                };
                self.access_lvalue(
                    context, (lvalue, span), (Shallow(Some(af)), Read(ReadKind::Copy)), flow_state);
                self.check_if_path_is_moved(context, "use", (lvalue, span), flow_state);
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
            Operand::Consume(ref lvalue) => {
                self.consume_lvalue(context, consume_via_drop, (lvalue, span), flow_state)
            }
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
            self.access_lvalue(context, lvalue_span, (Deep, Write(WriteKind::Move)), flow_state);
        } else {
            // copy of lvalue: check if this is "copy of frozen path" (FIXME: see check_loans.rs)
            self.access_lvalue(context, lvalue_span, (Deep, Read(ReadKind::Copy)), flow_state);
        }

        // Finally, check if path was already moved.
        match consume_via_drop {
            ConsumeKind::Drop => {
                // If path is merely being dropped, then we'll already
                // check the drop flag to see if it is moved (thus we
                // skip this check in that case).
            }
            ConsumeKind::Consume => {
                self.check_if_path_is_moved(context, "use", lvalue_span, flow_state);
            }
        }
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
                // FIXME: Not ideal, it only finds the assignment that lexically comes first
                let assigned_lvalue = &move_data.move_paths[mpi].lvalue;
                let assignment_stmt = self.mir.basic_blocks().iter().filter_map(|bb| {
                    bb.statements.iter().find(|stmt| {
                        if let StatementKind::Assign(ref lv, _) = stmt.kind {
                            *lv == *assigned_lvalue
                        } else {
                            false
                        }
                    })
                }).next().unwrap();
                self.report_illegal_reassignment(
                    context, (lvalue, span), assignment_stmt.source_info.span);
            }
        }
    }

    fn check_if_path_is_moved(&mut self,
                              context: Context,
                              desired_action: &str,
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
                self.report_use_of_moved(context, desired_action, lvalue_span);
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

                                    self.check_if_path_is_moved(
                                        context, "assignment", (base, span), flow_state);

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
}

impl<'c, 'b, 'a: 'b+'c, 'gcx, 'tcx: 'a> MirBorrowckCtxt<'c, 'b, 'a, 'gcx, 'tcx> {
    fn each_borrow_involving_path<F>(&mut self,
                                     _context: Context,
                                     access_lvalue: (ShallowOrDeep, &Lvalue<'gcx>),
                                     flow_state: &InProgress<'b, 'gcx>,
                                     mut op: F)
        where F: FnMut(&mut Self, BorrowIndex, &BorrowData<'gcx>, &Lvalue) -> Control
    {
        let (access, lvalue) = access_lvalue;

        // FIXME: analogous code in check_loans first maps `lvalue` to
        // its base_path.

        let domain = flow_state.borrows.base_results.operator();
        let data = domain.borrows();

        // check for loan restricting path P being used. Accounts for
        // borrows of P, P.a.b, etc.
        'next_borrow: for i in flow_state.borrows.elems_incoming() {
            let borrowed = &data[i];

            // Is `lvalue` (or a prefix of it) already borrowed? If
            // so, that's relevant.
            //
            // FIXME: Differs from AST-borrowck; includes drive-by fix
            // to #38899. Will probably need back-compat mode flag.
            for accessed_prefix in self.prefixes(lvalue, PrefixSet::All) {
                if *accessed_prefix == borrowed.lvalue {
                    // FIXME: pass in enum describing case we are in?
                    let ctrl = op(self, i, borrowed, accessed_prefix);
                    if ctrl == Control::Break { return; }
                }
            }

            // Is `lvalue` a prefix (modulo access type) of the
            // `borrowed.lvalue`? If so, that's relevant.

            let prefix_kind = match access {
                Shallow(Some(ArtificialField::Discriminant)) |
                Shallow(Some(ArtificialField::ArrayLength)) => {
                    // The discriminant and array length are like
                    // additional fields on the type; they do not
                    // overlap any existing data there. Furthermore,
                    // they cannot actually be a prefix of any
                    // borrowed lvalue (at least in MIR as it is
                    // currently.)
                    continue 'next_borrow;
                }
                Shallow(None) => PrefixSet::Shallow,
                Deep => PrefixSet::Supporting,
            };

            for borrowed_prefix in self.prefixes(&borrowed.lvalue, prefix_kind) {
                if borrowed_prefix == lvalue {
                    // FIXME: pass in enum describing case we are in?
                    let ctrl = op(self, i, borrowed, borrowed_prefix);
                    if ctrl == Control::Break { return; }
                }
            }
        }
    }
}

use self::prefixes::PrefixSet;

/// From the NLL RFC: "The deep [aka 'supporting'] prefixes for an
/// lvalue are formed by stripping away fields and derefs, except that
/// we stop when we reach the deref of a shared reference. [...] "
///
/// "Shallow prefixes are found by stripping away fields, but stop at
/// any dereference. So: writing a path like `a` is illegal if `a.b`
/// is borrowed. But: writing `a` is legal if `*a` is borrowed,
/// whether or not `a` is a shared or mutable reference. [...] "
mod prefixes {
    use super::{MirBorrowckCtxt};

    use rustc::hir;
    use rustc::ty::{self, TyCtxt};
    use rustc::mir::{Lvalue, Mir, ProjectionElem};

    pub trait IsPrefixOf<'tcx> {
        fn is_prefix_of(&self, other: &Lvalue<'tcx>) -> bool;
    }

    impl<'tcx> IsPrefixOf<'tcx> for Lvalue<'tcx> {
        fn is_prefix_of(&self, other: &Lvalue<'tcx>) -> bool {
            let mut cursor = other;
            loop {
                if self == cursor {
                    return true;
                }

                match *cursor {
                    Lvalue::Local(_) |
                    Lvalue::Static(_) => return false,
                    Lvalue::Projection(ref proj) => {
                        cursor = &proj.base;
                    }
                }
            }
        }
    }


    pub(super) struct Prefixes<'c, 'tcx: 'c> {
        mir: &'c Mir<'tcx>,
        tcx: TyCtxt<'c, 'tcx, 'tcx>,
        kind: PrefixSet,
        next: Option<&'c Lvalue<'tcx>>,
    }

    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub(super) enum PrefixSet {
        All,
        Shallow,
        Supporting,
    }

    impl<'c, 'b, 'a: 'b+'c, 'gcx, 'tcx: 'a> MirBorrowckCtxt<'c, 'b, 'a, 'gcx, 'tcx> {
        pub(super) fn prefixes<'d>(&self,
                                   lvalue: &'d Lvalue<'gcx>,
                                   kind: PrefixSet)
                                   -> Prefixes<'d, 'gcx> where 'b: 'd
        {
            Prefixes { next: Some(lvalue), kind, mir: self.mir, tcx: self.tcx }
        }
    }

    impl<'c, 'tcx> Iterator for Prefixes<'c, 'tcx> {
        type Item = &'c Lvalue<'tcx>;
        fn next(&mut self) -> Option<Self::Item> {
            let mut cursor = match self.next {
                None => return None,
                Some(lvalue) => lvalue,
            };

            // Post-processing `lvalue`: Enqueue any remaining
            // work. Also, `lvalue` may not be a prefix itself, but
            // may hold one further down (e.g. we never return
            // downcasts here, but may return a base of a downcast).

            'cursor: loop {
                let proj = match *cursor {
                    Lvalue::Local(_) | // search yielded this leaf
                    Lvalue::Static(_) => {
                        self.next = None;
                        return Some(cursor);
                    }

                    Lvalue::Projection(ref proj) => proj,
                };

                match proj.elem {
                    ProjectionElem::Field(_/*field*/, _/*ty*/) => {
                        // FIXME: add union handling
                        self.next = Some(&proj.base);
                        return Some(cursor);
                    }
                    ProjectionElem::Downcast(..) |
                    ProjectionElem::Subslice { .. } |
                    ProjectionElem::ConstantIndex { .. } |
                    ProjectionElem::Index(_) => {
                        cursor = &proj.base;
                        continue 'cursor;
                    }
                    ProjectionElem::Deref => {
                        // (handled below)
                    }
                }

                assert_eq!(proj.elem, ProjectionElem::Deref);

                match self.kind {
                    PrefixSet::Shallow => {
                        // shallow prefixes are found by stripping away
                        // fields, but stop at *any* dereference.
                        // So we can just stop the traversal now.
                        self.next = None;
                        return Some(cursor);
                    }
                    PrefixSet::All => {
                        // all prefixes: just blindly enqueue the base
                        // of the projection
                        self.next = Some(&proj.base);
                        return Some(cursor);
                    }
                    PrefixSet::Supporting => {
                        // fall through!
                    }
                }

                assert_eq!(self.kind, PrefixSet::Supporting);
                // supporting prefixes: strip away fields and
                // derefs, except we stop at the deref of a shared
                // reference.

                let ty = proj.base.ty(self.mir, self.tcx).to_ty(self.tcx);
                match ty.sty {
                    ty::TyRawPtr(_) |
                    ty::TyRef(_/*rgn*/, ty::TypeAndMut { ty: _, mutbl: hir::MutImmutable }) => {
                        // don't continue traversing over derefs of raw pointers or shared borrows.
                        self.next = None;
                        return Some(cursor);
                    }

                    ty::TyRef(_/*rgn*/, ty::TypeAndMut { ty: _, mutbl: hir::MutMutable }) => {
                        self.next = Some(&proj.base);
                        return Some(cursor);
                    }

                    ty::TyAdt(..) if ty.is_box() => {
                        self.next = Some(&proj.base);
                        return Some(cursor);
                    }

                    _ => panic!("unknown type fed to Projection Deref."),
                }
            }
        }
    }
}

impl<'c, 'b, 'a: 'b+'c, 'gcx, 'tcx: 'a> MirBorrowckCtxt<'c, 'b, 'a, 'gcx, 'tcx> {
    fn report_use_of_moved(&mut self,
                           _context: Context,
                           desired_action: &str,
                           (lvalue, span): (&Lvalue, Span)) {
        self.tcx.cannot_act_on_uninitialized_variable(span,
                                                      desired_action,
                                                      &self.describe_lvalue(lvalue),
                                                      Origin::Mir)
                .span_label(span, format!("use of possibly uninitialized `{}`",
                                          self.describe_lvalue(lvalue)))
                .emit();
    }

    fn report_move_out_while_borrowed(&mut self,
                                      _context: Context,
                                      (lvalue, span): (&Lvalue, Span),
                                      borrow: &BorrowData) {
        self.tcx.cannot_move_when_borrowed(span,
                                           &self.describe_lvalue(lvalue),
                                           Origin::Mir)
                .span_label(self.retrieve_borrow_span(borrow),
                            format!("borrow of `{}` occurs here",
                                    self.describe_lvalue(&borrow.lvalue)))
                .span_label(span, format!("move out of `{}` occurs here",
                                          self.describe_lvalue(lvalue)))
                .emit();
    }

    fn report_use_while_mutably_borrowed(&mut self,
                                         _context: Context,
                                         (lvalue, span): (&Lvalue, Span),
                                         borrow : &BorrowData) {

        let mut err = self.tcx.cannot_use_when_mutably_borrowed(
            span, &self.describe_lvalue(lvalue),
            self.retrieve_borrow_span(borrow), &self.describe_lvalue(&borrow.lvalue),
            Origin::Mir);

        err.emit();
    }

    fn report_conflicting_borrow(&mut self,
                                 _context: Context,
                                 common_prefix: &Lvalue,
                                 (lvalue, span): (&Lvalue, Span),
                                 gen_borrow_kind: BorrowKind,
                                 issued_borrow: &BorrowData,
                                 end_issued_loan_span: Span) {
        use self::prefixes::IsPrefixOf;

        assert!(common_prefix.is_prefix_of(lvalue));
        assert!(common_prefix.is_prefix_of(&issued_borrow.lvalue));

        let issued_span = self.retrieve_borrow_span(issued_borrow);

        // FIXME: supply non-"" `opt_via` when appropriate
        let mut err = match (gen_borrow_kind, "immutable", "mutable",
                             issued_borrow.kind, "immutable", "mutable") {
            (BorrowKind::Shared, lft, _, BorrowKind::Mut, _, rgt) |
            (BorrowKind::Mut, _, lft, BorrowKind::Shared, rgt, _) =>
                self.tcx.cannot_reborrow_already_borrowed(
                    span, &self.describe_lvalue(lvalue), "", lft, issued_span,
                    "it", rgt, "", end_issued_loan_span, Origin::Mir),

            (BorrowKind::Mut, _, _, BorrowKind::Mut, _, _) =>
                self.tcx.cannot_mutably_borrow_multiply(
                    span, &self.describe_lvalue(lvalue), "", issued_span,
                    "", end_issued_loan_span, Origin::Mir),

            (BorrowKind::Unique, _, _, BorrowKind::Unique, _, _) =>
                self.tcx.cannot_uniquely_borrow_by_two_closures(
                    span, &self.describe_lvalue(lvalue), issued_span,
                    end_issued_loan_span, Origin::Mir),

            (BorrowKind::Unique, _, _, _, _, _) =>
                self.tcx.cannot_uniquely_borrow_by_one_closure(
                    span, &self.describe_lvalue(lvalue), "",
                    issued_span, "it", "", end_issued_loan_span, Origin::Mir),

            (_, _, _, BorrowKind::Unique, _, _) =>
                self.tcx.cannot_reborrow_already_uniquely_borrowed(
                    span, &self.describe_lvalue(lvalue), "it", "",
                    issued_span, "", end_issued_loan_span, Origin::Mir),

            (BorrowKind::Shared, _, _, BorrowKind::Shared, _, _) =>
                unreachable!(),
        };
        err.emit();
    }

    fn report_illegal_mutation_of_borrowed(&mut self,
                                           _: Context,
                                           (lvalue, span): (&Lvalue, Span),
                                           loan: &BorrowData) {
        let mut err = self.tcx.cannot_assign_to_borrowed(
            span, self.retrieve_borrow_span(loan), &self.describe_lvalue(lvalue), Origin::Mir);

        err.emit();
    }

    fn report_illegal_reassignment(&mut self,
                                   _context: Context,
                                   (lvalue, span): (&Lvalue, Span),
                                   assigned_span: Span) {
        self.tcx.cannot_reassign_immutable(span,
                                           &self.describe_lvalue(lvalue),
                                           Origin::Mir)
                .span_label(span, "re-assignment of immutable variable")
                .span_label(assigned_span, format!("first assignment to `{}`",
                                                   self.describe_lvalue(lvalue)))
                .emit();
    }

    fn report_assignment_to_static(&mut self, _context: Context, (lvalue, span): (&Lvalue, Span)) {
        let mut err = self.tcx.cannot_assign_static(
            span, &self.describe_lvalue(lvalue), Origin::Mir);
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
                        ("",   format!(".{}", field.index()), None), // FIXME: report name of field
                    ProjectionElem::Index(index) =>
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
                    self.append_lvalue_to_string(&Lvalue::Local(index), buf);
                    buf.push_str("]");
                } else {
                    buf.push_str(&suffix);
                }
            }
        }
    }

    // Retrieve span of given borrow from the current MIR representation
    fn retrieve_borrow_span(&self, borrow: &BorrowData) -> Span {
        self.mir.basic_blocks()[borrow.location.block]
            .statements[borrow.location.statement_index]
            .source_info.span
    }
}

impl<'c, 'b, 'a: 'b+'c, 'gcx, 'tcx: 'a> MirBorrowckCtxt<'c, 'b, 'a, 'gcx, 'tcx> {
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
    Yield,
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

    fn elems_incoming(&self) -> indexed_set::Elems<BD::Idx> {
        let univ = self.base_results.sets().bits_per_block();
        self.curr_state.elems(univ)
    }
}
