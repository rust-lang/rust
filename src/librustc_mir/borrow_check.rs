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

use rustc::hir;
use rustc::hir::def_id::{DefId};
use rustc::infer::{InferCtxt};
use rustc::ty::{self, TyCtxt, ParamEnv};
use rustc::ty::maps::Providers;
use rustc::mir::{AssertMessage, BasicBlock, BorrowKind, Location, Lvalue, Local};
use rustc::mir::{Mir, Mutability, Operand, Projection, ProjectionElem, Rvalue};
use rustc::mir::{Statement, StatementKind, Terminator, TerminatorKind};
use transform::nll;

use rustc_data_structures::indexed_set::{self, IdxSetBuf};
use rustc_data_structures::indexed_vec::{Idx};

use syntax::ast::{self};
use syntax_pos::{DUMMY_SP, Span};

use dataflow::{do_dataflow};
use dataflow::{MoveDataParamEnv};
use dataflow::{BitDenotation, BlockSets, DataflowResults, DataflowResultsConsumer};
use dataflow::{MaybeInitializedLvals, MaybeUninitializedLvals};
use dataflow::{MovingOutStatements};
use dataflow::{Borrows, BorrowData, BorrowIndex};
use dataflow::move_paths::{MoveError, IllegalMoveOriginKind};
use dataflow::move_paths::{HasMoveData, MoveData, MovePathIndex, LookupResult, MoveOutIndex};
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
    let input_mir = tcx.mir_validated(def_id);
    debug!("run query mir_borrowck: {}", tcx.item_path_str(def_id));

    if {
        !tcx.has_attr(def_id, "rustc_mir_borrowck") &&
            !tcx.sess.opts.debugging_opts.borrowck_mir &&
            !tcx.sess.opts.debugging_opts.nll
    } {
        return;
    }

    tcx.infer_ctxt().enter(|infcx| {
        let input_mir: &Mir = &input_mir.borrow();
        do_mir_borrowck(&infcx, input_mir, def_id);
    });
    debug!("mir_borrowck done");
}

fn do_mir_borrowck<'a, 'gcx, 'tcx>(infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                                   input_mir: &Mir<'gcx>,
                                   def_id: DefId)
{
    let tcx = infcx.tcx;
    let attributes = tcx.get_attrs(def_id);
    let param_env = tcx.param_env(def_id);
    let id = tcx.hir.as_local_node_id(def_id)
        .expect("do_mir_borrowck: non-local DefId");

    let move_data: MoveData<'tcx> = match MoveData::gather_moves(input_mir, tcx, param_env) {
        Ok(move_data) => move_data,
        Err((move_data, move_errors)) => {
            for move_error in move_errors {
                let (span, kind): (Span, IllegalMoveOriginKind) = match move_error {
                    MoveError::UnionMove { .. } =>
                        unimplemented!("dont know how to report union move errors yet."),
                    MoveError::IllegalMove { cannot_move_out_of: o } => (o.span, o.kind),
                };
                let origin = Origin::Mir;
                let mut err = match kind {
                    IllegalMoveOriginKind::Static =>
                        tcx.cannot_move_out_of(span, "static item", origin),
                    IllegalMoveOriginKind::BorrowedContent =>
                        tcx.cannot_move_out_of(span, "borrowed_content", origin),
                    IllegalMoveOriginKind::InteriorOfTypeWithDestructor { container_ty: ty } =>
                        tcx.cannot_move_out_of_interior_of_drop(span, ty, origin),
                    IllegalMoveOriginKind::InteriorOfSliceOrArray { ty, is_index } =>
                        tcx.cannot_move_out_of_interior_noncopy(span, ty, is_index, origin),
                };
                err.emit();
            }
            move_data
        }
    };

    // Make our own copy of the MIR. This copy will be modified (in place) to
    // contain non-lexical lifetimes. It will have a lifetime tied
    // to the inference context.
    let mut mir: Mir<'tcx> = input_mir.clone();
    let mir = &mut mir;

    // If we are in non-lexical mode, compute the non-lexical lifetimes.
    let opt_regioncx = if !tcx.sess.opts.debugging_opts.nll {
        None
    } else {
        Some(nll::compute_regions(infcx, def_id, mir))
    };

    let mdpe = MoveDataParamEnv { move_data: move_data, param_env: param_env };
    let dead_unwinds = IdxSetBuf::new_empty(mir.basic_blocks().len());
    let flow_borrows = do_dataflow(tcx, mir, id, &attributes, &dead_unwinds,
                                   Borrows::new(tcx, mir, opt_regioncx.as_ref()),
                                   |bd, i| bd.location(i));
    let flow_inits = do_dataflow(tcx, mir, id, &attributes, &dead_unwinds,
                                 MaybeInitializedLvals::new(tcx, mir, &mdpe),
                                 |bd, i| &bd.move_data().move_paths[i]);
    let flow_uninits = do_dataflow(tcx, mir, id, &attributes, &dead_unwinds,
                                   MaybeUninitializedLvals::new(tcx, mir, &mdpe),
                                   |bd, i| &bd.move_data().move_paths[i]);
    let flow_move_outs = do_dataflow(tcx, mir, id, &attributes, &dead_unwinds,
                                     MovingOutStatements::new(tcx, mir, &mdpe),
                                     |bd, i| &bd.move_data().moves[i]);

    let mut mbcx = MirBorrowckCtxt {
        tcx: tcx,
        mir: mir,
        node_id: id,
        move_data: &mdpe.move_data,
        param_env: param_env,
        fake_infer_ctxt: &infcx,
    };

    let mut state = InProgress::new(flow_borrows,
                                    flow_inits,
                                    flow_uninits,
                                    flow_move_outs);

    mbcx.analyze_results(&mut state); // entry point for DataflowResultsConsumer
}

#[allow(dead_code)]
pub struct MirBorrowckCtxt<'c, 'b, 'a: 'b+'c, 'gcx: 'a+'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    mir: &'b Mir<'tcx>,
    node_id: ast::NodeId,
    move_data: &'b MoveData<'tcx>,
    param_env: ParamEnv<'tcx>,
    fake_infer_ctxt: &'c InferCtxt<'c, 'gcx, 'tcx>,
}

// (forced to be `pub` due to its use as an associated type below.)
pub struct InProgress<'b, 'gcx: 'tcx, 'tcx: 'b> {
    borrows: FlowInProgress<Borrows<'b, 'gcx, 'tcx>>,
    inits: FlowInProgress<MaybeInitializedLvals<'b, 'gcx, 'tcx>>,
    uninits: FlowInProgress<MaybeUninitializedLvals<'b, 'gcx, 'tcx>>,
    move_outs: FlowInProgress<MovingOutStatements<'b, 'gcx, 'tcx>>,
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
impl<'c, 'b, 'a: 'b+'c, 'gcx, 'tcx: 'a> DataflowResultsConsumer<'b, 'tcx>
    for MirBorrowckCtxt<'c, 'b, 'a, 'gcx, 'tcx>
{
    type FlowState = InProgress<'b, 'gcx, 'tcx>;

    fn mir(&self) -> &'b Mir<'tcx> { self.mir }

    fn reset_to_entry_of(&mut self, bb: BasicBlock, flow_state: &mut Self::FlowState) {
        flow_state.each_flow(|b| b.reset_to_entry_of(bb),
                             |i| i.reset_to_entry_of(bb),
                             |u| u.reset_to_entry_of(bb),
                             |m| m.reset_to_entry_of(bb));
    }

    fn reconstruct_statement_effect(&mut self,
                                    location: Location,
                                    flow_state: &mut Self::FlowState) {
        flow_state.each_flow(|b| b.reconstruct_statement_effect(location),
                             |i| i.reconstruct_statement_effect(location),
                             |u| u.reconstruct_statement_effect(location),
                             |m| m.reconstruct_statement_effect(location));
    }

    fn apply_local_effect(&mut self,
                          _location: Location,
                          flow_state: &mut Self::FlowState) {
        flow_state.each_flow(|b| b.apply_local_effect(),
                             |i| i.apply_local_effect(),
                             |u| u.apply_local_effect(),
                             |m| m.apply_local_effect());
    }

    fn reconstruct_terminator_effect(&mut self,
                                     location: Location,
                                     flow_state: &mut Self::FlowState) {
        flow_state.each_flow(|b| b.reconstruct_terminator_effect(location),
                             |i| i.reconstruct_terminator_effect(location),
                             |u| u.reconstruct_terminator_effect(location),
                             |m| m.reconstruct_terminator_effect(location));
    }

    fn visit_block_entry(&mut self,
                         bb: BasicBlock,
                         flow_state: &Self::FlowState) {
        let summary = flow_state.summary();
        debug!("MirBorrowckCtxt::process_block({:?}): {}", bb, summary);
    }

    fn visit_statement_entry(&mut self,
                             location: Location,
                             stmt: &Statement<'tcx>,
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
                              term: &Terminator<'tcx>,
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
            TerminatorKind::Unreachable |
            TerminatorKind::FalseEdges { .. } => {
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
                     lvalue_span: (&Lvalue<'tcx>, Span),
                     kind: (ShallowOrDeep, ReadOrWrite),
                     flow_state: &InProgress<'b, 'gcx, 'tcx>) {

        let (sd, rw) = kind;

        // Check permissions
        self.check_access_permissions(lvalue_span, rw);

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
                                    flow_state.borrows.base_results.operator().opt_region_end_span(
                                        &borrow.region);
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
                                    flow_state.borrows.base_results.operator().opt_region_end_span(
                                        &borrow.region);
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
                     lvalue_span: (&Lvalue<'tcx>, Span),
                     kind: ShallowOrDeep,
                     mode: MutateMode,
                     flow_state: &InProgress<'b, 'gcx, 'tcx>) {
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
                      (rvalue, span): (&Rvalue<'tcx>, Span),
                      _location: Location,
                      flow_state: &InProgress<'b, 'gcx, 'tcx>) {
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
                       (operand, span): (&Operand<'tcx>, Span),
                       flow_state: &InProgress<'b, 'gcx, 'tcx>) {
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
                      lvalue_span: (&Lvalue<'tcx>, Span),
                      flow_state: &InProgress<'b, 'gcx, 'tcx>) {
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
                                                (lvalue, span): (&Lvalue<'tcx>, Span),
                                                flow_state: &InProgress<'b, 'gcx, 'tcx>) {
        let move_data = self.move_data;

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

        if let Some(mpi) = self.move_path_for_lvalue(lvalue) {
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
                              lvalue_span: (&Lvalue<'tcx>, Span),
                              flow_state: &InProgress<'b, 'gcx, 'tcx>) {
        // FIXME: analogous code in check_loans first maps `lvalue` to
        // its base_path ... but is that what we want here?
        let lvalue = self.base_path(lvalue_span.0);

        let maybe_uninits = &flow_state.uninits;
        let curr_move_outs = &flow_state.move_outs.curr_state;

        // Bad scenarios:
        //
        // 1. Move of `a.b.c`, use of `a.b.c`
        // 2. Move of `a.b.c`, use of `a.b.c.d` (without first reinitializing `a.b.c.d`)
        // 3. Move of `a.b.c`, use of `a` or `a.b`
        // 4. Uninitialized `(a.b.c: &_)`, use of `*a.b.c`; note that with
        //    partial initialization support, one might have `a.x`
        //    initialized but not `a.b`.
        //
        // OK scenarios:
        //
        // 5. Move of `a.b.c`, use of `a.b.d`
        // 6. Uninitialized `a.x`, initialized `a.b`, use of `a.b`
        // 7. Copied `(a.b: &_)`, use of `*(a.b).c`; note that `a.b`
        //    must have been initialized for the use to be sound.
        // 8. Move of `a.b.c` then reinit of `a.b.c.d`, use of `a.b.c.d`

        // The dataflow tracks shallow prefixes distinctly (that is,
        // field-accesses on P distinctly from P itself), in order to
        // track substructure initialization separately from the whole
        // structure.
        //
        // E.g., when looking at (*a.b.c).d, if the closest prefix for
        // which we have a MovePath is `a.b`, then that means that the
        // initialization state of `a.b` is all we need to inspect to
        // know if `a.b.c` is valid (and from that we infer that the
        // dereference and `.d` access is also valid, since we assume
        // `a.b.c` is assigned a reference to a initialized and
        // well-formed record structure.)

        // Therefore, if we seek out the *closest* prefix for which we
        // have a MovePath, that should capture the initialization
        // state for the lvalue scenario.
        //
        // This code covers scenarios 1, 2, and 4.

        debug!("check_if_path_is_moved part1 lvalue: {:?}", lvalue);
        match self.move_path_closest_to(lvalue) {
            Ok(mpi) => {
                if maybe_uninits.curr_state.contains(&mpi) {
                    self.report_use_of_moved_or_uninitialized(context, desired_action,
                                                              lvalue_span, mpi,
                                                              curr_move_outs);
                    return; // don't bother finding other problems.
                }
            }
            Err(NoMovePathFound::ReachedStatic) => {
                // Okay: we do not build MoveData for static variables
            }

            // Only query longest prefix with a MovePath, not further
            // ancestors; dataflow recurs on children when parents
            // move (to support partial (re)inits).
            //
            // (I.e. querying parents breaks scenario 8; but may want
            // to do such a query based on partial-init feature-gate.)
        }

        // A move of any shallow suffix of `lvalue` also interferes
        // with an attempt to use `lvalue`. This is scenario 3 above.
        //
        // (Distinct from handling of scenarios 1+2+4 above because
        // `lvalue` does not interfere with suffixes of its prefixes,
        // e.g. `a.b.c` does not interfere with `a.b.d`)

        debug!("check_if_path_is_moved part2 lvalue: {:?}", lvalue);
        if let Some(mpi) = self.move_path_for_lvalue(lvalue) {
            if let Some(child_mpi) = maybe_uninits.has_any_child_of(mpi) {
                self.report_use_of_moved_or_uninitialized(context, desired_action,
                                                          lvalue_span, child_mpi,
                                                          curr_move_outs);
                return; // don't bother finding other problems.
            }
        }
    }

    /// Currently MoveData does not store entries for all lvalues in
    /// the input MIR. For example it will currently filter out
    /// lvalues that are Copy; thus we do not track lvalues of shared
    /// reference type. This routine will walk up an lvalue along its
    /// prefixes, searching for a foundational lvalue that *is*
    /// tracked in the MoveData.
    ///
    /// An Err result includes a tag indicated why the search failed.
    /// Currenly this can only occur if the lvalue is built off of a
    /// static variable, as we do not track those in the MoveData.
    fn move_path_closest_to(&mut self, lvalue: &Lvalue<'tcx>)
                            -> Result<MovePathIndex, NoMovePathFound>
    {
        let mut last_prefix = lvalue;
        for prefix in self.prefixes(lvalue, PrefixSet::All) {
            if let Some(mpi) = self.move_path_for_lvalue(prefix) {
                return Ok(mpi);
            }
            last_prefix = prefix;
        }
        match *last_prefix {
            Lvalue::Local(_) => panic!("should have move path for every Local"),
            Lvalue::Projection(_) => panic!("PrefixSet::All meant dont stop for Projection"),
            Lvalue::Static(_) => return Err(NoMovePathFound::ReachedStatic),
        }
    }

    fn move_path_for_lvalue(&mut self,
                            lvalue: &Lvalue<'tcx>)
                            -> Option<MovePathIndex>
    {
        // If returns None, then there is no move path corresponding
        // to a direct owner of `lvalue` (which means there is nothing
        // that borrowck tracks for its analysis).

        match self.move_data.rev_lookup.find(lvalue) {
            LookupResult::Parent(_) => None,
            LookupResult::Exact(mpi) => Some(mpi),
        }
    }

    fn check_if_assigned_path_is_moved(&mut self,
                                       context: Context,
                                       (lvalue, span): (&Lvalue<'tcx>, Span),
                                       flow_state: &InProgress<'b, 'gcx, 'tcx>) {
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

    /// Check the permissions for the given lvalue and read or write kind
    fn check_access_permissions(&self, (lvalue, span): (&Lvalue<'tcx>, Span), kind: ReadOrWrite) {
        match kind {
            Write(WriteKind::MutableBorrow(BorrowKind::Unique)) => {
                if let Err(_lvalue_err) = self.is_unique(lvalue) {
                    span_bug!(span, "&unique borrow for `{}` should not fail",
                        self.describe_lvalue(lvalue));
                }
            },
            Write(WriteKind::MutableBorrow(BorrowKind::Mut)) => {
                if let Err(lvalue_err) = self.is_mutable(lvalue) {
                    let mut err = self.tcx.cannot_borrow_path_as_mutable(span,
                        &format!("immutable item `{}`",
                                  self.describe_lvalue(lvalue)),
                        Origin::Mir);
                    err.span_label(span, "cannot borrow as mutable");

                    if lvalue != lvalue_err {
                        err.note(&format!("Value not mutable causing this error: `{}`",
                            self.describe_lvalue(lvalue_err)));
                    }

                    err.emit();
                }
            },
            _ => {}// Access authorized
        }
    }

    /// Can this value be written or borrowed mutably
    fn is_mutable<'d>(&self, lvalue: &'d Lvalue<'tcx>) -> Result<(), &'d Lvalue<'tcx>> {
        match *lvalue {
            Lvalue::Local(local) => {
                let local = &self.mir.local_decls[local];
                match local.mutability {
                    Mutability::Not => Err(lvalue),
                    Mutability::Mut => Ok(())
                }
            },
            Lvalue::Static(ref static_) => {
                if !self.tcx.is_static_mut(static_.def_id) {
                    Err(lvalue)
                } else {
                    Ok(())
                }
            },
            Lvalue::Projection(ref proj) => {
                match proj.elem {
                    ProjectionElem::Deref => {
                        let base_ty = proj.base.ty(self.mir, self.tcx).to_ty(self.tcx);

                        // `Box<T>` owns its content, so mutable if its location is mutable
                        if base_ty.is_box() {
                            return self.is_mutable(&proj.base);
                        }

                        // Otherwise we check the kind of deref to decide
                        match base_ty.sty {
                            ty::TyRef(_, tnm) => {
                                match tnm.mutbl {
                                    // Shared borrowed data is never mutable
                                    hir::MutImmutable => Err(lvalue),
                                    // Mutably borrowed data is mutable, but only if we have a
                                    // unique path to the `&mut`
                                    hir::MutMutable => self.is_unique(&proj.base),
                                }
                            },
                            ty::TyRawPtr(tnm) => {
                                match tnm.mutbl {
                                    // `*const` raw pointers are not mutable
                                    hir::MutImmutable => Err(lvalue),
                                    // `*mut` raw pointers are always mutable, regardless of context
                                    // The users have to check by themselve.
                                    hir::MutMutable => Ok(()),
                                }
                            },
                            // Deref should only be for reference, pointers or boxes
                            _ => bug!("Deref of unexpected type: {:?}", base_ty)
                        }
                    },
                    // All other projections are owned by their base path, so mutable if
                    // base path is mutable
                    ProjectionElem::Field(..) |
                    ProjectionElem::Index(..) |
                    ProjectionElem::ConstantIndex{..} |
                    ProjectionElem::Subslice{..} |
                    ProjectionElem::Downcast(..) =>
                        self.is_mutable(&proj.base)
                }
            }
        }
    }

    /// Does this lvalue have a unique path
    fn is_unique<'d>(&self, lvalue: &'d Lvalue<'tcx>) -> Result<(), &'d Lvalue<'tcx>> {
        match *lvalue {
            Lvalue::Local(..) => {
                // Local variables are unique
                Ok(())
            },
            Lvalue::Static(..) => {
                // Static variables are not
                Err(lvalue)
            },
            Lvalue::Projection(ref proj) => {
                match proj.elem {
                    ProjectionElem::Deref => {
                        let base_ty = proj.base.ty(self.mir, self.tcx).to_ty(self.tcx);

                        // `Box<T>` referent is unique if box is a unique spot
                        if base_ty.is_box() {
                            return self.is_unique(&proj.base);
                        }

                        // Otherwise we check the kind of deref to decide
                        match base_ty.sty {
                            ty::TyRef(_, tnm) => {
                                match tnm.mutbl {
                                    // lvalue represent an aliased location
                                    hir::MutImmutable => Err(lvalue),
                                    // `&mut T` is as unique as the context in which it is found
                                    hir::MutMutable => self.is_unique(&proj.base),
                                }
                            },
                            ty::TyRawPtr(tnm) => {
                                match tnm.mutbl {
                                    // `*mut` can be aliased, but we leave it to user
                                    hir::MutMutable => Ok(()),
                                    // `*const` is treated the same as `*mut`
                                    hir::MutImmutable => Ok(()),
                                }
                            },
                            // Deref should only be for reference, pointers or boxes
                            _ => bug!("Deref of unexpected type: {:?}", base_ty)
                        }
                    },
                    // Other projections are unique if the base is unique
                    ProjectionElem::Field(..) |
                    ProjectionElem::Index(..) |
                    ProjectionElem::ConstantIndex{..} |
                    ProjectionElem::Subslice{..} |
                    ProjectionElem::Downcast(..) =>
                        self.is_unique(&proj.base)
                }
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum NoMovePathFound {
    ReachedStatic,
}

impl<'c, 'b, 'a: 'b+'c, 'gcx, 'tcx: 'a> MirBorrowckCtxt<'c, 'b, 'a, 'gcx, 'tcx> {
    fn each_borrow_involving_path<F>(&mut self,
                                     _context: Context,
                                     access_lvalue: (ShallowOrDeep, &Lvalue<'tcx>),
                                     flow_state: &InProgress<'b, 'gcx, 'tcx>,
                                     mut op: F)
        where F: FnMut(&mut Self, BorrowIndex, &BorrowData<'tcx>, &Lvalue) -> Control
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


    pub(super) struct Prefixes<'c, 'gcx: 'tcx, 'tcx: 'c> {
        mir: &'c Mir<'tcx>,
        tcx: TyCtxt<'c, 'gcx, 'tcx>,
        kind: PrefixSet,
        next: Option<&'c Lvalue<'tcx>>,
    }

    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub(super) enum PrefixSet {
        /// Doesn't stop until it returns the base case (a Local or
        /// Static prefix).
        All,
        /// Stops at any dereference.
        Shallow,
        /// Stops at the deref of a shared reference.
        Supporting,
    }

    impl<'c, 'b, 'a: 'b+'c, 'gcx, 'tcx: 'a> MirBorrowckCtxt<'c, 'b, 'a, 'gcx, 'tcx> {
        /// Returns an iterator over the prefixes of `lvalue`
        /// (inclusive) from longest to smallest, potentially
        /// terminating the iteration early based on `kind`.
        pub(super) fn prefixes<'d>(&self,
                                   lvalue: &'d Lvalue<'tcx>,
                                   kind: PrefixSet)
                                   -> Prefixes<'d, 'gcx, 'tcx> where 'b: 'd
        {
            Prefixes { next: Some(lvalue), kind, mir: self.mir, tcx: self.tcx }
        }
    }

    impl<'c, 'gcx, 'tcx> Iterator for Prefixes<'c, 'gcx, 'tcx> {
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
    fn report_use_of_moved_or_uninitialized(&mut self,
                           _context: Context,
                           desired_action: &str,
                           (lvalue, span): (&Lvalue, Span),
                           mpi: MovePathIndex,
                           curr_move_out: &IdxSetBuf<MoveOutIndex>) {

        let mois = self.move_data.path_map[mpi].iter().filter(
            |moi| curr_move_out.contains(moi)).collect::<Vec<_>>();

        if mois.is_empty() {
            self.tcx.cannot_act_on_uninitialized_variable(span,
                                                          desired_action,
                                                          &self.describe_lvalue(lvalue),
                                                          Origin::Mir)
                    .span_label(span, format!("use of possibly uninitialized `{}`",
                                              self.describe_lvalue(lvalue)))
                    .emit();
        } else {
            let msg = ""; //FIXME: add "partially " or "collaterally "

            let mut err = self.tcx.cannot_act_on_moved_value(span,
                                                             desired_action,
                                                             msg,
                                                             &self.describe_lvalue(lvalue),
                                                             Origin::Mir);
            err.span_label(span, format!("value {} here after move", desired_action));
            for moi in mois {
                let move_msg = ""; //FIXME: add " (into closure)"
                let move_span = self.mir.source_info(self.move_data.moves[*moi].source).span;
                if span == move_span {
                    err.span_label(span,
                                   format!("value moved{} here in previous iteration of loop",
                                           move_msg));
                } else {
                    err.span_label(move_span, format!("value moved{} here", move_msg));
                };
            }
            //FIXME: add note for closure
            err.emit();
        }
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

    /// Finds the span of arguments of a closure (within `maybe_closure_span`) and its usage of
    /// the local assigned at `location`.
    /// This is done by searching in statements succeeding `location`
    /// and originating from `maybe_closure_span`.
    fn find_closure_span(
        &self,
        maybe_closure_span: Span,
        location: Location,
    ) -> Option<(Span, Span)> {
        use rustc::hir::ExprClosure;
        use rustc::mir::AggregateKind;

        let local = if let StatementKind::Assign(Lvalue::Local(local), _) =
            self.mir[location.block].statements[location.statement_index].kind
        {
            local
        } else {
            return None;
        };

        for stmt in &self.mir[location.block].statements[location.statement_index + 1..] {
            if maybe_closure_span != stmt.source_info.span {
                break;
            }

            if let StatementKind::Assign(_, Rvalue::Aggregate(ref kind, ref lvs)) = stmt.kind {
                if let AggregateKind::Closure(def_id, _) = **kind {
                    debug!("find_closure_span: found closure {:?}", lvs);

                    return if let Some(node_id) = self.tcx.hir.as_local_node_id(def_id) {
                        let args_span = if let ExprClosure(_, _, _, span, _) =
                            self.tcx.hir.expect_expr(node_id).node
                        {
                            span
                        } else {
                            return None;
                        };

                        self.tcx
                            .with_freevars(node_id, |freevars| {
                                for (v, lv) in freevars.iter().zip(lvs) {
                                    if let Operand::Consume(Lvalue::Local(l)) = *lv {
                                        if local == l {
                                            debug!(
                                                "find_closure_span: found captured local {:?}",
                                                l
                                            );
                                            return Some(v.span);
                                        }
                                    }
                                }
                                None
                            })
                            .map(|var_span| (args_span, var_span))
                    } else {
                        None
                    };
                }
            }
        }

        None
    }

    fn report_conflicting_borrow(&mut self,
                                 context: Context,
                                 common_prefix: &Lvalue,
                                 (lvalue, span): (&Lvalue, Span),
                                 gen_borrow_kind: BorrowKind,
                                 issued_borrow: &BorrowData,
                                 end_issued_loan_span: Option<Span>) {
        use self::prefixes::IsPrefixOf;

        assert!(common_prefix.is_prefix_of(lvalue));
        assert!(common_prefix.is_prefix_of(&issued_borrow.lvalue));

        let issued_span = self.retrieve_borrow_span(issued_borrow);

        let new_closure_span = self.find_closure_span(span, context.loc);
        let span = new_closure_span.map(|(args, _)| args).unwrap_or(span);
        let old_closure_span = self.find_closure_span(issued_span, issued_borrow.location);
        let issued_span = old_closure_span.map(|(args, _)| args).unwrap_or(issued_span);

        let desc_lvalue = self.describe_lvalue(lvalue);

        // FIXME: supply non-"" `opt_via` when appropriate
        let mut err = match (gen_borrow_kind, "immutable", "mutable",
                             issued_borrow.kind, "immutable", "mutable") {
            (BorrowKind::Shared, lft, _, BorrowKind::Mut, _, rgt) |
            (BorrowKind::Mut, _, lft, BorrowKind::Shared, rgt, _) =>
                self.tcx.cannot_reborrow_already_borrowed(
                    span, &desc_lvalue, "", lft, issued_span,
                    "it", rgt, "", end_issued_loan_span, Origin::Mir),

            (BorrowKind::Mut, _, _, BorrowKind::Mut, _, _) =>
                self.tcx.cannot_mutably_borrow_multiply(
                    span, &desc_lvalue, "", issued_span,
                    "", end_issued_loan_span, Origin::Mir),

            (BorrowKind::Unique, _, _, BorrowKind::Unique, _, _) =>
                self.tcx.cannot_uniquely_borrow_by_two_closures(
                    span, &desc_lvalue, issued_span,
                    end_issued_loan_span, Origin::Mir),

            (BorrowKind::Unique, _, _, _, _, _) =>
                self.tcx.cannot_uniquely_borrow_by_one_closure(
                    span, &desc_lvalue, "",
                    issued_span, "it", "", end_issued_loan_span, Origin::Mir),

            (_, _, _, BorrowKind::Unique, _, _) =>
                self.tcx.cannot_reborrow_already_uniquely_borrowed(
                    span, &desc_lvalue, "it", "",
                    issued_span, "", end_issued_loan_span, Origin::Mir),

            (BorrowKind::Shared, _, _, BorrowKind::Shared, _, _) =>
                unreachable!(),
        };

        if let Some((_, var_span)) = old_closure_span {
            err.span_label(
                var_span,
                format!("previous borrow occurs due to use of `{}` in closure", desc_lvalue),
            );
        }

        if let Some((_, var_span)) = new_closure_span {
            err.span_label(
                var_span,
                format!("borrow occurs due to use of `{}` in closure", desc_lvalue),
            );
        }

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
                .span_label(span, "cannot assign twice to immutable variable")
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
        self.append_lvalue_to_string(lvalue, &mut buf, None);
        buf
    }

    // Appends end-user visible description of `lvalue` to `buf`.
    fn append_lvalue_to_string(&self, lvalue: &Lvalue, buf: &mut String, autoderef: Option<bool>) {
        match *lvalue {
            Lvalue::Local(local) => {
                self.append_local_to_string(local, buf, "_");
            }
            Lvalue::Static(ref static_) => {
                buf.push_str(&format!("{}", &self.tcx.item_name(static_.def_id)));
            }
            Lvalue::Projection(ref proj) => {
                let mut autoderef = autoderef.unwrap_or(false);
                let (prefix, suffix, index_operand) = match proj.elem {
                    ProjectionElem::Deref => {
                        if autoderef {
                            ("", format!(""), None)
                        } else {
                            ("(*", format!(")"), None)
                        }
                    },
                    ProjectionElem::Downcast(..) =>
                        ("",   format!(""), None), // (dont emit downcast info)
                    ProjectionElem::Field(field, _ty) => {
                        autoderef = true;
                        ("", format!(".{}", self.describe_field(&proj.base, field.index())), None)
                    },
                    ProjectionElem::Index(index) => {
                        autoderef = true;
                        ("",   format!(""), Some(index))
                    },
                    ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. } => {
                        autoderef = true;
                        // Since it isn't possible to borrow an element on a particular index and
                        // then use another while the borrow is held, don't output indices details
                        // to avoid confusing the end-user
                        ("",   format!("[..]"), None)
                    },
                };
                buf.push_str(prefix);
                self.append_lvalue_to_string(&proj.base, buf, Some(autoderef));
                if let Some(index) = index_operand {
                    buf.push_str("[");
                    self.append_local_to_string(index, buf, "..");
                    buf.push_str("]");
                } else {
                    buf.push_str(&suffix);
                }
            }
        }
    }

    // Appends end-user visible description of the `local` lvalue to `buf`. If `local` doesn't have
    // a name, then `none_string` is appended instead
    fn append_local_to_string(&self, local_index: Local, buf: &mut String, none_string: &str) {
        let local = &self.mir.local_decls[local_index];
        match local.name {
            Some(name) => buf.push_str(&format!("{}", name)),
            None => buf.push_str(none_string)
        }
    }

    // End-user visible description of the `field_index`nth field of `base`
    fn describe_field(&self, base: &Lvalue, field_index: usize) -> String {
        match *base {
            Lvalue::Local(local) => {
                let local = &self.mir.local_decls[local];
                self.describe_field_from_ty(&local.ty, field_index)
            },
            Lvalue::Static(ref static_) => {
                self.describe_field_from_ty(&static_.ty, field_index)
            },
            Lvalue::Projection(ref proj) => {
                match proj.elem {
                    ProjectionElem::Deref =>
                        self.describe_field(&proj.base, field_index),
                    ProjectionElem::Downcast(def, variant_index) =>
                        format!("{}", def.variants[variant_index].fields[field_index].name),
                    ProjectionElem::Field(_, field_type) =>
                        self.describe_field_from_ty(&field_type, field_index),
                    ProjectionElem::Index(..)
                    | ProjectionElem::ConstantIndex { .. }
                    | ProjectionElem::Subslice { .. } =>
                        format!("{}", self.describe_field(&proj.base, field_index)),
                }
            }
        }
    }

    // End-user visible description of the `field_index`nth field of `ty`
    fn describe_field_from_ty(&self, ty: &ty::Ty, field_index: usize) -> String {
        if ty.is_box() {
            // If the type is a box, the field is described from the boxed type
            self.describe_field_from_ty(&ty.boxed_ty(), field_index)
        }
        else {
            match ty.sty {
                ty::TyAdt(def, _) => {
                    if def.is_enum() {
                        format!("{}", field_index)
                    }
                    else {
                        format!("{}", def.struct_variant().fields[field_index].name)
                    }
                },
                ty::TyTuple(_, _) => {
                    format!("{}", field_index)
                },
                ty::TyRef(_, tnm) | ty::TyRawPtr(tnm) => {
                    self.describe_field_from_ty(&tnm.ty, field_index)
                },
                ty::TyArray(ty, _) | ty::TySlice(ty) => {
                    self.describe_field_from_ty(&ty, field_index)
                }
                _ => {
                    // Might need a revision when the fields in trait RFC is implemented
                    // (https://github.com/rust-lang/rfcs/pull/1546)
                    bug!("End-user description not implemented for field access on `{:?}`", ty.sty);
                }
            }
        }
    }

    // Retrieve span of given borrow from the current MIR representation
    fn retrieve_borrow_span(&self, borrow: &BorrowData) -> Span {
        self.mir.source_info(borrow.location).span
    }
}

impl<'c, 'b, 'a: 'b+'c, 'gcx, 'tcx: 'a> MirBorrowckCtxt<'c, 'b, 'a, 'gcx, 'tcx> {
    // FIXME (#16118): function intended to allow the borrow checker
    // to be less precise in its handling of Box while still allowing
    // moves out of a Box. They should be removed when/if we stop
    // treating Box specially (e.g. when/if DerefMove is added...)

    fn base_path<'d>(&self, lvalue: &'d Lvalue<'tcx>) -> &'d Lvalue<'tcx> {
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

impl<'b, 'gcx, 'tcx> InProgress<'b, 'gcx, 'tcx> {
    pub(super) fn new(borrows: DataflowResults<Borrows<'b, 'gcx, 'tcx>>,
                      inits: DataflowResults<MaybeInitializedLvals<'b, 'gcx, 'tcx>>,
                      uninits: DataflowResults<MaybeUninitializedLvals<'b, 'gcx, 'tcx>>,
                      move_out: DataflowResults<MovingOutStatements<'b, 'gcx, 'tcx>>)
                      -> Self {
        InProgress {
            borrows: FlowInProgress::new(borrows),
            inits: FlowInProgress::new(inits),
            uninits: FlowInProgress::new(uninits),
            move_outs: FlowInProgress::new(move_out)
        }
    }

    fn each_flow<XB, XI, XU, XM>(&mut self,
                                 mut xform_borrows: XB,
                                 mut xform_inits: XI,
                                 mut xform_uninits: XU,
                                 mut xform_move_outs: XM) where
        XB: FnMut(&mut FlowInProgress<Borrows<'b, 'gcx, 'tcx>>),
        XI: FnMut(&mut FlowInProgress<MaybeInitializedLvals<'b, 'gcx, 'tcx>>),
        XU: FnMut(&mut FlowInProgress<MaybeUninitializedLvals<'b, 'gcx, 'tcx>>),
        XM: FnMut(&mut FlowInProgress<MovingOutStatements<'b, 'gcx, 'tcx>>),
    {
        xform_borrows(&mut self.borrows);
        xform_inits(&mut self.inits);
        xform_uninits(&mut self.uninits);
        xform_move_outs(&mut self.move_outs);
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
        s.push_str("] ");

        s.push_str("move_out: [");
        let mut saw_one = false;
        self.move_outs.each_state_bit(|mpi_move_out| {
            if saw_one { s.push_str(", "); };
            saw_one = true;
            let move_out =
                &self.move_outs.base_results.operator().move_data().moves[mpi_move_out];
            s.push_str(&format!("{:?}", move_out));
        });
        s.push_str("]");

        return s;
    }
}

impl<'b, 'gcx, 'tcx> FlowInProgress<MaybeUninitializedLvals<'b, 'gcx, 'tcx>> {
    fn has_any_child_of(&self, mpi: MovePathIndex) -> Option<MovePathIndex> {
        let move_data = self.base_results.operator().move_data();

        let mut todo = vec![mpi];
        let mut push_siblings = false; // don't look at siblings of original `mpi`.
        while let Some(mpi) = todo.pop() {
            if self.curr_state.contains(&mpi) {
                return Some(mpi);
            }
            let move_path = &move_data.move_paths[mpi];
            if let Some(child) = move_path.first_child {
                todo.push(child);
            }
            if push_siblings {
                if let Some(sibling) = move_path.next_sibling {
                    todo.push(sibling);
                }
            } else {
                // after we've processed the original `mpi`, we should
                // always traverse the siblings of any of its
                // children.
                push_siblings = true;
            }
        }
        return None;
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
