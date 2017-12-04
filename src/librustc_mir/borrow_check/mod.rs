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
use rustc::hir::def_id::DefId;
use rustc::infer::InferCtxt;
use rustc::ty::{self, ParamEnv, TyCtxt};
use rustc::ty::maps::Providers;
use rustc::mir::{AssertMessage, BasicBlock, BorrowKind, Local, Location, Place};
use rustc::mir::{Mir, Mutability, Operand, Projection, ProjectionElem, Rvalue};
use rustc::mir::{Field, Statement, StatementKind, Terminator, TerminatorKind};

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::indexed_set::{self, IdxSetBuf};
use rustc_data_structures::indexed_vec::Idx;

use syntax::ast;
use syntax_pos::Span;

use dataflow::do_dataflow;
use dataflow::MoveDataParamEnv;
use dataflow::{BitDenotation, BlockSets, DataflowResults, DataflowResultsConsumer};
use dataflow::{MaybeInitializedLvals, MaybeUninitializedLvals};
use dataflow::{EverInitializedLvals, MovingOutStatements};
use dataflow::{BorrowData, BorrowIndex, Borrows};
use dataflow::move_paths::{IllegalMoveOriginKind, MoveError};
use dataflow::move_paths::{HasMoveData, LookupResult, MoveData, MoveOutIndex, MovePathIndex};
use util::borrowck_errors::{BorrowckErrors, Origin};

use self::MutateMode::{JustWrite, WriteAndRead};

pub(crate) mod nll;

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
        !tcx.has_attr(def_id, "rustc_mir_borrowck") && !tcx.sess.opts.borrowck_mode.use_mir()
            && !tcx.sess.opts.debugging_opts.nll
    } {
        return;
    }

    tcx.infer_ctxt().enter(|infcx| {
        let input_mir: &Mir = &input_mir.borrow();
        do_mir_borrowck(&infcx, input_mir, def_id);
    });
    debug!("mir_borrowck done");
}

fn do_mir_borrowck<'a, 'gcx, 'tcx>(
    infcx: &InferCtxt<'a, 'gcx, 'tcx>,
    input_mir: &Mir<'gcx>,
    def_id: DefId,
) {
    let tcx = infcx.tcx;
    let attributes = tcx.get_attrs(def_id);
    let param_env = tcx.param_env(def_id);
    let id = tcx.hir
        .as_local_node_id(def_id)
        .expect("do_mir_borrowck: non-local DefId");

    // Make our own copy of the MIR. This copy will be modified (in place) to
    // contain non-lexical lifetimes. It will have a lifetime tied
    // to the inference context.
    let mut mir: Mir<'tcx> = input_mir.clone();
    let free_regions = if !tcx.sess.opts.debugging_opts.nll {
        None
    } else {
        let mir = &mut mir;

        // Replace all regions with fresh inference variables.
        Some(nll::replace_regions_in_mir(infcx, def_id, mir))
    };
    let mir = &mir;

    let move_data: MoveData<'tcx> = match MoveData::gather_moves(mir, tcx) {
        Ok(move_data) => move_data,
        Err((move_data, move_errors)) => {
            for move_error in move_errors {
                let (span, kind): (Span, IllegalMoveOriginKind) = match move_error {
                    MoveError::UnionMove { .. } => {
                        unimplemented!("dont know how to report union move errors yet.")
                    }
                    MoveError::IllegalMove {
                        cannot_move_out_of: o,
                    } => (o.span, o.kind),
                };
                let origin = Origin::Mir;
                let mut err = match kind {
                    IllegalMoveOriginKind::Static => {
                        tcx.cannot_move_out_of(span, "static item", origin)
                    }
                    IllegalMoveOriginKind::BorrowedContent => {
                        tcx.cannot_move_out_of(span, "borrowed content", origin)
                    }
                    IllegalMoveOriginKind::InteriorOfTypeWithDestructor { container_ty: ty } => {
                        tcx.cannot_move_out_of_interior_of_drop(span, ty, origin)
                    }
                    IllegalMoveOriginKind::InteriorOfSliceOrArray { ty, is_index } => {
                        tcx.cannot_move_out_of_interior_noncopy(span, ty, is_index, origin)
                    }
                };
                err.emit();
            }
            move_data
        }
    };

    let mdpe = MoveDataParamEnv {
        move_data: move_data,
        param_env: param_env,
    };
    let dead_unwinds = IdxSetBuf::new_empty(mir.basic_blocks().len());
    let mut flow_inits = FlowInProgress::new(do_dataflow(
        tcx,
        mir,
        id,
        &attributes,
        &dead_unwinds,
        MaybeInitializedLvals::new(tcx, mir, &mdpe),
        |bd, i| &bd.move_data().move_paths[i],
    ));
    let flow_uninits = FlowInProgress::new(do_dataflow(
        tcx,
        mir,
        id,
        &attributes,
        &dead_unwinds,
        MaybeUninitializedLvals::new(tcx, mir, &mdpe),
        |bd, i| &bd.move_data().move_paths[i],
    ));
    let flow_move_outs = FlowInProgress::new(do_dataflow(
        tcx,
        mir,
        id,
        &attributes,
        &dead_unwinds,
        MovingOutStatements::new(tcx, mir, &mdpe),
        |bd, i| &bd.move_data().moves[i],
    ));
    let flow_ever_inits = FlowInProgress::new(do_dataflow(
        tcx,
        mir,
        id,
        &attributes,
        &dead_unwinds,
        EverInitializedLvals::new(tcx, mir, &mdpe),
        |bd, i| &bd.move_data().inits[i],
    ));

    // If we are in non-lexical mode, compute the non-lexical lifetimes.
    let opt_regioncx = if let Some(free_regions) = free_regions {
        Some(nll::compute_regions(
            infcx,
            def_id,
            free_regions,
            mir,
            param_env,
            &mut flow_inits,
            &mdpe.move_data,
        ))
    } else {
        assert!(!tcx.sess.opts.debugging_opts.nll);
        None
    };
    let flow_inits = flow_inits; // remove mut

    let mut mbcx = MirBorrowckCtxt {
        tcx: tcx,
        mir: mir,
        node_id: id,
        move_data: &mdpe.move_data,
        param_env: param_env,
        storage_dead_or_drop_error_reported: FxHashSet(),
    };

    let flow_borrows = FlowInProgress::new(do_dataflow(
        tcx,
        mir,
        id,
        &attributes,
        &dead_unwinds,
        Borrows::new(tcx, mir, opt_regioncx),
        |bd, i| bd.location(i),
    ));

    let mut state = InProgress::new(
        flow_borrows,
        flow_inits,
        flow_uninits,
        flow_move_outs,
        flow_ever_inits,
    );

    mbcx.analyze_results(&mut state); // entry point for DataflowResultsConsumer
}

#[allow(dead_code)]
pub struct MirBorrowckCtxt<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    mir: &'cx Mir<'tcx>,
    node_id: ast::NodeId,
    move_data: &'cx MoveData<'tcx>,
    param_env: ParamEnv<'gcx>,
    /// This field keeps track of when storage dead or drop errors are reported
    /// in order to stop duplicate error reporting and identify the conditions required
    /// for a "temporary value dropped here while still borrowed" error. See #45360.
    storage_dead_or_drop_error_reported: FxHashSet<Local>,
}

// (forced to be `pub` due to its use as an associated type below.)
pub struct InProgress<'b, 'gcx: 'tcx, 'tcx: 'b> {
    borrows: FlowInProgress<Borrows<'b, 'gcx, 'tcx>>,
    inits: FlowInProgress<MaybeInitializedLvals<'b, 'gcx, 'tcx>>,
    uninits: FlowInProgress<MaybeUninitializedLvals<'b, 'gcx, 'tcx>>,
    move_outs: FlowInProgress<MovingOutStatements<'b, 'gcx, 'tcx>>,
    ever_inits: FlowInProgress<EverInitializedLvals<'b, 'gcx, 'tcx>>,
}

struct FlowInProgress<BD>
where
    BD: BitDenotation,
{
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
impl<'cx, 'gcx, 'tcx> DataflowResultsConsumer<'cx, 'tcx> for MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    type FlowState = InProgress<'cx, 'gcx, 'tcx>;

    fn mir(&self) -> &'cx Mir<'tcx> {
        self.mir
    }

    fn reset_to_entry_of(&mut self, bb: BasicBlock, flow_state: &mut Self::FlowState) {
        flow_state.each_flow(
            |b| b.reset_to_entry_of(bb),
            |i| i.reset_to_entry_of(bb),
            |u| u.reset_to_entry_of(bb),
            |m| m.reset_to_entry_of(bb),
            |e| e.reset_to_entry_of(bb),
        );
    }

    fn reconstruct_statement_effect(
        &mut self,
        location: Location,
        flow_state: &mut Self::FlowState,
    ) {
        flow_state.each_flow(
            |b| b.reconstruct_statement_effect(location),
            |i| i.reconstruct_statement_effect(location),
            |u| u.reconstruct_statement_effect(location),
            |m| m.reconstruct_statement_effect(location),
            |e| e.reconstruct_statement_effect(location),
        );
    }

    fn apply_local_effect(&mut self, _location: Location, flow_state: &mut Self::FlowState) {
        flow_state.each_flow(
            |b| b.apply_local_effect(),
            |i| i.apply_local_effect(),
            |u| u.apply_local_effect(),
            |m| m.apply_local_effect(),
            |e| e.apply_local_effect(),
        );
    }

    fn reconstruct_terminator_effect(
        &mut self,
        location: Location,
        flow_state: &mut Self::FlowState,
    ) {
        flow_state.each_flow(
            |b| b.reconstruct_terminator_effect(location),
            |i| i.reconstruct_terminator_effect(location),
            |u| u.reconstruct_terminator_effect(location),
            |m| m.reconstruct_terminator_effect(location),
            |e| e.reconstruct_terminator_effect(location),
        );
    }

    fn visit_block_entry(&mut self, bb: BasicBlock, flow_state: &Self::FlowState) {
        let summary = flow_state.summary();
        debug!("MirBorrowckCtxt::process_block({:?}): {}", bb, summary);
    }

    fn visit_statement_entry(
        &mut self,
        location: Location,
        stmt: &Statement<'tcx>,
        flow_state: &Self::FlowState,
    ) {
        let summary = flow_state.summary();
        debug!(
            "MirBorrowckCtxt::process_statement({:?}, {:?}): {}",
            location,
            stmt,
            summary
        );
        let span = stmt.source_info.span;
        match stmt.kind {
            StatementKind::Assign(ref lhs, ref rhs) => {
                // NOTE: NLL RFC calls for *shallow* write; using Deep
                // for short-term compat w/ AST-borrowck. Also, switch
                // to shallow requires to dataflow: "if this is an
                // assignment `place = <rvalue>`, then any loan for some
                // path P of which `place` is a prefix is killed."
                self.mutate_place(
                    ContextKind::AssignLhs.new(location),
                    (lhs, span),
                    Deep,
                    JustWrite,
                    flow_state,
                );

                self.consume_rvalue(
                    ContextKind::AssignRhs.new(location),
                    (rhs, span),
                    location,
                    flow_state,
                );
            }
            StatementKind::SetDiscriminant {
                ref place,
                variant_index: _,
            } => {
                self.mutate_place(
                    ContextKind::SetDiscrim.new(location),
                    (place, span),
                    Shallow(Some(ArtificialField::Discriminant)),
                    JustWrite,
                    flow_state,
                );
            }
            StatementKind::InlineAsm {
                ref asm,
                ref outputs,
                ref inputs,
            } => {
                let context = ContextKind::InlineAsm.new(location);
                for (o, output) in asm.outputs.iter().zip(outputs) {
                    if o.is_indirect {
                        // FIXME(eddyb) indirect inline asm outputs should
                        // be encoeded through MIR place derefs instead.
                        self.access_place(
                            context,
                            (output, span),
                            (Deep, Read(ReadKind::Copy)),
                            LocalMutationIsAllowed::No,
                            flow_state,
                        );
                        self.check_if_path_is_moved(
                            context,
                            InitializationRequiringAction::Use,
                            (output, span),
                            flow_state,
                        );
                    } else {
                        self.mutate_place(
                            context,
                            (output, span),
                            Deep,
                            if o.is_rw { WriteAndRead } else { JustWrite },
                            flow_state,
                        );
                    }
                }
                for input in inputs {
                    self.consume_operand(context, (input, span), flow_state);
                }
            }
            StatementKind::EndRegion(ref _rgn) => {
                // ignored when consuming results (update to
                // flow_state already handled).
            }
            StatementKind::Nop | StatementKind::Validate(..) | StatementKind::StorageLive(..) => {
                // `Nop`, `Validate`, and `StorageLive` are irrelevant
                // to borrow check.
            }

            StatementKind::StorageDead(local) => {
                self.access_place(
                    ContextKind::StorageDead.new(location),
                    (&Place::Local(local), span),
                    (Shallow(None), Write(WriteKind::StorageDeadOrDrop)),
                    LocalMutationIsAllowed::Yes,
                    flow_state,
                );
            }
        }
    }

    fn visit_terminator_entry(
        &mut self,
        location: Location,
        term: &Terminator<'tcx>,
        flow_state: &Self::FlowState,
    ) {
        let loc = location;
        let summary = flow_state.summary();
        debug!(
            "MirBorrowckCtxt::process_terminator({:?}, {:?}): {}",
            location,
            term,
            summary
        );
        let span = term.source_info.span;
        match term.kind {
            TerminatorKind::SwitchInt {
                ref discr,
                switch_ty: _,
                values: _,
                targets: _,
            } => {
                self.consume_operand(ContextKind::SwitchInt.new(loc), (discr, span), flow_state);
            }
            TerminatorKind::Drop {
                location: ref drop_place,
                target: _,
                unwind: _,
            } => {
                self.access_place(
                    ContextKind::Drop.new(loc),
                    (drop_place, span),
                    (Deep, Write(WriteKind::StorageDeadOrDrop)),
                    LocalMutationIsAllowed::Yes,
                    flow_state,
                );
            }
            TerminatorKind::DropAndReplace {
                location: ref drop_place,
                value: ref new_value,
                target: _,
                unwind: _,
            } => {
                self.mutate_place(
                    ContextKind::DropAndReplace.new(loc),
                    (drop_place, span),
                    Deep,
                    JustWrite,
                    flow_state,
                );
                self.consume_operand(
                    ContextKind::DropAndReplace.new(loc),
                    (new_value, span),
                    flow_state,
                );
            }
            TerminatorKind::Call {
                ref func,
                ref args,
                ref destination,
                cleanup: _,
            } => {
                self.consume_operand(ContextKind::CallOperator.new(loc), (func, span), flow_state);
                for arg in args {
                    self.consume_operand(
                        ContextKind::CallOperand.new(loc),
                        (arg, span),
                        flow_state,
                    );
                }
                if let Some((ref dest, _ /*bb*/)) = *destination {
                    self.mutate_place(
                        ContextKind::CallDest.new(loc),
                        (dest, span),
                        Deep,
                        JustWrite,
                        flow_state,
                    );
                }
            }
            TerminatorKind::Assert {
                ref cond,
                expected: _,
                ref msg,
                target: _,
                cleanup: _,
            } => {
                self.consume_operand(ContextKind::Assert.new(loc), (cond, span), flow_state);
                match *msg {
                    AssertMessage::BoundsCheck { ref len, ref index } => {
                        self.consume_operand(ContextKind::Assert.new(loc), (len, span), flow_state);
                        self.consume_operand(
                            ContextKind::Assert.new(loc),
                            (index, span),
                            flow_state,
                        );
                    }
                    AssertMessage::Math(_ /*const_math_err*/) => {}
                    AssertMessage::GeneratorResumedAfterReturn => {}
                    AssertMessage::GeneratorResumedAfterPanic => {}
                }
            }

            TerminatorKind::Yield {
                ref value,
                resume: _,
                drop: _,
            } => {
                self.consume_operand(ContextKind::Yield.new(loc), (value, span), flow_state);
            }

            TerminatorKind::Resume | TerminatorKind::Return | TerminatorKind::GeneratorDrop => {
                // Returning from the function implicitly kills storage for all locals and statics.
                // Often, the storage will already have been killed by an explicit
                // StorageDead, but we don't always emit those (notably on unwind paths),
                // so this "extra check" serves as a kind of backup.
                let domain = flow_state.borrows.base_results.operator();
                let data = domain.borrows();
                flow_state.borrows.with_elems_outgoing(|borrows| {
                    for i in borrows {
                        let borrow = &data[i];

                        if self.place_is_invalidated_at_exit(&borrow.place) {
                            debug!("borrow conflicts at exit {:?}", borrow);
                            let borrow_span = self.mir.source_info(borrow.location).span;
                            // FIXME: should be talking about the region lifetime instead
                            // of just a span here.
                            let end_span = domain.opt_region_end_span(&borrow.region);

                            self.report_borrowed_value_does_not_live_long_enough(
                                ContextKind::StorageDead.new(loc),
                                (&borrow.place, borrow_span),
                                end_span,
                            )
                        }
                    }
                });
            }
            TerminatorKind::Goto { target: _ } |
            TerminatorKind::Unreachable |
            TerminatorKind::FalseEdges { .. } => {
                // no data used, thus irrelevant to borrowck
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum MutateMode {
    JustWrite,
    WriteAndRead,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Control {
    Continue,
    Break,
}

use self::ShallowOrDeep::{Deep, Shallow};
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
    /// through the given place may be invalidated or accesses by
    /// this action."
    Deep,
}

/// Kind of access to a value: read or write
/// (For informational purposes only)
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

/// Kind of read access to a value
/// (For informational purposes only)
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum ReadKind {
    Borrow(BorrowKind),
    Copy,
}

/// Kind of write access to a value
/// (For informational purposes only)
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum WriteKind {
    StorageDeadOrDrop,
    MutableBorrow(BorrowKind),
    Mutate,
    Move,
}

/// When checking permissions for a place access, this flag is used to indicate that an immutable
/// local place can be mutated.
///
/// FIXME: @nikomatsakis suggested that this flag could be removed with the following modifications:
/// - Merge `check_access_permissions()` and `check_if_reassignment_to_immutable_state()`
/// - Split `is_mutable()` into `is_assignable()` (can be directly assigned) and
///   `is_declared_mutable()`
/// - Take flow state into consideration in `is_assignable()` for local variables
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum LocalMutationIsAllowed {
    Yes,
    No,
}

#[derive(Copy, Clone)]
enum InitializationRequiringAction {
    Update,
    Borrow,
    Use,
    Assignment,
}

impl InitializationRequiringAction {
    fn as_noun(self) -> &'static str {
        match self {
            InitializationRequiringAction::Update => "update",
            InitializationRequiringAction::Borrow => "borrow",
            InitializationRequiringAction::Use => "use",
            InitializationRequiringAction::Assignment => "assign",
        }
    }

    fn as_verb_in_past_tense(self) -> &'static str {
        match self {
            InitializationRequiringAction::Update => "updated",
            InitializationRequiringAction::Borrow => "borrowed",
            InitializationRequiringAction::Use => "used",
            InitializationRequiringAction::Assignment => "assigned",
        }
    }
}

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    /// Checks an access to the given place to see if it is allowed. Examines the set of borrows
    /// that are in scope, as well as which paths have been initialized, to ensure that (a) the
    /// place is initialized and (b) it is not borrowed in some way that would prevent this
    /// access.
    ///
    /// Returns true if an error is reported, false otherwise.
    fn access_place(
        &mut self,
        context: Context,
        place_span: (&Place<'tcx>, Span),
        kind: (ShallowOrDeep, ReadOrWrite),
        is_local_mutation_allowed: LocalMutationIsAllowed,
        flow_state: &InProgress<'cx, 'gcx, 'tcx>,
    ) {
        let (sd, rw) = kind;

        let storage_dead_or_drop_local = match (place_span.0, rw) {
            (&Place::Local(local), Write(WriteKind::StorageDeadOrDrop)) => Some(local),
            _ => None,
        };

        // Check if error has already been reported to stop duplicate reporting.
        if let Some(local) = storage_dead_or_drop_local {
            if self.storage_dead_or_drop_error_reported.contains(&local) {
                return;
            }
        }

        // Check permissions
        let mut error_reported =
            self.check_access_permissions(place_span, rw, is_local_mutation_allowed);

        self.each_borrow_involving_path(
            context,
            (sd, place_span.0),
            flow_state,
            |this, _index, borrow, common_prefix| match (rw, borrow.kind) {
                (Read(_), BorrowKind::Shared) => Control::Continue,
                (Read(kind), BorrowKind::Unique) | (Read(kind), BorrowKind::Mut) => {
                    match kind {
                        ReadKind::Copy => {
                            error_reported = true;
                            this.report_use_while_mutably_borrowed(context, place_span, borrow)
                        }
                        ReadKind::Borrow(bk) => {
                            let end_issued_loan_span = flow_state
                                .borrows
                                .base_results
                                .operator()
                                .opt_region_end_span(&borrow.region);
                            error_reported = true;
                            this.report_conflicting_borrow(
                                context,
                                common_prefix,
                                place_span,
                                bk,
                                &borrow,
                                end_issued_loan_span,
                            )
                        }
                    }
                    Control::Break
                }
                (Write(kind), _) => {
                    match kind {
                        WriteKind::MutableBorrow(bk) => {
                            let end_issued_loan_span = flow_state
                                .borrows
                                .base_results
                                .operator()
                                .opt_region_end_span(&borrow.region);
                            error_reported = true;
                            this.report_conflicting_borrow(
                                context,
                                common_prefix,
                                place_span,
                                bk,
                                &borrow,
                                end_issued_loan_span,
                            )
                        }
                        WriteKind::StorageDeadOrDrop => {
                            let end_span = flow_state
                                .borrows
                                .base_results
                                .operator()
                                .opt_region_end_span(&borrow.region);
                            error_reported = true;
                            this.report_borrowed_value_does_not_live_long_enough(
                                context,
                                place_span,
                                end_span,
                            )
                        }
                        WriteKind::Mutate => {
                            error_reported = true;
                            this.report_illegal_mutation_of_borrowed(context, place_span, borrow)
                        }
                        WriteKind::Move => {
                            error_reported = true;
                            this.report_move_out_while_borrowed(context, place_span, &borrow)
                        }
                    }
                    Control::Break
                }
            },
        );

        if error_reported {
            if let Some(local) = storage_dead_or_drop_local {
                self.storage_dead_or_drop_error_reported.insert(local);
            }
        }
    }

    fn mutate_place(
        &mut self,
        context: Context,
        place_span: (&Place<'tcx>, Span),
        kind: ShallowOrDeep,
        mode: MutateMode,
        flow_state: &InProgress<'cx, 'gcx, 'tcx>,
    ) {
        // Write of P[i] or *P, or WriteAndRead of any P, requires P init'd.
        match mode {
            MutateMode::WriteAndRead => {
                self.check_if_path_is_moved(
                    context,
                    InitializationRequiringAction::Update,
                    place_span,
                    flow_state,
                );
            }
            MutateMode::JustWrite => {
                self.check_if_assigned_path_is_moved(context, place_span, flow_state);
            }
        }

        self.access_place(
            context,
            place_span,
            (kind, Write(WriteKind::Mutate)),
            LocalMutationIsAllowed::Yes,
            flow_state,
        );

        // check for reassignments to immutable local variables
        self.check_if_reassignment_to_immutable_state(context, place_span, flow_state);
    }

    fn consume_rvalue(
        &mut self,
        context: Context,
        (rvalue, span): (&Rvalue<'tcx>, Span),
        _location: Location,
        flow_state: &InProgress<'cx, 'gcx, 'tcx>,
    ) {
        match *rvalue {
            Rvalue::Ref(_ /*rgn*/, bk, ref place) => {
                let access_kind = match bk {
                    BorrowKind::Shared => (Deep, Read(ReadKind::Borrow(bk))),
                    BorrowKind::Unique | BorrowKind::Mut => {
                        (Deep, Write(WriteKind::MutableBorrow(bk)))
                    }
                };
                self.access_place(
                    context,
                    (place, span),
                    access_kind,
                    LocalMutationIsAllowed::No,
                    flow_state,
                );
                self.check_if_path_is_moved(
                    context,
                    InitializationRequiringAction::Borrow,
                    (place, span),
                    flow_state,
                );
            }

            Rvalue::Use(ref operand) |
            Rvalue::Repeat(ref operand, _) |
            Rvalue::UnaryOp(_ /*un_op*/, ref operand) |
            Rvalue::Cast(_ /*cast_kind*/, ref operand, _ /*ty*/) => {
                self.consume_operand(context, (operand, span), flow_state)
            }

            Rvalue::Len(ref place) | Rvalue::Discriminant(ref place) => {
                let af = match *rvalue {
                    Rvalue::Len(..) => ArtificialField::ArrayLength,
                    Rvalue::Discriminant(..) => ArtificialField::Discriminant,
                    _ => unreachable!(),
                };
                self.access_place(
                    context,
                    (place, span),
                    (Shallow(Some(af)), Read(ReadKind::Copy)),
                    LocalMutationIsAllowed::No,
                    flow_state,
                );
                self.check_if_path_is_moved(
                    context,
                    InitializationRequiringAction::Use,
                    (place, span),
                    flow_state,
                );
            }

            Rvalue::BinaryOp(_bin_op, ref operand1, ref operand2) |
            Rvalue::CheckedBinaryOp(_bin_op, ref operand1, ref operand2) => {
                self.consume_operand(context, (operand1, span), flow_state);
                self.consume_operand(context, (operand2, span), flow_state);
            }

            Rvalue::NullaryOp(_op, _ty) => {
                // nullary ops take no dynamic input; no borrowck effect.
                //
                // FIXME: is above actually true? Do we want to track
                // the fact that uninitialized data can be created via
                // `NullOp::Box`?
            }

            Rvalue::Aggregate(ref _aggregate_kind, ref operands) => for operand in operands {
                self.consume_operand(context, (operand, span), flow_state);
            },
        }
    }

    fn consume_operand(
        &mut self,
        context: Context,
        (operand, span): (&Operand<'tcx>, Span),
        flow_state: &InProgress<'cx, 'gcx, 'tcx>,
    ) {
        match *operand {
            Operand::Copy(ref place) => {
                // copy of place: check if this is "copy of frozen path"
                // (FIXME: see check_loans.rs)
                self.access_place(
                    context,
                    (place, span),
                    (Deep, Read(ReadKind::Copy)),
                    LocalMutationIsAllowed::No,
                    flow_state,
                );

                // Finally, check if path was already moved.
                self.check_if_path_is_moved(
                    context,
                    InitializationRequiringAction::Use,
                    (place, span),
                    flow_state,
                );
            }
            Operand::Move(ref place) => {
                // move of place: check if this is move of already borrowed path
                self.access_place(
                    context,
                    (place, span),
                    (Deep, Write(WriteKind::Move)),
                    LocalMutationIsAllowed::Yes,
                    flow_state,
                );

                // Finally, check if path was already moved.
                self.check_if_path_is_moved(
                    context,
                    InitializationRequiringAction::Use,
                    (place, span),
                    flow_state,
                );
            }
            Operand::Constant(_) => {}
        }
    }

    /// Returns whether a borrow of this place is invalidated when the function
    /// exits
    fn place_is_invalidated_at_exit(&self, place: &Place<'tcx>) -> bool {
        debug!("place_is_invalidated_at_exit({:?})", place);
        let root_place = self.prefixes(place, PrefixSet::All).last().unwrap();

        // FIXME(nll-rfc#40): do more precise destructor tracking here. For now
        // we just know that all locals are dropped at function exit (otherwise
        // we'll have a memory leak) and assume that all statics have a destructor.
        let (might_be_alive, will_be_dropped) = match root_place {
            Place::Static(statik) => {
                // Thread-locals might be dropped after the function exits, but
                // "true" statics will never be.
                let is_thread_local = self.tcx
                    .get_attrs(statik.def_id)
                    .iter()
                    .any(|attr| attr.check_name("thread_local"));

                (true, is_thread_local)
            }
            Place::Local(_) => {
                // Locals are always dropped at function exit, and if they
                // have a destructor it would've been called already.
                (false, true)
            }
            Place::Projection(..) => {
                bug!("root of {:?} is a projection ({:?})?", place, root_place)
            }
        };

        if !will_be_dropped {
            debug!(
                "place_is_invalidated_at_exit({:?}) - won't be dropped",
                place
            );
            return false;
        }

        // FIXME: replace this with a proper borrow_conflicts_with_place when
        // that is merged.
        let prefix_set = if might_be_alive {
            PrefixSet::Supporting
        } else {
            PrefixSet::Shallow
        };

        self.prefixes(place, prefix_set)
            .any(|prefix| prefix == root_place)
    }
}

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    fn check_if_reassignment_to_immutable_state(
        &mut self,
        context: Context,
        (place, span): (&Place<'tcx>, Span),
        flow_state: &InProgress<'cx, 'gcx, 'tcx>,
    ) {
        let move_data = self.move_data;

        // determine if this path has a non-mut owner (and thus needs checking).
        if let Ok(()) = self.is_mutable(place, LocalMutationIsAllowed::No) {
            return;
        }

        if let Err(_) = self.is_mutable(place, LocalMutationIsAllowed::Yes) {
            return;
        }

        match self.move_path_closest_to(place) {
            Ok(mpi) => for ii in &move_data.init_path_map[mpi] {
                if flow_state.ever_inits.curr_state.contains(ii) {
                    let first_assign_span = self.move_data.inits[*ii].span;
                    self.report_illegal_reassignment(context, (place, span), first_assign_span);
                    break;
                }
            },
            Err(NoMovePathFound::ReachedStatic) => {
                let item_msg = match self.describe_place(place) {
                    Some(name) => format!("immutable static item `{}`", name),
                    None => "immutable static item".to_owned(),
                };
                self.tcx.sess.delay_span_bug(
                    span,
                    &format!(
                        "cannot assign to {}, should have been caught by \
                         `check_access_permissions()`",
                        item_msg
                    ),
                );
            }
        }
    }

    fn check_if_path_is_moved(
        &mut self,
        context: Context,
        desired_action: InitializationRequiringAction,
        place_span: (&Place<'tcx>, Span),
        flow_state: &InProgress<'cx, 'gcx, 'tcx>,
    ) {
        // FIXME: analogous code in check_loans first maps `place` to
        // its base_path ... but is that what we want here?
        let place = self.base_path(place_span.0);

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
        // state for the place scenario.
        //
        // This code covers scenarios 1, 2, and 4.

        debug!("check_if_path_is_moved part1 place: {:?}", place);
        match self.move_path_closest_to(place) {
            Ok(mpi) => {
                if maybe_uninits.curr_state.contains(&mpi) {
                    self.report_use_of_moved_or_uninitialized(
                        context,
                        desired_action,
                        place_span,
                        mpi,
                        curr_move_outs,
                    );
                    return; // don't bother finding other problems.
                }
            }
            Err(NoMovePathFound::ReachedStatic) => {
                // Okay: we do not build MoveData for static variables
            } // Only query longest prefix with a MovePath, not further
              // ancestors; dataflow recurs on children when parents
              // move (to support partial (re)inits).
              //
              // (I.e. querying parents breaks scenario 8; but may want
              // to do such a query based on partial-init feature-gate.)
        }

        // A move of any shallow suffix of `place` also interferes
        // with an attempt to use `place`. This is scenario 3 above.
        //
        // (Distinct from handling of scenarios 1+2+4 above because
        // `place` does not interfere with suffixes of its prefixes,
        // e.g. `a.b.c` does not interfere with `a.b.d`)

        debug!("check_if_path_is_moved part2 place: {:?}", place);
        if let Some(mpi) = self.move_path_for_place(place) {
            if let Some(child_mpi) = maybe_uninits.has_any_child_of(mpi) {
                self.report_use_of_moved_or_uninitialized(
                    context,
                    desired_action,
                    place_span,
                    child_mpi,
                    curr_move_outs,
                );
                return; // don't bother finding other problems.
            }
        }
    }

    /// Currently MoveData does not store entries for all places in
    /// the input MIR. For example it will currently filter out
    /// places that are Copy; thus we do not track places of shared
    /// reference type. This routine will walk up a place along its
    /// prefixes, searching for a foundational place that *is*
    /// tracked in the MoveData.
    ///
    /// An Err result includes a tag indicated why the search failed.
    /// Currenly this can only occur if the place is built off of a
    /// static variable, as we do not track those in the MoveData.
    fn move_path_closest_to(
        &mut self,
        place: &Place<'tcx>,
    ) -> Result<MovePathIndex, NoMovePathFound> {
        let mut last_prefix = place;
        for prefix in self.prefixes(place, PrefixSet::All) {
            if let Some(mpi) = self.move_path_for_place(prefix) {
                return Ok(mpi);
            }
            last_prefix = prefix;
        }
        match *last_prefix {
            Place::Local(_) => panic!("should have move path for every Local"),
            Place::Projection(_) => panic!("PrefixSet::All meant dont stop for Projection"),
            Place::Static(_) => return Err(NoMovePathFound::ReachedStatic),
        }
    }

    fn move_path_for_place(&mut self, place: &Place<'tcx>) -> Option<MovePathIndex> {
        // If returns None, then there is no move path corresponding
        // to a direct owner of `place` (which means there is nothing
        // that borrowck tracks for its analysis).

        match self.move_data.rev_lookup.find(place) {
            LookupResult::Parent(_) => None,
            LookupResult::Exact(mpi) => Some(mpi),
        }
    }

    fn check_if_assigned_path_is_moved(
        &mut self,
        context: Context,
        (place, span): (&Place<'tcx>, Span),
        flow_state: &InProgress<'cx, 'gcx, 'tcx>,
    ) {
        // recur down place; dispatch to check_if_path_is_moved when necessary
        let mut place = place;
        loop {
            match *place {
                Place::Local(_) | Place::Static(_) => {
                    // assigning to `x` does not require `x` be initialized.
                    break;
                }
                Place::Projection(ref proj) => {
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
                                        context, InitializationRequiringAction::Assignment,
                                        (base, span), flow_state);

                                    // (base initialized; no need to
                                    // recur further)
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }

                    place = base;
                    continue;
                }
            }
        }
    }

    /// Check the permissions for the given place and read or write kind
    ///
    /// Returns true if an error is reported, false otherwise.
    fn check_access_permissions(
        &self,
        (place, span): (&Place<'tcx>, Span),
        kind: ReadOrWrite,
        is_local_mutation_allowed: LocalMutationIsAllowed,
    ) -> bool {
        debug!(
            "check_access_permissions({:?}, {:?}, {:?})",
            place,
            kind,
            is_local_mutation_allowed
        );
        let mut error_reported = false;
        match kind {
            Write(WriteKind::MutableBorrow(BorrowKind::Unique)) => {
                if let Err(_place_err) = self.is_unique(place) {
                    span_bug!(span, "&unique borrow for {:?} should not fail", place);
                }
            }
            Write(WriteKind::MutableBorrow(BorrowKind::Mut)) => if let Err(place_err) =
                self.is_mutable(place, is_local_mutation_allowed)
            {
                error_reported = true;

                let item_msg = match self.describe_place(place) {
                    Some(name) => format!("immutable item `{}`", name),
                    None => "immutable item".to_owned(),
                };

                let mut err = self.tcx
                    .cannot_borrow_path_as_mutable(span, &item_msg, Origin::Mir);
                err.span_label(span, "cannot borrow as mutable");

                if place != place_err {
                    if let Some(name) = self.describe_place(place_err) {
                        err.note(&format!("Value not mutable causing this error: `{}`", name));
                    }
                }

                err.emit();
            },
            Write(WriteKind::Mutate) => {
                if let Err(place_err) = self.is_mutable(place, is_local_mutation_allowed) {
                    error_reported = true;

                    let item_msg = match self.describe_place(place) {
                        Some(name) => format!("immutable item `{}`", name),
                        None => "immutable item".to_owned(),
                    };

                    let mut err = self.tcx.cannot_assign(span, &item_msg, Origin::Mir);
                    err.span_label(span, "cannot mutate");

                    if place != place_err {
                        if let Some(name) = self.describe_place(place_err) {
                            err.note(&format!("Value not mutable causing this error: `{}`", name));
                        }
                    }

                    err.emit();
                }
            }
            Write(WriteKind::Move) |
            Write(WriteKind::StorageDeadOrDrop) |
            Write(WriteKind::MutableBorrow(BorrowKind::Shared)) => {
                if let Err(_place_err) = self.is_mutable(place, is_local_mutation_allowed) {
                    self.tcx.sess.delay_span_bug(
                        span,
                        &format!(
                            "Accessing `{:?}` with the kind `{:?}` shouldn't be possible",
                            place,
                            kind
                        ),
                    );
                }
            }
            Read(ReadKind::Borrow(BorrowKind::Unique)) |
            Read(ReadKind::Borrow(BorrowKind::Mut)) |
            Read(ReadKind::Borrow(BorrowKind::Shared)) |
            Read(ReadKind::Copy) => {} // Access authorized
        }

        error_reported
    }

    /// Can this value be written or borrowed mutably
    fn is_mutable<'d>(
        &self,
        place: &'d Place<'tcx>,
        is_local_mutation_allowed: LocalMutationIsAllowed,
    ) -> Result<(), &'d Place<'tcx>> {
        match *place {
            Place::Local(local) => {
                let local = &self.mir.local_decls[local];
                match local.mutability {
                    Mutability::Not => match is_local_mutation_allowed {
                        LocalMutationIsAllowed::Yes => Ok(()),
                        LocalMutationIsAllowed::No => Err(place),
                    },
                    Mutability::Mut => Ok(()),
                }
            }
            Place::Static(ref static_) => if !self.tcx.is_static_mut(static_.def_id) {
                Err(place)
            } else {
                Ok(())
            },
            Place::Projection(ref proj) => {
                match proj.elem {
                    ProjectionElem::Deref => {
                        let base_ty = proj.base.ty(self.mir, self.tcx).to_ty(self.tcx);

                        // Check the kind of deref to decide
                        match base_ty.sty {
                            ty::TyRef(_, tnm) => {
                                match tnm.mutbl {
                                    // Shared borrowed data is never mutable
                                    hir::MutImmutable => Err(place),
                                    // Mutably borrowed data is mutable, but only if we have a
                                    // unique path to the `&mut`
                                    hir::MutMutable => {
                                        if self.is_upvar_field_projection(&proj.base).is_some() {
                                            self.is_mutable(&proj.base, is_local_mutation_allowed)
                                        } else {
                                            self.is_unique(&proj.base)
                                        }
                                    }
                                }
                            }
                            ty::TyRawPtr(tnm) => {
                                match tnm.mutbl {
                                    // `*const` raw pointers are not mutable
                                    hir::MutImmutable => Err(place),
                                    // `*mut` raw pointers are always mutable, regardless of context
                                    // The users have to check by themselve.
                                    hir::MutMutable => Ok(()),
                                }
                            }
                            // `Box<T>` owns its content, so mutable if its location is mutable
                            _ if base_ty.is_box() => {
                                self.is_mutable(&proj.base, LocalMutationIsAllowed::No)
                            }
                            // Deref should only be for reference, pointers or boxes
                            _ => bug!("Deref of unexpected type: {:?}", base_ty),
                        }
                    }
                    // All other projections are owned by their base path, so mutable if
                    // base path is mutable
                    ProjectionElem::Field(..) |
                    ProjectionElem::Index(..) |
                    ProjectionElem::ConstantIndex { .. } |
                    ProjectionElem::Subslice { .. } |
                    ProjectionElem::Downcast(..) => {
                        let field_projection = self.is_upvar_field_projection(place);

                        if let Some(field) = field_projection {
                            let decl = &self.mir.upvar_decls[field.index()];

                            return match decl.mutability {
                                Mutability::Mut => self.is_unique(&proj.base),
                                Mutability::Not => Err(place),
                            };
                        }

                        self.is_mutable(&proj.base, LocalMutationIsAllowed::No)
                    }
                }
            }
        }
    }

    /// Does this place have a unique path
    fn is_unique<'d>(&self, place: &'d Place<'tcx>) -> Result<(), &'d Place<'tcx>> {
        match *place {
            Place::Local(..) => {
                // Local variables are unique
                Ok(())
            }
            Place::Static(..) => {
                // Static variables are not
                Err(place)
            }
            Place::Projection(ref proj) => {
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
                                    // place represent an aliased location
                                    hir::MutImmutable => Err(place),
                                    // `&mut T` is as unique as the context in which it is found
                                    hir::MutMutable => self.is_unique(&proj.base),
                                }
                            }
                            ty::TyRawPtr(tnm) => {
                                match tnm.mutbl {
                                    // `*mut` can be aliased, but we leave it to user
                                    hir::MutMutable => Ok(()),
                                    // `*const` is treated the same as `*mut`
                                    hir::MutImmutable => Ok(()),
                                }
                            }
                            // Deref should only be for reference, pointers or boxes
                            _ => bug!("Deref of unexpected type: {:?}", base_ty),
                        }
                    }
                    // Other projections are unique if the base is unique
                    ProjectionElem::Field(..) |
                    ProjectionElem::Index(..) |
                    ProjectionElem::ConstantIndex { .. } |
                    ProjectionElem::Subslice { .. } |
                    ProjectionElem::Downcast(..) => self.is_unique(&proj.base),
                }
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum NoMovePathFound {
    ReachedStatic,
}

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    fn each_borrow_involving_path<F>(
        &mut self,
        _context: Context,
        access_place: (ShallowOrDeep, &Place<'tcx>),
        flow_state: &InProgress<'cx, 'gcx, 'tcx>,
        mut op: F,
    ) where
        F: FnMut(&mut Self, BorrowIndex, &BorrowData<'tcx>, &Place<'tcx>) -> Control,
    {
        let (access, place) = access_place;

        // FIXME: analogous code in check_loans first maps `place` to
        // its base_path.

        let domain = flow_state.borrows.base_results.operator();
        let data = domain.borrows();

        // check for loan restricting path P being used. Accounts for
        // borrows of P, P.a.b, etc.
        'next_borrow: for i in flow_state.borrows.elems_incoming() {
            let borrowed = &data[i];

            // Is `place` (or a prefix of it) already borrowed? If
            // so, that's relevant.
            //
            // FIXME: Differs from AST-borrowck; includes drive-by fix
            // to #38899. Will probably need back-compat mode flag.
            for accessed_prefix in self.prefixes(place, PrefixSet::All) {
                if *accessed_prefix == borrowed.place {
                    // FIXME: pass in enum describing case we are in?
                    let ctrl = op(self, i, borrowed, accessed_prefix);
                    if ctrl == Control::Break {
                        return;
                    }
                }
            }

            // Is `place` a prefix (modulo access type) of the
            // `borrowed.place`? If so, that's relevant.

            let prefix_kind = match access {
                Shallow(Some(ArtificialField::Discriminant)) |
                Shallow(Some(ArtificialField::ArrayLength)) => {
                    // The discriminant and array length are like
                    // additional fields on the type; they do not
                    // overlap any existing data there. Furthermore,
                    // they cannot actually be a prefix of any
                    // borrowed place (at least in MIR as it is
                    // currently.)
                    continue 'next_borrow;
                }
                Shallow(None) => PrefixSet::Shallow,
                Deep => PrefixSet::Supporting,
            };

            for borrowed_prefix in self.prefixes(&borrowed.place, prefix_kind) {
                if borrowed_prefix == place {
                    // FIXME: pass in enum describing case we are in?
                    let ctrl = op(self, i, borrowed, borrowed_prefix);
                    if ctrl == Control::Break {
                        return;
                    }
                }
            }
        }
    }
}

use self::prefixes::PrefixSet;

/// From the NLL RFC: "The deep [aka 'supporting'] prefixes for an
/// place are formed by stripping away fields and derefs, except that
/// we stop when we reach the deref of a shared reference. [...] "
///
/// "Shallow prefixes are found by stripping away fields, but stop at
/// any dereference. So: writing a path like `a` is illegal if `a.b`
/// is borrowed. But: writing `a` is legal if `*a` is borrowed,
/// whether or not `a` is a shared or mutable reference. [...] "
mod prefixes {
    use super::MirBorrowckCtxt;

    use rustc::hir;
    use rustc::ty::{self, TyCtxt};
    use rustc::mir::{Mir, Place, ProjectionElem};

    pub trait IsPrefixOf<'tcx> {
        fn is_prefix_of(&self, other: &Place<'tcx>) -> bool;
    }

    impl<'tcx> IsPrefixOf<'tcx> for Place<'tcx> {
        fn is_prefix_of(&self, other: &Place<'tcx>) -> bool {
            let mut cursor = other;
            loop {
                if self == cursor {
                    return true;
                }

                match *cursor {
                    Place::Local(_) | Place::Static(_) => return false,
                    Place::Projection(ref proj) => {
                        cursor = &proj.base;
                    }
                }
            }
        }
    }


    pub(super) struct Prefixes<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
        mir: &'cx Mir<'tcx>,
        tcx: TyCtxt<'cx, 'gcx, 'tcx>,
        kind: PrefixSet,
        next: Option<&'cx Place<'tcx>>,
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

    impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
        /// Returns an iterator over the prefixes of `place`
        /// (inclusive) from longest to smallest, potentially
        /// terminating the iteration early based on `kind`.
        pub(super) fn prefixes(
            &self,
            place: &'cx Place<'tcx>,
            kind: PrefixSet,
        ) -> Prefixes<'cx, 'gcx, 'tcx> {
            Prefixes {
                next: Some(place),
                kind,
                mir: self.mir,
                tcx: self.tcx,
            }
        }
    }

    impl<'cx, 'gcx, 'tcx> Iterator for Prefixes<'cx, 'gcx, 'tcx> {
        type Item = &'cx Place<'tcx>;
        fn next(&mut self) -> Option<Self::Item> {
            let mut cursor = match self.next {
                None => return None,
                Some(place) => place,
            };

            // Post-processing `place`: Enqueue any remaining
            // work. Also, `place` may not be a prefix itself, but
            // may hold one further down (e.g. we never return
            // downcasts here, but may return a base of a downcast).

            'cursor: loop {
                let proj = match *cursor {
                    Place::Local(_) | // search yielded this leaf
                    Place::Static(_) => {
                        self.next = None;
                        return Some(cursor);
                    }

                    Place::Projection(ref proj) => proj,
                };

                match proj.elem {
                    ProjectionElem::Field(_ /*field*/, _ /*ty*/) => {
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
                    ty::TyRef(
                        _, /*rgn*/
                        ty::TypeAndMut {
                            ty: _,
                            mutbl: hir::MutImmutable,
                        },
                    ) => {
                        // don't continue traversing over derefs of raw pointers or shared borrows.
                        self.next = None;
                        return Some(cursor);
                    }

                    ty::TyRef(
                        _, /*rgn*/
                        ty::TypeAndMut {
                            ty: _,
                            mutbl: hir::MutMutable,
                        },
                    ) => {
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

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    fn report_use_of_moved_or_uninitialized(
        &mut self,
        _context: Context,
        desired_action: InitializationRequiringAction,
        (place, span): (&Place<'tcx>, Span),
        mpi: MovePathIndex,
        curr_move_out: &IdxSetBuf<MoveOutIndex>,
    ) {
        let mois = self.move_data.path_map[mpi]
            .iter()
            .filter(|moi| curr_move_out.contains(moi))
            .collect::<Vec<_>>();

        if mois.is_empty() {
            let item_msg = match self.describe_place(place) {
                Some(name) => format!("`{}`", name),
                None => "value".to_owned(),
            };
            self.tcx
                .cannot_act_on_uninitialized_variable(
                    span,
                    desired_action.as_noun(),
                    &self.describe_place(place).unwrap_or("_".to_owned()),
                    Origin::Mir,
                )
                .span_label(span, format!("use of possibly uninitialized {}", item_msg))
                .emit();
        } else {
            let msg = ""; //FIXME: add "partially " or "collaterally "

            let mut err = self.tcx.cannot_act_on_moved_value(
                span,
                desired_action.as_noun(),
                msg,
                &self.describe_place(place).unwrap_or("_".to_owned()),
                Origin::Mir,
            );

            err.span_label(
                span,
                format!(
                    "value {} here after move",
                    desired_action.as_verb_in_past_tense()
                ),
            );
            for moi in mois {
                let move_msg = ""; //FIXME: add " (into closure)"
                let move_span = self.mir.source_info(self.move_data.moves[*moi].source).span;
                if span == move_span {
                    err.span_label(
                        span,
                        format!("value moved{} here in previous iteration of loop", move_msg),
                    );
                } else {
                    err.span_label(move_span, format!("value moved{} here", move_msg));
                };
            }
            //FIXME: add note for closure
            err.emit();
        }
    }

    fn report_move_out_while_borrowed(
        &mut self,
        _context: Context,
        (place, span): (&Place<'tcx>, Span),
        borrow: &BorrowData<'tcx>,
    ) {
        let value_msg = match self.describe_place(place) {
            Some(name) => format!("`{}`", name),
            None => "value".to_owned(),
        };
        let borrow_msg = match self.describe_place(&borrow.place) {
            Some(name) => format!("`{}`", name),
            None => "value".to_owned(),
        };
        self.tcx
            .cannot_move_when_borrowed(
                span,
                &self.describe_place(place).unwrap_or("_".to_owned()),
                Origin::Mir,
            )
            .span_label(
                self.retrieve_borrow_span(borrow),
                format!("borrow of {} occurs here", borrow_msg),
            )
            .span_label(span, format!("move out of {} occurs here", value_msg))
            .emit();
    }

    fn report_use_while_mutably_borrowed(
        &mut self,
        _context: Context,
        (place, span): (&Place<'tcx>, Span),
        borrow: &BorrowData<'tcx>,
    ) {
        let mut err = self.tcx.cannot_use_when_mutably_borrowed(
            span,
            &self.describe_place(place).unwrap_or("_".to_owned()),
            self.retrieve_borrow_span(borrow),
            &self.describe_place(&borrow.place).unwrap_or("_".to_owned()),
            Origin::Mir,
        );

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

        let local = if let StatementKind::Assign(Place::Local(local), _) =
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

            if let StatementKind::Assign(_, Rvalue::Aggregate(ref kind, ref places)) = stmt.kind {
                if let AggregateKind::Closure(def_id, _) = **kind {
                    debug!("find_closure_span: found closure {:?}", places);

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
                                for (v, place) in freevars.iter().zip(places) {
                                    match *place {
                                        Operand::Copy(Place::Local(l)) |
                                        Operand::Move(Place::Local(l)) if local == l =>
                                        {
                                            debug!(
                                                "find_closure_span: found captured local {:?}",
                                                l
                                            );
                                            return Some(v.span);
                                        }
                                        _ => {}
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

    fn report_conflicting_borrow(
        &mut self,
        context: Context,
        common_prefix: &Place<'tcx>,
        (place, span): (&Place<'tcx>, Span),
        gen_borrow_kind: BorrowKind,
        issued_borrow: &BorrowData,
        end_issued_loan_span: Option<Span>,
    ) {
        use self::prefixes::IsPrefixOf;

        assert!(common_prefix.is_prefix_of(place));
        assert!(common_prefix.is_prefix_of(&issued_borrow.place));

        let issued_span = self.retrieve_borrow_span(issued_borrow);

        let new_closure_span = self.find_closure_span(span, context.loc);
        let span = new_closure_span.map(|(args, _)| args).unwrap_or(span);
        let old_closure_span = self.find_closure_span(issued_span, issued_borrow.location);
        let issued_span = old_closure_span
            .map(|(args, _)| args)
            .unwrap_or(issued_span);

        let desc_place = self.describe_place(place).unwrap_or("_".to_owned());

        // FIXME: supply non-"" `opt_via` when appropriate
        let mut err = match (
            gen_borrow_kind,
            "immutable",
            "mutable",
            issued_borrow.kind,
            "immutable",
            "mutable",
        ) {
            (BorrowKind::Shared, lft, _, BorrowKind::Mut, _, rgt) |
            (BorrowKind::Mut, _, lft, BorrowKind::Shared, rgt, _) => self.tcx
                .cannot_reborrow_already_borrowed(
                    span,
                    &desc_place,
                    "",
                    lft,
                    issued_span,
                    "it",
                    rgt,
                    "",
                    end_issued_loan_span,
                    Origin::Mir,
                ),

            (BorrowKind::Mut, _, _, BorrowKind::Mut, _, _) => self.tcx
                .cannot_mutably_borrow_multiply(
                    span,
                    &desc_place,
                    "",
                    issued_span,
                    "",
                    end_issued_loan_span,
                    Origin::Mir,
                ),

            (BorrowKind::Unique, _, _, BorrowKind::Unique, _, _) => self.tcx
                .cannot_uniquely_borrow_by_two_closures(
                    span,
                    &desc_place,
                    issued_span,
                    end_issued_loan_span,
                    Origin::Mir,
                ),

            (BorrowKind::Unique, _, _, _, _, _) => self.tcx.cannot_uniquely_borrow_by_one_closure(
                span,
                &desc_place,
                "",
                issued_span,
                "it",
                "",
                end_issued_loan_span,
                Origin::Mir,
            ),

            (_, _, _, BorrowKind::Unique, _, _) => self.tcx
                .cannot_reborrow_already_uniquely_borrowed(
                    span,
                    &desc_place,
                    "it",
                    "",
                    issued_span,
                    "",
                    end_issued_loan_span,
                    Origin::Mir,
                ),

            (BorrowKind::Shared, _, _, BorrowKind::Shared, _, _) => unreachable!(),
        };

        if let Some((_, var_span)) = old_closure_span {
            err.span_label(
                var_span,
                format!(
                    "previous borrow occurs due to use of `{}` in closure",
                    desc_place
                ),
            );
        }

        if let Some((_, var_span)) = new_closure_span {
            err.span_label(
                var_span,
                format!("borrow occurs due to use of `{}` in closure", desc_place),
            );
        }

        err.emit();
    }

    fn report_borrowed_value_does_not_live_long_enough(
        &mut self,
        _: Context,
        (place, span): (&Place<'tcx>, Span),
        end_span: Option<Span>,
    ) {
        let root_place = self.prefixes(place, PrefixSet::All).last().unwrap();
        let proper_span = match *root_place {
            Place::Local(local) => self.mir.local_decls[local].source_info.span,
            _ => span,
        };
        let mut err = self.tcx
            .path_does_not_live_long_enough(span, "borrowed value", Origin::Mir);
        err.span_label(proper_span, "temporary value created here");
        err.span_label(span, "temporary value dropped here while still borrowed");
        err.note("consider using a `let` binding to increase its lifetime");

        if let Some(end) = end_span {
            err.span_label(end, "temporary value needs to live until here");
        }

        err.emit();
    }

    fn report_illegal_mutation_of_borrowed(
        &mut self,
        _: Context,
        (place, span): (&Place<'tcx>, Span),
        loan: &BorrowData,
    ) {
        let mut err = self.tcx.cannot_assign_to_borrowed(
            span,
            self.retrieve_borrow_span(loan),
            &self.describe_place(place).unwrap_or("_".to_owned()),
            Origin::Mir,
        );

        err.emit();
    }

    fn report_illegal_reassignment(
        &mut self,
        _context: Context,
        (place, span): (&Place<'tcx>, Span),
        assigned_span: Span,
    ) {
        let mut err = self.tcx.cannot_reassign_immutable(
            span,
            &self.describe_place(place).unwrap_or("_".to_owned()),
            Origin::Mir,
        );
        err.span_label(span, "cannot assign twice to immutable variable");
        if span != assigned_span {
            let value_msg = match self.describe_place(place) {
                Some(name) => format!("`{}`", name),
                None => "value".to_owned(),
            };
            err.span_label(assigned_span, format!("first assignment to {}", value_msg));
        }
        err.emit();
    }
}

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    // End-user visible description of `place` if one can be found. If the
    // place is a temporary for instance, None will be returned.
    fn describe_place(&self, place: &Place<'tcx>) -> Option<String> {
        let mut buf = String::new();
        match self.append_place_to_string(place, &mut buf, false) {
            Ok(()) => Some(buf),
            Err(()) => None,
        }
    }

    /// If this is a field projection, and the field is being projected from a closure type,
    /// then returns the index of the field being projected. Note that this closure will always
    /// be `self` in the current MIR, because that is the only time we directly access the fields
    /// of a closure type.
    fn is_upvar_field_projection(&self, place: &Place<'tcx>) -> Option<Field> {
        match *place {
            Place::Projection(ref proj) => match proj.elem {
                ProjectionElem::Field(field, _ty) => {
                    let is_projection_from_ty_closure = proj.base
                        .ty(self.mir, self.tcx)
                        .to_ty(self.tcx)
                        .is_closure();

                    if is_projection_from_ty_closure {
                        Some(field)
                    } else {
                        None
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }

    // Appends end-user visible description of `place` to `buf`.
    fn append_place_to_string(
        &self,
        place: &Place<'tcx>,
        buf: &mut String,
        mut autoderef: bool,
    ) -> Result<(), ()> {
        match *place {
            Place::Local(local) => {
                self.append_local_to_string(local, buf)?;
            }
            Place::Static(ref static_) => {
                buf.push_str(&format!("{}", &self.tcx.item_name(static_.def_id)));
            }
            Place::Projection(ref proj) => {
                match proj.elem {
                    ProjectionElem::Deref => {
                        if let Some(field) = self.is_upvar_field_projection(&proj.base) {
                            let var_index = field.index();
                            let name = self.mir.upvar_decls[var_index].debug_name.to_string();
                            if self.mir.upvar_decls[var_index].by_ref {
                                buf.push_str(&name);
                            } else {
                                buf.push_str(&format!("*{}", &name));
                            }
                        } else {
                            if autoderef {
                                self.append_place_to_string(&proj.base, buf, autoderef)?;
                            } else {
                                buf.push_str(&"*");
                                self.append_place_to_string(&proj.base, buf, autoderef)?;
                            }
                        }
                    }
                    ProjectionElem::Downcast(..) => {
                        self.append_place_to_string(&proj.base, buf, autoderef)?;
                    }
                    ProjectionElem::Field(field, _ty) => {
                        autoderef = true;

                        if let Some(field) = self.is_upvar_field_projection(place) {
                            let var_index = field.index();
                            let name = self.mir.upvar_decls[var_index].debug_name.to_string();
                            buf.push_str(&name);
                        } else {
                            let field_name = self.describe_field(&proj.base, field);
                            self.append_place_to_string(&proj.base, buf, autoderef)?;
                            buf.push_str(&format!(".{}", field_name));
                        }
                    }
                    ProjectionElem::Index(index) => {
                        autoderef = true;

                        self.append_place_to_string(&proj.base, buf, autoderef)?;
                        buf.push_str("[");
                        if let Err(_) = self.append_local_to_string(index, buf) {
                            buf.push_str("..");
                        }
                        buf.push_str("]");
                    }
                    ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. } => {
                        autoderef = true;
                        // Since it isn't possible to borrow an element on a particular index and
                        // then use another while the borrow is held, don't output indices details
                        // to avoid confusing the end-user
                        self.append_place_to_string(&proj.base, buf, autoderef)?;
                        buf.push_str(&"[..]");
                    }
                };
            }
        }

        Ok(())
    }

    // Appends end-user visible description of the `local` place to `buf`. If `local` doesn't have
    // a name, then `Err` is returned
    fn append_local_to_string(&self, local_index: Local, buf: &mut String) -> Result<(), ()> {
        let local = &self.mir.local_decls[local_index];
        match local.name {
            Some(name) => {
                buf.push_str(&format!("{}", name));
                Ok(())
            }
            None => Err(()),
        }
    }

    // End-user visible description of the `field`nth field of `base`
    fn describe_field(&self, base: &Place, field: Field) -> String {
        match *base {
            Place::Local(local) => {
                let local = &self.mir.local_decls[local];
                self.describe_field_from_ty(&local.ty, field)
            }
            Place::Static(ref static_) => self.describe_field_from_ty(&static_.ty, field),
            Place::Projection(ref proj) => match proj.elem {
                ProjectionElem::Deref => self.describe_field(&proj.base, field),
                ProjectionElem::Downcast(def, variant_index) => {
                    format!("{}", def.variants[variant_index].fields[field.index()].name)
                }
                ProjectionElem::Field(_, field_type) => {
                    self.describe_field_from_ty(&field_type, field)
                }
                ProjectionElem::Index(..) |
                ProjectionElem::ConstantIndex { .. } |
                ProjectionElem::Subslice { .. } => {
                    format!("{}", self.describe_field(&proj.base, field))
                }
            },
        }
    }

    // End-user visible description of the `field_index`nth field of `ty`
    fn describe_field_from_ty(&self, ty: &ty::Ty, field: Field) -> String {
        if ty.is_box() {
            // If the type is a box, the field is described from the boxed type
            self.describe_field_from_ty(&ty.boxed_ty(), field)
        } else {
            match ty.sty {
                ty::TyAdt(def, _) => if def.is_enum() {
                    format!("{}", field.index())
                } else {
                    format!("{}", def.struct_variant().fields[field.index()].name)
                },
                ty::TyTuple(_, _) => format!("{}", field.index()),
                ty::TyRef(_, tnm) | ty::TyRawPtr(tnm) => {
                    self.describe_field_from_ty(&tnm.ty, field)
                }
                ty::TyArray(ty, _) | ty::TySlice(ty) => self.describe_field_from_ty(&ty, field),
                ty::TyClosure(closure_def_id, _) => {
                    // Convert the def-id into a node-id. node-ids are only valid for
                    // the local code in the current crate, so this returns an `Option` in case
                    // the closure comes from another crate. But in that case we wouldn't
                    // be borrowck'ing it, so we can just unwrap:
                    let node_id = self.tcx.hir.as_local_node_id(closure_def_id).unwrap();
                    let freevar = self.tcx.with_freevars(node_id, |fv| fv[field.index()]);

                    self.tcx.hir.name(freevar.var_id()).to_string()
                }
                _ => {
                    // Might need a revision when the fields in trait RFC is implemented
                    // (https://github.com/rust-lang/rfcs/pull/1546)
                    bug!(
                        "End-user description not implemented for field access on `{:?}`",
                        ty.sty
                    );
                }
            }
        }
    }

    // Retrieve span of given borrow from the current MIR representation
    fn retrieve_borrow_span(&self, borrow: &BorrowData) -> Span {
        self.mir.source_info(borrow.location).span
    }
}

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    // FIXME (#16118): function intended to allow the borrow checker
    // to be less precise in its handling of Box while still allowing
    // moves out of a Box. They should be removed when/if we stop
    // treating Box specially (e.g. when/if DerefMove is added...)

    fn base_path<'d>(&self, place: &'d Place<'tcx>) -> &'d Place<'tcx> {
        //! Returns the base of the leftmost (deepest) dereference of an
        //! Box in `place`. If there is no dereference of an Box
        //! in `place`, then it just returns `place` itself.

        let mut cursor = place;
        let mut deepest = place;
        loop {
            let proj = match *cursor {
                Place::Local(..) | Place::Static(..) => return deepest,
                Place::Projection(ref proj) => proj,
            };
            if proj.elem == ProjectionElem::Deref
                && place.ty(self.mir, self.tcx).to_ty(self.tcx).is_box()
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
    fn new(self, loc: Location) -> Context {
        Context {
            kind: self,
            loc: loc,
        }
    }
}

impl<'b, 'gcx, 'tcx> InProgress<'b, 'gcx, 'tcx> {
    fn new(
        borrows: FlowInProgress<Borrows<'b, 'gcx, 'tcx>>,
        inits: FlowInProgress<MaybeInitializedLvals<'b, 'gcx, 'tcx>>,
        uninits: FlowInProgress<MaybeUninitializedLvals<'b, 'gcx, 'tcx>>,
        move_outs: FlowInProgress<MovingOutStatements<'b, 'gcx, 'tcx>>,
        ever_inits: FlowInProgress<EverInitializedLvals<'b, 'gcx, 'tcx>>,
    ) -> Self {
        InProgress {
            borrows,
            inits,
            uninits,
            move_outs,
            ever_inits,
        }
    }

    fn each_flow<XB, XI, XU, XM, XE>(
        &mut self,
        mut xform_borrows: XB,
        mut xform_inits: XI,
        mut xform_uninits: XU,
        mut xform_move_outs: XM,
        mut xform_ever_inits: XE,
    ) where
        XB: FnMut(&mut FlowInProgress<Borrows<'b, 'gcx, 'tcx>>),
        XI: FnMut(&mut FlowInProgress<MaybeInitializedLvals<'b, 'gcx, 'tcx>>),
        XU: FnMut(&mut FlowInProgress<MaybeUninitializedLvals<'b, 'gcx, 'tcx>>),
        XM: FnMut(&mut FlowInProgress<MovingOutStatements<'b, 'gcx, 'tcx>>),
        XE: FnMut(&mut FlowInProgress<EverInitializedLvals<'b, 'gcx, 'tcx>>),
    {
        xform_borrows(&mut self.borrows);
        xform_inits(&mut self.inits);
        xform_uninits(&mut self.uninits);
        xform_move_outs(&mut self.move_outs);
        xform_ever_inits(&mut self.ever_inits);
    }

    fn summary(&self) -> String {
        let mut s = String::new();

        s.push_str("borrows in effect: [");
        let mut saw_one = false;
        self.borrows.each_state_bit(|borrow| {
            if saw_one {
                s.push_str(", ");
            };
            saw_one = true;
            let borrow_data = &self.borrows.base_results.operator().borrows()[borrow];
            s.push_str(&format!("{}", borrow_data));
        });
        s.push_str("] ");

        s.push_str("borrows generated: [");
        let mut saw_one = false;
        self.borrows.each_gen_bit(|borrow| {
            if saw_one {
                s.push_str(", ");
            };
            saw_one = true;
            let borrow_data = &self.borrows.base_results.operator().borrows()[borrow];
            s.push_str(&format!("{}", borrow_data));
        });
        s.push_str("] ");

        s.push_str("inits: [");
        let mut saw_one = false;
        self.inits.each_state_bit(|mpi_init| {
            if saw_one {
                s.push_str(", ");
            };
            saw_one = true;
            let move_path = &self.inits.base_results.operator().move_data().move_paths[mpi_init];
            s.push_str(&format!("{}", move_path));
        });
        s.push_str("] ");

        s.push_str("uninits: [");
        let mut saw_one = false;
        self.uninits.each_state_bit(|mpi_uninit| {
            if saw_one {
                s.push_str(", ");
            };
            saw_one = true;
            let move_path =
                &self.uninits.base_results.operator().move_data().move_paths[mpi_uninit];
            s.push_str(&format!("{}", move_path));
        });
        s.push_str("] ");

        s.push_str("move_out: [");
        let mut saw_one = false;
        self.move_outs.each_state_bit(|mpi_move_out| {
            if saw_one {
                s.push_str(", ");
            };
            saw_one = true;
            let move_out = &self.move_outs.base_results.operator().move_data().moves[mpi_move_out];
            s.push_str(&format!("{:?}", move_out));
        });
        s.push_str("] ");

        s.push_str("ever_init: [");
        let mut saw_one = false;
        self.ever_inits.each_state_bit(|mpi_ever_init| {
            if saw_one {
                s.push_str(", ");
            };
            saw_one = true;
            let ever_init =
                &self.ever_inits.base_results.operator().move_data().inits[mpi_ever_init];
            s.push_str(&format!("{:?}", ever_init));
        });
        s.push_str("]");

        return s;
    }
}

impl<'tcx, T> FlowInProgress<T>
where
    T: HasMoveData<'tcx> + BitDenotation<Idx = MovePathIndex>,
{
    fn has_any_child_of(&self, mpi: T::Idx) -> Option<T::Idx> {
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

impl<BD> FlowInProgress<BD>
where
    BD: BitDenotation,
{
    fn each_state_bit<F>(&self, f: F)
    where
        F: FnMut(BD::Idx),
    {
        self.curr_state
            .each_bit(self.base_results.operator().bits_per_block(), f)
    }

    fn each_gen_bit<F>(&self, f: F)
    where
        F: FnMut(BD::Idx),
    {
        self.stmt_gen
            .each_bit(self.base_results.operator().bits_per_block(), f)
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
            on_entry: &mut ignored,
            gen_set: &mut self.stmt_gen,
            kill_set: &mut self.stmt_kill,
        };
        self.base_results
            .operator()
            .statement_effect(&mut sets, loc);
    }

    fn reconstruct_terminator_effect(&mut self, loc: Location) {
        self.stmt_gen.reset_to_empty();
        self.stmt_kill.reset_to_empty();
        let mut ignored = IdxSetBuf::new_empty(0);
        let mut sets = BlockSets {
            on_entry: &mut ignored,
            gen_set: &mut self.stmt_gen,
            kill_set: &mut self.stmt_kill,
        };
        self.base_results
            .operator()
            .terminator_effect(&mut sets, loc);
    }

    fn apply_local_effect(&mut self) {
        self.curr_state.union(&self.stmt_gen);
        self.curr_state.subtract(&self.stmt_kill);
    }

    fn elems_incoming(&self) -> indexed_set::Elems<BD::Idx> {
        let univ = self.base_results.sets().bits_per_block();
        self.curr_state.elems(univ)
    }

    fn with_elems_outgoing<F>(&self, f: F)
    where
        F: FnOnce(indexed_set::Elems<BD::Idx>),
    {
        let mut curr_state = self.curr_state.clone();
        curr_state.union(&self.stmt_gen);
        curr_state.subtract(&self.stmt_kill);
        let univ = self.base_results.sets().bits_per_block();
        f(curr_state.elems(univ));
    }
}
