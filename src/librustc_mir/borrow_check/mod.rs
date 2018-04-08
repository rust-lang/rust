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

use borrow_check::nll::region_infer::{RegionCausalInfo, RegionInferenceContext};
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::hir::map::definitions::DefPathData;
use rustc::infer::InferCtxt;
use rustc::ty::{self, ParamEnv, TyCtxt};
use rustc::ty::maps::Providers;
use rustc::mir::{AssertMessage, BasicBlock, BorrowKind, Location, Place};
use rustc::mir::{Mir, Mutability, Operand, Projection, ProjectionElem, Rvalue};
use rustc::mir::{Field, Statement, StatementKind, Terminator, TerminatorKind};
use rustc::mir::ClosureRegionRequirements;

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::indexed_set::IdxSetBuf;
use rustc_data_structures::indexed_vec::Idx;

use std::rc::Rc;

use syntax::ast;
use syntax_pos::Span;

use dataflow::{do_dataflow, DebugFormatted};
use dataflow::FlowAtLocation;
use dataflow::MoveDataParamEnv;
use dataflow::{DataflowResultsConsumer};
use dataflow::{MaybeInitializedPlaces, MaybeUninitializedPlaces};
use dataflow::{EverInitializedPlaces, MovingOutStatements};
use dataflow::{BorrowData, Borrows, ReserveOrActivateIndex};
use dataflow::indexes::BorrowIndex;
use dataflow::move_paths::{IllegalMoveOriginKind, MoveError};
use dataflow::move_paths::{HasMoveData, LookupResult, MoveData, MovePathIndex};
use util::borrowck_errors::{BorrowckErrors, Origin};
use util::collect_writes::FindAssignments;

use std::iter;

use self::flows::Flows;
use self::prefixes::PrefixSet;
use self::MutateMode::{JustWrite, WriteAndRead};

mod error_reporting;
mod flows;
mod prefixes;

pub(crate) mod nll;

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        mir_borrowck,
        ..*providers
    };
}

fn mir_borrowck<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    def_id: DefId,
) -> Option<ClosureRegionRequirements<'tcx>> {
    let input_mir = tcx.mir_validated(def_id);
    debug!("run query mir_borrowck: {}", tcx.item_path_str(def_id));

    if !tcx.has_attr(def_id, "rustc_mir_borrowck") && !tcx.use_mir() {
        return None;
    }

    let opt_closure_req = tcx.infer_ctxt().enter(|infcx| {
        let input_mir: &Mir = &input_mir.borrow();
        do_mir_borrowck(&infcx, input_mir, def_id)
    });
    debug!("mir_borrowck done");

    opt_closure_req
}

fn do_mir_borrowck<'a, 'gcx, 'tcx>(
    infcx: &InferCtxt<'a, 'gcx, 'tcx>,
    input_mir: &Mir<'gcx>,
    def_id: DefId,
) -> Option<ClosureRegionRequirements<'gcx>> {
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
    let free_regions = if !tcx.nll() {
        None
    } else {
        let mir = &mut mir;

        // Replace all regions with fresh inference variables.
        Some(nll::replace_regions_in_mir(infcx, def_id, param_env, mir))
    };
    let mir = &mir;

    let move_data: MoveData<'tcx> = match MoveData::gather_moves(mir, tcx) {
        Ok(move_data) => move_data,
        Err((move_data, move_errors)) => {
            for move_error in move_errors {
                let (span, kind): (Span, IllegalMoveOriginKind) = match move_error {
                    MoveError::UnionMove { .. } => {
                        unimplemented!("don't know how to report union move errors yet.")
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
    let body_id = match tcx.def_key(def_id).disambiguated_data.data {
        DefPathData::StructCtor | DefPathData::EnumVariant(_) => None,
        _ => Some(tcx.hir.body_owned_by(id)),
    };

    let dead_unwinds = IdxSetBuf::new_empty(mir.basic_blocks().len());
    let mut flow_inits = FlowAtLocation::new(do_dataflow(
        tcx,
        mir,
        id,
        &attributes,
        &dead_unwinds,
        MaybeInitializedPlaces::new(tcx, mir, &mdpe),
        |bd, i| DebugFormatted::new(&bd.move_data().move_paths[i]),
    ));
    let flow_uninits = FlowAtLocation::new(do_dataflow(
        tcx,
        mir,
        id,
        &attributes,
        &dead_unwinds,
        MaybeUninitializedPlaces::new(tcx, mir, &mdpe),
        |bd, i| DebugFormatted::new(&bd.move_data().move_paths[i]),
    ));
    let flow_move_outs = FlowAtLocation::new(do_dataflow(
        tcx,
        mir,
        id,
        &attributes,
        &dead_unwinds,
        MovingOutStatements::new(tcx, mir, &mdpe),
        |bd, i| DebugFormatted::new(&bd.move_data().moves[i]),
    ));
    let flow_ever_inits = FlowAtLocation::new(do_dataflow(
        tcx,
        mir,
        id,
        &attributes,
        &dead_unwinds,
        EverInitializedPlaces::new(tcx, mir, &mdpe),
        |bd, i| DebugFormatted::new(&bd.move_data().inits[i]),
    ));

    // If we are in non-lexical mode, compute the non-lexical lifetimes.
    let (opt_regioncx, opt_closure_req) = if let Some(free_regions) = free_regions {
        let (regioncx, opt_closure_req) = nll::compute_regions(
            infcx,
            def_id,
            free_regions,
            mir,
            param_env,
            &mut flow_inits,
            &mdpe.move_data,
        );
        (Some(Rc::new(regioncx)), opt_closure_req)
    } else {
        assert!(!tcx.nll());
        (None, None)
    };
    let flow_inits = flow_inits; // remove mut

    let flow_borrows = FlowAtLocation::new(do_dataflow(
        tcx,
        mir,
        id,
        &attributes,
        &dead_unwinds,
        Borrows::new(tcx, mir, opt_regioncx.clone(), def_id, body_id),
        |rs, i| {
            DebugFormatted::new(&(i.kind(), rs.location(i.borrow_index())))
        }
    ));

    let movable_generator = !match tcx.hir.get(id) {
        hir::map::Node::NodeExpr(&hir::Expr {
            node: hir::ExprClosure(.., Some(hir::GeneratorMovability::Static)),
            ..
        }) => true,
        _ => false,
    };

    let mut mbcx = MirBorrowckCtxt {
        tcx: tcx,
        mir: mir,
        node_id: id,
        move_data: &mdpe.move_data,
        param_env: param_env,
        movable_generator,
        locals_are_invalidated_at_exit: match tcx.hir.body_owner_kind(id) {
            hir::BodyOwnerKind::Const | hir::BodyOwnerKind::Static(_) => false,
            hir::BodyOwnerKind::Fn => true,
        },
        access_place_error_reported: FxHashSet(),
        reservation_error_reported: FxHashSet(),
        nonlexical_regioncx: opt_regioncx,
        nonlexical_cause_info: None,
    };

    let mut state = Flows::new(
        flow_borrows,
        flow_inits,
        flow_uninits,
        flow_move_outs,
        flow_ever_inits,
    );

    mbcx.analyze_results(&mut state); // entry point for DataflowResultsConsumer

    opt_closure_req
}

#[allow(dead_code)]
pub struct MirBorrowckCtxt<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    mir: &'cx Mir<'tcx>,
    node_id: ast::NodeId,
    move_data: &'cx MoveData<'tcx>,
    param_env: ParamEnv<'gcx>,
    movable_generator: bool,
    /// This keeps track of whether local variables are free-ed when the function
    /// exits even without a `StorageDead`, which appears to be the case for
    /// constants.
    ///
    /// I'm not sure this is the right approach - @eddyb could you try and
    /// figure this out?
    locals_are_invalidated_at_exit: bool,
    /// This field keeps track of when borrow errors are reported in the access_place function
    /// so that there is no duplicate reporting. This field cannot also be used for the conflicting
    /// borrow errors that is handled by the `reservation_error_reported` field as the inclusion
    /// of the `Span` type (while required to mute some errors) stops the muting of the reservation
    /// errors.
    access_place_error_reported: FxHashSet<(Place<'tcx>, Span)>,
    /// This field keeps track of when borrow conflict errors are reported
    /// for reservations, so that we don't report seemingly duplicate
    /// errors for corresponding activations
    ///
    /// FIXME: Ideally this would be a set of BorrowIndex, not Places,
    /// but it is currently inconvenient to track down the BorrowIndex
    /// at the time we detect and report a reservation error.
    reservation_error_reported: FxHashSet<Place<'tcx>>,
    /// Non-lexical region inference context, if NLL is enabled.  This
    /// contains the results from region inference and lets us e.g.
    /// find out which CFG points are contained in each borrow region.
    nonlexical_regioncx: Option<Rc<RegionInferenceContext<'tcx>>>,
    nonlexical_cause_info: Option<RegionCausalInfo>,
}

// Check that:
// 1. assignments are always made to mutable locations (FIXME: does that still really go here?)
// 2. loans made in overlapping scopes do not conflict
// 3. assignments do not affect things loaned out as immutable
// 4. moves do not affect things loaned out in any way
impl<'cx, 'gcx, 'tcx> DataflowResultsConsumer<'cx, 'tcx> for MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    type FlowState = Flows<'cx, 'gcx, 'tcx>;

    fn mir(&self) -> &'cx Mir<'tcx> {
        self.mir
    }

    fn visit_block_entry(&mut self, bb: BasicBlock, flow_state: &Self::FlowState) {
        debug!("MirBorrowckCtxt::process_block({:?}): {}", bb, flow_state);
    }

    fn visit_statement_entry(
        &mut self,
        location: Location,
        stmt: &Statement<'tcx>,
        flow_state: &Self::FlowState,
    ) {
        debug!(
            "MirBorrowckCtxt::process_statement({:?}, {:?}): {}",
            location, stmt, flow_state
        );
        let span = stmt.source_info.span;

        self.check_activations(location, span, flow_state);

        match stmt.kind {
            StatementKind::Assign(ref lhs, ref rhs) => {
                self.consume_rvalue(
                    ContextKind::AssignRhs.new(location),
                    (rhs, span),
                    location,
                    flow_state,
                );

                self.mutate_place(
                    ContextKind::AssignLhs.new(location),
                    (lhs, span),
                    Shallow(None),
                    JustWrite,
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
                            if o.is_rw { Deep } else { Shallow(None) },
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
            StatementKind::Nop |
            StatementKind::UserAssertTy(..) |
            StatementKind::Validate(..) |
            StatementKind::StorageLive(..) => {
                // `Nop`, `UserAssertTy`, `Validate`, and `StorageLive` are irrelevant
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
        debug!(
            "MirBorrowckCtxt::process_terminator({:?}, {:?}): {}",
            location, term, flow_state
        );
        let span = term.source_info.span;

        self.check_activations(location, span, flow_state);

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
                let gcx = self.tcx.global_tcx();

                // Compute the type with accurate region information.
                let drop_place_ty = drop_place.ty(self.mir, self.tcx);

                // Erase the regions.
                let drop_place_ty = self.tcx.erase_regions(&drop_place_ty).to_ty(self.tcx);

                // "Lift" into the gcx -- once regions are erased, this type should be in the
                // global arenas; this "lift" operation basically just asserts that is true, but
                // that is useful later.
                let drop_place_ty = gcx.lift(&drop_place_ty).unwrap();

                self.visit_terminator_drop(loc, term, flow_state, drop_place, drop_place_ty, span);
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

                if self.movable_generator {
                    // Look for any active borrows to locals
                    let domain = flow_state.borrows.operator();
                    let data = domain.borrows();
                    flow_state.borrows.with_iter_outgoing(|borrows| {
                        for i in borrows {
                            let borrow = &data[i.borrow_index()];
                            self.check_for_local_borrow(borrow, span);
                        }
                    });
                }
            }

            TerminatorKind::Resume | TerminatorKind::Return | TerminatorKind::GeneratorDrop => {
                // Returning from the function implicitly kills storage for all locals and statics.
                // Often, the storage will already have been killed by an explicit
                // StorageDead, but we don't always emit those (notably on unwind paths),
                // so this "extra check" serves as a kind of backup.
                let domain = flow_state.borrows.operator();
                let data = domain.borrows();
                flow_state.borrows.with_iter_outgoing(|borrows| {
                    for i in borrows {
                        let borrow = &data[i.borrow_index()];
                        let context = ContextKind::StorageDead.new(loc);
                        self.check_for_invalidation_at_exit(context, borrow, span, flow_state);
                    }
                });
            }
            TerminatorKind::Goto { target: _ }
            | TerminatorKind::Abort
            | TerminatorKind::Unreachable
            | TerminatorKind::FalseEdges {
                real_target: _,
                imaginary_targets: _,
            }
            | TerminatorKind::FalseUnwind {
                real_target: _,
                unwind: _,
            } => {
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
use self::ReadOrWrite::{Activation, Read, Reservation, Write};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum ArtificialField {
    Discriminant,
    ArrayLength,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum ShallowOrDeep {
    /// From the RFC: "A *shallow* access means that the immediate
    /// fields reached at P are accessed, but references or pointers
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

    /// For two-phase borrows, we distinguish a reservation (which is treated
    /// like a Read) from an activation (which is treated like a write), and
    /// each of those is furthermore distinguished from Reads/Writes above.
    Reservation(WriteKind),
    Activation(WriteKind, BorrowIndex),
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
    /// We want use of immutable upvars to cause a "write to immutable upvar"
    /// error, not an "reassignment" error.
    ExceptUpvars,
    No,
}

struct AccessErrorsReported {
    mutability_error: bool,
    #[allow(dead_code)]
    conflict_error: bool,
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
    /// Returns true if the borrow represented by `kind` is
    /// allowed to be split into separate Reservation and
    /// Activation phases.
    fn allow_two_phase_borrow(&self, kind: BorrowKind) -> bool {
        self.tcx.two_phase_borrows()
            && (kind.allows_two_phase_borrow()
                || self.tcx.sess.opts.debugging_opts.two_phase_beyond_autoref)
    }

    /// Invokes `access_place` as appropriate for dropping the value
    /// at `drop_place`. Note that the *actual* `Drop` in the MIR is
    /// always for a variable (e.g., `Drop(x)`) -- but we recursively
    /// break this variable down into subpaths (e.g., `Drop(x.foo)`)
    /// to indicate more precisely which fields might actually be
    /// accessed by a destructor.
    fn visit_terminator_drop(
        &mut self,
        loc: Location,
        term: &Terminator<'tcx>,
        flow_state: &Flows<'cx, 'gcx, 'tcx>,
        drop_place: &Place<'tcx>,
        erased_drop_place_ty: ty::Ty<'gcx>,
        span: Span,
    ) {
        match erased_drop_place_ty.sty {
            // When a struct is being dropped, we need to check
            // whether it has a destructor, if it does, then we can
            // call it, if it does not then we need to check the
            // individual fields instead. This way if `foo` has a
            // destructor but `bar` does not, we will only check for
            // borrows of `x.foo` and not `x.bar`. See #47703.
            ty::TyAdt(def, substs) if def.is_struct() && !def.has_dtor(self.tcx) => {
                for (index, field) in def.all_fields().enumerate() {
                    let gcx = self.tcx.global_tcx();
                    let field_ty = field.ty(gcx, substs);
                    let field_ty = gcx.normalize_erasing_regions(self.param_env, field_ty);
                    let place = drop_place.clone().field(Field::new(index), field_ty);

                    self.visit_terminator_drop(loc, term, flow_state, &place, field_ty, span);
                }
            }
            _ => {
                // We have now refined the type of the value being
                // dropped (potentially) to just the type of a
                // subfield; so check whether that field's type still
                // "needs drop". If so, we assume that the destructor
                // may access any data it likes (i.e., a Deep Write).
                let gcx = self.tcx.global_tcx();
                if erased_drop_place_ty.needs_drop(gcx, self.param_env) {
                    self.access_place(
                        ContextKind::Drop.new(loc),
                        (drop_place, span),
                        (Deep, Write(WriteKind::StorageDeadOrDrop)),
                        LocalMutationIsAllowed::Yes,
                        flow_state,
                    );
                }
            }
        }
    }

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
        flow_state: &Flows<'cx, 'gcx, 'tcx>,
    ) -> AccessErrorsReported {
        let (sd, rw) = kind;

        if let Activation(_, borrow_index) = rw {
            if self.reservation_error_reported.contains(&place_span.0) {
                debug!(
                    "skipping access_place for activation of invalid reservation \
                     place: {:?} borrow_index: {:?}",
                    place_span.0, borrow_index
                );
                return AccessErrorsReported {
                    mutability_error: false,
                    conflict_error: true,
                };
            }
        }

        if self.access_place_error_reported
            .contains(&(place_span.0.clone(), place_span.1))
        {
            debug!(
                "access_place: suppressing error place_span=`{:?}` kind=`{:?}`",
                place_span, kind
            );
            return AccessErrorsReported {
                mutability_error: false,
                conflict_error: true,
            };
        }

        let mutability_error =
            self.check_access_permissions(place_span, rw, is_local_mutation_allowed);
        let conflict_error =
            self.check_access_for_conflict(context, place_span, sd, rw, flow_state);

        if conflict_error || mutability_error {
            debug!(
                "access_place: logging error place_span=`{:?}` kind=`{:?}`",
                place_span, kind
            );
            self.access_place_error_reported
                .insert((place_span.0.clone(), place_span.1));
        }

        AccessErrorsReported {
            mutability_error,
            conflict_error,
        }
    }

    fn check_access_for_conflict(
        &mut self,
        context: Context,
        place_span: (&Place<'tcx>, Span),
        sd: ShallowOrDeep,
        rw: ReadOrWrite,
        flow_state: &Flows<'cx, 'gcx, 'tcx>,
    ) -> bool {
        let mut error_reported = false;
        self.each_borrow_involving_path(
            context,
            (sd, place_span.0),
            flow_state,
            |this, index, borrow| match (rw, borrow.kind) {
                // Obviously an activation is compatible with its own
                // reservation (or even prior activating uses of same
                // borrow); so don't check if they interfere.
                //
                // NOTE: *reservations* do conflict with themselves;
                // thus aren't injecting unsoundenss w/ this check.)
                (Activation(_, activating), _) if activating == index.borrow_index() => {
                    debug!(
                        "check_access_for_conflict place_span: {:?} sd: {:?} rw: {:?} \
                         skipping {:?} b/c activation of same borrow_index: {:?}",
                        place_span,
                        sd,
                        rw,
                        (index, borrow),
                        index.borrow_index()
                    );
                    Control::Continue
                }

                (Read(_), BorrowKind::Shared) | (Reservation(..), BorrowKind::Shared) => {
                    Control::Continue
                }

                (Read(kind), BorrowKind::Unique) | (Read(kind), BorrowKind::Mut { .. }) => {
                    // Reading from mere reservations of mutable-borrows is OK.
                    if this.allow_two_phase_borrow(borrow.kind) && index.is_reservation() {
                        return Control::Continue;
                    }

                    match kind {
                        ReadKind::Copy => {
                            error_reported = true;
                            this.report_use_while_mutably_borrowed(context, place_span, borrow)
                        }
                        ReadKind::Borrow(bk) => {
                            let end_issued_loan_span = flow_state
                                .borrows
                                .operator()
                                .opt_region_end_span(&borrow.region);
                            error_reported = true;
                            this.report_conflicting_borrow(
                                context,
                                place_span,
                                bk,
                                &borrow,
                                end_issued_loan_span,
                            )
                        }
                    }
                    Control::Break
                }

                (Reservation(kind), BorrowKind::Unique)
                | (Reservation(kind), BorrowKind::Mut { .. })
                | (Activation(kind, _), _)
                | (Write(kind), _) => {
                    match rw {
                        Reservation(_) => {
                            debug!(
                                "recording invalid reservation of \
                                 place: {:?}",
                                place_span.0
                            );
                            this.reservation_error_reported.insert(place_span.0.clone());
                        }
                        Activation(_, activating) => {
                            debug!(
                                "observing check_place for activation of \
                                 borrow_index: {:?}",
                                activating
                            );
                        }
                        Read(..) | Write(..) => {}
                    }

                    match kind {
                        WriteKind::MutableBorrow(bk) => {
                            let end_issued_loan_span = flow_state
                                .borrows
                                .operator()
                                .opt_region_end_span(&borrow.region);

                            error_reported = true;
                            this.report_conflicting_borrow(
                                context,
                                place_span,
                                bk,
                                &borrow,
                                end_issued_loan_span,
                            )
                        }
                        WriteKind::StorageDeadOrDrop => {
                            error_reported = true;
                            this.report_borrowed_value_does_not_live_long_enough(
                                context,
                                borrow,
                                place_span.1,
                                flow_state.borrows.operator(),
                            );
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

        error_reported
    }

    fn mutate_place(
        &mut self,
        context: Context,
        place_span: (&Place<'tcx>, Span),
        kind: ShallowOrDeep,
        mode: MutateMode,
        flow_state: &Flows<'cx, 'gcx, 'tcx>,
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

        let errors_reported = self.access_place(
            context,
            place_span,
            (kind, Write(WriteKind::Mutate)),
            // We want immutable upvars to cause an "assignment to immutable var"
            // error, not an "reassignment of immutable var" error, because the
            // latter can't find a good previous assignment span.
            //
            // There's probably a better way to do this.
            LocalMutationIsAllowed::ExceptUpvars,
            flow_state,
        );

        if !errors_reported.mutability_error {
            // check for reassignments to immutable local variables
            self.check_if_reassignment_to_immutable_state(context, place_span, flow_state);
        }
    }

    fn consume_rvalue(
        &mut self,
        context: Context,
        (rvalue, span): (&Rvalue<'tcx>, Span),
        _location: Location,
        flow_state: &Flows<'cx, 'gcx, 'tcx>,
    ) {
        match *rvalue {
            Rvalue::Ref(_ /*rgn*/, bk, ref place) => {
                let access_kind = match bk {
                    BorrowKind::Shared => (Deep, Read(ReadKind::Borrow(bk))),
                    BorrowKind::Unique | BorrowKind::Mut { .. } => {
                        let wk = WriteKind::MutableBorrow(bk);
                        if self.allow_two_phase_borrow(bk) {
                            (Deep, Reservation(wk))
                        } else {
                            (Deep, Write(wk))
                        }
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

            Rvalue::Use(ref operand)
            | Rvalue::Repeat(ref operand, _)
            | Rvalue::UnaryOp(_ /*un_op*/, ref operand)
            | Rvalue::Cast(_ /*cast_kind*/, ref operand, _ /*ty*/) => {
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

            Rvalue::BinaryOp(_bin_op, ref operand1, ref operand2)
            | Rvalue::CheckedBinaryOp(_bin_op, ref operand1, ref operand2) => {
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
        flow_state: &Flows<'cx, 'gcx, 'tcx>,
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
    fn check_for_invalidation_at_exit(
        &mut self,
        context: Context,
        borrow: &BorrowData<'tcx>,
        span: Span,
        flow_state: &Flows<'cx, 'gcx, 'tcx>,
    ) {
        debug!("check_for_invalidation_at_exit({:?})", borrow);
        let place = &borrow.borrowed_place;
        let root_place = self.prefixes(place, PrefixSet::All).last().unwrap();

        // FIXME(nll-rfc#40): do more precise destructor tracking here. For now
        // we just know that all locals are dropped at function exit (otherwise
        // we'll have a memory leak) and assume that all statics have a destructor.
        //
        // FIXME: allow thread-locals to borrow other thread locals?
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
                (false, self.locals_are_invalidated_at_exit)
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
            return;
        }

        // FIXME: replace this with a proper borrow_conflicts_with_place when
        // that is merged.
        let sd = if might_be_alive { Deep } else { Shallow(None) };

        if self.places_conflict(place, root_place, sd) {
            debug!("check_for_invalidation_at_exit({:?}): INVALID", place);
            // FIXME: should be talking about the region lifetime instead
            // of just a span here.
            let span = self.tcx.sess.codemap().end_point(span);
            self.report_borrowed_value_does_not_live_long_enough(
                context,
                borrow,
                span,
                flow_state.borrows.operator(),
            )
        }
    }

    /// Reports an error if this is a borrow of local data.
    /// This is called for all Yield statements on movable generators
    fn check_for_local_borrow(&mut self, borrow: &BorrowData<'tcx>, yield_span: Span) {
        fn borrow_of_local_data<'tcx>(place: &Place<'tcx>) -> bool {
            match place {
                Place::Static(..) => false,
                Place::Local(..) => true,
                Place::Projection(box proj) => {
                    match proj.elem {
                        // Reborrow of already borrowed data is ignored
                        // Any errors will be caught on the initial borrow
                        ProjectionElem::Deref => false,

                        // For interior references and downcasts, find out if the base is local
                        ProjectionElem::Field(..)
                        | ProjectionElem::Index(..)
                        | ProjectionElem::ConstantIndex { .. }
                        | ProjectionElem::Subslice { .. }
                        | ProjectionElem::Downcast(..) => borrow_of_local_data(&proj.base),
                    }
                }
            }
        }

        debug!("check_for_local_borrow({:?})", borrow);

        if borrow_of_local_data(&borrow.borrowed_place) {
            self.tcx
                .cannot_borrow_across_generator_yield(
                    self.retrieve_borrow_span(borrow),
                    yield_span,
                    Origin::Mir,
                )
                .emit();
        }
    }

    fn check_activations(
        &mut self,
        location: Location,
        span: Span,
        flow_state: &Flows<'cx, 'gcx, 'tcx>,
    ) {
        if !self.tcx.two_phase_borrows() {
            return;
        }

        // Two-phase borrow support: For each activation that is newly
        // generated at this statement, check if it interferes with
        // another borrow.
        let domain = flow_state.borrows.operator();
        let data = domain.borrows();
        flow_state.borrows.each_gen_bit(|gen| {
            if gen.is_activation() {
                let borrow_index = gen.borrow_index();
                let borrow = &data[borrow_index];
                // currently the flow analysis registers
                // activations for both mutable and immutable
                // borrows. So make sure we are talking about a
                // mutable borrow before we check it.
                match borrow.kind {
                    BorrowKind::Shared => return,
                    BorrowKind::Unique | BorrowKind::Mut { .. } => {}
                }

                self.access_place(
                    ContextKind::Activation.new(location),
                    (&borrow.borrowed_place, span),
                    (
                        Deep,
                        Activation(WriteKind::MutableBorrow(borrow.kind), borrow_index),
                    ),
                    LocalMutationIsAllowed::No,
                    flow_state,
                );
                // We do not need to call `check_if_path_is_moved`
                // again, as we already called it when we made the
                // initial reservation.
            }
        });
    }
}

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    fn check_if_reassignment_to_immutable_state(
        &mut self,
        context: Context,
        (place, span): (&Place<'tcx>, Span),
        flow_state: &Flows<'cx, 'gcx, 'tcx>,
    ) {
        debug!("check_if_reassignment_to_immutable_state({:?})", place);
        // determine if this path has a non-mut owner (and thus needs checking).
        if let Ok(()) = self.is_mutable(place, LocalMutationIsAllowed::No) {
            return;
        }
        debug!(
            "check_if_reassignment_to_immutable_state({:?}) - is an imm local",
            place
        );

        for i in flow_state.ever_inits.iter_incoming() {
            let init = self.move_data.inits[i];
            let init_place = &self.move_data.move_paths[init.path].place;
            if self.places_conflict(&init_place, place, Deep) {
                self.report_illegal_reassignment(context, (place, span), init.span);
                break;
            }
        }
    }

    fn check_if_path_is_moved(
        &mut self,
        context: Context,
        desired_action: InitializationRequiringAction,
        place_span: (&Place<'tcx>, Span),
        flow_state: &Flows<'cx, 'gcx, 'tcx>,
    ) {
        // FIXME: analogous code in check_loans first maps `place` to
        // its base_path ... but is that what we want here?
        let place = self.base_path(place_span.0);

        let maybe_uninits = &flow_state.uninits;
        let curr_move_outs = &flow_state.move_outs;

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
                if maybe_uninits.contains(&mpi) {
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
    /// Currently this can only occur if the place is built off of a
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
            Place::Projection(_) => panic!("PrefixSet::All meant don't stop for Projection"),
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
        flow_state: &Flows<'cx, 'gcx, 'tcx>,
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
                            panic!("we don't allow assignments to subslices, context: {:?}",
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

    fn specialized_description(&self, place:&Place<'tcx>) -> Option<String>{
        if let Some(_name) = self.describe_place(place) {
            Some(format!("data in a `&` reference"))
        } else {
            None
        }
    }

    fn get_default_err_msg(&self, place:&Place<'tcx>) -> String{
        match self.describe_place(place) {
            Some(name) => format!("immutable item `{}`", name),
            None => "immutable item".to_owned(),
        }
    }

    fn get_secondary_err_msg(&self, place:&Place<'tcx>) -> String{
        match self.specialized_description(place) {
            Some(_) => format!("data in a `&` reference"),
            None => self.get_default_err_msg(place)
        }
    }

    fn get_primary_err_msg(&self, place:&Place<'tcx>) -> String{
        if let Some(name) = self.describe_place(place) {
            format!("`{}` is a `&` reference, so the data it refers to cannot be written", name) 
        } else {
            format!("cannot assign through `&`-reference")
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
            place, kind, is_local_mutation_allowed
        );
        let mut error_reported = false;
        match kind {
            Reservation(WriteKind::MutableBorrow(BorrowKind::Unique))
            | Write(WriteKind::MutableBorrow(BorrowKind::Unique)) => {
                if let Err(_place_err) = self.is_mutable(place, LocalMutationIsAllowed::Yes) {
                    span_bug!(span, "&unique borrow for {:?} should not fail", place);
                }
            }
            Reservation(WriteKind::MutableBorrow(BorrowKind::Mut { .. }))
            | Write(WriteKind::MutableBorrow(BorrowKind::Mut { .. })) => if let Err(place_err) =
                self.is_mutable(place, is_local_mutation_allowed)
            {
                error_reported = true;
                let item_msg = self.get_default_err_msg(place);
                let mut err = self.tcx
                    .cannot_borrow_path_as_mutable(span, &item_msg, Origin::Mir);
                err.span_label(span, "cannot borrow as mutable");

                if place != place_err {
                    if let Some(name) = self.describe_place(place_err) {
                        err.note(&format!("the value which is causing this path not to be mutable \
                                           is...: `{}`", name));
                    }
                }

                err.emit();
            },
            Reservation(WriteKind::Mutate) | Write(WriteKind::Mutate) => {

                if let Err(place_err) = self.is_mutable(place, is_local_mutation_allowed) {
                    error_reported = true;
                    let mut err_info = None;
                    match *place_err {

                        Place::Projection(box Projection {
                        ref base, elem:ProjectionElem::Deref}) => {
                            match *base {
                                Place::Local(local) => {
                                    let locations = self.mir.find_assignments(local);
                                        if locations.len() > 0 {
                                            let item_msg = if error_reported {
                                                self.get_secondary_err_msg(base)
                                            } else {
                                                self.get_default_err_msg(place)
                                            };   
                                            err_info = Some((
                                                self.mir.source_info(locations[0]).span,
                                                    "consider changing this to be a \
                                                    mutable reference: `&mut`", item_msg,
                                                    self.get_primary_err_msg(base)));
                                        }
                                },
                            _ => {},
                            }
                        },
                        _ => {},
                    }

                    if let Some((err_help_span, err_help_stmt, item_msg, sec_span)) = err_info {
                        let mut err = self.tcx.cannot_assign(span, &item_msg, Origin::Mir);
                        err.span_suggestion(err_help_span, err_help_stmt, format!(""));
                        if place != place_err {
                            err.span_label(span, sec_span);
                        }
                        err.emit()
                    } else {
                        let item_msg_ = self.get_default_err_msg(place);
                        let mut err = self.tcx.cannot_assign(span, &item_msg_, Origin::Mir);
                        err.span_label(span, "cannot mutate");
                        if place != place_err {
                            if let Some(name) = self.describe_place(place_err) {
                                err.note(&format!("the value which is causing this path not to be \
                                                   mutable is...: `{}`", name));
                            }
                        }
                        err.emit();
                    }
                }
            }
            Reservation(WriteKind::Move)
            | Reservation(WriteKind::StorageDeadOrDrop)
            | Reservation(WriteKind::MutableBorrow(BorrowKind::Shared))
            | Write(WriteKind::Move)
            | Write(WriteKind::StorageDeadOrDrop)
            | Write(WriteKind::MutableBorrow(BorrowKind::Shared)) => {
                if let Err(_place_err) = self.is_mutable(place, is_local_mutation_allowed) {
                    self.tcx.sess.delay_span_bug(
                        span,
                        &format!(
                            "Accessing `{:?}` with the kind `{:?}` shouldn't be possible",
                            place, kind
                        ),
                    );
                }
            }
            Activation(..) => {} // permission checks are done at Reservation point.
            Read(ReadKind::Borrow(BorrowKind::Unique))
            | Read(ReadKind::Borrow(BorrowKind::Mut { .. }))
            | Read(ReadKind::Borrow(BorrowKind::Shared))
            | Read(ReadKind::Copy) => {} // Access authorized
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
                        LocalMutationIsAllowed::Yes | LocalMutationIsAllowed::ExceptUpvars => {
                            Ok(())
                        }
                        LocalMutationIsAllowed::No => Err(place),
                    },
                    Mutability::Mut => Ok(()),
                }
            }
            Place::Static(ref static_) =>
                if self.tcx.is_static(static_.def_id) != Some(hir::Mutability::MutMutable) {
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
                                        let mode = match self.is_upvar_field_projection(&proj.base)
                                        {
                                            Some(field)
                                                if {
                                                    self.mir.upvar_decls[field.index()].by_ref
                                                } =>
                                            {
                                                is_local_mutation_allowed
                                            }
                                            _ => LocalMutationIsAllowed::Yes,
                                        };

                                        self.is_mutable(&proj.base, mode)
                                    }
                                }
                            }
                            ty::TyRawPtr(tnm) => {
                                match tnm.mutbl {
                                    // `*const` raw pointers are not mutable
                                    hir::MutImmutable => return Err(place),
                                    // `*mut` raw pointers are always mutable, regardless of context
                                    // The users have to check by themselve.
                                    hir::MutMutable => return Ok(()),
                                }
                            }
                            // `Box<T>` owns its content, so mutable if its location is mutable
                            _ if base_ty.is_box() => {
                                self.is_mutable(&proj.base, is_local_mutation_allowed)
                            }
                            // Deref should only be for reference, pointers or boxes
                            _ => bug!("Deref of unexpected type: {:?}", base_ty),
                        }
                    }
                    // All other projections are owned by their base path, so mutable if
                    // base path is mutable
                    ProjectionElem::Field(..)
                    | ProjectionElem::Index(..)
                    | ProjectionElem::ConstantIndex { .. }
                    | ProjectionElem::Subslice { .. }
                    | ProjectionElem::Downcast(..) => {
                        if let Some(field) = self.is_upvar_field_projection(place) {
                            let decl = &self.mir.upvar_decls[field.index()];
                            debug!(
                                "decl.mutability={:?} local_mutation_is_allowed={:?} place={:?}",
                                decl, is_local_mutation_allowed, place
                            );
                            match (decl.mutability, is_local_mutation_allowed) {
                                (Mutability::Not, LocalMutationIsAllowed::No)
                                | (Mutability::Not, LocalMutationIsAllowed::ExceptUpvars) => {
                                    Err(place)
                                }
                                (Mutability::Not, LocalMutationIsAllowed::Yes)
                                | (Mutability::Mut, _) => {
                                    self.is_mutable(&proj.base, is_local_mutation_allowed)
                                }
                            }
                        } else {
                            self.is_mutable(&proj.base, is_local_mutation_allowed)
                        }
                    }
                }
            }
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
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum NoMovePathFound {
    ReachedStatic,
}

/// The degree of overlap between 2 places for borrow-checking.
enum Overlap {
    /// The places might partially overlap - in this case, we give
    /// up and say that they might conflict. This occurs when
    /// different fields of a union are borrowed. For example,
    /// if `u` is a union, we have no way of telling how disjoint
    /// `u.a.x` and `a.b.y` are.
    Arbitrary,
    /// The places have the same type, and are either completely disjoint
    /// or equal - i.e. they can't "partially" overlap as can occur with
    /// unions. This is the "base case" on which we recur for extensions
    /// of the place.
    EqualOrDisjoint,
    /// The places are disjoint, so we know all extensions of them
    /// will also be disjoint.
    Disjoint,
}

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    // Given that the bases of `elem1` and `elem2` are always either equal
    // or disjoint (and have the same type!), return the overlap situation
    // between `elem1` and `elem2`.
    fn place_element_conflict(&self, elem1: &Place<'tcx>, elem2: &Place<'tcx>) -> Overlap {
        match (elem1, elem2) {
            (Place::Local(l1), Place::Local(l2)) => {
                if l1 == l2 {
                    // the same local - base case, equal
                    debug!("place_element_conflict: DISJOINT-OR-EQ-LOCAL");
                    Overlap::EqualOrDisjoint
                } else {
                    // different locals - base case, disjoint
                    debug!("place_element_conflict: DISJOINT-LOCAL");
                    Overlap::Disjoint
                }
            }
            (Place::Static(static1), Place::Static(static2)) => {
                if static1.def_id != static2.def_id {
                    debug!("place_element_conflict: DISJOINT-STATIC");
                    Overlap::Disjoint
                } else if self.tcx.is_static(static1.def_id) == Some(hir::Mutability::MutMutable) {
                    // We ignore mutable statics - they can only be unsafe code.
                    debug!("place_element_conflict: IGNORE-STATIC-MUT");
                    Overlap::Disjoint
                } else {
                    debug!("place_element_conflict: DISJOINT-OR-EQ-STATIC");
                    Overlap::EqualOrDisjoint
                }
            }
            (Place::Local(_), Place::Static(_)) | (Place::Static(_), Place::Local(_)) => {
                debug!("place_element_conflict: DISJOINT-STATIC-LOCAL");
                Overlap::Disjoint
            }
            (Place::Projection(pi1), Place::Projection(pi2)) => {
                match (&pi1.elem, &pi2.elem) {
                    (ProjectionElem::Deref, ProjectionElem::Deref) => {
                        // derefs (e.g. `*x` vs. `*x`) - recur.
                        debug!("place_element_conflict: DISJOINT-OR-EQ-DEREF");
                        Overlap::EqualOrDisjoint
                    }
                    (ProjectionElem::Field(f1, _), ProjectionElem::Field(f2, _)) => {
                        if f1 == f2 {
                            // same field (e.g. `a.y` vs. `a.y`) - recur.
                            debug!("place_element_conflict: DISJOINT-OR-EQ-FIELD");
                            Overlap::EqualOrDisjoint
                        } else {
                            let ty = pi1.base.ty(self.mir, self.tcx).to_ty(self.tcx);
                            match ty.sty {
                                ty::TyAdt(def, _) if def.is_union() => {
                                    // Different fields of a union, we are basically stuck.
                                    debug!("place_element_conflict: STUCK-UNION");
                                    Overlap::Arbitrary
                                }
                                _ => {
                                    // Different fields of a struct (`a.x` vs. `a.y`). Disjoint!
                                    debug!("place_element_conflict: DISJOINT-FIELD");
                                    Overlap::Disjoint
                                }
                            }
                        }
                    }
                    (ProjectionElem::Downcast(_, v1), ProjectionElem::Downcast(_, v2)) => {
                        // different variants are treated as having disjoint fields,
                        // even if they occupy the same "space", because it's
                        // impossible for 2 variants of the same enum to exist
                        // (and therefore, to be borrowed) at the same time.
                        //
                        // Note that this is different from unions - we *do* allow
                        // this code to compile:
                        //
                        // ```
                        // fn foo(x: &mut Result<i32, i32>) {
                        //     let mut v = None;
                        //     if let Ok(ref mut a) = *x {
                        //         v = Some(a);
                        //     }
                        //     // here, you would *think* that the
                        //     // *entirety* of `x` would be borrowed,
                        //     // but in fact only the `Ok` variant is,
                        //     // so the `Err` variant is *entirely free*:
                        //     if let Err(ref mut a) = *x {
                        //         v = Some(a);
                        //     }
                        //     drop(v);
                        // }
                        // ```
                        if v1 == v2 {
                            debug!("place_element_conflict: DISJOINT-OR-EQ-FIELD");
                            Overlap::EqualOrDisjoint
                        } else {
                            debug!("place_element_conflict: DISJOINT-FIELD");
                            Overlap::Disjoint
                        }
                    }
                    (ProjectionElem::Index(..), ProjectionElem::Index(..))
                    | (ProjectionElem::Index(..), ProjectionElem::ConstantIndex { .. })
                    | (ProjectionElem::Index(..), ProjectionElem::Subslice { .. })
                    | (ProjectionElem::ConstantIndex { .. }, ProjectionElem::Index(..))
                    | (
                        ProjectionElem::ConstantIndex { .. },
                        ProjectionElem::ConstantIndex { .. },
                    )
                    | (ProjectionElem::ConstantIndex { .. }, ProjectionElem::Subslice { .. })
                    | (ProjectionElem::Subslice { .. }, ProjectionElem::Index(..))
                    | (ProjectionElem::Subslice { .. }, ProjectionElem::ConstantIndex { .. })
                    | (ProjectionElem::Subslice { .. }, ProjectionElem::Subslice { .. }) => {
                        // Array indexes (`a[0]` vs. `a[i]`). These can either be disjoint
                        // (if the indexes differ) or equal (if they are the same), so this
                        // is the recursive case that gives "equal *or* disjoint" its meaning.
                        //
                        // Note that by construction, MIR at borrowck can't subdivide
                        // `Subslice` accesses (e.g. `a[2..3][i]` will never be present) - they
                        // are only present in slice patterns, and we "merge together" nested
                        // slice patterns. That means we don't have to think about these. It's
                        // probably a good idea to assert this somewhere, but I'm too lazy.
                        //
                        // FIXME(#8636) we might want to return Disjoint if
                        // both projections are constant and disjoint.
                        debug!("place_element_conflict: DISJOINT-OR-EQ-ARRAY");
                        Overlap::EqualOrDisjoint
                    }

                    (ProjectionElem::Deref, _)
                    | (ProjectionElem::Field(..), _)
                    | (ProjectionElem::Index(..), _)
                    | (ProjectionElem::ConstantIndex { .. }, _)
                    | (ProjectionElem::Subslice { .. }, _)
                    | (ProjectionElem::Downcast(..), _) => bug!(
                        "mismatched projections in place_element_conflict: {:?} and {:?}",
                        elem1,
                        elem2
                    ),
                }
            }
            (Place::Projection(_), _) | (_, Place::Projection(_)) => bug!(
                "unexpected elements in place_element_conflict: {:?} and {:?}",
                elem1,
                elem2
            ),
        }
    }

    /// Returns whether an access of kind `access` to `access_place` conflicts with
    /// a borrow/full access to `borrow_place` (for deep accesses to mutable
    /// locations, this function is symmetric between `borrow_place` & `access_place`).
    fn places_conflict(
        &mut self,
        borrow_place: &Place<'tcx>,
        access_place: &Place<'tcx>,
        access: ShallowOrDeep,
    ) -> bool {
        debug!(
            "places_conflict({:?},{:?},{:?})",
            borrow_place, access_place, access
        );

        // Return all the prefixes of `place` in reverse order, including
        // downcasts.
        fn place_elements<'a, 'tcx>(place: &'a Place<'tcx>) -> Vec<&'a Place<'tcx>> {
            let mut result = vec![];
            let mut place = place;
            loop {
                result.push(place);
                match place {
                    Place::Projection(interior) => {
                        place = &interior.base;
                    }
                    Place::Local(_) | Place::Static(_) => {
                        result.reverse();
                        return result;
                    }
                }
            }
        }

        let borrow_components = place_elements(borrow_place);
        let access_components = place_elements(access_place);
        debug!(
            "places_conflict: components {:?} / {:?}",
            borrow_components, access_components
        );

        let borrow_components = borrow_components
            .into_iter()
            .map(Some)
            .chain(iter::repeat(None));
        let access_components = access_components
            .into_iter()
            .map(Some)
            .chain(iter::repeat(None));
        // The borrowck rules for proving disjointness are applied from the "root" of the
        // borrow forwards, iterating over "similar" projections in lockstep until
        // we can prove overlap one way or another. Essentially, we treat `Overlap` as
        // a monoid and report a conflict if the product ends up not being `Disjoint`.
        //
        // At each step, if we didn't run out of borrow or place, we know that our elements
        // have the same type, and that they only overlap if they are the identical.
        //
        // For example, if we are comparing these:
        // BORROW:  (*x1[2].y).z.a
        // ACCESS:  (*x1[i].y).w.b
        //
        // Then our steps are:
        //       x1         |   x1          -- places are the same
        //       x1[2]      |   x1[i]       -- equal or disjoint (disjoint if indexes differ)
        //       x1[2].y    |   x1[i].y     -- equal or disjoint
        //      *x1[2].y    |  *x1[i].y     -- equal or disjoint
        //     (*x1[2].y).z | (*x1[i].y).w  -- we are disjoint and don't need to check more!
        //
        // Because `zip` does potentially bad things to the iterator inside, this loop
        // also handles the case where the access might be a *prefix* of the borrow, e.g.
        //
        // BORROW:  (*x1[2].y).z.a
        // ACCESS:  x1[i].y
        //
        // Then our steps are:
        //       x1         |   x1          -- places are the same
        //       x1[2]      |   x1[i]       -- equal or disjoint (disjoint if indexes differ)
        //       x1[2].y    |   x1[i].y     -- equal or disjoint
        //
        // -- here we run out of access - the borrow can access a part of it. If this
        // is a full deep access, then we *know* the borrow conflicts with it. However,
        // if the access is shallow, then we can proceed:
        //
        //       x1[2].y    | (*x1[i].y)    -- a deref! the access can't get past this, so we
        //                                     are disjoint
        //
        // Our invariant is, that at each step of the iteration:
        //  - If we didn't run out of access to match, our borrow and access are comparable
        //    and either equal or disjoint.
        //  - If we did run out of accesss, the borrow can access a part of it.
        for (borrow_c, access_c) in borrow_components.zip(access_components) {
            // loop invariant: borrow_c is always either equal to access_c or disjoint from it.
            debug!("places_conflict: {:?} vs. {:?}", borrow_c, access_c);
            match (borrow_c, access_c) {
                (None, _) => {
                    // If we didn't run out of access, the borrow can access all of our
                    // place (e.g. a borrow of `a.b` with an access to `a.b.c`),
                    // so we have a conflict.
                    //
                    // If we did, then we still know that the borrow can access a *part*
                    // of our place that our access cares about (a borrow of `a.b.c`
                    // with an access to `a.b`), so we still have a conflict.
                    //
                    // FIXME: Differs from AST-borrowck; includes drive-by fix
                    // to #38899. Will probably need back-compat mode flag.
                    debug!("places_conflict: full borrow, CONFLICT");
                    return true;
                }
                (Some(borrow_c), None) => {
                    // We know that the borrow can access a part of our place. This
                    // is a conflict if that is a part our access cares about.

                    let (base, elem) = match borrow_c {
                        Place::Projection(box Projection { base, elem }) => (base, elem),
                        _ => bug!("place has no base?"),
                    };
                    let base_ty = base.ty(self.mir, self.tcx).to_ty(self.tcx);

                    match (elem, &base_ty.sty, access) {
                        (_, _, Shallow(Some(ArtificialField::Discriminant)))
                        | (_, _, Shallow(Some(ArtificialField::ArrayLength))) => {
                            // The discriminant and array length are like
                            // additional fields on the type; they do not
                            // overlap any existing data there. Furthermore,
                            // they cannot actually be a prefix of any
                            // borrowed place (at least in MIR as it is
                            // currently.)
                            //
                            // e.g. a (mutable) borrow of `a[5]` while we read the
                            // array length of `a`.
                            debug!("places_conflict: implicit field");
                            return false;
                        }

                        (ProjectionElem::Deref, _, Shallow(None)) => {
                            // e.g. a borrow of `*x.y` while we shallowly access `x.y` or some
                            // prefix thereof - the shallow access can't touch anything behind
                            // the pointer.
                            debug!("places_conflict: shallow access behind ptr");
                            return false;
                        }
                        (
                            ProjectionElem::Deref,
                            ty::TyRef(
                                _,
                                ty::TypeAndMut {
                                    ty: _,
                                    mutbl: hir::MutImmutable,
                                },
                            ),
                            _,
                        ) => {
                            // the borrow goes through a dereference of a shared reference.
                            //
                            // I'm not sure why we are tracking these borrows - shared
                            // references can *always* be aliased, which means the
                            // permission check already account for this borrow.
                            debug!("places_conflict: behind a shared ref");
                            return false;
                        }

                        (ProjectionElem::Deref, _, Deep)
                        | (ProjectionElem::Field { .. }, _, _)
                        | (ProjectionElem::Index { .. }, _, _)
                        | (ProjectionElem::ConstantIndex { .. }, _, _)
                        | (ProjectionElem::Subslice { .. }, _, _)
                        | (ProjectionElem::Downcast { .. }, _, _) => {
                            // Recursive case. This can still be disjoint on a
                            // further iteration if this a shallow access and
                            // there's a deref later on, e.g. a borrow
                            // of `*x.y` while accessing `x`.
                        }
                    }
                }
                (Some(borrow_c), Some(access_c)) => {
                    match self.place_element_conflict(&borrow_c, access_c) {
                        Overlap::Arbitrary => {
                            // We have encountered different fields of potentially
                            // the same union - the borrow now partially overlaps.
                            //
                            // There is no *easy* way of comparing the fields
                            // further on, because they might have different types
                            // (e.g. borrows of `u.a.0` and `u.b.y` where `.0` and
                            // `.y` come from different structs).
                            //
                            // We could try to do some things here - e.g. count
                            // dereferences - but that's probably not a good
                            // idea, at least for now, so just give up and
                            // report a conflict. This is unsafe code anyway so
                            // the user could always use raw pointers.
                            debug!("places_conflict: arbitrary -> conflict");
                            return true;
                        }
                        Overlap::EqualOrDisjoint => {
                            // This is the recursive case - proceed to the next element.
                        }
                        Overlap::Disjoint => {
                            // We have proven the borrow disjoint - further
                            // projections will remain disjoint.
                            debug!("places_conflict: disjoint");
                            return false;
                        }
                    }
                }
            }
        }
        unreachable!("iter::repeat returned None")
    }

    /// This function iterates over all of the current borrows
    /// (represented by 1-bits in `flow_state.borrows`) that conflict
    /// with an access to a place, invoking the `op` callback for each
    /// one.
    ///
    /// "Current borrow" here means a borrow that reaches the point in
    /// the control-flow where the access occurs.
    ///
    /// The borrow's phase is represented by the ReserveOrActivateIndex
    /// passed to the callback: one can call `is_reservation()` and
    /// `is_activation()` to determine what phase the borrow is
    /// currently in, when such distinction matters.
    fn each_borrow_involving_path<F>(
        &mut self,
        _context: Context,
        access_place: (ShallowOrDeep, &Place<'tcx>),
        flow_state: &Flows<'cx, 'gcx, 'tcx>,
        mut op: F,
    ) where
        F: FnMut(&mut Self, ReserveOrActivateIndex, &BorrowData<'tcx>) -> Control,
    {
        let (access, place) = access_place;

        // FIXME: analogous code in check_loans first maps `place` to
        // its base_path.

        let data = flow_state.borrows.operator().borrows();

        // check for loan restricting path P being used. Accounts for
        // borrows of P, P.a.b, etc.
        let mut iter_incoming = flow_state.borrows.iter_incoming();
        while let Some(i) = iter_incoming.next() {
            let borrowed = &data[i.borrow_index()];

            if self.places_conflict(&borrowed.borrowed_place, place, access) {
                debug!(
                    "each_borrow_involving_path: {:?} @ {:?} vs. {:?}/{:?}",
                    i, borrowed, place, access
                );
                let ctrl = op(self, i, borrowed);
                if ctrl == Control::Break {
                    return;
                }
            }
        }
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
    Activation,
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

