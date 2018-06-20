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

use borrow_check::nll::region_infer::RegionInferenceContext;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::hir::map::definitions::DefPathData;
use rustc::infer::InferCtxt;
use rustc::ty::{self, ParamEnv, TyCtxt};
use rustc::ty::query::Providers;
use rustc::lint::builtin::UNUSED_MUT;
use rustc::mir::{self, AggregateKind, BasicBlock, BorrowCheckResult, BorrowKind};
use rustc::mir::{ClearCrossCrate, Local, Location, Place, Mir, Mutability, Operand};
use rustc::mir::{Projection, ProjectionElem, Rvalue, Field, Statement, StatementKind};
use rustc::mir::{Terminator, TerminatorKind};

use rustc_data_structures::control_flow_graph::dominators::Dominators;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::indexed_set::IdxSetBuf;
use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::small_vec::SmallVec;

use std::rc::Rc;

use syntax_pos::Span;

use dataflow::{do_dataflow, DebugFormatted};
use dataflow::FlowAtLocation;
use dataflow::MoveDataParamEnv;
use dataflow::{DataflowResultsConsumer};
use dataflow::{MaybeInitializedPlaces, MaybeUninitializedPlaces};
use dataflow::{EverInitializedPlaces, MovingOutStatements};
use dataflow::Borrows;
use dataflow::indexes::BorrowIndex;
use dataflow::move_paths::{IllegalMoveOriginKind, MoveError};
use dataflow::move_paths::{HasMoveData, LookupResult, MoveData, MovePathIndex};
use util::borrowck_errors::{BorrowckErrors, Origin};
use util::collect_writes::FindAssignments;

use self::borrow_set::{BorrowSet, BorrowData};
use self::flows::Flows;
use self::location::LocationTable;
use self::prefixes::PrefixSet;
use self::MutateMode::{JustWrite, WriteAndRead};

use self::path_utils::*;

crate mod borrow_set;
mod error_reporting;
mod flows;
mod location;
crate mod place_ext;
mod prefixes;
mod path_utils;
mod used_muts;

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
) -> BorrowCheckResult<'tcx> {
    let input_mir = tcx.mir_validated(def_id);
    debug!("run query mir_borrowck: {}", tcx.item_path_str(def_id));

    if !tcx.has_attr(def_id, "rustc_mir_borrowck") && !tcx.use_mir_borrowck() {
        return BorrowCheckResult {
            closure_requirements: None,
            used_mut_upvars: SmallVec::new(),
        };
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
) -> BorrowCheckResult<'gcx> {
    let tcx = infcx.tcx;
    let attributes = tcx.get_attrs(def_id);
    let param_env = tcx.param_env(def_id);
    let id = tcx.hir
        .as_local_node_id(def_id)
        .expect("do_mir_borrowck: non-local DefId");

    // Replace all regions with fresh inference variables. This
    // requires first making our own copy of the MIR. This copy will
    // be modified (in place) to contain non-lexical lifetimes. It
    // will have a lifetime tied to the inference context.
    let mut mir: Mir<'tcx> = input_mir.clone();
    let free_regions = nll::replace_regions_in_mir(infcx, def_id, param_env, &mut mir);
    let mir = &mir; // no further changes
    let location_table = &LocationTable::new(mir);

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
                    IllegalMoveOriginKind::BorrowedContent { target_ty: ty } => {
                        // Inspect the type of the content behind the
                        // borrow to provide feedback about why this
                        // was a move rather than a copy.
                        match ty.sty {
                            ty::TyArray(..) | ty::TySlice(..) =>
                                tcx.cannot_move_out_of_interior_noncopy(span, ty, None, origin),
                            _ => tcx.cannot_move_out_of(span, "borrowed content", origin)
                        }
                    }
                    IllegalMoveOriginKind::InteriorOfTypeWithDestructor { container_ty: ty } => {
                        tcx.cannot_move_out_of_interior_of_drop(span, ty, origin)
                    }
                    IllegalMoveOriginKind::InteriorOfSliceOrArray { ty, is_index } => {
                        tcx.cannot_move_out_of_interior_noncopy(span, ty, Some(is_index), origin)
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

    let borrow_set = Rc::new(BorrowSet::build(tcx, mir));

    // If we are in non-lexical mode, compute the non-lexical lifetimes.
    let (regioncx, polonius_output, opt_closure_req) = nll::compute_regions(
        infcx,
        def_id,
        free_regions,
        mir,
        location_table,
        param_env,
        &mut flow_inits,
        &mdpe.move_data,
        &borrow_set,
    );
    let regioncx = Rc::new(regioncx);
    let flow_inits = flow_inits; // remove mut

    let flow_borrows = FlowAtLocation::new(do_dataflow(
        tcx,
        mir,
        id,
        &attributes,
        &dead_unwinds,
        Borrows::new(tcx, mir, regioncx.clone(), def_id, body_id, &borrow_set),
        |rs, i| DebugFormatted::new(&rs.location(i)),
    ));

    let movable_generator = match tcx.hir.get(id) {
        hir::map::Node::NodeExpr(&hir::Expr {
            node: hir::ExprClosure(.., Some(hir::GeneratorMovability::Static)),
            ..
        }) => false,
        _ => true,
    };

    let dominators = mir.dominators();

    let mut mbcx = MirBorrowckCtxt {
        tcx: tcx,
        mir: mir,
        mir_def_id: def_id,
        move_data: &mdpe.move_data,
        param_env: param_env,
        movable_generator,
        locals_are_invalidated_at_exit: match tcx.hir.body_owner_kind(id) {
            hir::BodyOwnerKind::Const | hir::BodyOwnerKind::Static(_) => false,
            hir::BodyOwnerKind::Fn => true,
        },
        access_place_error_reported: FxHashSet(),
        reservation_error_reported: FxHashSet(),
        moved_error_reported: FxHashSet(),
        nonlexical_regioncx: regioncx,
        used_mut: FxHashSet(),
        used_mut_upvars: SmallVec::new(),
        borrow_set,
        dominators,
    };

    let mut state = Flows::new(
        flow_borrows,
        flow_inits,
        flow_uninits,
        flow_move_outs,
        flow_ever_inits,
        polonius_output,
    );

    mbcx.analyze_results(&mut state); // entry point for DataflowResultsConsumer

    // For each non-user used mutable variable, check if it's been assigned from
    // a user-declared local. If so, then put that local into the used_mut set.
    // Note that this set is expected to be small - only upvars from closures
    // would have a chance of erroneously adding non-user-defined mutable vars
    // to the set.
    let temporary_used_locals: FxHashSet<Local> =
        mbcx.used_mut.iter()
            .filter(|&local| !mbcx.mir.local_decls[*local].is_user_variable.is_some())
            .cloned()
            .collect();
    mbcx.gather_used_muts(temporary_used_locals);

    debug!("mbcx.used_mut: {:?}", mbcx.used_mut);

    for local in mbcx.mir.mut_vars_and_args_iter().filter(|local| !mbcx.used_mut.contains(local)) {
        if let ClearCrossCrate::Set(ref vsi) = mbcx.mir.source_scope_local_data {
            let local_decl = &mbcx.mir.local_decls[local];

            // Skip implicit `self` argument for closures
            if local.index() == 1 && tcx.is_closure(mbcx.mir_def_id) {
                continue;
            }

            // Skip over locals that begin with an underscore or have no name
            match local_decl.name {
                Some(name) => if name.as_str().starts_with("_") { continue; },
                None => continue,
            }

            let span = local_decl.source_info.span;
            let mut_span = tcx.sess.codemap().span_until_non_whitespace(span);

            tcx.struct_span_lint_node(
                UNUSED_MUT,
                vsi[local_decl.source_info.scope].lint_root,
                span,
                "variable does not need to be mutable"
            )
            .span_suggestion_short(mut_span, "remove this `mut`", "".to_owned())
            .emit();
        }
    }

    BorrowCheckResult {
        closure_requirements: opt_closure_req,
        used_mut_upvars: mbcx.used_mut_upvars,
    }
}

#[allow(dead_code)]
pub struct MirBorrowckCtxt<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    mir: &'cx Mir<'tcx>,
    mir_def_id: DefId,
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
    /// This field keeps track of errors reported in the checking of moved variables,
    /// so that we don't report report seemingly duplicate errors.
    moved_error_reported: FxHashSet<Place<'tcx>>,
    /// This field keeps track of all the local variables that are declared mut and are mutated.
    /// Used for the warning issued by an unused mutable local variable.
    used_mut: FxHashSet<Local>,
    /// If the function we're checking is a closure, then we'll need to report back the list of
    /// mutable upvars that have been used. This field keeps track of them.
    used_mut_upvars: SmallVec<[Field; 8]>,
    /// Non-lexical region inference context, if NLL is enabled.  This
    /// contains the results from region inference and lets us e.g.
    /// find out which CFG points are contained in each borrow region.
    nonlexical_regioncx: Rc<RegionInferenceContext<'tcx>>,

    /// The set of borrows extracted from the MIR
    borrow_set: Rc<BorrowSet<'tcx>>,

    /// Dominators for MIR
    dominators: Dominators<BasicBlock>,
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
            StatementKind::ReadForMatch(ref place) => {
                self.access_place(ContextKind::ReadForMatch.new(location),
                                  (place, span),
                                  (Deep, Read(ReadKind::Borrow(BorrowKind::Shared))),
                                  LocalMutationIsAllowed::No,
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
                        self.check_if_path_or_subpath_is_moved(
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
                use rustc::mir::interpret::EvalErrorKind::BoundsCheck;
                if let BoundsCheck { ref len, ref index } = *msg {
                    self.consume_operand(ContextKind::Assert.new(loc), (len, span), flow_state);
                    self.consume_operand(
                        ContextKind::Assert.new(loc),
                        (index, span),
                        flow_state,
                    );
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
                    let borrow_set = self.borrow_set.clone();
                    flow_state.with_outgoing_borrows(|borrows| {
                        for i in borrows {
                            let borrow = &borrow_set[i];
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
                let borrow_set = self.borrow_set.clone();
                flow_state.with_outgoing_borrows(|borrows| {
                    for i in borrows {
                        let borrow = &borrow_set[i];
                        let context = ContextKind::StorageDead.new(loc);
                        self.check_for_invalidation_at_exit(context, borrow, span);
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

struct RootPlace<'d, 'tcx: 'd> {
    place: &'d Place<'tcx>,
    is_local_mutation_allowed: LocalMutationIsAllowed,
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
        let gcx = self.tcx.global_tcx();
        let drop_field = |
            mir: &mut MirBorrowckCtxt<'cx, 'gcx, 'tcx>,
            (index, field): (usize, ty::Ty<'gcx>),
        | {
            let field_ty = gcx.normalize_erasing_regions(mir.param_env, field);
            let place = drop_place.clone().field(Field::new(index), field_ty);

            mir.visit_terminator_drop(loc, term, flow_state, &place, field_ty, span);
        };

        match erased_drop_place_ty.sty {
            // When a struct is being dropped, we need to check
            // whether it has a destructor, if it does, then we can
            // call it, if it does not then we need to check the
            // individual fields instead. This way if `foo` has a
            // destructor but `bar` does not, we will only check for
            // borrows of `x.foo` and not `x.bar`. See #47703.
            ty::TyAdt(def, substs) if def.is_struct() && !def.has_dtor(self.tcx) => {
                def.all_fields()
                    .map(|field| field.ty(gcx, substs))
                    .enumerate()
                    .for_each(|field| drop_field(self, field));
            }
            // Same as above, but for tuples.
            ty::TyTuple(tys) => {
                tys.iter().cloned().enumerate()
                    .for_each(|field| drop_field(self, field));
            }
            // Closures also have disjoint fields, but they are only
            // directly accessed in the body of the closure.
            ty::TyClosure(def, substs)
                if *drop_place == Place::Local(Local::new(1)) && !self.mir.upvar_decls.is_empty()
            => {
                substs.upvar_tys(def, self.tcx).enumerate()
                    .for_each(|field| drop_field(self, field));
            }
            // Generators also have disjoint fields, but they are only
            // directly accessed in the body of the generator.
            ty::TyGenerator(def, substs, _)
                if *drop_place == Place::Local(Local::new(1)) && !self.mir.upvar_decls.is_empty()
            => {
                substs.upvar_tys(def, self.tcx).enumerate()
                    .for_each(|field| drop_field(self, field));
            }
            _ => {
                // We have now refined the type of the value being
                // dropped (potentially) to just the type of a
                // subfield; so check whether that field's type still
                // "needs drop". If so, we assume that the destructor
                // may access any data it likes (i.e., a Deep Write).
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
            self.check_access_permissions(place_span, rw, is_local_mutation_allowed, flow_state);
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
        debug!(
            "check_access_for_conflict(context={:?}, place_span={:?}, sd={:?}, rw={:?})",
            context,
            place_span,
            sd,
            rw,
        );

        let mut error_reported = false;
        let tcx = self.tcx;
        let mir = self.mir;
        let location_table = &LocationTable::new(mir);
        let location = location_table.start_index(context.loc);
        let borrow_set = self.borrow_set.clone();
        each_borrow_involving_path(
            self,
            tcx,
            mir,
            context,
            (sd, place_span.0),
            &borrow_set,
            flow_state.borrows_in_scope(location),
            |this, borrow_index, borrow|
            match (rw, borrow.kind) {
                // Obviously an activation is compatible with its own
                // reservation (or even prior activating uses of same
                // borrow); so don't check if they interfere.
                //
                // NOTE: *reservations* do conflict with themselves;
                // thus aren't injecting unsoundenss w/ this check.)
                (Activation(_, activating), _) if activating == borrow_index => {
                    debug!(
                        "check_access_for_conflict place_span: {:?} sd: {:?} rw: {:?} \
                         skipping {:?} b/c activation of same borrow_index",
                        place_span,
                        sd,
                        rw,
                        (borrow_index, borrow),
                    );
                    Control::Continue
                }

                (Read(_), BorrowKind::Shared) | (Reservation(..), BorrowKind::Shared) => {
                    Control::Continue
                }

                (Read(kind), BorrowKind::Unique) | (Read(kind), BorrowKind::Mut { .. }) => {
                    // Reading from mere reservations of mutable-borrows is OK.
                    if !is_active(&this.dominators, borrow, context.loc) {
                        assert!(allow_two_phase_borrow(&this.tcx, borrow.kind));
                        return Control::Continue;
                    }

                    match kind {
                        ReadKind::Copy => {
                            error_reported = true;
                            this.report_use_while_mutably_borrowed(context, place_span, borrow)
                        }
                        ReadKind::Borrow(bk) => {
                            error_reported = true;
                            this.report_conflicting_borrow(
                                context,
                                place_span,
                                bk,
                                &borrow,
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
                            error_reported = true;
                            this.report_conflicting_borrow(
                                context,
                                place_span,
                                bk,
                                &borrow,
                            )
                        }
                        WriteKind::StorageDeadOrDrop => {
                            error_reported = true;
                            this.report_borrowed_value_does_not_live_long_enough(
                                context,
                                borrow,
                                place_span,
                                Some(kind),
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
                self.check_if_path_or_subpath_is_moved(
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
                        if allow_two_phase_borrow(&self.tcx, bk) {
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

                self.check_if_path_or_subpath_is_moved(
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
                self.check_if_path_or_subpath_is_moved(
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

            Rvalue::Aggregate(ref aggregate_kind, ref operands) => {
                // We need to report back the list of mutable upvars that were
                // moved into the closure and subsequently used by the closure,
                // in order to populate our used_mut set.
                if let AggregateKind::Closure(def_id, _) = &**aggregate_kind {
                    let BorrowCheckResult { used_mut_upvars, .. } = self.tcx.mir_borrowck(*def_id);
                    debug!("{:?} used_mut_upvars={:?}", def_id, used_mut_upvars);
                    for field in used_mut_upvars {
                        match operands[field.index()] {
                            Operand::Move(Place::Local(local)) => {
                                self.used_mut.insert(local);
                            }
                            Operand::Move(ref place @ Place::Projection(_)) => {
                                if let Some(field) = self.is_upvar_field_projection(place) {
                                    self.used_mut_upvars.push(field);
                                }
                            }
                            Operand::Move(Place::Static(..)) |
                            Operand::Copy(..) |
                            Operand::Constant(..) => {}
                        }
                    }
                }

                for operand in operands {
                    self.consume_operand(context, (operand, span), flow_state);
                }
            }
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
                self.check_if_path_or_subpath_is_moved(
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
                self.check_if_path_or_subpath_is_moved(
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

        if places_conflict(self.tcx, self.mir, place, root_place, sd) {
            debug!("check_for_invalidation_at_exit({:?}): INVALID", place);
            // FIXME: should be talking about the region lifetime instead
            // of just a span here.
            let span = self.tcx.sess.codemap().end_point(span);
            self.report_borrowed_value_does_not_live_long_enough(
                context,
                borrow,
                (place, span),
                None,
            )
        }
    }

    /// Reports an error if this is a borrow of local data.
    /// This is called for all Yield statements on movable generators
    fn check_for_local_borrow(&mut self, borrow: &BorrowData<'tcx>, yield_span: Span) {
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
        let borrow_set = self.borrow_set.clone();
        for &borrow_index in borrow_set.activations_at_location(location) {
            let borrow = &borrow_set[borrow_index];

            // only mutable borrows should be 2-phase
            assert!(match borrow.kind {
                BorrowKind::Shared => false,
                BorrowKind::Unique | BorrowKind::Mut { .. } => true,
            });

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
            // We do not need to call `check_if_path_or_subpath_is_moved`
            // again, as we already called it when we made the
            // initial reservation.
        }
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
        let err_place = match self.is_mutable(place, LocalMutationIsAllowed::No) {
            Ok(..) => return,
            Err(place) => place,
        };
        debug!(
            "check_if_reassignment_to_immutable_state({:?}) - is an imm local",
            place
        );

        for i in flow_state.ever_inits.iter_incoming() {
            let init = self.move_data.inits[i];
            let init_place = &self.move_data.move_paths[init.path].place;
            if places_conflict(self.tcx, self.mir, &init_place, place, Deep) {
                self.report_illegal_reassignment(context, (place, span), init.span, err_place);
                break;
            }
        }
    }

    fn check_if_full_path_is_moved(
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
        // 3. Uninitialized `(a.b.c: &_)`, use of `*a.b.c`; note that with
        //    partial initialization support, one might have `a.x`
        //    initialized but not `a.b`.
        //
        // OK scenarios:
        //
        // 4. Move of `a.b.c`, use of `a.b.d`
        // 5. Uninitialized `a.x`, initialized `a.b`, use of `a.b`
        // 6. Copied `(a.b: &_)`, use of `*(a.b).c`; note that `a.b`
        //    must have been initialized for the use to be sound.
        // 7. Move of `a.b.c` then reinit of `a.b.c.d`, use of `a.b.c.d`

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
        // This code covers scenarios 1, 2, and 3.

        debug!("check_if_full_path_is_moved place: {:?}", place);
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
              // (I.e. querying parents breaks scenario 7; but may want
              // to do such a query based on partial-init feature-gate.)
        }
    }

    fn check_if_path_or_subpath_is_moved(
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
        // 1. Move of `a.b.c`, use of `a` or `a.b`
        //    partial initialization support, one might have `a.x`
        //    initialized but not `a.b`.
        // 2. All bad scenarios from `check_if_full_path_is_moved`
        //
        // OK scenarios:
        //
        // 3. Move of `a.b.c`, use of `a.b.d`
        // 4. Uninitialized `a.x`, initialized `a.b`, use of `a.b`
        // 5. Copied `(a.b: &_)`, use of `*(a.b).c`; note that `a.b`
        //    must have been initialized for the use to be sound.
        // 6. Move of `a.b.c` then reinit of `a.b.c.d`, use of `a.b.c.d`

        self.check_if_full_path_is_moved(context, desired_action, place_span, flow_state);

        // A move of any shallow suffix of `place` also interferes
        // with an attempt to use `place`. This is scenario 3 above.
        //
        // (Distinct from handling of scenarios 1+2+4 above because
        // `place` does not interfere with suffixes of its prefixes,
        // e.g. `a.b.c` does not interfere with `a.b.d`)
        //
        // This code covers scenario 1.

        debug!("check_if_path_or_subpath_is_moved place: {:?}", place);
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
        debug!("check_if_assigned_path_is_moved place: {:?}", place);
        // recur down place; dispatch to external checks when necessary
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
                        ProjectionElem::Index(_/*operand*/) |
                        ProjectionElem::ConstantIndex { .. } |
                        // assigning to P[i] requires P to be valid.
                        ProjectionElem::Downcast(_/*adt_def*/, _/*variant_idx*/) =>
                        // assigning to (P->variant) is okay if assigning to `P` is okay
                        //
                        // FIXME: is this true even if P is a adt with a dtor?
                        { }

                        // assigning to (*P) requires P to be initialized
                        ProjectionElem::Deref => {
                            self.check_if_full_path_is_moved(
                                context, InitializationRequiringAction::Use,
                                (base, span), flow_state);
                            // (base initialized; no need to
                            // recur further)
                            break;
                        }

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

                                    self.check_if_path_or_subpath_is_moved(
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
        &mut self,
        (place, span): (&Place<'tcx>, Span),
        kind: ReadOrWrite,
        is_local_mutation_allowed: LocalMutationIsAllowed,
        flow_state: &Flows<'cx, 'gcx, 'tcx>,
    ) -> bool {
        debug!(
            "check_access_permissions({:?}, {:?}, {:?})",
            place, kind, is_local_mutation_allowed
        );

        #[derive(Copy, Clone, Debug)]
        enum AccessKind {
            MutableBorrow,
            Mutate,
        }
        let error_access;
        let the_place_err;

        match kind {
            Reservation(WriteKind::MutableBorrow(borrow_kind @ BorrowKind::Unique))
            | Reservation(WriteKind::MutableBorrow(borrow_kind @ BorrowKind::Mut { .. }))
            | Write(WriteKind::MutableBorrow(borrow_kind @ BorrowKind::Unique))
            | Write(WriteKind::MutableBorrow(borrow_kind @ BorrowKind::Mut { .. })) =>
            {
                let is_local_mutation_allowed = match borrow_kind {
                    BorrowKind::Unique => LocalMutationIsAllowed::Yes,
                    BorrowKind::Mut { .. } => is_local_mutation_allowed,
                    BorrowKind::Shared => unreachable!(),
                };
                match self.is_mutable(place, is_local_mutation_allowed) {
                    Ok(root_place) => {
                        self.add_used_mut(root_place, flow_state);
                        return false;
                    }
                    Err(place_err) => {
                        error_access = AccessKind::MutableBorrow;
                        the_place_err = place_err;
                    }
                }
            }
            Reservation(WriteKind::Mutate) | Write(WriteKind::Mutate) => {
                match self.is_mutable(place, is_local_mutation_allowed) {
                    Ok(root_place) => {
                        self.add_used_mut(root_place, flow_state);
                        return false;
                    }
                    Err(place_err) => {
                        error_access = AccessKind::Mutate;
                        the_place_err = place_err;
                    }
                }
            }

            Reservation(WriteKind::Move)
            | Write(WriteKind::Move)
            | Reservation(WriteKind::StorageDeadOrDrop)
            | Reservation(WriteKind::MutableBorrow(BorrowKind::Shared))
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
                return false;
            }
            Activation(..) => {
                // permission checks are done at Reservation point.
                return false;
            }
            Read(ReadKind::Borrow(BorrowKind::Unique))
            | Read(ReadKind::Borrow(BorrowKind::Mut { .. }))
            | Read(ReadKind::Borrow(BorrowKind::Shared))
            | Read(ReadKind::Copy) => {
                // Access authorized
                return false;
            }
        }

        // at this point, we have set up the error reporting state.

        let mut err;
        let item_msg = match self.describe_place(place) {
            Some(name) => format!("immutable item `{}`", name),
            None => "immutable item".to_owned(),
        };

        // `act` and `acted_on` are strings that let us abstract over
        // the verbs used in some diagnostic messages.
        let act; let acted_on;

        match error_access {
            AccessKind::Mutate => {
                let item_msg = match the_place_err {
                    Place::Projection(box Projection {
                        base: _,
                        elem: ProjectionElem::Deref }
                    ) => match self.describe_place(place) {
                        Some(description) =>
                            format!("`{}` which is behind a `&` reference", description),
                        None => format!("data in a `&` reference"),
                    },
                    _ => item_msg,
                };
                err = self.tcx.cannot_assign(span, &item_msg, Origin::Mir);
                act = "assign"; acted_on = "written";
            }
            AccessKind::MutableBorrow => {
                err = self.tcx
                    .cannot_borrow_path_as_mutable(span, &item_msg, Origin::Mir);
                act = "borrow as mutable"; acted_on = "borrowed as mutable";
            }
        }

        match the_place_err {
            // We want to suggest users use `let mut` for local (user
            // variable) mutations...
            Place::Local(local) if self.mir.local_decls[*local].can_be_made_mutable() => {
                // ... but it doesn't make sense to suggest it on
                // variables that are `ref x`, `ref mut x`, `&self`,
                // or `&mut self` (such variables are simply not
                // mutable)..
                let local_decl = &self.mir.local_decls[*local];
                assert_eq!(local_decl.mutability, Mutability::Not);

                err.span_label(span, format!("cannot {ACT}", ACT=act));
                err.span_suggestion(local_decl.source_info.span,
                                    "consider changing this to be mutable",
                                    format!("mut {}", local_decl.name.unwrap()));
            }

            // complete hack to approximate old AST-borrowck
            // diagnostic: if the span starts with a mutable borrow of
            // a local variable, then just suggest the user remove it.
            Place::Local(_) if {
                if let Ok(snippet) = self.tcx.sess.codemap().span_to_snippet(span) {
                    snippet.starts_with("&mut ")
                } else {
                    false
                }
            } => {
                err.span_label(span, format!("cannot {ACT}", ACT=act));
                err.span_label(span, "try removing `&mut` here");
            }

            // We want to point out when a `&` can be readily replaced
            // with an `&mut`.
            //
            // FIXME: can this case be generalized to work for an
            // arbitrary base for the projection?
            Place::Projection(box Projection { base: Place::Local(local),
                                               elem: ProjectionElem::Deref })
                if self.mir.local_decls[*local].is_nonref_binding() =>
            {
                let (err_help_span, suggested_code) =
                    find_place_to_suggest_ampmut(self.tcx, self.mir, *local);
                err.span_suggestion(err_help_span,
                                    "consider changing this to be a mutable reference",
                                    suggested_code);

                let local_decl = &self.mir.local_decls[*local];
                if let Some(name) = local_decl.name {
                    err.span_label(
                        span, format!("`{NAME}` is a `&` reference, \
                                       so the data it refers to cannot be {ACTED_ON}",
                                      NAME=name, ACTED_ON=acted_on));
                } else {
                    err.span_label(span, format!("cannot {ACT} through `&`-reference", ACT=act));
                }
            }

            _ => {
                err.span_label(span, format!("cannot {ACT}", ACT=act));
            }
        }

        err.emit();
        return true;

        // Returns the span to highlight and the associated text to
        // present when suggesting that the user use an `&mut`.
        //
        // When we want to suggest a user change a local variable to be a `&mut`, there
        // are three potential "obvious" things to highlight:
        //
        // let ident [: Type] [= RightHandSideExresssion];
        //     ^^^^^    ^^^^     ^^^^^^^^^^^^^^^^^^^^^^^
        //     (1.)     (2.)              (3.)
        //
        // We can always fallback on highlighting the first. But chances are good that
        // the user experience will be better if we highlight one of the others if possible;
        // for example, if the RHS is present and the Type is not, then the type is going to
        // be inferred *from* the RHS, which means we should highlight that (and suggest
        // that they borrow the RHS mutably).
        fn find_place_to_suggest_ampmut<'cx, 'gcx, 'tcx>(tcx: TyCtxt<'cx, 'gcx, 'tcx>,
                                                         mir: &Mir<'tcx>,
                                                         local: Local) -> (Span, String)
        {
            // This implementation attempts to emulate AST-borrowck prioritization
            // by trying (3.), then (2.) and finally falling back on (1.).
            let locations = mir.find_assignments(local);
            if locations.len() > 0 {
                let assignment_rhs_span = mir.source_info(locations[0]).span;
                let snippet = tcx.sess.codemap().span_to_snippet(assignment_rhs_span);
                if let Ok(src) = snippet {
                    // pnkfelix inherited code; believes intention is
                    // highlighted text will always be `&<expr>` and
                    // thus can transform to `&mut` by slicing off
                    // first ASCII character and prepending "&mut ".
                    let borrowed_expr = src[1..].to_string();
                    return (assignment_rhs_span, format!("&mut {}", borrowed_expr));
                }
            }

            let local_decl = &mir.local_decls[local];
            let highlight_span = match local_decl.is_user_variable {
                // if this is a variable binding with an explicit type,
                // try to highlight that for the suggestion.
                Some(ClearCrossCrate::Set(mir::BindingForm::Var(mir::VarBindingForm {
                    opt_ty_info: Some(ty_span), .. }))) => ty_span,

                Some(ClearCrossCrate::Clear) => bug!("saw cleared local state"),

                // otherwise, just highlight the span associated with
                // the (MIR) LocalDecl.
                _ => local_decl.source_info.span,
            };

            let ty_mut = local_decl.ty.builtin_deref(true).unwrap();
            assert_eq!(ty_mut.mutbl, hir::MutImmutable);
            return (highlight_span, format!("&mut {}", ty_mut.ty));
        }
    }

    /// Adds the place into the used mutable variables set
    fn add_used_mut<'d>(
        &mut self,
        root_place: RootPlace<'d, 'tcx>,
        flow_state: &Flows<'cx, 'gcx, 'tcx>
    ) {
        match root_place {
            RootPlace {
                place: Place::Local(local),
                is_local_mutation_allowed,
            } => {
                if is_local_mutation_allowed != LocalMutationIsAllowed::Yes {
                    // If the local may be initialized, and it is now currently being
                    // mutated, then it is justified to be annotated with the `mut`
                    // keyword, since the mutation may be a possible reassignment.
                    let mpi = self.move_data.rev_lookup.find_local(*local);
                    let ii = &self.move_data.init_path_map[mpi];
                    for index in ii {
                        if flow_state.ever_inits.contains(index) {
                            self.used_mut.insert(*local);
                            break;
                        }
                    }
                }
            }
            RootPlace {
                place: place @ Place::Projection(_),
                is_local_mutation_allowed: _,
            } => {
                if let Some(field) = self.is_upvar_field_projection(&place) {
                    self.used_mut_upvars.push(field);
                }
            }
            RootPlace {
                place: Place::Static(..),
                is_local_mutation_allowed: _,
            } => {}
        }
    }

    /// Whether this value be written or borrowed mutably.
    /// Returns the root place if the place passed in is a projection.
    fn is_mutable<'d>(
        &self,
        place: &'d Place<'tcx>,
        is_local_mutation_allowed: LocalMutationIsAllowed,
    ) -> Result<RootPlace<'d, 'tcx>, &'d Place<'tcx>> {
        match *place {
            Place::Local(local) => {
                let local = &self.mir.local_decls[local];
                match local.mutability {
                    Mutability::Not => match is_local_mutation_allowed {
                        LocalMutationIsAllowed::Yes => {
                            Ok(RootPlace {
                                place,
                                is_local_mutation_allowed: LocalMutationIsAllowed::Yes
                            })
                        }
                        LocalMutationIsAllowed::ExceptUpvars => {
                            Ok(RootPlace {
                                place,
                                is_local_mutation_allowed: LocalMutationIsAllowed::ExceptUpvars
                            })
                        }
                        LocalMutationIsAllowed::No => Err(place),
                    },
                    Mutability::Mut => Ok(RootPlace { place, is_local_mutation_allowed }),
                }
            }
            Place::Static(ref static_) =>
                if self.tcx.is_static(static_.def_id) != Some(hir::Mutability::MutMutable) {
                    Err(place)
                } else {
                    Ok(RootPlace { place, is_local_mutation_allowed })
                },
            Place::Projection(ref proj) => {
                match proj.elem {
                    ProjectionElem::Deref => {
                        let base_ty = proj.base.ty(self.mir, self.tcx).to_ty(self.tcx);

                        // Check the kind of deref to decide
                        match base_ty.sty {
                            ty::TyRef(_, _, mutbl) => {
                                match mutbl {
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
                                    // `*mut` raw pointers are always mutable, regardless of
                                    // context. The users have to check by themselves.
                                    hir::MutMutable => {
                                        return Ok(RootPlace { place, is_local_mutation_allowed });
                                    }
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
                                    // Subtle: this is an upvar
                                    // reference, so it looks like
                                    // `self.foo` -- we want to double
                                    // check that the context `*self`
                                    // is mutable (i.e., this is not a
                                    // `Fn` closure).  But if that
                                    // check succeeds, we want to
                                    // *blame* the mutability on
                                    // `place` (that is,
                                    // `self.foo`). This is used to
                                    // propagate the info about
                                    // whether mutability declarations
                                    // are used outwards, so that we register
                                    // the outer variable as mutable. Otherwise a
                                    // test like this fails to record the `mut`
                                    // as needed:
                                    //
                                    // ```
                                    // fn foo<F: FnOnce()>(_f: F) { }
                                    // fn main() {
                                    //     let var = Vec::new();
                                    //     foo(move || {
                                    //         var.push(1);
                                    //     });
                                    // }
                                    // ```
                                    let _ = self.is_mutable(&proj.base, is_local_mutation_allowed)?;
                                    Ok(RootPlace { place, is_local_mutation_allowed })
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
    ReadForMatch,
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

