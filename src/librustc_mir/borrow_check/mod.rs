//! This query borrow-checks the MIR to (further) ensure it is not broken.

use crate::borrow_check::nll::region_infer::RegionInferenceContext;
use rustc::hir::{self, HirId};
use rustc::hir::Node;
use rustc::hir::def_id::DefId;
use rustc::infer::InferCtxt;
use rustc::lint::builtin::UNUSED_MUT;
use rustc::lint::builtin::{MUTABLE_BORROW_RESERVATION_CONFLICT};
use rustc::middle::borrowck::SignalledError;
use rustc::mir::{AggregateKind, BasicBlock, BorrowCheckResult, BorrowKind};
use rustc::mir::{
    ClearCrossCrate, Local, Location, Body, Mutability, Operand, Place, PlaceBase, Static,

    StaticKind
};
use rustc::mir::{Field, Projection, ProjectionElem, Rvalue, Statement, StatementKind};
use rustc::mir::{Terminator, TerminatorKind};
use rustc::ty::query::Providers;
use rustc::ty::{self, TyCtxt};

use rustc_errors::{Applicability, Diagnostic, DiagnosticBuilder, Level};
use rustc_data_structures::bit_set::BitSet;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::graph::dominators::Dominators;
use smallvec::SmallVec;

use std::collections::BTreeMap;
use std::mem;
use std::rc::Rc;

use syntax::ast::Name;
use syntax_pos::{Span, DUMMY_SP};

use crate::dataflow::indexes::{BorrowIndex, InitIndex, MoveOutIndex, MovePathIndex};
use crate::dataflow::move_paths::{HasMoveData, InitLocation, LookupResult, MoveData, MoveError};
use crate::dataflow::Borrows;
use crate::dataflow::DataflowResultsConsumer;
use crate::dataflow::FlowAtLocation;
use crate::dataflow::MoveDataParamEnv;
use crate::dataflow::{do_dataflow, DebugFormatted};
use crate::dataflow::EverInitializedPlaces;
use crate::dataflow::{MaybeInitializedPlaces, MaybeUninitializedPlaces};
use crate::util::borrowck_errors::{BorrowckErrors, Origin};

use self::borrow_set::{BorrowData, BorrowSet};
use self::flows::Flows;
use self::location::LocationTable;
use self::prefixes::PrefixSet;
use self::MutateMode::{JustWrite, WriteAndRead};
use self::mutability_errors::AccessKind;

use self::path_utils::*;

crate mod borrow_set;
mod error_reporting;
mod flows;
mod location;
mod conflict_errors;
mod move_errors;
mod mutability_errors;
mod path_utils;
crate mod place_ext;
crate mod places_conflict;
mod prefixes;
mod used_muts;

pub(crate) mod nll;

// FIXME(eddyb) perhaps move this somewhere more centrally.
#[derive(Debug)]
crate struct Upvar {
    name: Name,

    var_hir_id: HirId,

    /// If true, the capture is behind a reference.
    by_ref: bool,

    mutability: Mutability,
}

pub fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        mir_borrowck,
        ..*providers
    };
}

fn mir_borrowck(tcx: TyCtxt<'_>, def_id: DefId) -> BorrowCheckResult<'_> {
    let input_body = tcx.mir_validated(def_id);
    debug!("run query mir_borrowck: {}", tcx.def_path_str(def_id));

    let opt_closure_req = tcx.infer_ctxt().enter(|infcx| {
        let input_body: &Body<'_> = &input_body.borrow();
        do_mir_borrowck(&infcx, input_body, def_id)
    });
    debug!("mir_borrowck done");

    opt_closure_req
}

fn do_mir_borrowck<'a, 'tcx>(
    infcx: &InferCtxt<'a, 'tcx>,
    input_body: &Body<'tcx>,
    def_id: DefId,
) -> BorrowCheckResult<'tcx> {
    debug!("do_mir_borrowck(def_id = {:?})", def_id);

    let tcx = infcx.tcx;
    let attributes = tcx.get_attrs(def_id);
    let param_env = tcx.param_env(def_id);
    let id = tcx
        .hir()
        .as_local_hir_id(def_id)
        .expect("do_mir_borrowck: non-local DefId");

    // Gather the upvars of a closure, if any.
    let tables = tcx.typeck_tables_of(def_id);
    let upvars: Vec<_> = tables
        .upvar_list
        .get(&def_id)
        .into_iter()
        .flat_map(|v| v.values())
        .map(|upvar_id| {
            let var_hir_id = upvar_id.var_path.hir_id;
            let capture = tables.upvar_capture(*upvar_id);
            let by_ref = match capture {
                ty::UpvarCapture::ByValue => false,
                ty::UpvarCapture::ByRef(..) => true,
            };
            let mut upvar = Upvar {
                name: tcx.hir().name(var_hir_id),
                var_hir_id,
                by_ref,
                mutability: Mutability::Not,
            };
            let bm = *tables.pat_binding_modes().get(var_hir_id)
                .expect("missing binding mode");
            if bm == ty::BindByValue(hir::MutMutable) {
                upvar.mutability = Mutability::Mut;
            }
            upvar
        })
        .collect();

    // Replace all regions with fresh inference variables. This
    // requires first making our own copy of the MIR. This copy will
    // be modified (in place) to contain non-lexical lifetimes. It
    // will have a lifetime tied to the inference context.
    let mut body: Body<'tcx> = input_body.clone();
    let free_regions = nll::replace_regions_in_mir(infcx, def_id, param_env, &mut body);
    let body = &body; // no further changes
    let location_table = &LocationTable::new(body);

    let mut errors_buffer = Vec::new();
    let (move_data, move_errors): (MoveData<'tcx>, Option<Vec<(Place<'tcx>, MoveError<'tcx>)>>) =
        match MoveData::gather_moves(body, tcx) {
            Ok(move_data) => (move_data, None),
            Err((move_data, move_errors)) => (move_data, Some(move_errors)),
        };

    let mdpe = MoveDataParamEnv {
        move_data: move_data,
        param_env: param_env,
    };

    let dead_unwinds = BitSet::new_empty(body.basic_blocks().len());
    let mut flow_inits = FlowAtLocation::new(do_dataflow(
        tcx,
        body,
        def_id,
        &attributes,
        &dead_unwinds,
        MaybeInitializedPlaces::new(tcx, body, &mdpe),
        |bd, i| DebugFormatted::new(&bd.move_data().move_paths[i]),
    ));

    let locals_are_invalidated_at_exit = tcx.hir().body_owner_kind(id).is_fn_or_closure();
    let borrow_set = Rc::new(BorrowSet::build(
            tcx, body, locals_are_invalidated_at_exit, &mdpe.move_data));

    // If we are in non-lexical mode, compute the non-lexical lifetimes.
    let (regioncx, polonius_output, opt_closure_req) = nll::compute_regions(
        infcx,
        def_id,
        free_regions,
        body,
        &upvars,
        location_table,
        param_env,
        &mut flow_inits,
        &mdpe.move_data,
        &borrow_set,
        &mut errors_buffer,
    );

    // The various `flow_*` structures can be large. We drop `flow_inits` here
    // so it doesn't overlap with the others below. This reduces peak memory
    // usage significantly on some benchmarks.
    drop(flow_inits);

    let regioncx = Rc::new(regioncx);

    let flow_borrows = FlowAtLocation::new(do_dataflow(
        tcx,
        body,
        def_id,
        &attributes,
        &dead_unwinds,
        Borrows::new(tcx, body, regioncx.clone(), &borrow_set),
        |rs, i| DebugFormatted::new(&rs.location(i)),
    ));
    let flow_uninits = FlowAtLocation::new(do_dataflow(
        tcx,
        body,
        def_id,
        &attributes,
        &dead_unwinds,
        MaybeUninitializedPlaces::new(tcx, body, &mdpe),
        |bd, i| DebugFormatted::new(&bd.move_data().move_paths[i]),
    ));
    let flow_ever_inits = FlowAtLocation::new(do_dataflow(
        tcx,
        body,
        def_id,
        &attributes,
        &dead_unwinds,
        EverInitializedPlaces::new(tcx, body, &mdpe),
        |bd, i| DebugFormatted::new(&bd.move_data().inits[i]),
    ));

    let movable_generator = match tcx.hir().get(id) {
        Node::Expr(&hir::Expr {
            node: hir::ExprKind::Closure(.., Some(hir::GeneratorMovability::Static)),
            ..
        }) => false,
        _ => true,
    };

    let dominators = body.dominators();

    let mut mbcx = MirBorrowckCtxt {
        infcx,
        body,
        mir_def_id: def_id,
        move_data: &mdpe.move_data,
        location_table,
        movable_generator,
        locals_are_invalidated_at_exit,
        access_place_error_reported: Default::default(),
        reservation_error_reported: Default::default(),
        reservation_warnings: Default::default(),
        move_error_reported: BTreeMap::new(),
        uninitialized_error_reported: Default::default(),
        errors_buffer,
        nonlexical_regioncx: regioncx,
        used_mut: Default::default(),
        used_mut_upvars: SmallVec::new(),
        borrow_set,
        dominators,
        upvars,
    };

    let mut state = Flows::new(
        flow_borrows,
        flow_uninits,
        flow_ever_inits,
        polonius_output,
    );

    if let Some(errors) = move_errors {
        mbcx.report_move_errors(errors);
    }
    mbcx.analyze_results(&mut state); // entry point for DataflowResultsConsumer

    // Convert any reservation warnings into lints.
    let reservation_warnings = mem::take(&mut mbcx.reservation_warnings);
    for (_, (place, span, location, bk, borrow)) in reservation_warnings {
        let mut initial_diag =
            mbcx.report_conflicting_borrow(location, (&place, span), bk, &borrow);

        let lint_root = if let ClearCrossCrate::Set(ref vsi) = mbcx.body.source_scope_local_data {
            let scope = mbcx.body.source_info(location).scope;
            vsi[scope].lint_root
        } else {
            id
        };

        // Span and message don't matter; we overwrite them below anyway
        let mut diag = mbcx.infcx.tcx.struct_span_lint_hir(
            MUTABLE_BORROW_RESERVATION_CONFLICT, lint_root, DUMMY_SP, "");

        diag.message = initial_diag.styled_message().clone();
        diag.span = initial_diag.span.clone();

        initial_diag.cancel();
        diag.buffer(&mut mbcx.errors_buffer);
    }

    // For each non-user used mutable variable, check if it's been assigned from
    // a user-declared local. If so, then put that local into the used_mut set.
    // Note that this set is expected to be small - only upvars from closures
    // would have a chance of erroneously adding non-user-defined mutable vars
    // to the set.
    let temporary_used_locals: FxHashSet<Local> = mbcx.used_mut.iter()
        .filter(|&local| mbcx.body.local_decls[*local].is_user_variable.is_none())
        .cloned()
        .collect();
    // For the remaining unused locals that are marked as mutable, we avoid linting any that
    // were never initialized. These locals may have been removed as unreachable code; or will be
    // linted as unused variables.
    let unused_mut_locals = mbcx.body.mut_vars_iter()
        .filter(|local| !mbcx.used_mut.contains(local))
        .collect();
    mbcx.gather_used_muts(temporary_used_locals, unused_mut_locals);

    debug!("mbcx.used_mut: {:?}", mbcx.used_mut);
    let used_mut = mbcx.used_mut;
    for local in mbcx.body.mut_vars_and_args_iter().filter(|local| !used_mut.contains(local)) {
        if let ClearCrossCrate::Set(ref vsi) = mbcx.body.source_scope_local_data {
            let local_decl = &mbcx.body.local_decls[local];

            // Skip implicit `self` argument for closures
            if local.index() == 1 && tcx.is_closure(mbcx.mir_def_id) {
                continue;
            }

            // Skip over locals that begin with an underscore or have no name
            match local_decl.name {
                Some(name) => if name.as_str().starts_with("_") {
                    continue;
                },
                None => continue,
            }

            let span = local_decl.source_info.span;
            if span.compiler_desugaring_kind().is_some() {
                // If the `mut` arises as part of a desugaring, we should ignore it.
                continue;
            }

            let mut_span = tcx.sess.source_map().span_until_non_whitespace(span);
            tcx.struct_span_lint_hir(
                UNUSED_MUT,
                vsi[local_decl.source_info.scope].lint_root,
                span,
                "variable does not need to be mutable",
            )
            .span_suggestion_short(
                mut_span,
                "remove this `mut`",
                String::new(),
                Applicability::MachineApplicable,
            )
            .emit();
        }
    }

    // Buffer any move errors that we collected and de-duplicated.
    for (_, (_, diag)) in mbcx.move_error_reported {
        diag.buffer(&mut mbcx.errors_buffer);
    }

    if !mbcx.errors_buffer.is_empty() {
        mbcx.errors_buffer.sort_by_key(|diag| diag.span.primary_span());

        if tcx.migrate_borrowck() {
            // When borrowck=migrate, check if AST-borrowck would
            // error on the given code.

            // rust-lang/rust#55492, rust-lang/rust#58776 check the base def id
            // for errors. AST borrowck is responsible for aggregating
            // `signalled_any_error` from all of the nested closures here.
            let base_def_id = tcx.closure_base_def_id(def_id);

            match tcx.borrowck(base_def_id).signalled_any_error {
                SignalledError::NoErrorsSeen => {
                    // if AST-borrowck signalled no errors, then
                    // downgrade all the buffered MIR-borrowck errors
                    // to warnings.

                    for err in mbcx.errors_buffer.iter_mut() {
                        downgrade_if_error(err);
                    }
                }
                SignalledError::SawSomeError => {
                    // if AST-borrowck signalled a (cancelled) error,
                    // then we will just emit the buffered
                    // MIR-borrowck errors as normal.
                }
            }
        }

        for diag in mbcx.errors_buffer.drain(..) {
            DiagnosticBuilder::new_diagnostic(mbcx.infcx.tcx.sess.diagnostic(), diag).emit();
        }
    }

    let result = BorrowCheckResult {
        closure_requirements: opt_closure_req,
        used_mut_upvars: mbcx.used_mut_upvars,
    };

    debug!("do_mir_borrowck: result = {:#?}", result);

    result
}

fn downgrade_if_error(diag: &mut Diagnostic) {
    if diag.is_error() {
        diag.level = Level::Warning;
        diag.warn(
            "this error has been downgraded to a warning for backwards \
            compatibility with previous releases",
        ).warn(
            "this represents potential undefined behavior in your code and \
            this warning will become a hard error in the future",
        ).note(
            "for more information, try `rustc --explain E0729`"
        );
    }
}

pub struct MirBorrowckCtxt<'cx, 'tcx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,
    body: &'cx Body<'tcx>,
    mir_def_id: DefId,
    move_data: &'cx MoveData<'tcx>,

    /// Map from MIR `Location` to `LocationIndex`; created
    /// when MIR borrowck begins.
    location_table: &'cx LocationTable,

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
    /// errors for corresponding activations.
    //
    // FIXME: ideally this would be a set of `BorrowIndex`, not `Place`s,
    // but it is currently inconvenient to track down the `BorrowIndex`
    // at the time we detect and report a reservation error.
    reservation_error_reported: FxHashSet<Place<'tcx>>,
    /// Migration warnings to be reported for #56254. We delay reporting these
    /// so that we can suppress the warning if there's a corresponding error
    /// for the activation of the borrow.
    reservation_warnings: FxHashMap<
        BorrowIndex,
        (Place<'tcx>, Span, Location, BorrowKind, BorrowData<'tcx>)
    >,
    /// This field keeps track of move errors that are to be reported for given move indicies.
    ///
    /// There are situations where many errors can be reported for a single move out (see #53807)
    /// and we want only the best of those errors.
    ///
    /// The `report_use_of_moved_or_uninitialized` function checks this map and replaces the
    /// diagnostic (if there is one) if the `Place` of the error being reported is a prefix of the
    /// `Place` of the previous most diagnostic. This happens instead of buffering the error. Once
    /// all move errors have been reported, any diagnostics in this map are added to the buffer
    /// to be emitted.
    ///
    /// `BTreeMap` is used to preserve the order of insertions when iterating. This is necessary
    /// when errors in the map are being re-added to the error buffer so that errors with the
    /// same primary span come out in a consistent order.
    move_error_reported: BTreeMap<Vec<MoveOutIndex>, (Place<'tcx>, DiagnosticBuilder<'cx>)>,
    /// This field keeps track of errors reported in the checking of uninitialized variables,
    /// so that we don't report seemingly duplicate errors.
    uninitialized_error_reported: FxHashSet<Place<'tcx>>,
    /// Errors to be reported buffer
    errors_buffer: Vec<Diagnostic>,
    /// This field keeps track of all the local variables that are declared mut and are mutated.
    /// Used for the warning issued by an unused mutable local variable.
    used_mut: FxHashSet<Local>,
    /// If the function we're checking is a closure, then we'll need to report back the list of
    /// mutable upvars that have been used. This field keeps track of them.
    used_mut_upvars: SmallVec<[Field; 8]>,
    /// Non-lexical region inference context, if NLL is enabled. This
    /// contains the results from region inference and lets us e.g.
    /// find out which CFG points are contained in each borrow region.
    nonlexical_regioncx: Rc<RegionInferenceContext<'tcx>>,

    /// The set of borrows extracted from the MIR
    borrow_set: Rc<BorrowSet<'tcx>>,

    /// Dominators for MIR
    dominators: Dominators<BasicBlock>,

    /// Information about upvars not necessarily preserved in types or MIR
    upvars: Vec<Upvar>,
}

// Check that:
// 1. assignments are always made to mutable locations (FIXME: does that still really go here?)
// 2. loans made in overlapping scopes do not conflict
// 3. assignments do not affect things loaned out as immutable
// 4. moves do not affect things loaned out in any way
impl<'cx, 'tcx> DataflowResultsConsumer<'cx, 'tcx> for MirBorrowckCtxt<'cx, 'tcx> {
    type FlowState = Flows<'cx, 'tcx>;

    fn body(&self) -> &'cx Body<'tcx> {
        self.body
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
                    location,
                    (rhs, span),
                    flow_state,
                );

                self.mutate_place(
                    location,
                    (lhs, span),
                    Shallow(None),
                    JustWrite,
                    flow_state,
                );
            }
            StatementKind::FakeRead(_, ref place) => {
                // Read for match doesn't access any memory and is used to
                // assert that a place is safe and live. So we don't have to
                // do any checks here.
                //
                // FIXME: Remove check that the place is initialized. This is
                // needed for now because matches don't have never patterns yet.
                // So this is the only place we prevent
                //      let x: !;
                //      match x {};
                // from compiling.
                self.check_if_path_or_subpath_is_moved(
                    location,
                    InitializationRequiringAction::Use,
                    (place, span),
                    flow_state,
                );
            }
            StatementKind::SetDiscriminant {
                ref place,
                variant_index: _,
            } => {
                self.mutate_place(
                    location,
                    (place, span),
                    Shallow(None),
                    JustWrite,
                    flow_state,
                );
            }
            StatementKind::InlineAsm(ref asm) => {
                for (o, output) in asm.asm.outputs.iter().zip(asm.outputs.iter()) {
                    if o.is_indirect {
                        // FIXME(eddyb) indirect inline asm outputs should
                        // be encoded through MIR place derefs instead.
                        self.access_place(
                            location,
                            (output, o.span),
                            (Deep, Read(ReadKind::Copy)),
                            LocalMutationIsAllowed::No,
                            flow_state,
                        );
                        self.check_if_path_or_subpath_is_moved(
                            location,
                            InitializationRequiringAction::Use,
                            (output, o.span),
                            flow_state,
                        );
                    } else {
                        self.mutate_place(
                            location,
                            (output, o.span),
                            if o.is_rw { Deep } else { Shallow(None) },
                            if o.is_rw { WriteAndRead } else { JustWrite },
                            flow_state,
                        );
                    }
                }
                for (_, input) in asm.inputs.iter() {
                    self.consume_operand(location, (input, span), flow_state);
                }
            }
            StatementKind::Nop
            | StatementKind::AscribeUserType(..)
            | StatementKind::Retag { .. }
            | StatementKind::StorageLive(..) => {
                // `Nop`, `AscribeUserType`, `Retag`, and `StorageLive` are irrelevant
                // to borrow check.
            }
            StatementKind::StorageDead(local) => {
                self.access_place(
                    location,
                    (&Place::from(local), span),
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
                self.consume_operand(loc, (discr, span), flow_state);
            }
            TerminatorKind::Drop {
                location: ref drop_place,
                target: _,
                unwind: _,
            } => {
                let gcx = self.infcx.tcx.global_tcx();

                // Compute the type with accurate region information.
                let drop_place_ty = drop_place.ty(self.body, self.infcx.tcx);

                // Erase the regions.
                let drop_place_ty = self.infcx.tcx.erase_regions(&drop_place_ty).ty;

                // "Lift" into the gcx -- once regions are erased, this type should be in the
                // global arenas; this "lift" operation basically just asserts that is true, but
                // that is useful later.
                gcx.lift_to_global(&drop_place_ty).unwrap();

                debug!("visit_terminator_drop \
                        loc: {:?} term: {:?} drop_place: {:?} drop_place_ty: {:?} span: {:?}",
                       loc, term, drop_place, drop_place_ty, span);

                self.access_place(
                    loc,
                    (drop_place, span),
                    (AccessDepth::Drop, Write(WriteKind::StorageDeadOrDrop)),
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
                    loc,
                    (drop_place, span),
                    Deep,
                    JustWrite,
                    flow_state,
                );
                self.consume_operand(
                    loc,
                    (new_value, span),
                    flow_state,
                );
            }
            TerminatorKind::Call {
                ref func,
                ref args,
                ref destination,
                cleanup: _,
                from_hir_call: _,
            } => {
                self.consume_operand(loc, (func, span), flow_state);
                for arg in args {
                    self.consume_operand(
                        loc,
                        (arg, span),
                        flow_state,
                    );
                }
                if let Some((ref dest, _ /*bb*/)) = *destination {
                    self.mutate_place(
                        loc,
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
                self.consume_operand(loc, (cond, span), flow_state);
                use rustc::mir::interpret::InterpError::BoundsCheck;
                if let BoundsCheck { ref len, ref index } = *msg {
                    self.consume_operand(loc, (len, span), flow_state);
                    self.consume_operand(loc, (index, span), flow_state);
                }
            }

            TerminatorKind::Yield {
                ref value,
                resume: _,
                drop: _,
            } => {
                self.consume_operand(loc, (value, span), flow_state);

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
                        self.check_for_invalidation_at_exit(loc, borrow, span);
                    }
                });
            }
            TerminatorKind::Goto { target: _ }
            | TerminatorKind::Abort
            | TerminatorKind::Unreachable
            | TerminatorKind::FalseEdges {
                real_target: _,
                imaginary_target: _,
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

use self::ReadOrWrite::{Activation, Read, Reservation, Write};
use self::AccessDepth::{Deep, Shallow};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum ArtificialField {
    ArrayLength,
    ShallowBorrow,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum AccessDepth {
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

    /// Access is Deep only when there is a Drop implementation that
    /// can reach the data behind the reference.
    Drop,
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
//
// FIXME: @nikomatsakis suggested that this flag could be removed with the following modifications:
// - Merge `check_access_permissions()` and `check_if_reassignment_to_immutable_state()`.
// - Split `is_mutable()` into `is_assignable()` (can be directly assigned) and
//   `is_declared_mutable()`.
// - Take flow state into consideration in `is_assignable()` for local variables.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum LocalMutationIsAllowed {
    Yes,
    /// We want use of immutable upvars to cause a "write to immutable upvar"
    /// error, not an "reassignment" error.
    ExceptUpvars,
    No,
}

#[derive(Copy, Clone, Debug)]
enum InitializationRequiringAction {
    Update,
    Borrow,
    MatchOn,
    Use,
    Assignment,
    PartialAssignment,
}

struct RootPlace<'d, 'tcx> {
    place: &'d Place<'tcx>,
    is_local_mutation_allowed: LocalMutationIsAllowed,
}

impl InitializationRequiringAction {
    fn as_noun(self) -> &'static str {
        match self {
            InitializationRequiringAction::Update => "update",
            InitializationRequiringAction::Borrow => "borrow",
            InitializationRequiringAction::MatchOn => "use", // no good noun
            InitializationRequiringAction::Use => "use",
            InitializationRequiringAction::Assignment => "assign",
            InitializationRequiringAction::PartialAssignment => "assign to part",
        }
    }

    fn as_verb_in_past_tense(self) -> &'static str {
        match self {
            InitializationRequiringAction::Update => "updated",
            InitializationRequiringAction::Borrow => "borrowed",
            InitializationRequiringAction::MatchOn => "matched on",
            InitializationRequiringAction::Use => "used",
            InitializationRequiringAction::Assignment => "assigned",
            InitializationRequiringAction::PartialAssignment => "partially assigned",
        }
    }
}

impl<'cx, 'tcx> MirBorrowckCtxt<'cx, 'tcx> {
    /// Checks an access to the given place to see if it is allowed. Examines the set of borrows
    /// that are in scope, as well as which paths have been initialized, to ensure that (a) the
    /// place is initialized and (b) it is not borrowed in some way that would prevent this
    /// access.
    ///
    /// Returns `true` if an error is reported.
    fn access_place(
        &mut self,
        location: Location,
        place_span: (&Place<'tcx>, Span),
        kind: (AccessDepth, ReadOrWrite),
        is_local_mutation_allowed: LocalMutationIsAllowed,
        flow_state: &Flows<'cx, 'tcx>,
    ) {
        let (sd, rw) = kind;

        if let Activation(_, borrow_index) = rw {
            if self.reservation_error_reported.contains(&place_span.0) {
                debug!(
                    "skipping access_place for activation of invalid reservation \
                     place: {:?} borrow_index: {:?}",
                    place_span.0, borrow_index
                );
                return;
            }
        }

        // Check is_empty() first because it's the common case, and doing that
        // way we avoid the clone() call.
        if !self.access_place_error_reported.is_empty() &&
           self
            .access_place_error_reported
            .contains(&(place_span.0.clone(), place_span.1))
        {
            debug!(
                "access_place: suppressing error place_span=`{:?}` kind=`{:?}`",
                place_span, kind
            );
            return;
        }

        let mutability_error =
            self.check_access_permissions(
                place_span,
                rw,
                is_local_mutation_allowed,
                flow_state,
                location,
            );
        let conflict_error =
            self.check_access_for_conflict(location, place_span, sd, rw, flow_state);

        if let (Activation(_, borrow_idx), true) = (kind.1, conflict_error) {
            // Suppress this warning when there's an error being emited for the
            // same borrow: fixing the error is likely to fix the warning.
            self.reservation_warnings.remove(&borrow_idx);
        }

        if conflict_error || mutability_error {
            debug!(
                "access_place: logging error place_span=`{:?}` kind=`{:?}`",
                place_span, kind
            );

            self.access_place_error_reported
                .insert((place_span.0.clone(), place_span.1));
        }
    }

    fn check_access_for_conflict(
        &mut self,
        location: Location,
        place_span: (&Place<'tcx>, Span),
        sd: AccessDepth,
        rw: ReadOrWrite,
        flow_state: &Flows<'cx, 'tcx>,
    ) -> bool {
        debug!(
            "check_access_for_conflict(location={:?}, place_span={:?}, sd={:?}, rw={:?})",
            location, place_span, sd, rw,
        );

        let mut error_reported = false;
        let tcx = self.infcx.tcx;
        let body = self.body;
        let location_table = self.location_table.start_index(location);
        let borrow_set = self.borrow_set.clone();
        each_borrow_involving_path(
            self,
            tcx,
            body,
            location,
            (sd, place_span.0),
            &borrow_set,
            flow_state.borrows_in_scope(location_table),
            |this, borrow_index, borrow| match (rw, borrow.kind) {
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

                (Read(_), BorrowKind::Shared)
                | (Read(_), BorrowKind::Shallow)
                | (Read(ReadKind::Borrow(BorrowKind::Shallow)), BorrowKind::Unique)
                | (Read(ReadKind::Borrow(BorrowKind::Shallow)), BorrowKind::Mut { .. }) => {
                    Control::Continue
                }

                (Write(WriteKind::Move), BorrowKind::Shallow) => {
                    // Handled by initialization checks.
                    Control::Continue
                }

                (Read(kind), BorrowKind::Unique) | (Read(kind), BorrowKind::Mut { .. }) => {
                    // Reading from mere reservations of mutable-borrows is OK.
                    if !is_active(&this.dominators, borrow, location) {
                        assert!(allow_two_phase_borrow(borrow.kind));
                        return Control::Continue;
                    }

                    error_reported = true;
                    match kind {
                        ReadKind::Copy  => {
                            this.report_use_while_mutably_borrowed(location, place_span, borrow)
                                .buffer(&mut this.errors_buffer);
                        }
                        ReadKind::Borrow(bk) => {
                            this.report_conflicting_borrow(location, place_span, bk, borrow)
                                .buffer(&mut this.errors_buffer);
                        }
                    }
                    Control::Break
                }

                (Reservation(WriteKind::MutableBorrow(bk)), BorrowKind::Shallow)
                | (Reservation(WriteKind::MutableBorrow(bk)), BorrowKind::Shared) if {
                    tcx.migrate_borrowck() && this.borrow_set.location_map.contains_key(&location)
                } => {
                    let bi = this.borrow_set.location_map[&location];
                    debug!(
                        "recording invalid reservation of place: {:?} with \
                         borrow index {:?} as warning",
                        place_span.0,
                        bi,
                    );
                    // rust-lang/rust#56254 - This was previously permitted on
                    // the 2018 edition so we emit it as a warning. We buffer
                    // these sepately so that we only emit a warning if borrow
                    // checking was otherwise successful.
                    this.reservation_warnings.insert(
                        bi,
                        (place_span.0.clone(), place_span.1, location, bk, borrow.clone()),
                    );

                    // Don't suppress actual errors.
                    Control::Continue
                }

                (Reservation(kind), _)
                | (Activation(kind, _), _)
                | (Write(kind), _) => {
                    match rw {
                        Reservation(..) => {
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

                    error_reported = true;
                    match kind {
                        WriteKind::MutableBorrow(bk) => {
                            this.report_conflicting_borrow(location, place_span, bk, borrow)
                                .buffer(&mut this.errors_buffer);
                        }
                        WriteKind::StorageDeadOrDrop => {
                            this.report_borrowed_value_does_not_live_long_enough(
                                location,
                                borrow,
                                place_span,
                                Some(kind))
                        }
                        WriteKind::Mutate => {
                            this.report_illegal_mutation_of_borrowed(location, place_span, borrow)
                        }
                        WriteKind::Move => {
                            this.report_move_out_while_borrowed(location, place_span, borrow)
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
        location: Location,
        place_span: (&Place<'tcx>, Span),
        kind: AccessDepth,
        mode: MutateMode,
        flow_state: &Flows<'cx, 'tcx>,
    ) {
        // Write of P[i] or *P, or WriteAndRead of any P, requires P init'd.
        match mode {
            MutateMode::WriteAndRead => {
                self.check_if_path_or_subpath_is_moved(
                    location,
                    InitializationRequiringAction::Update,
                    place_span,
                    flow_state,
                );
            }
            MutateMode::JustWrite => {
                self.check_if_assigned_path_is_moved(location, place_span, flow_state);
            }
        }

        // Special case: you can assign a immutable local variable
        // (e.g., `x = ...`) so long as it has never been initialized
        // before (at this point in the flow).
        if let &Place::Base(PlaceBase::Local(local)) = place_span.0 {
            if let Mutability::Not = self.body.local_decls[local].mutability {
                // check for reassignments to immutable local variables
                self.check_if_reassignment_to_immutable_state(
                    location,
                    local,
                    place_span,
                    flow_state,
                );
                return;
            }
        }

        // Otherwise, use the normal access permission rules.
        self.access_place(
            location,
            place_span,
            (kind, Write(WriteKind::Mutate)),
            LocalMutationIsAllowed::No,
            flow_state,
        );
    }

    fn consume_rvalue(
        &mut self,
        location: Location,
        (rvalue, span): (&Rvalue<'tcx>, Span),
        flow_state: &Flows<'cx, 'tcx>,
    ) {
        match *rvalue {
            Rvalue::Ref(_ /*rgn*/, bk, ref place) => {
                let access_kind = match bk {
                    BorrowKind::Shallow => {
                        (Shallow(Some(ArtificialField::ShallowBorrow)), Read(ReadKind::Borrow(bk)))
                    },
                    BorrowKind::Shared => (Deep, Read(ReadKind::Borrow(bk))),
                    BorrowKind::Unique | BorrowKind::Mut { .. } => {
                        let wk = WriteKind::MutableBorrow(bk);
                        if allow_two_phase_borrow(bk) {
                            (Deep, Reservation(wk))
                        } else {
                            (Deep, Write(wk))
                        }
                    }
                };

                self.access_place(
                    location,
                    (place, span),
                    access_kind,
                    LocalMutationIsAllowed::No,
                    flow_state,
                );

                let action = if bk == BorrowKind::Shallow {
                    InitializationRequiringAction::MatchOn
                } else {
                    InitializationRequiringAction::Borrow
                };

                self.check_if_path_or_subpath_is_moved(
                    location,
                    action,
                    (place, span),
                    flow_state,
                );
            }

            Rvalue::Use(ref operand)
            | Rvalue::Repeat(ref operand, _)
            | Rvalue::UnaryOp(_ /*un_op*/, ref operand)
            | Rvalue::Cast(_ /*cast_kind*/, ref operand, _ /*ty*/) => {
                self.consume_operand(location, (operand, span), flow_state)
            }

            Rvalue::Len(ref place) | Rvalue::Discriminant(ref place) => {
                let af = match *rvalue {
                    Rvalue::Len(..) => Some(ArtificialField::ArrayLength),
                    Rvalue::Discriminant(..) => None,
                    _ => unreachable!(),
                };
                self.access_place(
                    location,
                    (place, span),
                    (Shallow(af), Read(ReadKind::Copy)),
                    LocalMutationIsAllowed::No,
                    flow_state,
                );
                self.check_if_path_or_subpath_is_moved(
                    location,
                    InitializationRequiringAction::Use,
                    (place, span),
                    flow_state,
                );
            }

            Rvalue::BinaryOp(_bin_op, ref operand1, ref operand2)
            | Rvalue::CheckedBinaryOp(_bin_op, ref operand1, ref operand2) => {
                self.consume_operand(location, (operand1, span), flow_state);
                self.consume_operand(location, (operand2, span), flow_state);
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
                match **aggregate_kind {
                    AggregateKind::Closure(def_id, _)
                    | AggregateKind::Generator(def_id, _, _) => {
                        let BorrowCheckResult {
                            used_mut_upvars, ..
                        } = self.infcx.tcx.mir_borrowck(def_id);
                        debug!("{:?} used_mut_upvars={:?}", def_id, used_mut_upvars);
                        for field in used_mut_upvars {
                            self.propagate_closure_used_mut_upvar(&operands[field.index()]);
                        }
                    }
                    AggregateKind::Adt(..)
                    | AggregateKind::Array(..)
                    | AggregateKind::Tuple { .. } => (),
                }

                for operand in operands {
                    self.consume_operand(location, (operand, span), flow_state);
                }
            }
        }
    }

    fn propagate_closure_used_mut_upvar(&mut self, operand: &Operand<'tcx>) {
        let propagate_closure_used_mut_place = |this: &mut Self, place: &Place<'tcx>| {
            match *place {
                Place::Projection { .. } => {
                    if let Some(field) = this.is_upvar_field_projection(place) {
                        this.used_mut_upvars.push(field);
                    }
                }
                Place::Base(PlaceBase::Local(local)) => {
                    this.used_mut.insert(local);
                }
                Place::Base(PlaceBase::Static(_)) => {}
            }
        };

        // This relies on the current way that by-value
        // captures of a closure are copied/moved directly
        // when generating MIR.
        match *operand {
            Operand::Move(Place::Base(PlaceBase::Local(local)))
            | Operand::Copy(Place::Base(PlaceBase::Local(local)))
                if self.body.local_decls[local].is_user_variable.is_none() =>
            {
                if self.body.local_decls[local].ty.is_mutable_pointer() {
                    // The variable will be marked as mutable by the borrow.
                    return;
                }
                // This is an edge case where we have a `move` closure
                // inside a non-move closure, and the inner closure
                // contains a mutation:
                //
                // let mut i = 0;
                // || { move || { i += 1; }; };
                //
                // In this case our usual strategy of assuming that the
                // variable will be captured by mutable reference is
                // wrong, since `i` can be copied into the inner
                // closure from a shared reference.
                //
                // As such we have to search for the local that this
                // capture comes from and mark it as being used as mut.

                let temp_mpi = self.move_data.rev_lookup.find_local(local);
                let init = if let [init_index] = *self.move_data.init_path_map[temp_mpi] {
                    &self.move_data.inits[init_index]
                } else {
                    bug!("temporary should be initialized exactly once")
                };

                let loc = match init.location {
                    InitLocation::Statement(stmt) => stmt,
                    _ => bug!("temporary initialized in arguments"),
                };

                let bbd = &self.body[loc.block];
                let stmt = &bbd.statements[loc.statement_index];
                debug!("temporary assigned in: stmt={:?}", stmt);

                if let StatementKind::Assign(_, box Rvalue::Ref(_, _, ref source)) = stmt.kind {
                    propagate_closure_used_mut_place(self, source);
                } else {
                    bug!("closures should only capture user variables \
                        or references to user variables");
                }
            }
            Operand::Move(ref place)
            | Operand::Copy(ref place) => {
                propagate_closure_used_mut_place(self, place);
            }
            Operand::Constant(..) => {}
        }
    }

    fn consume_operand(
        &mut self,
        location: Location,
        (operand, span): (&Operand<'tcx>, Span),
        flow_state: &Flows<'cx, 'tcx>,
    ) {
        match *operand {
            Operand::Copy(ref place) => {
                // copy of place: check if this is "copy of frozen path"
                // (FIXME: see check_loans.rs)
                self.access_place(
                    location,
                    (place, span),
                    (Deep, Read(ReadKind::Copy)),
                    LocalMutationIsAllowed::No,
                    flow_state,
                );

                // Finally, check if path was already moved.
                self.check_if_path_or_subpath_is_moved(
                    location,
                    InitializationRequiringAction::Use,
                    (place, span),
                    flow_state,
                );
            }
            Operand::Move(ref place) => {
                // move of place: check if this is move of already borrowed path
                self.access_place(
                    location,
                    (place, span),
                    (Deep, Write(WriteKind::Move)),
                    LocalMutationIsAllowed::Yes,
                    flow_state,
                );

                // Finally, check if path was already moved.
                self.check_if_path_or_subpath_is_moved(
                    location,
                    InitializationRequiringAction::Use,
                    (place, span),
                    flow_state,
                );
            }
            Operand::Constant(_) => {}
        }
    }

    /// Checks whether a borrow of this place is invalidated when the function
    /// exits
    fn check_for_invalidation_at_exit(
        &mut self,
        location: Location,
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
            Place::Base(PlaceBase::Static(box Static{ kind: StaticKind::Promoted(_), .. })) => {
                (true, false)
            }
            Place::Base(PlaceBase::Static(box Static{ kind: StaticKind::Static(_), .. })) => {
                // Thread-locals might be dropped after the function exits, but
                // "true" statics will never be.
                (true, self.is_place_thread_local(&root_place))
            }
            Place::Base(PlaceBase::Local(_)) => {
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

        let sd = if might_be_alive { Deep } else { Shallow(None) };

        if places_conflict::borrow_conflicts_with_place(
            self.infcx.tcx,
            self.body,
            place,
            borrow.kind,
            root_place,
            sd,
            places_conflict::PlaceConflictBias::Overlap,
        ) {
            debug!("check_for_invalidation_at_exit({:?}): INVALID", place);
            // FIXME: should be talking about the region lifetime instead
            // of just a span here.
            let span = self.infcx.tcx.sess.source_map().end_point(span);
            self.report_borrowed_value_does_not_live_long_enough(
                location,
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
            let err = self.infcx.tcx
                .cannot_borrow_across_generator_yield(
                    self.retrieve_borrow_spans(borrow).var_or_use(),
                    yield_span,
                    Origin::Mir,
                );

            err.buffer(&mut self.errors_buffer);
        }
    }

    fn check_activations(&mut self, location: Location, span: Span, flow_state: &Flows<'cx, 'tcx>) {
        // Two-phase borrow support: For each activation that is newly
        // generated at this statement, check if it interferes with
        // another borrow.
        let borrow_set = self.borrow_set.clone();
        for &borrow_index in borrow_set.activations_at_location(location) {
            let borrow = &borrow_set[borrow_index];

            // only mutable borrows should be 2-phase
            assert!(match borrow.kind {
                BorrowKind::Shared | BorrowKind::Shallow => false,
                BorrowKind::Unique | BorrowKind::Mut { .. } => true,
            });

            self.access_place(
                location,
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

impl<'cx, 'tcx> MirBorrowckCtxt<'cx, 'tcx> {
    fn check_if_reassignment_to_immutable_state(
        &mut self,
        location: Location,
        local: Local,
        place_span: (&Place<'tcx>, Span),
        flow_state: &Flows<'cx, 'tcx>,
    ) {
        debug!("check_if_reassignment_to_immutable_state({:?})", local);

        // Check if any of the initializiations of `local` have happened yet:
        if let Some(init_index) = self.is_local_ever_initialized(local, flow_state) {
            // And, if so, report an error.
            let init = &self.move_data.inits[init_index];
            let span = init.span(&self.body);
            self.report_illegal_reassignment(
                location, place_span, span, place_span.0
            );
        }
    }

    fn check_if_full_path_is_moved(
        &mut self,
        location: Location,
        desired_action: InitializationRequiringAction,
        place_span: (&Place<'tcx>, Span),
        flow_state: &Flows<'cx, 'tcx>,
    ) {
        let maybe_uninits = &flow_state.uninits;

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

        debug!("check_if_full_path_is_moved place: {:?}", place_span.0);
        match self.move_path_closest_to(place_span.0) {
            Ok((prefix, mpi)) => {
                if maybe_uninits.contains(mpi) {
                    self.report_use_of_moved_or_uninitialized(
                        location,
                        desired_action,
                        (prefix, place_span.0, place_span.1),
                        mpi,
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
              // (I.e., querying parents breaks scenario 7; but may want
              // to do such a query based on partial-init feature-gate.)
        }
    }

    fn check_if_path_or_subpath_is_moved(
        &mut self,
        location: Location,
        desired_action: InitializationRequiringAction,
        place_span: (&Place<'tcx>, Span),
        flow_state: &Flows<'cx, 'tcx>,
    ) {
        let maybe_uninits = &flow_state.uninits;

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

        self.check_if_full_path_is_moved(location, desired_action, place_span, flow_state);

        // A move of any shallow suffix of `place` also interferes
        // with an attempt to use `place`. This is scenario 3 above.
        //
        // (Distinct from handling of scenarios 1+2+4 above because
        // `place` does not interfere with suffixes of its prefixes,
        // e.g., `a.b.c` does not interfere with `a.b.d`)
        //
        // This code covers scenario 1.

        debug!("check_if_path_or_subpath_is_moved place: {:?}", place_span.0);
        if let Some(mpi) = self.move_path_for_place(place_span.0) {
            if let Some(child_mpi) = maybe_uninits.has_any_child_of(mpi) {
                self.report_use_of_moved_or_uninitialized(
                    location,
                    desired_action,
                    (place_span.0, place_span.0, place_span.1),
                    child_mpi,
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
    fn move_path_closest_to<'a>(
        &mut self,
        place: &'a Place<'tcx>,
    ) -> Result<(&'a Place<'tcx>, MovePathIndex), NoMovePathFound> where 'cx: 'a {
        let mut last_prefix = place;
        for prefix in self.prefixes(place, PrefixSet::All) {
            if let Some(mpi) = self.move_path_for_place(prefix) {
                return Ok((prefix, mpi));
            }
            last_prefix = prefix;
        }
        match *last_prefix {
            Place::Base(PlaceBase::Local(_)) => panic!("should have move path for every Local"),
            Place::Projection(_) => panic!("PrefixSet::All meant don't stop for Projection"),
            Place::Base(PlaceBase::Static(_)) => Err(NoMovePathFound::ReachedStatic),
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
        location: Location,
        (place, span): (&Place<'tcx>, Span),
        flow_state: &Flows<'cx, 'tcx>,
    ) {
        debug!("check_if_assigned_path_is_moved place: {:?}", place);
        // recur down place; dispatch to external checks when necessary
        let mut place = place;
        loop {
            match *place {
                Place::Base(PlaceBase::Local(_)) | Place::Base(PlaceBase::Static(_)) => {
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
                                location, InitializationRequiringAction::Use,
                                (base, span), flow_state);
                            // (base initialized; no need to
                            // recur further)
                            break;
                        }

                        ProjectionElem::Subslice { .. } => {
                            panic!("we don't allow assignments to subslices, location: {:?}",
                                   location);
                        }

                        ProjectionElem::Field(..) => {
                            // if type of `P` has a dtor, then
                            // assigning to `P.f` requires `P` itself
                            // be already initialized
                            let tcx = self.infcx.tcx;
                            match base.ty(self.body, tcx).ty.sty {
                                ty::Adt(def, _) if def.has_dtor(tcx) => {
                                    self.check_if_path_or_subpath_is_moved(
                                        location, InitializationRequiringAction::Assignment,
                                        (base, span), flow_state);

                                    // (base initialized; no need to
                                    // recur further)
                                    break;
                                }


                                // Once `let s; s.x = V; read(s.x);`,
                                // is allowed, remove this match arm.
                                ty::Adt(..) | ty::Tuple(..) => {
                                    check_parent_of_field(self, location, base, span, flow_state);

                                    if let Some(local) = place.base_local() {
                                        // rust-lang/rust#21232,
                                        // #54499, #54986: during
                                        // period where we reject
                                        // partial initialization, do
                                        // not complain about
                                        // unnecessary `mut` on an
                                        // attempt to do a partial
                                        // initialization.
                                        self.used_mut.insert(local);
                                    }
                                }

                                _ => {}
                            }
                        }
                    }

                    place = base;
                }
            }
        }

        fn check_parent_of_field<'cx, 'tcx>(
            this: &mut MirBorrowckCtxt<'cx, 'tcx>,
            location: Location,
            base: &Place<'tcx>,
            span: Span,
            flow_state: &Flows<'cx, 'tcx>,
        ) {
            // rust-lang/rust#21232: Until Rust allows reads from the
            // initialized parts of partially initialized structs, we
            // will, starting with the 2018 edition, reject attempts
            // to write to structs that are not fully initialized.
            //
            // In other words, *until* we allow this:
            //
            // 1. `let mut s; s.x = Val; read(s.x);`
            //
            // we will for now disallow this:
            //
            // 2. `let mut s; s.x = Val;`
            //
            // and also this:
            //
            // 3. `let mut s = ...; drop(s); s.x=Val;`
            //
            // This does not use check_if_path_or_subpath_is_moved,
            // because we want to *allow* reinitializations of fields:
            // e.g., want to allow
            //
            // `let mut s = ...; drop(s.x); s.x=Val;`
            //
            // This does not use check_if_full_path_is_moved on
            // `base`, because that would report an error about the
            // `base` as a whole, but in this scenario we *really*
            // want to report an error about the actual thing that was
            // moved, which may be some prefix of `base`.

            // Shallow so that we'll stop at any dereference; we'll
            // report errors about issues with such bases elsewhere.
            let maybe_uninits = &flow_state.uninits;

            // Find the shortest uninitialized prefix you can reach
            // without going over a Deref.
            let mut shortest_uninit_seen = None;
            for prefix in this.prefixes(base, PrefixSet::Shallow) {
                let mpi = match this.move_path_for_place(prefix) {
                    Some(mpi) => mpi, None => continue,
                };

                if maybe_uninits.contains(mpi) {
                    debug!("check_parent_of_field updating shortest_uninit_seen from {:?} to {:?}",
                           shortest_uninit_seen, Some((prefix, mpi)));
                    shortest_uninit_seen = Some((prefix, mpi));
                } else {
                    debug!("check_parent_of_field {:?} is definitely initialized", (prefix, mpi));
                }
            }

            if let Some((prefix, mpi)) = shortest_uninit_seen {
                // Check for a reassignment into a uninitialized field of a union (for example,
                // after a move out). In this case, do not report a error here. There is an
                // exception, if this is the first assignment into the union (that is, there is
                // no move out from an earlier location) then this is an attempt at initialization
                // of the union - we should error in that case.
                let tcx = this.infcx.tcx;
                if let ty::Adt(def, _) = base.ty(this.body, tcx).ty.sty {
                    if def.is_union() {
                        if this.move_data.path_map[mpi].iter().any(|moi| {
                            this.move_data.moves[*moi].source.is_predecessor_of(
                                location, this.body,
                            )
                        }) {
                            return;
                        }
                    }
                }

                this.report_use_of_moved_or_uninitialized(
                    location,
                    InitializationRequiringAction::PartialAssignment,
                    (prefix, base, span),
                    mpi,
                );
            }
        }
    }

    /// Checks the permissions for the given place and read or write kind
    ///
    /// Returns `true` if an error is reported.
    fn check_access_permissions(
        &mut self,
        (place, span): (&Place<'tcx>, Span),
        kind: ReadOrWrite,
        is_local_mutation_allowed: LocalMutationIsAllowed,
        flow_state: &Flows<'cx, 'tcx>,
        location: Location,
    ) -> bool {
        debug!(
            "check_access_permissions({:?}, {:?}, is_local_mutation_allowed: {:?})",
            place, kind, is_local_mutation_allowed
        );

        let error_access;
        let the_place_err;

        // rust-lang/rust#21232, #54986: during period where we reject
        // partial initialization, do not complain about mutability
        // errors except for actual mutation (as opposed to an attempt
        // to do a partial initialization).
        let previously_initialized = if let Some(local) = place.base_local() {
            self.is_local_ever_initialized(local, flow_state).is_some()
        } else {
            true
        };

        match kind {
            Reservation(WriteKind::MutableBorrow(borrow_kind @ BorrowKind::Unique))
            | Reservation(WriteKind::MutableBorrow(borrow_kind @ BorrowKind::Mut { .. }))
            | Write(WriteKind::MutableBorrow(borrow_kind @ BorrowKind::Unique))
            | Write(WriteKind::MutableBorrow(borrow_kind @ BorrowKind::Mut { .. })) => {
                let is_local_mutation_allowed = match borrow_kind {
                    BorrowKind::Unique => LocalMutationIsAllowed::Yes,
                    BorrowKind::Mut { .. } => is_local_mutation_allowed,
                    BorrowKind::Shared | BorrowKind::Shallow => unreachable!(),
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

            Reservation(wk @ WriteKind::Move)
            | Write(wk @ WriteKind::Move)
            | Reservation(wk @ WriteKind::StorageDeadOrDrop)
            | Reservation(wk @ WriteKind::MutableBorrow(BorrowKind::Shared))
            | Reservation(wk @ WriteKind::MutableBorrow(BorrowKind::Shallow))
            | Write(wk @ WriteKind::StorageDeadOrDrop)
            | Write(wk @ WriteKind::MutableBorrow(BorrowKind::Shared))
            | Write(wk @ WriteKind::MutableBorrow(BorrowKind::Shallow)) => {
                if let (Err(_place_err), true) = (
                    self.is_mutable(place, is_local_mutation_allowed),
                    self.errors_buffer.is_empty()
                ) {
                    if self.infcx.tcx.migrate_borrowck() {
                        // rust-lang/rust#46908: In pure NLL mode this
                        // code path should be unreachable (and thus
                        // we signal an ICE in the else branch
                        // here). But we can legitimately get here
                        // under borrowck=migrate mode, so instead of
                        // ICE'ing we instead report a legitimate
                        // error (which will then be downgraded to a
                        // warning by the migrate machinery).
                        error_access = match wk {
                            WriteKind::MutableBorrow(_) => AccessKind::MutableBorrow,
                            WriteKind::Move => AccessKind::Move,
                            WriteKind::StorageDeadOrDrop |
                            WriteKind::Mutate => AccessKind::Mutate,
                        };
                        self.report_mutability_error(
                            place,
                            span,
                            _place_err,
                            error_access,
                            location,
                        );
                    } else {
                        span_bug!(
                            span,
                            "Accessing `{:?}` with the kind `{:?}` shouldn't be possible",
                            place,
                            kind,
                        );
                    }
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
            | Read(ReadKind::Borrow(BorrowKind::Shallow))
            | Read(ReadKind::Copy) => {
                // Access authorized
                return false;
            }
        }

        // at this point, we have set up the error reporting state.
        return if previously_initialized {
            self.report_mutability_error(
                place,
                span,
                the_place_err,
                error_access,
                location,
            );
            true
        } else {
            false
        };
    }

    fn is_local_ever_initialized(
        &self,
        local: Local,
        flow_state: &Flows<'cx, 'tcx>,
    ) -> Option<InitIndex> {
        let mpi = self.move_data.rev_lookup.find_local(local);
        let ii = &self.move_data.init_path_map[mpi];
        for &index in ii {
            if flow_state.ever_inits.contains(index) {
                return Some(index);
            }
        }
        None
    }

    /// Adds the place into the used mutable variables set
    fn add_used_mut<'d>(&mut self, root_place: RootPlace<'d, 'tcx>, flow_state: &Flows<'cx, 'tcx>) {
        match root_place {
            RootPlace {
                place: Place::Base(PlaceBase::Local(local)),
                is_local_mutation_allowed,
            } => {
                // If the local may have been initialized, and it is now currently being
                // mutated, then it is justified to be annotated with the `mut`
                // keyword, since the mutation may be a possible reassignment.
                if is_local_mutation_allowed != LocalMutationIsAllowed::Yes &&
                    self.is_local_ever_initialized(*local, flow_state).is_some()
                {
                    self.used_mut.insert(*local);
                }
            }
            RootPlace {
                place: _,
                is_local_mutation_allowed: LocalMutationIsAllowed::Yes,
            } => {}
            RootPlace {
                place: place @ Place::Projection(_),
                is_local_mutation_allowed: _,
            } => {
                if let Some(field) = self.is_upvar_field_projection(place) {
                    self.used_mut_upvars.push(field);
                }
            }
            RootPlace {
                place: Place::Base(PlaceBase::Static(..)),
                is_local_mutation_allowed: _,
            } => {}
        }
    }

    /// Whether this value can be written or borrowed mutably.
    /// Returns the root place if the place passed in is a projection.
    fn is_mutable<'d>(
        &self,
        place: &'d Place<'tcx>,
        is_local_mutation_allowed: LocalMutationIsAllowed,
    ) -> Result<RootPlace<'d, 'tcx>, &'d Place<'tcx>> {
        match *place {
            Place::Base(PlaceBase::Local(local)) => {
                let local = &self.body.local_decls[local];
                match local.mutability {
                    Mutability::Not => match is_local_mutation_allowed {
                        LocalMutationIsAllowed::Yes => Ok(RootPlace {
                            place,
                            is_local_mutation_allowed: LocalMutationIsAllowed::Yes,
                        }),
                        LocalMutationIsAllowed::ExceptUpvars => Ok(RootPlace {
                            place,
                            is_local_mutation_allowed: LocalMutationIsAllowed::ExceptUpvars,
                        }),
                        LocalMutationIsAllowed::No => Err(place),
                    },
                    Mutability::Mut => Ok(RootPlace {
                        place,
                        is_local_mutation_allowed,
                    }),
                }
            }
            // The rules for promotion are made by `qualify_consts`, there wouldn't even be a
            // `Place::Promoted` if the promotion weren't 100% legal. So we just forward this
            Place::Base(PlaceBase::Static(box Static{kind: StaticKind::Promoted(_), ..})) =>
                Ok(RootPlace {
                    place,
                    is_local_mutation_allowed,
                }),
            Place::Base(PlaceBase::Static(box Static{ kind: StaticKind::Static(def_id), .. })) => {
                if !self.infcx.tcx.is_mutable_static(def_id) {
                    Err(place)
                } else {
                    Ok(RootPlace {
                        place,
                        is_local_mutation_allowed,
                    })
                }
            }
            Place::Projection(ref proj) => {
                match proj.elem {
                    ProjectionElem::Deref => {
                        let base_ty = proj.base.ty(self.body, self.infcx.tcx).ty;

                        // Check the kind of deref to decide
                        match base_ty.sty {
                            ty::Ref(_, _, mutbl) => {
                                match mutbl {
                                    // Shared borrowed data is never mutable
                                    hir::MutImmutable => Err(place),
                                    // Mutably borrowed data is mutable, but only if we have a
                                    // unique path to the `&mut`
                                    hir::MutMutable => {
                                        let mode = match self.is_upvar_field_projection(place) {
                                            Some(field)
                                                if self.upvars[field.index()].by_ref =>
                                            {
                                                is_local_mutation_allowed
                                            }
                                            _ => LocalMutationIsAllowed::Yes,
                                        };

                                        self.is_mutable(&proj.base, mode)
                                    }
                                }
                            }
                            ty::RawPtr(tnm) => {
                                match tnm.mutbl {
                                    // `*const` raw pointers are not mutable
                                    hir::MutImmutable => Err(place),
                                    // `*mut` raw pointers are always mutable, regardless of
                                    // context. The users have to check by themselves.
                                    hir::MutMutable => {
                                        Ok(RootPlace {
                                            place,
                                            is_local_mutation_allowed,
                                        })
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
                        let upvar_field_projection = self.is_upvar_field_projection(place);
                        if let Some(field) = upvar_field_projection {
                            let upvar = &self.upvars[field.index()];
                            debug!(
                                "upvar.mutability={:?} local_mutation_is_allowed={:?} place={:?}",
                                upvar, is_local_mutation_allowed, place
                            );
                            match (upvar.mutability, is_local_mutation_allowed) {
                                (Mutability::Not, LocalMutationIsAllowed::No)
                                | (Mutability::Not, LocalMutationIsAllowed::ExceptUpvars) => {
                                    Err(place)
                                }
                                (Mutability::Not, LocalMutationIsAllowed::Yes)
                                | (Mutability::Mut, _) => {
                                    // Subtle: this is an upvar
                                    // reference, so it looks like
                                    // `self.foo` -- we want to double
                                    // check that the location `*self`
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
                                    Ok(RootPlace {
                                        place,
                                        is_local_mutation_allowed,
                                    })
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

    /// If `place` is a field projection, and the field is being projected from a closure type,
    /// then returns the index of the field being projected. Note that this closure will always
    /// be `self` in the current MIR, because that is the only time we directly access the fields
    /// of a closure type.
    pub fn is_upvar_field_projection(&self, place: &Place<'tcx>) -> Option<Field> {
        let (place, by_ref) = if let Place::Projection(ref proj) = place {
            if let ProjectionElem::Deref = proj.elem {
                (&proj.base, true)
            } else {
                (place, false)
            }
        } else {
            (place, false)
        };

        match place {
            Place::Projection(ref proj) => match proj.elem {
                ProjectionElem::Field(field, _ty) => {
                    let tcx = self.infcx.tcx;
                    let base_ty = proj.base.ty(self.body, tcx).ty;

                    if (base_ty.is_closure() || base_ty.is_generator()) &&
                        (!by_ref || self.upvars[field.index()].by_ref)
                    {
                        Some(field)
                    } else {
                        None
                    }
                },
                _ => None,
            }
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
    /// or equal - i.e., they can't "partially" overlap as can occur with
    /// unions. This is the "base case" on which we recur for extensions
    /// of the place.
    EqualOrDisjoint,
    /// The places are disjoint, so we know all extensions of them
    /// will also be disjoint.
    Disjoint,
}
