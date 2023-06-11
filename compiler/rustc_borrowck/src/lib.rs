//! This query borrow-checks the MIR to (further) ensure it is not broken.

#![feature(associated_type_bounds)]
#![feature(box_patterns)]
#![feature(let_chains)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(lazy_cell)]
#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]
#![feature(trusted_step)]
#![feature(try_blocks)]
#![recursion_limit = "256"]

#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate tracing;

use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::graph::dominators::Dominators;
use rustc_errors::{Diagnostic, DiagnosticBuilder, DiagnosticMessage, SubdiagnosticMessage};
use rustc_fluent_macro::fluent_messages;
use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_index::bit_set::ChunkedBitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_infer::infer::{
    InferCtxt, NllRegionVariableOrigin, RegionVariableOrigin, TyCtxtInferExt,
};
use rustc_middle::mir::{
    traversal, Body, ClearCrossCrate, Local, Location, Mutability, NonDivergingIntrinsic, Operand,
    Place, PlaceElem, PlaceRef, VarDebugInfoContents,
};
use rustc_middle::mir::{AggregateKind, BasicBlock, BorrowCheckResult, BorrowKind};
use rustc_middle::mir::{InlineAsmOperand, Terminator, TerminatorKind};
use rustc_middle::mir::{ProjectionElem, Promoted, Rvalue, Statement, StatementKind};
use rustc_middle::query::Providers;
use rustc_middle::traits::DefiningAnchor;
use rustc_middle::ty::{self, CapturedPlace, ParamEnv, RegionVid, TyCtxt};
use rustc_session::lint::builtin::UNUSED_MUT;
use rustc_span::{Span, Symbol};
use rustc_target::abi::FieldIdx;

use either::Either;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ops::Deref;
use std::rc::Rc;

use rustc_mir_dataflow::impls::{
    EverInitializedPlaces, MaybeInitializedPlaces, MaybeUninitializedPlaces,
};
use rustc_mir_dataflow::move_paths::{InitIndex, MoveOutIndex, MovePathIndex};
use rustc_mir_dataflow::move_paths::{InitLocation, LookupResult, MoveData, MoveError};
use rustc_mir_dataflow::Analysis;
use rustc_mir_dataflow::MoveDataParamEnv;

use crate::session_diagnostics::VarNeedNotMut;

use self::diagnostics::{AccessKind, RegionName};
use self::location::LocationTable;
use self::prefixes::PrefixSet;
use consumers::{BodyWithBorrowckFacts, ConsumerOptions};

use self::path_utils::*;

pub mod borrow_set;
mod borrowck_errors;
mod constraint_generation;
mod constraints;
mod dataflow;
mod def_use;
mod diagnostics;
mod facts;
mod invalidation;
mod location;
mod member_constraints;
mod nll;
mod path_utils;
mod place_ext;
mod places_conflict;
mod prefixes;
mod region_infer;
mod renumber;
mod session_diagnostics;
mod type_check;
mod universal_regions;
mod used_muts;
mod util;

/// A public API provided for the Rust compiler consumers.
pub mod consumers;

use borrow_set::{BorrowData, BorrowSet};
use dataflow::{BorrowIndex, BorrowckFlowState as Flows, BorrowckResults, Borrows};
use nll::PoloniusOutput;
use place_ext::PlaceExt;
use places_conflict::{places_conflict, PlaceConflictBias};
use region_infer::RegionInferenceContext;
use renumber::RegionCtxt;

fluent_messages! { "../messages.ftl" }

// FIXME(eddyb) perhaps move this somewhere more centrally.
#[derive(Debug)]
struct Upvar<'tcx> {
    place: CapturedPlace<'tcx>,

    /// If true, the capture is behind a reference.
    by_ref: bool,
}

/// Associate some local constants with the `'tcx` lifetime
struct TyCtxtConsts<'tcx>(TyCtxt<'tcx>);
impl<'tcx> TyCtxtConsts<'tcx> {
    const DEREF_PROJECTION: &'tcx [PlaceElem<'tcx>; 1] = &[ProjectionElem::Deref];
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { mir_borrowck, ..*providers };
}

fn mir_borrowck(tcx: TyCtxt<'_>, def: LocalDefId) -> &BorrowCheckResult<'_> {
    let (input_body, promoted) = tcx.mir_promoted(def);
    debug!("run query mir_borrowck: {}", tcx.def_path_str(def));

    if input_body.borrow().should_skip() {
        debug!("Skipping borrowck because of injected body");
        // Let's make up a borrowck result! Fun times!
        let result = BorrowCheckResult {
            concrete_opaque_types: FxIndexMap::default(),
            closure_requirements: None,
            used_mut_upvars: SmallVec::new(),
            tainted_by_errors: None,
        };
        return tcx.arena.alloc(result);
    }

    let hir_owner = tcx.hir().local_def_id_to_hir_id(def).owner;

    let infcx =
        tcx.infer_ctxt().with_opaque_type_inference(DefiningAnchor::Bind(hir_owner.def_id)).build();
    let input_body: &Body<'_> = &input_body.borrow();
    let promoted: &IndexSlice<_, _> = &promoted.borrow();
    let opt_closure_req = do_mir_borrowck(&infcx, input_body, promoted, None).0;
    debug!("mir_borrowck done");

    tcx.arena.alloc(opt_closure_req)
}

/// Perform the actual borrow checking.
///
/// Use `consumer_options: None` for the default behavior of returning
/// [`BorrowCheckResult`] only. Otherwise, return [`BodyWithBorrowckFacts`] according
/// to the given [`ConsumerOptions`].
#[instrument(skip(infcx, input_body, input_promoted), fields(id=?input_body.source.def_id()), level = "debug")]
fn do_mir_borrowck<'tcx>(
    infcx: &InferCtxt<'tcx>,
    input_body: &Body<'tcx>,
    input_promoted: &IndexSlice<Promoted, Body<'tcx>>,
    consumer_options: Option<ConsumerOptions>,
) -> (BorrowCheckResult<'tcx>, Option<Box<BodyWithBorrowckFacts<'tcx>>>) {
    let def = input_body.source.def_id().expect_local();
    debug!(?def);

    let tcx = infcx.tcx;
    let infcx = BorrowckInferCtxt::new(infcx);
    let param_env = tcx.param_env(def);

    let mut local_names = IndexVec::from_elem(None, &input_body.local_decls);
    for var_debug_info in &input_body.var_debug_info {
        if let VarDebugInfoContents::Place(place) = var_debug_info.value {
            if let Some(local) = place.as_local() {
                if let Some(prev_name) = local_names[local] && var_debug_info.name != prev_name {
                    span_bug!(
                        var_debug_info.source_info.span,
                        "local {:?} has many names (`{}` vs `{}`)",
                        local,
                        prev_name,
                        var_debug_info.name
                    );
                }
                local_names[local] = Some(var_debug_info.name);
            }
        }
    }

    let mut errors = error::BorrowckErrors::new(infcx.tcx);

    // Gather the upvars of a closure, if any.
    if let Some(e) = input_body.tainted_by_errors {
        infcx.set_tainted_by_errors(e);
        errors.set_tainted_by_errors(e);
    }
    let upvars: Vec<_> = tcx
        .closure_captures(def)
        .iter()
        .map(|&captured_place| {
            let capture = captured_place.info.capture_kind;
            let by_ref = match capture {
                ty::UpvarCapture::ByValue => false,
                ty::UpvarCapture::ByRef(..) => true,
            };
            Upvar { place: captured_place.clone(), by_ref }
        })
        .collect();

    // Replace all regions with fresh inference variables. This
    // requires first making our own copy of the MIR. This copy will
    // be modified (in place) to contain non-lexical lifetimes. It
    // will have a lifetime tied to the inference context.
    let mut body_owned = input_body.clone();
    let mut promoted = input_promoted.to_owned();
    let free_regions =
        nll::replace_regions_in_mir(&infcx, param_env, &mut body_owned, &mut promoted);
    let body = &body_owned; // no further changes

    let location_table_owned = LocationTable::new(body);
    let location_table = &location_table_owned;

    let (move_data, move_errors): (MoveData<'tcx>, Vec<(Place<'tcx>, MoveError<'tcx>)>) =
        match MoveData::gather_moves(&body, tcx, param_env) {
            Ok((_, move_data)) => (move_data, Vec::new()),
            Err((move_data, move_errors)) => (move_data, move_errors),
        };
    let promoted_errors = promoted
        .iter_enumerated()
        .map(|(idx, body)| (idx, MoveData::gather_moves(&body, tcx, param_env)));

    let mdpe = MoveDataParamEnv { move_data, param_env };

    let mut flow_inits = MaybeInitializedPlaces::new(tcx, &body, &mdpe)
        .into_engine(tcx, &body)
        .pass_name("borrowck")
        .iterate_to_fixpoint()
        .into_results_cursor(&body);

    let locals_are_invalidated_at_exit = tcx.hir().body_owner_kind(def).is_fn_or_closure();
    let borrow_set =
        Rc::new(BorrowSet::build(tcx, body, locals_are_invalidated_at_exit, &mdpe.move_data));

    // Compute non-lexical lifetimes.
    let nll::NllOutput {
        regioncx,
        opaque_type_values,
        polonius_input,
        polonius_output,
        opt_closure_req,
        nll_errors,
    } = nll::compute_regions(
        &infcx,
        free_regions,
        body,
        &promoted,
        location_table,
        param_env,
        &mut flow_inits,
        &mdpe.move_data,
        &borrow_set,
        &upvars,
        consumer_options,
    );

    // Dump MIR results into a file, if that is enabled. This let us
    // write unit-tests, as well as helping with debugging.
    nll::dump_mir_results(&infcx, &body, &regioncx, &opt_closure_req);

    // We also have a `#[rustc_regions]` annotation that causes us to dump
    // information.
    nll::dump_annotation(
        &infcx,
        &body,
        &regioncx,
        &opt_closure_req,
        &opaque_type_values,
        &mut errors,
    );

    // The various `flow_*` structures can be large. We drop `flow_inits` here
    // so it doesn't overlap with the others below. This reduces peak memory
    // usage significantly on some benchmarks.
    drop(flow_inits);

    let regioncx = Rc::new(regioncx);

    let flow_borrows = Borrows::new(tcx, body, &regioncx, &borrow_set)
        .into_engine(tcx, body)
        .pass_name("borrowck")
        .iterate_to_fixpoint();
    let flow_uninits = MaybeUninitializedPlaces::new(tcx, body, &mdpe)
        .into_engine(tcx, body)
        .pass_name("borrowck")
        .iterate_to_fixpoint();
    let flow_ever_inits = EverInitializedPlaces::new(tcx, body, &mdpe)
        .into_engine(tcx, body)
        .pass_name("borrowck")
        .iterate_to_fixpoint();

    let movable_generator =
        // The first argument is the generator type passed by value
        if let Some(local) = body.local_decls.raw.get(1)
        // Get the interior types and substs which typeck computed
        && let ty::Generator(_, _, hir::Movability::Static) = local.ty.kind()
    {
        false
    } else {
        true
    };

    for (idx, move_data_results) in promoted_errors {
        let promoted_body = &promoted[idx];

        if let Err((move_data, move_errors)) = move_data_results {
            let mut promoted_mbcx = MirBorrowckCtxt {
                infcx: &infcx,
                param_env,
                body: promoted_body,
                move_data: &move_data,
                location_table, // no need to create a real one for the promoted, it is not used
                movable_generator,
                fn_self_span_reported: Default::default(),
                locals_are_invalidated_at_exit,
                access_place_error_reported: Default::default(),
                reservation_error_reported: Default::default(),
                uninitialized_error_reported: Default::default(),
                regioncx: regioncx.clone(),
                used_mut: Default::default(),
                used_mut_upvars: SmallVec::new(),
                borrow_set: Rc::clone(&borrow_set),
                upvars: Vec::new(),
                local_names: IndexVec::from_elem(None, &promoted_body.local_decls),
                region_names: RefCell::default(),
                next_region_name: RefCell::new(1),
                polonius_output: None,
                errors,
            };
            promoted_mbcx.report_move_errors(move_errors);
            errors = promoted_mbcx.errors;
        };
    }

    let mut mbcx = MirBorrowckCtxt {
        infcx: &infcx,
        param_env,
        body,
        move_data: &mdpe.move_data,
        location_table,
        movable_generator,
        locals_are_invalidated_at_exit,
        fn_self_span_reported: Default::default(),
        access_place_error_reported: Default::default(),
        reservation_error_reported: Default::default(),
        uninitialized_error_reported: Default::default(),
        regioncx: Rc::clone(&regioncx),
        used_mut: Default::default(),
        used_mut_upvars: SmallVec::new(),
        borrow_set: Rc::clone(&borrow_set),
        upvars,
        local_names,
        region_names: RefCell::default(),
        next_region_name: RefCell::new(1),
        polonius_output,
        errors,
    };

    // Compute and report region errors, if any.
    mbcx.report_region_errors(nll_errors);

    let mut results = BorrowckResults {
        ever_inits: flow_ever_inits,
        uninits: flow_uninits,
        borrows: flow_borrows,
    };

    mbcx.report_move_errors(move_errors);

    rustc_mir_dataflow::visit_results(
        body,
        traversal::reverse_postorder(body).map(|(bb, _)| bb),
        &mut results,
        &mut mbcx,
    );

    // For each non-user used mutable variable, check if it's been assigned from
    // a user-declared local. If so, then put that local into the used_mut set.
    // Note that this set is expected to be small - only upvars from closures
    // would have a chance of erroneously adding non-user-defined mutable vars
    // to the set.
    let temporary_used_locals: FxIndexSet<Local> = mbcx
        .used_mut
        .iter()
        .filter(|&local| !mbcx.body.local_decls[*local].is_user_variable())
        .cloned()
        .collect();
    // For the remaining unused locals that are marked as mutable, we avoid linting any that
    // were never initialized. These locals may have been removed as unreachable code; or will be
    // linted as unused variables.
    let unused_mut_locals =
        mbcx.body.mut_vars_iter().filter(|local| !mbcx.used_mut.contains(local)).collect();
    mbcx.gather_used_muts(temporary_used_locals, unused_mut_locals);

    debug!("mbcx.used_mut: {:?}", mbcx.used_mut);
    let used_mut = std::mem::take(&mut mbcx.used_mut);
    for local in mbcx.body.mut_vars_and_args_iter().filter(|local| !used_mut.contains(local)) {
        let local_decl = &mbcx.body.local_decls[local];
        let lint_root = match &mbcx.body.source_scopes[local_decl.source_info.scope].local_data {
            ClearCrossCrate::Set(data) => data.lint_root,
            _ => continue,
        };

        // Skip over locals that begin with an underscore or have no name
        match mbcx.local_names[local] {
            Some(name) => {
                if name.as_str().starts_with('_') {
                    continue;
                }
            }
            None => continue,
        }

        let span = local_decl.source_info.span;
        if span.desugaring_kind().is_some() {
            // If the `mut` arises as part of a desugaring, we should ignore it.
            continue;
        }

        let mut_span = tcx.sess.source_map().span_until_non_whitespace(span);

        tcx.emit_spanned_lint(UNUSED_MUT, lint_root, span, VarNeedNotMut { span: mut_span })
    }

    let tainted_by_errors = mbcx.emit_errors();

    let result = BorrowCheckResult {
        concrete_opaque_types: opaque_type_values,
        closure_requirements: opt_closure_req,
        used_mut_upvars: mbcx.used_mut_upvars,
        tainted_by_errors,
    };

    let body_with_facts = if consumer_options.is_some() {
        let output_facts = mbcx.polonius_output;
        Some(Box::new(BodyWithBorrowckFacts {
            body: body_owned,
            promoted,
            borrow_set,
            region_inference_context: regioncx,
            location_table: polonius_input.as_ref().map(|_| location_table_owned),
            input_facts: polonius_input,
            output_facts,
        }))
    } else {
        None
    };

    debug!("do_mir_borrowck: result = {:#?}", result);

    (result, body_with_facts)
}

pub struct BorrowckInferCtxt<'cx, 'tcx> {
    pub(crate) infcx: &'cx InferCtxt<'tcx>,
    pub(crate) reg_var_to_origin: RefCell<FxIndexMap<ty::RegionVid, RegionCtxt>>,
}

impl<'cx, 'tcx> BorrowckInferCtxt<'cx, 'tcx> {
    pub(crate) fn new(infcx: &'cx InferCtxt<'tcx>) -> Self {
        BorrowckInferCtxt { infcx, reg_var_to_origin: RefCell::new(Default::default()) }
    }

    pub(crate) fn next_region_var<F>(
        &self,
        origin: RegionVariableOrigin,
        get_ctxt_fn: F,
    ) -> ty::Region<'tcx>
    where
        F: Fn() -> RegionCtxt,
    {
        let next_region = self.infcx.next_region_var(origin);
        let vid = next_region.as_var();

        if cfg!(debug_assertions) {
            debug!("inserting vid {:?} with origin {:?} into var_to_origin", vid, origin);
            let ctxt = get_ctxt_fn();
            let mut var_to_origin = self.reg_var_to_origin.borrow_mut();
            assert_eq!(var_to_origin.insert(vid, ctxt), None);
        }

        next_region
    }

    #[instrument(skip(self, get_ctxt_fn), level = "debug")]
    pub(crate) fn next_nll_region_var<F>(
        &self,
        origin: NllRegionVariableOrigin,
        get_ctxt_fn: F,
    ) -> ty::Region<'tcx>
    where
        F: Fn() -> RegionCtxt,
    {
        let next_region = self.infcx.next_nll_region_var(origin);
        let vid = next_region.as_var();

        if cfg!(debug_assertions) {
            debug!("inserting vid {:?} with origin {:?} into var_to_origin", vid, origin);
            let ctxt = get_ctxt_fn();
            let mut var_to_origin = self.reg_var_to_origin.borrow_mut();
            assert_eq!(var_to_origin.insert(vid, ctxt), None);
        }

        next_region
    }
}

impl<'cx, 'tcx> Deref for BorrowckInferCtxt<'cx, 'tcx> {
    type Target = InferCtxt<'tcx>;

    fn deref(&self) -> &'cx Self::Target {
        self.infcx
    }
}

struct MirBorrowckCtxt<'cx, 'tcx> {
    infcx: &'cx BorrowckInferCtxt<'cx, 'tcx>,
    param_env: ParamEnv<'tcx>,
    body: &'cx Body<'tcx>,
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
    access_place_error_reported: FxIndexSet<(Place<'tcx>, Span)>,
    /// This field keeps track of when borrow conflict errors are reported
    /// for reservations, so that we don't report seemingly duplicate
    /// errors for corresponding activations.
    //
    // FIXME: ideally this would be a set of `BorrowIndex`, not `Place`s,
    // but it is currently inconvenient to track down the `BorrowIndex`
    // at the time we detect and report a reservation error.
    reservation_error_reported: FxIndexSet<Place<'tcx>>,
    /// This fields keeps track of the `Span`s that we have
    /// used to report extra information for `FnSelfUse`, to avoid
    /// unnecessarily verbose errors.
    fn_self_span_reported: FxIndexSet<Span>,
    /// This field keeps track of errors reported in the checking of uninitialized variables,
    /// so that we don't report seemingly duplicate errors.
    uninitialized_error_reported: FxIndexSet<PlaceRef<'tcx>>,
    /// This field keeps track of all the local variables that are declared mut and are mutated.
    /// Used for the warning issued by an unused mutable local variable.
    used_mut: FxIndexSet<Local>,
    /// If the function we're checking is a closure, then we'll need to report back the list of
    /// mutable upvars that have been used. This field keeps track of them.
    used_mut_upvars: SmallVec<[FieldIdx; 8]>,
    /// Region inference context. This contains the results from region inference and lets us e.g.
    /// find out which CFG points are contained in each borrow region.
    regioncx: Rc<RegionInferenceContext<'tcx>>,

    /// The set of borrows extracted from the MIR
    borrow_set: Rc<BorrowSet<'tcx>>,

    /// Information about upvars not necessarily preserved in types or MIR
    upvars: Vec<Upvar<'tcx>>,

    /// Names of local (user) variables (extracted from `var_debug_info`).
    local_names: IndexVec<Local, Option<Symbol>>,

    /// Record the region names generated for each region in the given
    /// MIR def so that we can reuse them later in help/error messages.
    region_names: RefCell<FxIndexMap<RegionVid, RegionName>>,

    /// The counter for generating new region names.
    next_region_name: RefCell<usize>,

    /// Results of Polonius analysis.
    polonius_output: Option<Rc<PoloniusOutput>>,

    errors: error::BorrowckErrors<'tcx>,
}

// Check that:
// 1. assignments are always made to mutable locations (FIXME: does that still really go here?)
// 2. loans made in overlapping scopes do not conflict
// 3. assignments do not affect things loaned out as immutable
// 4. moves do not affect things loaned out in any way
impl<'cx, 'tcx, R> rustc_mir_dataflow::ResultsVisitor<'cx, 'tcx, R> for MirBorrowckCtxt<'cx, 'tcx> {
    type FlowState = Flows<'cx, 'tcx>;

    fn visit_statement_before_primary_effect(
        &mut self,
        _results: &R,
        flow_state: &Flows<'cx, 'tcx>,
        stmt: &'cx Statement<'tcx>,
        location: Location,
    ) {
        debug!("MirBorrowckCtxt::process_statement({:?}, {:?}): {:?}", location, stmt, flow_state);
        let span = stmt.source_info.span;

        self.check_activations(location, span, flow_state);

        match &stmt.kind {
            StatementKind::Assign(box (lhs, rhs)) => {
                self.consume_rvalue(location, (rhs, span), flow_state);

                self.mutate_place(location, (*lhs, span), Shallow(None), flow_state);
            }
            StatementKind::FakeRead(box (_, place)) => {
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
                    (place.as_ref(), span),
                    flow_state,
                );
            }
            StatementKind::Intrinsic(box kind) => match kind {
                NonDivergingIntrinsic::Assume(op) => self.consume_operand(location, (op, span), flow_state),
                NonDivergingIntrinsic::CopyNonOverlapping(..) => span_bug!(
                    span,
                    "Unexpected CopyNonOverlapping, should only appear after lower_intrinsics",
                )
            }
            // Only relevant for mir typeck
            StatementKind::AscribeUserType(..)
            // Only relevant for liveness and unsafeck
            | StatementKind::PlaceMention(..)
            // Doesn't have any language semantics
            | StatementKind::Coverage(..)
            // These do not actually affect borrowck
            | StatementKind::ConstEvalCounter
            | StatementKind::StorageLive(..) => {}
            StatementKind::StorageDead(local) => {
                self.access_place(
                    location,
                    (Place::from(*local), span),
                    (Shallow(None), Write(WriteKind::StorageDeadOrDrop)),
                    LocalMutationIsAllowed::Yes,
                    flow_state,
                );
            }
            StatementKind::Nop
            | StatementKind::Retag { .. }
            | StatementKind::Deinit(..)
            | StatementKind::SetDiscriminant { .. } => {
                bug!("Statement not allowed in this MIR phase")
            }
        }
    }

    fn visit_terminator_before_primary_effect(
        &mut self,
        _results: &R,
        flow_state: &Flows<'cx, 'tcx>,
        term: &'cx Terminator<'tcx>,
        loc: Location,
    ) {
        debug!("MirBorrowckCtxt::process_terminator({:?}, {:?}): {:?}", loc, term, flow_state);
        let span = term.source_info.span;

        self.check_activations(loc, span, flow_state);

        match &term.kind {
            TerminatorKind::SwitchInt { discr, targets: _ } => {
                self.consume_operand(loc, (discr, span), flow_state);
            }
            TerminatorKind::Drop { place, target: _, unwind: _, replace } => {
                debug!(
                    "visit_terminator_drop \
                     loc: {:?} term: {:?} place: {:?} span: {:?}",
                    loc, term, place, span
                );

                let write_kind =
                    if *replace { WriteKind::Replace } else { WriteKind::StorageDeadOrDrop };
                self.access_place(
                    loc,
                    (*place, span),
                    (AccessDepth::Drop, Write(write_kind)),
                    LocalMutationIsAllowed::Yes,
                    flow_state,
                );
            }
            TerminatorKind::Call {
                func,
                args,
                destination,
                target: _,
                unwind: _,
                from_hir_call: _,
                fn_span: _,
            } => {
                self.consume_operand(loc, (func, span), flow_state);
                for arg in args {
                    self.consume_operand(loc, (arg, span), flow_state);
                }
                self.mutate_place(loc, (*destination, span), Deep, flow_state);
            }
            TerminatorKind::Assert { cond, expected: _, msg, target: _, unwind: _ } => {
                self.consume_operand(loc, (cond, span), flow_state);
                use rustc_middle::mir::AssertKind;
                if let AssertKind::BoundsCheck { len, index } = &**msg {
                    self.consume_operand(loc, (len, span), flow_state);
                    self.consume_operand(loc, (index, span), flow_state);
                }
            }

            TerminatorKind::Yield { value, resume: _, resume_arg, drop: _ } => {
                self.consume_operand(loc, (value, span), flow_state);
                self.mutate_place(loc, (*resume_arg, span), Deep, flow_state);
            }

            TerminatorKind::InlineAsm {
                template: _,
                operands,
                options: _,
                line_spans: _,
                destination: _,
                unwind: _,
            } => {
                for op in operands {
                    match op {
                        InlineAsmOperand::In { reg: _, value } => {
                            self.consume_operand(loc, (value, span), flow_state);
                        }
                        InlineAsmOperand::Out { reg: _, late: _, place, .. } => {
                            if let Some(place) = place {
                                self.mutate_place(loc, (*place, span), Shallow(None), flow_state);
                            }
                        }
                        InlineAsmOperand::InOut { reg: _, late: _, in_value, out_place } => {
                            self.consume_operand(loc, (in_value, span), flow_state);
                            if let &Some(out_place) = out_place {
                                self.mutate_place(
                                    loc,
                                    (out_place, span),
                                    Shallow(None),
                                    flow_state,
                                );
                            }
                        }
                        InlineAsmOperand::Const { value: _ }
                        | InlineAsmOperand::SymFn { value: _ }
                        | InlineAsmOperand::SymStatic { def_id: _ } => {}
                    }
                }
            }

            TerminatorKind::Goto { target: _ }
            | TerminatorKind::Terminate
            | TerminatorKind::Unreachable
            | TerminatorKind::Resume
            | TerminatorKind::Return
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::FalseEdge { real_target: _, imaginary_target: _ }
            | TerminatorKind::FalseUnwind { real_target: _, unwind: _ } => {
                // no data used, thus irrelevant to borrowck
            }
        }
    }

    fn visit_terminator_after_primary_effect(
        &mut self,
        _results: &R,
        flow_state: &Flows<'cx, 'tcx>,
        term: &'cx Terminator<'tcx>,
        loc: Location,
    ) {
        let span = term.source_info.span;

        match term.kind {
            TerminatorKind::Yield { value: _, resume: _, resume_arg: _, drop: _ } => {
                if self.movable_generator {
                    // Look for any active borrows to locals
                    let borrow_set = self.borrow_set.clone();
                    for i in flow_state.borrows.iter() {
                        let borrow = &borrow_set[i];
                        self.check_for_local_borrow(borrow, span);
                    }
                }
            }

            TerminatorKind::Resume | TerminatorKind::Return | TerminatorKind::GeneratorDrop => {
                // Returning from the function implicitly kills storage for all locals and statics.
                // Often, the storage will already have been killed by an explicit
                // StorageDead, but we don't always emit those (notably on unwind paths),
                // so this "extra check" serves as a kind of backup.
                let borrow_set = self.borrow_set.clone();
                for i in flow_state.borrows.iter() {
                    let borrow = &borrow_set[i];
                    self.check_for_invalidation_at_exit(loc, borrow, span);
                }
            }

            TerminatorKind::Terminate
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Call { .. }
            | TerminatorKind::Drop { .. }
            | TerminatorKind::FalseEdge { real_target: _, imaginary_target: _ }
            | TerminatorKind::FalseUnwind { real_target: _, unwind: _ }
            | TerminatorKind::Goto { .. }
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Unreachable
            | TerminatorKind::InlineAsm { .. } => {}
        }
    }
}

use self::AccessDepth::{Deep, Shallow};
use self::ReadOrWrite::{Activation, Read, Reservation, Write};

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
    Replace,
    MutableBorrow(BorrowKind),
    Mutate,
    Move,
}

/// When checking permissions for a place access, this flag is used to indicate that an immutable
/// local place can be mutated.
//
// FIXME: @nikomatsakis suggested that this flag could be removed with the following modifications:
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
    Borrow,
    MatchOn,
    Use,
    Assignment,
    PartialAssignment,
}

#[derive(Debug)]
struct RootPlace<'tcx> {
    place_local: Local,
    place_projection: &'tcx [PlaceElem<'tcx>],
    is_local_mutation_allowed: LocalMutationIsAllowed,
}

impl InitializationRequiringAction {
    fn as_noun(self) -> &'static str {
        match self {
            InitializationRequiringAction::Borrow => "borrow",
            InitializationRequiringAction::MatchOn => "use", // no good noun
            InitializationRequiringAction::Use => "use",
            InitializationRequiringAction::Assignment => "assign",
            InitializationRequiringAction::PartialAssignment => "assign to part",
        }
    }

    fn as_verb_in_past_tense(self) -> &'static str {
        match self {
            InitializationRequiringAction::Borrow => "borrowed",
            InitializationRequiringAction::MatchOn => "matched on",
            InitializationRequiringAction::Use => "used",
            InitializationRequiringAction::Assignment => "assigned",
            InitializationRequiringAction::PartialAssignment => "partially assigned",
        }
    }

    fn as_general_verb_in_past_tense(self) -> &'static str {
        match self {
            InitializationRequiringAction::Borrow
            | InitializationRequiringAction::MatchOn
            | InitializationRequiringAction::Use => "used",
            InitializationRequiringAction::Assignment => "assigned",
            InitializationRequiringAction::PartialAssignment => "partially assigned",
        }
    }
}

impl<'cx, 'tcx> MirBorrowckCtxt<'cx, 'tcx> {
    fn body(&self) -> &'cx Body<'tcx> {
        self.body
    }

    /// Checks an access to the given place to see if it is allowed. Examines the set of borrows
    /// that are in scope, as well as which paths have been initialized, to ensure that (a) the
    /// place is initialized and (b) it is not borrowed in some way that would prevent this
    /// access.
    ///
    /// Returns `true` if an error is reported.
    fn access_place(
        &mut self,
        location: Location,
        place_span: (Place<'tcx>, Span),
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
        if !self.access_place_error_reported.is_empty()
            && self.access_place_error_reported.contains(&(place_span.0, place_span.1))
        {
            debug!(
                "access_place: suppressing error place_span=`{:?}` kind=`{:?}`",
                place_span, kind
            );
            return;
        }

        let mutability_error = self.check_access_permissions(
            place_span,
            rw,
            is_local_mutation_allowed,
            flow_state,
            location,
        );
        let conflict_error =
            self.check_access_for_conflict(location, place_span, sd, rw, flow_state);

        if conflict_error || mutability_error {
            debug!("access_place: logging error place_span=`{:?}` kind=`{:?}`", place_span, kind);
            self.access_place_error_reported.insert((place_span.0, place_span.1));
        }
    }

    #[instrument(level = "debug", skip(self, flow_state))]
    fn check_access_for_conflict(
        &mut self,
        location: Location,
        place_span: (Place<'tcx>, Span),
        sd: AccessDepth,
        rw: ReadOrWrite,
        flow_state: &Flows<'cx, 'tcx>,
    ) -> bool {
        let mut error_reported = false;
        let tcx = self.infcx.tcx;
        let body = self.body;
        let borrow_set = self.borrow_set.clone();

        // Use polonius output if it has been enabled.
        let polonius_output = self.polonius_output.clone();
        let borrows_in_scope = if let Some(polonius) = &polonius_output {
            let location = self.location_table.start_index(location);
            Either::Left(polonius.errors_at(location).iter().copied())
        } else {
            Either::Right(flow_state.borrows.iter())
        };

        each_borrow_involving_path(
            self,
            tcx,
            body,
            location,
            (sd, place_span.0),
            &borrow_set,
            borrows_in_scope,
            |this, borrow_index, borrow| match (rw, borrow.kind) {
                // Obviously an activation is compatible with its own
                // reservation (or even prior activating uses of same
                // borrow); so don't check if they interfere.
                //
                // NOTE: *reservations* do conflict with themselves;
                // thus aren't injecting unsoundness w/ this check.)
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

                (Read(_), BorrowKind::Shared | BorrowKind::Shallow)
                | (
                    Read(ReadKind::Borrow(BorrowKind::Shallow)),
                    BorrowKind::Unique | BorrowKind::Mut { .. },
                ) => Control::Continue,

                (Reservation(_), BorrowKind::Shallow | BorrowKind::Shared) => {
                    // This used to be a future compatibility warning (to be
                    // disallowed on NLL). See rust-lang/rust#56254
                    Control::Continue
                }

                (Write(WriteKind::Move), BorrowKind::Shallow) => {
                    // Handled by initialization checks.
                    Control::Continue
                }

                (Read(kind), BorrowKind::Unique | BorrowKind::Mut { .. }) => {
                    // Reading from mere reservations of mutable-borrows is OK.
                    if !is_active(this.dominators(), borrow, location) {
                        assert!(allow_two_phase_borrow(borrow.kind));
                        return Control::Continue;
                    }

                    error_reported = true;
                    match kind {
                        ReadKind::Copy => {
                            let err = this
                                .report_use_while_mutably_borrowed(location, place_span, borrow);
                            this.buffer_error(err);
                        }
                        ReadKind::Borrow(bk) => {
                            let err =
                                this.report_conflicting_borrow(location, place_span, bk, borrow);
                            this.buffer_error(err);
                        }
                    }
                    Control::Break
                }

                (Reservation(kind) | Activation(kind, _) | Write(kind), _) => {
                    match rw {
                        Reservation(..) => {
                            debug!(
                                "recording invalid reservation of \
                                 place: {:?}",
                                place_span.0
                            );
                            this.reservation_error_reported.insert(place_span.0);
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
                            let err =
                                this.report_conflicting_borrow(location, place_span, bk, borrow);
                            this.buffer_error(err);
                        }
                        WriteKind::StorageDeadOrDrop => this
                            .report_borrowed_value_does_not_live_long_enough(
                                location,
                                borrow,
                                place_span,
                                Some(WriteKind::StorageDeadOrDrop),
                            ),
                        WriteKind::Mutate => {
                            this.report_illegal_mutation_of_borrowed(location, place_span, borrow)
                        }
                        WriteKind::Move => {
                            this.report_move_out_while_borrowed(location, place_span, borrow)
                        }
                        WriteKind::Replace => {
                            this.report_illegal_mutation_of_borrowed(location, place_span, borrow)
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
        place_span: (Place<'tcx>, Span),
        kind: AccessDepth,
        flow_state: &Flows<'cx, 'tcx>,
    ) {
        // Write of P[i] or *P requires P init'd.
        self.check_if_assigned_path_is_moved(location, place_span, flow_state);

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
        (rvalue, span): (&'cx Rvalue<'tcx>, Span),
        flow_state: &Flows<'cx, 'tcx>,
    ) {
        match rvalue {
            &Rvalue::Ref(_ /*rgn*/, bk, place) => {
                let access_kind = match bk {
                    BorrowKind::Shallow => {
                        (Shallow(Some(ArtificialField::ShallowBorrow)), Read(ReadKind::Borrow(bk)))
                    }
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
                    (place.as_ref(), span),
                    flow_state,
                );
            }

            &Rvalue::AddressOf(mutability, place) => {
                let access_kind = match mutability {
                    Mutability::Mut => (
                        Deep,
                        Write(WriteKind::MutableBorrow(BorrowKind::Mut {
                            allow_two_phase_borrow: false,
                        })),
                    ),
                    Mutability::Not => (Deep, Read(ReadKind::Borrow(BorrowKind::Shared))),
                };

                self.access_place(
                    location,
                    (place, span),
                    access_kind,
                    LocalMutationIsAllowed::No,
                    flow_state,
                );

                self.check_if_path_or_subpath_is_moved(
                    location,
                    InitializationRequiringAction::Borrow,
                    (place.as_ref(), span),
                    flow_state,
                );
            }

            Rvalue::ThreadLocalRef(_) => {}

            Rvalue::Use(operand)
            | Rvalue::Repeat(operand, _)
            | Rvalue::UnaryOp(_ /*un_op*/, operand)
            | Rvalue::Cast(_ /*cast_kind*/, operand, _ /*ty*/)
            | Rvalue::ShallowInitBox(operand, _ /*ty*/) => {
                self.consume_operand(location, (operand, span), flow_state)
            }

            &Rvalue::CopyForDeref(place) => {
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
                    (place.as_ref(), span),
                    flow_state,
                );
            }

            &(Rvalue::Len(place) | Rvalue::Discriminant(place)) => {
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
                    (place.as_ref(), span),
                    flow_state,
                );
            }

            Rvalue::BinaryOp(_bin_op, box (operand1, operand2))
            | Rvalue::CheckedBinaryOp(_bin_op, box (operand1, operand2)) => {
                self.consume_operand(location, (operand1, span), flow_state);
                self.consume_operand(location, (operand2, span), flow_state);
            }

            Rvalue::NullaryOp(_op, _ty) => {
                // nullary ops take no dynamic input; no borrowck effect.
            }

            Rvalue::Aggregate(aggregate_kind, operands) => {
                // We need to report back the list of mutable upvars that were
                // moved into the closure and subsequently used by the closure,
                // in order to populate our used_mut set.
                match **aggregate_kind {
                    AggregateKind::Closure(def_id, _) | AggregateKind::Generator(def_id, _, _) => {
                        let def_id = def_id.expect_local();
                        let BorrowCheckResult { used_mut_upvars, .. } =
                            self.infcx.tcx.mir_borrowck(def_id);
                        debug!("{:?} used_mut_upvars={:?}", def_id, used_mut_upvars);
                        for field in used_mut_upvars {
                            self.propagate_closure_used_mut_upvar(&operands[*field]);
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
        let propagate_closure_used_mut_place = |this: &mut Self, place: Place<'tcx>| {
            // We have three possibilities here:
            // a. We are modifying something through a mut-ref
            // b. We are modifying something that is local to our parent
            // c. Current body is a nested closure, and we are modifying path starting from
            //    a Place captured by our parent closure.

            // Handle (c), the path being modified is exactly the path captured by our parent
            if let Some(field) = this.is_upvar_field_projection(place.as_ref()) {
                this.used_mut_upvars.push(field);
                return;
            }

            for (place_ref, proj) in place.iter_projections().rev() {
                // Handle (a)
                if proj == ProjectionElem::Deref {
                    match place_ref.ty(this.body(), this.infcx.tcx).ty.kind() {
                        // We aren't modifying a variable directly
                        ty::Ref(_, _, hir::Mutability::Mut) => return,

                        _ => {}
                    }
                }

                // Handle (c)
                if let Some(field) = this.is_upvar_field_projection(place_ref) {
                    this.used_mut_upvars.push(field);
                    return;
                }
            }

            // Handle(b)
            this.used_mut.insert(place.local);
        };

        // This relies on the current way that by-value
        // captures of a closure are copied/moved directly
        // when generating MIR.
        match *operand {
            Operand::Move(place) | Operand::Copy(place) => {
                match place.as_local() {
                    Some(local) if !self.body.local_decls[local].is_user_variable() => {
                        if self.body.local_decls[local].ty.is_mutable_ptr() {
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

                        let InitLocation::Statement(loc) = init.location else {
                            bug!("temporary initialized in arguments")
                        };

                        let body = self.body;
                        let bbd = &body[loc.block];
                        let stmt = &bbd.statements[loc.statement_index];
                        debug!("temporary assigned in: stmt={:?}", stmt);

                        if let StatementKind::Assign(box (_, Rvalue::Ref(_, _, source))) = stmt.kind
                        {
                            propagate_closure_used_mut_place(self, source);
                        } else {
                            bug!(
                                "closures should only capture user variables \
                                 or references to user variables"
                            );
                        }
                    }
                    _ => propagate_closure_used_mut_place(self, place),
                }
            }
            Operand::Constant(..) => {}
        }
    }

    fn consume_operand(
        &mut self,
        location: Location,
        (operand, span): (&'cx Operand<'tcx>, Span),
        flow_state: &Flows<'cx, 'tcx>,
    ) {
        match *operand {
            Operand::Copy(place) => {
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
                    (place.as_ref(), span),
                    flow_state,
                );
            }
            Operand::Move(place) => {
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
                    (place.as_ref(), span),
                    flow_state,
                );
            }
            Operand::Constant(_) => {}
        }
    }

    /// Checks whether a borrow of this place is invalidated when the function
    /// exits
    #[instrument(level = "debug", skip(self))]
    fn check_for_invalidation_at_exit(
        &mut self,
        location: Location,
        borrow: &BorrowData<'tcx>,
        span: Span,
    ) {
        let place = borrow.borrowed_place;
        let mut root_place = PlaceRef { local: place.local, projection: &[] };

        // FIXME(nll-rfc#40): do more precise destructor tracking here. For now
        // we just know that all locals are dropped at function exit (otherwise
        // we'll have a memory leak) and assume that all statics have a destructor.
        //
        // FIXME: allow thread-locals to borrow other thread locals?

        let (might_be_alive, will_be_dropped) =
            if self.body.local_decls[root_place.local].is_ref_to_thread_local() {
                // Thread-locals might be dropped after the function exits
                // We have to dereference the outer reference because
                // borrows don't conflict behind shared references.
                root_place.projection = TyCtxtConsts::DEREF_PROJECTION;
                (true, true)
            } else {
                (false, self.locals_are_invalidated_at_exit)
            };

        if !will_be_dropped {
            debug!("place_is_invalidated_at_exit({:?}) - won't be dropped", place);
            return;
        }

        let sd = if might_be_alive { Deep } else { Shallow(None) };

        if places_conflict::borrow_conflicts_with_place(
            self.infcx.tcx,
            &self.body,
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
    /// This is called for all Yield expressions on movable generators
    fn check_for_local_borrow(&mut self, borrow: &BorrowData<'tcx>, yield_span: Span) {
        debug!("check_for_local_borrow({:?})", borrow);

        if borrow_of_local_data(borrow.borrowed_place) {
            let err = self.cannot_borrow_across_generator_yield(
                self.retrieve_borrow_spans(borrow).var_or_use(),
                yield_span,
            );

            self.buffer_error(err);
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
                (borrow.borrowed_place, span),
                (Deep, Activation(WriteKind::MutableBorrow(borrow.kind), borrow_index)),
                LocalMutationIsAllowed::No,
                flow_state,
            );
            // We do not need to call `check_if_path_or_subpath_is_moved`
            // again, as we already called it when we made the
            // initial reservation.
        }
    }

    fn check_if_full_path_is_moved(
        &mut self,
        location: Location,
        desired_action: InitializationRequiringAction,
        place_span: (PlaceRef<'tcx>, Span),
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
        // `a.b.c` is assigned a reference to an initialized and
        // well-formed record structure.)

        // Therefore, if we seek out the *closest* prefix for which we
        // have a MovePath, that should capture the initialization
        // state for the place scenario.
        //
        // This code covers scenarios 1, 2, and 3.

        debug!("check_if_full_path_is_moved place: {:?}", place_span.0);
        let (prefix, mpi) = self.move_path_closest_to(place_span.0);
        if maybe_uninits.contains(mpi) {
            self.report_use_of_moved_or_uninitialized(
                location,
                desired_action,
                (prefix, place_span.0, place_span.1),
                mpi,
            );
        } // Only query longest prefix with a MovePath, not further
        // ancestors; dataflow recurs on children when parents
        // move (to support partial (re)inits).
        //
        // (I.e., querying parents breaks scenario 7; but may want
        // to do such a query based on partial-init feature-gate.)
    }

    /// Subslices correspond to multiple move paths, so we iterate through the
    /// elements of the base array. For each element we check
    ///
    /// * Does this element overlap with our slice.
    /// * Is any part of it uninitialized.
    fn check_if_subslice_element_is_moved(
        &mut self,
        location: Location,
        desired_action: InitializationRequiringAction,
        place_span: (PlaceRef<'tcx>, Span),
        maybe_uninits: &ChunkedBitSet<MovePathIndex>,
        from: u64,
        to: u64,
    ) {
        if let Some(mpi) = self.move_path_for_place(place_span.0) {
            let move_paths = &self.move_data.move_paths;

            let root_path = &move_paths[mpi];
            for (child_mpi, child_move_path) in root_path.children(move_paths) {
                let last_proj = child_move_path.place.projection.last().unwrap();
                if let ProjectionElem::ConstantIndex { offset, from_end, .. } = last_proj {
                    debug_assert!(!from_end, "Array constant indexing shouldn't be `from_end`.");

                    if (from..to).contains(offset) {
                        let uninit_child =
                            self.move_data.find_in_move_path_or_its_descendants(child_mpi, |mpi| {
                                maybe_uninits.contains(mpi)
                            });

                        if let Some(uninit_child) = uninit_child {
                            self.report_use_of_moved_or_uninitialized(
                                location,
                                desired_action,
                                (place_span.0, place_span.0, place_span.1),
                                uninit_child,
                            );
                            return; // don't bother finding other problems.
                        }
                    }
                }
            }
        }
    }

    fn check_if_path_or_subpath_is_moved(
        &mut self,
        location: Location,
        desired_action: InitializationRequiringAction,
        place_span: (PlaceRef<'tcx>, Span),
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

        if let Some((place_base, ProjectionElem::Subslice { from, to, from_end: false })) =
            place_span.0.last_projection()
        {
            let place_ty = place_base.ty(self.body(), self.infcx.tcx);
            if let ty::Array(..) = place_ty.ty.kind() {
                self.check_if_subslice_element_is_moved(
                    location,
                    desired_action,
                    (place_base, place_span.1),
                    maybe_uninits,
                    from,
                    to,
                );
                return;
            }
        }

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
            let uninit_mpi = self
                .move_data
                .find_in_move_path_or_its_descendants(mpi, |mpi| maybe_uninits.contains(mpi));

            if let Some(uninit_mpi) = uninit_mpi {
                self.report_use_of_moved_or_uninitialized(
                    location,
                    desired_action,
                    (place_span.0, place_span.0, place_span.1),
                    uninit_mpi,
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
    fn move_path_closest_to(&mut self, place: PlaceRef<'tcx>) -> (PlaceRef<'tcx>, MovePathIndex) {
        match self.move_data.rev_lookup.find(place) {
            LookupResult::Parent(Some(mpi)) | LookupResult::Exact(mpi) => {
                (self.move_data.move_paths[mpi].place.as_ref(), mpi)
            }
            LookupResult::Parent(None) => panic!("should have move path for every Local"),
        }
    }

    fn move_path_for_place(&mut self, place: PlaceRef<'tcx>) -> Option<MovePathIndex> {
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
        (place, span): (Place<'tcx>, Span),
        flow_state: &Flows<'cx, 'tcx>,
    ) {
        debug!("check_if_assigned_path_is_moved place: {:?}", place);

        // None case => assigning to `x` does not require `x` be initialized.
        for (place_base, elem) in place.iter_projections().rev() {
            match elem {
                ProjectionElem::Index(_/*operand*/) |
                ProjectionElem::OpaqueCast(_) |
                ProjectionElem::ConstantIndex { .. } |
                // assigning to P[i] requires P to be valid.
                ProjectionElem::Downcast(_/*adt_def*/, _/*variant_idx*/) =>
                // assigning to (P->variant) is okay if assigning to `P` is okay
                //
                // FIXME: is this true even if P is an adt with a dtor?
                { }

                // assigning to (*P) requires P to be initialized
                ProjectionElem::Deref => {
                    self.check_if_full_path_is_moved(
                        location, InitializationRequiringAction::Use,
                        (place_base, span), flow_state);
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
                    let base_ty = place_base.ty(self.body(), tcx).ty;
                    match base_ty.kind() {
                        ty::Adt(def, _) if def.has_dtor(tcx) => {
                            self.check_if_path_or_subpath_is_moved(
                                location, InitializationRequiringAction::Assignment,
                                (place_base, span), flow_state);

                            // (base initialized; no need to
                            // recur further)
                            break;
                        }

                        // Once `let s; s.x = V; read(s.x);`,
                        // is allowed, remove this match arm.
                        ty::Adt(..) | ty::Tuple(..) => {
                            check_parent_of_field(self, location, place_base, span, flow_state);
                        }

                        _ => {}
                    }
                }
            }
        }

        fn check_parent_of_field<'cx, 'tcx>(
            this: &mut MirBorrowckCtxt<'cx, 'tcx>,
            location: Location,
            base: PlaceRef<'tcx>,
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
                let Some(mpi) = this.move_path_for_place(prefix) else { continue };

                if maybe_uninits.contains(mpi) {
                    debug!(
                        "check_parent_of_field updating shortest_uninit_seen from {:?} to {:?}",
                        shortest_uninit_seen,
                        Some((prefix, mpi))
                    );
                    shortest_uninit_seen = Some((prefix, mpi));
                } else {
                    debug!("check_parent_of_field {:?} is definitely initialized", (prefix, mpi));
                }
            }

            if let Some((prefix, mpi)) = shortest_uninit_seen {
                // Check for a reassignment into an uninitialized field of a union (for example,
                // after a move out). In this case, do not report an error here. There is an
                // exception, if this is the first assignment into the union (that is, there is
                // no move out from an earlier location) then this is an attempt at initialization
                // of the union - we should error in that case.
                let tcx = this.infcx.tcx;
                if base.ty(this.body(), tcx).ty.is_union() {
                    if this.move_data.path_map[mpi].iter().any(|moi| {
                        this.move_data.moves[*moi].source.is_predecessor_of(location, this.body)
                    }) {
                        return;
                    }
                }

                this.report_use_of_moved_or_uninitialized(
                    location,
                    InitializationRequiringAction::PartialAssignment,
                    (prefix, base, span),
                    mpi,
                );

                // rust-lang/rust#21232, #54499, #54986: during period where we reject
                // partial initialization, do not complain about unnecessary `mut` on
                // an attempt to do a partial initialization.
                this.used_mut.insert(base.local);
            }
        }
    }

    /// Checks the permissions for the given place and read or write kind
    ///
    /// Returns `true` if an error is reported.
    fn check_access_permissions(
        &mut self,
        (place, span): (Place<'tcx>, Span),
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

        match kind {
            Reservation(WriteKind::MutableBorrow(
                borrow_kind @ (BorrowKind::Unique | BorrowKind::Mut { .. }),
            ))
            | Write(WriteKind::MutableBorrow(
                borrow_kind @ (BorrowKind::Unique | BorrowKind::Mut { .. }),
            )) => {
                let is_local_mutation_allowed = match borrow_kind {
                    BorrowKind::Unique => LocalMutationIsAllowed::Yes,
                    BorrowKind::Mut { .. } => is_local_mutation_allowed,
                    BorrowKind::Shared | BorrowKind::Shallow => unreachable!(),
                };
                match self.is_mutable(place.as_ref(), is_local_mutation_allowed) {
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
                match self.is_mutable(place.as_ref(), is_local_mutation_allowed) {
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

            Reservation(
                WriteKind::Move
                | WriteKind::Replace
                | WriteKind::StorageDeadOrDrop
                | WriteKind::MutableBorrow(BorrowKind::Shared)
                | WriteKind::MutableBorrow(BorrowKind::Shallow),
            )
            | Write(
                WriteKind::Move
                | WriteKind::Replace
                | WriteKind::StorageDeadOrDrop
                | WriteKind::MutableBorrow(BorrowKind::Shared)
                | WriteKind::MutableBorrow(BorrowKind::Shallow),
            ) => {
                if self.is_mutable(place.as_ref(), is_local_mutation_allowed).is_err()
                    && !self.has_buffered_errors()
                {
                    // rust-lang/rust#46908: In pure NLL mode this code path should be
                    // unreachable, but we use `delay_span_bug` because we can hit this when
                    // dereferencing a non-Copy raw pointer *and* have `-Ztreat-err-as-bug`
                    // enabled. We don't want to ICE for that case, as other errors will have
                    // been emitted (#52262).
                    self.infcx.tcx.sess.delay_span_bug(
                        span,
                        format!(
                            "Accessing `{:?}` with the kind `{:?}` shouldn't be possible",
                            place, kind,
                        ),
                    );
                }
                return false;
            }
            Activation(..) => {
                // permission checks are done at Reservation point.
                return false;
            }
            Read(
                ReadKind::Borrow(
                    BorrowKind::Unique
                    | BorrowKind::Mut { .. }
                    | BorrowKind::Shared
                    | BorrowKind::Shallow,
                )
                | ReadKind::Copy,
            ) => {
                // Access authorized
                return false;
            }
        }

        // rust-lang/rust#21232, #54986: during period where we reject
        // partial initialization, do not complain about mutability
        // errors except for actual mutation (as opposed to an attempt
        // to do a partial initialization).
        let previously_initialized = self.is_local_ever_initialized(place.local, flow_state);

        // at this point, we have set up the error reporting state.
        if let Some(init_index) = previously_initialized {
            if let (AccessKind::Mutate, Some(_)) = (error_access, place.as_local()) {
                // If this is a mutate access to an immutable local variable with no projections
                // report the error as an illegal reassignment
                let init = &self.move_data.inits[init_index];
                let assigned_span = init.span(&self.body);
                self.report_illegal_reassignment(location, (place, span), assigned_span, place);
            } else {
                self.report_mutability_error(place, span, the_place_err, error_access, location)
            }
            true
        } else {
            false
        }
    }

    fn is_local_ever_initialized(
        &self,
        local: Local,
        flow_state: &Flows<'cx, 'tcx>,
    ) -> Option<InitIndex> {
        let mpi = self.move_data.rev_lookup.find_local(local);
        let ii = &self.move_data.init_path_map[mpi];
        ii.into_iter().find(|&&index| flow_state.ever_inits.contains(index)).copied()
    }

    /// Adds the place into the used mutable variables set
    fn add_used_mut(&mut self, root_place: RootPlace<'tcx>, flow_state: &Flows<'cx, 'tcx>) {
        match root_place {
            RootPlace { place_local: local, place_projection: [], is_local_mutation_allowed } => {
                // If the local may have been initialized, and it is now currently being
                // mutated, then it is justified to be annotated with the `mut`
                // keyword, since the mutation may be a possible reassignment.
                if is_local_mutation_allowed != LocalMutationIsAllowed::Yes
                    && self.is_local_ever_initialized(local, flow_state).is_some()
                {
                    self.used_mut.insert(local);
                }
            }
            RootPlace {
                place_local: _,
                place_projection: _,
                is_local_mutation_allowed: LocalMutationIsAllowed::Yes,
            } => {}
            RootPlace {
                place_local,
                place_projection: place_projection @ [.., _],
                is_local_mutation_allowed: _,
            } => {
                if let Some(field) = self.is_upvar_field_projection(PlaceRef {
                    local: place_local,
                    projection: place_projection,
                }) {
                    self.used_mut_upvars.push(field);
                }
            }
        }
    }

    /// Whether this value can be written or borrowed mutably.
    /// Returns the root place if the place passed in is a projection.
    fn is_mutable(
        &self,
        place: PlaceRef<'tcx>,
        is_local_mutation_allowed: LocalMutationIsAllowed,
    ) -> Result<RootPlace<'tcx>, PlaceRef<'tcx>> {
        debug!("is_mutable: place={:?}, is_local...={:?}", place, is_local_mutation_allowed);
        match place.last_projection() {
            None => {
                let local = &self.body.local_decls[place.local];
                match local.mutability {
                    Mutability::Not => match is_local_mutation_allowed {
                        LocalMutationIsAllowed::Yes => Ok(RootPlace {
                            place_local: place.local,
                            place_projection: place.projection,
                            is_local_mutation_allowed: LocalMutationIsAllowed::Yes,
                        }),
                        LocalMutationIsAllowed::ExceptUpvars => Ok(RootPlace {
                            place_local: place.local,
                            place_projection: place.projection,
                            is_local_mutation_allowed: LocalMutationIsAllowed::ExceptUpvars,
                        }),
                        LocalMutationIsAllowed::No => Err(place),
                    },
                    Mutability::Mut => Ok(RootPlace {
                        place_local: place.local,
                        place_projection: place.projection,
                        is_local_mutation_allowed,
                    }),
                }
            }
            Some((place_base, elem)) => {
                match elem {
                    ProjectionElem::Deref => {
                        let base_ty = place_base.ty(self.body(), self.infcx.tcx).ty;

                        // Check the kind of deref to decide
                        match base_ty.kind() {
                            ty::Ref(_, _, mutbl) => {
                                match mutbl {
                                    // Shared borrowed data is never mutable
                                    hir::Mutability::Not => Err(place),
                                    // Mutably borrowed data is mutable, but only if we have a
                                    // unique path to the `&mut`
                                    hir::Mutability::Mut => {
                                        let mode = match self.is_upvar_field_projection(place) {
                                            Some(field) if self.upvars[field.index()].by_ref => {
                                                is_local_mutation_allowed
                                            }
                                            _ => LocalMutationIsAllowed::Yes,
                                        };

                                        self.is_mutable(place_base, mode)
                                    }
                                }
                            }
                            ty::RawPtr(tnm) => {
                                match tnm.mutbl {
                                    // `*const` raw pointers are not mutable
                                    hir::Mutability::Not => Err(place),
                                    // `*mut` raw pointers are always mutable, regardless of
                                    // context. The users have to check by themselves.
                                    hir::Mutability::Mut => Ok(RootPlace {
                                        place_local: place.local,
                                        place_projection: place.projection,
                                        is_local_mutation_allowed,
                                    }),
                                }
                            }
                            // `Box<T>` owns its content, so mutable if its location is mutable
                            _ if base_ty.is_box() => {
                                self.is_mutable(place_base, is_local_mutation_allowed)
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
                    | ProjectionElem::OpaqueCast { .. }
                    | ProjectionElem::Downcast(..) => {
                        let upvar_field_projection = self.is_upvar_field_projection(place);
                        if let Some(field) = upvar_field_projection {
                            let upvar = &self.upvars[field.index()];
                            debug!(
                                "is_mutable: upvar.mutability={:?} local_mutation_is_allowed={:?} \
                                 place={:?}, place_base={:?}",
                                upvar, is_local_mutation_allowed, place, place_base
                            );
                            match (upvar.place.mutability, is_local_mutation_allowed) {
                                (
                                    Mutability::Not,
                                    LocalMutationIsAllowed::No
                                    | LocalMutationIsAllowed::ExceptUpvars,
                                ) => Err(place),
                                (Mutability::Not, LocalMutationIsAllowed::Yes)
                                | (Mutability::Mut, _) => {
                                    // Subtle: this is an upvar
                                    // reference, so it looks like
                                    // `self.foo` -- we want to double
                                    // check that the location `*self`
                                    // is mutable (i.e., this is not a
                                    // `Fn` closure). But if that
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
                                    let _ =
                                        self.is_mutable(place_base, is_local_mutation_allowed)?;
                                    Ok(RootPlace {
                                        place_local: place.local,
                                        place_projection: place.projection,
                                        is_local_mutation_allowed,
                                    })
                                }
                            }
                        } else {
                            self.is_mutable(place_base, is_local_mutation_allowed)
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
    fn is_upvar_field_projection(&self, place_ref: PlaceRef<'tcx>) -> Option<FieldIdx> {
        path_utils::is_upvar_field_projection(self.infcx.tcx, &self.upvars, place_ref, self.body())
    }

    fn dominators(&self) -> &Dominators<BasicBlock> {
        // `BasicBlocks` computes dominators on-demand and caches them.
        self.body.basic_blocks.dominators()
    }
}

mod error {
    use rustc_errors::ErrorGuaranteed;

    use super::*;

    pub struct BorrowckErrors<'tcx> {
        tcx: TyCtxt<'tcx>,
        /// This field keeps track of move errors that are to be reported for given move indices.
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
        buffered_move_errors:
            BTreeMap<Vec<MoveOutIndex>, (PlaceRef<'tcx>, DiagnosticBuilder<'tcx, ErrorGuaranteed>)>,
        buffered_mut_errors: FxIndexMap<Span, (DiagnosticBuilder<'tcx, ErrorGuaranteed>, usize)>,
        /// Diagnostics to be reported buffer.
        buffered: Vec<Diagnostic>,
        /// Set to Some if we emit an error during borrowck
        tainted_by_errors: Option<ErrorGuaranteed>,
    }

    impl<'tcx> BorrowckErrors<'tcx> {
        pub fn new(tcx: TyCtxt<'tcx>) -> Self {
            BorrowckErrors {
                tcx,
                buffered_move_errors: BTreeMap::new(),
                buffered_mut_errors: Default::default(),
                buffered: Default::default(),
                tainted_by_errors: None,
            }
        }

        pub fn buffer_error(&mut self, t: DiagnosticBuilder<'_, ErrorGuaranteed>) {
            if let None = self.tainted_by_errors {
                self.tainted_by_errors = Some(
                    self.tcx
                        .sess
                        .delay_span_bug(t.span.clone(), "diagnostic buffered but not emitted"),
                )
            }
            t.buffer(&mut self.buffered);
        }

        pub fn buffer_non_error_diag(&mut self, t: DiagnosticBuilder<'_, ()>) {
            t.buffer(&mut self.buffered);
        }

        pub fn set_tainted_by_errors(&mut self, e: ErrorGuaranteed) {
            self.tainted_by_errors = Some(e);
        }
    }

    impl<'cx, 'tcx> MirBorrowckCtxt<'cx, 'tcx> {
        pub fn buffer_error(&mut self, t: DiagnosticBuilder<'_, ErrorGuaranteed>) {
            self.errors.buffer_error(t);
        }

        pub fn buffer_non_error_diag(&mut self, t: DiagnosticBuilder<'_, ()>) {
            self.errors.buffer_non_error_diag(t);
        }

        pub fn buffer_move_error(
            &mut self,
            move_out_indices: Vec<MoveOutIndex>,
            place_and_err: (PlaceRef<'tcx>, DiagnosticBuilder<'tcx, ErrorGuaranteed>),
        ) -> bool {
            if let Some((_, diag)) =
                self.errors.buffered_move_errors.insert(move_out_indices, place_and_err)
            {
                // Cancel the old diagnostic so we don't ICE
                diag.cancel();
                false
            } else {
                true
            }
        }

        pub fn get_buffered_mut_error(
            &mut self,
            span: Span,
        ) -> Option<(DiagnosticBuilder<'tcx, ErrorGuaranteed>, usize)> {
            self.errors.buffered_mut_errors.remove(&span)
        }

        pub fn buffer_mut_error(
            &mut self,
            span: Span,
            t: DiagnosticBuilder<'tcx, ErrorGuaranteed>,
            count: usize,
        ) {
            self.errors.buffered_mut_errors.insert(span, (t, count));
        }

        pub fn emit_errors(&mut self) -> Option<ErrorGuaranteed> {
            // Buffer any move errors that we collected and de-duplicated.
            for (_, (_, diag)) in std::mem::take(&mut self.errors.buffered_move_errors) {
                // We have already set tainted for this error, so just buffer it.
                diag.buffer(&mut self.errors.buffered);
            }
            for (_, (mut diag, count)) in std::mem::take(&mut self.errors.buffered_mut_errors) {
                if count > 10 {
                    diag.note(format!("...and {} other attempted mutable borrows", count - 10));
                }
                diag.buffer(&mut self.errors.buffered);
            }

            if !self.errors.buffered.is_empty() {
                self.errors.buffered.sort_by_key(|diag| diag.sort_span);

                for mut diag in self.errors.buffered.drain(..) {
                    self.infcx.tcx.sess.diagnostic().emit_diagnostic(&mut diag);
                }
            }

            self.errors.tainted_by_errors
        }

        pub fn has_buffered_errors(&self) -> bool {
            self.errors.buffered.is_empty()
        }

        pub fn has_move_error(
            &self,
            move_out_indices: &[MoveOutIndex],
        ) -> Option<&(PlaceRef<'tcx>, DiagnosticBuilder<'cx, ErrorGuaranteed>)> {
            self.errors.buffered_move_errors.get(move_out_indices)
        }
    }
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
