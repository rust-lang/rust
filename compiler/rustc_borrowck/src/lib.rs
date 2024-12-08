//! This query borrow-checks the MIR to (further) ensure it is not broken.

// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(rust_logo)]
#![feature(assert_matches)]
#![feature(box_patterns)]
#![feature(file_buffered)]
#![feature(let_chains)]
#![feature(never_type)]
#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
#![feature(stmt_expr_attributes)]
#![feature(try_blocks)]
#![warn(unreachable_pub)]
// tidy-alphabetical-end

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::ops::Deref;

use rustc_abi::FieldIdx;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::graph::dominators::Dominators;
use rustc_errors::Diag;
use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_index::bit_set::{BitSet, ChunkedBitSet};
use rustc_index::{IndexSlice, IndexVec};
use rustc_infer::infer::{
    InferCtxt, NllRegionVariableOrigin, RegionVariableOrigin, TyCtxtInferExt,
};
use rustc_middle::mir::tcx::PlaceTy;
use rustc_middle::mir::*;
use rustc_middle::query::Providers;
use rustc_middle::ty::fold::fold_regions;
use rustc_middle::ty::{self, ParamEnv, RegionVid, TyCtxt, TypingMode};
use rustc_middle::{bug, span_bug};
use rustc_mir_dataflow::impls::{
    EverInitializedPlaces, MaybeInitializedPlaces, MaybeUninitializedPlaces,
};
use rustc_mir_dataflow::move_paths::{
    InitIndex, InitLocation, LookupResult, MoveData, MoveOutIndex, MovePathIndex,
};
use rustc_mir_dataflow::{Analysis, EntrySets, Results, ResultsVisitor, visit_results};
use rustc_session::lint::builtin::UNUSED_MUT;
use rustc_span::{Span, Symbol};
use smallvec::SmallVec;
use tracing::{debug, instrument};

use crate::borrow_set::{BorrowData, BorrowSet};
use crate::consumers::{BodyWithBorrowckFacts, ConsumerOptions};
use crate::dataflow::{BorrowIndex, Borrowck, BorrowckDomain, Borrows};
use crate::diagnostics::{AccessKind, IllegalMoveOriginKind, MoveError, RegionName};
use crate::location::LocationTable;
use crate::nll::PoloniusOutput;
use crate::path_utils::*;
use crate::place_ext::PlaceExt;
use crate::places_conflict::{PlaceConflictBias, places_conflict};
use crate::prefixes::PrefixSet;
use crate::region_infer::RegionInferenceContext;
use crate::renumber::RegionCtxt;
use crate::session_diagnostics::VarNeedNotMut;

mod borrow_set;
mod borrowck_errors;
mod constraints;
mod dataflow;
mod def_use;
mod diagnostics;
mod facts;
mod location;
mod member_constraints;
mod nll;
mod path_utils;
mod place_ext;
mod places_conflict;
mod polonius;
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

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

/// Associate some local constants with the `'tcx` lifetime
struct TyCtxtConsts<'tcx>(PhantomData<&'tcx ()>);

impl<'tcx> TyCtxtConsts<'tcx> {
    const DEREF_PROJECTION: &'tcx [PlaceElem<'tcx>; 1] = &[ProjectionElem::Deref];
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { mir_borrowck, ..*providers };
}

fn mir_borrowck(tcx: TyCtxt<'_>, def: LocalDefId) -> &BorrowCheckResult<'_> {
    let (input_body, promoted) = tcx.mir_promoted(def);
    debug!("run query mir_borrowck: {}", tcx.def_path_str(def));

    let input_body: &Body<'_> = &input_body.borrow();

    if input_body.should_skip() || input_body.tainted_by_errors.is_some() {
        debug!("Skipping borrowck because of injected body or tainted body");
        // Let's make up a borrowck result! Fun times!
        let result = BorrowCheckResult {
            concrete_opaque_types: FxIndexMap::default(),
            closure_requirements: None,
            used_mut_upvars: SmallVec::new(),
            tainted_by_errors: input_body.tainted_by_errors,
        };
        return tcx.arena.alloc(result);
    }

    let promoted: &IndexSlice<_, _> = &promoted.borrow();
    let opt_closure_req = do_mir_borrowck(tcx, input_body, promoted, None).0;
    debug!("mir_borrowck done");

    tcx.arena.alloc(opt_closure_req)
}

/// Perform the actual borrow checking.
///
/// Use `consumer_options: None` for the default behavior of returning
/// [`BorrowCheckResult`] only. Otherwise, return [`BodyWithBorrowckFacts`] according
/// to the given [`ConsumerOptions`].
#[instrument(skip(tcx, input_body, input_promoted), fields(id=?input_body.source.def_id()), level = "debug")]
fn do_mir_borrowck<'tcx>(
    tcx: TyCtxt<'tcx>,
    input_body: &Body<'tcx>,
    input_promoted: &IndexSlice<Promoted, Body<'tcx>>,
    consumer_options: Option<ConsumerOptions>,
) -> (BorrowCheckResult<'tcx>, Option<Box<BodyWithBorrowckFacts<'tcx>>>) {
    let def = input_body.source.def_id().expect_local();
    let infcx = BorrowckInferCtxt::new(tcx, def);

    let mut local_names = IndexVec::from_elem(None, &input_body.local_decls);
    for var_debug_info in &input_body.var_debug_info {
        if let VarDebugInfoContents::Place(place) = var_debug_info.value {
            if let Some(local) = place.as_local() {
                if let Some(prev_name) = local_names[local]
                    && var_debug_info.name != prev_name
                {
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

    let diags = &mut diags::BorrowckDiags::new();

    // Gather the upvars of a closure, if any.
    if let Some(e) = input_body.tainted_by_errors {
        infcx.set_tainted_by_errors(e);
    }

    // Replace all regions with fresh inference variables. This
    // requires first making our own copy of the MIR. This copy will
    // be modified (in place) to contain non-lexical lifetimes. It
    // will have a lifetime tied to the inference context.
    let mut body_owned = input_body.clone();
    let mut promoted = input_promoted.to_owned();
    let free_regions = nll::replace_regions_in_mir(&infcx, &mut body_owned, &mut promoted);
    let body = &body_owned; // no further changes

    // FIXME(-Znext-solver): A bit dubious that we're only registering
    // predefined opaques in the typeck root.
    if infcx.next_trait_solver() && !infcx.tcx.is_typeck_child(body.source.def_id()) {
        infcx.register_predefined_opaques_for_next_solver(def);
    }

    let location_table = LocationTable::new(body);

    let move_data = MoveData::gather_moves(body, tcx, |_| true);
    let promoted_move_data = promoted
        .iter_enumerated()
        .map(|(idx, body)| (idx, MoveData::gather_moves(body, tcx, |_| true)));

    let flow_inits = MaybeInitializedPlaces::new(tcx, body, &move_data)
        .iterate_to_fixpoint(tcx, body, Some("borrowck"))
        .into_results_cursor(body);

    let locals_are_invalidated_at_exit = tcx.hir().body_owner_kind(def).is_fn_or_closure();
    let borrow_set = BorrowSet::build(tcx, body, locals_are_invalidated_at_exit, &move_data);

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
        &location_table,
        flow_inits,
        &move_data,
        &borrow_set,
        consumer_options,
    );

    // Dump MIR results into a file, if that is enabled. This let us
    // write unit-tests, as well as helping with debugging.
    nll::dump_nll_mir(&infcx, body, &regioncx, &opt_closure_req, &borrow_set);

    // We also have a `#[rustc_regions]` annotation that causes us to dump
    // information.
    nll::dump_annotation(&infcx, body, &regioncx, &opt_closure_req, &opaque_type_values, diags);

    let movable_coroutine =
        // The first argument is the coroutine type passed by value
        if let Some(local) = body.local_decls.raw.get(1)
        // Get the interior types and args which typeck computed
        && let ty::Coroutine(def_id, _) = *local.ty.kind()
        && tcx.coroutine_movability(def_id) == hir::Movability::Movable
    {
        true
    } else {
        false
    };

    for (idx, move_data) in promoted_move_data {
        use rustc_middle::mir::visit::Visitor;

        let promoted_body = &promoted[idx];
        let mut promoted_mbcx = MirBorrowckCtxt {
            infcx: &infcx,
            body: promoted_body,
            move_data: &move_data,
            location_table: &location_table, // no need to create a real one for the promoted, it is not used
            movable_coroutine,
            fn_self_span_reported: Default::default(),
            locals_are_invalidated_at_exit,
            access_place_error_reported: Default::default(),
            reservation_error_reported: Default::default(),
            uninitialized_error_reported: Default::default(),
            regioncx: &regioncx,
            used_mut: Default::default(),
            used_mut_upvars: SmallVec::new(),
            borrow_set: &borrow_set,
            upvars: &[],
            local_names: IndexVec::from_elem(None, &promoted_body.local_decls),
            region_names: RefCell::default(),
            next_region_name: RefCell::new(1),
            polonius_output: None,
            move_errors: Vec::new(),
            diags,
        };
        MoveVisitor { ctxt: &mut promoted_mbcx }.visit_body(promoted_body);
        promoted_mbcx.report_move_errors();

        struct MoveVisitor<'a, 'b, 'infcx, 'tcx> {
            ctxt: &'a mut MirBorrowckCtxt<'b, 'infcx, 'tcx>,
        }

        impl<'tcx> Visitor<'tcx> for MoveVisitor<'_, '_, '_, 'tcx> {
            fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
                if let Operand::Move(place) = operand {
                    self.ctxt.check_movable_place(location, *place);
                }
            }
        }
    }

    let mut mbcx = MirBorrowckCtxt {
        infcx: &infcx,
        body,
        move_data: &move_data,
        location_table: &location_table,
        movable_coroutine,
        locals_are_invalidated_at_exit,
        fn_self_span_reported: Default::default(),
        access_place_error_reported: Default::default(),
        reservation_error_reported: Default::default(),
        uninitialized_error_reported: Default::default(),
        regioncx: &regioncx,
        used_mut: Default::default(),
        used_mut_upvars: SmallVec::new(),
        borrow_set: &borrow_set,
        upvars: tcx.closure_captures(def),
        local_names,
        region_names: RefCell::default(),
        next_region_name: RefCell::new(1),
        polonius_output,
        move_errors: Vec::new(),
        diags,
    };

    // Compute and report region errors, if any.
    mbcx.report_region_errors(nll_errors);

    let mut flow_results = get_flow_results(tcx, body, &move_data, &borrow_set, &regioncx);
    visit_results(
        body,
        traversal::reverse_postorder(body).map(|(bb, _)| bb),
        &mut flow_results,
        &mut mbcx,
    );

    mbcx.report_move_errors();

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

        tcx.emit_node_span_lint(UNUSED_MUT, lint_root, span, VarNeedNotMut { span: mut_span })
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
            location_table: polonius_input.as_ref().map(|_| location_table),
            input_facts: polonius_input,
            output_facts,
        }))
    } else {
        None
    };

    debug!("do_mir_borrowck: result = {:#?}", result);

    (result, body_with_facts)
}

fn get_flow_results<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    move_data: &'a MoveData<'tcx>,
    borrow_set: &'a BorrowSet<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
) -> Results<'tcx, Borrowck<'a, 'tcx>> {
    // We compute these three analyses individually, but them combine them into
    // a single results so that `mbcx` can visit them all together.
    let borrows = Borrows::new(tcx, body, regioncx, borrow_set).iterate_to_fixpoint(
        tcx,
        body,
        Some("borrowck"),
    );
    let uninits = MaybeUninitializedPlaces::new(tcx, body, move_data).iterate_to_fixpoint(
        tcx,
        body,
        Some("borrowck"),
    );
    let ever_inits = EverInitializedPlaces::new(body, move_data).iterate_to_fixpoint(
        tcx,
        body,
        Some("borrowck"),
    );

    let analysis = Borrowck {
        borrows: borrows.analysis,
        uninits: uninits.analysis,
        ever_inits: ever_inits.analysis,
    };

    assert_eq!(borrows.entry_sets.len(), uninits.entry_sets.len());
    assert_eq!(borrows.entry_sets.len(), ever_inits.entry_sets.len());
    let entry_sets: EntrySets<'_, Borrowck<'_, '_>> =
        itertools::izip!(borrows.entry_sets, uninits.entry_sets, ever_inits.entry_sets)
            .map(|(borrows, uninits, ever_inits)| BorrowckDomain { borrows, uninits, ever_inits })
            .collect();

    Results { analysis, entry_sets }
}

pub(crate) struct BorrowckInferCtxt<'tcx> {
    pub(crate) infcx: InferCtxt<'tcx>,
    pub(crate) reg_var_to_origin: RefCell<FxIndexMap<ty::RegionVid, RegionCtxt>>,
    pub(crate) param_env: ParamEnv<'tcx>,
}

impl<'tcx> BorrowckInferCtxt<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> Self {
        let infcx = tcx.infer_ctxt().build(TypingMode::analysis_in_body(tcx, def_id));
        let param_env = tcx.param_env(def_id);
        BorrowckInferCtxt { infcx, reg_var_to_origin: RefCell::new(Default::default()), param_env }
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

    /// With the new solver we prepopulate the opaque type storage during
    /// MIR borrowck with the hidden types from HIR typeck. This is necessary
    /// to avoid ambiguities as earlier goals can rely on the hidden type
    /// of an opaque which is only constrained by a later goal.
    fn register_predefined_opaques_for_next_solver(&self, def_id: LocalDefId) {
        let tcx = self.tcx;
        // OK to use the identity arguments for each opaque type key, since
        // we remap opaques from HIR typeck back to their definition params.
        for data in tcx.typeck(def_id).concrete_opaque_types.iter().map(|(k, v)| (*k, *v)) {
            // HIR typeck did not infer the regions of the opaque, so we instantiate
            // them with fresh inference variables.
            let (key, hidden_ty) = fold_regions(tcx, data, |_, _| {
                self.next_nll_region_var_in_universe(
                    NllRegionVariableOrigin::Existential { from_forall: false },
                    ty::UniverseIndex::ROOT,
                )
            });

            self.inject_new_hidden_type_unchecked(key, hidden_ty);
        }
    }
}

impl<'tcx> Deref for BorrowckInferCtxt<'tcx> {
    type Target = InferCtxt<'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.infcx
    }
}

struct MirBorrowckCtxt<'a, 'infcx, 'tcx> {
    infcx: &'infcx BorrowckInferCtxt<'tcx>,
    body: &'a Body<'tcx>,
    move_data: &'a MoveData<'tcx>,

    /// Map from MIR `Location` to `LocationIndex`; created
    /// when MIR borrowck begins.
    location_table: &'a LocationTable,

    movable_coroutine: bool,
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
    uninitialized_error_reported: FxIndexSet<Local>,
    /// This field keeps track of all the local variables that are declared mut and are mutated.
    /// Used for the warning issued by an unused mutable local variable.
    used_mut: FxIndexSet<Local>,
    /// If the function we're checking is a closure, then we'll need to report back the list of
    /// mutable upvars that have been used. This field keeps track of them.
    used_mut_upvars: SmallVec<[FieldIdx; 8]>,
    /// Region inference context. This contains the results from region inference and lets us e.g.
    /// find out which CFG points are contained in each borrow region.
    regioncx: &'a RegionInferenceContext<'tcx>,

    /// The set of borrows extracted from the MIR
    borrow_set: &'a BorrowSet<'tcx>,

    /// Information about upvars not necessarily preserved in types or MIR
    upvars: &'tcx [&'tcx ty::CapturedPlace<'tcx>],

    /// Names of local (user) variables (extracted from `var_debug_info`).
    local_names: IndexVec<Local, Option<Symbol>>,

    /// Record the region names generated for each region in the given
    /// MIR def so that we can reuse them later in help/error messages.
    region_names: RefCell<FxIndexMap<RegionVid, RegionName>>,

    /// The counter for generating new region names.
    next_region_name: RefCell<usize>,

    /// Results of Polonius analysis.
    polonius_output: Option<Box<PoloniusOutput>>,

    diags: &'a mut diags::BorrowckDiags<'infcx, 'tcx>,
    move_errors: Vec<MoveError<'tcx>>,
}

// Check that:
// 1. assignments are always made to mutable locations (FIXME: does that still really go here?)
// 2. loans made in overlapping scopes do not conflict
// 3. assignments do not affect things loaned out as immutable
// 4. moves do not affect things loaned out in any way
impl<'a, 'tcx> ResultsVisitor<'a, 'tcx, Borrowck<'a, 'tcx>> for MirBorrowckCtxt<'a, '_, 'tcx> {
    fn visit_statement_before_primary_effect(
        &mut self,
        _results: &mut Results<'tcx, Borrowck<'a, 'tcx>>,
        state: &BorrowckDomain<'a, 'tcx>,
        stmt: &'a Statement<'tcx>,
        location: Location,
    ) {
        debug!("MirBorrowckCtxt::process_statement({:?}, {:?}): {:?}", location, stmt, state);
        let span = stmt.source_info.span;

        self.check_activations(location, span, state);

        match &stmt.kind {
            StatementKind::Assign(box (lhs, rhs)) => {
                self.consume_rvalue(location, (rhs, span), state);

                self.mutate_place(location, (*lhs, span), Shallow(None), state);
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
                    state,
                );
            }
            StatementKind::Intrinsic(box kind) => match kind {
                NonDivergingIntrinsic::Assume(op) => {
                    self.consume_operand(location, (op, span), state);
                }
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
            // This do not affect borrowck
            | StatementKind::BackwardIncompatibleDropHint { .. }
            | StatementKind::StorageLive(..) => {}
            StatementKind::StorageDead(local) => {
                self.access_place(
                    location,
                    (Place::from(*local), span),
                    (Shallow(None), Write(WriteKind::StorageDeadOrDrop)),
                    LocalMutationIsAllowed::Yes,
                    state,
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
        _results: &mut Results<'tcx, Borrowck<'a, 'tcx>>,
        state: &BorrowckDomain<'a, 'tcx>,
        term: &'a Terminator<'tcx>,
        loc: Location,
    ) {
        debug!("MirBorrowckCtxt::process_terminator({:?}, {:?}): {:?}", loc, term, state);
        let span = term.source_info.span;

        self.check_activations(loc, span, state);

        match &term.kind {
            TerminatorKind::SwitchInt { discr, targets: _ } => {
                self.consume_operand(loc, (discr, span), state);
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
                    state,
                );
            }
            TerminatorKind::Call {
                func,
                args,
                destination,
                target: _,
                unwind: _,
                call_source: _,
                fn_span: _,
            } => {
                self.consume_operand(loc, (func, span), state);
                for arg in args {
                    self.consume_operand(loc, (&arg.node, arg.span), state);
                }
                self.mutate_place(loc, (*destination, span), Deep, state);
            }
            TerminatorKind::TailCall { func, args, fn_span: _ } => {
                self.consume_operand(loc, (func, span), state);
                for arg in args {
                    self.consume_operand(loc, (&arg.node, arg.span), state);
                }
            }
            TerminatorKind::Assert { cond, expected: _, msg, target: _, unwind: _ } => {
                self.consume_operand(loc, (cond, span), state);
                if let AssertKind::BoundsCheck { len, index } = &**msg {
                    self.consume_operand(loc, (len, span), state);
                    self.consume_operand(loc, (index, span), state);
                }
            }

            TerminatorKind::Yield { value, resume: _, resume_arg, drop: _ } => {
                self.consume_operand(loc, (value, span), state);
                self.mutate_place(loc, (*resume_arg, span), Deep, state);
            }

            TerminatorKind::InlineAsm {
                asm_macro: _,
                template: _,
                operands,
                options: _,
                line_spans: _,
                targets: _,
                unwind: _,
            } => {
                for op in operands {
                    match op {
                        InlineAsmOperand::In { reg: _, value } => {
                            self.consume_operand(loc, (value, span), state);
                        }
                        InlineAsmOperand::Out { reg: _, late: _, place, .. } => {
                            if let Some(place) = place {
                                self.mutate_place(loc, (*place, span), Shallow(None), state);
                            }
                        }
                        InlineAsmOperand::InOut { reg: _, late: _, in_value, out_place } => {
                            self.consume_operand(loc, (in_value, span), state);
                            if let &Some(out_place) = out_place {
                                self.mutate_place(loc, (out_place, span), Shallow(None), state);
                            }
                        }
                        InlineAsmOperand::Const { value: _ }
                        | InlineAsmOperand::SymFn { value: _ }
                        | InlineAsmOperand::SymStatic { def_id: _ }
                        | InlineAsmOperand::Label { target_index: _ } => {}
                    }
                }
            }

            TerminatorKind::Goto { target: _ }
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Unreachable
            | TerminatorKind::UnwindResume
            | TerminatorKind::Return
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::FalseEdge { real_target: _, imaginary_target: _ }
            | TerminatorKind::FalseUnwind { real_target: _, unwind: _ } => {
                // no data used, thus irrelevant to borrowck
            }
        }
    }

    fn visit_terminator_after_primary_effect(
        &mut self,
        _results: &mut Results<'tcx, Borrowck<'a, 'tcx>>,
        state: &BorrowckDomain<'a, 'tcx>,
        term: &'a Terminator<'tcx>,
        loc: Location,
    ) {
        let span = term.source_info.span;

        match term.kind {
            TerminatorKind::Yield { value: _, resume: _, resume_arg: _, drop: _ } => {
                if self.movable_coroutine {
                    // Look for any active borrows to locals
                    for i in state.borrows.iter() {
                        let borrow = &self.borrow_set[i];
                        self.check_for_local_borrow(borrow, span);
                    }
                }
            }

            TerminatorKind::UnwindResume
            | TerminatorKind::Return
            | TerminatorKind::TailCall { .. }
            | TerminatorKind::CoroutineDrop => {
                // Returning from the function implicitly kills storage for all locals and statics.
                // Often, the storage will already have been killed by an explicit
                // StorageDead, but we don't always emit those (notably on unwind paths),
                // so this "extra check" serves as a kind of backup.
                for i in state.borrows.iter() {
                    let borrow = &self.borrow_set[i];
                    self.check_for_invalidation_at_exit(loc, borrow, span);
                }
            }

            TerminatorKind::UnwindTerminate(_)
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
    FakeBorrow,
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

impl<'a, 'tcx> MirBorrowckCtxt<'a, '_, 'tcx> {
    fn body(&self) -> &'a Body<'tcx> {
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
        state: &BorrowckDomain<'a, 'tcx>,
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
            state,
            location,
        );
        let conflict_error = self.check_access_for_conflict(location, place_span, sd, rw, state);

        if conflict_error || mutability_error {
            debug!("access_place: logging error place_span=`{:?}` kind=`{:?}`", place_span, kind);
            self.access_place_error_reported.insert((place_span.0, place_span.1));
        }
    }

    #[instrument(level = "debug", skip(self, state))]
    fn check_access_for_conflict(
        &mut self,
        location: Location,
        place_span: (Place<'tcx>, Span),
        sd: AccessDepth,
        rw: ReadOrWrite,
        state: &BorrowckDomain<'a, 'tcx>,
    ) -> bool {
        let mut error_reported = false;

        // Use polonius output if it has been enabled.
        let mut polonius_output;
        let borrows_in_scope = if let Some(polonius) = &self.polonius_output {
            let location = self.location_table.start_index(location);
            polonius_output = BitSet::new_empty(self.borrow_set.len());
            for &idx in polonius.errors_at(location) {
                polonius_output.insert(idx);
            }
            &polonius_output
        } else {
            &state.borrows
        };

        each_borrow_involving_path(
            self,
            self.infcx.tcx,
            self.body,
            (sd, place_span.0),
            self.borrow_set,
            |borrow_index| borrows_in_scope.contains(borrow_index),
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

                (Read(_), BorrowKind::Shared | BorrowKind::Fake(_))
                | (
                    Read(ReadKind::Borrow(BorrowKind::Fake(FakeBorrowKind::Shallow))),
                    BorrowKind::Mut { .. },
                ) => Control::Continue,

                (Reservation(_), BorrowKind::Fake(_) | BorrowKind::Shared) => {
                    // This used to be a future compatibility warning (to be
                    // disallowed on NLL). See rust-lang/rust#56254
                    Control::Continue
                }

                (Write(WriteKind::Move), BorrowKind::Fake(FakeBorrowKind::Shallow)) => {
                    // Handled by initialization checks.
                    Control::Continue
                }

                (Read(kind), BorrowKind::Mut { .. }) => {
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
        state: &BorrowckDomain<'a, 'tcx>,
    ) {
        // Write of P[i] or *P requires P init'd.
        self.check_if_assigned_path_is_moved(location, place_span, state);

        self.access_place(
            location,
            place_span,
            (kind, Write(WriteKind::Mutate)),
            LocalMutationIsAllowed::No,
            state,
        );
    }

    fn consume_rvalue(
        &mut self,
        location: Location,
        (rvalue, span): (&'a Rvalue<'tcx>, Span),
        state: &BorrowckDomain<'a, 'tcx>,
    ) {
        match rvalue {
            &Rvalue::Ref(_ /*rgn*/, bk, place) => {
                let access_kind = match bk {
                    BorrowKind::Fake(FakeBorrowKind::Shallow) => {
                        (Shallow(Some(ArtificialField::FakeBorrow)), Read(ReadKind::Borrow(bk)))
                    }
                    BorrowKind::Shared | BorrowKind::Fake(FakeBorrowKind::Deep) => {
                        (Deep, Read(ReadKind::Borrow(bk)))
                    }
                    BorrowKind::Mut { .. } => {
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
                    state,
                );

                let action = if bk == BorrowKind::Fake(FakeBorrowKind::Shallow) {
                    InitializationRequiringAction::MatchOn
                } else {
                    InitializationRequiringAction::Borrow
                };

                self.check_if_path_or_subpath_is_moved(
                    location,
                    action,
                    (place.as_ref(), span),
                    state,
                );
            }

            &Rvalue::RawPtr(mutability, place) => {
                let access_kind = match mutability {
                    Mutability::Mut => (
                        Deep,
                        Write(WriteKind::MutableBorrow(BorrowKind::Mut {
                            kind: MutBorrowKind::Default,
                        })),
                    ),
                    Mutability::Not => (Deep, Read(ReadKind::Borrow(BorrowKind::Shared))),
                };

                self.access_place(
                    location,
                    (place, span),
                    access_kind,
                    LocalMutationIsAllowed::No,
                    state,
                );

                self.check_if_path_or_subpath_is_moved(
                    location,
                    InitializationRequiringAction::Borrow,
                    (place.as_ref(), span),
                    state,
                );
            }

            Rvalue::ThreadLocalRef(_) => {}

            Rvalue::Use(operand)
            | Rvalue::Repeat(operand, _)
            | Rvalue::UnaryOp(_ /*un_op*/, operand)
            | Rvalue::Cast(_ /*cast_kind*/, operand, _ /*ty*/)
            | Rvalue::ShallowInitBox(operand, _ /*ty*/) => {
                self.consume_operand(location, (operand, span), state)
            }

            &Rvalue::CopyForDeref(place) => {
                self.access_place(
                    location,
                    (place, span),
                    (Deep, Read(ReadKind::Copy)),
                    LocalMutationIsAllowed::No,
                    state,
                );

                // Finally, check if path was already moved.
                self.check_if_path_or_subpath_is_moved(
                    location,
                    InitializationRequiringAction::Use,
                    (place.as_ref(), span),
                    state,
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
                    state,
                );
                self.check_if_path_or_subpath_is_moved(
                    location,
                    InitializationRequiringAction::Use,
                    (place.as_ref(), span),
                    state,
                );
            }

            Rvalue::BinaryOp(_bin_op, box (operand1, operand2)) => {
                self.consume_operand(location, (operand1, span), state);
                self.consume_operand(location, (operand2, span), state);
            }

            Rvalue::NullaryOp(_op, _ty) => {
                // nullary ops take no dynamic input; no borrowck effect.
            }

            Rvalue::Aggregate(aggregate_kind, operands) => {
                // We need to report back the list of mutable upvars that were
                // moved into the closure and subsequently used by the closure,
                // in order to populate our used_mut set.
                match **aggregate_kind {
                    AggregateKind::Closure(def_id, _)
                    | AggregateKind::CoroutineClosure(def_id, _)
                    | AggregateKind::Coroutine(def_id, _) => {
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
                    | AggregateKind::Tuple { .. }
                    | AggregateKind::RawPtr(..) => (),
                }

                for operand in operands {
                    self.consume_operand(location, (operand, span), state);
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

                        let Some(temp_mpi) = self.move_data.rev_lookup.find_local(local) else {
                            bug!("temporary should be tracked");
                        };
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
        (operand, span): (&'a Operand<'tcx>, Span),
        state: &BorrowckDomain<'a, 'tcx>,
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
                    state,
                );

                // Finally, check if path was already moved.
                self.check_if_path_or_subpath_is_moved(
                    location,
                    InitializationRequiringAction::Use,
                    (place.as_ref(), span),
                    state,
                );
            }
            Operand::Move(place) => {
                // Check if moving from this place makes sense.
                self.check_movable_place(location, place);

                // move of place: check if this is move of already borrowed path
                self.access_place(
                    location,
                    (place, span),
                    (Deep, Write(WriteKind::Move)),
                    LocalMutationIsAllowed::Yes,
                    state,
                );

                // Finally, check if path was already moved.
                self.check_if_path_or_subpath_is_moved(
                    location,
                    InitializationRequiringAction::Use,
                    (place.as_ref(), span),
                    state,
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
    /// This is called for all Yield expressions on movable coroutines
    fn check_for_local_borrow(&mut self, borrow: &BorrowData<'tcx>, yield_span: Span) {
        debug!("check_for_local_borrow({:?})", borrow);

        if borrow_of_local_data(borrow.borrowed_place) {
            let err = self.cannot_borrow_across_coroutine_yield(
                self.retrieve_borrow_spans(borrow).var_or_use(),
                yield_span,
            );

            self.buffer_error(err);
        }
    }

    fn check_activations(
        &mut self,
        location: Location,
        span: Span,
        state: &BorrowckDomain<'a, 'tcx>,
    ) {
        // Two-phase borrow support: For each activation that is newly
        // generated at this statement, check if it interferes with
        // another borrow.
        for &borrow_index in self.borrow_set.activations_at_location(location) {
            let borrow = &self.borrow_set[borrow_index];

            // only mutable borrows should be 2-phase
            assert!(match borrow.kind {
                BorrowKind::Shared | BorrowKind::Fake(_) => false,
                BorrowKind::Mut { .. } => true,
            });

            self.access_place(
                location,
                (borrow.borrowed_place, span),
                (Deep, Activation(WriteKind::MutableBorrow(borrow.kind), borrow_index)),
                LocalMutationIsAllowed::No,
                state,
            );
            // We do not need to call `check_if_path_or_subpath_is_moved`
            // again, as we already called it when we made the
            // initial reservation.
        }
    }

    fn check_movable_place(&mut self, location: Location, place: Place<'tcx>) {
        use IllegalMoveOriginKind::*;

        let body = self.body;
        let tcx = self.infcx.tcx;
        let mut place_ty = PlaceTy::from_ty(body.local_decls[place.local].ty);
        for (place_ref, elem) in place.iter_projections() {
            match elem {
                ProjectionElem::Deref => match place_ty.ty.kind() {
                    ty::Ref(..) | ty::RawPtr(..) => {
                        self.move_errors.push(MoveError::new(place, location, BorrowedContent {
                            target_place: place_ref.project_deeper(&[elem], tcx),
                        }));
                        return;
                    }
                    ty::Adt(adt, _) => {
                        if !adt.is_box() {
                            bug!("Adt should be a box type when Place is deref");
                        }
                    }
                    ty::Bool
                    | ty::Char
                    | ty::Int(_)
                    | ty::Uint(_)
                    | ty::Float(_)
                    | ty::Foreign(_)
                    | ty::Str
                    | ty::Array(_, _)
                    | ty::Pat(_, _)
                    | ty::Slice(_)
                    | ty::FnDef(_, _)
                    | ty::FnPtr(..)
                    | ty::Dynamic(_, _, _)
                    | ty::Closure(_, _)
                    | ty::CoroutineClosure(_, _)
                    | ty::Coroutine(_, _)
                    | ty::CoroutineWitness(..)
                    | ty::Never
                    | ty::Tuple(_)
                    | ty::Alias(_, _)
                    | ty::Param(_)
                    | ty::Bound(_, _)
                    | ty::Infer(_)
                    | ty::Error(_)
                    | ty::Placeholder(_) => {
                        bug!("When Place is Deref it's type shouldn't be {place_ty:#?}")
                    }
                },
                ProjectionElem::Field(_, _) => match place_ty.ty.kind() {
                    ty::Adt(adt, _) => {
                        if adt.has_dtor(tcx) {
                            self.move_errors.push(MoveError::new(
                                place,
                                location,
                                InteriorOfTypeWithDestructor { container_ty: place_ty.ty },
                            ));
                            return;
                        }
                    }
                    ty::Closure(..)
                    | ty::CoroutineClosure(..)
                    | ty::Coroutine(_, _)
                    | ty::Tuple(_) => (),
                    ty::Bool
                    | ty::Char
                    | ty::Int(_)
                    | ty::Uint(_)
                    | ty::Float(_)
                    | ty::Foreign(_)
                    | ty::Str
                    | ty::Array(_, _)
                    | ty::Pat(_, _)
                    | ty::Slice(_)
                    | ty::RawPtr(_, _)
                    | ty::Ref(_, _, _)
                    | ty::FnDef(_, _)
                    | ty::FnPtr(..)
                    | ty::Dynamic(_, _, _)
                    | ty::CoroutineWitness(..)
                    | ty::Never
                    | ty::Alias(_, _)
                    | ty::Param(_)
                    | ty::Bound(_, _)
                    | ty::Infer(_)
                    | ty::Error(_)
                    | ty::Placeholder(_) => bug!(
                        "When Place contains ProjectionElem::Field it's type shouldn't be {place_ty:#?}"
                    ),
                },
                ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. } => {
                    match place_ty.ty.kind() {
                        ty::Slice(_) => {
                            self.move_errors.push(MoveError::new(
                                place,
                                location,
                                InteriorOfSliceOrArray { ty: place_ty.ty, is_index: false },
                            ));
                            return;
                        }
                        ty::Array(_, _) => (),
                        _ => bug!("Unexpected type {:#?}", place_ty.ty),
                    }
                }
                ProjectionElem::Index(_) => match place_ty.ty.kind() {
                    ty::Array(..) | ty::Slice(..) => {
                        self.move_errors.push(MoveError::new(
                            place,
                            location,
                            InteriorOfSliceOrArray { ty: place_ty.ty, is_index: true },
                        ));
                        return;
                    }
                    _ => bug!("Unexpected type {place_ty:#?}"),
                },
                // `OpaqueCast`: only transmutes the type, so no moves there.
                // `Downcast`  : only changes information about a `Place` without moving.
                // `Subtype`   : only transmutes the type, so no moves.
                // So it's safe to skip these.
                ProjectionElem::OpaqueCast(_)
                | ProjectionElem::Subtype(_)
                | ProjectionElem::Downcast(_, _) => (),
            }

            place_ty = place_ty.projection_ty(tcx, elem);
        }
    }

    fn check_if_full_path_is_moved(
        &mut self,
        location: Location,
        desired_action: InitializationRequiringAction,
        place_span: (PlaceRef<'tcx>, Span),
        state: &BorrowckDomain<'a, 'tcx>,
    ) {
        let maybe_uninits = &state.uninits;

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
        state: &BorrowckDomain<'a, 'tcx>,
    ) {
        let maybe_uninits = &state.uninits;

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

        self.check_if_full_path_is_moved(location, desired_action, place_span, state);

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
        state: &BorrowckDomain<'a, 'tcx>,
    ) {
        debug!("check_if_assigned_path_is_moved place: {:?}", place);

        // None case => assigning to `x` does not require `x` be initialized.
        for (place_base, elem) in place.iter_projections().rev() {
            match elem {
                ProjectionElem::Index(_/*operand*/) |
                ProjectionElem::Subtype(_) |
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
                        (place_base, span), state);
                    // (base initialized; no need to
                    // recur further)
                    break;
                }

                ProjectionElem::Subslice { .. } => {
                    panic!("we don't allow assignments to subslices, location: {location:?}");
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
                                (place_base, span), state);

                            // (base initialized; no need to
                            // recur further)
                            break;
                        }

                        // Once `let s; s.x = V; read(s.x);`,
                        // is allowed, remove this match arm.
                        ty::Adt(..) | ty::Tuple(..) => {
                            check_parent_of_field(self, location, place_base, span, state);
                        }

                        _ => {}
                    }
                }
            }
        }

        fn check_parent_of_field<'a, 'tcx>(
            this: &mut MirBorrowckCtxt<'a, '_, 'tcx>,
            location: Location,
            base: PlaceRef<'tcx>,
            span: Span,
            state: &BorrowckDomain<'a, 'tcx>,
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
            let maybe_uninits = &state.uninits;

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
                if base.ty(this.body(), tcx).ty.is_union()
                    && this.move_data.path_map[mpi].iter().any(|moi| {
                        this.move_data.moves[*moi].source.is_predecessor_of(location, this.body)
                    })
                {
                    return;
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
        state: &BorrowckDomain<'a, 'tcx>,
        location: Location,
    ) -> bool {
        debug!(
            "check_access_permissions({:?}, {:?}, is_local_mutation_allowed: {:?})",
            place, kind, is_local_mutation_allowed
        );

        let error_access;
        let the_place_err;

        match kind {
            Reservation(WriteKind::MutableBorrow(BorrowKind::Mut { kind: mut_borrow_kind }))
            | Write(WriteKind::MutableBorrow(BorrowKind::Mut { kind: mut_borrow_kind })) => {
                let is_local_mutation_allowed = match mut_borrow_kind {
                    // `ClosureCapture` is used for mutable variable with an immutable binding.
                    // This is only behaviour difference between `ClosureCapture` and mutable
                    // borrows.
                    MutBorrowKind::ClosureCapture => LocalMutationIsAllowed::Yes,
                    MutBorrowKind::Default | MutBorrowKind::TwoPhaseBorrow => {
                        is_local_mutation_allowed
                    }
                };
                match self.is_mutable(place.as_ref(), is_local_mutation_allowed) {
                    Ok(root_place) => {
                        self.add_used_mut(root_place, state);
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
                        self.add_used_mut(root_place, state);
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
                | WriteKind::MutableBorrow(BorrowKind::Fake(_)),
            )
            | Write(
                WriteKind::Move
                | WriteKind::Replace
                | WriteKind::StorageDeadOrDrop
                | WriteKind::MutableBorrow(BorrowKind::Shared)
                | WriteKind::MutableBorrow(BorrowKind::Fake(_)),
            ) => {
                if self.is_mutable(place.as_ref(), is_local_mutation_allowed).is_err()
                    && !self.has_buffered_diags()
                {
                    // rust-lang/rust#46908: In pure NLL mode this code path should be
                    // unreachable, but we use `span_delayed_bug` because we can hit this when
                    // dereferencing a non-Copy raw pointer *and* have `-Ztreat-err-as-bug`
                    // enabled. We don't want to ICE for that case, as other errors will have
                    // been emitted (#52262).
                    self.dcx().span_delayed_bug(
                        span,
                        format!(
                            "Accessing `{place:?}` with the kind `{kind:?}` shouldn't be possible",
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
                ReadKind::Borrow(BorrowKind::Mut { .. } | BorrowKind::Shared | BorrowKind::Fake(_))
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
        let previously_initialized = self.is_local_ever_initialized(place.local, state);

        // at this point, we have set up the error reporting state.
        if let Some(init_index) = previously_initialized {
            if let (AccessKind::Mutate, Some(_)) = (error_access, place.as_local()) {
                // If this is a mutate access to an immutable local variable with no projections
                // report the error as an illegal reassignment
                let init = &self.move_data.inits[init_index];
                let assigned_span = init.span(self.body);
                self.report_illegal_reassignment((place, span), assigned_span, place);
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
        state: &BorrowckDomain<'a, 'tcx>,
    ) -> Option<InitIndex> {
        let mpi = self.move_data.rev_lookup.find_local(local)?;
        let ii = &self.move_data.init_path_map[mpi];
        ii.into_iter().find(|&&index| state.ever_inits.contains(index)).copied()
    }

    /// Adds the place into the used mutable variables set
    fn add_used_mut(&mut self, root_place: RootPlace<'tcx>, state: &BorrowckDomain<'a, 'tcx>) {
        match root_place {
            RootPlace { place_local: local, place_projection: [], is_local_mutation_allowed } => {
                // If the local may have been initialized, and it is now currently being
                // mutated, then it is justified to be annotated with the `mut`
                // keyword, since the mutation may be a possible reassignment.
                if is_local_mutation_allowed != LocalMutationIsAllowed::Yes
                    && self.is_local_ever_initialized(local, state).is_some()
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
                                            Some(field)
                                                if self.upvars[field.index()].is_by_ref() =>
                                            {
                                                is_local_mutation_allowed
                                            }
                                            _ => LocalMutationIsAllowed::Yes,
                                        };

                                        self.is_mutable(place_base, mode)
                                    }
                                }
                            }
                            ty::RawPtr(_, mutbl) => {
                                match mutbl {
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
                    | ProjectionElem::Subtype(..)
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
                            match (upvar.mutability, is_local_mutation_allowed) {
                                (
                                    Mutability::Not,
                                    LocalMutationIsAllowed::No
                                    | LocalMutationIsAllowed::ExceptUpvars,
                                ) => Err(place),
                                (Mutability::Not, LocalMutationIsAllowed::Yes)
                                | (Mutability::Mut, _) => {
                                    // Subtle: this is an upvar reference, so it looks like
                                    // `self.foo` -- we want to double check that the location
                                    // `*self` is mutable (i.e., this is not a `Fn` closure). But
                                    // if that check succeeds, we want to *blame* the mutability on
                                    // `place` (that is, `self.foo`). This is used to propagate the
                                    // info about whether mutability declarations are used
                                    // outwards, so that we register the outer variable as mutable.
                                    // Otherwise a test like this fails to record the `mut` as
                                    // needed:
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

mod diags {
    use rustc_errors::ErrorGuaranteed;

    use super::*;

    enum BufferedDiag<'infcx> {
        Error(Diag<'infcx>),
        NonError(Diag<'infcx, ()>),
    }

    impl<'infcx> BufferedDiag<'infcx> {
        fn sort_span(&self) -> Span {
            match self {
                BufferedDiag::Error(diag) => diag.sort_span,
                BufferedDiag::NonError(diag) => diag.sort_span,
            }
        }
    }

    pub(crate) struct BorrowckDiags<'infcx, 'tcx> {
        /// This field keeps track of move errors that are to be reported for given move indices.
        ///
        /// There are situations where many errors can be reported for a single move out (see
        /// #53807) and we want only the best of those errors.
        ///
        /// The `report_use_of_moved_or_uninitialized` function checks this map and replaces the
        /// diagnostic (if there is one) if the `Place` of the error being reported is a prefix of
        /// the `Place` of the previous most diagnostic. This happens instead of buffering the
        /// error. Once all move errors have been reported, any diagnostics in this map are added
        /// to the buffer to be emitted.
        ///
        /// `BTreeMap` is used to preserve the order of insertions when iterating. This is necessary
        /// when errors in the map are being re-added to the error buffer so that errors with the
        /// same primary span come out in a consistent order.
        buffered_move_errors: BTreeMap<Vec<MoveOutIndex>, (PlaceRef<'tcx>, Diag<'infcx>)>,

        buffered_mut_errors: FxIndexMap<Span, (Diag<'infcx>, usize)>,

        /// Buffer of diagnostics to be reported. A mixture of error and non-error diagnostics.
        buffered_diags: Vec<BufferedDiag<'infcx>>,
    }

    impl<'infcx, 'tcx> BorrowckDiags<'infcx, 'tcx> {
        pub(crate) fn new() -> Self {
            BorrowckDiags {
                buffered_move_errors: BTreeMap::new(),
                buffered_mut_errors: Default::default(),
                buffered_diags: Default::default(),
            }
        }

        pub(crate) fn buffer_error(&mut self, diag: Diag<'infcx>) {
            self.buffered_diags.push(BufferedDiag::Error(diag));
        }

        pub(crate) fn buffer_non_error(&mut self, diag: Diag<'infcx, ()>) {
            self.buffered_diags.push(BufferedDiag::NonError(diag));
        }
    }

    impl<'infcx, 'tcx> MirBorrowckCtxt<'_, 'infcx, 'tcx> {
        pub(crate) fn buffer_error(&mut self, diag: Diag<'infcx>) {
            self.diags.buffer_error(diag);
        }

        pub(crate) fn buffer_non_error(&mut self, diag: Diag<'infcx, ()>) {
            self.diags.buffer_non_error(diag);
        }

        pub(crate) fn buffer_move_error(
            &mut self,
            move_out_indices: Vec<MoveOutIndex>,
            place_and_err: (PlaceRef<'tcx>, Diag<'infcx>),
        ) -> bool {
            if let Some((_, diag)) =
                self.diags.buffered_move_errors.insert(move_out_indices, place_and_err)
            {
                // Cancel the old diagnostic so we don't ICE
                diag.cancel();
                false
            } else {
                true
            }
        }

        pub(crate) fn get_buffered_mut_error(
            &mut self,
            span: Span,
        ) -> Option<(Diag<'infcx>, usize)> {
            // FIXME(#120456) - is `swap_remove` correct?
            self.diags.buffered_mut_errors.swap_remove(&span)
        }

        pub(crate) fn buffer_mut_error(&mut self, span: Span, diag: Diag<'infcx>, count: usize) {
            self.diags.buffered_mut_errors.insert(span, (diag, count));
        }

        pub(crate) fn emit_errors(&mut self) -> Option<ErrorGuaranteed> {
            let mut res = self.infcx.tainted_by_errors();

            // Buffer any move errors that we collected and de-duplicated.
            for (_, (_, diag)) in std::mem::take(&mut self.diags.buffered_move_errors) {
                // We have already set tainted for this error, so just buffer it.
                self.diags.buffer_error(diag);
            }
            for (_, (mut diag, count)) in std::mem::take(&mut self.diags.buffered_mut_errors) {
                if count > 10 {
                    #[allow(rustc::diagnostic_outside_of_impl)]
                    #[allow(rustc::untranslatable_diagnostic)]
                    diag.note(format!("...and {} other attempted mutable borrows", count - 10));
                }
                self.diags.buffer_error(diag);
            }

            if !self.diags.buffered_diags.is_empty() {
                self.diags.buffered_diags.sort_by_key(|buffered_diag| buffered_diag.sort_span());
                for buffered_diag in self.diags.buffered_diags.drain(..) {
                    match buffered_diag {
                        BufferedDiag::Error(diag) => res = Some(diag.emit()),
                        BufferedDiag::NonError(diag) => diag.emit(),
                    }
                }
            }

            res
        }

        pub(crate) fn has_buffered_diags(&self) -> bool {
            self.diags.buffered_diags.is_empty()
        }

        pub(crate) fn has_move_error(
            &self,
            move_out_indices: &[MoveOutIndex],
        ) -> Option<&(PlaceRef<'tcx>, Diag<'infcx>)> {
            self.diags.buffered_move_errors.get(move_out_indices)
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
