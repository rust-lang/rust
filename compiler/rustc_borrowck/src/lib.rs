//! This query borrow-checks the MIR to (further) ensure it is not broken.

// ignore-tidy-filelength
// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(rust_logo)]
#![feature(assert_matches)]
#![feature(box_patterns)]
#![feature(file_buffered)]
#![feature(if_let_guard)]
#![feature(negative_impls)]
#![feature(never_type)]
#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
#![feature(stmt_expr_attributes)]
#![feature(try_blocks)]
// tidy-alphabetical-end

use std::borrow::Cow;
use std::cell::{OnceCell, RefCell};
use std::marker::PhantomData;
use std::ops::{ControlFlow, Deref};

use borrow_set::LocalsStateAtExit;
use root_cx::BorrowCheckRootCtxt;
use rustc_abi::FieldIdx;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::graph::dominators::Dominators;
use rustc_errors::LintDiagnostic;
use rustc_hir as hir;
use rustc_hir::CRATE_HIR_ID;
use rustc_hir::def_id::LocalDefId;
use rustc_index::bit_set::MixedBitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_infer::infer::{
    InferCtxt, NllRegionVariableOrigin, RegionVariableOrigin, TyCtxtInferExt,
};
use rustc_middle::mir::*;
use rustc_middle::query::Providers;
use rustc_middle::ty::{
    self, ParamEnv, RegionVid, Ty, TyCtxt, TypeFoldable, TypeVisitable, TypingMode, fold_regions,
};
use rustc_middle::{bug, span_bug};
use rustc_mir_dataflow::impls::{EverInitializedPlaces, MaybeUninitializedPlaces};
use rustc_mir_dataflow::move_paths::{
    InitIndex, InitLocation, LookupResult, MoveData, MovePathIndex,
};
use rustc_mir_dataflow::{Analysis, ResultsVisitor};
// use rustc_mir_dataflow::{Analysis, Results, ResultsVisitor, visit_results};
use rustc_session::lint::builtin::{TAIL_EXPR_DROP_ORDER, UNUSED_MUT};
use rustc_span::{ErrorGuaranteed, Span, Symbol};
use smallvec::SmallVec;
use tracing::{debug, instrument};

use crate::borrow_set::{BorrowData, BorrowSet};
use crate::consumers::BodyWithBorrowckFacts;
use crate::dataflow::{BorrowIndex, Borrowck, BorrowckDomain, Borrows};
use crate::diagnostics::{
    AccessKind, BorrowckDiagnosticsBuffer, IllegalMoveOriginKind, MoveError, RegionName,
};
use crate::path_utils::*;
use crate::place_ext::PlaceExt;
use crate::places_conflict::{PlaceConflictBias, places_conflict};
use crate::polonius::PoloniusDiagnosticsContext;
use crate::polonius::legacy::{PoloniusLocationTable, PoloniusOutput};
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
mod handle_placeholders;
mod member_constraints;
mod nll;
mod path_utils;
mod place_ext;
mod places_conflict;
mod polonius;
mod prefixes;
mod region_infer;
mod renumber;
mod root_cx;
mod session_diagnostics;
mod type_check;
mod universal_regions;
mod used_muts;

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

/// Provider for `query mir_borrowck`. Similar to `typeck`, this must
/// only be called for typeck roots which will then borrowck all
/// nested bodies as well.
fn mir_borrowck(
    tcx: TyCtxt<'_>,
    def: LocalDefId,
) -> Result<&ConcreteOpaqueTypes<'_>, ErrorGuaranteed> {
    assert!(!tcx.is_typeck_child(def.to_def_id()));
    let (input_body, _) = tcx.mir_promoted(def);
    debug!("run query mir_borrowck: {}", tcx.def_path_str(def));

    let input_body: &Body<'_> = &input_body.borrow();
    if let Some(guar) = input_body.tainted_by_errors {
        debug!("Skipping borrowck because of tainted body");
        Err(guar)
    } else if input_body.should_skip() {
        debug!("Skipping borrowck because of injected body");
        let opaque_types = ConcreteOpaqueTypes(Default::default());
        Ok(tcx.arena.alloc(opaque_types))
    } else {
        let mut root_cx = BorrowCheckRootCtxt::new(tcx, def, None);
        // We need to manually borrowck all nested bodies from the HIR as
        // we do not generate MIR for dead code. Not doing so causes us to
        // never check closures in dead code.
        let nested_bodies = tcx.nested_bodies_within(def);
        for def_id in nested_bodies {
            root_cx.get_or_insert_nested(def_id);
        }

        let PropagatedBorrowCheckResults { closure_requirements, used_mut_upvars } =
            do_mir_borrowck(&mut root_cx, def);
        debug_assert!(closure_requirements.is_none());
        debug_assert!(used_mut_upvars.is_empty());
        root_cx.finalize()
    }
}

/// Data propagated to the typeck parent by nested items.
/// This should always be empty for the typeck root.
#[derive(Debug)]
struct PropagatedBorrowCheckResults<'tcx> {
    closure_requirements: Option<ClosureRegionRequirements<'tcx>>,
    used_mut_upvars: SmallVec<[FieldIdx; 8]>,
}

/// After we borrow check a closure, we are left with various
/// requirements that we have inferred between the free regions that
/// appear in the closure's signature or on its field types. These
/// requirements are then verified and proved by the closure's
/// creating function. This struct encodes those requirements.
///
/// The requirements are listed as being between various `RegionVid`. The 0th
/// region refers to `'static`; subsequent region vids refer to the free
/// regions that appear in the closure (or coroutine's) type, in order of
/// appearance. (This numbering is actually defined by the `UniversalRegions`
/// struct in the NLL region checker. See for example
/// `UniversalRegions::closure_mapping`.) Note the free regions in the
/// closure's signature and captures are erased.
///
/// Example: If type check produces a closure with the closure args:
///
/// ```text
/// ClosureArgs = [
///     'a,                                         // From the parent.
///     'b,
///     i8,                                         // the "closure kind"
///     for<'x> fn(&'<erased> &'x u32) -> &'x u32,  // the "closure signature"
///     &'<erased> String,                          // some upvar
/// ]
/// ```
///
/// We would "renumber" each free region to a unique vid, as follows:
///
/// ```text
/// ClosureArgs = [
///     '1,                                         // From the parent.
///     '2,
///     i8,                                         // the "closure kind"
///     for<'x> fn(&'3 &'x u32) -> &'x u32,         // the "closure signature"
///     &'4 String,                                 // some upvar
/// ]
/// ```
///
/// Now the code might impose a requirement like `'1: '2`. When an
/// instance of the closure is created, the corresponding free regions
/// can be extracted from its type and constrained to have the given
/// outlives relationship.
#[derive(Clone, Debug)]
pub struct ClosureRegionRequirements<'tcx> {
    /// The number of external regions defined on the closure. In our
    /// example above, it would be 3 -- one for `'static`, then `'1`
    /// and `'2`. This is just used for a sanity check later on, to
    /// make sure that the number of regions we see at the callsite
    /// matches.
    pub num_external_vids: usize,

    /// Requirements between the various free regions defined in
    /// indices.
    pub outlives_requirements: Vec<ClosureOutlivesRequirement<'tcx>>,
}

/// Indicates an outlives-constraint between a type or between two
/// free regions declared on the closure.
#[derive(Copy, Clone, Debug)]
pub struct ClosureOutlivesRequirement<'tcx> {
    // This region or type ...
    pub subject: ClosureOutlivesSubject<'tcx>,

    // ... must outlive this one.
    pub outlived_free_region: ty::RegionVid,

    // If not, report an error here ...
    pub blame_span: Span,

    // ... due to this reason.
    pub category: ConstraintCategory<'tcx>,
}

// Make sure this enum doesn't unintentionally grow
#[cfg(target_pointer_width = "64")]
rustc_data_structures::static_assert_size!(ConstraintCategory<'_>, 16);

/// The subject of a `ClosureOutlivesRequirement` -- that is, the thing
/// that must outlive some region.
#[derive(Copy, Clone, Debug)]
pub enum ClosureOutlivesSubject<'tcx> {
    /// Subject is a type, typically a type parameter, but could also
    /// be a projection. Indicates a requirement like `T: 'a` being
    /// passed to the caller, where the type here is `T`.
    Ty(ClosureOutlivesSubjectTy<'tcx>),

    /// Subject is a free region from the closure. Indicates a requirement
    /// like `'a: 'b` being passed to the caller; the region here is `'a`.
    Region(ty::RegionVid),
}

/// Represents a `ty::Ty` for use in [`ClosureOutlivesSubject`].
///
/// This abstraction is necessary because the type may include `ReVar` regions,
/// which is what we use internally within NLL code, and they can't be used in
/// a query response.
#[derive(Copy, Clone, Debug)]
pub struct ClosureOutlivesSubjectTy<'tcx> {
    inner: Ty<'tcx>,
}
// DO NOT implement `TypeVisitable` or `TypeFoldable` traits, because this
// type is not recognized as a binder for late-bound region.
impl<'tcx, I> !TypeVisitable<I> for ClosureOutlivesSubjectTy<'tcx> {}
impl<'tcx, I> !TypeFoldable<I> for ClosureOutlivesSubjectTy<'tcx> {}

impl<'tcx> ClosureOutlivesSubjectTy<'tcx> {
    /// All regions of `ty` must be of kind `ReVar` and must represent
    /// universal regions *external* to the closure.
    pub fn bind(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Self {
        let inner = fold_regions(tcx, ty, |r, depth| match r.kind() {
            ty::ReVar(vid) => {
                let br = ty::BoundRegion {
                    var: ty::BoundVar::from_usize(vid.index()),
                    kind: ty::BoundRegionKind::Anon,
                };
                ty::Region::new_bound(tcx, depth, br)
            }
            _ => bug!("unexpected region in ClosureOutlivesSubjectTy: {r:?}"),
        });

        Self { inner }
    }

    pub fn instantiate(
        self,
        tcx: TyCtxt<'tcx>,
        mut map: impl FnMut(ty::RegionVid) -> ty::Region<'tcx>,
    ) -> Ty<'tcx> {
        fold_regions(tcx, self.inner, |r, depth| match r.kind() {
            ty::ReBound(debruijn, br) => {
                debug_assert_eq!(debruijn, depth);
                map(ty::RegionVid::from_usize(br.var.index()))
            }
            _ => bug!("unexpected region {r:?}"),
        })
    }
}

/// Perform the actual borrow checking.
///
/// For nested bodies this should only be called through `root_cx.get_or_insert_nested`.
#[instrument(skip(root_cx), level = "debug")]
fn do_mir_borrowck<'tcx>(
    root_cx: &mut BorrowCheckRootCtxt<'tcx>,
    def: LocalDefId,
) -> PropagatedBorrowCheckResults<'tcx> {
    let tcx = root_cx.tcx;
    let infcx = BorrowckInferCtxt::new(tcx, def);
    let (input_body, promoted) = tcx.mir_promoted(def);
    let input_body: &Body<'_> = &input_body.borrow();
    let input_promoted: &IndexSlice<_, _> = &promoted.borrow();
    if let Some(e) = input_body.tainted_by_errors {
        infcx.set_tainted_by_errors(e);
        root_cx.set_tainted_by_errors(e);
    }

    // Replace all regions with fresh inference variables. This
    // requires first making our own copy of the MIR. This copy will
    // be modified (in place) to contain non-lexical lifetimes. It
    // will have a lifetime tied to the inference context.
    let mut body_owned = input_body.clone();
    let mut promoted = input_promoted.to_owned();
    let universal_regions = nll::replace_regions_in_mir(&infcx, &mut body_owned, &mut promoted);
    let body = &body_owned; // no further changes

    let location_table = PoloniusLocationTable::new(body);

    let move_data = MoveData::gather_moves(body, tcx, |_| true);

    let locals_are_invalidated_at_exit = tcx.hir_body_owner_kind(def).is_fn_or_closure();
    let borrow_set = BorrowSet::build(tcx, body, locals_are_invalidated_at_exit, &move_data);

    // Compute non-lexical lifetimes.
    let nll::NllOutput {
        regioncx,
        polonius_input,
        polonius_output,
        opt_closure_req,
        nll_errors,
        polonius_diagnostics,
    } = nll::compute_regions(
        root_cx,
        &infcx,
        universal_regions,
        body,
        &promoted,
        &location_table,
        &move_data,
        &borrow_set,
    );

    // use rustc_data_structures::unify;

    // #[derive(Debug, PartialEq, Copy, Clone)]
    // struct Yo(RegionVid);

    // impl unify::UnifyKey for Yo {
    //     type Value = ();

    //     fn index(&self) -> u32 {
    //         self.0.as_u32()
    //     }

    //     fn from_index(u: u32) -> Self {
    //         Self(RegionVid::from_u32(u))
    //     }

    //     fn tag() -> &'static str {
    //         "Yo!"
    //     }
    // }

    // let timer = std::time::Instant::now();
    // let mut ena = unify::UnificationTable::<unify::InPlace<Yo>>::new();
    // let mut keys = Vec::new();
    // for _ in regioncx.definitions.indices() {
    //     keys.push(ena.new_key(()));
    // }
    // for c in regioncx.outlives_constraints() {
    //     // let sup = Yo(c.sup);
    //     // let sub = Yo(c.sub);
    //     ena.union(keys[c.sup.as_usize()], keys[c.sub.as_usize()]);
    //     // table.unify_var_var(sup, sub).unwrap();
    //     // table.unify_var_var(a_id, b_id)
    // }
    // let elapsed_2 = timer.elapsed();

    // if borrow_set.len() > 0 {
    //     // if elapsed_1 < elapsed_2 {
    //     //     eprintln!(
    //     //         "table wins: by {} ns",
    //     //         elapsed_2.as_nanos() - elapsed_1.as_nanos()
    //     //     );
    //     // } else {
    //     //     eprintln!(
    //     //         "ena wins: by {} ns",
    //     //         elapsed_1.as_nanos() - elapsed_2.as_nanos()
    //     //     );
    //     // }

    //     // FIXME: check how much it takes if we use this in loans in scope pre-check
    //     // eprintln!("ena unification took: {} ns", elapsed_2.as_nanos());

    //     // eprintln!(
    //     //     "region union find: {} sets in {} ns, region count: {:?}, scc count: {}, loan count: {}, {:?}",
    //     //     table.count(),
    //     //     elapsed.as_nanos(),
    //     //     regioncx.definitions.len(),
    //     //     regioncx.constraint_sccs().num_sccs(),
    //     //     borrow_set.len(),
    //     //     body.span
    //     // );

    //     // eprintln!(
    //     //     "region union find: {:5} sets in {} ns, region count: {:?}, scc count: {}, loan count: {}, {:?}",
    //     //     table.count(),
    //     //     elapsed_1.as_nanos(),
    //     //     regioncx.definitions.len(),
    //     //     regioncx.constraint_sccs().num_sccs(),
    //     //     borrow_set.len(),
    //     //     body.span
    //     // );

    //     // eprintln!(
    //     //     "ena    union find: {:5} sets in {} ns, region count: {:?}, scc count: {}, loan count: {}, {:?}",
    //     //     ena.len(),
    //     //     elapsed_2.as_nanos(),
    //     //     regioncx.definitions.len(),
    //     //     regioncx.constraint_sccs().num_sccs(),
    //     //     borrow_set.len(),
    //     //     body.span
    //     // );

    //     // for (idx, loan) in borrow_set.iter_enumerated() {
    //     //     let borrow_region = loan.region;
    //     //     let same_set: Vec<_> = regioncx
    //     //         .definitions
    //     //         .indices()
    //     //         .filter(|r| table.is_same_set(&borrow_region, &r))
    //     //         .collect();
    //     //     // let different_set = regioncx.outlives_constraints().count() - same_set.len();
    //     //     // eprint!(
    //     //     //     "loan {} from region {} involves {} regions in the set, and {} unrelated regions",
    //     //     //     idx.as_usize(),
    //     //     //     borrow_region.as_usize(),
    //     //     //     same_set.len(),
    //     //     //     different_set,
    //     //     // );
    //     //     // if same_set.len() < 15 {
    //     //     //     eprintln!(", same: {:?}", same_set);
    //     //     // } else {
    //     //     //     eprintln!();
    //     //     // }

    //     //     let ena_same_set: Vec<_> = regioncx
    //     //         .definitions
    //     //         .indices()
    //     //         .filter(|r| ena.unioned(keys[borrow_region.as_usize()], keys[r.as_usize()]))
    //     //         .collect();

    //     //     assert_eq!(same_set, ena_same_set);

    //     //     // eprintln!(
    //     //     //     "loan involves {} regions in the set, ena found {} regions in the set",
    //     //     //     same_set.len(),
    //     //     //     ena_same_set.len()
    //     //     // );
    //     // }
    // }

    // Dump MIR results into a file, if that is enabled. This lets us
    // write unit-tests, as well as helping with debugging.
    nll::dump_nll_mir(&infcx, body, &regioncx, &opt_closure_req, &borrow_set);
    polonius::dump_polonius_mir(
        &infcx,
        body,
        &regioncx,
        &opt_closure_req,
        &borrow_set,
        polonius_diagnostics.as_ref(),
    );

    // We also have a `#[rustc_regions]` annotation that causes us to dump
    // information.
    nll::dump_annotation(&infcx, body, &regioncx, &opt_closure_req);

    let movable_coroutine = body.coroutine.is_some()
        && tcx.coroutine_movability(def.to_def_id()) == hir::Movability::Movable;

    let diags_buffer = &mut BorrowckDiagnosticsBuffer::default();
    // While promoteds should mostly be correct by construction, we need to check them for
    // invalid moves to detect moving out of arrays:`struct S; fn main() { &([S][0]); }`.
    for promoted_body in &promoted {
        use rustc_middle::mir::visit::Visitor;
        // This assumes that we won't use some of the fields of the `promoted_mbcx`
        // when detecting and reporting move errors. While it would be nice to move
        // this check out of `MirBorrowckCtxt`, actually doing so is far from trivial.
        let move_data = MoveData::gather_moves(promoted_body, tcx, |_| true);
        let mut promoted_mbcx = MirBorrowckCtxt {
            root_cx,
            infcx: &infcx,
            body: promoted_body,
            move_data: &move_data,
            // no need to create a real location table for the promoted, it is not used
            location_table: &location_table,
            movable_coroutine,
            fn_self_span_reported: Default::default(),
            access_place_error_reported: Default::default(),
            reservation_error_reported: Default::default(),
            uninitialized_error_reported: Default::default(),
            regioncx: &regioncx,
            used_mut: Default::default(),
            used_mut_upvars: SmallVec::new(),
            borrow_set: &borrow_set,
            upvars: &[],
            local_names: OnceCell::from(IndexVec::from_elem(None, &promoted_body.local_decls)),
            region_names: RefCell::default(),
            next_region_name: RefCell::new(1),
            polonius_output: None,
            move_errors: Vec::new(),
            diags_buffer,
            polonius_diagnostics: polonius_diagnostics.as_ref(),
            #[cfg(test)]
            nuutila: None,
            #[cfg(test)]
            duration: 0,
            #[cfg(test)]
            duration2: 0,
            #[cfg(test)]
            duration3: 0,
            #[cfg(test)]
            transitive_predecessors: None,
            #[cfg(test)]
            locals_checked_for_initialization: FxHashMap::default(),
        };
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
        MoveVisitor { ctxt: &mut promoted_mbcx }.visit_body(promoted_body);
        promoted_mbcx.report_move_errors();
    }

    let mut mbcx = MirBorrowckCtxt {
        root_cx,
        infcx: &infcx,
        body,
        move_data: &move_data,
        location_table: &location_table,
        movable_coroutine,
        fn_self_span_reported: Default::default(),
        access_place_error_reported: Default::default(),
        reservation_error_reported: Default::default(),
        uninitialized_error_reported: Default::default(),
        regioncx: &regioncx,
        used_mut: Default::default(),
        used_mut_upvars: SmallVec::new(),
        borrow_set: &borrow_set,
        upvars: tcx.closure_captures(def),
        local_names: OnceCell::new(),
        region_names: RefCell::default(),
        next_region_name: RefCell::new(1),
        move_errors: Vec::new(),
        diags_buffer,
        polonius_output: polonius_output.as_deref(),
        polonius_diagnostics: polonius_diagnostics.as_ref(),
        #[cfg(test)]
        nuutila: None,
        #[cfg(test)]
        duration: 0,
        #[cfg(test)]
        duration2: 0,
        #[cfg(test)]
        duration3: 0,
        #[cfg(test)]
        transitive_predecessors: None,
        #[cfg(test)]
        locals_checked_for_initialization: FxHashMap::default(),
    };

    // Compute and report region errors, if any.
    mbcx.report_region_errors(nll_errors);

    // if body.basic_blocks.len() > 5000 {
    //     let stmts: usize =
    //         body.basic_blocks.iter_enumerated().map(|(_idx, block)| block.statements.len()).sum();
    //     eprintln!(
    //         "\nCFG stats, blocks: {}, statements: {}, is cyclic: {}, {:?}",
    //         body.basic_blocks.len(),
    //         stmts,
    //         rustc_data_structures::graph::is_cyclic(&body.basic_blocks),
    //         body.span,
    //     );
    // }

    if body.basic_blocks.is_cfg_cyclic() {
        // let (mut flow_analysis, flow_entry_states) =
        //     get_flow_results(tcx, body, &move_data, &borrow_set, &regioncx);
        // visit_results(
        //     body,
        //     traversal::reverse_postorder(body).map(|(bb, _)| bb),
        //     &mut flow_analysis,
        //     &flow_entry_states,
        //     &mut mbcx,
        // );

        // let sccs = body.basic_blocks.sccs();
        // let mut single_block = 0; let mut single_successor = 0; let mut single_predecessor = 0;
        // for &scc in &sccs.queue {
        //     if sccs.sccs[scc as usize].len() == 1 {
        //         single_block += 1;
        //     }

        //     for block in sccs.sccs[scc as usize].iter().copied() {
        //         if body[block].terminator().successors().count() == 1 {
        //             single_successor += 1;
        //         }

        //         if body.basic_blocks.predecessors()[block].len() == 1 {
        //             single_predecessor += 1;
        //         }
        //     }
        // }

        // eprintln!(
        //     "CFG, {} blocks, SCCs: {}, single-block SCCs: {}, single-successor blocks: {}, single-predecessor blocks: {}, {:?}",
        //     body.basic_blocks.len(),
        //     sccs.component_count,
        //     single_block,
        //     single_successor,
        //     single_predecessor,
        //     body.span,
        // );

        let borrows = Borrows::new(tcx, body, &regioncx, &borrow_set);
        let uninits = MaybeUninitializedPlaces::new(tcx, body, &move_data);
        let ever_inits = EverInitializedPlaces::new(body, &move_data);
        compute_cyclic_dataflow(body, borrows, uninits, ever_inits, &mut mbcx);

        // let (_, flow_entry_states) =
        //     get_flow_results(tcx, body, &move_data, &borrow_set, &regioncx);
        // compute_cyclic_dataflow(body, borrows, uninits, ever_inits, &mut mbcx, &flow_entry_states);
    } else {
        // compute_dataflow(tcx, body, &move_data, &borrow_set, &regioncx, &mut mbcx);

        let borrows = Borrows::new(tcx, body, &regioncx, &borrow_set);
        let uninits = MaybeUninitializedPlaces::new(tcx, body, &move_data);
        let ever_inits = EverInitializedPlaces::new(body, &move_data);
        let mut analysis = Borrowck { borrows, uninits, ever_inits };
        compute_rpo_dataflow(body, &mut analysis, &mut mbcx);
    }

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
    mbcx.lint_unused_mut();
    if let Some(guar) = mbcx.emit_errors() {
        mbcx.root_cx.set_tainted_by_errors(guar);
    }

    let result = PropagatedBorrowCheckResults {
        closure_requirements: opt_closure_req,
        used_mut_upvars: mbcx.used_mut_upvars,
    };

    #[cfg(test)]
    if body.basic_blocks.len() > 5000 {
        eprintln!("borrow stats, locals: {}, loans: {}", body.local_decls.len(), borrow_set.len());
        eprintln!("nuutila duration:     {} ns", mbcx.duration);
        eprintln!("predecessor duration: {} ns", mbcx.duration2);
        eprintln!("NLL scopes duration:  {} ns", mbcx.duration3);

        use std::collections::VecDeque;

        use rustc_data_structures::graph::scc::*;

        //
        {
            eprint!("SCC tests - {:>30}", "rustc");
            let timer = std::time::Instant::now();

            type CfgScc = Sccs<BasicBlock, usize>;
            let sccs = CfgScc::new(&body.basic_blocks);

            let elapsed = timer.elapsed();
            // eprintln!(", computed {} SCCs in {} ns", sccs.num_sccs(), elapsed.as_nanos());
            eprint!(", computed {} SCCs in {} ns", sccs.num_sccs(), elapsed.as_nanos());

            use rustc_index::interval::IntervalSet;

            let timer = std::time::Instant::now();
            let mut components = vec![IntervalSet::new(body.basic_blocks.len()); sccs.num_sccs()];
            for block in body.basic_blocks.indices() {
                let scc = sccs.scc(block);
                components[scc].insert(block);
            }
            let elapsed = timer.elapsed();

            eprintln!(" and SCCs contents in {} ns (intervals)", elapsed.as_nanos(),);
        }

        //
        {
            eprint!("SCC tests - {:>30}", "tarjan SCCs (dense/usize)");

            struct Scc {
                candidate_component_roots: Vec<usize>,
                components: Vec<isize>,
                component_count: usize,
                dfs_numbers: Vec<u32>,
                d: u32,
                stack: VecDeque<usize>,
                visited: DenseBitSet<usize>,
            }

            impl Scc {
                fn new(node_count: usize) -> Self {
                    Self {
                        candidate_component_roots: vec![0; node_count],
                        components: vec![-1; node_count],
                        component_count: 0,
                        dfs_numbers: vec![0; node_count],
                        d: 0,
                        stack: VecDeque::new(),
                        visited: DenseBitSet::new_empty(node_count),
                    }
                }

                fn compute_sccs(&mut self, blocks: &BasicBlocks<'_>) {
                    for (idx, block) in blocks.iter_enumerated() {
                        let edges = block.terminator().edges();
                        if matches!(edges, TerminatorEdges::None) {
                            continue;
                        }

                        let idx = idx.as_usize();
                        if !self.visited.contains(idx) {
                            self.dfs_visit(idx, blocks);
                        }
                    }
                }

                fn dfs_visit(&mut self, v: usize, blocks: &BasicBlocks<'_>) {
                    self.candidate_component_roots[v] = v;
                    self.components[v] = -1;

                    self.d += 1;
                    self.dfs_numbers[v] = self.d;

                    self.visited.insert(v);

                    // println!(
                    //     "dfs_visit, v = {v}, CCR[{v}] = {}, D[{v}] = {}",
                    //     self.candidate_component_roots[v],
                    //     self.dfs_numbers[v],
                    //     v = v
                    // );

                    let idx = BasicBlock::from_usize(v);
                    for succ in blocks[idx].terminator().successors() {
                        let w = succ.as_usize();

                        // if w == v {
                        //     panic!("a dang self loop ?!");
                        // }

                        if !self.visited.contains(w) {
                            self.dfs_visit(w, blocks);
                        }

                        // println!(
                        //     "v = {v} - adjacent vertex w = {w}: C[w] = {}, C[v] = {} / D[CCR[w]] = {}, D[CCR[v]] = {} / CCR[v] = {}, CCR[w] = {}",
                        //     self.components[w],
                        //     self.components[v],
                        //     self.dfs_numbers[self.candidate_component_roots[w]],
                        //     self.dfs_numbers[self.candidate_component_roots[v]],
                        //     self.candidate_component_roots[v],
                        //     self.candidate_component_roots[w],
                        //     w = w,
                        //     v = v,
                        // );

                        if self.components[w] == -1
                            && self.dfs_numbers[self.candidate_component_roots[w]]
                                < self.dfs_numbers[self.candidate_component_roots[v]]
                        {
                            self.candidate_component_roots[v] = self.candidate_component_roots[w];
                        }
                    }

                    // println!(
                    //     "v = {v} - CCR[v] = {}",
                    //     self.candidate_component_roots[v],
                    //     v = v,
                    // );

                    if self.candidate_component_roots[v] == v {
                        self.components[v] = self.component_count as isize;
                        self.component_count += 1;

                        // println!(
                        //     "v = {v} - creating component {} / C[v] = {}",
                        //     self.component_count,
                        //     self.components[v],
                        //     v = v,
                        // );

                        while self.stack.front().is_some()
                            && self.dfs_numbers[*self.stack.front().expect("peek front failed")]
                                > self.dfs_numbers[v]
                        {
                            let w = self.stack.pop_front().expect("pop front failed");
                            self.components[w] = self.components[v];
                            // println!(
                            //     "v = {v} - popping w = {w} off the stack (contents: {:?}) / C[w] = {}, C[v] = {}",
                            //     self.stack,
                            //     self.components[w],
                            //     self.components[v],
                            //     v = v,
                            //     w = w,
                            // );
                        }
                    } else {
                        // println!(
                        //     "v = {v}: pushing v on the stack (contents: {:?})",
                        //     self.stack,
                        //     v = v,
                        // );
                        self.stack.push_front(v);
                    }
                }
            }

            let timer = std::time::Instant::now();
            let mut sccs = Scc::new(body.basic_blocks.len());
            sccs.compute_sccs(&body.basic_blocks);

            let elapsed = timer.elapsed();
            // eprintln!(", computed {} SCCs in {} ns", sccs.component_count, elapsed.as_nanos());
            eprint!(", computed {} SCCs in {} ns", sccs.component_count, elapsed.as_nanos());

            // sanity checks
            // eprintln!("-----");
            // eprintln!("blocks: {}", body.basic_blocks.len());
            // for block in body.basic_blocks.indices() {
            //     let successors: Vec<_> =
            //         body.basic_blocks[block].terminator().successors().collect();
            //     eprintln!("block: {:?}, successors: {:?}", block, successors);
            // }

            let rustc_sccs = Sccs::<BasicBlock, usize>::new(&body.basic_blocks);

            for block in body.basic_blocks.indices() {
                let rustc_scc = rustc_sccs.scc(block);
                let scc = sccs.components[block.as_usize()] as usize;
                assert_eq!(
                    rustc_scc, scc,
                    "sccs differ for {block:?} between tarjan sccs: {scc}, and rustc's scc: {rustc_scc}"
                );
            }

            // eprintln!("-----");
            // eprintln!(
            //     "rustc SCCs: {} (min: {:?}, max: {:?})",
            //     rustc_sccs.num_sccs(),
            //     rustc_sccs.all_sccs().min().unwrap(),
            //     rustc_sccs.all_sccs().max().unwrap()
            // );
            // for scc in rustc_sccs.all_sccs() {
            //     let mut blocks = Vec::new();
            //     for block in body.basic_blocks.indices() {
            //         if rustc_sccs.scc(block) == scc {
            //             blocks.push(block);
            //         }
            //     }
            //     blocks.sort();

            //     let mut successors: Vec<_> = rustc_sccs.successors(scc).into_iter().collect();
            //     successors.sort();

            //     eprintln!("scc: {:?}, contains: {:?}, successors: {:?}", scc, blocks, successors);
            // }

            // eprintln!("-----");
            // eprintln!(
            //     "tarjan SCCs: {} (min: {:?}, max: {:?})",
            //     sccs.component_count,
            //     sccs.components.iter().min().unwrap(),
            //     sccs.components.iter().max().unwrap()
            // );
            // for scc in 0..sccs.component_count {
            //     let mut blocks = Vec::new();
            //     for block in body.basic_blocks.indices() {
            //         if sccs.components[block.as_usize()] == scc {
            //             blocks.push(block);
            //         }
            //     }
            //     blocks.sort();

            //     let mut successors = Vec::new();
            //     for &block in &blocks {
            //         let block_successors = body.basic_blocks[block].terminator().successors();
            //         let scc_successors = block_successors
            //             .map(|block| sccs.components[block.as_usize()])
            //             .filter(|&succ_scc| succ_scc != scc);
            //         successors.extend(scc_successors);
            //     }

            //     successors.sort();
            //     successors.dedup();

            //     eprintln!("scc: {:?}, contains: {:?}, successors: {:?}", scc, blocks, successors);
            // }

            // let timer = std::time::Instant::now();

            // use rustc_index::interval::IntervalSet;
            // let mut successors = vec![IntervalSet::new(sccs.component_count); sccs.component_count];
            // for block in body.basic_blocks.indices() {
            //     let scc = sccs.components[block.as_usize()] as usize;
            //     let scc_successors = body.basic_blocks[block]
            //         .terminator()
            //         .successors()
            //         .map(|block| sccs.components[block.as_usize()] as usize)
            //         .filter(|&succ_scc| succ_scc != scc);
            //     for succ in scc_successors {
            //         successors[scc].insert(succ);
            //     }
            // }
            // let elapsed = timer.elapsed();

            // let timer = std::time::Instant::now();
            // let mut components =
            //     vec![IntervalSet::new(body.basic_blocks.len()); sccs.component_count];
            // for block in body.basic_blocks.indices() {
            //     let scc = sccs.components[block.as_usize()] as usize;
            //     components[scc].insert(block.as_usize());
            // }
            // let elapsed2 = timer.elapsed();

            use rustc_index::interval::IntervalSet;

            let timer = std::time::Instant::now();

            let mut components =
                vec![IntervalSet::new(body.basic_blocks.len()); sccs.component_count];
            let mut successors = vec![IntervalSet::new(sccs.component_count); sccs.component_count];
            for block in body.basic_blocks.indices() {
                let scc = sccs.components[block.as_usize()] as usize;
                let scc_successors = body.basic_blocks[block]
                    .terminator()
                    .successors()
                    .map(|block| sccs.components[block.as_usize()] as usize)
                    .filter(|&succ_scc| succ_scc != scc);
                for succ in scc_successors {
                    successors[scc].insert(succ);
                }
                components[scc].insert(block.as_usize());
            }
            let elapsed2 = timer.elapsed();

            eprintln!(" and SCCs successors/contents in {} ns (intervals)", elapsed2.as_nanos(),);
        }

        //
        {
            eprint!("SCC tests - {:>30}", "tarjan SCCs (mixed/usize)");

            struct Scc {
                candidate_component_roots: Vec<usize>,
                components: Vec<isize>,
                component_count: usize,
                dfs_numbers: Vec<u32>,
                d: u32,
                stack: VecDeque<usize>,
                visited: MixedBitSet<usize>,
            }

            impl Scc {
                fn new(node_count: usize) -> Self {
                    Self {
                        candidate_component_roots: vec![0; node_count],
                        components: vec![-1; node_count],
                        component_count: 0,
                        dfs_numbers: vec![0; node_count],
                        d: 0,
                        stack: VecDeque::new(),
                        visited: MixedBitSet::new_empty(node_count),
                    }
                }

                fn compute_sccs(&mut self, blocks: &BasicBlocks<'_>) {
                    for (idx, block) in blocks.iter_enumerated() {
                        let edges = block.terminator().edges();
                        if matches!(edges, TerminatorEdges::None) {
                            continue;
                        }

                        let idx = idx.as_usize();
                        if !self.visited.contains(idx) {
                            self.dfs_visit(idx, blocks);
                        }
                    }
                }

                fn dfs_visit(&mut self, v: usize, blocks: &BasicBlocks<'_>) {
                    self.candidate_component_roots[v] = v;
                    self.components[v] = -1;

                    self.d += 1;
                    self.dfs_numbers[v] = self.d;

                    self.visited.insert(v);

                    // println!(
                    //     "dfs_visit, v = {v}, CCR[{v}] = {}, D[{v}] = {}",
                    //     self.candidate_component_roots[v],
                    //     self.dfs_numbers[v],
                    //     v = v
                    // );

                    let idx = BasicBlock::from_usize(v);
                    for succ in blocks[idx].terminator().successors() {
                        let w = succ.as_usize();

                        // if w == v {
                        //     panic!("a dang self loop ?!");
                        // }

                        if !self.visited.contains(w) {
                            self.dfs_visit(w, blocks);
                        }

                        // println!(
                        //     "v = {v} - adjacent vertex w = {w}: C[w] = {}, C[v] = {} / D[CCR[w]] = {}, D[CCR[v]] = {} / CCR[v] = {}, CCR[w] = {}",
                        //     self.components[w],
                        //     self.components[v],
                        //     self.dfs_numbers[self.candidate_component_roots[w]],
                        //     self.dfs_numbers[self.candidate_component_roots[v]],
                        //     self.candidate_component_roots[v],
                        //     self.candidate_component_roots[w],
                        //     w = w,
                        //     v = v,
                        // );

                        if self.components[w] == -1
                            && self.dfs_numbers[self.candidate_component_roots[w]]
                                < self.dfs_numbers[self.candidate_component_roots[v]]
                        {
                            self.candidate_component_roots[v] = self.candidate_component_roots[w];
                        }
                    }

                    // println!(
                    //     "v = {v} - CCR[v] = {}",
                    //     self.candidate_component_roots[v],
                    //     v = v,
                    // );

                    if self.candidate_component_roots[v] == v {
                        self.component_count += 1;
                        self.components[v] = self.component_count as isize;

                        // println!(
                        //     "v = {v} - creating component {} / C[v] = {}",
                        //     self.component_count,
                        //     self.components[v],
                        //     v = v,
                        // );

                        while self.stack.front().is_some()
                            && self.dfs_numbers[*self.stack.front().expect("peek front failed")]
                                > self.dfs_numbers[v]
                        {
                            let w = self.stack.pop_front().expect("pop front failed");
                            self.components[w] = self.components[v];
                            // println!(
                            //     "v = {v} - popping w = {w} off the stack (contents: {:?}) / C[w] = {}, C[v] = {}",
                            //     self.stack,
                            //     self.components[w],
                            //     self.components[v],
                            //     v = v,
                            //     w = w,
                            // );
                        }
                    } else {
                        // println!(
                        //     "v = {v}: pushing v on the stack (contents: {:?})",
                        //     self.stack,
                        //     v = v,
                        // );
                        self.stack.push_front(v);
                    }
                }
            }

            let timer = std::time::Instant::now();
            let mut sccs = Scc::new(body.basic_blocks.len());
            sccs.compute_sccs(&body.basic_blocks);

            let elapsed = timer.elapsed();
            eprintln!(", computed {} SCCs in {} ns", sccs.component_count, elapsed.as_nanos());
        }

        //
        {
            eprint!("SCC tests - {:>30}", "tarjan SCCs (mixed/u32)");

            struct Scc {
                candidate_component_roots: Vec<u32>,
                components: Vec<i32>,
                component_count: usize,
                dfs_numbers: Vec<u32>,
                d: u32,
                stack: VecDeque<u32>,
                visited: MixedBitSet<u32>,
            }

            impl Scc {
                fn new(node_count: usize) -> Self {
                    Self {
                        candidate_component_roots: vec![0; node_count],
                        components: vec![-1; node_count],
                        component_count: 0,
                        dfs_numbers: vec![0; node_count],
                        d: 0,
                        stack: VecDeque::new(),
                        visited: MixedBitSet::new_empty(node_count),
                    }
                }

                fn compute_sccs(&mut self, blocks: &BasicBlocks<'_>) {
                    for (idx, block) in blocks.iter_enumerated() {
                        let edges = block.terminator().edges();
                        if matches!(edges, TerminatorEdges::None) {
                            continue;
                        }

                        let idx = idx.as_u32();
                        if !self.visited.contains(idx) {
                            self.dfs_visit(idx, blocks);
                        }
                    }
                }

                fn dfs_visit(&mut self, v: u32, blocks: &BasicBlocks<'_>) {
                    self.candidate_component_roots[v as usize] = v;
                    self.components[v as usize] = -1;

                    self.d += 1;
                    self.dfs_numbers[v as usize] = self.d;

                    self.visited.insert(v);

                    // println!(
                    //     "dfs_visit, v = {v}, CCR[{v}] = {}, D[{v}] = {}",
                    //     self.candidate_component_roots[v],
                    //     self.dfs_numbers[v],
                    //     v = v
                    // );

                    let idx = unsafe { BasicBlock::from_u32_unchecked(v) };
                    for succ in blocks[idx].terminator().successors() {
                        let w = succ.as_u32();

                        // if w == v {
                        //     panic!("a dang self loop ?!");
                        // }

                        if !self.visited.contains(w) {
                            self.dfs_visit(w, blocks);
                        }

                        // println!(
                        //     "v = {v} - adjacent vertex w = {w}: C[w] = {}, C[v] = {} / D[CCR[w]] = {}, D[CCR[v]] = {} / CCR[v] = {}, CCR[w] = {}",
                        //     self.components[w],
                        //     self.components[v],
                        //     self.dfs_numbers[self.candidate_component_roots[w]],
                        //     self.dfs_numbers[self.candidate_component_roots[v]],
                        //     self.candidate_component_roots[v],
                        //     self.candidate_component_roots[w],
                        //     w = w,
                        //     v = v,
                        // );

                        if self.components[w as usize] == -1
                            && self.dfs_numbers[self.candidate_component_roots[w as usize] as usize]
                                < self.dfs_numbers
                                    [self.candidate_component_roots[v as usize] as usize]
                        {
                            self.candidate_component_roots[v as usize] =
                                self.candidate_component_roots[w as usize];
                        }
                    }

                    // println!(
                    //     "v = {v} - CCR[v] = {}",
                    //     self.candidate_component_roots[v],
                    //     v = v,
                    // );

                    if self.candidate_component_roots[v as usize] == v {
                        self.component_count += 1;
                        self.components[v as usize] = self.component_count as i32;

                        // println!(
                        //     "v = {v} - creating component {} / C[v] = {}",
                        //     self.component_count,
                        //     self.components[v],
                        //     v = v,
                        // );

                        while self.stack.front().is_some()
                            && self.dfs_numbers
                                [*self.stack.front().expect("peek front failed") as usize]
                                > self.dfs_numbers[v as usize]
                        {
                            let w = self.stack.pop_front().expect("pop front failed");
                            self.components[w as usize] = self.components[v as usize];
                            // println!(
                            //     "v = {v} - popping w = {w} off the stack (contents: {:?}) / C[w] = {}, C[v] = {}",
                            //     self.stack,
                            //     self.components[w],
                            //     self.components[v],
                            //     v = v,
                            //     w = w,
                            // );
                        }
                    } else {
                        // println!(
                        //     "v = {v}: pushing v on the stack (contents: {:?})",
                        //     self.stack,
                        //     v = v,
                        // );
                        self.stack.push_front(v);
                    }
                }
            }

            let timer = std::time::Instant::now();
            let mut sccs = Scc::new(body.basic_blocks.len());
            sccs.compute_sccs(&body.basic_blocks);

            let elapsed = timer.elapsed();
            eprintln!(", computed {} SCCs in {} ns", sccs.component_count, elapsed.as_nanos());
        }

        //
        {
            eprint!("SCC tests - {:>30}", "tarjan SCCs (mixed/idx)");

            struct Scc {
                candidate_component_roots: IndexVec<BasicBlock, BasicBlock>,
                components: IndexVec<BasicBlock, Option<BasicBlock>>,
                component_count: u32,
                dfs_numbers: IndexVec<BasicBlock, u32>,
                d: u32,
                stack: VecDeque<BasicBlock>,
                visited: MixedBitSet<BasicBlock>,
            }

            impl Scc {
                fn new(node_count: usize) -> Self {
                    Self {
                        candidate_component_roots: IndexVec::from_raw(vec![
                            unsafe {
                                BasicBlock::from_u32_unchecked(0)
                            };
                            node_count
                        ]),
                        components: IndexVec::from_raw(vec![None; node_count]),
                        component_count: 0,
                        dfs_numbers: IndexVec::from_raw(vec![0; node_count]),
                        d: 0,
                        stack: VecDeque::new(),
                        visited: MixedBitSet::new_empty(node_count),
                    }
                }

                fn compute_sccs(&mut self, blocks: &BasicBlocks<'_>) {
                    for (idx, block) in blocks.iter_enumerated() {
                        let edges = block.terminator().edges();
                        if matches!(edges, TerminatorEdges::None) {
                            continue;
                        }

                        if !self.visited.contains(idx) {
                            self.dfs_visit(idx, blocks);
                        }
                    }
                }

                fn dfs_visit(&mut self, v: BasicBlock, blocks: &BasicBlocks<'_>) {
                    self.candidate_component_roots[v] = v;
                    self.components[v] = None;

                    self.d += 1;
                    self.dfs_numbers[v] = self.d;

                    self.visited.insert(v);

                    // println!(
                    //     "dfs_visit, v = {v}, CCR[{v}] = {}, D[{v}] = {}",
                    //     self.candidate_component_roots[v],
                    //     self.dfs_numbers[v],
                    //     v = v
                    // );

                    for w in blocks[v].terminator().successors() {
                        // if w == v {
                        //     panic!("a dang self loop ?!");
                        // }

                        if !self.visited.contains(w) {
                            self.dfs_visit(w, blocks);
                        }

                        // println!(
                        //     "v = {v} - adjacent vertex w = {w}: C[w] = {}, C[v] = {} / D[CCR[w]] = {}, D[CCR[v]] = {} / CCR[v] = {}, CCR[w] = {}",
                        //     self.components[w],
                        //     self.components[v],
                        //     self.dfs_numbers[self.candidate_component_roots[w]],
                        //     self.dfs_numbers[self.candidate_component_roots[v]],
                        //     self.candidate_component_roots[v],
                        //     self.candidate_component_roots[w],
                        //     w = w,
                        //     v = v,
                        // );

                        if self.components[w].is_none()
                            && self.dfs_numbers[self.candidate_component_roots[w]]
                                < self.dfs_numbers[self.candidate_component_roots[v]]
                        {
                            self.candidate_component_roots[v] = self.candidate_component_roots[w];
                        }
                    }

                    // println!(
                    //     "v = {v} - CCR[v] = {}",
                    //     self.candidate_component_roots[v],
                    //     v = v,
                    // );

                    if self.candidate_component_roots[v] == v {
                        self.component_count += 1;
                        self.components[v] =
                            Some(unsafe { BasicBlock::from_u32_unchecked(self.component_count) });

                        // println!(
                        //     "v = {v} - creating component {} / C[v] = {}",
                        //     self.component_count,
                        //     self.components[v],
                        //     v = v,
                        // );

                        while self.stack.front().is_some()
                            && self.dfs_numbers[*self.stack.front().expect("peek front failed")]
                                > self.dfs_numbers[v]
                        {
                            let w = self.stack.pop_front().expect("pop front failed");
                            self.components[w] = self.components[v];
                            // println!(
                            //     "v = {v} - popping w = {w} off the stack (contents: {:?}) / C[w] = {}, C[v] = {}",
                            //     self.stack,
                            //     self.components[w],
                            //     self.components[v],
                            //     v = v,
                            //     w = w,
                            // );
                        }
                    } else {
                        // println!(
                        //     "v = {v}: pushing v on the stack (contents: {:?})",
                        //     self.stack,
                        //     v = v,
                        // );
                        self.stack.push_front(v);
                    }
                }
            }

            let timer = std::time::Instant::now();
            let mut sccs = Scc::new(body.basic_blocks.len());
            sccs.compute_sccs(&body.basic_blocks);

            let elapsed = timer.elapsed();
            eprintln!(", computed {} SCCs in {} ns", sccs.component_count, elapsed.as_nanos());
        }

        // ---

        {
            eprint!("SCC tests - {:>30}", "nuutila (interval/vec/usize)");

            use rustc_index::interval::IntervalSet;

            struct Nuutila {
                candidate_component_roots: Vec<usize>,
                components: Vec<isize>,
                component_count: usize,
                dfs_numbers: Vec<u32>,
                d: u32,
                visited: Vec<bool>,
                stack_vertex: VecDeque<usize>,
                stack_component: VecDeque<usize>,
                reachability: Vec<IntervalSet<usize>>,
                // reachabilly: Vec<HybridBitSet<usize>>,
            }

            impl Nuutila {
                fn new(node_count: usize) -> Self {
                    Self {
                        candidate_component_roots: vec![0; node_count],
                        components: vec![-1; node_count],
                        component_count: 0,
                        dfs_numbers: vec![0; node_count],
                        d: 0,
                        visited: vec![false; node_count],
                        stack_vertex: VecDeque::new(),
                        stack_component: VecDeque::new(),
                        reachability: vec![IntervalSet::new(node_count); node_count + 1],
                        // ^--- la reachability c'est celle que des composants donc il en faut moins que `node_count` s'il y a au moins un SCC avec > 1 nodes
                        // reachabilly: vec![HybridBitSet::new_empty(node_count); node_count],
                    }
                }

                fn compute_sccs(&mut self, blocks: &BasicBlocks<'_>) {
                    for (idx, block) in blocks.iter_enumerated() {
                        let edges = block.terminator().edges();
                        if matches!(edges, TerminatorEdges::None) {
                            continue;
                        }

                        let idx = idx.as_usize();
                        if !self.visited[idx] {
                            self.dfs_visit(idx, blocks);
                        }
                    }
                }

                // Compute SCCs and reachability only starting where loans appear.
                // We still have the unused blocks in our domain, but won't traverse them.
                fn compute_for_loans(
                    &mut self,
                    borrow_set: &BorrowSet<'_>,
                    blocks: &BasicBlocks<'_>,
                ) {
                    for (_loan_idx, loan) in borrow_set.iter_enumerated() {
                        let block_idx = loan.reserve_location.block;
                        let block = &blocks[block_idx];

                        let edges = block.terminator().edges();
                        if matches!(edges, TerminatorEdges::None) {
                            continue;
                        }

                        let idx = block_idx.as_usize();
                        if !self.visited[idx] {
                            self.dfs_visit(idx, blocks);
                        }
                    }
                }

                fn dfs_visit(&mut self, v: usize, blocks: &BasicBlocks<'_>) {
                    self.candidate_component_roots[v] = v;
                    self.components[v] = -1;

                    self.d += 1;
                    self.dfs_numbers[v] = self.d;

                    self.stack_vertex.push_front(v);
                    let stack_component_height = self.stack_component.len();

                    self.visited[v] = true;

                    // println!(
                    //     "dfs_visit, v = {v}, CCR[{v}] = {}, D[{v}] = {}",
                    //     self.candidate_component_roots[v],
                    //     self.dfs_numbers[v],
                    //     v = v
                    // );

                    let idx = BasicBlock::from_usize(v);
                    for succ in blocks[idx].terminator().successors() {
                        let w = succ.as_usize();

                        if w == v {
                            panic!("a dang self loop ?! at {}", w);
                        }

                        if !self.visited[w] {
                            self.dfs_visit(w, blocks);
                        }

                        // println!(
                        //     "v = {v} - adjacent vertex w = {w}: C[w] = {}, C[v] = {} / D[CCR[w]] = {}, D[CCR[v]] = {} / CCR[v] = {}, CCR[w] = {}",
                        //     self.components[w],
                        //     self.components[v],
                        //     self.dfs_numbers[self.candidate_component_roots[w]],
                        //     self.dfs_numbers[self.candidate_component_roots[v]],
                        //     self.candidate_component_roots[v],
                        //     self.candidate_component_roots[w],
                        //     w = w,
                        //     v = v,
                        // );

                        let component_w = self.components[w];
                        if component_w == -1 {
                            if self.dfs_numbers[self.candidate_component_roots[w]]
                                < self.dfs_numbers[self.candidate_component_roots[v]]
                            {
                                self.candidate_component_roots[v] =
                                    self.candidate_component_roots[w];
                            }
                        } else {
                            assert!(component_w >= 0);

                            // FIXME: check if v -> w is actually a forward edge or not, to avoid unnecessary work if it is
                            self.stack_component.push_front(self.components[w] as usize);
                        }
                    }

                    // println!(
                    //     "v = {v} - CCR[v] = {}",
                    //     self.candidate_component_roots[v],
                    //     v = v,
                    // );

                    if self.candidate_component_roots[v] == v {
                        self.component_count += 1;
                        self.components[v] = self.component_count as isize;

                        // println!(
                        //     "v = {v} - creating component {} / C[v] = {}",
                        //     self.component_count,
                        //     self.components[v],
                        //     v = v,
                        // );

                        // Reachability of C[v]
                        assert!(self.reachability[self.component_count].is_empty());
                        // assert!(self.reachabilly[self.component_count].is_empty());

                        if let Some(&top) = self.stack_vertex.front() {
                            if top != v {
                                // we're adding new component, initialize its reachability: self-loop,
                                // the component can reach itself
                                // self.reachability[self.component_count] =
                                //     (self.component_count, self.component_count).to_interval_set();
                                self.reachability[self.component_count]
                                    .insert(self.component_count);
                            } else {
                                // R[C[v]] should be empty here already, do nothing
                                // if we don't always initialize the reachability of C by default, it would need to be
                                // initialized to "empty" here.
                            }
                        }

                        // process adjacent components
                        while self.stack_component.len() != stack_component_height {
                            let x = self
                                .stack_component
                                .pop_front()
                                .expect("Sc can't be empty at this point");
                            // prevent performing duplicate operations
                            if !self.reachability[self.component_count].contains(x) {
                                // merge reachability information
                                // let r_c_v = self.reachability[self.component_count]
                                //     .union(&self.reachability[x])
                                //     .union(&(x, x).to_interval_set());
                                // self.reachability[self.component_count] = r_c_v;
                                assert_ne!(x, self.component_count);

                                // self.reachability[self.component_count].union(&self.reachability[x]);
                                // self.reachability[self.component_count].insert(x);

                                // let mut r_c_v = self.reachability[self.component_count].clone();
                                // r_c_v.union(&self.reachability[x]);
                                // r_c_v.insert(x);
                                // self.reachability[self.component_count] = r_c_v;

                                let zzz = unsafe {
                                    self.reachability.get_unchecked(x) as *const IntervalSet<usize>
                                };
                                let r_c_v = unsafe {
                                    self.reachability.get_unchecked_mut(self.component_count)
                                };
                                // r_c_v.union(&self.reachability[x]);
                                r_c_v.union(unsafe { &*zzz });
                                r_c_v.insert(x);
                            }

                            // // prevent performing duplicate operations
                            // if !self.reachabilly[self.component_count].contains(x) {
                            //     // merge reachability information
                            //     assert!(x != self.component_count);

                            //     self.reachabilly[self.component_count].insert(x);

                            //     // split the array into two slices, starting at the lowest of the 2
                            //     // the lowest will be the first in the array and the highest the last
                            //     let low = self.component_count.min(x);
                            //     let high = self.component_count.max(x);
                            //     let interval = &mut self.reachabilly[low..=high];
                            //     let (a, b) = interval.split_at_mut(1);
                            //     let (component_reachabilly, x_reachabilly) = if self.component_count == low {
                            //         (&mut a[0], &mut b[b.len() - 1])
                            //     } else {
                            //         (&mut b[0], &mut a[a.len() - 1])
                            //     };

                            //     component_reachabilly.union(x_reachabilly);
                            // }
                        }

                        while let Some(w) = self.stack_vertex.pop_front() {
                            self.components[w] = self.components[v];

                            if w == v {
                                break;
                            }
                        }
                    }
                }
            }

            let timer = std::time::Instant::now();
            let mut sccs = Nuutila::new(body.basic_blocks.len());
            sccs.compute_sccs(&body.basic_blocks);
            // sccs.compute_for_loans(&borrow_set, &body.basic_blocks);

            let elapsed = timer.elapsed();
            eprintln!(", computed {} SCCs in {} ns", sccs.component_count, elapsed.as_nanos());
        }

        {
            eprint!("SCC tests - {:>30}", "nuutila' (interval/vec/usize)");

            use rustc_index::interval::IntervalSet;

            struct Nuutila {
                candidate_component_roots: Vec<usize>,
                components: Vec<isize>,
                component_count: usize,
                dfs_numbers: Vec<u32>,
                d: u32,
                visited: Vec<bool>,
                stack_vertex: VecDeque<usize>,
                stack_component: VecDeque<usize>,
                reachability: Vec<IntervalSet<usize>>,
                // reachabilly: Vec<HybridBitSet<usize>>,
            }

            impl Nuutila {
                fn new(node_count: usize) -> Self {
                    Self {
                        candidate_component_roots: vec![0; node_count],
                        components: vec![-1; node_count],
                        component_count: 0,
                        dfs_numbers: vec![0; node_count],
                        d: 0,
                        visited: vec![false; node_count],
                        stack_vertex: VecDeque::new(),
                        stack_component: VecDeque::new(),
                        reachability: vec![IntervalSet::new(node_count); node_count + 1],
                        // ^--- la reachability c'est celle que des composants donc il en faut moins que `node_count` s'il y a au moins un SCC avec > 1 nodes
                        // reachabilly: vec![HybridBitSet::new_empty(node_count); node_count],
                    }
                }

                fn compute_sccs(&mut self, blocks: &BasicBlocks<'_>) {
                    for (idx, block) in blocks.iter_enumerated() {
                        let edges = block.terminator().edges();
                        if matches!(edges, TerminatorEdges::None) {
                            continue;
                        }

                        let idx = idx.as_usize();
                        if !self.visited[idx] {
                            self.dfs_visit(idx, blocks);
                        }
                    }
                }

                // Compute SCCs and reachability only starting where loans appear.
                // We still have the unused blocks in our domain, but won't traverse them.
                fn compute_for_loans(
                    &mut self,
                    borrow_set: &BorrowSet<'_>,
                    blocks: &BasicBlocks<'_>,
                ) {
                    for (_loan_idx, loan) in borrow_set.iter_enumerated() {
                        let block_idx = loan.reserve_location.block;
                        let block = &blocks[block_idx];

                        let edges = block.terminator().edges();
                        if matches!(edges, TerminatorEdges::None) {
                            continue;
                        }

                        let idx = block_idx.as_usize();
                        if !self.visited[idx] {
                            self.dfs_visit(idx, blocks);
                        }
                    }
                }

                fn dfs_visit(&mut self, v: usize, blocks: &BasicBlocks<'_>) {
                    self.candidate_component_roots[v] = v;
                    self.components[v] = -1;

                    self.d += 1;
                    self.dfs_numbers[v] = self.d;

                    self.stack_vertex.push_front(v);
                    let stack_component_height = self.stack_component.len();

                    self.visited[v] = true;

                    // println!(
                    //     "dfs_visit, v = {v}, CCR[{v}] = {}, D[{v}] = {}",
                    //     self.candidate_component_roots[v],
                    //     self.dfs_numbers[v],
                    //     v = v
                    // );

                    let idx = BasicBlock::from_usize(v);
                    for succ in blocks[idx].terminator().successors() {
                        let w = succ.as_usize();

                        if w == v {
                            panic!("a dang self loop ?! at {}", w);
                        }

                        if !self.visited[w] {
                            self.dfs_visit(w, blocks);
                        }

                        // println!(
                        //     "v = {v} - adjacent vertex w = {w}: C[w] = {}, C[v] = {} / D[CCR[w]] = {}, D[CCR[v]] = {} / CCR[v] = {}, CCR[w] = {}",
                        //     self.components[w],
                        //     self.components[v],
                        //     self.dfs_numbers[self.candidate_component_roots[w]],
                        //     self.dfs_numbers[self.candidate_component_roots[v]],
                        //     self.candidate_component_roots[v],
                        //     self.candidate_component_roots[w],
                        //     w = w,
                        //     v = v,
                        // );

                        let component_w = self.components[w];
                        if component_w == -1 {
                            if self.dfs_numbers[self.candidate_component_roots[w]]
                                < self.dfs_numbers[self.candidate_component_roots[v]]
                            {
                                self.candidate_component_roots[v] =
                                    self.candidate_component_roots[w];
                            }
                        } else {
                            assert!(component_w >= 0);

                            // FIXME: check if v -> w is actually a forward edge or not, to avoid unnecessary work if it is
                            self.stack_component.push_front(self.components[w] as usize);
                        }
                    }

                    // println!(
                    //     "v = {v} - CCR[v] = {}",
                    //     self.candidate_component_roots[v],
                    //     v = v,
                    // );

                    if self.candidate_component_roots[v] == v {
                        self.component_count += 1;
                        self.components[v] = self.component_count as isize;

                        // println!(
                        //     "v = {v} - creating component {} / C[v] = {}",
                        //     self.component_count,
                        //     self.components[v],
                        //     v = v,
                        // );

                        // Reachability of C[v]
                        assert!(self.reachability[self.component_count].is_empty());
                        // assert!(self.reachabilly[self.component_count].is_empty());

                        if let Some(&top) = self.stack_vertex.front() {
                            if top != v {
                                // we're adding new component, initialize its reachability: self-loop,
                                // the component can reach itself
                                // self.reachability[self.component_count] =
                                //     (self.component_count, self.component_count).to_interval_set();
                                self.reachability[self.component_count]
                                    .insert(self.component_count);
                            } else {
                                // R[C[v]] should be empty here already, do nothing
                                // if we don't always initialize the reachability of C by default, it would need to be
                                // initialized to "empty" here.
                            }
                        }

                        // process adjacent components
                        while self.stack_component.len() != stack_component_height {
                            let x = self
                                .stack_component
                                .pop_front()
                                .expect("Sc can't be empty at this point");
                            // prevent performing duplicate operations
                            if !self.reachability[self.component_count].contains(x) {
                                // merge reachability information
                                // let r_c_v = self.reachability[self.component_count]
                                //     .union(&self.reachability[x])
                                //     .union(&(x, x).to_interval_set());
                                // self.reachability[self.component_count] = r_c_v;
                                assert_ne!(x, self.component_count);

                                // self.reachability[self.component_count].union(&self.reachability[x]);
                                // self.reachability[self.component_count].insert(x);

                                // let mut r_c_v = self.reachability[self.component_count].clone();
                                // r_c_v.union(&self.reachability[x]);
                                // r_c_v.insert(x);
                                // self.reachability[self.component_count] = r_c_v;

                                let zzz = unsafe {
                                    self.reachability.get_unchecked(x) as *const IntervalSet<usize>
                                };
                                let r_c_v = unsafe {
                                    self.reachability.get_unchecked_mut(self.component_count)
                                };
                                // r_c_v.union(&self.reachability[x]);
                                r_c_v.union(unsafe { &*zzz });
                                r_c_v.insert(x);
                            }

                            // // prevent performing duplicate operations
                            // if !self.reachabilly[self.component_count].contains(x) {
                            //     // merge reachability information
                            //     assert!(x != self.component_count);

                            //     self.reachabilly[self.component_count].insert(x);

                            //     // split the array into two slices, starting at the lowest of the 2
                            //     // the lowest will be the first in the array and the highest the last
                            //     let low = self.component_count.min(x);
                            //     let high = self.component_count.max(x);
                            //     let interval = &mut self.reachabilly[low..=high];
                            //     let (a, b) = interval.split_at_mut(1);
                            //     let (component_reachabilly, x_reachabilly) = if self.component_count == low {
                            //         (&mut a[0], &mut b[b.len() - 1])
                            //     } else {
                            //         (&mut b[0], &mut a[a.len() - 1])
                            //     };

                            //     component_reachabilly.union(x_reachabilly);
                            // }
                        }

                        while let Some(w) = self.stack_vertex.pop_front() {
                            self.components[w] = self.components[v];

                            if w == v {
                                break;
                            }
                        }
                    }
                }
            }

            let timer = std::time::Instant::now();
            let mut sccs = Nuutila::new(body.basic_blocks.len());
            // sccs.compute_sccs(&body.basic_blocks);
            sccs.compute_for_loans(&borrow_set, &body.basic_blocks);

            let elapsed = timer.elapsed();
            eprintln!(", computed {} SCCs in {} ns", sccs.component_count, elapsed.as_nanos());
        }

        {
            eprint!("SCC tests - {:>30}", "nuutila (interval/dense/usize)");

            use rustc_index::interval::IntervalSet;

            struct Nuutila {
                candidate_component_roots: Vec<usize>,
                components: Vec<isize>,
                component_count: usize,
                dfs_numbers: Vec<u32>,
                d: u32,
                visited: DenseBitSet<usize>,
                stack_vertex: VecDeque<usize>,
                stack_component: VecDeque<usize>,
                reachability: Vec<IntervalSet<usize>>,
                // reachabilly: Vec<HybridBitSet<usize>>,
            }

            impl Nuutila {
                fn new(node_count: usize) -> Self {
                    Self {
                        candidate_component_roots: vec![0; node_count],
                        components: vec![-1; node_count],
                        component_count: 0,
                        dfs_numbers: vec![0; node_count],
                        d: 0,
                        visited: DenseBitSet::new_empty(node_count),
                        stack_vertex: VecDeque::new(),
                        stack_component: VecDeque::new(),
                        reachability: vec![IntervalSet::new(node_count); node_count + 1],
                        // ^--- la reachability c'est celle que des composants donc il en faut moins que `node_count` s'il y a au moins un SCC avec > 1 nodes
                        // reachabilly: vec![HybridBitSet::new_empty(node_count); node_count],
                    }
                }

                // Compute SCCs and reachability only starting where loans appear.
                // We still have the unused blocks in our domain, but won't traverse them.
                fn compute_for_loans(
                    &mut self,
                    borrow_set: &BorrowSet<'_>,
                    blocks: &BasicBlocks<'_>,
                ) {
                    for (_loan_idx, loan) in borrow_set.iter_enumerated() {
                        let block_idx = loan.reserve_location.block;
                        let block = &blocks[block_idx];

                        let edges = block.terminator().edges();
                        if matches!(edges, TerminatorEdges::None) {
                            continue;
                        }

                        let idx = block_idx.as_usize();
                        if !self.visited.contains(idx) {
                            self.dfs_visit(idx, blocks);
                        }
                    }
                }

                fn compute_sccs(&mut self, blocks: &BasicBlocks<'_>) {
                    for (idx, block) in blocks.iter_enumerated() {
                        let edges = block.terminator().edges();
                        if matches!(edges, TerminatorEdges::None) {
                            continue;
                        }

                        let idx = idx.as_usize();
                        if !self.visited.contains(idx) {
                            self.dfs_visit(idx, blocks);
                        }
                    }
                }

                fn dfs_visit(&mut self, v: usize, blocks: &BasicBlocks<'_>) {
                    self.candidate_component_roots[v] = v;
                    self.components[v] = -1;

                    self.d += 1;
                    self.dfs_numbers[v] = self.d;

                    self.stack_vertex.push_front(v);
                    let stack_component_height = self.stack_component.len();

                    self.visited.insert(v);

                    // println!(
                    //     "dfs_visit, v = {v}, CCR[{v}] = {}, D[{v}] = {}",
                    //     self.candidate_component_roots[v],
                    //     self.dfs_numbers[v],
                    //     v = v
                    // );

                    let idx = BasicBlock::from_usize(v);
                    for succ in blocks[idx].terminator().successors() {
                        let w = succ.as_usize();

                        if w == v {
                            panic!("a dang self loop ?! at {}", w);
                        }

                        if !self.visited.contains(w) {
                            self.dfs_visit(w, blocks);
                        }

                        // println!(
                        //     "v = {v} - adjacent vertex w = {w}: C[w] = {}, C[v] = {} / D[CCR[w]] = {}, D[CCR[v]] = {} / CCR[v] = {}, CCR[w] = {}",
                        //     self.components[w],
                        //     self.components[v],
                        //     self.dfs_numbers[self.candidate_component_roots[w]],
                        //     self.dfs_numbers[self.candidate_component_roots[v]],
                        //     self.candidate_component_roots[v],
                        //     self.candidate_component_roots[w],
                        //     w = w,
                        //     v = v,
                        // );

                        let component_w = self.components[w];
                        if component_w == -1 {
                            if self.dfs_numbers[self.candidate_component_roots[w]]
                                < self.dfs_numbers[self.candidate_component_roots[v]]
                            {
                                self.candidate_component_roots[v] =
                                    self.candidate_component_roots[w];
                            }
                        } else {
                            assert!(component_w >= 0);

                            // FIXME: check if v -> w is actually a forward edge or not, to avoid unnecessary work if it is
                            self.stack_component.push_front(self.components[w] as usize);
                        }
                    }

                    // println!(
                    //     "v = {v} - CCR[v] = {}",
                    //     self.candidate_component_roots[v],
                    //     v = v,
                    // );

                    if self.candidate_component_roots[v] == v {
                        self.component_count += 1;
                        self.components[v] = self.component_count as isize;

                        // println!(
                        //     "v = {v} - creating component {} / C[v] = {}",
                        //     self.component_count,
                        //     self.components[v],
                        //     v = v,
                        // );

                        // Reachability of C[v]
                        assert!(self.reachability[self.component_count].is_empty());
                        // assert!(self.reachabilly[self.component_count].is_empty());

                        if let Some(&top) = self.stack_vertex.front() {
                            if top != v {
                                // we're adding new component, initialize its reachability: self-loop,
                                // the component can reach itself
                                // self.reachability[self.component_count] =
                                //     (self.component_count, self.component_count).to_interval_set();
                                self.reachability[self.component_count]
                                    .insert(self.component_count);
                            } else {
                                // R[C[v]] should be empty here already, do nothing
                                // if we don't always initialize the reachability of C by default, it would need to be
                                // initialized to "empty" here.
                            }
                        }

                        // process adjacent components
                        while self.stack_component.len() != stack_component_height {
                            let x = self
                                .stack_component
                                .pop_front()
                                .expect("Sc can't be empty at this point");
                            // prevent performing duplicate operations
                            if !self.reachability[self.component_count].contains(x) {
                                // merge reachability information
                                // let r_c_v = self.reachability[self.component_count]
                                //     .union(&self.reachability[x])
                                //     .union(&(x, x).to_interval_set());
                                // self.reachability[self.component_count] = r_c_v;
                                assert_ne!(x, self.component_count);

                                // self.reachability[self.component_count].union(&self.reachability[x]);
                                // self.reachability[self.component_count].insert(x);

                                // let mut r_c_v = self.reachability[self.component_count].clone();
                                // r_c_v.union(&self.reachability[x]);
                                // r_c_v.insert(x);
                                // self.reachability[self.component_count] = r_c_v;

                                let zzz = unsafe {
                                    self.reachability.get_unchecked(x) as *const IntervalSet<usize>
                                };
                                let r_c_v = unsafe {
                                    self.reachability.get_unchecked_mut(self.component_count)
                                };
                                // r_c_v.union(&self.reachability[x]);
                                r_c_v.union(unsafe { &*zzz });
                                r_c_v.insert(x);
                            }

                            // // prevent performing duplicate operations
                            // if !self.reachabilly[self.component_count].contains(x) {
                            //     // merge reachability information
                            //     assert!(x != self.component_count);

                            //     self.reachabilly[self.component_count].insert(x);

                            //     // split the array into two slices, starting at the lowest of the 2
                            //     // the lowest will be the first in the array and the highest the last
                            //     let low = self.component_count.min(x);
                            //     let high = self.component_count.max(x);
                            //     let interval = &mut self.reachabilly[low..=high];
                            //     let (a, b) = interval.split_at_mut(1);
                            //     let (component_reachabilly, x_reachabilly) = if self.component_count == low {
                            //         (&mut a[0], &mut b[b.len() - 1])
                            //     } else {
                            //         (&mut b[0], &mut a[a.len() - 1])
                            //     };

                            //     component_reachabilly.union(x_reachabilly);
                            // }
                        }

                        while let Some(w) = self.stack_vertex.pop_front() {
                            self.components[w] = self.components[v];

                            if w == v {
                                break;
                            }
                        }
                    }
                }
            }

            let timer = std::time::Instant::now();
            let mut sccs = Nuutila::new(body.basic_blocks.len());
            // sccs.compute_sccs(&body.basic_blocks);
            sccs.compute_for_loans(&borrow_set, &body.basic_blocks);

            let elapsed = timer.elapsed();
            eprintln!(", computed {} SCCs in {} ns", sccs.component_count, elapsed.as_nanos());
        }

        {
            eprint!("SCC tests - {:>30}", "nuutila (mixed/vec/usize)");

            struct Nuutila {
                candidate_component_roots: Vec<usize>,
                components: Vec<isize>,
                component_count: usize,
                dfs_numbers: Vec<u32>,
                d: u32,
                visited: Vec<bool>,
                stack_vertex: VecDeque<usize>,
                stack_component: VecDeque<usize>,
                // reachability: Vec<IntervalSet<usize>>,
                reachability: Vec<MixedBitSet<usize>>,
            }

            impl Nuutila {
                fn new(node_count: usize) -> Self {
                    Self {
                        candidate_component_roots: vec![0; node_count],
                        components: vec![-1; node_count],
                        component_count: 0,
                        dfs_numbers: vec![0; node_count],
                        d: 0,
                        visited: vec![false; node_count],
                        stack_vertex: VecDeque::new(),
                        stack_component: VecDeque::new(),
                        reachability: vec![MixedBitSet::new_empty(node_count); node_count + 1],
                        // ^--- la reachability c'est celle que des composants donc il en faut moins que `node_count` s'il y a au moins un SCC avec > 1 nodes
                        // reachabilly: vec![HybridBitSet::new_empty(node_count); node_count],
                    }
                }

                // Compute SCCs and reachability only starting where loans appear.
                // We still have the unused blocks in our domain, but won't traverse them.
                fn compute_for_loans(
                    &mut self,
                    borrow_set: &BorrowSet<'_>,
                    blocks: &BasicBlocks<'_>,
                ) {
                    for (_loan_idx, loan) in borrow_set.iter_enumerated() {
                        let block_idx = loan.reserve_location.block;
                        let block = &blocks[block_idx];

                        let edges = block.terminator().edges();
                        if matches!(edges, TerminatorEdges::None) {
                            continue;
                        }

                        let idx = block_idx.as_usize();
                        if !self.visited[idx] {
                            self.dfs_visit(idx, blocks);
                        }
                    }
                }

                fn compute_sccs(&mut self, blocks: &BasicBlocks<'_>) {
                    for (idx, block) in blocks.iter_enumerated() {
                        let edges = block.terminator().edges();
                        if matches!(edges, TerminatorEdges::None) {
                            continue;
                        }

                        let idx = idx.as_usize();
                        if !self.visited[idx] {
                            self.dfs_visit(idx, blocks);
                        }
                    }
                }

                fn dfs_visit(&mut self, v: usize, blocks: &BasicBlocks<'_>) {
                    self.candidate_component_roots[v] = v;
                    self.components[v] = -1;

                    self.d += 1;
                    self.dfs_numbers[v] = self.d;

                    self.stack_vertex.push_front(v);
                    let stack_component_height = self.stack_component.len();

                    self.visited[v] = true;

                    // println!(
                    //     "dfs_visit, v = {v}, CCR[{v}] = {}, D[{v}] = {}",
                    //     self.candidate_component_roots[v],
                    //     self.dfs_numbers[v],
                    //     v = v
                    // );

                    let idx = BasicBlock::from_usize(v);
                    for succ in blocks[idx].terminator().successors() {
                        let w = succ.as_usize();

                        if w == v {
                            panic!("a dang self loop ?! at {}", w);
                        }

                        if !self.visited[w] {
                            self.dfs_visit(w, blocks);
                        }

                        // println!(
                        //     "v = {v} - adjacent vertex w = {w}: C[w] = {}, C[v] = {} / D[CCR[w]] = {}, D[CCR[v]] = {} / CCR[v] = {}, CCR[w] = {}",
                        //     self.components[w],
                        //     self.components[v],
                        //     self.dfs_numbers[self.candidate_component_roots[w]],
                        //     self.dfs_numbers[self.candidate_component_roots[v]],
                        //     self.candidate_component_roots[v],
                        //     self.candidate_component_roots[w],
                        //     w = w,
                        //     v = v,
                        // );

                        let component_w = self.components[w];
                        if component_w == -1 {
                            if self.dfs_numbers[self.candidate_component_roots[w]]
                                < self.dfs_numbers[self.candidate_component_roots[v]]
                            {
                                self.candidate_component_roots[v] =
                                    self.candidate_component_roots[w];
                            }
                        } else {
                            assert!(component_w >= 0);

                            // FIXME: check if v -> w is actually a forward edge or not, to avoid unnecessary work if it is
                            self.stack_component.push_front(self.components[w] as usize);
                        }
                    }

                    // println!(
                    //     "v = {v} - CCR[v] = {}",
                    //     self.candidate_component_roots[v],
                    //     v = v,
                    // );

                    if self.candidate_component_roots[v] == v {
                        self.component_count += 1;
                        self.components[v] = self.component_count as isize;

                        // println!(
                        //     "v = {v} - creating component {} / C[v] = {}",
                        //     self.component_count,
                        //     self.components[v],
                        //     v = v,
                        // );

                        // Reachability of C[v]
                        assert!(self.reachability[self.component_count].is_empty());
                        // assert!(self.reachabilly[self.component_count].is_empty());

                        if let Some(&top) = self.stack_vertex.front() {
                            if top != v {
                                // we're adding new component, initialize its reachability: self-loop,
                                // the component can reach itself
                                // self.reachability[self.component_count] =
                                //     (self.component_count, self.component_count).to_interval_set();
                                self.reachability[self.component_count]
                                    .insert(self.component_count);
                            } else {
                                // R[C[v]] should be empty here already, do nothing
                                // if we don't always initialize the reachability of C by default, it would need to be
                                // initialized to "empty" here.
                            }
                        }

                        // process adjacent components
                        while self.stack_component.len() != stack_component_height {
                            let x = self
                                .stack_component
                                .pop_front()
                                .expect("Sc can't be empty at this point");
                            // prevent performing duplicate operations
                            if !self.reachability[self.component_count].contains(x) {
                                // merge reachability information
                                // let r_c_v = self.reachability[self.component_count]
                                //     .union(&self.reachability[x])
                                //     .union(&(x, x).to_interval_set());
                                // self.reachability[self.component_count] = r_c_v;
                                assert_ne!(x, self.component_count);

                                // self.reachability[self.component_count].union(&self.reachability[x]);
                                // self.reachability[self.component_count].insert(x);

                                // let mut r_c_v = self.reachability[self.component_count].clone();
                                // r_c_v.union(&self.reachability[x]);
                                // r_c_v.insert(x);
                                // self.reachability[self.component_count] = r_c_v;

                                let zzz = unsafe { self.reachability.get_unchecked(x) as *const _ };
                                let r_c_v = unsafe {
                                    self.reachability.get_unchecked_mut(self.component_count)
                                };
                                // r_c_v.union(&self.reachability[x]);
                                r_c_v.union(unsafe { &*zzz });
                                r_c_v.insert(x);
                            }

                            // // prevent performing duplicate operations
                            // if !self.reachabilly[self.component_count].contains(x) {
                            //     // merge reachability information
                            //     assert!(x != self.component_count);

                            //     self.reachabilly[self.component_count].insert(x);

                            //     // split the array into two slices, starting at the lowest of the 2
                            //     // the lowest will be the first in the array and the highest the last
                            //     let low = self.component_count.min(x);
                            //     let high = self.component_count.max(x);
                            //     let interval = &mut self.reachabilly[low..=high];
                            //     let (a, b) = interval.split_at_mut(1);
                            //     let (component_reachabilly, x_reachabilly) = if self.component_count == low {
                            //         (&mut a[0], &mut b[b.len() - 1])
                            //     } else {
                            //         (&mut b[0], &mut a[a.len() - 1])
                            //     };

                            //     component_reachabilly.union(x_reachabilly);
                            // }
                        }

                        while let Some(w) = self.stack_vertex.pop_front() {
                            self.components[w] = self.components[v];

                            if w == v {
                                break;
                            }
                        }
                    }
                }
            }

            let timer = std::time::Instant::now();
            let mut sccs = Nuutila::new(body.basic_blocks.len());
            // sccs.compute_sccs(&body.basic_blocks);
            sccs.compute_for_loans(&borrow_set, &body.basic_blocks);

            let elapsed = timer.elapsed();
            eprintln!(", computed {} SCCs in {} ns", sccs.component_count, elapsed.as_nanos());
        }

        {
            // // CFG stats, blocks: 18073, statements: 67351
            // // borrow stats, locals: 21513, loans: 6232
            // if body.basic_blocks.len() == 18073
            //     && body.local_decls.len() == 21513
            //     && borrow_set.len() == 6232
            {
                eprint!("SCC tests - {:>30}", "tagebility");

                let timer = std::time::Instant::now();

                // Compute `transitive_predecessors` and `adjacent_predecessors`.
                let mut transitive_predecessors = IndexVec::from_elem_n(
                    DenseBitSet::new_empty(body.basic_blocks.len()),
                    body.basic_blocks.len(),
                );
                let mut adjacent_predecessors = transitive_predecessors.clone();
                // The stack is initially a reversed postorder traversal of the CFG. However, we might add
                // add blocks again to the stack if we have loops.
                let mut stack =
                    body.basic_blocks.reverse_postorder().iter().rev().copied().collect::<Vec<_>>();
                // We keep track of all blocks that are currently not in the stack.
                let mut not_in_stack = DenseBitSet::new_empty(body.basic_blocks.len());
                while let Some(block) = stack.pop() {
                    not_in_stack.insert(block);

                    // Loop over all successors to the block and add `block` to their predecessors.
                    for succ_block in body.basic_blocks[block].terminator().successors() {
                        // Keep track of whether the transitive predecessors of `succ_block` has changed.
                        let mut changed = false;

                        // Insert `block` in `succ_block`s predecessors.
                        if adjacent_predecessors[succ_block].insert(block) {
                            // Remember that `adjacent_predecessors` is a subset of
                            // `transitive_predecessors`.
                            changed |= transitive_predecessors[succ_block].insert(block);
                        }

                        // Add all transitive predecessors of `block` to the transitive predecessors of
                        // `succ_block`.
                        if block != succ_block {
                            let (blocks_predecessors, succ_blocks_predecessors) =
                                transitive_predecessors.pick2_mut(block, succ_block);
                            changed |= succ_blocks_predecessors.union(blocks_predecessors);

                            // Check if the `succ_block`s transitive predecessors changed. If so, we may
                            // need to add it to the stack again.
                            if changed && not_in_stack.remove(succ_block) {
                                stack.push(succ_block);
                            }
                        }
                    }

                    // debug_assert!(
                    //     transitive_predecessors[block].superset(&adjacent_predecessors[block])
                    // );
                }

                let elapsed = timer.elapsed();

                let w = 2 + Sccs::<BasicBlock, usize>::new(&body.basic_blocks).num_sccs().ilog10()
                    as usize;
                //  computed 2757 SCCs in 1535259 ns
                //  computed 11600 SCCs in 6472961 ns
                //  2 + ilog
                eprintln!(", {:w$} predecessors in {} ns", " ", elapsed.as_nanos());
            }
        }
    }

    // if std::env::var("LETSGO").is_ok() {
    //     eprintln!(
    //         "initialization checks: {} locals out of {} total, at {} statements out of {} total, in {} blocks out of {} total, in {} transitive blocks out {} total, cyclic cfg: {}",
    //         mbcx.locals_checked_for_initialization.len(),
    //         body.local_decls.len(),
    //         mbcx.locals_checked_for_initialization
    //             .iter()
    //             .flat_map(|(_, locations)| locations.iter())
    //             .collect::<FxHashSet<_>>()
    //             .len(),
    //         body.basic_blocks
    //             .iter_enumerated()
    //             .map(|(_, bb)| bb.statements.len() + 1)
    //             .sum::<usize>(),
    //         mbcx.locals_checked_for_initialization
    //             .iter()
    //             .flat_map(|(_, locations)| locations.iter().map(|l| l.block))
    //             .collect::<FxHashSet<_>>()
    //             .len(),
    //         body.basic_blocks.len(),
    //         mbcx.locals_checked_for_initialization
    //             .iter()
    //             .flat_map(|(_, locations)| locations.iter().map(|l| l.block))
    //             .flat_map(|block| std::iter::once(block)
    //                 .chain(body.basic_blocks.predecessors()[block].iter().copied()))
    //             .collect::<FxHashSet<_>>()
    //             .len(),
    //         body.basic_blocks.len(),
    //         body.basic_blocks.is_cfg_cyclic(),
    //     );

    //     // eprintln!(
    //     //     "initialization checks: {} move paths out of {} total, at {} statements out of {} total",
    //     //     mbcx.locals_checked_for_initialization.len(),
    //     //     move_data.init_path_map.len(),
    //     //     mbcx.locals_checked_for_initialization
    //     //         .iter()
    //     //         .flat_map(|(_, locations)| locations.iter())
    //     //         .collect::<FxHashSet<_>>()
    //     //         .len(),
    //     //     body.basic_blocks
    //     //         .iter_enumerated()
    //     //         .map(|(_, bb)| bb.statements.len() + 1)
    //     //         .sum::<usize>(),
    //     // );
    // }

    if let Some(consumer) = &mut root_cx.consumer {
        consumer.insert_body(
            def,
            BodyWithBorrowckFacts {
                body: body_owned,
                promoted,
                borrow_set,
                region_inference_context: regioncx,
                location_table: polonius_input.as_ref().map(|_| location_table),
                input_facts: polonius_input,
                output_facts: polonius_output,
            },
        );
    }

    debug!("do_mir_borrowck: result = {:#?}", result);

    result
}

fn compute_cyclic_dataflow<'mir, 'tcx>(
    body: &Body<'tcx>,
    borrows: Borrows<'mir, 'tcx>,
    uninits: MaybeUninitializedPlaces<'mir, 'tcx>,
    ever_inits: EverInitializedPlaces<'mir, 'tcx>,
    vis: &mut MirBorrowckCtxt<'mir, '_, 'tcx>,
    // flow_entry_states: &IndexVec<BasicBlock, BorrowckDomain>,
) {
    use rustc_data_structures::work_queue::WorkQueue;
    use rustc_middle::mir;
    use rustc_mir_dataflow::{Direction, Forward, JoinSemiLattice};

    struct AnalysisHolder<'tcx, T: Analysis<'tcx>> {
        // results: IndexVec<BasicBlock, T::Domain>,
        lazy_results: IndexVec<BasicBlock, Option<T::Domain>>,
        dirty_queue: WorkQueue<BasicBlock>,
    }

    impl<'tcx, T: Analysis<'tcx>> AnalysisHolder<'tcx, T> {
        fn new(body: &Body<'tcx>, analysis: &T) -> Self {
            // let mut results =
            //     IndexVec::from_fn_n(|_| analysis.bottom_value(body), body.basic_blocks.len());
            // analysis.initialize_start_block(body, &mut results[mir::START_BLOCK]);

            let mut lazy_results = IndexVec::from_elem_n(None, body.basic_blocks.len());
            lazy_results[mir::START_BLOCK] = Some(analysis.bottom_value(body));
            analysis.initialize_start_block(body, lazy_results[mir::START_BLOCK].as_mut().unwrap());

            Self {
                // results,
                lazy_results,
                dirty_queue: WorkQueue::with_none(body.basic_blocks.len()),
            }
        }
    }

    // FIXME: lazify this
    // let mut results = IndexVec::from_fn_n(|_| analysis.bottom_value(body), body.basic_blocks.len());
    // analysis.initialize_start_block(body, &mut results[mir::START_BLOCK]);

    let mut borrows_holder = AnalysisHolder::new(body, &borrows);
    let mut uninits_holder = AnalysisHolder::new(body, &uninits);
    let mut ever_inits_holder = AnalysisHolder::new(body, &ever_inits);

    // let mut results: IndexVec<BasicBlock, Option<A::Domain>> =
    //     IndexVec::from_elem_n(None, body.basic_blocks.len());
    // // Ensure the start block has some state in it;
    // results[mir::START_BLOCK] = Some(analysis.bottom_value(body));
    // analysis.initialize_start_block(body, results[mir::START_BLOCK].as_mut().unwrap());

    // We'll compute dataflow over the SCCs.
    let sccs = body.basic_blocks.sccs();

    // Worklist for per-SCC iterations
    // let mut dirty_queue: WorkQueue<BasicBlock> = WorkQueue::with_none(body.basic_blocks.len());

    // `state` is not actually used between iterations; this is just an optimization to avoid
    // reallocating every iteration.
    // let mut state = BorrowckDomain {
    //     borrows: borrows.bottom_value(body),
    //     uninits: uninits.bottom_value(body),
    //     ever_inits: ever_inits.bottom_value(body),
    // };

    let mut analysis = Borrowck { borrows, uninits, ever_inits };
    let mut state = analysis.bottom_value(body);

    for &scc in &sccs.queue {
        let blocks_in_scc = sccs.sccs[scc as usize].len();
        // eprintln!(
        //     "X - entering scc {} out of {}, there are {} blocks in there: {:?}",
        //     scc, sccs.component_count, blocks_in_scc, sccs.sccs[scc as usize]
        // );

        #[inline(always)]
        fn propagate<'tcx, A: Analysis<'tcx>>(
            body: &Body<'tcx>,
            block: BasicBlock,
            holder: &mut AnalysisHolder<'tcx, A>,
            state: &mut A::Domain,
            analysis: &mut A,
            mut propagate: impl FnMut(&mut AnalysisHolder<'tcx, A>, BasicBlock, &A::Domain),
        ) {
            // Apply the block's effects without visiting
            Forward::apply_effects_in_block(
                analysis,
                body,
                state,
                block,
                &body[block],
                |target: BasicBlock, state: &A::Domain| {
                    propagate(holder, target, state);
                },
            );
        }

        // Fast-path, the overwhelmingly most common case of having a single block in this SCC.
        if blocks_in_scc == 1 {
            let block = sccs.sccs[scc as usize][0];
            let block_data = &body[block];

            // eprintln!("A1 - entering scc {scc}'s block: {:?}", block);
            // eprintln!(
            //     "A2 - is {block:?} state ready, borrows: {}, uninits: {}, ever_inits: {}",
            //     borrows_holder.lazy_results[block].is_some(),
            //     uninits_holder.lazy_results[block].is_some(),
            //     ever_inits_holder.lazy_results[block].is_some(),
            // );

            // eprintln!(
            //     "1b - scc {scc} is a single block, computing and visiting {block:?} at the same time"
            // );

            // // tmp: verifying contents on entry
            // if let Some(borrows) = &borrows_holder.lazy_results[block] {
            //     assert_eq!(
            //         &flow_entry_states[block].borrows, borrows,
            //         "borrows of block {block:?} differ"
            //     );
            // }
            // if let Some(ever_inits) = &ever_inits_holder.lazy_results[block] {
            //     assert_eq!(
            //         &flow_entry_states[block].ever_inits, ever_inits,
            //         "ever_inits of block {block:?} differ"
            //     );
            // }
            // if let Some(uninits) = &uninits_holder.lazy_results[block] {
            //     assert_eq!(
            //         &flow_entry_states[block].uninits, uninits,
            //         "uninits of block {block:?} differ"
            //     );
            // }

            // Apply effects in the block's statements.
            let analysis = &mut analysis;
            let Some(borrows) = borrows_holder.lazy_results[block].take() else {
                continue;
            };
            let Some(uninits) = uninits_holder.lazy_results[block].take() else {
                continue;
            };
            let Some(ever_inits) = ever_inits_holder.lazy_results[block].take() else {
                continue;
            };
            let mut block_state = BorrowckDomain { borrows, uninits, ever_inits };

            // // tmp: verifying the contents on entry
            // assert_eq!(
            //     flow_entry_states[block].borrows, block_state.borrows,
            //     "borrows of block {block:?} differ"
            // );
            // assert_eq!(
            //     flow_entry_states[block].ever_inits, block_state.ever_inits,
            //     "ever_inits of block {block:?} differ"
            // );
            // assert_eq!(
            //     flow_entry_states[block].uninits, block_state.uninits,
            //     "uninits of block {block:?} differ"
            // );

            // eprintln!("1c1 - {block:?} uninits start state: \n{:?}", block_state.uninits);

            vis.visit_block_start(&mut block_state);

            for (statement_index, statement) in block_data.statements.iter().enumerate() {
                let location = Location { block, statement_index };
                analysis.apply_early_statement_effect(&mut block_state, statement, location);
                vis.visit_after_early_statement_effect(analysis, &block_state, statement, location);

                analysis.apply_primary_statement_effect(&mut block_state, statement, location);
                vis.visit_after_primary_statement_effect(
                    analysis,
                    &block_state,
                    statement,
                    location,
                );
            }

            // eprintln!("1c2 - {block:?} uninits post statements state: \n{:?}", block_state.uninits);

            // Apply effects in the block terminator.
            let terminator = block_data.terminator();
            let location = Location { block, statement_index: block_data.statements.len() };
            analysis.apply_early_terminator_effect(&mut block_state, terminator, location);
            vis.visit_after_early_terminator_effect(analysis, &block_state, terminator, location);

            // eprintln!("1c3 - {block:?} uninits early terminator state: \n{:?}", block_state.uninits);

            let edges =
                analysis.apply_primary_terminator_effect(&mut block_state, terminator, location);
            vis.visit_after_primary_terminator_effect(analysis, &block_state, terminator, location);

            // eprintln!("1c4 - {block:?} uninits post terminator state: \n{:?}", block_state.uninits);

            vis.visit_block_end(&mut block_state);

            #[inline(always)]
            fn propagate_single_edge<'tcx, A: Analysis<'tcx>>(
                holder: &mut AnalysisHolder<'tcx, A>,
                state: A::Domain,
                _block: BasicBlock,
                target: BasicBlock,
                _tag: &str,
                _kind: &str,
                _edge: &str,
            ) {
                // eprintln!(
                //     "{tag} - propagating {kind} state from {block:?} to {edge} edge: {target:?} (init: {})",
                //     holder.lazy_results[target].is_some()
                // );
                match holder.lazy_results[target].as_mut() {
                    None => {
                        holder.lazy_results[target] = Some(state);
                    }
                    Some(existing_state) => {
                        existing_state.join(&state);
                    }
                }
            }

            #[inline(always)]
            fn propagate_optional_single_edge<'tcx, A: Analysis<'tcx>>(
                holder: &mut AnalysisHolder<'tcx, A>,
                state: &A::Domain,
                _block: BasicBlock,
                target: BasicBlock,
                _tag: &str,
                _kind: &str,
                _edge: &str,
            ) {
                // eprintln!(
                //     "{tag} - propagating {kind} state from {block:?} to {edge} edge: {target:?} (init: {})",
                //     holder.lazy_results[target].is_some()
                // );
                match holder.lazy_results[target].as_mut() {
                    None => {
                        holder.lazy_results[target] = Some(state.clone());
                    }
                    Some(existing_state) => {
                        existing_state.join(state);
                    }
                }
            }

            #[inline(always)]
            fn propagate_double_edge<'tcx, A: Analysis<'tcx>>(
                holder: &mut AnalysisHolder<'tcx, A>,
                state: A::Domain,
                _block: BasicBlock,
                target: BasicBlock,
                unwind: BasicBlock,
                _tag: &str,
                _kind: &str,
                _edge: &str,
            ) {
                // eprintln!(
                //     "{tag} - propagating {kind} state from {block:?} to {edge} edge: {target:?} (init: {}), {unwind:?} (init: {})",
                //     holder.lazy_results[target].is_some(),
                //     holder.lazy_results[unwind].is_some(),
                // );
                // We have two *distinct* successors.
                //
                // We could use an `_unchecked` version of `pick2_mut` if it existed: we know the
                // indices are disjoint and in-bounds.
                match holder.lazy_results.pick2_mut(target, unwind) {
                    (None, None) => {
                        // We need to initialize both successors with our own block state, we need a
                        // clone.
                        holder.lazy_results[target] = Some(state.clone());
                        holder.lazy_results[unwind] = Some(state);
                    }
                    (None, Some(unwind_state)) => {
                        // No need to clone, only one successor is not initialized yet.
                        unwind_state.join(&state);
                        holder.lazy_results[target] = Some(state);
                    }
                    (Some(target_state), None) => {
                        // No need to clone, only one successor is not initialized yet.
                        target_state.join(&state);
                        holder.lazy_results[unwind] = Some(state);
                    }
                    (Some(target_state), Some(unwind_state)) => {
                        // The successors have already been initialized by their other parents, we
                        // merge our block state there.
                        target_state.join(&state);
                        unwind_state.join(&state);
                    }
                }
            }

            // The current block is done, and the visitor was notified at every step. We now take care
            // of the successors' state.
            match edges {
                TerminatorEdges::None => {}
                TerminatorEdges::Single(target) => {
                    // We have a single successor, our own state can either be moved to it, or dropped.
                    // propagate(target, block_state);
                    propagate_single_edge(
                        &mut borrows_holder,
                        block_state.borrows,
                        block,
                        target,
                        "A3a",
                        "borrows",
                        "single",
                    );
                    propagate_single_edge(
                        &mut uninits_holder,
                        block_state.uninits,
                        block,
                        target,
                        "A3b",
                        "uninits",
                        "single",
                    );
                    propagate_single_edge(
                        &mut ever_inits_holder,
                        block_state.ever_inits,
                        block,
                        target,
                        "A3c",
                        "ever_inits",
                        "single",
                    );
                }
                TerminatorEdges::Double(target, unwind) if target == unwind => {
                    // Why are we generating this shape in MIR building :thinking: ? Either way, we also
                    // have a single successor here.
                    // propagate(target, block_state);
                    propagate_single_edge(
                        &mut borrows_holder,
                        block_state.borrows,
                        block,
                        target,
                        "A4a",
                        "borrows",
                        "single double",
                    );
                    propagate_single_edge(
                        &mut uninits_holder,
                        block_state.uninits,
                        block,
                        target,
                        "A4b",
                        "uninits",
                        "single double",
                    );
                    propagate_single_edge(
                        &mut ever_inits_holder,
                        block_state.ever_inits,
                        block,
                        target,
                        "A4c",
                        "ever_inits",
                        "single double",
                    );
                }
                TerminatorEdges::Double(target, unwind) => {
                    propagate_double_edge(
                        &mut borrows_holder,
                        block_state.borrows,
                        block,
                        target,
                        unwind,
                        "A5a",
                        "borrows",
                        "double",
                    );
                    propagate_double_edge(
                        &mut uninits_holder,
                        block_state.uninits,
                        block,
                        target,
                        unwind,
                        "A5b",
                        "uninits",
                        "double",
                    );
                    propagate_double_edge(
                        &mut ever_inits_holder,
                        block_state.ever_inits,
                        block,
                        target,
                        unwind,
                        "A5c",
                        "ever_inits",
                        "double",
                    );
                }
                TerminatorEdges::AssignOnReturn { return_, cleanup, place } => {
                    // FIXME: we could optimize the move/clones here:
                    // - we only need to clone if there's >1 non-initialized block in the return and
                    //   cleanup blocks
                    // - if the cleanup block has been initialized, we don't need to pass clone to
                    //   propagate (until polonius is stabilized, not using propagate would also be a
                    //   compile error)
                    // FIXME: check if the return blocks are actually disjoint.

                    // This must be done *first*, otherwise the unwind path will see the assignments.
                    if let Some(cleanup) = cleanup {
                        // We don't `propagate`: we'd have to clone the block state, but that's only
                        // necessary if the cleanup state wasn't already initialized.
                        //
                        // FIXME: we wouldn't need to clone either if the cleanup block is one of the
                        // return blocks, similarly to `TerminatorEdges::Double` which can be 2 edges to
                        // the same block.
                        // propagate(cleanup, block_state.clone());

                        propagate_optional_single_edge(
                            &mut borrows_holder,
                            &block_state.borrows,
                            block,
                            cleanup,
                            "A6a",
                            "borrows",
                            "assign on return cleanup",
                        );
                        propagate_optional_single_edge(
                            &mut uninits_holder,
                            &block_state.uninits,
                            block,
                            cleanup,
                            "A6b",
                            "uninits",
                            "assign on return cleanup",
                        );
                        propagate_optional_single_edge(
                            &mut ever_inits_holder,
                            &block_state.ever_inits,
                            block,
                            cleanup,
                            "A6c",
                            "ever_inits",
                            "assign on return cleanup",
                        );
                    }

                    if !return_.is_empty() {
                        analysis.apply_call_return_effect(&mut block_state, block, place);

                        let target_count = return_.len();
                        for &target in return_.iter().take(target_count - 1) {
                            propagate_optional_single_edge(
                                &mut borrows_holder,
                                &block_state.borrows,
                                block,
                                target,
                                "A7a",
                                "borrows",
                                "return target",
                            );
                            propagate_optional_single_edge(
                                &mut uninits_holder,
                                &block_state.uninits,
                                block,
                                target,
                                "A7b",
                                "uninits",
                                "return target",
                            );
                            propagate_optional_single_edge(
                                &mut ever_inits_holder,
                                &block_state.ever_inits,
                                block,
                                target,
                                "A7c",
                                "ever_inits",
                                "return target",
                            );
                        }

                        let target = *return_.last().unwrap();
                        propagate_single_edge(
                            &mut borrows_holder,
                            block_state.borrows,
                            block,
                            target,
                            "A7d",
                            "borrows",
                            "return target",
                        );
                        propagate_single_edge(
                            &mut uninits_holder,
                            block_state.uninits,
                            block,
                            target,
                            "A7e",
                            "uninits",
                            "return target",
                        );
                        propagate_single_edge(
                            &mut ever_inits_holder,
                            block_state.ever_inits,
                            block,
                            target,
                            "A7f",
                            "ever_inits",
                            "return target",
                        );
                    }
                }
                TerminatorEdges::SwitchInt { targets, discr } => {
                    if let Some(_data) = analysis.get_switch_int_data(block, discr) {
                        todo!("wat. this is unused in tests");
                    } else {
                        let target_count = targets.all_targets().len();
                        for &target in targets.all_targets().iter().take(target_count - 1) {
                            propagate_optional_single_edge(
                                &mut borrows_holder,
                                &block_state.borrows,
                                block,
                                target,
                                "A8a",
                                "borrows",
                                "switchint",
                            );
                            propagate_optional_single_edge(
                                &mut uninits_holder,
                                &block_state.uninits,
                                block,
                                target,
                                "A8b",
                                "uninits",
                                "switchint",
                            );
                            propagate_optional_single_edge(
                                &mut ever_inits_holder,
                                &block_state.ever_inits,
                                block,
                                target,
                                "A8c",
                                "ever_inits",
                                "switchint",
                            );
                        }

                        let target = *targets.all_targets().last().unwrap();
                        propagate_single_edge(
                            &mut borrows_holder,
                            block_state.borrows,
                            block,
                            target,
                            "A8d",
                            "borrows",
                            "switchint",
                        );
                        propagate_single_edge(
                            &mut uninits_holder,
                            block_state.uninits,
                            block,
                            target,
                            "A8e",
                            "uninits",
                            "switchint",
                        );
                        propagate_single_edge(
                            &mut ever_inits_holder,
                            block_state.ever_inits,
                            block,
                            target,
                            "A8f",
                            "ever_inits",
                            "switchint",
                        );
                    }
                }
            }
        } else {
            for block in sccs.sccs[scc as usize].iter().copied() {
                borrows_holder.dirty_queue.insert(block);
                uninits_holder.dirty_queue.insert(block);
                ever_inits_holder.dirty_queue.insert(block);
            }

            while let Some(block) = borrows_holder.dirty_queue.pop() {
                // eprintln!("B1 - entering scc {scc}'s block: {:?}", block);

                // eprintln!(
                //     "B2 - is {block:?} state ready, borrows: {}, uninits: {}, ever_inits: {}",
                //     borrows_holder.lazy_results[block].is_some(),
                //     uninits_holder.lazy_results[block].is_some(),
                //     ever_inits_holder.lazy_results[block].is_some(),
                // );

                // We're in an SCC:
                // - we need to retain our entry state and can't move it to our children: we need to
                //   reach a fixpoint, and *then* to visit the blocks starting from their entry
                //   state.
                // - our parent have initialized our state, so we use this to set the domain cursor
                state.borrows.clone_from(
                    borrows_holder.lazy_results[block].as_ref().unwrap_or_else(|| {
                        panic!("the parents of {block:?} haven't initialized its state!");
                    }),
                );
                propagate(
                    body,
                    block,
                    &mut borrows_holder,
                    &mut state.borrows,
                    &mut analysis.borrows,
                    |holder, target, state| {
                        let set_changed = match holder.lazy_results[target].as_mut() {
                            None => {
                                holder.lazy_results[target] = Some(state.clone());
                                true
                            }
                            Some(existing_state) => existing_state.join(&state),
                        };

                        // let set_changed = holder.results[target].join(&state);
                        let target_scc = sccs.components[target];
                        // eprintln!(
                        //     "B3a - propagating borrows from {block:?} to target {target:?} (init: {}) of scc {target_scc}",
                        //     holder.lazy_results[target].is_some()
                        // );
                        if set_changed && target_scc == scc {
                            // The target block is in the SCC we're currently processing, and we
                            // want to process this block until fixpoint. Otherwise, the target
                            // block is in a successor SCC and it will be processed when that SCC is
                            // encountered later.
                            holder.dirty_queue.insert(target);
                        }
                    },
                );
            }
            while let Some(block) = uninits_holder.dirty_queue.pop() {
                // eprintln!("C1 - entering scc {scc}'s block: {:?}", block);

                // eprintln!(
                //     "C2 - is {block:?} state ready, borrows: {}, uninits: {}, ever_inits: {}",
                //     borrows_holder.lazy_results[block].is_some(),
                //     uninits_holder.lazy_results[block].is_some(),
                //     ever_inits_holder.lazy_results[block].is_some(),
                // );

                state.uninits.clone_from(
                    uninits_holder.lazy_results[block].as_ref().unwrap_or_else(|| {
                        panic!("the parents of {block:?} haven't initialized its state!");
                    }),
                );
                propagate(
                    body,
                    block,
                    &mut uninits_holder,
                    &mut state.uninits,
                    &mut analysis.uninits,
                    |holder, target, state| {
                        let set_changed = match holder.lazy_results[target].as_mut() {
                            None => {
                                holder.lazy_results[target] = Some(state.clone());
                                true
                            }
                            Some(existing_state) => existing_state.join(&state),
                        };

                        // let set_changed = holder.results[target].join(&state);
                        let target_scc = sccs.components[target];
                        // eprintln!(
                        //     "C3a - propagating uninits from {block:?} to target {target:?} (init: {}) of scc {target_scc}",
                        //     holder.lazy_results[target].is_some()
                        // );
                        if set_changed && target_scc == scc {
                            // The target block is in the SCC we're currently processing, and we
                            // want to process this block until fixpoint. Otherwise, the target
                            // block is in a successor SCC and it will be processed when that SCC is
                            // encountered later.
                            holder.dirty_queue.insert(target);
                        }
                    },
                );
            }
            while let Some(block) = ever_inits_holder.dirty_queue.pop() {
                // eprintln!("D1 - entering scc {scc}'s block: {:?}", block);

                // eprintln!(
                //     "D2 - is {block:?} state ready, ever_inits: {}",
                //     ever_inits_holder.lazy_results[block].is_some(),
                // );

                state.ever_inits.clone_from(
                    ever_inits_holder.lazy_results[block].as_ref().unwrap_or_else(|| {
                        panic!("the parents of {block:?} haven't initialized its state!");
                    }),
                );
                propagate(
                    body,
                    block,
                    &mut ever_inits_holder,
                    &mut state.ever_inits,
                    &mut analysis.ever_inits,
                    |holder, target, state| {
                        let set_changed = match holder.lazy_results[target].as_mut() {
                            None => {
                                holder.lazy_results[target] = Some(state.clone());
                                true
                            }
                            Some(existing_state) => existing_state.join(&state),
                        };

                        // let set_changed = holder.results[target].join(&state);
                        let target_scc = sccs.components[target];
                        // eprintln!(
                        //     "D3a - propagating ever_inits from {block:?} to target {target:?} (init: {}) of scc {target_scc}",
                        //     holder.lazy_results[target].is_some()
                        // );
                        if set_changed && target_scc == scc {
                            // The target block is in the SCC we're currently processing, and we
                            // want to process this block until fixpoint. Otherwise, the target
                            // block is in a successor SCC and it will be processed when that SCC is
                            // encountered later.
                            holder.dirty_queue.insert(target);
                        }
                    },
                );
            }

            // eprintln!(
            //     "1b - scc {scc} has reached fixpoint, visiting it again from its entry states"
            // );

            // let state = &mut state;
            let analysis = &mut analysis;

            // The SCC has reached fixpoint, we can now visit it.
            for block in sccs.sccs[scc as usize].iter().copied() {
                // eprintln!("2 - re-entering scc {scc}'s block: {:?}", block);

                // // tmp: verifying contents on entry
                // assert_eq!(
                //     &flow_entry_states[block].borrows,
                //     borrows_holder.lazy_results[block].as_ref().unwrap(),
                //     "borrows of block {block:?} differ"
                // );
                // assert_eq!(
                //     &flow_entry_states[block].ever_inits,
                //     ever_inits_holder.lazy_results[block].as_ref().unwrap(),
                //     "ever_inits of block {block:?} differ"
                // );
                // assert_eq!(
                //     &flow_entry_states[block].uninits,
                //     uninits_holder.lazy_results[block].as_ref().unwrap(),
                //     "uninits of block {block:?} differ"
                // );

                // state.borrows.clone_from(&borrows_holder.lazy_results[block].as_ref().unwrap());
                // state
                //     .ever_inits
                //     .clone_from(&ever_inits_holder.lazy_results[block].as_ref().unwrap());
                // state.uninits.clone_from(&uninits_holder.lazy_results[block].as_ref().unwrap());

                let Some(borrows) = borrows_holder.lazy_results[block].take() else {
                    continue;
                };
                let Some(uninits) = uninits_holder.lazy_results[block].take() else {
                    continue;
                };
                let Some(ever_inits) = ever_inits_holder.lazy_results[block].take() else {
                    continue;
                };
                let mut state = BorrowckDomain { borrows, uninits, ever_inits };
                let state = &mut state;

                // // tmp: verifying contents on entry
                // assert_eq!(
                //     flow_entry_states[block].borrows, state.borrows,
                //     "borrows of block {block:?} differ"
                // );
                // assert_eq!(
                //     flow_entry_states[block].ever_inits, state.ever_inits,
                //     "ever_inits of block {block:?} differ"
                // );
                // assert_eq!(
                //     flow_entry_states[block].uninits, state.uninits,
                //     "uninits of block {block:?} differ"
                // );

                let block_data = &body[block];

                vis.visit_block_start(state);

                for (statement_index, statement) in block_data.statements.iter().enumerate() {
                    let location = Location { block, statement_index };
                    analysis.apply_early_statement_effect(state, statement, location);
                    vis.visit_after_early_statement_effect(analysis, state, statement, location);
                    analysis.apply_primary_statement_effect(state, statement, location);
                    vis.visit_after_primary_statement_effect(analysis, state, statement, location);
                }

                let terminator = block_data.terminator();
                let location = Location { block, statement_index: block_data.statements.len() };
                analysis.apply_early_terminator_effect(state, terminator, location);
                vis.visit_after_early_terminator_effect(analysis, state, terminator, location);
                analysis.apply_primary_terminator_effect(state, terminator, location);
                vis.visit_after_primary_terminator_effect(analysis, state, terminator, location);

                vis.visit_block_end(state);
            }
        }

        // eprintln!();
    }
}

// When a CFG is acyclic, reaching fixpoint is a single iteration over the blocks in RPO order. If
// we do the computation at the same time as we're visiting results, we can avoid computing
// per-block state in `iterate_to_fixpoint` and then per-statement state (again, since we have to do
// it in `iterate_to_fixpoint` to compute the per-block exit state) in `visit_results`.
//
// The callers need to ensure that the CFG is acyclic, e.g. via `body.basic_blocks.is_cfg_cyclic()`,
// and that the analysis is a forward analysis.
fn compute_rpo_dataflow<'mir, 'tcx, A>(
    body: &'mir Body<'tcx>,
    analysis: &mut A,
    vis: &mut impl ResultsVisitor<'tcx, A>,
) where
    A: Analysis<'tcx, Direction = rustc_mir_dataflow::Forward>,
{
    use rustc_middle::mir;
    use rustc_mir_dataflow::JoinSemiLattice;

    // Instead of storing a domain state per-block, we only do it lazily. This means that we can
    // re-use that state when for example a block has a single successor. The visitor will be
    // notified that the entire block is complete, before we mutate the same piece of state, and
    // thus avoid creating it or cloning it in many cases.

    // Set up lazy state for the CFG
    let mut results: IndexVec<BasicBlock, Option<A::Domain>> =
        IndexVec::from_elem_n(None, body.basic_blocks.len());

    // Ensure the start block has some state in it;
    results[mir::START_BLOCK] = Some(analysis.bottom_value(body));
    analysis.initialize_start_block(body, results[mir::START_BLOCK].as_mut().unwrap());

    // eprintln!("CFG, {} blocks, {:?}, {:#?}", body.basic_blocks.len(), body.span, body);
    // for (block, bb) in body.basic_blocks.iter_enumerated() {
    //     let terminator = bb.terminator();
    //     let successors: Vec<_> = terminator.successors().collect();
    //     eprintln!(
    //         "block: {block:?}, {} successors: {:?}, edges: {:?}, terminator: {:?}",
    //         successors.len(),
    //         successors,
    //         terminator.edges(),
    //         terminator.kind,
    //     );

    //     match terminator.kind {
    //         TerminatorKind::Drop { place: _, target, unwind, replace: _, drop, async_fut: _ } => {
    //             eprintln!(
    //                 "Drop terminator, target: {target:?}, unwind: {unwind:?}, drop: {drop:?}"
    //             );
    //         }
    //         _ => {}
    //     }
    // }

    // Visit this *acyclic* CFG in RPO.
    for (block, block_data) in traversal::reverse_postorder(body) {
        // `reverse_postorder` doesn't yield unreachable blocks, so every block we visit will have
        // at least one of its parents visited first.
        //
        // Therefore, we get our per-block state:
        // - from one of our predecessors initializing it
        // - or from the analysis' initial value, for the START_BLOCK
        //
        // That is true in general, except for some bug/issue with async drops today: we can visit
        // successors of a block that are not present in the data used to propagate dataflow to
        // successor blocks, `TerminatorEdges`.
        //
        // We temporarily ignore these unreachable-in-practice blocks for now: they are ignored by
        // the dataflow engine, and wouldn't have any state computed or propagated other than the
        // bottom value of the analysis.
        let Some(mut block_state) = results[block].take() else {
            continue;
        };

        // FIXME(async_drop): we assert here to fail when this issue is fixed, just expect() above
        // and remove the assertion below when traversal and dataflow agree.
        assert!(
            {
                let terminator = block_data.terminator();
                if matches!(terminator.kind, TerminatorKind::Drop { drop: Some(_), .. }) {
                    terminator.successors().count() == 3
                        && matches!(terminator.edges(), TerminatorEdges::Double(_, _))
                } else {
                    true
                }
            },
            "dataflow mismatch between async_drop TerminatorKind successors() and edges()"
        );

        // Apply effects in the block's statements.
        vis.visit_block_start(&mut block_state);

        for (statement_index, statement) in block_data.statements.iter().enumerate() {
            let location = Location { block, statement_index };
            analysis.apply_early_statement_effect(&mut block_state, statement, location);
            vis.visit_after_early_statement_effect(analysis, &block_state, statement, location);

            analysis.apply_primary_statement_effect(&mut block_state, statement, location);
            vis.visit_after_primary_statement_effect(analysis, &block_state, statement, location);
        }

        // Apply effects in the block terminator.
        let terminator = block_data.terminator();
        let location = Location { block, statement_index: block_data.statements.len() };
        analysis.apply_early_terminator_effect(&mut block_state, terminator, location);
        vis.visit_after_early_terminator_effect(analysis, &block_state, terminator, location);

        let edges =
            analysis.apply_primary_terminator_effect(&mut block_state, terminator, location);
        vis.visit_after_primary_terminator_effect(analysis, &block_state, terminator, location);

        vis.visit_block_end(&mut block_state);

        // The current block is done, and the visitor was notified at every step. We now take care
        // of the successor's state.

        // let mut propagate = |target: BasicBlock, state: A::Domain| {
        //     // Look at the target block state holder:
        //     // - either it's empty, and we initialize it by moving the state there
        //     // - or it's been initialized, and we merge it with the given state
        //     match results[target].as_mut() {
        //         None => {
        //             results[target] = Some(state);
        //         }
        //         Some(existing_state) => {
        //             existing_state.join(&state);
        //         }
        //     }
        // };

        match edges {
            TerminatorEdges::None => {}
            TerminatorEdges::Single(target) => {
                // eprintln!(
                //     "Propagating state from {block:?} to single edge: {target:?} (init: {})",
                //     results[target].is_some()
                // );
                match results[target].as_mut() {
                    None => {
                        results[target] = Some(block_state);
                    }
                    Some(existing_state) => {
                        existing_state.join(&block_state);
                    }
                }

                // We have a single successor, our own state can either be moved to it, or dropped.
                // propagate(target, block_state);
            }
            TerminatorEdges::Double(target, unwind) if target == unwind => {
                // eprintln!(
                //     "Propagating state from {block:?} to single double edge: {target:?} (init: {})",
                //     results[target].is_some()
                // );

                // Why are we generating this shape in MIR building :thinking: ? Either way, we also
                // have a single successor here.
                // propagate(target, block_state);
                match results[target].as_mut() {
                    None => {
                        results[target] = Some(block_state);
                    }
                    Some(existing_state) => {
                        existing_state.join(&block_state);
                    }
                }
            }
            TerminatorEdges::Double(target, unwind) => {
                // eprintln!(
                //     "Propagating state from {block:?} to double edge: {target:?} (init: {}), {unwind:?} (init: {})",
                //     results[target].is_some(),
                //     results[unwind].is_some()
                // );

                // We have two *distinct* successors.
                //
                // We could use an `_unchecked` version of `pick2_mut` if it existed: we know the
                // indices are disjoint and in-bounds.
                match results.pick2_mut(target, unwind) {
                    (None, None) => {
                        // We need to initialize both successors with our own block state, we need a
                        // clone.
                        results[target] = Some(block_state.clone());
                        results[unwind] = Some(block_state);
                    }
                    (None, Some(unwind_state)) => {
                        // No need to clone, only one successor is not initialized yet.
                        unwind_state.join(&block_state);
                        results[target] = Some(block_state);
                    }
                    (Some(target_state), None) => {
                        // No need to clone, only one successor is not initialized yet.
                        target_state.join(&block_state);
                        results[unwind] = Some(block_state);
                    }
                    (Some(target_state), Some(unwind_state)) => {
                        // The successors have already been initialized by their other parents, we
                        // merge our block state there.
                        target_state.join(&block_state);
                        unwind_state.join(&block_state);
                    }
                }
            }
            TerminatorEdges::AssignOnReturn { return_, cleanup, place } => {
                // FIXME: we could optimize the move/clones here:
                // - we only need to clone if there's >1 non-initialized block in the return and
                //   cleanup blocks
                // - if the cleanup block has been initialized, we don't need to pass clone to
                //   propagate (until polonius is stabilized, not using propagate would also be a
                //   compile error)
                // FIXME: check if the return blocks are actually disjoint.

                // This must be done *first*, otherwise the unwind path will see the assignments.
                if let Some(cleanup) = cleanup {
                    // We don't `propagate`: we'd have to clone the block state, but that's only
                    // necessary if the cleanup state wasn't already initialized.
                    //
                    // FIXME: we wouldn't need to clone either if the cleanup block is one of the
                    // return blocks, similarly to `TerminatorEdges::Double` which can be 2 edges to
                    // the same block.
                    // propagate(cleanup, block_state.clone());

                    match results[cleanup].as_mut() {
                        None => {
                            results[cleanup] = Some(block_state.clone());
                        }
                        Some(existing_state) => {
                            existing_state.join(&block_state);
                        }
                    }
                }

                if !return_.is_empty() {
                    analysis.apply_call_return_effect(&mut block_state, block, place);

                    let target_count = return_.len();
                    for &target in return_.iter().take(target_count - 1) {
                        // propagate(target, block_state.clone());
                        match results[target].as_mut() {
                            None => {
                                results[target] = Some(block_state.clone());
                            }
                            Some(existing_state) => {
                                existing_state.join(&block_state);
                            }
                        }
                    }

                    let target = *return_.last().unwrap();
                    // propagate(target, block_state);
                    match results[target].as_mut() {
                        None => {
                            results[target] = Some(block_state);
                        }
                        Some(existing_state) => {
                            existing_state.join(&block_state);
                        }
                    }
                }
            }
            TerminatorEdges::SwitchInt { targets, discr } => {
                if let Some(_data) = analysis.get_switch_int_data(block, discr) {
                    todo!("wat. this is unused in tests");
                } else {
                    let target_count = targets.all_targets().len();
                    for &target in targets.all_targets().iter().take(target_count - 1) {
                        match results[target].as_mut() {
                            None => {
                                results[target] = Some(block_state.clone());
                            }
                            Some(existing_state) => {
                                existing_state.join(&block_state);
                            }
                        }
                    }

                    let target = *targets.all_targets().last().unwrap();
                    match results[target].as_mut() {
                        None => {
                            results[target] = Some(block_state);
                        }
                        Some(existing_state) => {
                            existing_state.join(&block_state);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
fn compute_dataflow<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,

    move_data: &'a MoveData<'tcx>,
    borrow_set: &'a BorrowSet<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,

    vis: &mut MirBorrowckCtxt<'a, '_, 'tcx>,
) {
    let borrows = Borrows::new(tcx, body, regioncx, borrow_set);
    let uninits = MaybeUninitializedPlaces::new(tcx, body, move_data);
    let ever_inits = EverInitializedPlaces::new(body, move_data);

    let mut analysis = Borrowck { borrows, uninits, ever_inits };

    // Set up lazy state for the CFG
    use rustc_middle::mir;
    use rustc_mir_dataflow::JoinSemiLattice;

    let mut results: IndexVec<BasicBlock, Option<BorrowckDomain>> =
        IndexVec::from_elem_n(None, body.basic_blocks.len());

    // Ensure the start block has some state in it;
    results[mir::START_BLOCK] = Some(analysis.bottom_value(body));
    analysis.initialize_start_block(body, results[mir::START_BLOCK].as_mut().unwrap());

    for (_idx, (block, block_data)) in traversal::reverse_postorder(body).enumerate() {
        // Apply effects in block
        let mut block_state = results[block].take().unwrap_or_else(|| analysis.bottom_value(body));

        vis.visit_block_start(&mut block_state);

        for (statement_index, statement) in block_data.statements.iter().enumerate() {
            let location = Location { block, statement_index };
            analysis.apply_early_statement_effect(&mut block_state, statement, location);
            vis.visit_after_early_statement_effect(
                &mut analysis,
                &block_state,
                statement,
                location,
            );

            analysis.apply_primary_statement_effect(&mut block_state, statement, location);
            vis.visit_after_primary_statement_effect(
                &mut analysis,
                &block_state,
                statement,
                location,
            );
        }
        let terminator = block_data.terminator();
        let location = Location { block, statement_index: block_data.statements.len() };
        analysis.apply_early_terminator_effect(&mut block_state, terminator, location);
        vis.visit_after_early_terminator_effect(&mut analysis, &block_state, terminator, location);

        let edges =
            analysis.apply_primary_terminator_effect(&mut block_state, terminator, location);
        vis.visit_after_primary_terminator_effect(
            &mut analysis,
            &block_state,
            terminator,
            location,
        );

        // notify visitor the block is ready
        vis.visit_block_end(&mut block_state);

        match edges {
            TerminatorEdges::None => {}
            TerminatorEdges::Single(target) => match results[target].as_mut() {
                None => {
                    results[target] = Some(block_state);
                }
                Some(existing_state) => {
                    existing_state.join(&block_state);
                }
            },
            TerminatorEdges::Double(target, unwind) if target == unwind => {
                // wtf
                match results[target].as_mut() {
                    None => {
                        results[target] = Some(block_state);
                    }
                    Some(existing_state) => {
                        existing_state.join(&block_state);
                    }
                }
            }
            TerminatorEdges::Double(target, unwind) => match results.pick2_mut(target, unwind) {
                (None, None) => {
                    results[target] = Some(block_state.clone());
                    results[unwind] = Some(block_state);
                }
                (None, Some(unwind_state)) => {
                    unwind_state.join(&block_state);
                    results[target] = Some(block_state);
                }
                (Some(target_state), None) => {
                    target_state.join(&block_state);
                    results[unwind] = Some(block_state);
                }
                (Some(target_state), Some(unwind_state)) => {
                    target_state.join(&block_state);
                    unwind_state.join(&block_state);
                }
            },
            TerminatorEdges::AssignOnReturn { return_, cleanup, place } => {
                // This must be done *first*, otherwise the unwind path will see the assignments.
                if let Some(cleanup) = cleanup {
                    match results[cleanup].as_mut() {
                        None => {
                            results[cleanup] = Some(block_state.clone());
                        }
                        Some(existing_state) => {
                            existing_state.join(&block_state);
                        }
                    }
                }

                if !return_.is_empty() {
                    analysis.apply_call_return_effect(&mut block_state, block, place);

                    // fixme: optimize, if we've merged the previous target states instead
                    // of moving, we don't need to clone it.

                    let target_count = return_.len();
                    for &target in return_.iter().take(target_count - 1) {
                        match results[target].as_mut() {
                            None => {
                                results[target] = Some(block_state.clone());
                            }
                            Some(existing_state) => {
                                existing_state.join(&block_state);
                            }
                        }
                    }

                    let target = *return_.last().unwrap();
                    match results[target].as_mut() {
                        None => {
                            results[target] = Some(block_state.clone());
                        }
                        Some(existing_state) => {
                            existing_state.join(&block_state);
                        }
                    }
                }
            }
            TerminatorEdges::SwitchInt { targets, discr } => {
                if let Some(_data) = analysis.get_switch_int_data(block, discr) {
                    todo!("wat. this is unused in tests");
                } else {
                    let target_count = targets.all_targets().len();
                    for &target in targets.all_targets().iter().take(target_count - 1) {
                        match results[target].as_mut() {
                            None => {
                                results[target] = Some(block_state.clone());
                            }
                            Some(existing_state) => {
                                existing_state.join(&block_state);
                            }
                        }
                    }

                    let target = *targets.all_targets().last().unwrap();
                    match results[target].as_mut() {
                        None => {
                            results[target] = Some(block_state.clone());
                        }
                        Some(existing_state) => {
                            existing_state.join(&block_state);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
fn get_flow_results<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    move_data: &'a MoveData<'tcx>,
    borrow_set: &'a BorrowSet<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
) -> (Borrowck<'a, 'tcx>, Results<BorrowckDomain>) {
    // We compute these three analyses individually, but them combine them into
    // a single results so that `mbcx` can visit them all together.
    // let timer = std::time::Instant::now();
    let borrows = Borrows::new(tcx, body, regioncx, borrow_set).iterate_to_fixpoint(
        tcx,
        body,
        Some("borrowck"),
    );
    // let elapsed = timer.elapsed();
    // if body.basic_blocks.len() > 5000 && elapsed.as_millis() > 1 {
    //     eprintln!("dataflow {}, took {} ns, {:?}", "borrows", elapsed.as_nanos(), body.span);
    // }

    // ---
    // if rustc_data_structures::graph::is_cyclic(&body.basic_blocks) {
    //     // let timer = std::time::Instant::now();
    //     let borrowz = Borrows::new(tcx, body, regioncx, borrow_set).iterate_to_fixpoint_per_scc(
    //         tcx,
    //         body,
    //         Some("borrowz"),
    //     );
    //     // let elapsed = timer.elapsed();
    //     // if body.basic_blocks.len() > 5000 && elapsed.as_millis() > 1 {
    //     //     eprintln!("dataflow {}, took {} ns, {:?}", "borrowz", elapsed.as_nanos(), body.span);
    //     // }
    //     assert_eq!(
    //         borrows.results, borrowz.results,
    //         "oh noes, borrows dataflow results are different"
    //     );
    // }

    // ---

    // let timer = std::time::Instant::now();
    let uninits = MaybeUninitializedPlaces::new(tcx, body, move_data).iterate_to_fixpoint(
        tcx,
        body,
        Some("borrowck"),
    );
    // let elapsed = timer.elapsed();
    // if body.basic_blocks.len() > 5000 && elapsed.as_millis() > 1 {
    //     eprintln!("dataflow {}, took {} ns, {:?}", "uninits", elapsed.as_nanos(), body.span);
    // }

    // if rustc_data_structures::graph::is_cyclic(&body.basic_blocks) {
    //     let uninitz = MaybeUninitializedPlaces::new(tcx, body, move_data)
    //         .iterate_to_fixpoint_per_scc(tcx, body, Some("borrowck"));
    //     assert_eq!(
    //         uninits.results, uninitz.results,
    //         "oh noes, uninits dataflow results are different"
    //     );
    // }

    // ---
    // let timer = std::time::Instant::now();
    let ever_inits = EverInitializedPlaces::new(body, move_data).iterate_to_fixpoint(
        tcx,
        body,
        Some("borrowck"),
    );
    // let elapsed = timer.elapsed();
    // if body.basic_blocks.len() > 5000 && elapsed.as_millis() > 1 {
    //     eprintln!("dataflow {}, took {} ns, {:?}", "e_inits", elapsed.as_nanos(), body.span);
    // }

    // let timer = std::time::Instant::now();
    // let _ever_initz = EverInitializedPlaces2::new(body, move_data).iterate_to_fixpoint(
    //     tcx,
    //     body,
    //     Some("borrowck"),
    // );
    // let elapsed = timer.elapsed();
    // if body.basic_blocks.len() > 5000 && elapsed.as_millis() > 1 {
    //     eprintln!("dataflow {}, took {} ns, {:?}", "e_initz", elapsed.as_nanos(), body.span);
    // }

    // if rustc_data_structures::graph::is_cyclic(&body.basic_blocks) {
    //     // let timer = std::time::Instant::now();
    //     let ever_initz = EverInitializedPlaces::new(body, move_data).iterate_to_fixpoint_per_scc(
    //         tcx,
    //         body,
    //         Some("e_initz"),
    //     );
    //     // let elapsed = timer.elapsed();
    //     // if body.basic_blocks.len() > 5000 && elapsed.as_millis() > 1 {
    //     //     eprintln!("dataflow {}, took {} ns, {:?}", "e_initz", elapsed.as_nanos(), body.span);
    //     // }
    //     assert_eq!(
    //         ever_inits.results, ever_initz.results,
    //         "oh noes, ever_inits dataflow results are different"
    //     );
    // }

    let analysis = Borrowck {
        borrows: borrows.analysis,
        uninits: uninits.analysis,
        ever_inits: ever_inits.analysis,
    };

    assert_eq!(borrows.results.len(), uninits.results.len());
    assert_eq!(borrows.results.len(), ever_inits.results.len());
    let results: Results<_> =
        itertools::izip!(borrows.results, uninits.results, ever_inits.results)
            .map(|(borrows, uninits, ever_inits)| BorrowckDomain { borrows, uninits, ever_inits })
            .collect();

    (analysis, results)
}

pub(crate) struct BorrowckInferCtxt<'tcx> {
    pub(crate) infcx: InferCtxt<'tcx>,
    pub(crate) reg_var_to_origin: RefCell<FxIndexMap<ty::RegionVid, RegionCtxt>>,
    pub(crate) param_env: ParamEnv<'tcx>,
}

impl<'tcx> BorrowckInferCtxt<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> Self {
        let typing_mode = if tcx.use_typing_mode_borrowck() {
            TypingMode::borrowck(tcx, def_id)
        } else {
            TypingMode::analysis_in_body(tcx, def_id)
        };
        let infcx = tcx.infer_ctxt().build(typing_mode);
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
}

impl<'tcx> Deref for BorrowckInferCtxt<'tcx> {
    type Target = InferCtxt<'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.infcx
    }
}

struct MirBorrowckCtxt<'a, 'infcx, 'tcx> {
    root_cx: &'a mut BorrowCheckRootCtxt<'tcx>,
    infcx: &'infcx BorrowckInferCtxt<'tcx>,
    body: &'a Body<'tcx>,
    move_data: &'a MoveData<'tcx>,

    /// Map from MIR `Location` to `LocationIndex`; created
    /// when MIR borrowck begins.
    location_table: &'a PoloniusLocationTable,

    movable_coroutine: bool,
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
    local_names: OnceCell<IndexVec<Local, Option<Symbol>>>,

    /// Record the region names generated for each region in the given
    /// MIR def so that we can reuse them later in help/error messages.
    region_names: RefCell<FxIndexMap<RegionVid, RegionName>>,

    /// The counter for generating new region names.
    next_region_name: RefCell<usize>,

    diags_buffer: &'a mut BorrowckDiagnosticsBuffer<'infcx, 'tcx>,
    move_errors: Vec<MoveError<'tcx>>,

    /// Results of Polonius analysis.
    polonius_output: Option<&'a PoloniusOutput>,
    /// When using `-Zpolonius=next`: the data used to compute errors and diagnostics.
    polonius_diagnostics: Option<&'a PoloniusDiagnosticsContext>,

    #[cfg(test)]
    nuutila: Option<nuutila::Nuutila>,
    #[cfg(test)]
    duration: u128,
    #[cfg(test)]
    duration2: u128,
    #[cfg(test)]
    duration3: u128,
    #[cfg(test)]
    transitive_predecessors: Option<IndexVec<BasicBlock, DenseBitSet<BasicBlock>>>,

    // locals_checked_for_initialization: FxHashMap<MovePathIndex, FxHashSet<Location>>,
    #[cfg(test)]
    locals_checked_for_initialization: FxHashMap<Local, FxHashSet<Location>>,
}

// Check that:
// 1. assignments are always made to mutable locations (FIXME: does that still really go here?)
// 2. loans made in overlapping scopes do not conflict
// 3. assignments do not affect things loaned out as immutable
// 4. moves do not affect things loaned out in any way
impl<'a, 'tcx> ResultsVisitor<'tcx, Borrowck<'a, 'tcx>> for MirBorrowckCtxt<'a, '_, 'tcx> {
    fn visit_after_early_statement_effect(
        &mut self,
        _analysis: &mut Borrowck<'a, 'tcx>,
        state: &BorrowckDomain,
        stmt: &Statement<'tcx>,
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
            | StatementKind::StorageLive(..) => {}
            // This does not affect borrowck
            StatementKind::BackwardIncompatibleDropHint { place, reason: BackwardIncompatibleDropReason::Edition2024 } => {
                self.check_backward_incompatible_drop(location, **place, state);
            }
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

    fn visit_after_early_terminator_effect(
        &mut self,
        _analysis: &mut Borrowck<'a, 'tcx>,
        state: &BorrowckDomain,
        term: &Terminator<'tcx>,
        loc: Location,
    ) {
        debug!("MirBorrowckCtxt::process_terminator({:?}, {:?}): {:?}", loc, term, state);
        let span = term.source_info.span;

        self.check_activations(loc, span, state);

        match &term.kind {
            TerminatorKind::SwitchInt { discr, targets: _ } => {
                self.consume_operand(loc, (discr, span), state);
            }
            TerminatorKind::Drop {
                place,
                target: _,
                unwind: _,
                replace,
                drop: _,
                async_fut: _,
            } => {
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

    fn visit_after_primary_terminator_effect(
        &mut self,
        _analysis: &mut Borrowck<'a, 'tcx>,
        state: &BorrowckDomain,
        term: &Terminator<'tcx>,
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
                match self.borrow_set.locals_state_at_exit() {
                    LocalsStateAtExit::AllAreInvalidated => {
                        // Returning from the function implicitly kills storage for all locals and statics.
                        // Often, the storage will already have been killed by an explicit
                        // StorageDead, but we don't always emit those (notably on unwind paths),
                        // so this "extra check" serves as a kind of backup.
                        for i in state.borrows.iter() {
                            let borrow = &self.borrow_set[i];
                            self.check_for_invalidation_at_exit(loc, borrow, span);
                        }
                    }
                    // If we do not implicitly invalidate all locals on exit,
                    // we check for conflicts when dropping or moving this local.
                    LocalsStateAtExit::SomeAreInvalidated { has_storage_dead_or_moved: _ } => {}
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

#[cfg(test)]
mod nuutila {
    use std::collections::VecDeque;

    use rustc_index::interval::IntervalSet;
    use rustc_middle::mir::{BasicBlock, BasicBlocks, TerminatorEdges};

    use crate::consumers::BorrowSet;

    pub(super) struct Nuutila {
        candidate_component_roots: Vec<usize>,
        components: Vec<isize>,
        pub component_count: usize,
        dfs_numbers: Vec<u32>,
        d: u32,
        visited: Vec<bool>,
        stack_vertex: VecDeque<usize>,
        stack_component: VecDeque<usize>,
        pub reachability: Vec<IntervalSet<usize>>,
    }

    impl Nuutila {
        pub(crate) fn new(node_count: usize) -> Self {
            Self {
                candidate_component_roots: vec![0; node_count],
                components: vec![-1; node_count],
                component_count: 0,
                dfs_numbers: vec![0; node_count],
                d: 0,
                visited: vec![false; node_count],
                stack_vertex: VecDeque::new(),
                stack_component: VecDeque::new(),
                reachability: vec![IntervalSet::new(node_count); node_count + 1],
                // ^--- la reachability c'est celle que des composants donc il en faut moins que `node_count` s'il y a au moins un SCC avec > 1 nodes
                // reachabilly: vec![HybridBitSet::new_empty(node_count); node_count],
            }
        }

        // fn compute_sccs(&mut self, blocks: &BasicBlocks<'_>) {
        //     for (idx, block) in blocks.iter_enumerated() {
        //         let edges = block.terminator().edges();
        //         if matches!(edges, TerminatorEdges::None) {
        //             continue;
        //         }
        //
        //         let idx = idx.as_usize();
        //         if !self.visited[idx] {
        //             self.dfs_visit(idx, blocks);
        //         }
        //     }
        // }

        // Compute SCCs and reachability only starting where loans appear.
        // We still have the unused blocks in our domain, but won't traverse them.
        pub(crate) fn compute_for_loans(
            &mut self,
            borrow_set: &BorrowSet<'_>,
            blocks: &BasicBlocks<'_>,
        ) {
            for (_loan_idx, loan) in borrow_set.iter_enumerated() {
                let block_idx = loan.reserve_location.block;
                let block = &blocks[block_idx];

                let edges = block.terminator().edges();
                if matches!(edges, TerminatorEdges::None) {
                    continue;
                }

                let idx = block_idx.as_usize();
                if !self.visited[idx] {
                    self.dfs_visit(idx, blocks);
                }
            }
        }

        fn dfs_visit(&mut self, v: usize, blocks: &BasicBlocks<'_>) {
            self.candidate_component_roots[v] = v;
            self.components[v] = -1;

            self.d += 1;
            self.dfs_numbers[v] = self.d;

            self.stack_vertex.push_front(v);
            let stack_component_height = self.stack_component.len();

            self.visited[v] = true;

            let idx = BasicBlock::from_usize(v);
            for succ in blocks[idx].terminator().successors() {
                let w = succ.as_usize();

                if w == v {
                    panic!("a dang self loop ?! at {}", w);
                }

                if !self.visited[w] {
                    self.dfs_visit(w, blocks);
                }

                let component_w = self.components[w];
                if component_w == -1 {
                    if self.dfs_numbers[self.candidate_component_roots[w]]
                        < self.dfs_numbers[self.candidate_component_roots[v]]
                    {
                        self.candidate_component_roots[v] = self.candidate_component_roots[w];
                    }
                } else {
                    assert!(component_w >= 0);

                    // FIXME: check if v -> w is actually a forward edge or not, to avoid unnecessary work if it is
                    self.stack_component.push_front(self.components[w] as usize);
                }
            }

            if self.candidate_component_roots[v] == v {
                self.component_count += 1;
                self.components[v] = self.component_count as isize;

                // Reachability of C[v]
                assert!(self.reachability[self.component_count].is_empty());
                // assert!(self.reachabilly[self.component_count].is_empty());

                if let Some(&top) = self.stack_vertex.front() {
                    if top != v {
                        // we're adding new component, initialize its reachability: self-loop,
                        // the component can reach itself
                        // self.reachability[self.component_count] =
                        //     (self.component_count, self.component_count).to_interval_set();
                        self.reachability[self.component_count].insert(self.component_count);
                    } else {
                        // R[C[v]] should be empty here already, do nothing
                        // if we don't always initialize the reachability of C by default, it would need to be
                        // initialized to "empty" here.
                    }
                }

                // process adjacent components
                while self.stack_component.len() != stack_component_height {
                    let x =
                        self.stack_component.pop_front().expect("Sc can't be empty at this point");
                    // prevent performing duplicate operations
                    if !self.reachability[self.component_count].contains(x) {
                        // merge reachability information
                        assert_ne!(x, self.component_count);

                        let zzz = unsafe {
                            self.reachability.get_unchecked(x) as *const IntervalSet<usize>
                        };
                        let r_c_v =
                            unsafe { self.reachability.get_unchecked_mut(self.component_count) };
                        r_c_v.union(unsafe { &*zzz });
                        r_c_v.insert(x);
                    }
                }

                while let Some(w) = self.stack_vertex.pop_front() {
                    self.components[w] = self.components[v];

                    if w == v {
                        break;
                    }
                }
            }
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
        state: &BorrowckDomain,
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

    fn borrows_in_scope<'s>(
        &self,
        location: Location,
        state: &'s BorrowckDomain,
    ) -> Cow<'s, MixedBitSet<BorrowIndex>> {
        if let Some(polonius) = &self.polonius_output {
            // Use polonius output if it has been enabled.
            let location = self.location_table.start_index(location);
            let mut polonius_output = MixedBitSet::new_empty(self.borrow_set.len());
            for &idx in polonius.errors_at(location) {
                polonius_output.insert(idx);
            }
            Cow::Owned(polonius_output)
        } else {
            Cow::Borrowed(&state.borrows)
        }
    }

    #[cfg(test)]
    fn compute_nuutila(&mut self) {
        use nuutila::Nuutila;

        let timer = std::time::Instant::now();
        let mut sccs = Nuutila::new(self.body.basic_blocks.len());
        sccs.compute_for_loans(&self.borrow_set, &self.body.basic_blocks);

        let elapsed = timer.elapsed();
        eprintln!(
            "compute_nuutila, found {} SCCs in {} ns",
            sccs.component_count,
            elapsed.as_nanos(),
        );

        self.nuutila = Some(sccs);
    }

    #[cfg(test)]
    fn compute_transitive_predecessors(&mut self) {
        let block_count = self.body.basic_blocks.len();

        // Compute `transitive_predecessors` and `adjacent_predecessors`.
        let mut transitive_predecessors =
            IndexVec::from_elem_n(DenseBitSet::new_empty(block_count), block_count);
        let mut adjacent_predecessors = transitive_predecessors.clone();
        // The stack is initially a reversed postorder traversal of the CFG. However, we might add
        // add blocks again to the stack if we have loops.
        let mut stack =
            self.body.basic_blocks.reverse_postorder().iter().rev().copied().collect::<Vec<_>>();
        // We keep track of all blocks that are currently not in the stack.
        let mut not_in_stack = DenseBitSet::new_empty(block_count);
        while let Some(block) = stack.pop() {
            not_in_stack.insert(block);

            // Loop over all successors to the block and add `block` to their predecessors.
            for succ_block in self.body.basic_blocks[block].terminator().successors() {
                // Keep track of whether the transitive predecessors of `succ_block` has changed.
                let mut changed = false;

                // Insert `block` in `succ_block`s predecessors.
                if adjacent_predecessors[succ_block].insert(block) {
                    // Remember that `adjacent_predecessors` is a subset of
                    // `transitive_predecessors`.
                    changed |= transitive_predecessors[succ_block].insert(block);
                }

                // Add all transitive predecessors of `block` to the transitive predecessors of
                // `succ_block`.
                if block != succ_block {
                    let (blocks_predecessors, succ_blocks_predecessors) =
                        transitive_predecessors.pick2_mut(block, succ_block);
                    changed |= succ_blocks_predecessors.union(blocks_predecessors);

                    // Check if the `succ_block`s transitive predecessors changed. If so, we may
                    // need to add it to the stack again.
                    if changed && not_in_stack.remove(succ_block) {
                        stack.push(succ_block);
                    }
                }
            }

            // debug_assert!(
            //     transitive_predecessors[block].superset(&adjacent_predecessors[block])
            // );
        }

        self.transitive_predecessors = Some(transitive_predecessors);
    }

    #[instrument(level = "debug", skip(self, state))]
    fn check_access_for_conflict(
        &mut self,
        location: Location,
        place_span: (Place<'tcx>, Span),
        sd: AccessDepth,
        rw: ReadOrWrite,
        state: &BorrowckDomain,
    ) -> bool {
        let mut error_reported = false;

        // For each region that is live at this location
        //   does it unify with the loan introduction region
        //      if so, it may be live
        //   if the live region doesn't unify with any borrow region, we don't need to check it

        let borrows_in_scope = self.borrows_in_scope(location, state);

        // if self.borrow_set.len() > 6230 {
        //     if let Some(borrows_for_place_base) = self.borrow_set.local_map.get(&place_span.0.local)
        //     {
        //         if self.nuutila.is_none() {
        //             eprintln!(
        //                 "we have {} borrows for the place local on the entire function, computing sccs",
        //                 borrows_for_place_base.len(),
        //             );
        //             self.compute_nuutila();
        //         }

        //         let timer = std::time::Instant::now();

        //         let nuutila = self.nuutila.as_ref().unwrap();

        //         let current_block = location.block;
        //         let mut reachable_loans = 0;
        //         for &idx in borrows_for_place_base {
        //             let loan = &self.borrow_set[idx];
        //             let loan_introduction_block = loan.reserve_location.block;

        //             let loan_block_reachability =
        //                 &nuutila.reachability[loan_introduction_block.as_usize()];
        //             if loan_block_reachability.contains(current_block.as_usize()) {
        //                 reachable_loans += 1;
        //             }
        //         }

        //         let elapsed = timer.elapsed();
        //         if reachable_loans > 0 {
        //             self.duration += elapsed.as_nanos();

        //             // eprintln!(
        //             //     "{} invalidations are reachable from their loan introduction, took {} ns, {:?}",
        //             //     reachable_loans,
        //             //     elapsed.as_nanos(),
        //             //     self.body.span
        //             // );
        //         }

        //         // ---
        //         let timer = std::time::Instant::now();

        //         let mut reachable_loans = 0;
        //         for &idx in borrows_for_place_base {
        //             if borrows_in_scope.contains(idx) {
        //                 reachable_loans += 1;
        //             }
        //         }

        //         let elapsed = timer.elapsed();
        //         if reachable_loans > 0 {
        //             self.duration3 += elapsed.as_nanos();
        //         }

        //         // ---

        //         if self.transitive_predecessors.is_none() {
        //             self.compute_transitive_predecessors();
        //         }

        //         let timer = std::time::Instant::now();

        //         let transitive_predecessors = self.transitive_predecessors.as_ref().unwrap();

        //         let mut reachable_loans = 0;
        //         for &idx in borrows_for_place_base {
        //             let loan = &self.borrow_set[idx];
        //             let source = loan.reserve_location;

        //             #[inline(always)]
        //             fn is_predecessor(
        //                 transitive_predecessors: &IndexVec<BasicBlock, DenseBitSet<BasicBlock>>,
        //                 a: Location,
        //                 b: Location,
        //             ) -> bool {
        //                 a.block == b.block && a.statement_index < b.statement_index
        //                     || transitive_predecessors[b.block].contains(a.block)
        //             }

        //             if is_predecessor(transitive_predecessors, source, location) {
        //                 reachable_loans += 1;
        //             }
        //         }

        //         let elapsed = timer.elapsed();
        //         if reachable_loans > 0 {
        //             self.duration2 += elapsed.as_nanos();
        //         }
        //     }
        // }

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
                    ControlFlow::Continue(())
                }

                (Read(_), BorrowKind::Shared | BorrowKind::Fake(_))
                | (
                    Read(ReadKind::Borrow(BorrowKind::Fake(FakeBorrowKind::Shallow))),
                    BorrowKind::Mut { .. },
                ) => ControlFlow::Continue(()),

                (Reservation(_), BorrowKind::Fake(_) | BorrowKind::Shared) => {
                    // This used to be a future compatibility warning (to be
                    // disallowed on NLL). See rust-lang/rust#56254
                    ControlFlow::Continue(())
                }

                (Write(WriteKind::Move), BorrowKind::Fake(FakeBorrowKind::Shallow)) => {
                    // Handled by initialization checks.
                    ControlFlow::Continue(())
                }

                (Read(kind), BorrowKind::Mut { .. }) => {
                    // Reading from mere reservations of mutable-borrows is OK.
                    if !is_active(this.dominators(), borrow, location) {
                        assert!(borrow.kind.allows_two_phase_borrow());
                        return ControlFlow::Continue(());
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
                    ControlFlow::Break(())
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
                    ControlFlow::Break(())
                }
            },
        );

        error_reported
    }

    /// Through #123739, `BackwardIncompatibleDropHint`s (BIDs) are introduced.
    /// We would like to emit lints whether borrow checking fails at these future drop locations.
    #[instrument(level = "debug", skip(self, state))]
    fn check_backward_incompatible_drop(
        &mut self,
        location: Location,
        place: Place<'tcx>,
        state: &BorrowckDomain,
    ) {
        let tcx = self.infcx.tcx;
        // If this type does not need `Drop`, then treat it like a `StorageDead`.
        // This is needed because we track the borrows of refs to thread locals,
        // and we'll ICE because we don't track borrows behind shared references.
        let sd = if place.ty(self.body, tcx).ty.needs_drop(tcx, self.body.typing_env(tcx)) {
            AccessDepth::Drop
        } else {
            AccessDepth::Shallow(None)
        };

        let borrows_in_scope = self.borrows_in_scope(location, state);

        // This is a very simplified version of `Self::check_access_for_conflict`.
        // We are here checking on BIDs and specifically still-live borrows of data involving the BIDs.
        each_borrow_involving_path(
            self,
            self.infcx.tcx,
            self.body,
            (sd, place),
            self.borrow_set,
            |borrow_index| borrows_in_scope.contains(borrow_index),
            |this, _borrow_index, borrow| {
                if matches!(borrow.kind, BorrowKind::Fake(_)) {
                    return ControlFlow::Continue(());
                }
                let borrowed = this.retrieve_borrow_spans(borrow).var_or_use_path_span();
                let explain = this.explain_why_borrow_contains_point(
                    location,
                    borrow,
                    Some((WriteKind::StorageDeadOrDrop, place)),
                );
                this.infcx.tcx.node_span_lint(
                    TAIL_EXPR_DROP_ORDER,
                    CRATE_HIR_ID,
                    borrowed,
                    |diag| {
                        session_diagnostics::TailExprDropOrder { borrowed }.decorate_lint(diag);
                        explain.add_explanation_to_diagnostic(&this, diag, "", None, None);
                    },
                );
                // We may stop at the first case
                ControlFlow::Break(())
            },
        );
    }

    fn mutate_place(
        &mut self,
        location: Location,
        place_span: (Place<'tcx>, Span),
        kind: AccessDepth,
        state: &BorrowckDomain,
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
        (rvalue, span): (&Rvalue<'tcx>, Span),
        state: &BorrowckDomain,
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
                        if bk.allows_two_phase_borrow() {
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

            &Rvalue::RawPtr(kind, place) => {
                let access_kind = match kind {
                    RawPtrKind::Mut => (
                        Deep,
                        Write(WriteKind::MutableBorrow(BorrowKind::Mut {
                            kind: MutBorrowKind::Default,
                        })),
                    ),
                    RawPtrKind::Const => (Deep, Read(ReadKind::Borrow(BorrowKind::Shared))),
                    RawPtrKind::FakeForPtrMetadata => {
                        (Shallow(Some(ArtificialField::ArrayLength)), Read(ReadKind::Copy))
                    }
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
                        let used_mut_upvars = self.root_cx.used_mut_upvars(def_id);
                        debug!("{:?} used_mut_upvars={:?}", def_id, used_mut_upvars);
                        // FIXME: We're cloning the `SmallVec` here to avoid borrowing `root_cx`
                        // when calling `propagate_closure_used_mut_upvar`. This should ideally
                        // be unnecessary.
                        for field in used_mut_upvars.clone() {
                            self.propagate_closure_used_mut_upvar(&operands[field]);
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

            Rvalue::WrapUnsafeBinder(op, _) => {
                self.consume_operand(location, (op, span), state);
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

                        match stmt.kind {
                            StatementKind::Assign(box (
                                _,
                                Rvalue::Ref(_, _, source)
                                | Rvalue::Use(Operand::Copy(source) | Operand::Move(source)),
                            )) => {
                                propagate_closure_used_mut_place(self, source);
                            }
                            _ => {
                                bug!(
                                    "closures should only capture user variables \
                                 or references to user variables"
                                );
                            }
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
        (operand, span): (&Operand<'tcx>, Span),
        state: &BorrowckDomain,
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
        let might_be_alive = if self.body.local_decls[root_place.local].is_ref_to_thread_local() {
            // Thread-locals might be dropped after the function exits
            // We have to dereference the outer reference because
            // borrows don't conflict behind shared references.
            root_place.projection = TyCtxtConsts::DEREF_PROJECTION;
            true
        } else {
            false
        };

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

    fn check_activations(&mut self, location: Location, span: Span, state: &BorrowckDomain) {
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
                        self.move_errors.push(MoveError::new(
                            place,
                            location,
                            BorrowedContent {
                                target_place: place_ref.project_deeper(&[elem], tcx),
                            },
                        ));
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
                    | ty::UnsafeBinder(_)
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
                    | ty::UnsafeBinder(_)
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
                | ProjectionElem::Downcast(_, _)
                | ProjectionElem::UnwrapUnsafeBinder(_) => (),
            }

            place_ty = place_ty.projection_ty(tcx, elem);
        }
    }

    fn check_if_full_path_is_moved(
        &mut self,
        location: Location,
        desired_action: InitializationRequiringAction,
        place_span: (PlaceRef<'tcx>, Span),
        state: &BorrowckDomain,
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
        // self.locals_checked_for_initialization.entry(mpi).or_default().insert(location);
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
        maybe_uninits: &MixedBitSet<MovePathIndex>,
        from: u64,
        to: u64,
    ) {
        if let Some(mpi) = self.move_path_for_place(place_span.0) {
            // self.locals_checked_for_initialization.entry(mpi).or_default().insert(location);
            let move_paths = &self.move_data.move_paths;

            let root_path = &move_paths[mpi];
            for (child_mpi, child_move_path) in root_path.children(move_paths) {
                let last_proj = child_move_path.place.projection.last().unwrap();
                if let ProjectionElem::ConstantIndex { offset, from_end, .. } = last_proj {
                    debug_assert!(!from_end, "Array constant indexing shouldn't be `from_end`.");

                    if (from..to).contains(offset) {
                        // self.locals_checked_for_initialization.entry(child_mpi).or_default().insert(location);
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
        state: &BorrowckDomain,
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

            // self.locals_checked_for_initialization.entry(mpi).or_default().insert(location);

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
        state: &BorrowckDomain,
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

                ProjectionElem::UnwrapUnsafeBinder(_) => {
                    check_parent_of_field(self, location, place_base, span, state);
                }

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
            state: &BorrowckDomain,
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
                // this.locals_checked_for_initialization.entry(mpi).or_default().insert(location);

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
        state: &BorrowckDomain,
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
                        self.add_used_mut(root_place, state, location);
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
                        self.add_used_mut(root_place, state, location);
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
        // self.locals_checked_for_initialization.entry(place.local).or_default().insert(location);
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

    fn is_local_ever_initialized(&self, local: Local, state: &BorrowckDomain) -> Option<InitIndex> {
        let mpi = self.move_data.rev_lookup.find_local(local)?;
        let ii = &self.move_data.init_path_map[mpi];
        ii.into_iter().find(|&&index| state.ever_inits.contains(index)).copied()
    }

    /// Adds the place into the used mutable variables set
    fn add_used_mut(
        &mut self,
        root_place: RootPlace<'tcx>,
        state: &BorrowckDomain,
        _location: Location,
    ) {
        match root_place {
            RootPlace { place_local: local, place_projection: [], is_local_mutation_allowed } => {
                // If the local may have been initialized, and it is now currently being
                // mutated, then it is justified to be annotated with the `mut`
                // keyword, since the mutation may be a possible reassignment.

                // FIXME: C'est pas super important ce use case de unused mut, on pourrait l'ignorer
                // ou le faire différemment.
                // if is_local_mutation_allowed != LocalMutationIsAllowed::Yes {
                //     self.locals_checked_for_initialization
                //         .entry(local)
                //         .or_default()
                //         .insert(_location);
                // }

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
                    | ProjectionElem::Downcast(..)
                    | ProjectionElem::UnwrapUnsafeBinder(_) => {
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

    fn lint_unused_mut(&self) {
        let tcx = self.infcx.tcx;
        let body = self.body;
        for local in body.mut_vars_and_args_iter().filter(|local| !self.used_mut.contains(local)) {
            let local_decl = &body.local_decls[local];
            let ClearCrossCrate::Set(SourceScopeLocalData { lint_root, .. }) =
                body.source_scopes[local_decl.source_info.scope].local_data
            else {
                continue;
            };

            // Skip over locals that begin with an underscore or have no name
            if self.local_excluded_from_unused_mut_lint(local) {
                continue;
            }

            let span = local_decl.source_info.span;
            if span.desugaring_kind().is_some() {
                // If the `mut` arises as part of a desugaring, we should ignore it.
                continue;
            }

            let mut_span = tcx.sess.source_map().span_until_non_whitespace(span);

            tcx.emit_node_span_lint(UNUSED_MUT, lint_root, span, VarNeedNotMut { span: mut_span })
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
