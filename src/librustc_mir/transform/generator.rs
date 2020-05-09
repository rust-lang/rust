//! This is the implementation of the pass which transforms generators into state machines.
//!
//! MIR generation for generators creates a function which has a self argument which
//! passes by value. This argument is effectively a generator type which only contains upvars and
//! is only used for this argument inside the MIR for the generator.
//! It is passed by value to enable upvars to be moved out of it. Drop elaboration runs on that
//! MIR before this pass and creates drop flags for MIR locals.
//! It will also drop the generator argument (which only consists of upvars) if any of the upvars
//! are moved out of. This pass elaborates the drops of upvars / generator argument in the case
//! that none of the upvars were moved out of. This is because we cannot have any drops of this
//! generator in the MIR, since it is used to create the drop glue for the generator. We'd get
//! infinite recursion otherwise.
//!
//! This pass creates the implementation for the Generator::resume function and the drop shim
//! for the generator based on the MIR input. It converts the generator argument from Self to
//! &mut Self adding derefs in the MIR as needed. It computes the final layout of the generator
//! struct which looks like this:
//!     First upvars are stored
//!     It is followed by the generator state field.
//!     Then finally the MIR locals which are live across a suspension point are stored.
//!
//!     struct Generator {
//!         upvars...,
//!         state: u32,
//!         mir_locals...,
//!     }
//!
//! This pass computes the meaning of the state field and the MIR locals which are live
//! across a suspension point. There are however three hardcoded generator states:
//!     0 - Generator have not been resumed yet
//!     1 - Generator has returned / is completed
//!     2 - Generator has been poisoned
//!
//! It also rewrites `return x` and `yield y` as setting a new generator state and returning
//! GeneratorState::Complete(x) and GeneratorState::Yielded(y) respectively.
//! MIR locals which are live across a suspension point are moved to the generator struct
//! with references to them being updated with references to the generator struct.
//!
//! The pass creates two functions which have a switch on the generator state giving
//! the action to take.
//!
//! One of them is the implementation of Generator::resume.
//! For generators with state 0 (unresumed) it starts the execution of the generator.
//! For generators with state 1 (returned) and state 2 (poisoned) it panics.
//! Otherwise it continues the execution from the last suspension point.
//!
//! The other function is the drop glue for the generator.
//! For generators with state 0 (unresumed) it drops the upvars of the generator.
//! For generators with state 1 (returned) and state 2 (poisoned) it does nothing.
//! Otherwise it drops all the values in scope at the last suspension point.

use crate::dataflow::impls::{
    MaybeBorrowedLocals, MaybeLiveLocals, MaybeRequiresStorage, MaybeStorageLive,
};
use crate::dataflow::{self, Analysis};
use crate::transform::no_landing_pads::no_landing_pads;
use crate::transform::simplify;
use crate::transform::{MirPass, MirSource};
use crate::util::dump_mir;
use crate::util::storage;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_index::bit_set::{BitMatrix, BitSet};
use rustc_index::vec::{Idx, IndexVec};
use rustc_middle::mir::visit::{MutVisitor, PlaceContext};
use rustc_middle::mir::*;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::GeneratorSubsts;
use rustc_middle::ty::{self, AdtDef, Ty, TyCtxt};
use rustc_target::abi::VariantIdx;
use rustc_target::spec::PanicStrategy;
use std::borrow::Cow;
use std::iter;

pub struct StateTransform;

struct RenameLocalVisitor<'tcx> {
    from: Local,
    to: Local,
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for RenameLocalVisitor<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _: Location) {
        if *local == self.from {
            *local = self.to;
        }
    }
}

struct DerefArgVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for DerefArgVisitor<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _: Location) {
        assert_ne!(*local, SELF_ARG);
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, context: PlaceContext, location: Location) {
        if place.local == SELF_ARG {
            replace_base(
                place,
                Place {
                    local: SELF_ARG,
                    projection: self.tcx().intern_place_elems(&[ProjectionElem::Deref]),
                },
                self.tcx,
            );
        } else {
            self.visit_local(&mut place.local, context, location);

            for elem in place.projection.iter() {
                if let PlaceElem::Index(local) = elem {
                    assert_ne!(*local, SELF_ARG);
                }
            }
        }
    }
}

struct PinArgVisitor<'tcx> {
    ref_gen_ty: Ty<'tcx>,
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for PinArgVisitor<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _: Location) {
        assert_ne!(*local, SELF_ARG);
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, context: PlaceContext, location: Location) {
        if place.local == SELF_ARG {
            replace_base(
                place,
                Place {
                    local: SELF_ARG,
                    projection: self.tcx().intern_place_elems(&[ProjectionElem::Field(
                        Field::new(0),
                        self.ref_gen_ty,
                    )]),
                },
                self.tcx,
            );
        } else {
            self.visit_local(&mut place.local, context, location);

            for elem in place.projection.iter() {
                if let PlaceElem::Index(local) = elem {
                    assert_ne!(*local, SELF_ARG);
                }
            }
        }
    }
}

fn replace_base<'tcx>(place: &mut Place<'tcx>, new_base: Place<'tcx>, tcx: TyCtxt<'tcx>) {
    place.local = new_base.local;

    let mut new_projection = new_base.projection.to_vec();
    new_projection.append(&mut place.projection.to_vec());

    place.projection = tcx.intern_place_elems(&new_projection);
}

const SELF_ARG: Local = Local::from_u32(1);

/// Generator has not been resumed yet.
const UNRESUMED: usize = GeneratorSubsts::UNRESUMED;
/// Generator has returned / is completed.
const RETURNED: usize = GeneratorSubsts::RETURNED;
/// Generator has panicked and is poisoned.
const POISONED: usize = GeneratorSubsts::POISONED;

/// A `yield` point in the generator.
struct SuspensionPoint<'tcx> {
    /// State discriminant used when suspending or resuming at this point.
    state: usize,
    /// The block to jump to after resumption.
    resume: BasicBlock,
    /// Where to move the resume argument after resumption.
    resume_arg: Place<'tcx>,
    /// Which block to jump to if the generator is dropped in this state.
    drop: Option<BasicBlock>,
    /// Set of locals that have live storage while at this suspension point.
    storage_liveness: BitSet<Local>,
}

struct TransformVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    state_adt_ref: &'tcx AdtDef,
    state_substs: SubstsRef<'tcx>,

    // The type of the discriminant in the generator struct
    discr_ty: Ty<'tcx>,

    // Mapping from Local to (type of local, generator struct index)
    // FIXME(eddyb) This should use `IndexVec<Local, Option<_>>`.
    remap: FxHashMap<Local, (Ty<'tcx>, VariantIdx, usize)>,

    // A map from a suspension point in a block to the locals which have live storage at that point
    storage_liveness: IndexVec<BasicBlock, Option<BitSet<Local>>>,

    // A list of suspension points, generated during the transform
    suspension_points: Vec<SuspensionPoint<'tcx>>,

    // The set of locals that have no `StorageLive`/`StorageDead` annotations.
    always_live_locals: storage::AlwaysLiveLocals,

    // The original RETURN_PLACE local
    new_ret_local: Local,
}

impl TransformVisitor<'tcx> {
    // Make a GeneratorState rvalue
    fn make_state(&self, idx: VariantIdx, val: Operand<'tcx>) -> Rvalue<'tcx> {
        let adt = AggregateKind::Adt(self.state_adt_ref, idx, self.state_substs, None, None);
        Rvalue::Aggregate(box adt, vec![val])
    }

    // Create a Place referencing a generator struct field
    fn make_field(&self, variant_index: VariantIdx, idx: usize, ty: Ty<'tcx>) -> Place<'tcx> {
        let self_place = Place::from(SELF_ARG);
        let base = self.tcx.mk_place_downcast_unnamed(self_place, variant_index);
        let mut projection = base.projection.to_vec();
        projection.push(ProjectionElem::Field(Field::new(idx), ty));

        Place { local: base.local, projection: self.tcx.intern_place_elems(&projection) }
    }

    // Create a statement which changes the discriminant
    fn set_discr(&self, state_disc: VariantIdx, source_info: SourceInfo) -> Statement<'tcx> {
        let self_place = Place::from(SELF_ARG);
        Statement {
            source_info,
            kind: StatementKind::SetDiscriminant {
                place: box self_place,
                variant_index: state_disc,
            },
        }
    }

    // Create a statement which reads the discriminant into a temporary
    fn get_discr(&self, body: &mut Body<'tcx>) -> (Statement<'tcx>, Place<'tcx>) {
        let temp_decl = LocalDecl::new(self.tcx.types.isize, body.span).internal();
        let local_decls_len = body.local_decls.push(temp_decl);
        let temp = Place::from(local_decls_len);

        let self_place = Place::from(SELF_ARG);
        let assign = Statement {
            source_info: SourceInfo::outermost(body.span),
            kind: StatementKind::Assign(box (temp, Rvalue::Discriminant(self_place))),
        };
        (assign, temp)
    }
}

impl MutVisitor<'tcx> for TransformVisitor<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _: Location) {
        assert_eq!(self.remap.get(local), None);
    }

    fn visit_place(
        &mut self,
        place: &mut Place<'tcx>,
        _context: PlaceContext,
        _location: Location,
    ) {
        // Replace an Local in the remap with a generator struct access
        if let Some(&(ty, variant_index, idx)) = self.remap.get(&place.local) {
            replace_base(place, self.make_field(variant_index, idx, ty), self.tcx);
        }
    }

    fn visit_basic_block_data(&mut self, block: BasicBlock, data: &mut BasicBlockData<'tcx>) {
        // Remove StorageLive and StorageDead statements for remapped locals
        data.retain_statements(|s| match s.kind {
            StatementKind::StorageLive(l) | StatementKind::StorageDead(l) => {
                !self.remap.contains_key(&l)
            }
            _ => true,
        });

        let ret_val = match data.terminator().kind {
            TerminatorKind::Return => Some((
                VariantIdx::new(1),
                None,
                Operand::Move(Place::from(self.new_ret_local)),
                None,
            )),
            TerminatorKind::Yield { ref value, resume, resume_arg, drop } => {
                Some((VariantIdx::new(0), Some((resume, resume_arg)), value.clone(), drop))
            }
            _ => None,
        };

        if let Some((state_idx, resume, v, drop)) = ret_val {
            let source_info = data.terminator().source_info;
            // We must assign the value first in case it gets declared dead below
            data.statements.push(Statement {
                source_info,
                kind: StatementKind::Assign(box (
                    Place::return_place(),
                    self.make_state(state_idx, v),
                )),
            });
            let state = if let Some((resume, resume_arg)) = resume {
                // Yield
                let state = 3 + self.suspension_points.len();

                // The resume arg target location might itself be remapped if its base local is
                // live across a yield.
                let resume_arg =
                    if let Some(&(ty, variant, idx)) = self.remap.get(&resume_arg.local) {
                        self.make_field(variant, idx, ty)
                    } else {
                        resume_arg
                    };

                self.suspension_points.push(SuspensionPoint {
                    state,
                    resume,
                    resume_arg,
                    drop,
                    storage_liveness: self.storage_liveness[block].clone().unwrap(),
                });

                VariantIdx::new(state)
            } else {
                // Return
                VariantIdx::new(RETURNED) // state for returned
            };
            data.statements.push(self.set_discr(state, source_info));
            data.terminator_mut().kind = TerminatorKind::Return;
        }

        self.super_basic_block_data(block, data);
    }
}

fn make_generator_state_argument_indirect<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let gen_ty = body.local_decls.raw[1].ty;

    let ref_gen_ty =
        tcx.mk_ref(tcx.lifetimes.re_erased, ty::TypeAndMut { ty: gen_ty, mutbl: Mutability::Mut });

    // Replace the by value generator argument
    body.local_decls.raw[1].ty = ref_gen_ty;

    // Add a deref to accesses of the generator state
    DerefArgVisitor { tcx }.visit_body(body);
}

fn make_generator_state_argument_pinned<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let ref_gen_ty = body.local_decls.raw[1].ty;

    let pin_did = tcx.lang_items().pin_type().unwrap();
    let pin_adt_ref = tcx.adt_def(pin_did);
    let substs = tcx.intern_substs(&[ref_gen_ty.into()]);
    let pin_ref_gen_ty = tcx.mk_adt(pin_adt_ref, substs);

    // Replace the by ref generator argument
    body.local_decls.raw[1].ty = pin_ref_gen_ty;

    // Add the Pin field access to accesses of the generator state
    PinArgVisitor { ref_gen_ty, tcx }.visit_body(body);
}

/// Allocates a new local and replaces all references of `local` with it. Returns the new local.
///
/// `local` will be changed to a new local decl with type `ty`.
///
/// Note that the new local will be uninitialized. It is the caller's responsibility to assign some
/// valid value to it before its first use.
fn replace_local<'tcx>(
    local: Local,
    ty: Ty<'tcx>,
    body: &mut Body<'tcx>,
    tcx: TyCtxt<'tcx>,
) -> Local {
    let new_decl = LocalDecl::new(ty, body.span);
    let new_local = body.local_decls.push(new_decl);
    body.local_decls.swap(local, new_local);

    RenameLocalVisitor { from: local, to: new_local, tcx }.visit_body(body);

    new_local
}

struct LivenessInfo {
    /// Which locals are live across any suspension point.
    ///
    /// GeneratorSavedLocal is indexed in terms of the elements in this set;
    /// i.e. GeneratorSavedLocal::new(1) corresponds to the second local
    /// included in this set.
    live_locals: BitSet<Local>,

    /// The set of saved locals live at each suspension point.
    live_locals_at_suspension_points: Vec<BitSet<GeneratorSavedLocal>>,

    /// For every saved local, the set of other saved locals that are
    /// storage-live at the same time as this local. We cannot overlap locals in
    /// the layout which have conflicting storage.
    storage_conflicts: BitMatrix<GeneratorSavedLocal, GeneratorSavedLocal>,

    /// For every suspending block, the locals which are storage-live across
    /// that suspension point.
    storage_liveness: IndexVec<BasicBlock, Option<BitSet<Local>>>,
}

fn locals_live_across_suspend_points(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    source: MirSource<'tcx>,
    always_live_locals: &storage::AlwaysLiveLocals,
    movable: bool,
) -> LivenessInfo {
    let def_id = source.def_id();
    let body_ref: &Body<'_> = &body;

    // Calculate when MIR locals have live storage. This gives us an upper bound of their
    // lifetimes.
    let mut storage_live = MaybeStorageLive::new(always_live_locals.clone())
        .into_engine(tcx, body_ref, def_id)
        .iterate_to_fixpoint()
        .into_results_cursor(body_ref);

    // Calculate the MIR locals which have been previously
    // borrowed (even if they are still active).
    let borrowed_locals_results =
        MaybeBorrowedLocals::all_borrows().into_engine(tcx, body_ref, def_id).iterate_to_fixpoint();

    let mut borrowed_locals_cursor =
        dataflow::ResultsCursor::new(body_ref, &borrowed_locals_results);

    // Calculate the MIR locals that we actually need to keep storage around
    // for.
    let requires_storage_results = MaybeRequiresStorage::new(body, &borrowed_locals_results)
        .into_engine(tcx, body_ref, def_id)
        .iterate_to_fixpoint();
    let mut requires_storage_cursor =
        dataflow::ResultsCursor::new(body_ref, &requires_storage_results);

    // Calculate the liveness of MIR locals ignoring borrows.
    let mut liveness = MaybeLiveLocals
        .into_engine(tcx, body_ref, def_id)
        .iterate_to_fixpoint()
        .into_results_cursor(body_ref);

    let mut storage_liveness_map = IndexVec::from_elem(None, body.basic_blocks());
    let mut live_locals_at_suspension_points = Vec::new();
    let mut live_locals_at_any_suspension_point = BitSet::new_empty(body.local_decls.len());

    for (block, data) in body.basic_blocks().iter_enumerated() {
        if let TerminatorKind::Yield { .. } = data.terminator().kind {
            let loc = Location { block, statement_index: data.statements.len() };

            liveness.seek_to_block_end(block);
            let mut live_locals = liveness.get().clone();

            if !movable {
                // The `liveness` variable contains the liveness of MIR locals ignoring borrows.
                // This is correct for movable generators since borrows cannot live across
                // suspension points. However for immovable generators we need to account for
                // borrows, so we conseratively assume that all borrowed locals are live until
                // we find a StorageDead statement referencing the locals.
                // To do this we just union our `liveness` result with `borrowed_locals`, which
                // contains all the locals which has been borrowed before this suspension point.
                // If a borrow is converted to a raw reference, we must also assume that it lives
                // forever. Note that the final liveness is still bounded by the storage liveness
                // of the local, which happens using the `intersect` operation below.
                borrowed_locals_cursor.seek_before_primary_effect(loc);
                live_locals.union(borrowed_locals_cursor.get());
            }

            // Store the storage liveness for later use so we can restore the state
            // after a suspension point
            storage_live.seek_before_primary_effect(loc);
            storage_liveness_map[block] = Some(storage_live.get().clone());

            // Locals live are live at this point only if they are used across
            // suspension points (the `liveness` variable)
            // and their storage is required (the `storage_required` variable)
            requires_storage_cursor.seek_before_primary_effect(loc);
            live_locals.intersect(requires_storage_cursor.get());

            // The generator argument is ignored.
            live_locals.remove(SELF_ARG);

            debug!("loc = {:?}, live_locals = {:?}", loc, live_locals);

            // Add the locals live at this suspension point to the set of locals which live across
            // any suspension points
            live_locals_at_any_suspension_point.union(&live_locals);

            live_locals_at_suspension_points.push(live_locals);
        }
    }
    debug!("live_locals_anywhere = {:?}", live_locals_at_any_suspension_point);

    // Renumber our liveness_map bitsets to include only the locals we are
    // saving.
    let live_locals_at_suspension_points = live_locals_at_suspension_points
        .iter()
        .map(|live_here| renumber_bitset(&live_here, &live_locals_at_any_suspension_point))
        .collect();

    let storage_conflicts = compute_storage_conflicts(
        body_ref,
        &live_locals_at_any_suspension_point,
        always_live_locals.clone(),
        requires_storage_results,
    );

    LivenessInfo {
        live_locals: live_locals_at_any_suspension_point,
        live_locals_at_suspension_points,
        storage_conflicts,
        storage_liveness: storage_liveness_map,
    }
}

/// Renumbers the items present in `stored_locals` and applies the renumbering
/// to 'input`.
///
/// For example, if `stored_locals = [1, 3, 5]`, this would be renumbered to
/// `[0, 1, 2]`. Thus, if `input = [3, 5]` we would return `[1, 2]`.
fn renumber_bitset(
    input: &BitSet<Local>,
    stored_locals: &BitSet<Local>,
) -> BitSet<GeneratorSavedLocal> {
    assert!(stored_locals.superset(&input), "{:?} not a superset of {:?}", stored_locals, input);
    let mut out = BitSet::new_empty(stored_locals.count());
    for (idx, local) in stored_locals.iter().enumerate() {
        let saved_local = GeneratorSavedLocal::from(idx);
        if input.contains(local) {
            out.insert(saved_local);
        }
    }
    debug!("renumber_bitset({:?}, {:?}) => {:?}", input, stored_locals, out);
    out
}

/// For every saved local, looks for which locals are StorageLive at the same
/// time. Generates a bitset for every local of all the other locals that may be
/// StorageLive simultaneously with that local. This is used in the layout
/// computation; see `GeneratorLayout` for more.
fn compute_storage_conflicts(
    body: &'mir Body<'tcx>,
    stored_locals: &BitSet<Local>,
    always_live_locals: storage::AlwaysLiveLocals,
    requires_storage: dataflow::Results<'tcx, MaybeRequiresStorage<'mir, 'tcx>>,
) -> BitMatrix<GeneratorSavedLocal, GeneratorSavedLocal> {
    assert_eq!(body.local_decls.len(), stored_locals.domain_size());

    debug!("compute_storage_conflicts({:?})", body.span);
    debug!("always_live = {:?}", always_live_locals);

    // Locals that are always live or ones that need to be stored across
    // suspension points are not eligible for overlap.
    let mut ineligible_locals = always_live_locals.into_inner();
    ineligible_locals.intersect(stored_locals);

    // Compute the storage conflicts for all eligible locals.
    let mut visitor = StorageConflictVisitor {
        body,
        stored_locals: &stored_locals,
        local_conflicts: BitMatrix::from_row_n(&ineligible_locals, body.local_decls.len()),
    };

    // Visit only reachable basic blocks. The exact order is not important.
    let reachable_blocks = traversal::preorder(body).map(|(bb, _)| bb);
    requires_storage.visit_with(body, reachable_blocks, &mut visitor);

    let local_conflicts = visitor.local_conflicts;

    // Compress the matrix using only stored locals (Local -> GeneratorSavedLocal).
    //
    // NOTE: Today we store a full conflict bitset for every local. Technically
    // this is twice as many bits as we need, since the relation is symmetric.
    // However, in practice these bitsets are not usually large. The layout code
    // also needs to keep track of how many conflicts each local has, so it's
    // simpler to keep it this way for now.
    let mut storage_conflicts = BitMatrix::new(stored_locals.count(), stored_locals.count());
    for (idx_a, local_a) in stored_locals.iter().enumerate() {
        let saved_local_a = GeneratorSavedLocal::new(idx_a);
        if ineligible_locals.contains(local_a) {
            // Conflicts with everything.
            storage_conflicts.insert_all_into_row(saved_local_a);
        } else {
            // Keep overlap information only for stored locals.
            for (idx_b, local_b) in stored_locals.iter().enumerate() {
                let saved_local_b = GeneratorSavedLocal::new(idx_b);
                if local_conflicts.contains(local_a, local_b) {
                    storage_conflicts.insert(saved_local_a, saved_local_b);
                }
            }
        }
    }
    storage_conflicts
}

struct StorageConflictVisitor<'mir, 'tcx, 's> {
    body: &'mir Body<'tcx>,
    stored_locals: &'s BitSet<Local>,
    // FIXME(tmandry): Consider using sparse bitsets here once we have good
    // benchmarks for generators.
    local_conflicts: BitMatrix<Local, Local>,
}

impl dataflow::ResultsVisitor<'mir, 'tcx> for StorageConflictVisitor<'mir, 'tcx, '_> {
    type FlowState = BitSet<Local>;

    fn visit_statement_before_primary_effect(
        &mut self,
        state: &Self::FlowState,
        _statement: &'mir Statement<'tcx>,
        loc: Location,
    ) {
        self.apply_state(state, loc);
    }

    fn visit_terminator_before_primary_effect(
        &mut self,
        state: &Self::FlowState,
        _terminator: &'mir Terminator<'tcx>,
        loc: Location,
    ) {
        self.apply_state(state, loc);
    }
}

impl<'body, 'tcx, 's> StorageConflictVisitor<'body, 'tcx, 's> {
    fn apply_state(&mut self, flow_state: &BitSet<Local>, loc: Location) {
        // Ignore unreachable blocks.
        if self.body.basic_blocks()[loc.block].terminator().kind == TerminatorKind::Unreachable {
            return;
        }

        let mut eligible_storage_live = flow_state.clone();
        eligible_storage_live.intersect(&self.stored_locals);

        for local in eligible_storage_live.iter() {
            self.local_conflicts.union_row_with(&eligible_storage_live, local);
        }

        if eligible_storage_live.count() > 1 {
            trace!("at {:?}, eligible_storage_live={:?}", loc, eligible_storage_live);
        }
    }
}

fn compute_layout<'tcx>(
    tcx: TyCtxt<'tcx>,
    source: MirSource<'tcx>,
    upvars: &Vec<Ty<'tcx>>,
    interior: Ty<'tcx>,
    always_live_locals: &storage::AlwaysLiveLocals,
    movable: bool,
    body: &mut Body<'tcx>,
) -> (
    FxHashMap<Local, (Ty<'tcx>, VariantIdx, usize)>,
    GeneratorLayout<'tcx>,
    IndexVec<BasicBlock, Option<BitSet<Local>>>,
) {
    // Use a liveness analysis to compute locals which are live across a suspension point
    let LivenessInfo {
        live_locals,
        live_locals_at_suspension_points,
        storage_conflicts,
        storage_liveness,
    } = locals_live_across_suspend_points(tcx, body, source, always_live_locals, movable);

    // Erase regions from the types passed in from typeck so we can compare them with
    // MIR types
    let allowed_upvars = tcx.erase_regions(upvars);
    let allowed = match interior.kind {
        ty::GeneratorWitness(s) => tcx.erase_late_bound_regions(&s),
        _ => bug!(),
    };

    let param_env = tcx.param_env(source.def_id());

    for (local, decl) in body.local_decls.iter_enumerated() {
        // Ignore locals which are internal or not live
        if !live_locals.contains(local) || decl.internal {
            continue;
        }
        let decl_ty = tcx.normalize_erasing_regions(param_env, decl.ty);

        // Sanity check that typeck knows about the type of locals which are
        // live across a suspension point
        if !allowed.contains(&decl_ty) && !allowed_upvars.contains(&decl_ty) {
            span_bug!(
                body.span,
                "Broken MIR: generator contains type {} in MIR, \
                       but typeck only knows about {}",
                decl.ty,
                interior
            );
        }
    }

    // Gather live local types and their indices.
    let mut locals = IndexVec::<GeneratorSavedLocal, _>::new();
    let mut tys = IndexVec::<GeneratorSavedLocal, _>::new();
    for (idx, local) in live_locals.iter().enumerate() {
        locals.push(local);
        tys.push(body.local_decls[local].ty);
        debug!("generator saved local {:?} => {:?}", GeneratorSavedLocal::from(idx), local);
    }

    // Leave empty variants for the UNRESUMED, RETURNED, and POISONED states.
    const RESERVED_VARIANTS: usize = 3;

    // Build the generator variant field list.
    // Create a map from local indices to generator struct indices.
    let mut variant_fields: IndexVec<VariantIdx, IndexVec<Field, GeneratorSavedLocal>> =
        iter::repeat(IndexVec::new()).take(RESERVED_VARIANTS).collect();
    let mut remap = FxHashMap::default();
    for (suspension_point_idx, live_locals) in live_locals_at_suspension_points.iter().enumerate() {
        let variant_index = VariantIdx::from(RESERVED_VARIANTS + suspension_point_idx);
        let mut fields = IndexVec::new();
        for (idx, saved_local) in live_locals.iter().enumerate() {
            fields.push(saved_local);
            // Note that if a field is included in multiple variants, we will
            // just use the first one here. That's fine; fields do not move
            // around inside generators, so it doesn't matter which variant
            // index we access them by.
            remap.entry(locals[saved_local]).or_insert((tys[saved_local], variant_index, idx));
        }
        variant_fields.push(fields);
    }
    debug!("generator variant_fields = {:?}", variant_fields);
    debug!("generator storage_conflicts = {:#?}", storage_conflicts);

    let layout = GeneratorLayout { field_tys: tys, variant_fields, storage_conflicts };

    (remap, layout, storage_liveness)
}

/// Replaces the entry point of `body` with a block that switches on the generator discriminant and
/// dispatches to blocks according to `cases`.
///
/// After this function, the former entry point of the function will be bb1.
fn insert_switch<'tcx>(
    body: &mut Body<'tcx>,
    cases: Vec<(usize, BasicBlock)>,
    transform: &TransformVisitor<'tcx>,
    default: TerminatorKind<'tcx>,
) {
    let default_block = insert_term_block(body, default);
    let (assign, discr) = transform.get_discr(body);
    let switch = TerminatorKind::SwitchInt {
        discr: Operand::Move(discr),
        switch_ty: transform.discr_ty,
        values: Cow::from(cases.iter().map(|&(i, _)| i as u128).collect::<Vec<_>>()),
        targets: cases.iter().map(|&(_, d)| d).chain(iter::once(default_block)).collect(),
    };

    let source_info = SourceInfo::outermost(body.span);
    body.basic_blocks_mut().raw.insert(
        0,
        BasicBlockData {
            statements: vec![assign],
            terminator: Some(Terminator { source_info, kind: switch }),
            is_cleanup: false,
        },
    );

    let blocks = body.basic_blocks_mut().iter_mut();

    for target in blocks.flat_map(|b| b.terminator_mut().successors_mut()) {
        *target = BasicBlock::new(target.index() + 1);
    }
}

fn elaborate_generator_drops<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, body: &mut Body<'tcx>) {
    use crate::shim::DropShimElaborator;
    use crate::util::elaborate_drops::{elaborate_drop, Unwind};
    use crate::util::patch::MirPatch;

    // Note that `elaborate_drops` only drops the upvars of a generator, and
    // this is ok because `open_drop` can only be reached within that own
    // generator's resume function.

    let param_env = tcx.param_env(def_id);

    let mut elaborator = DropShimElaborator { body, patch: MirPatch::new(body), tcx, param_env };

    for (block, block_data) in body.basic_blocks().iter_enumerated() {
        let (target, unwind, source_info) = match block_data.terminator() {
            Terminator { source_info, kind: TerminatorKind::Drop { location, target, unwind } } => {
                if let Some(local) = location.as_local() {
                    if local == SELF_ARG {
                        (target, unwind, source_info)
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            _ => continue,
        };
        let unwind = if block_data.is_cleanup {
            Unwind::InCleanup
        } else {
            Unwind::To(unwind.unwrap_or_else(|| elaborator.patch.resume_block()))
        };
        elaborate_drop(
            &mut elaborator,
            *source_info,
            Place::from(SELF_ARG),
            (),
            *target,
            unwind,
            block,
        );
    }
    elaborator.patch.apply(body);
}

fn create_generator_drop_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    transform: &TransformVisitor<'tcx>,
    source: MirSource<'tcx>,
    gen_ty: Ty<'tcx>,
    body: &mut Body<'tcx>,
    drop_clean: BasicBlock,
) -> Body<'tcx> {
    let mut body = body.clone();
    body.arg_count = 1; // make sure the resume argument is not included here

    let source_info = SourceInfo::outermost(body.span);

    let mut cases = create_cases(&mut body, transform, Operation::Drop);

    cases.insert(0, (UNRESUMED, drop_clean));

    // The returned state and the poisoned state fall through to the default
    // case which is just to return

    insert_switch(&mut body, cases, &transform, TerminatorKind::Return);

    for block in body.basic_blocks_mut() {
        let kind = &mut block.terminator_mut().kind;
        if let TerminatorKind::GeneratorDrop = *kind {
            *kind = TerminatorKind::Return;
        }
    }

    // Replace the return variable
    body.local_decls[RETURN_PLACE] = LocalDecl::with_source_info(tcx.mk_unit(), source_info);

    make_generator_state_argument_indirect(tcx, &mut body);

    // Change the generator argument from &mut to *mut
    body.local_decls[SELF_ARG] = LocalDecl::with_source_info(
        tcx.mk_ptr(ty::TypeAndMut { ty: gen_ty, mutbl: hir::Mutability::Mut }),
        source_info,
    );
    if tcx.sess.opts.debugging_opts.mir_emit_retag {
        // Alias tracking must know we changed the type
        body.basic_blocks_mut()[START_BLOCK].statements.insert(
            0,
            Statement {
                source_info,
                kind: StatementKind::Retag(RetagKind::Raw, box Place::from(SELF_ARG)),
            },
        )
    }

    no_landing_pads(tcx, &mut body);

    // Make sure we remove dead blocks to remove
    // unrelated code from the resume part of the function
    simplify::remove_dead_blocks(&mut body);

    dump_mir(tcx, None, "generator_drop", &0, source, &body, |_, _| Ok(()));

    body
}

fn insert_term_block<'tcx>(body: &mut Body<'tcx>, kind: TerminatorKind<'tcx>) -> BasicBlock {
    let source_info = SourceInfo::outermost(body.span);
    body.basic_blocks_mut().push(BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator { source_info, kind }),
        is_cleanup: false,
    })
}

fn insert_panic_block<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    message: AssertMessage<'tcx>,
) -> BasicBlock {
    let assert_block = BasicBlock::new(body.basic_blocks().len());
    let term = TerminatorKind::Assert {
        cond: Operand::Constant(box Constant {
            span: body.span,
            user_ty: None,
            literal: ty::Const::from_bool(tcx, false),
        }),
        expected: true,
        msg: message,
        target: assert_block,
        cleanup: None,
    };

    let source_info = SourceInfo::outermost(body.span);
    body.basic_blocks_mut().push(BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator { source_info, kind: term }),
        is_cleanup: false,
    });

    assert_block
}

fn can_return<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> bool {
    // Returning from a function with an uninhabited return type is undefined behavior.
    if body.return_ty().conservative_is_privately_uninhabited(tcx) {
        return false;
    }

    // If there's a return terminator the function may return.
    for block in body.basic_blocks() {
        if let TerminatorKind::Return = block.terminator().kind {
            return true;
        }
    }

    // Otherwise the function can't return.
    false
}

fn can_unwind<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> bool {
    // Nothing can unwind when landing pads are off.
    if tcx.sess.panic_strategy() == PanicStrategy::Abort {
        return false;
    }

    // Unwinds can only start at certain terminators.
    for block in body.basic_blocks() {
        match block.terminator().kind {
            // These never unwind.
            TerminatorKind::Goto { .. }
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Abort
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::FalseEdges { .. }
            | TerminatorKind::FalseUnwind { .. } => {}

            // Resume will *continue* unwinding, but if there's no other unwinding terminator it
            // will never be reached.
            TerminatorKind::Resume => {}

            TerminatorKind::Yield { .. } => {
                unreachable!("`can_unwind` called before generator transform")
            }

            // These may unwind.
            TerminatorKind::Drop { .. }
            | TerminatorKind::DropAndReplace { .. }
            | TerminatorKind::Call { .. }
            | TerminatorKind::Assert { .. } => return true,
        }
    }

    // If we didn't find an unwinding terminator, the function cannot unwind.
    false
}

fn create_generator_resume_function<'tcx>(
    tcx: TyCtxt<'tcx>,
    transform: TransformVisitor<'tcx>,
    source: MirSource<'tcx>,
    body: &mut Body<'tcx>,
    can_return: bool,
) {
    let can_unwind = can_unwind(tcx, body);

    // Poison the generator when it unwinds
    if can_unwind {
        let source_info = SourceInfo::outermost(body.span);
        let poison_block = body.basic_blocks_mut().push(BasicBlockData {
            statements: vec![transform.set_discr(VariantIdx::new(POISONED), source_info)],
            terminator: Some(Terminator { source_info, kind: TerminatorKind::Resume }),
            is_cleanup: true,
        });

        for (idx, block) in body.basic_blocks_mut().iter_enumerated_mut() {
            let source_info = block.terminator().source_info;

            if let TerminatorKind::Resume = block.terminator().kind {
                // An existing `Resume` terminator is redirected to jump to our dedicated
                // "poisoning block" above.
                if idx != poison_block {
                    *block.terminator_mut() = Terminator {
                        source_info,
                        kind: TerminatorKind::Goto { target: poison_block },
                    };
                }
            } else if !block.is_cleanup {
                // Any terminators that *can* unwind but don't have an unwind target set are also
                // pointed at our poisoning block (unless they're part of the cleanup path).
                if let Some(unwind @ None) = block.terminator_mut().unwind_mut() {
                    *unwind = Some(poison_block);
                }
            }
        }
    }

    let mut cases = create_cases(body, &transform, Operation::Resume);

    use rustc_middle::mir::AssertKind::{ResumedAfterPanic, ResumedAfterReturn};

    // Jump to the entry point on the unresumed
    cases.insert(0, (UNRESUMED, BasicBlock::new(0)));

    // Panic when resumed on the returned or poisoned state
    let generator_kind = body.generator_kind.unwrap();

    if can_unwind {
        cases.insert(
            1,
            (POISONED, insert_panic_block(tcx, body, ResumedAfterPanic(generator_kind))),
        );
    }

    if can_return {
        cases.insert(
            1,
            (RETURNED, insert_panic_block(tcx, body, ResumedAfterReturn(generator_kind))),
        );
    }

    insert_switch(body, cases, &transform, TerminatorKind::Unreachable);

    make_generator_state_argument_indirect(tcx, body);
    make_generator_state_argument_pinned(tcx, body);

    no_landing_pads(tcx, body);

    // Make sure we remove dead blocks to remove
    // unrelated code from the drop part of the function
    simplify::remove_dead_blocks(body);

    dump_mir(tcx, None, "generator_resume", &0, source, body, |_, _| Ok(()));
}

fn insert_clean_drop(body: &mut Body<'_>) -> BasicBlock {
    let return_block = insert_term_block(body, TerminatorKind::Return);

    let term = TerminatorKind::Drop {
        location: Place::from(SELF_ARG),
        target: return_block,
        unwind: None,
    };
    let source_info = SourceInfo::outermost(body.span);

    // Create a block to destroy an unresumed generators. This can only destroy upvars.
    body.basic_blocks_mut().push(BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator { source_info, kind: term }),
        is_cleanup: false,
    })
}

/// An operation that can be performed on a generator.
#[derive(PartialEq, Copy, Clone)]
enum Operation {
    Resume,
    Drop,
}

impl Operation {
    fn target_block(self, point: &SuspensionPoint<'_>) -> Option<BasicBlock> {
        match self {
            Operation::Resume => Some(point.resume),
            Operation::Drop => point.drop,
        }
    }
}

fn create_cases<'tcx>(
    body: &mut Body<'tcx>,
    transform: &TransformVisitor<'tcx>,
    operation: Operation,
) -> Vec<(usize, BasicBlock)> {
    let source_info = SourceInfo::outermost(body.span);

    transform
        .suspension_points
        .iter()
        .filter_map(|point| {
            // Find the target for this suspension point, if applicable
            operation.target_block(point).map(|target| {
                let mut statements = Vec::new();

                // Create StorageLive instructions for locals with live storage
                for i in 0..(body.local_decls.len()) {
                    if i == 2 {
                        // The resume argument is live on function entry. Don't insert a
                        // `StorageLive`, or the following `Assign` will read from uninitialized
                        // memory.
                        continue;
                    }

                    let l = Local::new(i);
                    let needs_storage_live = point.storage_liveness.contains(l)
                        && !transform.remap.contains_key(&l)
                        && !transform.always_live_locals.contains(l);
                    if needs_storage_live {
                        statements
                            .push(Statement { source_info, kind: StatementKind::StorageLive(l) });
                    }
                }

                if operation == Operation::Resume {
                    // Move the resume argument to the destination place of the `Yield` terminator
                    let resume_arg = Local::new(2); // 0 = return, 1 = self
                    statements.push(Statement {
                        source_info,
                        kind: StatementKind::Assign(box (
                            point.resume_arg,
                            Rvalue::Use(Operand::Move(resume_arg.into())),
                        )),
                    });
                }

                // Then jump to the real target
                let block = body.basic_blocks_mut().push(BasicBlockData {
                    statements,
                    terminator: Some(Terminator {
                        source_info,
                        kind: TerminatorKind::Goto { target },
                    }),
                    is_cleanup: false,
                });

                (point.state, block)
            })
        })
        .collect()
}

impl<'tcx> MirPass<'tcx> for StateTransform {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut Body<'tcx>) {
        let yield_ty = if let Some(yield_ty) = body.yield_ty {
            yield_ty
        } else {
            // This only applies to generators
            return;
        };

        assert!(body.generator_drop.is_none());

        let def_id = source.def_id();

        // The first argument is the generator type passed by value
        let gen_ty = body.local_decls.raw[1].ty;

        // Get the interior types and substs which typeck computed
        let (upvars, interior, discr_ty, movable) = match gen_ty.kind {
            ty::Generator(_, substs, movability) => {
                let substs = substs.as_generator();
                (
                    substs.upvar_tys().collect(),
                    substs.witness(),
                    substs.discr_ty(tcx),
                    movability == hir::Movability::Movable,
                )
            }
            _ => bug!(),
        };

        // Compute GeneratorState<yield_ty, return_ty>
        let state_did = tcx.lang_items().gen_state().unwrap();
        let state_adt_ref = tcx.adt_def(state_did);
        let state_substs = tcx.intern_substs(&[yield_ty.into(), body.return_ty().into()]);
        let ret_ty = tcx.mk_adt(state_adt_ref, state_substs);

        // We rename RETURN_PLACE which has type mir.return_ty to new_ret_local
        // RETURN_PLACE then is a fresh unused local with type ret_ty.
        let new_ret_local = replace_local(RETURN_PLACE, ret_ty, body, tcx);

        // We also replace the resume argument and insert an `Assign`.
        // This is needed because the resume argument `_2` might be live across a `yield`, in which
        // case there is no `Assign` to it that the transform can turn into a store to the generator
        // state. After the yield the slot in the generator state would then be uninitialized.
        let resume_local = Local::new(2);
        let new_resume_local =
            replace_local(resume_local, body.local_decls[resume_local].ty, body, tcx);

        // When first entering the generator, move the resume argument into its new local.
        let source_info = SourceInfo::outermost(body.span);
        let stmts = &mut body.basic_blocks_mut()[BasicBlock::new(0)].statements;
        stmts.insert(
            0,
            Statement {
                source_info,
                kind: StatementKind::Assign(box (
                    new_resume_local.into(),
                    Rvalue::Use(Operand::Move(resume_local.into())),
                )),
            },
        );

        let always_live_locals = storage::AlwaysLiveLocals::new(&body);

        // Extract locals which are live across suspension point into `layout`
        // `remap` gives a mapping from local indices onto generator struct indices
        // `storage_liveness` tells us which locals have live storage at suspension points
        let (remap, layout, storage_liveness) =
            compute_layout(tcx, source, &upvars, interior, &always_live_locals, movable, body);

        let can_return = can_return(tcx, body);

        // Run the transformation which converts Places from Local to generator struct
        // accesses for locals in `remap`.
        // It also rewrites `return x` and `yield y` as writing a new generator state and returning
        // GeneratorState::Complete(x) and GeneratorState::Yielded(y) respectively.
        let mut transform = TransformVisitor {
            tcx,
            state_adt_ref,
            state_substs,
            remap,
            storage_liveness,
            always_live_locals,
            suspension_points: Vec::new(),
            new_ret_local,
            discr_ty,
        };
        transform.visit_body(body);

        // Update our MIR struct to reflect the changes we've made
        body.yield_ty = None;
        body.arg_count = 2; // self, resume arg
        body.spread_arg = None;
        body.generator_layout = Some(layout);

        // Insert `drop(generator_struct)` which is used to drop upvars for generators in
        // the unresumed state.
        // This is expanded to a drop ladder in `elaborate_generator_drops`.
        let drop_clean = insert_clean_drop(body);

        dump_mir(tcx, None, "generator_pre-elab", &0, source, body, |_, _| Ok(()));

        // Expand `drop(generator_struct)` to a drop ladder which destroys upvars.
        // If any upvars are moved out of, drop elaboration will handle upvar destruction.
        // However we need to also elaborate the code generated by `insert_clean_drop`.
        elaborate_generator_drops(tcx, def_id, body);

        dump_mir(tcx, None, "generator_post-transform", &0, source, body, |_, _| Ok(()));

        // Create a copy of our MIR and use it to create the drop shim for the generator
        let drop_shim =
            create_generator_drop_shim(tcx, &transform, source, gen_ty, body, drop_clean);

        body.generator_drop = Some(box drop_shim);

        // Create the Generator::resume function
        create_generator_resume_function(tcx, transform, source, body, can_return);
    }
}
