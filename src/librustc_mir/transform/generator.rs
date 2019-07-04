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

use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::mir::*;
use rustc::mir::visit::{PlaceContext, Visitor, MutVisitor};
use rustc::ty::{self, TyCtxt, AdtDef, Ty};
use rustc::ty::GeneratorSubsts;
use rustc::ty::layout::VariantIdx;
use rustc::ty::subst::SubstsRef;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc_data_structures::bit_set::{BitSet, BitMatrix};
use std::borrow::Cow;
use std::iter;
use std::mem;
use crate::transform::{MirPass, MirSource};
use crate::transform::simplify;
use crate::transform::no_landing_pads::no_landing_pads;
use crate::dataflow::{DataflowResults, DataflowResultsConsumer, FlowAtLocation};
use crate::dataflow::{do_dataflow, DebugFormatted, state_for_location};
use crate::dataflow::{MaybeStorageLive, HaveBeenBorrowedLocals, RequiresStorage};
use crate::util::dump_mir;
use crate::util::liveness;

pub struct StateTransform;

struct RenameLocalVisitor {
    from: Local,
    to: Local,
}

impl<'tcx> MutVisitor<'tcx> for RenameLocalVisitor {
    fn visit_local(&mut self,
                   local: &mut Local,
                   _: PlaceContext,
                   _: Location) {
        if *local == self.from {
            *local = self.to;
        }
    }
}

struct DerefArgVisitor;

impl<'tcx> MutVisitor<'tcx> for DerefArgVisitor {
    fn visit_local(&mut self,
                   local: &mut Local,
                   _: PlaceContext,
                   _: Location) {
        assert_ne!(*local, self_arg());
    }

    fn visit_place(&mut self,
                    place: &mut Place<'tcx>,
                    context: PlaceContext,
                    location: Location) {
        if place.base_local() == Some(self_arg()) {
            replace_base(place, Place::Projection(Box::new(Projection {
                base: Place::Base(PlaceBase::Local(self_arg())),
                elem: ProjectionElem::Deref,
            })));
        } else {
            self.super_place(place, context, location);
        }
    }
}

struct PinArgVisitor<'tcx> {
    ref_gen_ty: Ty<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for PinArgVisitor<'tcx> {
    fn visit_local(&mut self,
                   local: &mut Local,
                   _: PlaceContext,
                   _: Location) {
        assert_ne!(*local, self_arg());
    }

    fn visit_place(&mut self,
                    place: &mut Place<'tcx>,
                    context: PlaceContext,
                    location: Location) {
        if place.base_local() == Some(self_arg()) {
            replace_base(place, Place::Projection(Box::new(Projection {
                base: Place::Base(PlaceBase::Local(self_arg())),
                elem: ProjectionElem::Field(Field::new(0), self.ref_gen_ty),
            })));
        } else {
            self.super_place(place, context, location);
        }
    }
}

fn replace_base(place: &mut Place<'tcx>, new_base: Place<'tcx>) {
    if let Place::Projection(proj) = place {
        replace_base(&mut proj.base, new_base);
    } else {
        *place = new_base;
    }
}

fn self_arg() -> Local {
    Local::new(1)
}

/// Generator have not been resumed yet
const UNRESUMED: usize = GeneratorSubsts::UNRESUMED;
/// Generator has returned / is completed
const RETURNED: usize = GeneratorSubsts::RETURNED;
/// Generator has been poisoned
const POISONED: usize = GeneratorSubsts::POISONED;

struct SuspensionPoint {
    state: usize,
    resume: BasicBlock,
    drop: Option<BasicBlock>,
    storage_liveness: liveness::LiveVarSet,
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
    // FIXME(eddyb) This should use `IndexVec<BasicBlock, Option<_>>`.
    storage_liveness: FxHashMap<BasicBlock, liveness::LiveVarSet>,

    // A list of suspension points, generated during the transform
    suspension_points: Vec<SuspensionPoint>,

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
        let self_place = Place::from(self_arg());
        let base = self_place.downcast_unnamed(variant_index);
        let field = Projection {
            base: base,
            elem: ProjectionElem::Field(Field::new(idx), ty),
        };
        Place::Projection(Box::new(field))
    }

    // Create a statement which changes the discriminant
    fn set_discr(&self, state_disc: VariantIdx, source_info: SourceInfo) -> Statement<'tcx> {
        let self_place = Place::from(self_arg());
        Statement {
            source_info,
            kind: StatementKind::SetDiscriminant { place: self_place, variant_index: state_disc },
        }
    }

    // Create a statement which reads the discriminant into a temporary
    fn get_discr(&self, body: &mut Body<'tcx>) -> (Statement<'tcx>, Place<'tcx>) {
        let temp_decl = LocalDecl::new_internal(self.tcx.types.isize, body.span);
        let local_decls_len = body.local_decls.push(temp_decl);
        let temp = Place::from(local_decls_len);

        let self_place = Place::from(self_arg());
        let assign = Statement {
            source_info: source_info(body),
            kind: StatementKind::Assign(temp.clone(), box Rvalue::Discriminant(self_place)),
        };
        (assign, temp)
    }
}

impl MutVisitor<'tcx> for TransformVisitor<'tcx> {
    fn visit_local(&mut self,
                   local: &mut Local,
                   _: PlaceContext,
                   _: Location) {
        assert_eq!(self.remap.get(local), None);
    }

    fn visit_place(&mut self,
                    place: &mut Place<'tcx>,
                    context: PlaceContext,
                    location: Location) {
        if let Some(l) = place.base_local() {
            // Replace an Local in the remap with a generator struct access
            if let Some(&(ty, variant_index, idx)) = self.remap.get(&l) {
                replace_base(place, self.make_field(variant_index, idx, ty));
            }
        } else {
            self.super_place(place, context, location);
        }
    }

    fn visit_basic_block_data(&mut self,
                              block: BasicBlock,
                              data: &mut BasicBlockData<'tcx>) {
        // Remove StorageLive and StorageDead statements for remapped locals
        data.retain_statements(|s| {
            match s.kind {
                StatementKind::StorageLive(l) | StatementKind::StorageDead(l) => {
                    !self.remap.contains_key(&l)
                }
                _ => true
            }
        });

        let ret_val = match data.terminator().kind {
            TerminatorKind::Return => Some((VariantIdx::new(1),
                None,
                Operand::Move(Place::from(self.new_ret_local)),
                None)),
            TerminatorKind::Yield { ref value, resume, drop } => Some((VariantIdx::new(0),
                Some(resume),
                value.clone(),
                drop)),
            _ => None
        };

        if let Some((state_idx, resume, v, drop)) = ret_val {
            let source_info = data.terminator().source_info;
            // We must assign the value first in case it gets declared dead below
            data.statements.push(Statement {
                source_info,
                kind: StatementKind::Assign(Place::RETURN_PLACE,
                                            box self.make_state(state_idx, v)),
            });
            let state = if let Some(resume) = resume { // Yield
                let state = 3 + self.suspension_points.len();

                self.suspension_points.push(SuspensionPoint {
                    state,
                    resume,
                    drop,
                    storage_liveness: self.storage_liveness.get(&block).unwrap().clone(),
                });

                VariantIdx::new(state)
            } else { // Return
                VariantIdx::new(RETURNED) // state for returned
            };
            data.statements.push(self.set_discr(state, source_info));
            data.terminator.as_mut().unwrap().kind = TerminatorKind::Return;
        }

        self.super_basic_block_data(block, data);
    }
}

fn make_generator_state_argument_indirect<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    body: &mut Body<'tcx>,
) {
    let gen_ty = body.local_decls.raw[1].ty;

    let region = ty::ReFree(ty::FreeRegion {
        scope: def_id,
        bound_region: ty::BoundRegion::BrEnv,
    });

    let region = tcx.mk_region(region);

    let ref_gen_ty = tcx.mk_ref(region, ty::TypeAndMut {
        ty: gen_ty,
        mutbl: hir::MutMutable
    });

    // Replace the by value generator argument
    body.local_decls.raw[1].ty = ref_gen_ty;

    // Add a deref to accesses of the generator state
    DerefArgVisitor.visit_body(body);
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
    PinArgVisitor { ref_gen_ty }.visit_body(body);
}

fn replace_result_variable<'tcx>(
    ret_ty: Ty<'tcx>,
    body: &mut Body<'tcx>,
) -> Local {
    let source_info = source_info(body);
    let new_ret = LocalDecl {
        mutability: Mutability::Mut,
        ty: ret_ty,
        user_ty: UserTypeProjections::none(),
        name: None,
        source_info,
        visibility_scope: source_info.scope,
        internal: false,
        is_block_tail: None,
        is_user_variable: None,
    };
    let new_ret_local = Local::new(body.local_decls.len());
    body.local_decls.push(new_ret);
    body.local_decls.swap(RETURN_PLACE, new_ret_local);

    RenameLocalVisitor {
        from: RETURN_PLACE,
        to: new_ret_local,
    }.visit_body(body);

    new_ret_local
}

struct StorageIgnored(liveness::LiveVarSet);

impl<'tcx> Visitor<'tcx> for StorageIgnored {
    fn visit_statement(&mut self,
                       statement: &Statement<'tcx>,
                       _location: Location) {
        match statement.kind {
            StatementKind::StorageLive(l) |
            StatementKind::StorageDead(l) => { self.0.remove(l); }
            _ => (),
        }
    }
}

struct LivenessInfo {
    /// Which locals are live across any suspension point.
    ///
    /// GeneratorSavedLocal is indexed in terms of the elements in this set;
    /// i.e. GeneratorSavedLocal::new(1) corresponds to the second local
    /// included in this set.
    live_locals: liveness::LiveVarSet,

    /// The set of saved locals live at each suspension point.
    live_locals_at_suspension_points: Vec<BitSet<GeneratorSavedLocal>>,

    /// For every saved local, the set of other saved locals that are
    /// storage-live at the same time as this local. We cannot overlap locals in
    /// the layout which have conflicting storage.
    storage_conflicts: BitMatrix<GeneratorSavedLocal, GeneratorSavedLocal>,

    /// For every suspending block, the locals which are storage-live across
    /// that suspension point.
    storage_liveness: FxHashMap<BasicBlock, liveness::LiveVarSet>,
}

fn locals_live_across_suspend_points(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    source: MirSource<'tcx>,
    movable: bool,
) -> LivenessInfo {
    let dead_unwinds = BitSet::new_empty(body.basic_blocks().len());
    let def_id = source.def_id();

    // Calculate when MIR locals have live storage. This gives us an upper bound of their
    // lifetimes.
    let storage_live_analysis = MaybeStorageLive::new(body);
    let storage_live =
        do_dataflow(tcx, body, def_id, &[], &dead_unwinds, storage_live_analysis,
                    |bd, p| DebugFormatted::new(&bd.body().local_decls[p]));

    // Find the MIR locals which do not use StorageLive/StorageDead statements.
    // The storage of these locals are always live.
    let mut ignored = StorageIgnored(BitSet::new_filled(body.local_decls.len()));
    ignored.visit_body(body);

    // Calculate the MIR locals which have been previously
    // borrowed (even if they are still active).
    let borrowed_locals_analysis = HaveBeenBorrowedLocals::new(body);
    let borrowed_locals_result =
        do_dataflow(tcx, body, def_id, &[], &dead_unwinds, borrowed_locals_analysis,
                    |bd, p| DebugFormatted::new(&bd.body().local_decls[p]));

    // Calculate the MIR locals that we actually need to keep storage around
    // for.
    let requires_storage_analysis = RequiresStorage::new(body, &borrowed_locals_result);
    let requires_storage =
        do_dataflow(tcx, body, def_id, &[], &dead_unwinds, requires_storage_analysis,
                    |bd, p| DebugFormatted::new(&bd.body().local_decls[p]));
    let requires_storage_analysis = RequiresStorage::new(body, &borrowed_locals_result);

    // Calculate the liveness of MIR locals ignoring borrows.
    let mut live_locals = liveness::LiveVarSet::new_empty(body.local_decls.len());
    let mut liveness = liveness::liveness_of_locals(
        body,
    );
    liveness::dump_mir(
        tcx,
        "generator_liveness",
        source,
        body,
        &liveness,
    );

    let mut storage_liveness_map = FxHashMap::default();
    let mut live_locals_at_suspension_points = Vec::new();

    for (block, data) in body.basic_blocks().iter_enumerated() {
        if let TerminatorKind::Yield { .. } = data.terminator().kind {
            let loc = Location {
                block: block,
                statement_index: data.statements.len(),
            };

            if !movable {
                let borrowed_locals = state_for_location(loc,
                                                         &borrowed_locals_analysis,
                                                         &borrowed_locals_result,
                                                         body);
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
                liveness.outs[block].union(&borrowed_locals);
            }

            let storage_liveness = state_for_location(loc,
                                                      &storage_live_analysis,
                                                      &storage_live,
                                                      body);

            // Store the storage liveness for later use so we can restore the state
            // after a suspension point
            storage_liveness_map.insert(block, storage_liveness.clone());

            let mut storage_required = state_for_location(loc,
                                                          &requires_storage_analysis,
                                                          &requires_storage,
                                                          body);

            // Mark locals without storage statements as always requiring storage
            storage_required.union(&ignored.0);

            // Locals live are live at this point only if they are used across
            // suspension points (the `liveness` variable)
            // and their storage is required (the `storage_required` variable)
            let mut live_locals_here = storage_required;
            live_locals_here.intersect(&liveness.outs[block]);

            // The generator argument is ignored
            live_locals_here.remove(self_arg());

            debug!("loc = {:?}, live_locals_here = {:?}", loc, live_locals_here);

            // Add the locals live at this suspension point to the set of locals which live across
            // any suspension points
            live_locals.union(&live_locals_here);

            live_locals_at_suspension_points.push(live_locals_here);
        }
    }
    debug!("live_locals = {:?}", live_locals);

    // Renumber our liveness_map bitsets to include only the locals we are
    // saving.
    let live_locals_at_suspension_points = live_locals_at_suspension_points
        .iter()
        .map(|live_here| renumber_bitset(&live_here, &live_locals))
        .collect();

    let storage_conflicts = compute_storage_conflicts(
        body,
        &live_locals,
        &ignored,
        requires_storage,
        requires_storage_analysis);

    LivenessInfo {
        live_locals,
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
fn renumber_bitset(input: &BitSet<Local>, stored_locals: &liveness::LiveVarSet)
-> BitSet<GeneratorSavedLocal> {
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
    stored_locals: &liveness::LiveVarSet,
    ignored: &StorageIgnored,
    requires_storage: DataflowResults<'tcx, RequiresStorage<'mir, 'tcx>>,
    _requires_storage_analysis: RequiresStorage<'mir, 'tcx>,
) -> BitMatrix<GeneratorSavedLocal, GeneratorSavedLocal> {
    assert_eq!(body.local_decls.len(), ignored.0.domain_size());
    assert_eq!(body.local_decls.len(), stored_locals.domain_size());
    debug!("compute_storage_conflicts({:?})", body.span);
    debug!("ignored = {:?}", ignored.0);

    // Storage ignored locals are not eligible for overlap, since their storage
    // is always live.
    let mut ineligible_locals = ignored.0.clone();
    ineligible_locals.intersect(&stored_locals);

    // Compute the storage conflicts for all eligible locals.
    let mut visitor = StorageConflictVisitor {
        body,
        stored_locals: &stored_locals,
        local_conflicts: BitMatrix::from_row_n(&ineligible_locals, body.local_decls.len()),
    };
    let mut state = FlowAtLocation::new(requires_storage);
    visitor.analyze_results(&mut state);
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

struct StorageConflictVisitor<'body, 'tcx, 's> {
    body: &'body Body<'tcx>,
    stored_locals: &'s liveness::LiveVarSet,
    // FIXME(tmandry): Consider using sparse bitsets here once we have good
    // benchmarks for generators.
    local_conflicts: BitMatrix<Local, Local>,
}

impl<'body, 'tcx, 's> DataflowResultsConsumer<'body, 'tcx>
    for StorageConflictVisitor<'body, 'tcx, 's>
{
    type FlowState = FlowAtLocation<'tcx, RequiresStorage<'body, 'tcx>>;

    fn body(&self) -> &'body Body<'tcx> {
        self.body
    }

    fn visit_block_entry(&mut self,
                         block: BasicBlock,
                         flow_state: &Self::FlowState) {
        // statement_index is only used for logging, so this is fine.
        self.apply_state(flow_state, Location { block, statement_index: 0 });
    }

    fn visit_statement_entry(&mut self,
                             loc: Location,
                             _stmt: &Statement<'tcx>,
                             flow_state: &Self::FlowState) {
        self.apply_state(flow_state, loc);
    }

    fn visit_terminator_entry(&mut self,
                              loc: Location,
                              _term: &Terminator<'tcx>,
                              flow_state: &Self::FlowState) {
        self.apply_state(flow_state, loc);
    }
}

impl<'body, 'tcx, 's> StorageConflictVisitor<'body, 'tcx, 's> {
    fn apply_state(&mut self,
                   flow_state: &FlowAtLocation<'tcx, RequiresStorage<'body, 'tcx>>,
                   loc: Location) {
        // Ignore unreachable blocks.
        match self.body.basic_blocks()[loc.block].terminator().kind {
            TerminatorKind::Unreachable => return,
            _ => (),
        };

        let mut eligible_storage_live = flow_state.as_dense().clone();
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
    movable: bool,
    body: &mut Body<'tcx>,
) -> (
    FxHashMap<Local, (Ty<'tcx>, VariantIdx, usize)>,
    GeneratorLayout<'tcx>,
    FxHashMap<BasicBlock, liveness::LiveVarSet>,
) {
    // Use a liveness analysis to compute locals which are live across a suspension point
    let LivenessInfo {
        live_locals, live_locals_at_suspension_points, storage_conflicts, storage_liveness
    } = locals_live_across_suspend_points(tcx, body, source, movable);

    // Erase regions from the types passed in from typeck so we can compare them with
    // MIR types
    let allowed_upvars = tcx.erase_regions(upvars);
    let allowed = match interior.sty {
        ty::GeneratorWitness(s) => tcx.erase_late_bound_regions(&s),
        _ => bug!(),
    };

    for (local, decl) in body.local_decls.iter_enumerated() {
        // Ignore locals which are internal or not live
        if !live_locals.contains(local) || decl.internal {
            continue;
        }

        // Sanity check that typeck knows about the type of locals which are
        // live across a suspension point
        if !allowed.contains(&decl.ty) && !allowed_upvars.contains(&decl.ty) {
            span_bug!(body.span,
                      "Broken MIR: generator contains type {} in MIR, \
                       but typeck only knows about {}",
                      decl.ty,
                      interior);
        }
    }

    let dummy_local = LocalDecl::new_internal(tcx.mk_unit(), body.span);

    // Gather live locals and their indices replacing values in body.local_decls
    // with a dummy to avoid changing local indices.
    let mut locals = IndexVec::<GeneratorSavedLocal, _>::new();
    let mut tys = IndexVec::<GeneratorSavedLocal, _>::new();
    let mut decls = IndexVec::<GeneratorSavedLocal, _>::new();
    for (idx, local) in live_locals.iter().enumerate() {
        let var = mem::replace(&mut body.local_decls[local], dummy_local.clone());
        locals.push(local);
        tys.push(var.ty);
        decls.push(var);
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

    let layout = GeneratorLayout {
        field_tys: tys,
        variant_fields,
        storage_conflicts,
        __local_debuginfo_codegen_only_do_not_use: decls,
    };

    (remap, layout, storage_liveness)
}

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

    let source_info = source_info(body);
    body.basic_blocks_mut().raw.insert(0, BasicBlockData {
        statements: vec![assign],
        terminator: Some(Terminator {
            source_info,
            kind: switch,
        }),
        is_cleanup: false,
    });

    let blocks = body.basic_blocks_mut().iter_mut();

    for target in blocks.flat_map(|b| b.terminator_mut().successors_mut()) {
        *target = BasicBlock::new(target.index() + 1);
    }
}

fn elaborate_generator_drops<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, body: &mut Body<'tcx>) {
    use crate::util::elaborate_drops::{elaborate_drop, Unwind};
    use crate::util::patch::MirPatch;
    use crate::shim::DropShimElaborator;

    // Note that `elaborate_drops` only drops the upvars of a generator, and
    // this is ok because `open_drop` can only be reached within that own
    // generator's resume function.

    let param_env = tcx.param_env(def_id);
    let gen = self_arg();

    let mut elaborator = DropShimElaborator {
        body: body,
        patch: MirPatch::new(body),
        tcx,
        param_env
    };

    for (block, block_data) in body.basic_blocks().iter_enumerated() {
        let (target, unwind, source_info) = match block_data.terminator() {
            &Terminator {
                source_info,
                kind: TerminatorKind::Drop {
                    location: Place::Base(PlaceBase::Local(local)),
                    target,
                    unwind
                }
            } if local == gen => (target, unwind, source_info),
            _ => continue,
        };
        let unwind = if block_data.is_cleanup {
            Unwind::InCleanup
        } else {
            Unwind::To(unwind.unwrap_or_else(|| elaborator.patch.resume_block()))
        };
        elaborate_drop(
            &mut elaborator,
            source_info,
            &Place::from(gen),
            (),
            target,
            unwind,
            block,
        );
    }
    elaborator.patch.apply(body);
}

fn create_generator_drop_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    transform: &TransformVisitor<'tcx>,
    def_id: DefId,
    source: MirSource<'tcx>,
    gen_ty: Ty<'tcx>,
    body: &Body<'tcx>,
    drop_clean: BasicBlock,
) -> Body<'tcx> {
    let mut body = body.clone();

    let source_info = source_info(&body);

    let mut cases = create_cases(&mut body, transform, |point| point.drop);

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
    body.local_decls[RETURN_PLACE] = LocalDecl {
        mutability: Mutability::Mut,
        ty: tcx.mk_unit(),
        user_ty: UserTypeProjections::none(),
        name: None,
        source_info,
        visibility_scope: source_info.scope,
        internal: false,
        is_block_tail: None,
        is_user_variable: None,
    };

    make_generator_state_argument_indirect(tcx, def_id, &mut body);

    // Change the generator argument from &mut to *mut
    body.local_decls[self_arg()] = LocalDecl {
        mutability: Mutability::Mut,
        ty: tcx.mk_ptr(ty::TypeAndMut {
            ty: gen_ty,
            mutbl: hir::Mutability::MutMutable,
        }),
        user_ty: UserTypeProjections::none(),
        name: None,
        source_info,
        visibility_scope: source_info.scope,
        internal: false,
        is_block_tail: None,
        is_user_variable: None,
    };
    if tcx.sess.opts.debugging_opts.mir_emit_retag {
        // Alias tracking must know we changed the type
        body.basic_blocks_mut()[START_BLOCK].statements.insert(0, Statement {
            source_info,
            kind: StatementKind::Retag(RetagKind::Raw, Place::from(self_arg())),
        })
    }

    no_landing_pads(tcx, &mut body);

    // Make sure we remove dead blocks to remove
    // unrelated code from the resume part of the function
    simplify::remove_dead_blocks(&mut body);

    dump_mir(tcx, None, "generator_drop", &0, source, &mut body, |_, _| Ok(()) );

    body
}

fn insert_term_block<'tcx>(body: &mut Body<'tcx>, kind: TerminatorKind<'tcx>) -> BasicBlock {
    let term_block = BasicBlock::new(body.basic_blocks().len());
    let source_info = source_info(body);
    body.basic_blocks_mut().push(BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator {
            source_info,
            kind,
        }),
        is_cleanup: false,
    });
    term_block
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
            ty: tcx.types.bool,
            user_ty: None,
            literal: ty::Const::from_bool(tcx, false),
        }),
        expected: true,
        msg: message,
        target: assert_block,
        cleanup: None,
    };

    let source_info = source_info(body);
    body.basic_blocks_mut().push(BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator {
            source_info,
            kind: term,
        }),
        is_cleanup: false,
    });

    assert_block
}

fn create_generator_resume_function<'tcx>(
    tcx: TyCtxt<'tcx>,
    transform: TransformVisitor<'tcx>,
    def_id: DefId,
    source: MirSource<'tcx>,
    body: &mut Body<'tcx>,
) {
    // Poison the generator when it unwinds
    for block in body.basic_blocks_mut() {
        let source_info = block.terminator().source_info;
        if let &TerminatorKind::Resume = &block.terminator().kind {
            block.statements.push(
                transform.set_discr(VariantIdx::new(POISONED), source_info));
        }
    }

    let mut cases = create_cases(body, &transform, |point| Some(point.resume));

    use rustc::mir::interpret::InterpError::{
        GeneratorResumedAfterPanic,
        GeneratorResumedAfterReturn,
    };

    // Jump to the entry point on the unresumed
    cases.insert(0, (UNRESUMED, BasicBlock::new(0)));
    // Panic when resumed on the returned state
    cases.insert(1, (RETURNED, insert_panic_block(tcx, body, GeneratorResumedAfterReturn)));
    // Panic when resumed on the poisoned state
    cases.insert(2, (POISONED, insert_panic_block(tcx, body, GeneratorResumedAfterPanic)));

    insert_switch(body, cases, &transform, TerminatorKind::Unreachable);

    make_generator_state_argument_indirect(tcx, def_id, body);
    make_generator_state_argument_pinned(tcx, body);

    no_landing_pads(tcx, body);

    // Make sure we remove dead blocks to remove
    // unrelated code from the drop part of the function
    simplify::remove_dead_blocks(body);

    dump_mir(tcx, None, "generator_resume", &0, source, body, |_, _| Ok(()) );
}

fn source_info(body: &Body<'_>) -> SourceInfo {
    SourceInfo {
        span: body.span,
        scope: OUTERMOST_SOURCE_SCOPE,
    }
}

fn insert_clean_drop(body: &mut Body<'_>) -> BasicBlock {
    let return_block = insert_term_block(body, TerminatorKind::Return);

    // Create a block to destroy an unresumed generators. This can only destroy upvars.
    let drop_clean = BasicBlock::new(body.basic_blocks().len());
    let term = TerminatorKind::Drop {
        location: Place::from(self_arg()),
        target: return_block,
        unwind: None,
    };
    let source_info = source_info(body);
    body.basic_blocks_mut().push(BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator {
            source_info,
            kind: term,
        }),
        is_cleanup: false,
    });

    drop_clean
}

fn create_cases<'tcx, F>(
    body: &mut Body<'tcx>,
    transform: &TransformVisitor<'tcx>,
    target: F,
) -> Vec<(usize, BasicBlock)>
where
    F: Fn(&SuspensionPoint) -> Option<BasicBlock>,
{
    let source_info = source_info(body);

    transform.suspension_points.iter().filter_map(|point| {
        // Find the target for this suspension point, if applicable
        target(point).map(|target| {
            let block = BasicBlock::new(body.basic_blocks().len());
            let mut statements = Vec::new();

            // Create StorageLive instructions for locals with live storage
            for i in 0..(body.local_decls.len()) {
                let l = Local::new(i);
                if point.storage_liveness.contains(l) && !transform.remap.contains_key(&l) {
                    statements.push(Statement {
                        source_info,
                        kind: StatementKind::StorageLive(l),
                    });
                }
            }

            // Then jump to the real target
            body.basic_blocks_mut().push(BasicBlockData {
                statements,
                terminator: Some(Terminator {
                    source_info,
                    kind: TerminatorKind::Goto {
                        target,
                    },
                }),
                is_cleanup: false,
            });

            (point.state, block)
        })
    }).collect()
}

impl MirPass for StateTransform {
    fn run_pass<'tcx>(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut Body<'tcx>) {
        let yield_ty = if let Some(yield_ty) = body.yield_ty {
            yield_ty
        } else {
            // This only applies to generators
            return
        };

        assert!(body.generator_drop.is_none());

        let def_id = source.def_id();

        // The first argument is the generator type passed by value
        let gen_ty = body.local_decls.raw[1].ty;

        // Get the interior types and substs which typeck computed
        let (upvars, interior, discr_ty, movable) = match gen_ty.sty {
            ty::Generator(_, substs, movability) => {
                (substs.upvar_tys(def_id, tcx).collect(),
                 substs.witness(def_id, tcx),
                 substs.discr_ty(tcx),
                 movability == hir::GeneratorMovability::Movable)
            }
            _ => bug!(),
        };

        // Compute GeneratorState<yield_ty, return_ty>
        let state_did = tcx.lang_items().gen_state().unwrap();
        let state_adt_ref = tcx.adt_def(state_did);
        let state_substs = tcx.intern_substs(&[
            yield_ty.into(),
            body.return_ty().into(),
        ]);
        let ret_ty = tcx.mk_adt(state_adt_ref, state_substs);

        // We rename RETURN_PLACE which has type mir.return_ty to new_ret_local
        // RETURN_PLACE then is a fresh unused local with type ret_ty.
        let new_ret_local = replace_result_variable(ret_ty, body);

        // Extract locals which are live across suspension point into `layout`
        // `remap` gives a mapping from local indices onto generator struct indices
        // `storage_liveness` tells us which locals have live storage at suspension points
        let (remap, layout, storage_liveness) = compute_layout(
            tcx,
            source,
            &upvars,
            interior,
            movable,
            body);

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
            suspension_points: Vec::new(),
            new_ret_local,
            discr_ty,
        };
        transform.visit_body(body);

        // Update our MIR struct to reflect the changed we've made
        body.yield_ty = None;
        body.arg_count = 1;
        body.spread_arg = None;
        body.generator_layout = Some(layout);

        // Insert `drop(generator_struct)` which is used to drop upvars for generators in
        // the unresumed state.
        // This is expanded to a drop ladder in `elaborate_generator_drops`.
        let drop_clean = insert_clean_drop(body);

        dump_mir(tcx, None, "generator_pre-elab", &0, source, body, |_, _| Ok(()) );

        // Expand `drop(generator_struct)` to a drop ladder which destroys upvars.
        // If any upvars are moved out of, drop elaboration will handle upvar destruction.
        // However we need to also elaborate the code generated by `insert_clean_drop`.
        elaborate_generator_drops(tcx, def_id, body);

        dump_mir(tcx, None, "generator_post-transform", &0, source, body, |_, _| Ok(()) );

        // Create a copy of our MIR and use it to create the drop shim for the generator
        let drop_shim = create_generator_drop_shim(tcx,
            &transform,
            def_id,
            source,
            gen_ty,
            &body,
            drop_clean);

        body.generator_drop = Some(box drop_shim);

        // Create the Generator::resume function
        create_generator_resume_function(tcx, transform, def_id, source, body);
    }
}
