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
//! This pass creates the implementation for either the `Generator::resume` or `Future::poll`
//! function and the drop shim for the generator based on the MIR input.
//! It converts the generator argument from Self to &mut Self adding derefs in the MIR as needed.
//! It computes the final layout of the generator struct which looks like this:
//!     First upvars are stored
//!     It is followed by the generator state field.
//!     Then finally the MIR locals which are live across a suspension point are stored.
//!     ```ignore (illustrative)
//!     struct Generator {
//!         upvars...,
//!         state: u32,
//!         mir_locals...,
//!     }
//!     ```
//! This pass computes the meaning of the state field and the MIR locals which are live
//! across a suspension point. There are however three hardcoded generator states:
//!     0 - Generator have not been resumed yet
//!     1 - Generator has returned / is completed
//!     2 - Generator has been poisoned
//!
//! It also rewrites `return x` and `yield y` as setting a new generator state and returning
//! `GeneratorState::Complete(x)` and `GeneratorState::Yielded(y)`,
//! or `Poll::Ready(x)` and `Poll::Pending` respectively.
//! MIR locals which are live across a suspension point are moved to the generator struct
//! with references to them being updated with references to the generator struct.
//!
//! The pass creates two functions which have a switch on the generator state giving
//! the action to take.
//!
//! One of them is the implementation of `Generator::resume` / `Future::poll`.
//! For generators with state 0 (unresumed) it starts the execution of the generator.
//! For generators with state 1 (returned) and state 2 (poisoned) it panics.
//! Otherwise it continues the execution from the last suspension point.
//!
//! The other function is the drop glue for the generator.
//! For generators with state 0 (unresumed) it drops the upvars of the generator.
//! For generators with state 1 (returned) and state 2 (poisoned) it does nothing.
//! Otherwise it drops all the values in scope at the last suspension point.

use crate::deref_separator::deref_finder;
use crate::errors;
use crate::simplify;
use crate::MirPass;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::pluralize;
use rustc_hir as hir;
use rustc_hir::lang_items::LangItem;
use rustc_hir::GeneratorKind;
use rustc_index::bit_set::{BitMatrix, BitSet, GrowableBitSet};
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::dump_mir;
use rustc_middle::mir::visit::{MutVisitor, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, AdtDef, Ty, TyCtxt};
use rustc_middle::ty::{GeneratorSubsts, SubstsRef};
use rustc_mir_dataflow::impls::{
    MaybeBorrowedLocals, MaybeLiveLocals, MaybeRequiresStorage, MaybeStorageLive,
};
use rustc_mir_dataflow::storage::always_storage_live_locals;
use rustc_mir_dataflow::{self, Analysis};
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::symbol::sym;
use rustc_span::Span;
use rustc_target::abi::{FieldIdx, VariantIdx};
use rustc_target::spec::PanicStrategy;
use std::{iter, ops};

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

    fn visit_terminator(&mut self, terminator: &mut Terminator<'tcx>, location: Location) {
        match terminator.kind {
            TerminatorKind::Return => {
                // Do not replace the implicit `_0` access here, as that's not possible. The
                // transform already handles `return` correctly.
            }
            _ => self.super_terminator(terminator, location),
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
                    projection: self.tcx().mk_place_elems(&[ProjectionElem::Deref]),
                },
                self.tcx,
            );
        } else {
            self.visit_local(&mut place.local, context, location);

            for elem in place.projection.iter() {
                if let PlaceElem::Index(local) = elem {
                    assert_ne!(local, SELF_ARG);
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
                    projection: self.tcx().mk_place_elems(&[ProjectionElem::Field(
                        FieldIdx::new(0),
                        self.ref_gen_ty,
                    )]),
                },
                self.tcx,
            );
        } else {
            self.visit_local(&mut place.local, context, location);

            for elem in place.projection.iter() {
                if let PlaceElem::Index(local) = elem {
                    assert_ne!(local, SELF_ARG);
                }
            }
        }
    }
}

fn replace_base<'tcx>(place: &mut Place<'tcx>, new_base: Place<'tcx>, tcx: TyCtxt<'tcx>) {
    place.local = new_base.local;

    let mut new_projection = new_base.projection.to_vec();
    new_projection.append(&mut place.projection.to_vec());

    place.projection = tcx.mk_place_elems(&new_projection);
}

const SELF_ARG: Local = Local::from_u32(1);

/// Generator has not been resumed yet.
const UNRESUMED: usize = GeneratorSubsts::UNRESUMED;
/// Generator has returned / is completed.
const RETURNED: usize = GeneratorSubsts::RETURNED;
/// Generator has panicked and is poisoned.
const POISONED: usize = GeneratorSubsts::POISONED;

/// Number of variants to reserve in generator state. Corresponds to
/// `UNRESUMED` (beginning of a generator) and `RETURNED`/`POISONED`
/// (end of a generator) states.
const RESERVED_VARIANTS: usize = 3;

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
    storage_liveness: GrowableBitSet<Local>,
}

struct TransformVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    is_async_kind: bool,
    state_adt_ref: AdtDef<'tcx>,
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
    always_live_locals: BitSet<Local>,

    // The original RETURN_PLACE local
    new_ret_local: Local,
}

impl<'tcx> TransformVisitor<'tcx> {
    // Make a `GeneratorState` or `Poll` variant assignment.
    //
    // `core::ops::GeneratorState` only has single element tuple variants,
    // so we can just write to the downcasted first field and then set the
    // discriminant to the appropriate variant.
    fn make_state(
        &self,
        val: Operand<'tcx>,
        source_info: SourceInfo,
        is_return: bool,
        statements: &mut Vec<Statement<'tcx>>,
    ) {
        let idx = VariantIdx::new(match (is_return, self.is_async_kind) {
            (true, false) => 1,  // GeneratorState::Complete
            (false, false) => 0, // GeneratorState::Yielded
            (true, true) => 0,   // Poll::Ready
            (false, true) => 1,  // Poll::Pending
        });

        let kind = AggregateKind::Adt(self.state_adt_ref.did(), idx, self.state_substs, None, None);

        // `Poll::Pending`
        if self.is_async_kind && idx == VariantIdx::new(1) {
            assert_eq!(self.state_adt_ref.variant(idx).fields.len(), 0);

            // FIXME(swatinem): assert that `val` is indeed unit?
            statements.push(Statement {
                kind: StatementKind::Assign(Box::new((
                    Place::return_place(),
                    Rvalue::Aggregate(Box::new(kind), IndexVec::new()),
                ))),
                source_info,
            });
            return;
        }

        // else: `Poll::Ready(x)`, `GeneratorState::Yielded(x)` or `GeneratorState::Complete(x)`
        assert_eq!(self.state_adt_ref.variant(idx).fields.len(), 1);

        statements.push(Statement {
            kind: StatementKind::Assign(Box::new((
                Place::return_place(),
                Rvalue::Aggregate(Box::new(kind), [val].into()),
            ))),
            source_info,
        });
    }

    // Create a Place referencing a generator struct field
    fn make_field(&self, variant_index: VariantIdx, idx: usize, ty: Ty<'tcx>) -> Place<'tcx> {
        let self_place = Place::from(SELF_ARG);
        let base = self.tcx.mk_place_downcast_unnamed(self_place, variant_index);
        let mut projection = base.projection.to_vec();
        projection.push(ProjectionElem::Field(FieldIdx::new(idx), ty));

        Place { local: base.local, projection: self.tcx.mk_place_elems(&projection) }
    }

    // Create a statement which changes the discriminant
    fn set_discr(&self, state_disc: VariantIdx, source_info: SourceInfo) -> Statement<'tcx> {
        let self_place = Place::from(SELF_ARG);
        Statement {
            source_info,
            kind: StatementKind::SetDiscriminant {
                place: Box::new(self_place),
                variant_index: state_disc,
            },
        }
    }

    // Create a statement which reads the discriminant into a temporary
    fn get_discr(&self, body: &mut Body<'tcx>) -> (Statement<'tcx>, Place<'tcx>) {
        let temp_decl = LocalDecl::new(self.discr_ty, body.span).internal();
        let local_decls_len = body.local_decls.push(temp_decl);
        let temp = Place::from(local_decls_len);

        let self_place = Place::from(SELF_ARG);
        let assign = Statement {
            source_info: SourceInfo::outermost(body.span),
            kind: StatementKind::Assign(Box::new((temp, Rvalue::Discriminant(self_place)))),
        };
        (assign, temp)
    }
}

impl<'tcx> MutVisitor<'tcx> for TransformVisitor<'tcx> {
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
            TerminatorKind::Return => {
                Some((true, None, Operand::Move(Place::from(self.new_ret_local)), None))
            }
            TerminatorKind::Yield { ref value, resume, resume_arg, drop } => {
                Some((false, Some((resume, resume_arg)), value.clone(), drop))
            }
            _ => None,
        };

        if let Some((is_return, resume, v, drop)) = ret_val {
            let source_info = data.terminator().source_info;
            // We must assign the value first in case it gets declared dead below
            self.make_state(v, source_info, is_return, &mut data.statements);
            let state = if let Some((resume, mut resume_arg)) = resume {
                // Yield
                let state = RESERVED_VARIANTS + self.suspension_points.len();

                // The resume arg target location might itself be remapped if its base local is
                // live across a yield.
                let resume_arg =
                    if let Some(&(ty, variant, idx)) = self.remap.get(&resume_arg.local) {
                        replace_base(&mut resume_arg, self.make_field(variant, idx, ty), self.tcx);
                        resume_arg
                    } else {
                        resume_arg
                    };

                self.suspension_points.push(SuspensionPoint {
                    state,
                    resume,
                    resume_arg,
                    drop,
                    storage_liveness: self.storage_liveness[block].clone().unwrap().into(),
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

    let pin_did = tcx.require_lang_item(LangItem::Pin, Some(body.span));
    let pin_adt_ref = tcx.adt_def(pin_did);
    let substs = tcx.mk_substs(&[ref_gen_ty.into()]);
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

/// Transforms the `body` of the generator applying the following transforms:
///
/// - Eliminates all the `get_context` calls that async lowering created.
/// - Replace all `Local` `ResumeTy` types with `&mut Context<'_>` (`context_mut_ref`).
///
/// The `Local`s that have their types replaced are:
/// - The `resume` argument itself.
/// - The argument to `get_context`.
/// - The yielded value of a `yield`.
///
/// The `ResumeTy` hides a `&mut Context<'_>` behind an unsafe raw pointer, and the
/// `get_context` function is being used to convert that back to a `&mut Context<'_>`.
///
/// Ideally the async lowering would not use the `ResumeTy`/`get_context` indirection,
/// but rather directly use `&mut Context<'_>`, however that would currently
/// lead to higher-kinded lifetime errors.
/// See <https://github.com/rust-lang/rust/issues/105501>.
///
/// The async lowering step and the type / lifetime inference / checking are
/// still using the `ResumeTy` indirection for the time being, and that indirection
/// is removed here. After this transform, the generator body only knows about `&mut Context<'_>`.
fn transform_async_context<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let context_mut_ref = tcx.mk_task_context();

    // replace the type of the `resume` argument
    replace_resume_ty_local(tcx, body, Local::new(2), context_mut_ref);

    let get_context_def_id = tcx.require_lang_item(LangItem::GetContext, None);

    for bb in START_BLOCK..body.basic_blocks.next_index() {
        let bb_data = &body[bb];
        if bb_data.is_cleanup {
            continue;
        }

        match &bb_data.terminator().kind {
            TerminatorKind::Call { func, .. } => {
                let func_ty = func.ty(body, tcx);
                if let ty::FnDef(def_id, _) = *func_ty.kind() {
                    if def_id == get_context_def_id {
                        let local = eliminate_get_context_call(&mut body[bb]);
                        replace_resume_ty_local(tcx, body, local, context_mut_ref);
                    }
                } else {
                    continue;
                }
            }
            TerminatorKind::Yield { resume_arg, .. } => {
                replace_resume_ty_local(tcx, body, resume_arg.local, context_mut_ref);
            }
            _ => {}
        }
    }
}

fn eliminate_get_context_call<'tcx>(bb_data: &mut BasicBlockData<'tcx>) -> Local {
    let terminator = bb_data.terminator.take().unwrap();
    if let TerminatorKind::Call { mut args, destination, target, .. } = terminator.kind {
        let arg = args.pop().unwrap();
        let local = arg.place().unwrap().local;

        let arg = Rvalue::Use(arg);
        let assign = Statement {
            source_info: terminator.source_info,
            kind: StatementKind::Assign(Box::new((destination, arg))),
        };
        bb_data.statements.push(assign);
        bb_data.terminator = Some(Terminator {
            source_info: terminator.source_info,
            kind: TerminatorKind::Goto { target: target.unwrap() },
        });
        local
    } else {
        bug!();
    }
}

#[cfg_attr(not(debug_assertions), allow(unused))]
fn replace_resume_ty_local<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    local: Local,
    context_mut_ref: Ty<'tcx>,
) {
    let local_ty = std::mem::replace(&mut body.local_decls[local].ty, context_mut_ref);
    // We have to replace the `ResumeTy` that is used for type and borrow checking
    // with `&mut Context<'_>` in MIR.
    #[cfg(debug_assertions)]
    {
        if let ty::Adt(resume_ty_adt, _) = local_ty.kind() {
            let expected_adt = tcx.adt_def(tcx.require_lang_item(LangItem::ResumeTy, None));
            assert_eq!(*resume_ty_adt, expected_adt);
        } else {
            panic!("expected `ResumeTy`, found `{:?}`", local_ty);
        };
    }
}

struct LivenessInfo {
    /// Which locals are live across any suspension point.
    saved_locals: GeneratorSavedLocals,

    /// The set of saved locals live at each suspension point.
    live_locals_at_suspension_points: Vec<BitSet<GeneratorSavedLocal>>,

    /// Parallel vec to the above with SourceInfo for each yield terminator.
    source_info_at_suspension_points: Vec<SourceInfo>,

    /// For every saved local, the set of other saved locals that are
    /// storage-live at the same time as this local. We cannot overlap locals in
    /// the layout which have conflicting storage.
    storage_conflicts: BitMatrix<GeneratorSavedLocal, GeneratorSavedLocal>,

    /// For every suspending block, the locals which are storage-live across
    /// that suspension point.
    storage_liveness: IndexVec<BasicBlock, Option<BitSet<Local>>>,
}

fn locals_live_across_suspend_points<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    always_live_locals: &BitSet<Local>,
    movable: bool,
) -> LivenessInfo {
    let body_ref: &Body<'_> = &body;

    // Calculate when MIR locals have live storage. This gives us an upper bound of their
    // lifetimes.
    let mut storage_live = MaybeStorageLive::new(std::borrow::Cow::Borrowed(always_live_locals))
        .into_engine(tcx, body_ref)
        .iterate_to_fixpoint()
        .into_results_cursor(body_ref);

    // Calculate the MIR locals which have been previously
    // borrowed (even if they are still active).
    let borrowed_locals_results =
        MaybeBorrowedLocals.into_engine(tcx, body_ref).pass_name("generator").iterate_to_fixpoint();

    let mut borrowed_locals_cursor =
        rustc_mir_dataflow::ResultsCursor::new(body_ref, &borrowed_locals_results);

    // Calculate the MIR locals that we actually need to keep storage around
    // for.
    let requires_storage_results = MaybeRequiresStorage::new(body, &borrowed_locals_results)
        .into_engine(tcx, body_ref)
        .iterate_to_fixpoint();
    let mut requires_storage_cursor =
        rustc_mir_dataflow::ResultsCursor::new(body_ref, &requires_storage_results);

    // Calculate the liveness of MIR locals ignoring borrows.
    let mut liveness = MaybeLiveLocals
        .into_engine(tcx, body_ref)
        .pass_name("generator")
        .iterate_to_fixpoint()
        .into_results_cursor(body_ref);

    let mut storage_liveness_map = IndexVec::from_elem(None, &body.basic_blocks);
    let mut live_locals_at_suspension_points = Vec::new();
    let mut source_info_at_suspension_points = Vec::new();
    let mut live_locals_at_any_suspension_point = BitSet::new_empty(body.local_decls.len());

    for (block, data) in body.basic_blocks.iter_enumerated() {
        if let TerminatorKind::Yield { .. } = data.terminator().kind {
            let loc = Location { block, statement_index: data.statements.len() };

            liveness.seek_to_block_end(block);
            let mut live_locals: BitSet<_> = BitSet::new_empty(body.local_decls.len());
            live_locals.union(liveness.get());

            if !movable {
                // The `liveness` variable contains the liveness of MIR locals ignoring borrows.
                // This is correct for movable generators since borrows cannot live across
                // suspension points. However for immovable generators we need to account for
                // borrows, so we conservatively assume that all borrowed locals are live until
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
            source_info_at_suspension_points.push(data.terminator().source_info);
        }
    }

    debug!("live_locals_anywhere = {:?}", live_locals_at_any_suspension_point);
    let saved_locals = GeneratorSavedLocals(live_locals_at_any_suspension_point);

    // Renumber our liveness_map bitsets to include only the locals we are
    // saving.
    let live_locals_at_suspension_points = live_locals_at_suspension_points
        .iter()
        .map(|live_here| saved_locals.renumber_bitset(&live_here))
        .collect();

    let storage_conflicts = compute_storage_conflicts(
        body_ref,
        &saved_locals,
        always_live_locals.clone(),
        requires_storage_results,
    );

    LivenessInfo {
        saved_locals,
        live_locals_at_suspension_points,
        source_info_at_suspension_points,
        storage_conflicts,
        storage_liveness: storage_liveness_map,
    }
}

/// The set of `Local`s that must be saved across yield points.
///
/// `GeneratorSavedLocal` is indexed in terms of the elements in this set;
/// i.e. `GeneratorSavedLocal::new(1)` corresponds to the second local
/// included in this set.
struct GeneratorSavedLocals(BitSet<Local>);

impl GeneratorSavedLocals {
    /// Returns an iterator over each `GeneratorSavedLocal` along with the `Local` it corresponds
    /// to.
    fn iter_enumerated(&self) -> impl '_ + Iterator<Item = (GeneratorSavedLocal, Local)> {
        self.iter().enumerate().map(|(i, l)| (GeneratorSavedLocal::from(i), l))
    }

    /// Transforms a `BitSet<Local>` that contains only locals saved across yield points to the
    /// equivalent `BitSet<GeneratorSavedLocal>`.
    fn renumber_bitset(&self, input: &BitSet<Local>) -> BitSet<GeneratorSavedLocal> {
        assert!(self.superset(&input), "{:?} not a superset of {:?}", self.0, input);
        let mut out = BitSet::new_empty(self.count());
        for (saved_local, local) in self.iter_enumerated() {
            if input.contains(local) {
                out.insert(saved_local);
            }
        }
        out
    }

    fn get(&self, local: Local) -> Option<GeneratorSavedLocal> {
        if !self.contains(local) {
            return None;
        }

        let idx = self.iter().take_while(|&l| l < local).count();
        Some(GeneratorSavedLocal::new(idx))
    }
}

impl ops::Deref for GeneratorSavedLocals {
    type Target = BitSet<Local>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// For every saved local, looks for which locals are StorageLive at the same
/// time. Generates a bitset for every local of all the other locals that may be
/// StorageLive simultaneously with that local. This is used in the layout
/// computation; see `GeneratorLayout` for more.
fn compute_storage_conflicts<'mir, 'tcx>(
    body: &'mir Body<'tcx>,
    saved_locals: &GeneratorSavedLocals,
    always_live_locals: BitSet<Local>,
    requires_storage: rustc_mir_dataflow::Results<'tcx, MaybeRequiresStorage<'mir, 'tcx>>,
) -> BitMatrix<GeneratorSavedLocal, GeneratorSavedLocal> {
    assert_eq!(body.local_decls.len(), saved_locals.domain_size());

    debug!("compute_storage_conflicts({:?})", body.span);
    debug!("always_live = {:?}", always_live_locals);

    // Locals that are always live or ones that need to be stored across
    // suspension points are not eligible for overlap.
    let mut ineligible_locals = always_live_locals;
    ineligible_locals.intersect(&**saved_locals);

    // Compute the storage conflicts for all eligible locals.
    let mut visitor = StorageConflictVisitor {
        body,
        saved_locals: &saved_locals,
        local_conflicts: BitMatrix::from_row_n(&ineligible_locals, body.local_decls.len()),
    };

    requires_storage.visit_reachable_with(body, &mut visitor);

    let local_conflicts = visitor.local_conflicts;

    // Compress the matrix using only stored locals (Local -> GeneratorSavedLocal).
    //
    // NOTE: Today we store a full conflict bitset for every local. Technically
    // this is twice as many bits as we need, since the relation is symmetric.
    // However, in practice these bitsets are not usually large. The layout code
    // also needs to keep track of how many conflicts each local has, so it's
    // simpler to keep it this way for now.
    let mut storage_conflicts = BitMatrix::new(saved_locals.count(), saved_locals.count());
    for (saved_local_a, local_a) in saved_locals.iter_enumerated() {
        if ineligible_locals.contains(local_a) {
            // Conflicts with everything.
            storage_conflicts.insert_all_into_row(saved_local_a);
        } else {
            // Keep overlap information only for stored locals.
            for (saved_local_b, local_b) in saved_locals.iter_enumerated() {
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
    saved_locals: &'s GeneratorSavedLocals,
    // FIXME(tmandry): Consider using sparse bitsets here once we have good
    // benchmarks for generators.
    local_conflicts: BitMatrix<Local, Local>,
}

impl<'mir, 'tcx> rustc_mir_dataflow::ResultsVisitor<'mir, 'tcx>
    for StorageConflictVisitor<'mir, 'tcx, '_>
{
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

impl StorageConflictVisitor<'_, '_, '_> {
    fn apply_state(&mut self, flow_state: &BitSet<Local>, loc: Location) {
        // Ignore unreachable blocks.
        if self.body.basic_blocks[loc.block].terminator().kind == TerminatorKind::Unreachable {
            return;
        }

        let mut eligible_storage_live = flow_state.clone();
        eligible_storage_live.intersect(&**self.saved_locals);

        for local in eligible_storage_live.iter() {
            self.local_conflicts.union_row_with(&eligible_storage_live, local);
        }

        if eligible_storage_live.count() > 1 {
            trace!("at {:?}, eligible_storage_live={:?}", loc, eligible_storage_live);
        }
    }
}

/// Validates the typeck view of the generator against the actual set of types saved between
/// yield points.
fn sanitize_witness<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    witness: Ty<'tcx>,
    upvars: Vec<Ty<'tcx>>,
    layout: &GeneratorLayout<'tcx>,
) {
    let did = body.source.def_id();
    let param_env = tcx.param_env(did);

    let allowed_upvars = tcx.normalize_erasing_regions(param_env, upvars);
    let allowed = match witness.kind() {
        &ty::GeneratorWitness(interior_tys) => {
            tcx.normalize_erasing_late_bound_regions(param_env, interior_tys)
        }
        _ => {
            tcx.sess.delay_span_bug(
                body.span,
                format!("unexpected generator witness type {:?}", witness.kind()),
            );
            return;
        }
    };

    let mut mismatches = Vec::new();
    for fty in &layout.field_tys {
        if fty.ignore_for_traits {
            continue;
        }
        let decl_ty = tcx.normalize_erasing_regions(param_env, fty.ty);

        // Sanity check that typeck knows about the type of locals which are
        // live across a suspension point
        if !allowed.contains(&decl_ty) && !allowed_upvars.contains(&decl_ty) {
            mismatches.push(decl_ty);
        }
    }

    if !mismatches.is_empty() {
        span_bug!(
            body.span,
            "Broken MIR: generator contains type {:?} in MIR, \
                       but typeck only knows about {} and {:?}",
            mismatches,
            allowed,
            allowed_upvars
        );
    }
}

fn compute_layout<'tcx>(
    tcx: TyCtxt<'tcx>,
    liveness: LivenessInfo,
    body: &Body<'tcx>,
) -> (
    FxHashMap<Local, (Ty<'tcx>, VariantIdx, usize)>,
    GeneratorLayout<'tcx>,
    IndexVec<BasicBlock, Option<BitSet<Local>>>,
) {
    let LivenessInfo {
        saved_locals,
        live_locals_at_suspension_points,
        source_info_at_suspension_points,
        storage_conflicts,
        storage_liveness,
    } = liveness;

    // Gather live local types and their indices.
    let mut locals = IndexVec::<GeneratorSavedLocal, _>::new();
    let mut tys = IndexVec::<GeneratorSavedLocal, _>::new();
    for (saved_local, local) in saved_locals.iter_enumerated() {
        debug!("generator saved local {:?} => {:?}", saved_local, local);

        locals.push(local);
        let decl = &body.local_decls[local];
        debug!(?decl);

        let ignore_for_traits = if tcx.sess.opts.unstable_opts.drop_tracking_mir {
            // Do not `assert_crate_local` here, as post-borrowck cleanup may have already cleared
            // the information. This is alright, since `ignore_for_traits` is only relevant when
            // this code runs on pre-cleanup MIR, and `ignore_for_traits = false` is the safer
            // default.
            match decl.local_info {
                // Do not include raw pointers created from accessing `static` items, as those could
                // well be re-created by another access to the same static.
                ClearCrossCrate::Set(box LocalInfo::StaticRef { is_thread_local, .. }) => {
                    !is_thread_local
                }
                // Fake borrows are only read by fake reads, so do not have any reality in
                // post-analysis MIR.
                ClearCrossCrate::Set(box LocalInfo::FakeBorrow) => true,
                _ => false,
            }
        } else {
            // FIXME(#105084) HIR-based drop tracking does not account for all the temporaries that
            // MIR building may introduce. This leads to wrongly ignored types, but this is
            // necessary for internal consistency and to avoid ICEs.
            decl.internal
        };
        let decl =
            GeneratorSavedTy { ty: decl.ty, source_info: decl.source_info, ignore_for_traits };
        debug!(?decl);

        tys.push(decl);
    }

    // Leave empty variants for the UNRESUMED, RETURNED, and POISONED states.
    // In debuginfo, these will correspond to the beginning (UNRESUMED) or end
    // (RETURNED, POISONED) of the function.
    let body_span = body.source_scopes[OUTERMOST_SOURCE_SCOPE].span;
    let mut variant_source_info: IndexVec<VariantIdx, SourceInfo> = [
        SourceInfo::outermost(body_span.shrink_to_lo()),
        SourceInfo::outermost(body_span.shrink_to_hi()),
        SourceInfo::outermost(body_span.shrink_to_hi()),
    ]
    .iter()
    .copied()
    .collect();

    // Build the generator variant field list.
    // Create a map from local indices to generator struct indices.
    let mut variant_fields: IndexVec<VariantIdx, IndexVec<FieldIdx, GeneratorSavedLocal>> =
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
            remap.entry(locals[saved_local]).or_insert((tys[saved_local].ty, variant_index, idx));
        }
        variant_fields.push(fields);
        variant_source_info.push(source_info_at_suspension_points[suspension_point_idx]);
    }
    debug!("generator variant_fields = {:?}", variant_fields);
    debug!("generator storage_conflicts = {:#?}", storage_conflicts);

    let layout =
        GeneratorLayout { field_tys: tys, variant_fields, variant_source_info, storage_conflicts };
    debug!(?layout);

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
    let switch_targets =
        SwitchTargets::new(cases.iter().map(|(i, bb)| ((*i) as u128, *bb)), default_block);
    let switch = TerminatorKind::SwitchInt { discr: Operand::Move(discr), targets: switch_targets };

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

fn elaborate_generator_drops<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    use crate::shim::DropShimElaborator;
    use rustc_middle::mir::patch::MirPatch;
    use rustc_mir_dataflow::elaborate_drops::{elaborate_drop, Unwind};

    // Note that `elaborate_drops` only drops the upvars of a generator, and
    // this is ok because `open_drop` can only be reached within that own
    // generator's resume function.

    let def_id = body.source.def_id();
    let param_env = tcx.param_env(def_id);

    let mut elaborator = DropShimElaborator { body, patch: MirPatch::new(body), tcx, param_env };

    for (block, block_data) in body.basic_blocks.iter_enumerated() {
        let (target, unwind, source_info) = match block_data.terminator() {
            Terminator { source_info, kind: TerminatorKind::Drop { place, target, unwind } } => {
                if let Some(local) = place.as_local() {
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
            Unwind::To(match *unwind {
                UnwindAction::Cleanup(tgt) => tgt,
                UnwindAction::Continue => elaborator.patch.resume_block(),
                UnwindAction::Unreachable => elaborator.patch.unreachable_cleanup_block(),
                UnwindAction::Terminate => elaborator.patch.terminate_block(),
            })
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

    // Make sure we remove dead blocks to remove
    // unrelated code from the resume part of the function
    simplify::remove_dead_blocks(tcx, &mut body);

    dump_mir(tcx, false, "generator_drop", &0, &body, |_, _| Ok(()));

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
    let assert_block = BasicBlock::new(body.basic_blocks.len());
    let term = TerminatorKind::Assert {
        cond: Operand::Constant(Box::new(Constant {
            span: body.span,
            user_ty: None,
            literal: ConstantKind::from_bool(tcx, false),
        })),
        expected: true,
        msg: Box::new(message),
        target: assert_block,
        unwind: UnwindAction::Continue,
    };

    let source_info = SourceInfo::outermost(body.span);
    body.basic_blocks_mut().push(BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator { source_info, kind: term }),
        is_cleanup: false,
    });

    assert_block
}

fn can_return<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, param_env: ty::ParamEnv<'tcx>) -> bool {
    // Returning from a function with an uninhabited return type is undefined behavior.
    if body.return_ty().is_privately_uninhabited(tcx, param_env) {
        return false;
    }

    // If there's a return terminator the function may return.
    for block in body.basic_blocks.iter() {
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
    for block in body.basic_blocks.iter() {
        match block.terminator().kind {
            // These never unwind.
            TerminatorKind::Goto { .. }
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Terminate
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. } => {}

            // Resume will *continue* unwinding, but if there's no other unwinding terminator it
            // will never be reached.
            TerminatorKind::Resume => {}

            TerminatorKind::Yield { .. } => {
                unreachable!("`can_unwind` called before generator transform")
            }

            // These may unwind.
            TerminatorKind::Drop { .. }
            | TerminatorKind::Call { .. }
            | TerminatorKind::InlineAsm { .. }
            | TerminatorKind::Assert { .. } => return true,
        }
    }

    // If we didn't find an unwinding terminator, the function cannot unwind.
    false
}

fn create_generator_resume_function<'tcx>(
    tcx: TyCtxt<'tcx>,
    transform: TransformVisitor<'tcx>,
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
                if let Some(unwind @ UnwindAction::Continue) = block.terminator_mut().unwind_mut() {
                    *unwind = UnwindAction::Cleanup(poison_block);
                }
            }
        }
    }

    let mut cases = create_cases(body, &transform, Operation::Resume);

    use rustc_middle::mir::AssertKind::{ResumedAfterPanic, ResumedAfterReturn};

    // Jump to the entry point on the unresumed
    cases.insert(0, (UNRESUMED, START_BLOCK));

    // Panic when resumed on the returned or poisoned state
    let generator_kind = body.generator_kind().unwrap();

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

    // Make sure we remove dead blocks to remove
    // unrelated code from the drop part of the function
    simplify::remove_dead_blocks(tcx, body);

    dump_mir(tcx, false, "generator_resume", &0, body, |_, _| Ok(()));
}

fn insert_clean_drop(body: &mut Body<'_>) -> BasicBlock {
    let return_block = insert_term_block(body, TerminatorKind::Return);

    let term = TerminatorKind::Drop {
        place: Place::from(SELF_ARG),
        target: return_block,
        unwind: UnwindAction::Continue,
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
                        kind: StatementKind::Assign(Box::new((
                            point.resume_arg,
                            Rvalue::Use(Operand::Move(resume_arg.into())),
                        ))),
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

#[instrument(level = "debug", skip(tcx), ret)]
pub(crate) fn mir_generator_witnesses<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> GeneratorLayout<'tcx> {
    assert!(tcx.sess.opts.unstable_opts.drop_tracking_mir);

    let (body, _) = tcx.mir_promoted(def_id);
    let body = body.borrow();
    let body = &*body;

    // The first argument is the generator type passed by value
    let gen_ty = body.local_decls[ty::CAPTURE_STRUCT_LOCAL].ty;

    // Get the interior types and substs which typeck computed
    let movable = match *gen_ty.kind() {
        ty::Generator(_, _, movability) => movability == hir::Movability::Movable,
        _ => span_bug!(body.span, "unexpected generator type {}", gen_ty),
    };

    // When first entering the generator, move the resume argument into its new local.
    let always_live_locals = always_storage_live_locals(&body);

    let liveness_info = locals_live_across_suspend_points(tcx, body, &always_live_locals, movable);

    // Extract locals which are live across suspension point into `layout`
    // `remap` gives a mapping from local indices onto generator struct indices
    // `storage_liveness` tells us which locals have live storage at suspension points
    let (_, generator_layout, _) = compute_layout(tcx, liveness_info, body);

    check_suspend_tys(tcx, &generator_layout, &body);

    generator_layout
}

impl<'tcx> MirPass<'tcx> for StateTransform {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let Some(yield_ty) = body.yield_ty() else {
            // This only applies to generators
            return;
        };

        assert!(body.generator_drop().is_none());

        // The first argument is the generator type passed by value
        let gen_ty = body.local_decls.raw[1].ty;

        // Get the discriminant type and substs which typeck computed
        let (discr_ty, upvars, interior, movable) = match *gen_ty.kind() {
            ty::Generator(_, substs, movability) => {
                let substs = substs.as_generator();
                (
                    substs.discr_ty(tcx),
                    substs.upvar_tys().collect::<Vec<_>>(),
                    substs.witness(),
                    movability == hir::Movability::Movable,
                )
            }
            _ => {
                tcx.sess.delay_span_bug(body.span, format!("unexpected generator type {}", gen_ty));
                return;
            }
        };

        let is_async_kind = matches!(body.generator_kind(), Some(GeneratorKind::Async(_)));
        let (state_adt_ref, state_substs) = if is_async_kind {
            // Compute Poll<return_ty>
            let poll_did = tcx.require_lang_item(LangItem::Poll, None);
            let poll_adt_ref = tcx.adt_def(poll_did);
            let poll_substs = tcx.mk_substs(&[body.return_ty().into()]);
            (poll_adt_ref, poll_substs)
        } else {
            // Compute GeneratorState<yield_ty, return_ty>
            let state_did = tcx.require_lang_item(LangItem::GeneratorState, None);
            let state_adt_ref = tcx.adt_def(state_did);
            let state_substs = tcx.mk_substs(&[yield_ty.into(), body.return_ty().into()]);
            (state_adt_ref, state_substs)
        };
        let ret_ty = tcx.mk_adt(state_adt_ref, state_substs);

        // We rename RETURN_PLACE which has type mir.return_ty to new_ret_local
        // RETURN_PLACE then is a fresh unused local with type ret_ty.
        let new_ret_local = replace_local(RETURN_PLACE, ret_ty, body, tcx);

        // Replace all occurrences of `ResumeTy` with `&mut Context<'_>` within async bodies.
        if is_async_kind {
            transform_async_context(tcx, body);
        }

        // We also replace the resume argument and insert an `Assign`.
        // This is needed because the resume argument `_2` might be live across a `yield`, in which
        // case there is no `Assign` to it that the transform can turn into a store to the generator
        // state. After the yield the slot in the generator state would then be uninitialized.
        let resume_local = Local::new(2);
        let resume_ty =
            if is_async_kind { tcx.mk_task_context() } else { body.local_decls[resume_local].ty };
        let new_resume_local = replace_local(resume_local, resume_ty, body, tcx);

        // When first entering the generator, move the resume argument into its new local.
        let source_info = SourceInfo::outermost(body.span);
        let stmts = &mut body.basic_blocks_mut()[START_BLOCK].statements;
        stmts.insert(
            0,
            Statement {
                source_info,
                kind: StatementKind::Assign(Box::new((
                    new_resume_local.into(),
                    Rvalue::Use(Operand::Move(resume_local.into())),
                ))),
            },
        );

        let always_live_locals = always_storage_live_locals(&body);

        let liveness_info =
            locals_live_across_suspend_points(tcx, body, &always_live_locals, movable);

        if tcx.sess.opts.unstable_opts.validate_mir {
            let mut vis = EnsureGeneratorFieldAssignmentsNeverAlias {
                assigned_local: None,
                saved_locals: &liveness_info.saved_locals,
                storage_conflicts: &liveness_info.storage_conflicts,
            };

            vis.visit_body(body);
        }

        // Extract locals which are live across suspension point into `layout`
        // `remap` gives a mapping from local indices onto generator struct indices
        // `storage_liveness` tells us which locals have live storage at suspension points
        let (remap, layout, storage_liveness) = compute_layout(tcx, liveness_info, body);

        if tcx.sess.opts.unstable_opts.validate_mir
            && !tcx.sess.opts.unstable_opts.drop_tracking_mir
        {
            sanitize_witness(tcx, body, interior, upvars, &layout);
        }

        let can_return = can_return(tcx, body, tcx.param_env(body.source.def_id()));

        // Run the transformation which converts Places from Local to generator struct
        // accesses for locals in `remap`.
        // It also rewrites `return x` and `yield y` as writing a new generator state and returning
        // either GeneratorState::Complete(x) and GeneratorState::Yielded(y),
        // or Poll::Ready(x) and Poll::Pending respectively depending on `is_async_kind`.
        let mut transform = TransformVisitor {
            tcx,
            is_async_kind,
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
        body.arg_count = 2; // self, resume arg
        body.spread_arg = None;

        // The original arguments to the function are no longer arguments, mark them as such.
        // Otherwise they'll conflict with our new arguments, which although they don't have
        // argument_index set, will get emitted as unnamed arguments.
        for var in &mut body.var_debug_info {
            var.argument_index = None;
        }

        body.generator.as_mut().unwrap().yield_ty = None;
        body.generator.as_mut().unwrap().generator_layout = Some(layout);

        // Insert `drop(generator_struct)` which is used to drop upvars for generators in
        // the unresumed state.
        // This is expanded to a drop ladder in `elaborate_generator_drops`.
        let drop_clean = insert_clean_drop(body);

        dump_mir(tcx, false, "generator_pre-elab", &0, body, |_, _| Ok(()));

        // Expand `drop(generator_struct)` to a drop ladder which destroys upvars.
        // If any upvars are moved out of, drop elaboration will handle upvar destruction.
        // However we need to also elaborate the code generated by `insert_clean_drop`.
        elaborate_generator_drops(tcx, body);

        dump_mir(tcx, false, "generator_post-transform", &0, body, |_, _| Ok(()));

        // Create a copy of our MIR and use it to create the drop shim for the generator
        let drop_shim = create_generator_drop_shim(tcx, &transform, gen_ty, body, drop_clean);

        body.generator.as_mut().unwrap().generator_drop = Some(drop_shim);

        // Create the Generator::resume / Future::poll function
        create_generator_resume_function(tcx, transform, body, can_return);

        // Run derefer to fix Derefs that are not in the first place
        deref_finder(tcx, body);
    }
}

/// Looks for any assignments between locals (e.g., `_4 = _5`) that will both be converted to fields
/// in the generator state machine but whose storage is not marked as conflicting
///
/// Validation needs to happen immediately *before* `TransformVisitor` is invoked, not after.
///
/// This condition would arise when the assignment is the last use of `_5` but the initial
/// definition of `_4` if we weren't extra careful to mark all locals used inside a statement as
/// conflicting. Non-conflicting generator saved locals may be stored at the same location within
/// the generator state machine, which would result in ill-formed MIR: the left-hand and right-hand
/// sides of an assignment may not alias. This caused a miscompilation in [#73137].
///
/// [#73137]: https://github.com/rust-lang/rust/issues/73137
struct EnsureGeneratorFieldAssignmentsNeverAlias<'a> {
    saved_locals: &'a GeneratorSavedLocals,
    storage_conflicts: &'a BitMatrix<GeneratorSavedLocal, GeneratorSavedLocal>,
    assigned_local: Option<GeneratorSavedLocal>,
}

impl EnsureGeneratorFieldAssignmentsNeverAlias<'_> {
    fn saved_local_for_direct_place(&self, place: Place<'_>) -> Option<GeneratorSavedLocal> {
        if place.is_indirect() {
            return None;
        }

        self.saved_locals.get(place.local)
    }

    fn check_assigned_place(&mut self, place: Place<'_>, f: impl FnOnce(&mut Self)) {
        if let Some(assigned_local) = self.saved_local_for_direct_place(place) {
            assert!(self.assigned_local.is_none(), "`check_assigned_place` must not recurse");

            self.assigned_local = Some(assigned_local);
            f(self);
            self.assigned_local = None;
        }
    }
}

impl<'tcx> Visitor<'tcx> for EnsureGeneratorFieldAssignmentsNeverAlias<'_> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        let Some(lhs) = self.assigned_local else {
            // This visitor only invokes `visit_place` for the right-hand side of an assignment
            // and only after setting `self.assigned_local`. However, the default impl of
            // `Visitor::super_body` may call `visit_place` with a `NonUseContext` for places
            // with debuginfo. Ignore them here.
            assert!(!context.is_use());
            return;
        };

        let Some(rhs) = self.saved_local_for_direct_place(*place) else { return };

        if !self.storage_conflicts.contains(lhs, rhs) {
            bug!(
                "Assignment between generator saved locals whose storage is not \
                    marked as conflicting: {:?}: {:?} = {:?}",
                location,
                lhs,
                rhs,
            );
        }
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match &statement.kind {
            StatementKind::Assign(box (lhs, rhs)) => {
                self.check_assigned_place(*lhs, |this| this.visit_rvalue(rhs, location));
            }

            StatementKind::FakeRead(..)
            | StatementKind::SetDiscriminant { .. }
            | StatementKind::Deinit(..)
            | StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Retag(..)
            | StatementKind::AscribeUserType(..)
            | StatementKind::PlaceMention(..)
            | StatementKind::Coverage(..)
            | StatementKind::Intrinsic(..)
            | StatementKind::ConstEvalCounter
            | StatementKind::Nop => {}
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        // Checking for aliasing in terminators is probably overkill, but until we have actual
        // semantics, we should be conservative here.
        match &terminator.kind {
            TerminatorKind::Call {
                func,
                args,
                destination,
                target: Some(_),
                unwind: _,
                from_hir_call: _,
                fn_span: _,
            } => {
                self.check_assigned_place(*destination, |this| {
                    this.visit_operand(func, location);
                    for arg in args {
                        this.visit_operand(arg, location);
                    }
                });
            }

            TerminatorKind::Yield { value, resume: _, resume_arg, drop: _ } => {
                self.check_assigned_place(*resume_arg, |this| this.visit_operand(value, location));
            }

            // FIXME: Does `asm!` have any aliasing requirements?
            TerminatorKind::InlineAsm { .. } => {}

            TerminatorKind::Call { .. }
            | TerminatorKind::Goto { .. }
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Resume
            | TerminatorKind::Terminate
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::Drop { .. }
            | TerminatorKind::Assert { .. }
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. } => {}
        }
    }
}

fn check_suspend_tys<'tcx>(tcx: TyCtxt<'tcx>, layout: &GeneratorLayout<'tcx>, body: &Body<'tcx>) {
    let mut linted_tys = FxHashSet::default();

    // We want a user-facing param-env.
    let param_env = tcx.param_env(body.source.def_id());

    for (variant, yield_source_info) in
        layout.variant_fields.iter().zip(&layout.variant_source_info)
    {
        debug!(?variant);
        for &local in variant {
            let decl = &layout.field_tys[local];
            debug!(?decl);

            if !decl.ignore_for_traits && linted_tys.insert(decl.ty) {
                let Some(hir_id) = decl.source_info.scope.lint_root(&body.source_scopes) else { continue };

                check_must_not_suspend_ty(
                    tcx,
                    decl.ty,
                    hir_id,
                    param_env,
                    SuspendCheckData {
                        source_span: decl.source_info.span,
                        yield_span: yield_source_info.span,
                        plural_len: 1,
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[derive(Default)]
struct SuspendCheckData<'a> {
    source_span: Span,
    yield_span: Span,
    descr_pre: &'a str,
    descr_post: &'a str,
    plural_len: usize,
}

// Returns whether it emitted a diagnostic or not
// Note that this fn and the proceeding one are based on the code
// for creating must_use diagnostics
//
// Note that this technique was chosen over things like a `Suspend` marker trait
// as it is simpler and has precedent in the compiler
fn check_must_not_suspend_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    hir_id: hir::HirId,
    param_env: ty::ParamEnv<'tcx>,
    data: SuspendCheckData<'_>,
) -> bool {
    if ty.is_unit() {
        return false;
    }

    let plural_suffix = pluralize!(data.plural_len);

    debug!("Checking must_not_suspend for {}", ty);

    match *ty.kind() {
        ty::Adt(..) if ty.is_box() => {
            let boxed_ty = ty.boxed_ty();
            let descr_pre = &format!("{}boxed ", data.descr_pre);
            check_must_not_suspend_ty(
                tcx,
                boxed_ty,
                hir_id,
                param_env,
                SuspendCheckData { descr_pre, ..data },
            )
        }
        ty::Adt(def, _) => check_must_not_suspend_def(tcx, def.did(), hir_id, data),
        // FIXME: support adding the attribute to TAITs
        ty::Alias(ty::Opaque, ty::AliasTy { def_id: def, .. }) => {
            let mut has_emitted = false;
            for &(predicate, _) in tcx.explicit_item_bounds(def).skip_binder() {
                // We only look at the `DefId`, so it is safe to skip the binder here.
                if let ty::PredicateKind::Clause(ty::Clause::Trait(ref poly_trait_predicate)) =
                    predicate.kind().skip_binder()
                {
                    let def_id = poly_trait_predicate.trait_ref.def_id;
                    let descr_pre = &format!("{}implementer{} of ", data.descr_pre, plural_suffix);
                    if check_must_not_suspend_def(
                        tcx,
                        def_id,
                        hir_id,
                        SuspendCheckData { descr_pre, ..data },
                    ) {
                        has_emitted = true;
                        break;
                    }
                }
            }
            has_emitted
        }
        ty::Dynamic(binder, _, _) => {
            let mut has_emitted = false;
            for predicate in binder.iter() {
                if let ty::ExistentialPredicate::Trait(ref trait_ref) = predicate.skip_binder() {
                    let def_id = trait_ref.def_id;
                    let descr_post = &format!(" trait object{}{}", plural_suffix, data.descr_post);
                    if check_must_not_suspend_def(
                        tcx,
                        def_id,
                        hir_id,
                        SuspendCheckData { descr_post, ..data },
                    ) {
                        has_emitted = true;
                        break;
                    }
                }
            }
            has_emitted
        }
        ty::Tuple(fields) => {
            let mut has_emitted = false;
            for (i, ty) in fields.iter().enumerate() {
                let descr_post = &format!(" in tuple element {i}");
                if check_must_not_suspend_ty(
                    tcx,
                    ty,
                    hir_id,
                    param_env,
                    SuspendCheckData { descr_post, ..data },
                ) {
                    has_emitted = true;
                }
            }
            has_emitted
        }
        ty::Array(ty, len) => {
            let descr_pre = &format!("{}array{} of ", data.descr_pre, plural_suffix);
            check_must_not_suspend_ty(
                tcx,
                ty,
                hir_id,
                param_env,
                SuspendCheckData {
                    descr_pre,
                    plural_len: len.try_eval_target_usize(tcx, param_env).unwrap_or(0) as usize + 1,
                    ..data
                },
            )
        }
        // If drop tracking is enabled, we want to look through references, since the referent
        // may not be considered live across the await point.
        ty::Ref(_region, ty, _mutability) => {
            let descr_pre = &format!("{}reference{} to ", data.descr_pre, plural_suffix);
            check_must_not_suspend_ty(
                tcx,
                ty,
                hir_id,
                param_env,
                SuspendCheckData { descr_pre, ..data },
            )
        }
        _ => false,
    }
}

fn check_must_not_suspend_def(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    hir_id: hir::HirId,
    data: SuspendCheckData<'_>,
) -> bool {
    if let Some(attr) = tcx.get_attr(def_id, sym::must_not_suspend) {
        let reason = attr.value_str().map(|s| errors::MustNotSuspendReason {
            span: data.source_span,
            reason: s.as_str().to_string(),
        });
        tcx.emit_spanned_lint(
            rustc_session::lint::builtin::MUST_NOT_SUSPEND,
            hir_id,
            data.source_span,
            errors::MustNotSupend {
                yield_sp: data.yield_span,
                reason,
                src_sp: data.source_span,
                pre: data.descr_pre,
                def_path: tcx.def_path_str(def_id),
                post: data.descr_post,
            },
        );

        true
    } else {
        false
    }
}
