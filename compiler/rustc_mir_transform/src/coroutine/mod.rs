//! This is the implementation of the pass which transforms coroutines into state machines.
//!
//! MIR generation for coroutines creates a function which has a self argument which
//! passes by value. This argument is effectively a coroutine type which only contains upvars and
//! is only used for this argument inside the MIR for the coroutine.
//! It is passed by value to enable upvars to be moved out of it. Drop elaboration runs on that
//! MIR before this pass and creates drop flags for MIR locals.
//! It will also drop the coroutine argument (which only consists of upvars) if any of the upvars
//! are moved out of. This pass elaborates the drops of upvars / coroutine argument in the case
//! that none of the upvars were moved out of. This is because we cannot have any drops of this
//! coroutine in the MIR, since it is used to create the drop glue for the coroutine. We'd get
//! infinite recursion otherwise.
//!
//! This pass creates the implementation for either the `Coroutine::resume` or `Future::poll`
//! function and the drop shim for the coroutine based on the MIR input.
//! It converts the coroutine argument from Self to &mut Self adding derefs in the MIR as needed.
//! It computes the final layout of the coroutine struct which looks like this:
//!     First upvars are stored
//!     It is followed by the coroutine state field.
//!     Then finally the MIR locals which are live across a suspension point are stored.
//!     ```ignore (illustrative)
//!     struct Coroutine {
//!         upvars...,
//!         state: u32,
//!         mir_locals...,
//!     }
//!     ```
//! This pass computes the meaning of the state field and the MIR locals which are live
//! across a suspension point. There are however three hardcoded coroutine states:
//!     0 - Coroutine have not been resumed yet
//!     1 - Coroutine has returned / is completed
//!     2 - Coroutine has been poisoned
//!
//! It also rewrites `return x` and `yield y` as setting a new coroutine state and returning
//! `CoroutineState::Complete(x)` and `CoroutineState::Yielded(y)`,
//! or `Poll::Ready(x)` and `Poll::Pending` respectively.
//! MIR locals which are live across a suspension point are moved to the coroutine struct
//! with references to them being updated with references to the coroutine struct.
//!
//! The pass creates two functions which have a switch on the coroutine state giving
//! the action to take.
//!
//! One of them is the implementation of `Coroutine::resume` / `Future::poll`.
//! For coroutines with state 0 (unresumed) it starts the execution of the coroutine.
//! For coroutines with state 1 (returned) and state 2 (poisoned) it panics.
//! Otherwise it continues the execution from the last suspension point.
//!
//! The other function is the drop glue for the coroutine.
//! For coroutines with state 0 (unresumed) it drops the upvars of the coroutine.
//! For coroutines with state 1 (returned) and state 2 (poisoned) it does nothing.
//! Otherwise it drops all the values in scope at the last suspension point.

mod by_move_body;
mod drop;
mod layout;

pub(super) use by_move_body::coroutine_by_move_body_def_id;
use drop::{
    create_coroutine_drop_shim, create_coroutine_drop_shim_async,
    create_coroutine_drop_shim_proxy_async, elaborate_coroutine_drops, has_async_drops,
    insert_clean_drop,
};
pub(super) use layout::mir_coroutine_witnesses;
use layout::{CoroutineSavedLocals, compute_layout, locals_live_across_suspend_points};
use rustc_abi::{FieldIdx, VariantIdx};
use rustc_hir::lang_items::LangItem;
use rustc_hir::{self as hir, CoroutineDesugaring, CoroutineKind};
use rustc_index::bit_set::{BitMatrix, DenseBitSet, GrowableBitSet};
use rustc_index::{Idx, IndexVec, indexvec};
use rustc_middle::mir::visit::{MutVisitor, MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{
    self, CoroutineArgs, CoroutineArgsExt, GenericArgsRef, InstanceKind, Ty, TyCtxt,
};
use rustc_middle::{bug, span_bug};
use rustc_mir_dataflow::impls::always_storage_live_locals;
use rustc_span::def_id::DefId;
use tracing::{debug, instrument};

use crate::deref_separator::deref_finder;
use crate::{abort_unwinding_calls, pass_manager as pm, simplify};

pub(super) struct StateTransform;

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
        } else if *local == self.to {
            *local = self.from;
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

struct SelfArgVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    new_base: Place<'tcx>,
}

impl<'tcx> SelfArgVisitor<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, new_base: Place<'tcx>) -> Self {
        Self { tcx, new_base }
    }
}

impl<'tcx> MutVisitor<'tcx> for SelfArgVisitor<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _: Location) {
        assert_ne!(*local, SELF_ARG);
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, _: PlaceContext, _: Location) {
        if place.local == SELF_ARG {
            replace_base(place, self.new_base, self.tcx);
        }

        for elem in place.projection.iter() {
            if let PlaceElem::Index(local) = elem {
                assert_ne!(local, SELF_ARG);
            }
        }
    }
}

#[tracing::instrument(level = "trace", skip(tcx))]
fn replace_base<'tcx>(place: &mut Place<'tcx>, new_base: Place<'tcx>, tcx: TyCtxt<'tcx>) {
    place.local = new_base.local;

    let mut new_projection = new_base.projection.to_vec();
    new_projection.append(&mut place.projection.to_vec());

    place.projection = tcx.mk_place_elems(&new_projection);
    tracing::trace!(?place);
}

const SELF_ARG: Local = Local::arg(0);
pub(crate) const CTX_ARG: Local = Local::arg(1);

/// A `yield` point in the coroutine.
struct SuspensionPoint<'tcx> {
    /// State discriminant used when suspending or resuming at this point.
    state: usize,
    /// The block to jump to after resumption.
    resume: BasicBlock,
    /// Where to move the resume argument after resumption.
    resume_arg: Place<'tcx>,
    /// Which block to jump to if the coroutine is dropped in this state.
    drop: Option<BasicBlock>,
    /// Set of locals that have live storage while at this suspension point.
    storage_liveness: GrowableBitSet<Local>,
}

struct TransformVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    coroutine_kind: hir::CoroutineKind,

    // The type of the discriminant in the coroutine struct
    discr_ty: Ty<'tcx>,

    // Mapping from Local to (type of local, coroutine struct index)
    remap: IndexVec<Local, Option<(Ty<'tcx>, VariantIdx, FieldIdx)>>,

    // A map from a suspension point in a block to the locals which have live storage at that point
    storage_liveness: IndexVec<BasicBlock, Option<DenseBitSet<Local>>>,

    // A list of suspension points, generated during the transform
    suspension_points: Vec<SuspensionPoint<'tcx>>,

    // The set of locals that have no `StorageLive`/`StorageDead` annotations.
    always_live_locals: DenseBitSet<Local>,

    // New local we just create to hold the `CoroutineState` value.
    new_ret_local: Local,

    old_yield_ty: Ty<'tcx>,

    old_ret_ty: Ty<'tcx>,
}

impl<'tcx> TransformVisitor<'tcx> {
    fn insert_none_ret_block(&self, body: &mut Body<'tcx>) -> BasicBlock {
        let block = body.basic_blocks.next_index();
        let source_info = SourceInfo::outermost(body.span);

        let none_value = match self.coroutine_kind {
            CoroutineKind::Desugared(CoroutineDesugaring::Async, _) => {
                span_bug!(body.span, "`Future`s are not fused inherently")
            }
            CoroutineKind::Coroutine(_) => span_bug!(body.span, "`Coroutine`s cannot be fused"),
            // `gen` and `async gen` continue to return `CoroutineState::Complete(())`.
            CoroutineKind::Desugared(
                CoroutineDesugaring::Gen | CoroutineDesugaring::AsyncGen,
                _,
            ) => {
                let coroutine_state_def_id =
                    self.tcx.require_lang_item(LangItem::CoroutineState, source_info.span);
                let args = self.tcx.mk_args(&[self.old_yield_ty.into(), self.old_ret_ty.into()]);
                let val = Operand::Constant(Box::new(ConstOperand {
                    span: source_info.span,
                    user_ty: None,
                    const_: Const::zero_sized(self.tcx.types.unit),
                }));
                make_aggregate_adt(
                    coroutine_state_def_id,
                    VariantIdx::from_usize(1),
                    args,
                    indexvec![val],
                )
            }
        };

        let statements = vec![Statement::new(
            source_info,
            StatementKind::Assign(Box::new((Place::return_place(), none_value))),
        )];

        body.basic_blocks_mut().push(BasicBlockData::new_stmts(
            statements,
            Some(Terminator { source_info, kind: TerminatorKind::Return }),
            false,
        ));

        block
    }

    // Make a `CoroutineState` or `Poll` variant assignment.
    //
    // `core::ops::CoroutineState` only has single element tuple variants,
    // so we can just write to the downcasted first field and then set the
    // discriminant to the appropriate variant.
    #[tracing::instrument(level = "trace", skip(self, statements))]
    fn make_state(
        &self,
        val: Operand<'tcx>,
        source_info: SourceInfo,
        is_return: bool,
        statements: &mut Vec<Statement<'tcx>>,
    ) {
        let coroutine_state_def_id =
            self.tcx.require_lang_item(LangItem::CoroutineState, source_info.span);
        let args = self.tcx.mk_args(&[self.old_yield_ty.into(), self.old_ret_ty.into()]);
        let variant_idx = if is_return {
            VariantIdx::from_usize(1) // CoroutineState::Complete(val)
        } else {
            VariantIdx::ZERO // CoroutineState::Yielded(val)
        };

        let rvalue = make_aggregate_adt(coroutine_state_def_id, variant_idx, args, indexvec![val]);

        // Assign to `new_ret_local`, which will be replaced by `RETURN_PLACE` later.
        statements.push(Statement::new(
            source_info,
            StatementKind::Assign(Box::new((self.new_ret_local.into(), rvalue))),
        ));
    }

    // Create a Place referencing a coroutine struct field
    #[tracing::instrument(level = "trace", skip(self), ret)]
    fn make_field(&self, variant_index: VariantIdx, idx: FieldIdx, ty: Ty<'tcx>) -> Place<'tcx> {
        let self_place = Place::from(SELF_ARG);
        let base = self.tcx.mk_place_downcast_unnamed(self_place, variant_index);
        let mut projection = base.projection.to_vec();
        projection.push(ProjectionElem::Field(idx, ty));

        Place { local: base.local, projection: self.tcx.mk_place_elems(&projection) }
    }

    // Create a statement which changes the discriminant
    #[tracing::instrument(level = "trace", skip(self))]
    fn set_discr(&self, state_disc: VariantIdx, source_info: SourceInfo) -> Statement<'tcx> {
        let self_place = Place::from(SELF_ARG);
        Statement::new(
            source_info,
            StatementKind::SetDiscriminant {
                place: Box::new(self_place),
                variant_index: state_disc,
            },
        )
    }

    // Create a statement which reads the discriminant into a temporary
    #[tracing::instrument(level = "trace", skip(self, body))]
    fn get_discr(&self, body: &mut Body<'tcx>) -> (Statement<'tcx>, Place<'tcx>) {
        let temp_decl = LocalDecl::new(self.discr_ty, body.span);
        let local_decls_len = body.local_decls.push(temp_decl);
        let temp = Place::from(local_decls_len);

        let self_place = Place::from(SELF_ARG);
        let assign = Statement::new(
            SourceInfo::outermost(body.span),
            StatementKind::Assign(Box::new((temp, Rvalue::Discriminant(self_place)))),
        );
        (assign, temp)
    }

    /// Swaps all references of `old_local` and `new_local`.
    #[tracing::instrument(level = "trace", skip(self, body))]
    fn replace_local(&mut self, old_local: Local, new_local: Local, body: &mut Body<'tcx>) {
        body.local_decls.swap(old_local, new_local);

        let mut visitor = RenameLocalVisitor { from: old_local, to: new_local, tcx: self.tcx };
        visitor.visit_body(body);
        for suspension in &mut self.suspension_points {
            let ctxt = PlaceContext::MutatingUse(MutatingUseContext::Yield);
            let location = Location { block: START_BLOCK, statement_index: 0 };
            visitor.visit_place(&mut suspension.resume_arg, ctxt, location);
        }
    }
}

impl<'tcx> MutVisitor<'tcx> for TransformVisitor<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    #[tracing::instrument(level = "trace", skip(self), ret)]
    fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _location: Location) {
        assert!(!self.remap.contains(*local));
    }

    #[tracing::instrument(level = "trace", skip(self), ret)]
    fn visit_place(&mut self, place: &mut Place<'tcx>, _: PlaceContext, _location: Location) {
        // Replace an Local in the remap with a coroutine struct access
        if let Some(&Some((ty, variant_index, idx))) = self.remap.get(place.local) {
            replace_base(place, self.make_field(variant_index, idx, ty), self.tcx);
        }
    }

    #[tracing::instrument(level = "trace", skip(self, stmt), ret)]
    fn visit_statement(&mut self, stmt: &mut Statement<'tcx>, location: Location) {
        // Remove StorageLive and StorageDead statements for remapped locals
        if let StatementKind::StorageLive(l) | StatementKind::StorageDead(l) = stmt.kind
            && self.remap.contains(l)
        {
            stmt.make_nop(true);
        }
        self.super_statement(stmt, location);
    }

    #[tracing::instrument(level = "trace", skip(self, term), ret)]
    fn visit_terminator(&mut self, term: &mut Terminator<'tcx>, location: Location) {
        if let TerminatorKind::Return = term.kind {
            // `visit_basic_block_data` introduces `Return` terminators which read `RETURN_PLACE`.
            // But this `RETURN_PLACE` is already remapped, so we should not touch it again.
            return;
        }
        self.super_terminator(term, location);
    }

    #[tracing::instrument(level = "trace", skip(self, data), ret)]
    fn visit_basic_block_data(&mut self, block: BasicBlock, data: &mut BasicBlockData<'tcx>) {
        match data.terminator().kind {
            TerminatorKind::Return => {
                let source_info = data.terminator().source_info;
                // We must assign the value first in case it gets declared dead below
                self.make_state(
                    Operand::Move(Place::return_place()),
                    source_info,
                    true,
                    &mut data.statements,
                );
                // Return state.
                let state = VariantIdx::new(CoroutineArgs::RETURNED);
                data.statements.push(self.set_discr(state, source_info));
                data.terminator_mut().kind = TerminatorKind::Return;
            }
            TerminatorKind::Yield { ref value, resume, mut resume_arg, drop } => {
                let source_info = data.terminator().source_info;
                // We must assign the value first in case it gets declared dead below
                self.make_state(value.clone(), source_info, false, &mut data.statements);
                // Yield state.
                let state = CoroutineArgs::RESERVED_VARIANTS + self.suspension_points.len();

                // The resume arg target location might itself be remapped if its base local is
                // live across a yield.
                if let Some(&Some((ty, variant, idx))) = self.remap.get(resume_arg.local) {
                    replace_base(&mut resume_arg, self.make_field(variant, idx, ty), self.tcx);
                }

                let storage_liveness: GrowableBitSet<Local> =
                    self.storage_liveness[block].clone().unwrap().into();

                for i in 0..self.always_live_locals.domain_size() {
                    let l = Local::new(i);
                    let needs_storage_dead = storage_liveness.contains(l)
                        && !self.remap.contains(l)
                        && !self.always_live_locals.contains(l);
                    if needs_storage_dead {
                        data.statements
                            .push(Statement::new(source_info, StatementKind::StorageDead(l)));
                    }
                }

                self.suspension_points.push(SuspensionPoint {
                    state,
                    resume,
                    resume_arg,
                    drop,
                    storage_liveness,
                });

                let state = VariantIdx::new(state);
                data.statements.push(self.set_discr(state, source_info));
                data.terminator_mut().kind = TerminatorKind::Return;
            }
            _ => {}
        }

        self.super_basic_block_data(block, data);
    }
}

fn make_aggregate_adt<'tcx>(
    def_id: DefId,
    variant_idx: VariantIdx,
    args: GenericArgsRef<'tcx>,
    operands: IndexVec<FieldIdx, Operand<'tcx>>,
) -> Rvalue<'tcx> {
    Rvalue::Aggregate(Box::new(AggregateKind::Adt(def_id, variant_idx, args, None, None)), operands)
}

#[tracing::instrument(level = "trace", skip(tcx, body))]
fn make_coroutine_state_argument_indirect<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let coroutine_ty = body.local_decls[SELF_ARG].ty;

    let ref_coroutine_ty = Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, coroutine_ty);

    // Replace the by value coroutine argument
    body.local_decls[SELF_ARG].ty = ref_coroutine_ty;

    // Add a deref to accesses of the coroutine state
    SelfArgVisitor::new(tcx, tcx.mk_place_deref(SELF_ARG.into())).visit_body(body);
}

#[tracing::instrument(level = "trace", skip(tcx, body))]
fn make_coroutine_state_argument_pinned<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let coroutine_ty = body.local_decls[SELF_ARG].ty;

    let ref_coroutine_ty = Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, coroutine_ty);

    let pin_did = tcx.require_lang_item(LangItem::Pin, body.span);
    let pin_adt_ref = tcx.adt_def(pin_did);
    let args = tcx.mk_args(&[ref_coroutine_ty.into()]);
    let pin_ref_coroutine_ty = Ty::new_adt(tcx, pin_adt_ref, args);

    // Replace the by ref coroutine argument
    body.local_decls[SELF_ARG].ty = pin_ref_coroutine_ty;

    let unpinned_local = body.local_decls.push(LocalDecl::new(ref_coroutine_ty, body.span));

    // Add the Pin field access to accesses of the coroutine state
    SelfArgVisitor::new(tcx, tcx.mk_place_deref(unpinned_local.into())).visit_body(body);

    let source_info = SourceInfo::outermost(body.span);
    let pin_field = tcx.mk_place_field(SELF_ARG.into(), FieldIdx::ZERO, ref_coroutine_ty);

    let statements = &mut body.basic_blocks.as_mut_preserves_cfg()[START_BLOCK].statements;
    statements.insert(
        0,
        Statement::new(
            source_info,
            StatementKind::Assign(Box::new((
                unpinned_local.into(),
                Rvalue::Use(Operand::Copy(pin_field), WithRetag::Yes),
            ))),
        ),
    );
}

/// HIR uses `get_context` to unwrap a `&mut Context<'_>` from a `ResumeTy`.
/// Both types are just a single pointer, but liveness analysis does not know that and
/// supposes that the operand and the destination are live at the same time.
/// Forcibly inline those calls to avoid this.
fn eliminate_get_context_calls<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let context_mut_ref = Ty::new_task_context(tcx);
    let resume_ty_def_id = tcx.require_lang_item(LangItem::ResumeTy, body.span);
    let resume_nonnull_ty = tcx.instantiate_and_normalize_erasing_regions(
        ty::GenericArgs::empty(),
        body.typing_env(tcx),
        tcx.type_of(tcx.adt_def(resume_ty_def_id).non_enum_variant().fields[FieldIdx::ZERO].did),
    );

    let get_context_def_id = tcx.require_lang_item(LangItem::GetContext, body.span);
    for bb_data in body.basic_blocks.as_mut().iter_mut() {
        if bb_data.is_cleanup {
            continue;
        }

        let terminator = bb_data.terminator_mut();
        if let TerminatorKind::Call { func, args, destination, target, .. } = &terminator.kind
            && let func_ty = func.ty(&body.local_decls, tcx)
            && let ty::FnDef(def_id, _) = *func_ty.kind()
            && def_id == get_context_def_id
            && let [arg] = &**args
            && let Some(place) = arg.node.place()
        {
            let arg =
                Rvalue::Cast(
                    CastKind::Transmute,
                    Operand::Copy(place.project_deeper(
                        &[PlaceElem::Field(FieldIdx::ZERO, resume_nonnull_ty)],
                        tcx,
                    )),
                    context_mut_ref,
                );
            let assign = Statement::new(
                terminator.source_info,
                StatementKind::Assign(Box::new((*destination, arg))),
            );
            terminator.kind = TerminatorKind::Goto { target: target.unwrap() };
            bb_data.statements.push(assign);
        }
    }
}

/// Replaces the entry point of `body` with a block that switches on the coroutine discriminant and
/// dispatches to blocks according to `cases`.
///
/// After this function, the former entry point of the function will be the last block.
fn insert_switch<'tcx>(
    body: &mut Body<'tcx>,
    cases: Vec<(usize, BasicBlock)>,
    transform: &TransformVisitor<'tcx>,
    default_block: BasicBlock,
) {
    let (assign, discr) = transform.get_discr(body);

    // MIR validation ensures that no block targets `ENTRY_BLOCK`.
    #[cfg(debug_assertions)]
    for bb in body.basic_blocks.iter() {
        for target in bb.terminator().successors() {
            assert_ne!(target, START_BLOCK);
        }
    }

    // Add the switch as entry block, and put the former entry block at the end.
    let former_entry = std::mem::replace(
        &mut body.basic_blocks_mut()[START_BLOCK],
        BasicBlockData::new_stmts(vec![assign], None, false),
    );
    let former_entry = body.basic_blocks_mut().push(former_entry);

    // We may point to `START_BLOCK` in our `cases`, replace it with `former_entry`.
    let mut switch_targets =
        SwitchTargets::new(cases.iter().map(|(i, bb)| ((*i) as u128, *bb)), default_block);
    for bb in switch_targets.all_targets_mut() {
        if *bb == START_BLOCK {
            *bb = former_entry;
        }
    }

    let switch = TerminatorKind::SwitchInt { discr: Operand::Move(discr), targets: switch_targets };
    body.basic_blocks_mut()[START_BLOCK].terminator =
        Some(Terminator { source_info: SourceInfo::outermost(body.span), kind: switch });
}

fn insert_term_block<'tcx>(body: &mut Body<'tcx>, kind: TerminatorKind<'tcx>) -> BasicBlock {
    let source_info = SourceInfo::outermost(body.span);
    body.basic_blocks_mut().push(BasicBlockData::new(Some(Terminator { source_info, kind }), false))
}

fn return_poll_ready_assign<'tcx>(tcx: TyCtxt<'tcx>, source_info: SourceInfo) -> Statement<'tcx> {
    // Coroutine::Complete(const ())
    let coroutine_state_def_id = tcx.require_lang_item(LangItem::CoroutineState, source_info.span);
    let args = tcx.mk_args(&[tcx.types.unit.into(), tcx.types.unit.into()]);
    let val = Operand::Constant(Box::new(ConstOperand {
        span: source_info.span,
        user_ty: None,
        const_: Const::zero_sized(tcx.types.unit),
    }));
    let rvalue =
        make_aggregate_adt(coroutine_state_def_id, VariantIdx::from_usize(1), args, indexvec![val]);
    Statement::new(source_info, StatementKind::Assign(Box::new((Place::return_place(), rvalue))))
}

fn insert_poll_ready_block<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) -> BasicBlock {
    let source_info = SourceInfo::outermost(body.span);
    body.basic_blocks_mut().push(BasicBlockData::new_stmts(
        [return_poll_ready_assign(tcx, source_info)].to_vec(),
        Some(Terminator { source_info, kind: TerminatorKind::Return }),
        false,
    ))
}

fn insert_panic_block<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    message: AssertMessage<'tcx>,
) -> BasicBlock {
    let assert_block = body.basic_blocks.next_index();
    let kind = TerminatorKind::Assert {
        cond: Operand::Constant(Box::new(ConstOperand {
            span: body.span,
            user_ty: None,
            const_: Const::from_bool(tcx, false),
        })),
        expected: true,
        msg: Box::new(message),
        target: assert_block,
        unwind: UnwindAction::Continue,
    };

    insert_term_block(body, kind)
}

fn can_return<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, typing_env: ty::TypingEnv<'tcx>) -> bool {
    // Returning from a function with an uninhabited return type is undefined behavior.
    if body.return_ty().is_privately_uninhabited(tcx, typing_env) {
        return false;
    }

    // If there's a return terminator the function may return.
    body.basic_blocks.iter().any(|block| matches!(block.terminator().kind, TerminatorKind::Return))
    // Otherwise the function can't return.
}

fn can_unwind<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> bool {
    // Nothing can unwind when landing pads are off.
    if !tcx.sess.panic_strategy().unwinds() {
        return false;
    }

    // If we don't find an unwinding terminator, the function cannot unwind.
    body.basic_blocks.iter().any(|block| block.terminator().unwind().is_some())
}

// Poison the coroutine when it unwinds
fn generate_poison_block_and_redirect_unwinds_there<'tcx>(
    transform: &TransformVisitor<'tcx>,
    body: &mut Body<'tcx>,
) {
    let source_info = SourceInfo::outermost(body.span);
    let poison_block = body.basic_blocks_mut().push(BasicBlockData::new_stmts(
        vec![transform.set_discr(VariantIdx::new(CoroutineArgs::POISONED), source_info)],
        Some(Terminator { source_info, kind: TerminatorKind::UnwindResume }),
        true,
    ));

    for (idx, block) in body.basic_blocks_mut().iter_enumerated_mut() {
        let source_info = block.terminator().source_info;

        if let TerminatorKind::UnwindResume = block.terminator().kind {
            // An existing `Resume` terminator is redirected to jump to our dedicated
            // "poisoning block" above.
            if idx != poison_block {
                *block.terminator_mut() =
                    Terminator { source_info, kind: TerminatorKind::Goto { target: poison_block } };
            }
        } else if !block.is_cleanup
            // Any terminators that *can* unwind but don't have an unwind target set are also
            // pointed at our poisoning block (unless they're part of the cleanup path).
            && let Some(unwind @ UnwindAction::Continue) = block.terminator_mut().unwind_mut()
        {
            *unwind = UnwindAction::Cleanup(poison_block);
        }
    }
}

#[tracing::instrument(level = "trace", skip(tcx, transform, body))]
fn create_coroutine_resume_function<'tcx>(
    tcx: TyCtxt<'tcx>,
    transform: TransformVisitor<'tcx>,
    body: &mut Body<'tcx>,
    can_return: bool,
    can_unwind: bool,
) {
    // Poison the coroutine when it unwinds
    if can_unwind {
        generate_poison_block_and_redirect_unwinds_there(&transform, body);
    }

    let mut cases = create_cases(body, &transform, Operation::Resume);

    use rustc_middle::mir::AssertKind::{ResumedAfterPanic, ResumedAfterReturn};

    // Jump to the entry point on the unresumed
    cases.insert(0, (CoroutineArgs::UNRESUMED, START_BLOCK));

    // Panic when resumed on the returned or poisoned state
    if can_unwind {
        cases.insert(
            1,
            (
                CoroutineArgs::POISONED,
                insert_panic_block(tcx, body, ResumedAfterPanic(transform.coroutine_kind)),
            ),
        );
    }

    if can_return {
        let block = match transform.coroutine_kind {
            CoroutineKind::Desugared(CoroutineDesugaring::Async, _)
            | CoroutineKind::Coroutine(_) => {
                // For `async_drop_in_place<T>::{closure}` we just keep return Poll::Ready,
                // because async drop of such coroutine keeps polling original coroutine
                if tcx.is_async_drop_in_place_coroutine(body.source.def_id()) {
                    insert_poll_ready_block(tcx, body)
                } else {
                    insert_panic_block(tcx, body, ResumedAfterReturn(transform.coroutine_kind))
                }
            }
            CoroutineKind::Desugared(CoroutineDesugaring::AsyncGen, _)
            | CoroutineKind::Desugared(CoroutineDesugaring::Gen, _) => {
                transform.insert_none_ret_block(body)
            }
        };
        cases.insert(1, (CoroutineArgs::RETURNED, block));
    }

    let default_block = insert_term_block(body, TerminatorKind::Unreachable);
    insert_switch(body, cases, &transform, default_block);

    make_coroutine_state_argument_pinned(tcx, body);

    // Make sure we remove dead blocks to remove
    // unrelated code from the drop part of the function
    simplify::remove_dead_blocks(body);

    pm::run_passes_no_validate(tcx, body, &[&abort_unwinding_calls::AbortUnwindingCalls], None);

    // Run derefer to fix Derefs that are not in the first place
    deref_finder(tcx, body, false);

    if let Some(dumper) = MirDumper::new(tcx, "coroutine_resume", body) {
        dumper.dump_mir(body);
    }
}

/// An operation that can be performed on a coroutine.
#[derive(PartialEq, Copy, Clone, Debug)]
enum Operation {
    Resume,
    Drop,
    AsyncDrop,
}

impl Operation {
    fn target_block(self, point: &SuspensionPoint<'_>) -> Option<BasicBlock> {
        match self {
            Operation::Resume => Some(point.resume),
            Operation::Drop | Operation::AsyncDrop => point.drop,
        }
    }

    fn resume_place<'tcx>(self, point: &SuspensionPoint<'tcx>) -> Option<Place<'tcx>> {
        match self {
            Operation::Resume | Operation::AsyncDrop => Some(point.resume_arg),
            Operation::Drop => None,
        }
    }
}

#[tracing::instrument(level = "trace", skip(transform, body))]
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
                for l in body.local_decls.indices() {
                    let needs_storage_live = point.storage_liveness.contains(l)
                        && !transform.remap.contains(l)
                        && !transform.always_live_locals.contains(l);
                    if needs_storage_live {
                        statements.push(Statement::new(source_info, StatementKind::StorageLive(l)));
                    }
                }

                // Move the resume argument to the destination place of the `Yield` terminator
                if let Some(resume_arg) = operation.resume_place(point)
                    && resume_arg != CTX_ARG.into()
                {
                    statements.push(Statement::new(
                        source_info,
                        StatementKind::Assign(Box::new((
                            resume_arg,
                            Rvalue::Use(Operand::Move(CTX_ARG.into()), WithRetag::Yes),
                        ))),
                    ));
                }

                // Then jump to the real target
                let block = body.basic_blocks_mut().push(BasicBlockData::new_stmts(
                    statements,
                    Some(Terminator { source_info, kind: TerminatorKind::Goto { target } }),
                    false,
                ));

                (point.state, block)
            })
        })
        .collect()
}

impl<'tcx> crate::MirPass<'tcx> for StateTransform {
    #[instrument(level = "debug", skip(self, tcx, body), ret)]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        debug!(def_id = ?body.source.def_id());

        let Some(old_yield_ty) = body.yield_ty() else {
            // This only applies to coroutines
            return;
        };
        tracing::trace!(def_id = ?body.source.def_id());

        let old_ret_ty = body.return_ty();

        assert!(body.coroutine_drop().is_none() && body.coroutine_drop_async().is_none());

        if let Some(dumper) = MirDumper::new(tcx, "coroutine_before", body) {
            dumper.dump_mir(body);
        }

        // The first argument is the coroutine type passed by value
        let coroutine_ty = body.local_decls.raw[1].ty;
        let coroutine_kind = body.coroutine_kind().unwrap();

        // Get the discriminant type and args which typeck computed
        let ty::Coroutine(_, args) = coroutine_ty.kind() else {
            tcx.dcx().span_bug(body.span, format!("unexpected coroutine type {coroutine_ty}"));
        };
        let discr_ty = args.as_coroutine().discr_ty(tcx);

        // Compute CoroutineState<yield_ty, return_ty>
        let state_did = tcx.require_lang_item(LangItem::CoroutineState, body.span);
        let state_adt_ref = tcx.adt_def(state_did);
        let state_args = tcx.mk_args(&[old_yield_ty.into(), old_ret_ty.into()]);
        let new_ret_ty = Ty::new_adt(tcx, state_adt_ref, state_args);

        // We need to insert clean drop for unresumed state and perform drop elaboration
        // (finally in open_drop_for_tuple) before async drop expansion.
        // Async drops, produced by this drop elaboration, will be expanded,
        // and corresponding futures kept in layout.
        let has_async_drops = has_async_drops(body);

        if coroutine_kind.is_async_desugaring() {
            eliminate_get_context_calls(tcx, body);
        }

        let always_live_locals = always_storage_live_locals(body);
        let movable = coroutine_kind.movability() == hir::Movability::Movable;
        let liveness_info =
            locals_live_across_suspend_points(tcx, body, &always_live_locals, movable);

        if tcx.sess.opts.unstable_opts.validate_mir {
            let mut vis = EnsureCoroutineFieldAssignmentsNeverAlias {
                assigned_local: None,
                saved_locals: &liveness_info.saved_locals,
                storage_conflicts: &liveness_info.storage_conflicts,
            };

            vis.visit_body(body);
        }

        // Extract locals which are live across suspension point into `layout`
        // `remap` gives a mapping from local indices onto coroutine struct indices
        // `storage_liveness` tells us which locals have live storage at suspension points
        let (remap, layout, storage_liveness) = compute_layout(liveness_info, body);

        let can_return = can_return(tcx, body, body.typing_env(tcx));

        // We rename RETURN_PLACE which has type mir.return_ty to new_ret_local
        // RETURN_PLACE then is a fresh unused local with type ret_ty.
        let new_ret_local = body.local_decls.push(LocalDecl::new(new_ret_ty, body.span));
        tracing::trace!(?new_ret_local);

        // Run the transformation which converts Places from Local to coroutine struct
        // accesses for locals in `remap`.
        // It also rewrites `return x` and `yield y` as writing a new coroutine state and returning
        // either `CoroutineState::Complete(x)` and `CoroutineState::Yielded(y)`,
        // or `Poll::Ready(x)` and `Poll::Pending` respectively depending on the coroutine kind.
        let mut transform = TransformVisitor {
            tcx,
            coroutine_kind,
            remap,
            storage_liveness,
            always_live_locals,
            suspension_points: Vec::new(),
            discr_ty,
            new_ret_local,
            old_ret_ty,
            old_yield_ty,
        };
        transform.visit_body(body);

        // Swap the actual `RETURN_PLACE` and the provisional `new_ret_local`.
        transform.replace_local(RETURN_PLACE, new_ret_local, body);

        // MIR parameters are not explicitly assigned-to when entering the MIR body.
        // If we want to save their values inside the coroutine state, we need to do so explicitly.
        let source_info = SourceInfo::outermost(body.span);
        let args_iter = body.args_iter();
        body.basic_blocks.as_mut()[START_BLOCK].statements.splice(
            0..0,
            args_iter.filter_map(|local| {
                let (ty, variant_index, idx) = transform.remap[local]?;
                let lhs = transform.make_field(variant_index, idx, ty);
                let rhs = Rvalue::Use(Operand::Move(local.into()), WithRetag::Yes);
                let assign = StatementKind::Assign(Box::new((lhs, rhs)));
                Some(Statement::new(source_info, assign))
            }),
        );

        body.coroutine.as_mut().unwrap().yield_ty = None;
        body.coroutine.as_mut().unwrap().resume_ty = None;
        body.coroutine.as_mut().unwrap().coroutine_layout = Some(layout);

        // Insert `drop(coroutine_struct)` which is used to drop upvars for coroutines in
        // the unresumed state.
        // This is expanded to a drop ladder in `elaborate_coroutine_drops`.
        let drop_clean = insert_clean_drop(tcx, body, has_async_drops);

        if let Some(dumper) = MirDumper::new(tcx, "coroutine_pre-elab", body) {
            dumper.dump_mir(body);
        }

        // Expand `drop(coroutine_struct)` to a drop ladder which destroys upvars.
        // If any upvars are moved out of, drop elaboration will handle upvar destruction.
        // However we need to also elaborate the code generated by `insert_clean_drop`.
        elaborate_coroutine_drops(tcx, body);

        if let Some(dumper) = MirDumper::new(tcx, "coroutine_post-transform", body) {
            dumper.dump_mir(body);
        }

        let can_unwind = can_unwind(tcx, body);

        // Create a copy of our MIR and use it to create the drop shim for the coroutine
        if has_async_drops {
            // If coroutine has async drops, generating async drop shim
            let drop_shim =
                create_coroutine_drop_shim_async(tcx, &transform, body, drop_clean, can_unwind);
            body.coroutine.as_mut().unwrap().coroutine_drop_async = Some(drop_shim);
        } else {
            // If coroutine has no async drops, generating sync drop shim
            let drop_shim =
                create_coroutine_drop_shim(tcx, &transform, coroutine_ty, body, drop_clean);
            body.coroutine.as_mut().unwrap().coroutine_drop = Some(drop_shim);

            // For coroutine with sync drop, generating async proxy for `future_drop_poll` call
            let proxy_shim = create_coroutine_drop_shim_proxy_async(tcx, body);
            body.coroutine.as_mut().unwrap().coroutine_drop_proxy_async = Some(proxy_shim);
        }

        // Create the Coroutine::resume / Future::poll function
        create_coroutine_resume_function(tcx, transform, body, can_return, can_unwind);
    }

    fn is_required(&self) -> bool {
        true
    }
}

/// Looks for any assignments between locals (e.g., `_4 = _5`) that will both be converted to fields
/// in the coroutine state machine but whose storage is not marked as conflicting
///
/// Validation needs to happen immediately *before* `TransformVisitor` is invoked, not after.
///
/// This condition would arise when the assignment is the last use of `_5` but the initial
/// definition of `_4` if we weren't extra careful to mark all locals used inside a statement as
/// conflicting. Non-conflicting coroutine saved locals may be stored at the same location within
/// the coroutine state machine, which would result in ill-formed MIR: the left-hand and right-hand
/// sides of an assignment may not alias. This caused a miscompilation in [#73137].
///
/// [#73137]: https://github.com/rust-lang/rust/issues/73137
struct EnsureCoroutineFieldAssignmentsNeverAlias<'a> {
    saved_locals: &'a CoroutineSavedLocals,
    storage_conflicts: &'a BitMatrix<CoroutineSavedLocal, CoroutineSavedLocal>,
    assigned_local: Option<CoroutineSavedLocal>,
}

impl EnsureCoroutineFieldAssignmentsNeverAlias<'_> {
    fn saved_local_for_direct_place(&self, place: Place<'_>) -> Option<CoroutineSavedLocal> {
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

impl<'tcx> Visitor<'tcx> for EnsureCoroutineFieldAssignmentsNeverAlias<'_> {
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
                "Assignment between coroutine saved locals whose storage is not \
                    marked as conflicting: {:?}: {:?} = {:?}",
                location,
                lhs,
                rhs,
            );
        }
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match &statement.kind {
            StatementKind::Assign((lhs, rhs)) => {
                self.check_assigned_place(*lhs, |this| this.visit_rvalue(rhs, location));
            }

            StatementKind::FakeRead(..)
            | StatementKind::SetDiscriminant { .. }
            | StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::AscribeUserType(..)
            | StatementKind::PlaceMention(..)
            | StatementKind::Coverage(..)
            | StatementKind::Intrinsic(..)
            | StatementKind::ConstEvalCounter
            | StatementKind::BackwardIncompatibleDropHint { .. }
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
                call_source: _,
                fn_span: _,
            } => {
                self.check_assigned_place(*destination, |this| {
                    this.visit_operand(func, location);
                    for arg in args {
                        this.visit_operand(&arg.node, location);
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
            | TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Return
            | TerminatorKind::TailCall { .. }
            | TerminatorKind::Unreachable
            | TerminatorKind::Drop { .. }
            | TerminatorKind::Assert { .. }
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. } => {}
        }
    }
}
