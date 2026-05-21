use std::{fmt, iter, mem};

use rustc_abi::{FIRST_VARIANT, FieldIdx, VariantIdx};
use rustc_data_structures::thin_vec::ThinVec;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{CoroutineDesugaring, CoroutineKind};
use rustc_index::Idx;
use rustc_middle::mir::*;
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::util::{Discr, IntTypeExt};
use rustc_middle::ty::{self, GenericArgsRef, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_span::{DUMMY_SP, dummy_spanned};
use tracing::{debug, instrument};

use crate::coroutine::CTX_ARG;
use crate::patch::MirPatch;

/// Describes how/if a value should be dropped.
#[derive(Debug)]
pub(crate) enum DropStyle {
    /// The value is already dead at the drop location, no drop will be executed.
    Dead,

    /// The value is known to always be initialized at the drop location, drop will always be
    /// executed.
    Static,

    /// Whether the value needs to be dropped depends on its drop flag.
    Conditional,

    /// An "open" drop is one where only the fields of a value are dropped.
    ///
    /// For example, this happens when moving out of a struct field: The rest of the struct will be
    /// dropped in such an "open" drop. It is also used to generate drop glue for the individual
    /// components of a value, for example for dropping array elements.
    Open,
}

/// Which drop flags to affect/check with an operation.
#[derive(Debug)]
pub(crate) enum DropFlagMode {
    /// Only affect the top-level drop flag, not that of any contained fields.
    Shallow,
    /// Affect all nested drop flags in addition to the top-level one.
    Deep,
}

/// Describes if unwinding is necessary and where to unwind to if a panic occurs.
#[derive(Copy, Clone, Debug)]
pub(crate) enum Unwind {
    /// Unwind to this block.
    To(BasicBlock),
    /// Already in an unwind path, any panic will cause an abort.
    InCleanup,
}

impl Unwind {
    fn is_cleanup(self) -> bool {
        match self {
            Unwind::To(..) => false,
            Unwind::InCleanup => true,
        }
    }

    fn into_action(self) -> UnwindAction {
        match self {
            Unwind::To(bb) => UnwindAction::Cleanup(bb),
            Unwind::InCleanup => UnwindAction::Terminate(UnwindTerminateReason::InCleanup),
        }
    }

    fn map<F>(self, f: F) -> Self
    where
        F: FnOnce(BasicBlock) -> BasicBlock,
    {
        match self {
            Unwind::To(bb) => Unwind::To(f(bb)),
            Unwind::InCleanup => Unwind::InCleanup,
        }
    }
}

pub(crate) trait DropElaborator<'a, 'tcx>: fmt::Debug {
    /// The type representing paths that can be moved out of.
    ///
    /// Users can move out of individual fields of a struct, such as `a.b.c`. This type is used to
    /// represent such move paths. Sometimes tracking individual move paths is not necessary, in
    /// which case this may be set to (for example) `()`.
    type Path: Copy + fmt::Debug;

    // Accessors

    fn patch_ref(&self) -> &MirPatch<'tcx>;
    fn patch(&mut self) -> &mut MirPatch<'tcx>;
    fn body(&self) -> &'a Body<'tcx>;
    fn tcx(&self) -> TyCtxt<'tcx>;
    fn typing_env(&self) -> ty::TypingEnv<'tcx>;
    fn allow_async_drops(&self) -> bool;

    // Drop logic

    /// Returns how `path` should be dropped, given `mode`.
    fn drop_style(&self, path: Self::Path, mode: DropFlagMode) -> DropStyle;

    /// Returns the drop flag of `path` as a MIR `Operand` (or `None` if `path` has no drop flag).
    fn get_drop_flag(&mut self, path: Self::Path) -> Option<Operand<'tcx>>;

    /// Modifies the MIR patch so that the drop flag of `path` (if any) is cleared at `location`.
    ///
    /// If `mode` is deep, drop flags of all child paths should also be cleared by inserting
    /// additional statements.
    fn clear_drop_flag(&mut self, location: Location, path: Self::Path, mode: DropFlagMode);

    // Subpaths

    /// Returns the subpath of a field of `path` (or `None` if there is no dedicated subpath).
    ///
    /// If this returns `None`, `field` will not get a dedicated drop flag.
    fn field_subpath(&self, path: Self::Path, field: FieldIdx) -> Option<Self::Path>;

    /// Returns the subpath of a dereference of `path` (or `None` if there is no dedicated subpath).
    ///
    /// If this returns `None`, `*path` will not get a dedicated drop flag.
    ///
    /// This is only relevant for `Box<T>`, where the contained `T` can be moved out of the box.
    fn deref_subpath(&self, path: Self::Path) -> Option<Self::Path>;

    /// Returns the subpath of downcasting `path` to one of its variants.
    ///
    /// If this returns `None`, the downcast of `path` will not get a dedicated drop flag.
    fn downcast_subpath(&self, path: Self::Path, variant: VariantIdx) -> Option<Self::Path>;

    /// Returns the subpath of indexing a fixed-size array `path`.
    ///
    /// If this returns `None`, elements of `path` will not get a dedicated drop flag.
    ///
    /// This is only relevant for array patterns, which can move out of individual array elements.
    fn array_subpath(&self, path: Self::Path, index: u64, size: u64) -> Option<Self::Path>;
}

#[derive(Debug)]
struct DropCtxt<'a, 'b, 'tcx, D>
where
    D: DropElaborator<'b, 'tcx>,
{
    elaborator: &'a mut D,

    source_info: SourceInfo,

    place: Place<'tcx>,
    path: D::Path,
    succ: BasicBlock,
    unwind: Unwind,
    dropline: Option<BasicBlock>,
}

/// "Elaborates" a drop of `place`/`path` and patches `bb`'s terminator to execute it.
///
/// The passed `elaborator` is used to determine what should happen at the drop terminator. It
/// decides whether the drop can be statically determined or whether it needs a dynamic drop flag,
/// and whether the drop is "open", i.e. should be expanded to drop all subfields of the dropped
/// value.
///
/// When this returns, the MIR patch in the `elaborator` contains the necessary changes.
pub(crate) fn elaborate_drop<'b, 'tcx, D>(
    elaborator: &mut D,
    source_info: SourceInfo,
    place: Place<'tcx>,
    path: D::Path,
    succ: BasicBlock,
    unwind: Unwind,
    bb: BasicBlock,
    dropline: Option<BasicBlock>,
) where
    D: DropElaborator<'b, 'tcx>,
    'tcx: 'b,
{
    DropCtxt { elaborator, source_info, place, path, succ, unwind, dropline }.elaborate_drop(bb)
}

impl<'a, 'b, 'tcx, D> DropCtxt<'a, 'b, 'tcx, D>
where
    D: DropElaborator<'b, 'tcx>,
    'tcx: 'b,
{
    #[instrument(level = "trace", skip(self), ret)]
    fn place_ty(&self, place: Place<'tcx>) -> Ty<'tcx> {
        if place.local < self.elaborator.body().local_decls.next_index() {
            place.ty(self.elaborator.body(), self.tcx()).ty
        } else {
            // We don't have a slice with all the locals, since some are in the patch.
            PlaceTy::from_ty(self.elaborator.patch_ref().local_ty(place.local))
                .multi_projection_ty(self.elaborator.tcx(), place.projection)
                .ty
        }
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.elaborator.tcx()
    }

    /// Async-drop `place: drop_ty`.
    ///
    /// Conceptually, we want to run `async_drop_in_place(&mut obj).await`.
    ///
    /// Await syntax does not exist in MIR, so we need to manually expand it into a poll-yield
    /// loop, essentially:
    /// ```mir
    ///   let fut = async_drop_in_place(&mut obj);
    ///   loop {
    ///     let pin_fut = Pin::new_unchecked(&mut fut);
    ///     match Future::poll(pin_fut, CTX_ARG) {
    ///       Poll::Ready => break,
    ///       Poll::Pending(..) => CTX_ARG = yield (),
    ///     }
    ///   }
    ///   // continue to `succ`
    /// ```
    ///
    /// We also need to ensure that async drop also happens on the coroutine drop path, ie. when
    /// `yield` branches along its `drop` target. This requires a second loop, this time jumping to
    /// `dropline`.
    ///
    /// Arguments:
    ///   `call_destructor_only`: call only `AsyncDrop::drop`, not full `async_drop_in_place` glue
    #[instrument(level = "debug", skip(self), ret)]
    fn build_async_drop(
        &mut self,
        place: Place<'tcx>,
        drop_ty: Ty<'tcx>,
        succ: BasicBlock,
        unwind: Unwind,
        dropline: Option<BasicBlock>,
        call_destructor_only: bool,
    ) -> BasicBlock {
        let tcx = self.tcx();
        let span = self.source_info.span;
        let obj_ref_ty = Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, drop_ty);

        let async_drop_fn_def_id = if call_destructor_only {
            // Resolving obj.<AsyncDrop::drop>()
            let async_drop_trait = tcx.require_lang_item(LangItem::AsyncDrop, span);
            tcx.associated_item_def_ids(async_drop_trait)[0]
        } else {
            // Resolving async_drop_in_place<T> function for drop_ty
            tcx.require_lang_item(LangItem::AsyncDropInPlace, span)
        };

        let fut_ty = tcx
            .instantiate_bound_regions_with_erased(
                Ty::new_fn_def(tcx, async_drop_fn_def_id, [drop_ty]).fn_sig(tcx),
            )
            .output();
        let fut = self.new_temp(fut_ty);

        // Create an intermediate block that does StorageDead(fut) then jumps to succ.
        // This is necessary because we do not want to modify statements
        // in existing blocks, in case those are used somewhere else in MIR.
        let succ_with_dead = self.new_block_with_statements(
            unwind,
            vec![self.storage_dead(fut)],
            TerminatorKind::Goto { target: succ },
        );
        let dropline_with_dead = dropline.map(|target| {
            self.new_block_with_statements(
                unwind,
                vec![self.storage_dead(fut)],
                TerminatorKind::Goto { target },
            )
        });
        let unwind_with_dead = unwind.map(|target| {
            self.new_block_with_statements(
                Unwind::InCleanup,
                vec![self.storage_dead(fut)],
                TerminatorKind::Goto { target },
            )
        });

        // The yielded value depends on the kind of coroutine, to match what AST lowering does.
        let coroutine_kind = self.elaborator.body().coroutine_kind().unwrap();
        let yield_value = match coroutine_kind {
            // For async gen, we need `yield Poll<OptRet>::Pending`.
            CoroutineKind::Desugared(CoroutineDesugaring::AsyncGen, _) => {
                let full_yield_ty = self.elaborator.body().yield_ty().unwrap();
                let ty::Adt(_poll_adt, args) = *full_yield_ty.kind() else { bug!() };
                let ty::Adt(_option_adt, args) = *args.type_at(0).kind() else { bug!() };
                let yield_ty = args.type_at(0);
                Operand::unevaluated_constant(
                    tcx,
                    tcx.require_lang_item(LangItem::AsyncGenPending, span),
                    tcx.mk_args(&[yield_ty.into()]),
                    span,
                )
            }
            // For regular async fn, we need `yield ()`.
            CoroutineKind::Desugared(CoroutineDesugaring::Async, _) => {
                Operand::zero_sized_constant(tcx.types.unit, span)
            }
            // `is_async_drop` should have checked that.
            _ => panic!("unexpected coroutine for async drop {coroutine_kind:?}"),
        };

        // The branching here is tricky and deserves some explanation.
        //
        // If we are in the drop code path, ie. we are currently dropping the coroutine.
        // The state machine follows the `drop` branch in the `yield` terminator.
        // To repeatedly poll the future, the `drop` branch must loop.
        // Meanwhile, the `resume` branch corresponds to anomalous execution,
        // trying to resume the coroutine while it is being dropped. So that branch panics
        // (`panic_bb`).
        let panic_bb = self.build_resumed_after_drop_abort_block(unwind_with_dead, coroutine_kind);
        let (drop_pin_bb, drop_resume_bb, drop_drop_bb) = self.build_pin_poll_yield_loop(
            CTX_ARG.into(),
            fut.into(),
            yield_value.clone(),
            // If `dropline_with_dead` is set, it points to the continuation of the drop execution.
            // Otherwise, we are already dropping the coroutine, and `succ_with_dead` does.
            dropline_with_dead.unwrap_or(succ_with_dead),
            unwind_with_dead,
        );
        self.elaborator
            .patch()
            .patch_terminator(drop_resume_bb, TerminatorKind::Goto { target: panic_bb });
        self.elaborator
            .patch()
            .patch_terminator(drop_drop_bb, TerminatorKind::Goto { target: drop_pin_bb });

        // If we are in the regular code path, `dropline_with_dead` is `Some`.
        //
        // In that case, the logic is reversed. Normal execution branches on `resume` from the
        // `yield` terminator. To repeatedly poll the future, that `resume` branch must loop.
        // When the future is dropped, the `yield` terminator branches to `drop`, which follows to
        // the previous loop `drop_pin_bb`.
        let succ_yield_loop = if dropline_with_dead.is_some() {
            let (pin_bb, resume_bb, drop_bb) = self.build_pin_poll_yield_loop(
                CTX_ARG.into(),
                fut.into(),
                yield_value,
                // `dropline_with_dead` is `Some`, so the previous loop point to it.
                succ_with_dead,
                unwind_with_dead,
            );
            self.elaborator
                .patch()
                .patch_terminator(resume_bb, TerminatorKind::Goto { target: pin_bb });
            self.elaborator
                .patch()
                .patch_terminator(drop_bb, TerminatorKind::Goto { target: drop_pin_bb });
            pin_bb
        } else {
            // We were already in the drop line, so return the loop we created for it.
            drop_pin_bb
        };

        // #2:call_drop_bb >>>
        //    call AsyncDrop::drop(pin_obj)
        // OR call async_drop_in_place(pin_obj.pointer)
        let pin_adt_def = tcx.adt_def(tcx.require_lang_item(LangItem::Pin, span));
        let pin_obj_ty = Ty::new_adt(tcx, pin_adt_def, tcx.mk_args(&[obj_ref_ty.into()]));
        // Where we store the result of Pin<&drop_ty>::new_unchecked(&mut place).
        let pin_obj_local = self.new_temp(pin_obj_ty);
        let drop_arg = if call_destructor_only {
            // `AsyncDrop::drop` takes `self: Pin<&mut Self>`.
            Operand::Move(pin_obj_local.into())
        } else {
            // `async_drop_in_place` takes `obj: &mut T`.
            Operand::Copy(tcx.mk_place_field(pin_obj_local.into(), FieldIdx::ZERO, obj_ref_ty))
        };
        let call_drop_bb = self.new_block_with_statements(
            unwind_with_dead,
            vec![self.storage_live(fut)],
            TerminatorKind::Call {
                func: Operand::function_handle(tcx, async_drop_fn_def_id, [drop_ty.into()], span),
                args: [dummy_spanned(drop_arg)].into(),
                destination: fut.into(),
                target: Some(succ_yield_loop),
                unwind: unwind_with_dead.into_action(),
                call_source: CallSource::Misc,
                fn_span: self.source_info.span,
            },
        );

        // #1:pin_obj_bb >>> call Pin<ObjTy>::new_unchecked(&mut obj)
        let obj_ref_place = Place::from(self.new_temp(obj_ref_ty));
        let pin_obj_new_unchecked_fn = tcx.require_lang_item(LangItem::PinNewUnchecked, span);
        let assign_obj_ref_place = self.assign(
            obj_ref_place,
            Rvalue::Ref(
                tcx.lifetimes.re_erased,
                BorrowKind::Mut { kind: MutBorrowKind::Default },
                place,
            ),
        );
        self.new_block_with_statements(
            unwind,
            vec![assign_obj_ref_place],
            TerminatorKind::Call {
                func: Operand::function_handle(
                    tcx,
                    pin_obj_new_unchecked_fn,
                    [obj_ref_ty.into()],
                    span,
                ),
                args: [dummy_spanned(Operand::Move(obj_ref_place))].into(),
                destination: pin_obj_local.into(),
                target: Some(call_drop_bb),
                unwind: unwind.into_action(),
                call_source: CallSource::Misc,
                fn_span: span,
            },
        )
    }

    fn build_resumed_after_drop_abort_block(
        &mut self,
        unwind: Unwind,
        coroutine_kind: CoroutineKind,
    ) -> BasicBlock {
        let tcx = self.tcx();
        let panic_bb = self.new_block(unwind, TerminatorKind::Unreachable);
        let msg = AssertMessage::ResumedAfterDrop(coroutine_kind);
        let false_op = Operand::Constant(Box::new(ConstOperand {
            span: self.source_info.span,
            user_ty: None,
            const_: Const::from_bool(tcx, false),
        }));
        self.elaborator.patch().patch_terminator(
            panic_bb,
            TerminatorKind::Assert {
                cond: false_op,
                expected: true,
                msg: Box::new(msg),
                target: panic_bb,
                unwind: unwind.into_action(),
            },
        );
        panic_bb
    }

    /// Build a small MIR loop that pins and polls a future, yielding when
    /// the future returns `Poll::Pending` and continuing to `ready_target`
    /// when it returns `Poll::Ready`.
    ///
    /// Pseudo-code:
    /// ```mir
    /// pin_bb:
    ///   let pin_fut = Pin::new_unchecked(&mut fut_place);
    ///   match Future::poll(pin_fut, CTX_ARG) {
    ///     Poll::Ready => goto succ,
    ///     Poll::Pending(..) => CTX_ARG = yield () [resume: resume_bb, drop: drop_bb],
    ///   }
    /// ```
    ///
    ///  Returns: the tuple `(pin_bb, resume_bb, drop_bb)`.
    #[instrument(level = "trace", skip(self), ret)]
    fn build_pin_poll_yield_loop(
        &mut self,
        resume_place: Place<'tcx>,
        fut_place: Place<'tcx>,
        yield_value: Operand<'tcx>,
        succ: BasicBlock,
        unwind: Unwind,
    ) -> (BasicBlock, BasicBlock, BasicBlock) {
        let tcx = self.tcx();
        let source_info = self.source_info;

        let resume_arg_ty = resume_place.ty(self.elaborator.body(), tcx).ty;
        let context_ref_ty = Ty::new_task_context(tcx);

        let poll_adt_def = tcx.adt_def(tcx.require_lang_item(LangItem::Poll, source_info.span));
        let poll_enum = Ty::new_adt(tcx, poll_adt_def, tcx.mk_args(&[tcx.types.unit.into()]));

        let fut_ty = self.elaborator.patch_ref().local_ty(fut_place.local);
        let fut_ref_ty = Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, fut_ty);

        let pin_adt_def = tcx.adt_def(tcx.require_lang_item(LangItem::Pin, source_info.span));
        let fut_pin_ty = Ty::new_adt(tcx, pin_adt_def, tcx.mk_args(&[fut_ref_ty.into()]));

        // Coroutine `transform_async_context` assumes that the local `resume_arg` to a yield
        // is not used once, so create a special temp for it.
        let yield_resume_local = self.new_temp(resume_arg_ty);
        let resume_bb = self.new_block_with_statements(
            unwind,
            vec![
                self.assign(
                    resume_place,
                    Rvalue::Use(Operand::Move(yield_resume_local.into()), WithRetag::Yes),
                ),
                self.storage_dead(yield_resume_local),
            ],
            // This will be transformed by the caller.
            TerminatorKind::Unreachable,
        );
        let dropline_bb = self.new_block_with_statements(
            unwind,
            vec![
                self.assign(
                    resume_place,
                    Rvalue::Use(Operand::Move(yield_resume_local.into()), WithRetag::Yes),
                ),
                self.storage_dead(yield_resume_local),
            ],
            // This will be transformed by the caller.
            TerminatorKind::Unreachable,
        );
        let yield_bb = self.new_block_with_statements(
            unwind,
            vec![self.storage_live(yield_resume_local)],
            TerminatorKind::Yield {
                value: yield_value,
                resume: resume_bb,
                resume_arg: yield_resume_local.into(),
                drop: Some(dropline_bb),
            },
        );

        let poll_unit_local = self.new_temp(poll_enum);
        let switch_bb = {
            let poll_ready_variant =
                tcx.require_lang_item(LangItem::PollReady, self.source_info.span);
            let poll_ready_variant_idx = poll_adt_def.variant_index_with_id(poll_ready_variant);
            let poll_pending_variant =
                tcx.require_lang_item(LangItem::PollPending, self.source_info.span);
            let poll_pending_variant_idx = poll_adt_def.variant_index_with_id(poll_pending_variant);

            let Discr { val: poll_ready_discr, ty: poll_discr_ty } =
                poll_enum.discriminant_for_variant(tcx, poll_ready_variant_idx).unwrap();
            let Discr { val: poll_pending_discr, ty: _ } =
                poll_enum.discriminant_for_variant(tcx, poll_pending_variant_idx).unwrap();

            let poll_discr_local = self.new_temp(poll_discr_ty);
            let otherwise_bb = self.elaborator.patch().unreachable_no_cleanup_block();
            self.new_block_with_statements(
                unwind,
                vec![
                    self.assign(
                        poll_discr_local.into(),
                        Rvalue::Discriminant(poll_unit_local.into()),
                    ),
                ],
                TerminatorKind::SwitchInt {
                    discr: Operand::Move(poll_discr_local.into()),
                    targets: SwitchTargets::new(
                        [
                            // on `Ready`, exit the loop, jump to `succ`
                            (poll_ready_discr, succ),
                            // on `Pending`, yield and resume back into the loop
                            (poll_pending_discr, yield_bb),
                        ]
                        .into_iter(),
                        // otherwise: unreachable
                        otherwise_bb,
                    ),
                },
            )
        };

        let fut_pin_local = self.new_temp(fut_pin_ty);
        let context_ref_local = self.new_temp(context_ref_ty);

        let poll_fn = tcx.require_lang_item(LangItem::FuturePoll, source_info.span);
        let poll_bb = self.new_block_with_statements(
            unwind,
            Vec::new(),
            TerminatorKind::Call {
                func: Operand::function_handle(tcx, poll_fn, [fut_ty.into()], source_info.span),
                args: [
                    dummy_spanned(Operand::Move(fut_pin_local.into())),
                    dummy_spanned(Operand::Move(context_ref_local.into())),
                ]
                .into(),
                destination: poll_unit_local.into(),
                target: Some(switch_bb),
                unwind: unwind.into_action(),
                call_source: CallSource::Misc,
                fn_span: source_info.span,
            },
        );

        let get_context_fn = tcx.require_lang_item(LangItem::GetContext, source_info.span);
        let get_context_bb = {
            // Coroutine `transform_async_context` assumes that the local argument to `GetContext`
            // is not used once, so create a special temp for it.
            let entry_resume_local = self.new_temp(resume_arg_ty);
            self.new_block_with_statements(
                unwind,
                vec![self.assign(
                    entry_resume_local.into(),
                    Rvalue::Use(Operand::Move(resume_place), WithRetag::Yes),
                )],
                TerminatorKind::Call {
                    func: Operand::function_handle(
                        tcx,
                        get_context_fn,
                        [tcx.lifetimes.re_erased.into(), tcx.lifetimes.re_erased.into()],
                        source_info.span,
                    ),
                    args: [dummy_spanned(Operand::Move(entry_resume_local.into()))].into(),
                    destination: context_ref_local.into(),
                    target: Some(poll_bb),
                    unwind: unwind.into_action(),
                    call_source: CallSource::Misc,
                    fn_span: source_info.span,
                },
            )
        };

        let fut_ref_local = self.new_temp(fut_ref_ty);
        let fut_pin_new_unchecked_fn =
            tcx.require_lang_item(LangItem::PinNewUnchecked, source_info.span);
        let pin_bb = self.new_block_with_statements(
            unwind,
            vec![self.assign(
                fut_ref_local.into(),
                Rvalue::Ref(
                    tcx.lifetimes.re_erased,
                    BorrowKind::Mut { kind: MutBorrowKind::Default },
                    fut_place,
                ),
            )],
            TerminatorKind::Call {
                func: Operand::function_handle(
                    tcx,
                    fut_pin_new_unchecked_fn,
                    [fut_ref_ty.into()],
                    source_info.span,
                ),
                args: [dummy_spanned(Operand::Move(fut_ref_local.into()))].into(),
                destination: fut_pin_local.into(),
                target: Some(get_context_bb),
                unwind: unwind.into_action(),
                call_source: CallSource::Misc,
                fn_span: source_info.span,
            },
        );

        (pin_bb, resume_bb, dropline_bb)
    }

    fn build_drop(&mut self, bb: BasicBlock) {
        let drop_ty = self.place_ty(self.place);
        if !self.elaborator.patch_ref().block(self.elaborator.body(), bb).is_cleanup
            && self.check_if_can_async_drop(drop_ty, false)
        {
            let async_drop_bb = self.build_async_drop(
                self.place,
                drop_ty,
                self.succ,
                self.unwind,
                self.dropline,
                false,
            );
            self.elaborator
                .patch()
                .patch_terminator(bb, TerminatorKind::Goto { target: async_drop_bb });
        } else {
            self.elaborator.patch().patch_terminator(
                bb,
                TerminatorKind::Drop {
                    place: self.place,
                    target: self.succ,
                    unwind: self.unwind.into_action(),
                    replace: false,
                    drop: None,
                },
            );
        }
    }

    /// Function to check if we can generate an async drop here
    fn check_if_can_async_drop(&mut self, drop_ty: Ty<'tcx>, call_destructor_only: bool) -> bool {
        if !self.elaborator.allow_async_drops()
            || !self
                .elaborator
                .body()
                .coroutine
                .as_ref()
                .is_some_and(|ck| ck.coroutine_kind.is_async_desugaring())
        {
            return false;
        }

        if drop_ty == self.place_ty(Local::arg(0).into()) {
            return false;
        }

        let is_async_drop_feature_enabled = if self.tcx().features().async_drop() {
            true
        } else {
            // Check if the type needing async drop comes from a dependency crate.
            if let ty::Adt(adt_def, _) = drop_ty.kind() {
                !adt_def.did().is_local() && adt_def.async_destructor(self.tcx()).is_some()
            } else {
                false
            }
        };

        // Short-circuit before calling needs_async_drop/is_async_drop, as those
        // require the `async_drop` lang item to exist (which may not be present
        // in minimal/custom core environments like cranelift's mini_core).
        if !is_async_drop_feature_enabled {
            return false;
        }

        let needs_async_drop = if call_destructor_only {
            drop_ty.is_async_drop(self.tcx(), self.elaborator.typing_env())
        } else {
            drop_ty.needs_async_drop(self.tcx(), self.elaborator.typing_env())
        };

        // Async drop in libstd/libcore would become insta-stable — catch that mistake.
        if needs_async_drop && self.tcx().features().staged_api() {
            span_bug!(
                self.source_info.span,
                "don't use async drop in libstd, it becomes insta-stable"
            );
        }

        needs_async_drop
    }

    /// This elaborates a single drop instruction, located at `bb`, and
    /// patches over it.
    ///
    /// The elaborated drop checks the drop flags to only drop what
    /// is initialized.
    ///
    /// In addition, the relevant drop flags also need to be cleared
    /// to avoid double-drops. However, in the middle of a complex
    /// drop, one must avoid clearing some of the flags before they
    /// are read, as that would cause a memory leak.
    ///
    /// In particular, when dropping an ADT, multiple fields may be
    /// joined together under the `rest` subpath. They are all controlled
    /// by the primary drop flag, but only the last rest-field dropped
    /// should clear it (and it must also not clear anything else).
    //
    // FIXME: I think we should just control the flags externally,
    // and then we do not need this machinery.
    #[instrument(level = "debug")]
    fn elaborate_drop(&mut self, bb: BasicBlock) {
        match self.elaborator.drop_style(self.path, DropFlagMode::Deep) {
            DropStyle::Dead => {
                self.elaborator
                    .patch()
                    .patch_terminator(bb, TerminatorKind::Goto { target: self.succ });
            }
            DropStyle::Static => {
                self.build_drop(bb);
            }
            DropStyle::Conditional => {
                let drop_bb = self.complete_drop(self.succ, self.unwind);
                self.elaborator
                    .patch()
                    .patch_terminator(bb, TerminatorKind::Goto { target: drop_bb });
            }
            DropStyle::Open => {
                let drop_bb = self.open_drop();
                self.elaborator
                    .patch()
                    .patch_terminator(bb, TerminatorKind::Goto { target: drop_bb });
            }
        }
    }

    /// Returns the place and move path for each field of `variant`,
    /// (the move path is `None` if the field is a rest field).
    fn move_paths_for_fields(
        &self,
        base_place: Place<'tcx>,
        variant_path: D::Path,
        variant: &'tcx ty::VariantDef,
        args: GenericArgsRef<'tcx>,
    ) -> Vec<(Place<'tcx>, Option<D::Path>)> {
        variant
            .fields
            .iter_enumerated()
            .map(|(field_idx, field)| {
                let subpath = self.elaborator.field_subpath(variant_path, field_idx);
                let tcx = self.tcx();

                match self.elaborator.typing_env().typing_mode().assert_not_erased() {
                    ty::TypingMode::PostAnalysis | ty::TypingMode::Codegen => {}
                    ty::TypingMode::Coherence
                    | ty::TypingMode::Analysis { .. }
                    | ty::TypingMode::Borrowck { .. }
                    | ty::TypingMode::PostBorrowckAnalysis { .. } => {
                        bug!()
                    }
                }

                let field_ty = field.ty(tcx, args);
                // We silently leave an unnormalized type here to support polymorphic drop
                // elaboration for users of rustc internal APIs
                let field_ty = tcx
                    .try_normalize_erasing_regions(self.elaborator.typing_env(), field_ty)
                    .unwrap_or(field_ty.skip_norm_wip());

                (tcx.mk_place_field(base_place, field_idx, field_ty), subpath)
            })
            .collect()
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn drop_subpath(
        &mut self,
        place: Place<'tcx>,
        path: Option<D::Path>,
        succ: BasicBlock,
        unwind: Unwind,
        dropline: Option<BasicBlock>,
    ) -> BasicBlock {
        if let Some(path) = path {
            DropCtxt {
                elaborator: self.elaborator,
                source_info: self.source_info,
                path,
                place,
                succ,
                unwind,
                dropline,
            }
            .elaborated_drop_block()
        } else {
            DropCtxt {
                elaborator: self.elaborator,
                source_info: self.source_info,
                place,
                succ,
                unwind,
                dropline,
                // Using `self.path` here to condition the drop on our own drop flag.
                path: self.path,
            }
            .complete_drop(succ, unwind)
        }
    }

    /// Creates one-half of the drop ladder for a list of fields, and return
    /// the list of steps in it in reverse order, with the first step
    /// dropping 0 fields and so on.
    ///
    /// `unwind_ladder` is such a list of steps in reverse order,
    /// which is called if the matching step of the drop glue panics.
    ///
    /// `dropline_ladder` is a similar list of steps in reverse order,
    /// which is called if the matching step of the drop glue will contain async drop
    /// (expanded later to Yield) and the containing coroutine will be dropped at this point.
    #[instrument(level = "debug", skip(self), ret)]
    fn drop_halfladder(
        &mut self,
        unwind_ladder: &[Unwind],
        dropline_ladder: &[Option<BasicBlock>],
        mut succ: BasicBlock,
        fields: &[(Place<'tcx>, Option<D::Path>)],
    ) -> Vec<BasicBlock> {
        iter::once(succ)
            .chain(itertools::izip!(fields.iter().rev(), unwind_ladder, dropline_ladder).map(
                |(&(place, path), &unwind_succ, &dropline_to)| {
                    succ = self.drop_subpath(place, path, succ, unwind_succ, dropline_to);
                    succ
                },
            ))
            .collect()
    }

    fn drop_ladder_bottom(&mut self) -> (BasicBlock, Unwind, Option<BasicBlock>) {
        // Clear the "master" drop flag at the end. This is needed
        // because the "master" drop protects the ADT's discriminant,
        // which is invalidated after the ADT is dropped.
        (
            self.drop_flag_reset_block(DropFlagMode::Shallow, self.succ, self.unwind),
            self.unwind,
            self.dropline,
        )
    }

    /// Creates a full drop ladder, consisting of 2 connected half-drop-ladders
    ///
    /// For example, with 3 fields, the drop ladder is
    ///
    /// ```text
    /// .d0:
    ///     ELAB(drop location.0 [target=.d1, unwind=.c1])
    /// .d1:
    ///     ELAB(drop location.1 [target=.d2, unwind=.c2])
    /// .d2:
    ///     ELAB(drop location.2 [target=`self.succ`, unwind=`self.unwind`])
    /// .c1:
    ///     ELAB(drop location.1 [target=.c2])
    /// .c2:
    ///     ELAB(drop location.2 [target=`self.unwind`])
    /// ```
    ///
    /// For possible-async drops in coroutines we also need dropline ladder
    /// ```text
    /// .d0 (mainline):
    ///     ELAB(drop location.0 [target=.d1, unwind=.c1, drop=.e1])
    /// .d1 (mainline):
    ///     ELAB(drop location.1 [target=.d2, unwind=.c2, drop=.e2])
    /// .d2 (mainline):
    ///     ELAB(drop location.2 [target=`self.succ`, unwind=`self.unwind`, drop=`self.drop`])
    /// .c1 (unwind):
    ///     ELAB(drop location.1 [target=.c2])
    /// .c2 (unwind):
    ///     ELAB(drop location.2 [target=`self.unwind`])
    /// .e1 (dropline):
    ///     ELAB(drop location.1 [target=.e2, unwind=.c2])
    /// .e2 (dropline):
    ///     ELAB(drop location.2 [target=`self.drop`, unwind=`self.unwind`])
    /// ```
    ///
    /// NOTE: this does not clear the master drop flag, so you need
    /// to point succ/unwind on a `drop_ladder_bottom`.
    #[instrument(level = "debug", skip(self), ret)]
    fn drop_ladder(
        &mut self,
        fields: Vec<(Place<'tcx>, Option<D::Path>)>,
        succ: BasicBlock,
        unwind: Unwind,
        dropline: Option<BasicBlock>,
    ) -> (BasicBlock, Unwind, Option<BasicBlock>) {
        assert!(
            if unwind.is_cleanup() { dropline.is_none() } else { true },
            "Dropline is set for cleanup drop ladder"
        );

        let mut fields = fields;
        fields.retain(|&(place, _)| {
            self.place_ty(place).needs_drop(self.tcx(), self.elaborator.typing_env())
        });

        debug!("drop_ladder - fields needing drop: {:?}", fields);

        let dropline_ladder: Vec<Option<BasicBlock>> = vec![None; fields.len() + 1];
        let unwind_ladder = vec![Unwind::InCleanup; fields.len() + 1];
        let unwind_ladder: Vec<_> = if let Unwind::To(succ) = unwind {
            let halfladder = self.drop_halfladder(&unwind_ladder, &dropline_ladder, succ, &fields);
            halfladder.into_iter().map(Unwind::To).collect()
        } else {
            unwind_ladder
        };
        let dropline_ladder: Vec<_> = if let Some(succ) = dropline {
            let halfladder = self.drop_halfladder(&unwind_ladder, &dropline_ladder, succ, &fields);
            halfladder.into_iter().map(Some).collect()
        } else {
            dropline_ladder
        };

        let normal_ladder = self.drop_halfladder(&unwind_ladder, &dropline_ladder, succ, &fields);

        (
            *normal_ladder.last().unwrap(),
            *unwind_ladder.last().unwrap(),
            *dropline_ladder.last().unwrap(),
        )
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn open_drop_for_tuple(&mut self, tys: &[Ty<'tcx>]) -> BasicBlock {
        let fields = tys
            .iter()
            .enumerate()
            .map(|(i, &ty)| {
                (
                    self.tcx().mk_place_field(self.place, FieldIdx::new(i), ty),
                    self.elaborator.field_subpath(self.path, FieldIdx::new(i)),
                )
            })
            .collect();

        let (succ, unwind, dropline) = self.drop_ladder_bottom();
        self.drop_ladder(fields, succ, unwind, dropline).0
    }

    /// Drops the T contained in a `Box<T>` if it has not been moved out of
    #[instrument(level = "debug", ret)]
    fn open_drop_for_box_contents(
        &mut self,
        adt: ty::AdtDef<'tcx>,
        args: GenericArgsRef<'tcx>,
        succ: BasicBlock,
        unwind: Unwind,
        dropline: Option<BasicBlock>,
    ) -> BasicBlock {
        // drop glue is sent straight to codegen
        // box cannot be directly dereferenced
        let unique_ty =
            adt.non_enum_variant().fields[FieldIdx::ZERO].ty(self.tcx(), args).skip_norm_wip();
        let unique_variant = unique_ty.ty_adt_def().unwrap().non_enum_variant();
        let nonnull_ty = unique_variant.fields[FieldIdx::ZERO].ty(self.tcx(), args).skip_norm_wip();
        let ptr_ty = Ty::new_imm_ptr(self.tcx(), args[0].expect_ty());

        let unique_place = self.tcx().mk_place_field(self.place, FieldIdx::ZERO, unique_ty);
        let nonnull_place = self.tcx().mk_place_field(unique_place, FieldIdx::ZERO, nonnull_ty);

        let ptr_local = self.new_temp(ptr_ty);

        let interior = self.tcx().mk_place_deref(Place::from(ptr_local));
        let interior_path = self.elaborator.deref_subpath(self.path);

        let do_drop_bb = self.drop_subpath(interior, interior_path, succ, unwind, dropline);

        self.new_block_with_statements(
            unwind,
            vec![self.assign(
                Place::from(ptr_local),
                Rvalue::Cast(CastKind::Transmute, Operand::Copy(nonnull_place), ptr_ty),
            )],
            TerminatorKind::Goto { target: do_drop_bb },
        )
    }

    #[instrument(level = "debug", ret)]
    fn open_drop_for_adt(
        &mut self,
        adt: ty::AdtDef<'tcx>,
        args: GenericArgsRef<'tcx>,
    ) -> BasicBlock {
        if adt.variants().is_empty() {
            return self.new_block(self.unwind, TerminatorKind::Unreachable);
        }

        let skip_contents = adt.is_union() || adt.is_manually_drop();
        let (contents_succ, contents_unwind, contents_dropline) = if skip_contents {
            if adt.has_dtor(self.tcx()) && self.elaborator.get_drop_flag(self.path).is_some() {
                // the top-level drop flag is usually cleared by open_drop_for_adt_contents
                // types with destructors would still need an empty drop ladder to clear it

                // however, these types are only open dropped in `DropShimElaborator`
                // which does not have drop flags
                // a future box-like "DerefMove" trait would allow for this case to happen
                span_bug!(self.source_info.span, "open dropping partially moved union");
            }

            (self.succ, self.unwind, self.dropline)
        } else {
            self.open_drop_for_adt_contents(adt, args)
        };

        if adt.has_dtor(self.tcx()) {
            let destructor_block = if adt.is_box() {
                // we need to drop the inside of the box before running the destructor
                let succ = self.destructor_call_block_sync(contents_succ, contents_unwind);
                let unwind = contents_unwind
                    .map(|unwind| self.destructor_call_block_sync(unwind, Unwind::InCleanup));
                let dropline = contents_dropline
                    .map(|dropline| self.destructor_call_block_sync(dropline, contents_unwind));
                self.open_drop_for_box_contents(adt, args, succ, unwind, dropline)
            } else {
                self.destructor_call_block(contents_succ, contents_unwind, contents_dropline)
            };

            self.drop_flag_test_block(destructor_block, contents_succ, contents_unwind)
        } else {
            contents_succ
        }
    }

    fn open_drop_for_adt_contents(
        &mut self,
        adt: ty::AdtDef<'tcx>,
        args: GenericArgsRef<'tcx>,
    ) -> (BasicBlock, Unwind, Option<BasicBlock>) {
        let (succ, unwind, dropline) = self.drop_ladder_bottom();
        if !adt.is_enum() {
            let fields =
                self.move_paths_for_fields(self.place, self.path, adt.variant(FIRST_VARIANT), args);
            self.drop_ladder(fields, succ, unwind, dropline)
        } else {
            self.open_drop_for_multivariant(adt, args, succ, unwind, dropline)
        }
    }

    fn open_drop_for_multivariant(
        &mut self,
        adt: ty::AdtDef<'tcx>,
        args: GenericArgsRef<'tcx>,
        succ: BasicBlock,
        unwind: Unwind,
        dropline: Option<BasicBlock>,
    ) -> (BasicBlock, Unwind, Option<BasicBlock>) {
        let mut values = Vec::with_capacity(adt.variants().len());
        let mut normal_blocks = Vec::with_capacity(adt.variants().len());
        let mut unwind_blocks =
            Vec::with_capacity(if unwind.is_cleanup() { 0 } else { adt.variants().len() });
        let mut dropline_blocks =
            Vec::with_capacity(if dropline.is_none() { 0 } else { adt.variants().len() });

        let mut have_otherwise_with_drop_glue = false;
        let mut have_otherwise = false;
        let tcx = self.tcx();

        for (variant_index, discr) in adt.discriminants(tcx) {
            let variant = &adt.variant(variant_index);
            let subpath = self.elaborator.downcast_subpath(self.path, variant_index);

            if let Some(variant_path) = subpath {
                let base_place = tcx.mk_place_elem(
                    self.place,
                    ProjectionElem::Downcast(Some(variant.name), variant_index),
                );
                let fields = self.move_paths_for_fields(base_place, variant_path, variant, args);
                values.push(discr.val);
                if let Unwind::To(unwind) = unwind {
                    // We can't use the half-ladder from the original
                    // drop ladder, because this breaks the
                    // "funclet can't have 2 successor funclets"
                    // requirement from MSVC:
                    //
                    //           switch       unwind-switch
                    //          /      \         /        \
                    //         v1.0    v2.0  v2.0-unwind  v1.0-unwind
                    //         |        |      /             |
                    //    v1.1-unwind  v2.1-unwind           |
                    //      ^                                |
                    //       \-------------------------------/
                    //
                    // Create a duplicate half-ladder to avoid that. We
                    // could technically only do this on MSVC, but I
                    // I want to minimize the divergence between MSVC
                    // and non-MSVC.

                    let unwind_ladder = vec![Unwind::InCleanup; fields.len() + 1];
                    let dropline_ladder: Vec<Option<BasicBlock>> = vec![None; fields.len() + 1];
                    let halfladder =
                        self.drop_halfladder(&unwind_ladder, &dropline_ladder, unwind, &fields);
                    unwind_blocks.push(halfladder.last().cloned().unwrap());
                }
                let (normal, _, drop_bb) = self.drop_ladder(fields, succ, unwind, dropline);
                normal_blocks.push(normal);
                if dropline.is_some() {
                    dropline_blocks.push(drop_bb.unwrap());
                }
            } else {
                have_otherwise = true;

                let typing_env = self.elaborator.typing_env();
                let have_field_with_drop_glue = variant
                    .fields
                    .iter()
                    .any(|field| field.ty(tcx, args).skip_norm_wip().needs_drop(tcx, typing_env));
                if have_field_with_drop_glue {
                    have_otherwise_with_drop_glue = true;
                }
            }
        }

        if !have_otherwise {
            values.pop();
        } else if !have_otherwise_with_drop_glue {
            normal_blocks.push(self.goto_block(succ, unwind));
            if let Unwind::To(unwind) = unwind {
                unwind_blocks.push(self.goto_block(unwind, Unwind::InCleanup));
            }
        } else {
            normal_blocks.push(self.drop_block(succ, unwind));
            if let Unwind::To(unwind) = unwind {
                unwind_blocks.push(self.drop_block(unwind, Unwind::InCleanup));
            }
        }

        (
            self.adt_switch_block(adt, normal_blocks, &values, succ, unwind),
            unwind.map(|unwind| {
                self.adt_switch_block(adt, unwind_blocks, &values, unwind, Unwind::InCleanup)
            }),
            dropline.map(|dropline| {
                self.adt_switch_block(adt, dropline_blocks, &values, dropline, unwind)
            }),
        )
    }

    fn adt_switch_block(
        &mut self,
        adt: ty::AdtDef<'tcx>,
        blocks: Vec<BasicBlock>,
        values: &[u128],
        succ: BasicBlock,
        unwind: Unwind,
    ) -> BasicBlock {
        // If there are multiple variants, then if something
        // is present within the enum the discriminant, tracked
        // by the rest path, must be initialized.
        //
        // Additionally, we do not want to switch on the
        // discriminant after it is free-ed, because that
        // way lies only trouble.
        let discr_ty = adt.repr().discr_type().to_ty(self.tcx());
        let discr = Place::from(self.new_temp(discr_ty));
        let discr_rv = Rvalue::Discriminant(self.place);
        let switch_block = self.new_block_with_statements(
            unwind,
            vec![self.assign(discr, discr_rv)],
            TerminatorKind::SwitchInt {
                discr: Operand::Move(discr),
                targets: SwitchTargets::new(
                    values.iter().copied().zip(blocks.iter().copied()),
                    *blocks.last().unwrap(),
                ),
            },
        );
        self.drop_flag_test_block(switch_block, succ, unwind)
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn destructor_call_block_sync(&mut self, succ: BasicBlock, unwind: Unwind) -> BasicBlock {
        let tcx = self.tcx();
        let drop_trait = tcx.require_lang_item(LangItem::Drop, DUMMY_SP);
        let drop_fn = tcx.associated_item_def_ids(drop_trait)[0];
        let ty = self.place_ty(self.place);

        let ref_ty = Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, ty);
        let ref_place = self.new_temp(ref_ty);
        let unit_temp = Place::from(self.new_temp(tcx.types.unit));

        self.new_block_with_statements(
            unwind,
            vec![self.assign(
                Place::from(ref_place),
                Rvalue::Ref(
                    tcx.lifetimes.re_erased,
                    BorrowKind::Mut { kind: MutBorrowKind::Default },
                    self.place,
                ),
            )],
            TerminatorKind::Call {
                func: Operand::function_handle(tcx, drop_fn, [ty.into()], self.source_info.span),
                args: [dummy_spanned(Operand::Move(Place::from(ref_place)))].into(),
                destination: unit_temp,
                target: Some(succ),
                unwind: unwind.into_action(),
                call_source: CallSource::Misc,
                fn_span: self.source_info.span,
            },
        )
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn destructor_call_block(
        &mut self,
        succ: BasicBlock,
        unwind: Unwind,
        dropline: Option<BasicBlock>,
    ) -> BasicBlock {
        let ty = self.place_ty(self.place);
        if !unwind.is_cleanup() && self.check_if_can_async_drop(ty, true) {
            self.build_async_drop(self.place, ty, succ, unwind, dropline, true)
        } else {
            self.destructor_call_block_sync(succ, unwind)
        }
    }

    /// Create a loop that drops an array:
    ///
    /// ```text
    /// loop-block:
    ///    can_go = cur == len
    ///    if can_go then succ else drop-block
    /// drop-block:
    ///    ptr = &raw mut P[cur]
    ///    cur = cur + 1
    ///    drop(ptr)
    /// ```
    fn drop_loop(
        &mut self,
        succ: BasicBlock,
        cur: Local,
        len: Local,
        ety: Ty<'tcx>,
        unwind: Unwind,
        dropline: Option<BasicBlock>,
    ) -> BasicBlock {
        let copy = |place: Place<'tcx>| Operand::Copy(place);
        let move_ = |place: Place<'tcx>| Operand::Move(place);
        let tcx = self.tcx();

        let ptr_ty = Ty::new_mut_ptr(tcx, ety);
        let ptr = Place::from(self.new_temp(ptr_ty));
        let can_go = Place::from(self.new_temp(tcx.types.bool));
        let one = self.constant_usize(1);

        let drop_block = self.new_block_with_statements(
            unwind,
            vec![
                self.assign(
                    ptr,
                    Rvalue::RawPtr(RawPtrKind::Mut, tcx.mk_place_index(self.place, cur)),
                ),
                self.assign(
                    cur.into(),
                    Rvalue::BinaryOp(BinOp::Add, Box::new((move_(cur.into()), one))),
                ),
            ],
            // this gets overwritten by drop elaboration.
            TerminatorKind::Unreachable,
        );

        let loop_block = self.new_block_with_statements(
            unwind,
            vec![self.assign(
                can_go,
                Rvalue::BinaryOp(BinOp::Eq, Box::new((copy(Place::from(cur)), copy(len.into())))),
            )],
            TerminatorKind::if_(move_(can_go), succ, drop_block),
        );

        let place = tcx.mk_place_deref(ptr);
        if !unwind.is_cleanup() && self.check_if_can_async_drop(ety, false) {
            let async_drop_bb =
                self.build_async_drop(place, ety, loop_block, unwind, dropline, false);
            self.elaborator
                .patch()
                .patch_terminator(drop_block, TerminatorKind::Goto { target: async_drop_bb });
        } else {
            self.elaborator.patch().patch_terminator(
                drop_block,
                TerminatorKind::Drop {
                    place,
                    target: loop_block,
                    unwind: unwind.into_action(),
                    replace: false,
                    drop: None,
                },
            );
        }
        loop_block
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn open_drop_for_array(
        &mut self,
        array_ty: Ty<'tcx>,
        ety: Ty<'tcx>,
        opt_size: Option<u64>,
    ) -> BasicBlock {
        let tcx = self.tcx();

        if let Some(size) = opt_size {
            enum ProjectionKind<Path> {
                Drop(std::ops::Range<u64>),
                Keep(u64, Path),
            }
            // Previously, we'd make a projection for every element in the array and create a drop
            // ladder if any `array_subpath` was `Some`, i.e. moving out with an array pattern.
            // This caused huge memory usage when generating the drops for large arrays, so we instead
            // record the *subslices* which are dropped and the *indexes* which are kept
            let mut drop_ranges = vec![];
            let mut dropping = true;
            let mut start = 0;
            for i in 0..size {
                let path = self.elaborator.array_subpath(self.path, i, size);
                if dropping && path.is_some() {
                    drop_ranges.push(ProjectionKind::Drop(start..i));
                    dropping = false;
                } else if !dropping && path.is_none() {
                    dropping = true;
                    start = i;
                }
                if let Some(path) = path {
                    drop_ranges.push(ProjectionKind::Keep(i, path));
                }
            }
            if !drop_ranges.is_empty() {
                if dropping {
                    drop_ranges.push(ProjectionKind::Drop(start..size));
                }
                let fields = drop_ranges
                    .iter()
                    .rev()
                    .map(|p| {
                        let (project, path) = match p {
                            ProjectionKind::Drop(r) => (
                                ProjectionElem::Subslice {
                                    from: r.start,
                                    to: r.end,
                                    from_end: false,
                                },
                                None,
                            ),
                            &ProjectionKind::Keep(offset, path) => (
                                ProjectionElem::ConstantIndex {
                                    offset,
                                    min_length: size,
                                    from_end: false,
                                },
                                Some(path),
                            ),
                        };
                        (tcx.mk_place_elem(self.place, project), path)
                    })
                    .collect::<Vec<_>>();
                let (succ, unwind, dropline) = self.drop_ladder_bottom();
                return self.drop_ladder(fields, succ, unwind, dropline).0;
            }
        }

        let array_ptr_ty = Ty::new_mut_ptr(tcx, array_ty);
        let array_ptr = self.new_temp(array_ptr_ty);

        let slice_ty = Ty::new_slice(tcx, ety);
        let slice_ptr_ty = Ty::new_mut_ptr(tcx, slice_ty);
        let slice_ptr = self.new_temp(slice_ptr_ty);

        let array_place = mem::replace(
            &mut self.place,
            Place::from(slice_ptr).project_deeper(&[PlaceElem::Deref], tcx),
        );
        let slice_block = self.drop_loop_trio_for_slice(ety);
        self.place = array_place;

        self.new_block_with_statements(
            self.unwind,
            vec![
                self.assign(Place::from(array_ptr), Rvalue::RawPtr(RawPtrKind::Mut, self.place)),
                self.assign(
                    Place::from(slice_ptr),
                    Rvalue::Cast(
                        CastKind::PointerCoercion(
                            PointerCoercion::Unsize,
                            CoercionSource::Implicit,
                        ),
                        Operand::Move(Place::from(array_ptr)),
                        slice_ptr_ty,
                    ),
                ),
            ],
            TerminatorKind::Goto { target: slice_block },
        )
    }

    /// Creates a trio of drop-loops of `place`, which drops its contents, even
    /// in the case of 1 panic or in the case of coroutine drop
    #[instrument(level = "debug", skip(self), ret)]
    fn drop_loop_trio_for_slice(&mut self, ety: Ty<'tcx>) -> BasicBlock {
        let tcx = self.tcx();
        let len = self.new_temp(tcx.types.usize);
        let cur = self.new_temp(tcx.types.usize);

        let unwind = self
            .unwind
            .map(|unwind| self.drop_loop(unwind, cur, len, ety, Unwind::InCleanup, None));

        let dropline =
            self.dropline.map(|dropline| self.drop_loop(dropline, cur, len, ety, unwind, None));

        let loop_block = self.drop_loop(self.succ, cur, len, ety, unwind, dropline);

        let [PlaceElem::Deref] = self.place.projection.as_slice() else {
            span_bug!(
                self.source_info.span,
                "Expected place for slice drop shim to be *_n, but it's {:?}",
                self.place,
            );
        };

        let zero = self.constant_usize(0);
        let drop_block = self.new_block_with_statements(
            unwind,
            vec![
                self.assign(
                    len.into(),
                    Rvalue::UnaryOp(
                        UnOp::PtrMetadata,
                        Operand::Copy(Place::from(self.place.local)),
                    ),
                ),
                self.assign(cur.into(), Rvalue::Use(zero, WithRetag::Yes)),
            ],
            TerminatorKind::Goto { target: loop_block },
        );

        // FIXME(#34708): handle partially-dropped array/slice elements.
        let reset_block = self.drop_flag_reset_block(DropFlagMode::Deep, drop_block, unwind);
        self.drop_flag_test_block(reset_block, self.succ, unwind)
    }

    /// The slow-path - create an "open", elaborated drop for a type
    /// which is moved-out-of only partially, and patch `bb` to a jump
    /// to it. This must not be called on ADTs with a destructor,
    /// as these can't be moved-out-of, except for `Box<T>`, which is
    /// special-cased.
    ///
    /// This creates a "drop ladder" that drops the needed fields of the
    /// ADT, both in the success case or if one of the destructors fail.
    fn open_drop(&mut self) -> BasicBlock {
        let ty = self.place_ty(self.place);
        match ty.kind() {
            ty::Closure(_, args) => self.open_drop_for_tuple(args.as_closure().upvar_tys()),
            ty::CoroutineClosure(_, args) => {
                self.open_drop_for_tuple(args.as_coroutine_closure().upvar_tys())
            }
            // Note that `elaborate_drops` only drops the upvars of a coroutine,
            // and this is ok because `open_drop` here can only be reached
            // within that own coroutine's resume function.
            // This should only happen for the self argument on the resume function.
            // It effectively only contains upvars until the coroutine transformation runs.
            // See librustc_body/transform/coroutine.rs for more details.
            ty::Coroutine(_, args) => self.open_drop_for_tuple(args.as_coroutine().upvar_tys()),
            ty::Tuple(fields) => self.open_drop_for_tuple(fields),
            ty::Adt(def, args) => self.open_drop_for_adt(*def, args),
            ty::Dynamic(..) => self.complete_drop(self.succ, self.unwind),
            ty::Array(ety, size) => {
                let size = size.try_to_target_usize(self.tcx());
                self.open_drop_for_array(ty, *ety, size)
            }
            ty::Slice(ety) => self.drop_loop_trio_for_slice(*ety),

            ty::UnsafeBinder(_) => {
                // Unsafe binders may elaborate drops if their inner type isn't copy.
                // This is enforced in typeck, so this should never happen.
                self.tcx().dcx().span_delayed_bug(
                    self.source_info.span,
                    "open drop for unsafe binder shouldn't be encountered",
                );
                self.new_block(self.unwind, TerminatorKind::Unreachable)
            }

            _ => span_bug!(self.source_info.span, "open drop from non-ADT `{:?}`", ty),
        }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn complete_drop(&mut self, succ: BasicBlock, unwind: Unwind) -> BasicBlock {
        let drop_block = self.drop_block(succ, unwind);
        self.drop_flag_test_block(drop_block, succ, unwind)
    }

    /// Creates a block that resets the drop flag. If `mode` is deep, all children drop flags will
    /// also be cleared.
    #[instrument(level = "debug", skip(self), ret)]
    fn drop_flag_reset_block(
        &mut self,
        mode: DropFlagMode,
        succ: BasicBlock,
        unwind: Unwind,
    ) -> BasicBlock {
        if unwind.is_cleanup() {
            // The drop flag isn't read again on the unwind path, so don't
            // bother setting it.
            return succ;
        }
        let block = self.new_block(unwind, TerminatorKind::Goto { target: succ });
        let block_start = Location { block, statement_index: 0 };
        self.elaborator.clear_drop_flag(block_start, self.path, mode);
        block
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn elaborated_drop_block(&mut self) -> BasicBlock {
        let blk = self.new_block(
            self.unwind,
            TerminatorKind::Drop {
                place: self.place,
                target: self.succ,
                unwind: self.unwind.into_action(),
                replace: false,
                drop: self.dropline,
            },
        );
        self.elaborate_drop(blk);
        blk
    }

    fn drop_block(&mut self, target: BasicBlock, unwind: Unwind) -> BasicBlock {
        let drop_ty = self.place_ty(self.place);
        if !unwind.is_cleanup() && self.check_if_can_async_drop(drop_ty, false) {
            self.build_async_drop(self.place, drop_ty, self.succ, unwind, self.dropline, false)
        } else {
            self.new_block(
                unwind,
                TerminatorKind::Drop {
                    place: self.place,
                    target,
                    unwind: unwind.into_action(),
                    replace: false,
                    drop: None,
                },
            )
        }
    }

    fn goto_block(&mut self, target: BasicBlock, unwind: Unwind) -> BasicBlock {
        let block = TerminatorKind::Goto { target };
        self.new_block(unwind, block)
    }

    /// Returns the block to jump to in order to test the drop flag and execute the drop.
    ///
    /// Depending on the required `DropStyle`, this might be a generated block with an `if`
    /// terminator (for dynamic/open drops), or it might be `on_set` or `on_unset` itself, in case
    /// the drop can be statically determined.
    #[instrument(level = "debug", skip(self), ret)]
    fn drop_flag_test_block(
        &mut self,
        on_set: BasicBlock,
        on_unset: BasicBlock,
        unwind: Unwind,
    ) -> BasicBlock {
        let style = self.elaborator.drop_style(self.path, DropFlagMode::Shallow);
        match style {
            DropStyle::Dead => on_unset,
            DropStyle::Static => on_set,
            DropStyle::Conditional | DropStyle::Open => {
                let flag = self.elaborator.get_drop_flag(self.path).unwrap();
                let term = TerminatorKind::if_(flag, on_set, on_unset);
                self.new_block(unwind, term)
            }
        }
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn new_block(&mut self, unwind: Unwind, k: TerminatorKind<'tcx>) -> BasicBlock {
        self.elaborator.patch().new_block(BasicBlockData::new(
            Some(Terminator { source_info: self.source_info, kind: k, attributes: ThinVec::new() }),
            unwind.is_cleanup(),
        ))
    }

    #[instrument(level = "trace", skip(self, statements), ret)]
    fn new_block_with_statements(
        &mut self,
        unwind: Unwind,
        statements: Vec<Statement<'tcx>>,
        k: TerminatorKind<'tcx>,
    ) -> BasicBlock {
        self.elaborator.patch().new_block(BasicBlockData::new_stmts(
            statements,
            Some(Terminator { source_info: self.source_info, kind: k, attributes: ThinVec::new() }),
            unwind.is_cleanup(),
        ))
    }

    fn new_temp(&mut self, ty: Ty<'tcx>) -> Local {
        self.elaborator.patch().new_temp(ty, self.source_info.span)
    }

    fn constant_usize(&self, val: u16) -> Operand<'tcx> {
        Operand::Constant(Box::new(ConstOperand {
            span: self.source_info.span,
            user_ty: None,
            const_: Const::from_usize(self.tcx(), val.into()),
        }))
    }

    fn assign(&self, lhs: Place<'tcx>, rhs: Rvalue<'tcx>) -> Statement<'tcx> {
        Statement::new(self.source_info, StatementKind::Assign(Box::new((lhs, rhs))))
    }

    fn storage_live(&self, local: Local) -> Statement<'tcx> {
        Statement::new(self.source_info, StatementKind::StorageLive(local))
    }

    fn storage_dead(&self, local: Local) -> Statement<'tcx> {
        Statement::new(self.source_info, StatementKind::StorageDead(local))
    }
}
