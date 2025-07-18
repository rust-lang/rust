use std::{fmt, iter, mem};

use rustc_abi::{FIRST_VARIANT, FieldIdx, VariantIdx};
use rustc_hir::def::DefKind;
use rustc_hir::lang_items::LangItem;
use rustc_index::Idx;
use rustc_middle::mir::*;
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::util::IntTypeExt;
use rustc_middle::ty::{self, GenericArg, GenericArgsRef, Ty, TyCtxt};
use rustc_middle::{bug, span_bug, traits};
use rustc_span::DUMMY_SP;
use rustc_span::source_map::{Spanned, dummy_spanned};
use tracing::{debug, instrument};

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

    fn terminator_loc(&self, bb: BasicBlock) -> Location;

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
/// and whether the drop is "open", ie. should be expanded to drop all subfields of the dropped
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

    // Generates three blocks:
    // * #1:pin_obj_bb:   call Pin<ObjTy>::new_unchecked(&mut obj)
    // * #2:call_drop_bb: fut = call obj.<AsyncDrop::drop>() OR call async_drop_in_place<T>(obj)
    // * #3:drop_term_bb: drop (obj, fut, ...)
    // We keep async drop unexpanded to poll-loop here, to expand it later, at StateTransform -
    //   into states expand.
    // call_destructor_only - to call only AsyncDrop::drop, not full async_drop_in_place glue
    fn build_async_drop(
        &mut self,
        place: Place<'tcx>,
        drop_ty: Ty<'tcx>,
        bb: Option<BasicBlock>,
        succ: BasicBlock,
        unwind: Unwind,
        dropline: Option<BasicBlock>,
        call_destructor_only: bool,
    ) -> BasicBlock {
        let tcx = self.tcx();
        let span = self.source_info.span;

        let pin_obj_bb = bb.unwrap_or_else(|| {
            self.elaborator.patch().new_block(BasicBlockData::new(
                Some(Terminator {
                    // Temporary terminator, will be replaced by patch
                    source_info: self.source_info,
                    kind: TerminatorKind::Return,
                }),
                false,
            ))
        });

        let (fut_ty, drop_fn_def_id, trait_args) = if call_destructor_only {
            // Resolving obj.<AsyncDrop::drop>()
            let trait_ref =
                ty::TraitRef::new(tcx, tcx.require_lang_item(LangItem::AsyncDrop, span), [drop_ty]);
            let (drop_trait, trait_args) = match tcx.codegen_select_candidate(
                ty::TypingEnv::fully_monomorphized().as_query_input(trait_ref),
            ) {
                Ok(traits::ImplSource::UserDefined(traits::ImplSourceUserDefinedData {
                    impl_def_id,
                    args,
                    ..
                })) => (*impl_def_id, *args),
                impl_source => {
                    span_bug!(span, "invalid `AsyncDrop` impl_source: {:?}", impl_source);
                }
            };
            // impl_item_refs may be empty if drop fn is not implemented in 'impl AsyncDrop for ...'
            // (#140974).
            // Such code will report error, so just generate sync drop here and return
            let Some(drop_fn_def_id) = tcx
                .associated_item_def_ids(drop_trait)
                .first()
                .and_then(|def_id| {
                    if tcx.def_kind(def_id) == DefKind::AssocFn
                        && tcx.check_args_compatible(*def_id, trait_args)
                    {
                        Some(def_id)
                    } else {
                        None
                    }
                })
                .copied()
            else {
                tcx.dcx().span_delayed_bug(
                    self.elaborator.body().span,
                    "AsyncDrop type without correct `async fn drop(...)`.",
                );
                self.elaborator.patch().patch_terminator(
                    pin_obj_bb,
                    TerminatorKind::Drop {
                        place,
                        target: succ,
                        unwind: unwind.into_action(),
                        replace: false,
                        drop: None,
                        async_fut: None,
                    },
                );
                return pin_obj_bb;
            };
            let drop_fn = Ty::new_fn_def(tcx, drop_fn_def_id, trait_args);
            let sig = drop_fn.fn_sig(tcx);
            let sig = tcx.instantiate_bound_regions_with_erased(sig);
            (sig.output(), drop_fn_def_id, trait_args)
        } else {
            // Resolving async_drop_in_place<T> function for drop_ty
            let drop_fn_def_id = tcx.require_lang_item(LangItem::AsyncDropInPlace, span);
            let trait_args = tcx.mk_args(&[drop_ty.into()]);
            let sig = tcx.fn_sig(drop_fn_def_id).instantiate(tcx, trait_args);
            let sig = tcx.instantiate_bound_regions_with_erased(sig);
            (sig.output(), drop_fn_def_id, trait_args)
        };

        let fut = Place::from(self.new_temp(fut_ty));

        // #1:pin_obj_bb >>> obj_ref = &mut obj
        let obj_ref_ty = Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, drop_ty);
        let obj_ref_place = Place::from(self.new_temp(obj_ref_ty));

        let term_loc = self.elaborator.terminator_loc(pin_obj_bb);
        self.elaborator.patch().add_assign(
            term_loc,
            obj_ref_place,
            Rvalue::Ref(
                tcx.lifetimes.re_erased,
                BorrowKind::Mut { kind: MutBorrowKind::Default },
                place,
            ),
        );

        // pin_obj_place preparation
        let pin_obj_new_unchecked_fn = Ty::new_fn_def(
            tcx,
            tcx.require_lang_item(LangItem::PinNewUnchecked, span),
            [GenericArg::from(obj_ref_ty)],
        );
        let pin_obj_ty = pin_obj_new_unchecked_fn.fn_sig(tcx).output().no_bound_vars().unwrap();
        let pin_obj_place = Place::from(self.new_temp(pin_obj_ty));
        let pin_obj_new_unchecked_fn = Operand::Constant(Box::new(ConstOperand {
            span,
            user_ty: None,
            const_: Const::zero_sized(pin_obj_new_unchecked_fn),
        }));

        // #3:drop_term_bb
        let drop_term_bb = self.new_block(
            unwind,
            TerminatorKind::Drop {
                place,
                target: succ,
                unwind: unwind.into_action(),
                replace: false,
                drop: dropline,
                async_fut: Some(fut.local),
            },
        );

        // #2:call_drop_bb
        let mut call_statements = Vec::new();
        let drop_arg = if call_destructor_only {
            pin_obj_place
        } else {
            let ty::Adt(adt_def, adt_args) = pin_obj_ty.kind() else {
                bug!();
            };
            let obj_ptr_ty = Ty::new_mut_ptr(tcx, drop_ty);
            let unwrap_ty = adt_def.non_enum_variant().fields[FieldIdx::ZERO].ty(tcx, adt_args);
            let obj_ref_place = Place::from(self.new_temp(unwrap_ty));
            call_statements.push(self.assign(
                obj_ref_place,
                Rvalue::Use(Operand::Copy(tcx.mk_place_field(
                    pin_obj_place,
                    FieldIdx::ZERO,
                    unwrap_ty,
                ))),
            ));

            let obj_ptr_place = Place::from(self.new_temp(obj_ptr_ty));

            let addr = Rvalue::RawPtr(RawPtrKind::Mut, tcx.mk_place_deref(obj_ref_place));
            call_statements.push(self.assign(obj_ptr_place, addr));
            obj_ptr_place
        };
        call_statements
            .push(Statement::new(self.source_info, StatementKind::StorageLive(fut.local)));

        let call_drop_bb = self.new_block_with_statements(
            unwind,
            call_statements,
            TerminatorKind::Call {
                func: Operand::function_handle(tcx, drop_fn_def_id, trait_args, span),
                args: [Spanned { node: Operand::Move(drop_arg), span: DUMMY_SP }].into(),
                destination: fut,
                target: Some(drop_term_bb),
                unwind: unwind.into_action(),
                call_source: CallSource::Misc,
                fn_span: self.source_info.span,
            },
        );

        // StorageDead(fut) in self.succ block (at the begin)
        self.elaborator.patch().add_statement(
            Location { block: self.succ, statement_index: 0 },
            StatementKind::StorageDead(fut.local),
        );
        // StorageDead(fut) in unwind block (at the begin)
        if let Unwind::To(block) = unwind {
            self.elaborator.patch().add_statement(
                Location { block, statement_index: 0 },
                StatementKind::StorageDead(fut.local),
            );
        }
        // StorageDead(fut) in dropline block (at the begin)
        if let Some(block) = dropline {
            self.elaborator.patch().add_statement(
                Location { block, statement_index: 0 },
                StatementKind::StorageDead(fut.local),
            );
        }

        // #1:pin_obj_bb >>> call Pin<ObjTy>::new_unchecked(&mut obj)
        self.elaborator.patch().patch_terminator(
            pin_obj_bb,
            TerminatorKind::Call {
                func: pin_obj_new_unchecked_fn,
                args: [dummy_spanned(Operand::Move(obj_ref_place))].into(),
                destination: pin_obj_place,
                target: Some(call_drop_bb),
                unwind: unwind.into_action(),
                call_source: CallSource::Misc,
                fn_span: span,
            },
        );
        pin_obj_bb
    }

    fn build_drop(&mut self, bb: BasicBlock) {
        let drop_ty = self.place_ty(self.place);
        if self.tcx().features().async_drop()
            && self.elaborator.body().coroutine.is_some()
            && self.elaborator.allow_async_drops()
            && !self.elaborator.patch_ref().block(self.elaborator.body(), bb).is_cleanup
            && drop_ty.needs_async_drop(self.tcx(), self.elaborator.typing_env())
        {
            self.build_async_drop(
                self.place,
                drop_ty,
                Some(bb),
                self.succ,
                self.unwind,
                self.dropline,
                false,
            );
        } else {
            self.elaborator.patch().patch_terminator(
                bb,
                TerminatorKind::Drop {
                    place: self.place,
                    target: self.succ,
                    unwind: self.unwind.into_action(),
                    replace: false,
                    drop: None,
                    async_fut: None,
                },
            );
        }
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

                assert_eq!(self.elaborator.typing_env().typing_mode, ty::TypingMode::PostAnalysis);
                let field_ty = match tcx.try_normalize_erasing_regions(
                    self.elaborator.typing_env(),
                    field.ty(tcx, args),
                ) {
                    Ok(t) => t,
                    Err(_) => Ty::new_error(
                        self.tcx(),
                        self.tcx().dcx().span_delayed_bug(
                            self.elaborator.body().span,
                            "Error normalizing in drop elaboration.",
                        ),
                    ),
                };

                (tcx.mk_place_field(base_place, field_idx, field_ty), subpath)
            })
            .collect()
    }

    fn drop_subpath(
        &mut self,
        place: Place<'tcx>,
        path: Option<D::Path>,
        succ: BasicBlock,
        unwind: Unwind,
        dropline: Option<BasicBlock>,
    ) -> BasicBlock {
        if let Some(path) = path {
            debug!("drop_subpath: for std field {:?}", place);

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
            debug!("drop_subpath: for rest field {:?}", place);

            DropCtxt {
                elaborator: self.elaborator,
                source_info: self.source_info,
                place,
                succ,
                unwind,
                dropline,
                // Using `self.path` here to condition the drop on
                // our own drop flag.
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
    ///
    /// For possible-async drops in coroutines we also need dropline ladder
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
    ///
    /// NOTE: this does not clear the master drop flag, so you need
    /// to point succ/unwind on a `drop_ladder_bottom`.
    fn drop_ladder(
        &mut self,
        fields: Vec<(Place<'tcx>, Option<D::Path>)>,
        succ: BasicBlock,
        unwind: Unwind,
        dropline: Option<BasicBlock>,
    ) -> (BasicBlock, Unwind, Option<BasicBlock>) {
        debug!("drop_ladder({:?}, {:?})", self, fields);
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

    fn open_drop_for_tuple(&mut self, tys: &[Ty<'tcx>]) -> BasicBlock {
        debug!("open_drop_for_tuple({:?}, {:?})", self, tys);

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
        let unique_ty = adt.non_enum_variant().fields[FieldIdx::ZERO].ty(self.tcx(), args);
        let unique_variant = unique_ty.ty_adt_def().unwrap().non_enum_variant();
        let nonnull_ty = unique_variant.fields[FieldIdx::ZERO].ty(self.tcx(), args);
        let ptr_ty = Ty::new_imm_ptr(self.tcx(), args[0].expect_ty());

        let unique_place = self.tcx().mk_place_field(self.place, FieldIdx::ZERO, unique_ty);
        let nonnull_place = self.tcx().mk_place_field(unique_place, FieldIdx::ZERO, nonnull_ty);

        let ptr_local = self.new_temp(ptr_ty);

        let interior = self.tcx().mk_place_deref(Place::from(ptr_local));
        let interior_path = self.elaborator.deref_subpath(self.path);

        let do_drop_bb = self.drop_subpath(interior, interior_path, succ, unwind, dropline);

        let setup_bbd = BasicBlockData::new_stmts(
            vec![self.assign(
                Place::from(ptr_local),
                Rvalue::Cast(CastKind::Transmute, Operand::Copy(nonnull_place), ptr_ty),
            )],
            Some(Terminator {
                kind: TerminatorKind::Goto { target: do_drop_bb },
                source_info: self.source_info,
            }),
            unwind.is_cleanup(),
        );
        self.elaborator.patch().new_block(setup_bbd)
    }

    #[instrument(level = "debug", ret)]
    fn open_drop_for_adt(
        &mut self,
        adt: ty::AdtDef<'tcx>,
        args: GenericArgsRef<'tcx>,
    ) -> BasicBlock {
        if adt.variants().is_empty() {
            return self.elaborator.patch().new_block(BasicBlockData::new(
                Some(Terminator {
                    source_info: self.source_info,
                    kind: TerminatorKind::Unreachable,
                }),
                self.unwind.is_cleanup(),
            ));
        }

        let skip_contents = adt.is_union() || adt.is_manually_drop();
        let contents_drop = if skip_contents {
            (self.succ, self.unwind, self.dropline)
        } else {
            self.open_drop_for_adt_contents(adt, args)
        };

        if adt.is_box() {
            // we need to drop the inside of the box before running the destructor
            let succ = self.destructor_call_block_sync((contents_drop.0, contents_drop.1));
            let unwind = contents_drop
                .1
                .map(|unwind| self.destructor_call_block_sync((unwind, Unwind::InCleanup)));
            let dropline = contents_drop
                .2
                .map(|dropline| self.destructor_call_block_sync((dropline, contents_drop.1)));

            self.open_drop_for_box_contents(adt, args, succ, unwind, dropline)
        } else if adt.has_dtor(self.tcx()) {
            self.destructor_call_block(contents_drop)
        } else {
            contents_drop.0
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
            if unwind.is_cleanup() { None } else { Some(Vec::with_capacity(adt.variants().len())) };
        let mut dropline_blocks =
            if dropline.is_none() { None } else { Some(Vec::with_capacity(adt.variants().len())) };

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

                    let unwind_blocks = unwind_blocks.as_mut().unwrap();
                    let unwind_ladder = vec![Unwind::InCleanup; fields.len() + 1];
                    let dropline_ladder: Vec<Option<BasicBlock>> = vec![None; fields.len() + 1];
                    let halfladder =
                        self.drop_halfladder(&unwind_ladder, &dropline_ladder, unwind, &fields);
                    unwind_blocks.push(halfladder.last().cloned().unwrap());
                }
                let (normal, _, drop_bb) = self.drop_ladder(fields, succ, unwind, dropline);
                normal_blocks.push(normal);
                if dropline.is_some() {
                    dropline_blocks.as_mut().unwrap().push(drop_bb.unwrap());
                }
            } else {
                have_otherwise = true;

                let typing_env = self.elaborator.typing_env();
                let have_field_with_drop_glue = variant
                    .fields
                    .iter()
                    .any(|field| field.ty(tcx, args).needs_drop(tcx, typing_env));
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
                unwind_blocks.as_mut().unwrap().push(self.goto_block(unwind, Unwind::InCleanup));
            }
        } else {
            normal_blocks.push(self.drop_block(succ, unwind));
            if let Unwind::To(unwind) = unwind {
                unwind_blocks.as_mut().unwrap().push(self.drop_block(unwind, Unwind::InCleanup));
            }
        }

        (
            self.adt_switch_block(adt, normal_blocks, &values, succ, unwind),
            unwind.map(|unwind| {
                self.adt_switch_block(
                    adt,
                    unwind_blocks.unwrap(),
                    &values,
                    unwind,
                    Unwind::InCleanup,
                )
            }),
            dropline.map(|dropline| {
                self.adt_switch_block(adt, dropline_blocks.unwrap(), &values, dropline, unwind)
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
        let switch_block = BasicBlockData::new_stmts(
            vec![self.assign(discr, discr_rv)],
            Some(Terminator {
                source_info: self.source_info,
                kind: TerminatorKind::SwitchInt {
                    discr: Operand::Move(discr),
                    targets: SwitchTargets::new(
                        values.iter().copied().zip(blocks.iter().copied()),
                        *blocks.last().unwrap(),
                    ),
                },
            }),
            unwind.is_cleanup(),
        );
        let switch_block = self.elaborator.patch().new_block(switch_block);
        self.drop_flag_test_block(switch_block, succ, unwind)
    }

    fn destructor_call_block_sync(&mut self, (succ, unwind): (BasicBlock, Unwind)) -> BasicBlock {
        debug!("destructor_call_block_sync({:?}, {:?})", self, succ);
        let tcx = self.tcx();
        let drop_trait = tcx.require_lang_item(LangItem::Drop, DUMMY_SP);
        let drop_fn = tcx.associated_item_def_ids(drop_trait)[0];
        let ty = self.place_ty(self.place);

        let ref_ty = Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, ty);
        let ref_place = self.new_temp(ref_ty);
        let unit_temp = Place::from(self.new_temp(tcx.types.unit));

        let result = BasicBlockData::new_stmts(
            vec![self.assign(
                Place::from(ref_place),
                Rvalue::Ref(
                    tcx.lifetimes.re_erased,
                    BorrowKind::Mut { kind: MutBorrowKind::Default },
                    self.place,
                ),
            )],
            Some(Terminator {
                kind: TerminatorKind::Call {
                    func: Operand::function_handle(
                        tcx,
                        drop_fn,
                        [ty.into()],
                        self.source_info.span,
                    ),
                    args: [Spanned { node: Operand::Move(Place::from(ref_place)), span: DUMMY_SP }]
                        .into(),
                    destination: unit_temp,
                    target: Some(succ),
                    unwind: unwind.into_action(),
                    call_source: CallSource::Misc,
                    fn_span: self.source_info.span,
                },
                source_info: self.source_info,
            }),
            unwind.is_cleanup(),
        );

        let destructor_block = self.elaborator.patch().new_block(result);

        let block_start = Location { block: destructor_block, statement_index: 0 };
        self.elaborator.clear_drop_flag(block_start, self.path, DropFlagMode::Shallow);

        self.drop_flag_test_block(destructor_block, succ, unwind)
    }

    fn destructor_call_block(
        &mut self,
        (succ, unwind, dropline): (BasicBlock, Unwind, Option<BasicBlock>),
    ) -> BasicBlock {
        debug!("destructor_call_block({:?}, {:?})", self, succ);
        let ty = self.place_ty(self.place);
        if self.tcx().features().async_drop()
            && self.elaborator.body().coroutine.is_some()
            && self.elaborator.allow_async_drops()
            && !unwind.is_cleanup()
            && ty.is_async_drop(self.tcx(), self.elaborator.typing_env())
        {
            let destructor_block =
                self.build_async_drop(self.place, ty, None, succ, unwind, dropline, true);

            let block_start = Location { block: destructor_block, statement_index: 0 };
            self.elaborator.clear_drop_flag(block_start, self.path, DropFlagMode::Shallow);

            self.drop_flag_test_block(destructor_block, succ, unwind)
        } else {
            self.destructor_call_block_sync((succ, unwind))
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

        let drop_block = BasicBlockData::new_stmts(
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
            Some(Terminator {
                source_info: self.source_info,
                // this gets overwritten by drop elaboration.
                kind: TerminatorKind::Unreachable,
            }),
            unwind.is_cleanup(),
        );
        let drop_block = self.elaborator.patch().new_block(drop_block);

        let loop_block = BasicBlockData::new_stmts(
            vec![self.assign(
                can_go,
                Rvalue::BinaryOp(BinOp::Eq, Box::new((copy(Place::from(cur)), copy(len.into())))),
            )],
            Some(Terminator {
                source_info: self.source_info,
                kind: TerminatorKind::if_(move_(can_go), succ, drop_block),
            }),
            unwind.is_cleanup(),
        );
        let loop_block = self.elaborator.patch().new_block(loop_block);

        let place = tcx.mk_place_deref(ptr);
        if self.tcx().features().async_drop()
            && self.elaborator.body().coroutine.is_some()
            && self.elaborator.allow_async_drops()
            && !unwind.is_cleanup()
            && ety.needs_async_drop(self.tcx(), self.elaborator.typing_env())
        {
            self.build_async_drop(
                place,
                ety,
                Some(drop_block),
                loop_block,
                unwind,
                dropline,
                false,
            );
        } else {
            self.elaborator.patch().patch_terminator(
                drop_block,
                TerminatorKind::Drop {
                    place,
                    target: loop_block,
                    unwind: unwind.into_action(),
                    replace: false,
                    drop: None,
                    async_fut: None,
                },
            );
        }
        loop_block
    }

    fn open_drop_for_array(
        &mut self,
        array_ty: Ty<'tcx>,
        ety: Ty<'tcx>,
        opt_size: Option<u64>,
    ) -> BasicBlock {
        debug!("open_drop_for_array({:?}, {:?}, {:?})", array_ty, ety, opt_size);
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

        let mut delegate_block = BasicBlockData::new_stmts(
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
            None,
            self.unwind.is_cleanup(),
        );

        let array_place = mem::replace(
            &mut self.place,
            Place::from(slice_ptr).project_deeper(&[PlaceElem::Deref], tcx),
        );
        let slice_block = self.drop_loop_trio_for_slice(ety);
        self.place = array_place;

        delegate_block.terminator = Some(Terminator {
            source_info: self.source_info,
            kind: TerminatorKind::Goto { target: slice_block },
        });
        self.elaborator.patch().new_block(delegate_block)
    }

    /// Creates a trio of drop-loops of `place`, which drops its contents, even
    /// in the case of 1 panic or in the case of coroutine drop
    fn drop_loop_trio_for_slice(&mut self, ety: Ty<'tcx>) -> BasicBlock {
        debug!("drop_loop_trio_for_slice({:?})", ety);
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
        let block = BasicBlockData::new_stmts(
            vec![
                self.assign(
                    len.into(),
                    Rvalue::UnaryOp(
                        UnOp::PtrMetadata,
                        Operand::Copy(Place::from(self.place.local)),
                    ),
                ),
                self.assign(cur.into(), Rvalue::Use(zero)),
            ],
            Some(Terminator {
                source_info: self.source_info,
                kind: TerminatorKind::Goto { target: loop_block },
            }),
            unwind.is_cleanup(),
        );

        let drop_block = self.elaborator.patch().new_block(block);
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
                self.elaborator.patch().new_block(BasicBlockData::new(
                    Some(Terminator {
                        source_info: self.source_info,
                        kind: TerminatorKind::Unreachable,
                    }),
                    self.unwind.is_cleanup(),
                ))
            }

            _ => span_bug!(self.source_info.span, "open drop from non-ADT `{:?}`", ty),
        }
    }

    fn complete_drop(&mut self, succ: BasicBlock, unwind: Unwind) -> BasicBlock {
        debug!("complete_drop(succ={:?}, unwind={:?})", succ, unwind);

        let drop_block = self.drop_block(succ, unwind);

        self.drop_flag_test_block(drop_block, succ, unwind)
    }

    /// Creates a block that resets the drop flag. If `mode` is deep, all children drop flags will
    /// also be cleared.
    fn drop_flag_reset_block(
        &mut self,
        mode: DropFlagMode,
        succ: BasicBlock,
        unwind: Unwind,
    ) -> BasicBlock {
        debug!("drop_flag_reset_block({:?},{:?})", self, mode);

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

    fn elaborated_drop_block(&mut self) -> BasicBlock {
        debug!("elaborated_drop_block({:?})", self);
        let blk = self.drop_block_simple(self.succ, self.unwind);
        self.elaborate_drop(blk);
        blk
    }

    fn drop_block_simple(&mut self, target: BasicBlock, unwind: Unwind) -> BasicBlock {
        let block = TerminatorKind::Drop {
            place: self.place,
            target,
            unwind: unwind.into_action(),
            replace: false,
            drop: self.dropline,
            async_fut: None,
        };
        self.new_block(unwind, block)
    }

    fn drop_block(&mut self, target: BasicBlock, unwind: Unwind) -> BasicBlock {
        let drop_ty = self.place_ty(self.place);
        if self.tcx().features().async_drop()
            && self.elaborator.body().coroutine.is_some()
            && self.elaborator.allow_async_drops()
            && !unwind.is_cleanup()
            && drop_ty.needs_async_drop(self.tcx(), self.elaborator.typing_env())
        {
            self.build_async_drop(
                self.place,
                drop_ty,
                None,
                self.succ,
                unwind,
                self.dropline,
                false,
            )
        } else {
            let block = TerminatorKind::Drop {
                place: self.place,
                target,
                unwind: unwind.into_action(),
                replace: false,
                drop: None,
                async_fut: None,
            };
            self.new_block(unwind, block)
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
    fn drop_flag_test_block(
        &mut self,
        on_set: BasicBlock,
        on_unset: BasicBlock,
        unwind: Unwind,
    ) -> BasicBlock {
        let style = self.elaborator.drop_style(self.path, DropFlagMode::Shallow);
        debug!(
            "drop_flag_test_block({:?},{:?},{:?},{:?}) - {:?}",
            self, on_set, on_unset, unwind, style
        );

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

    fn new_block(&mut self, unwind: Unwind, k: TerminatorKind<'tcx>) -> BasicBlock {
        self.elaborator.patch().new_block(BasicBlockData::new(
            Some(Terminator { source_info: self.source_info, kind: k }),
            unwind.is_cleanup(),
        ))
    }

    fn new_block_with_statements(
        &mut self,
        unwind: Unwind,
        statements: Vec<Statement<'tcx>>,
        k: TerminatorKind<'tcx>,
    ) -> BasicBlock {
        self.elaborator.patch().new_block(BasicBlockData::new_stmts(
            statements,
            Some(Terminator { source_info: self.source_info, kind: k }),
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
}
