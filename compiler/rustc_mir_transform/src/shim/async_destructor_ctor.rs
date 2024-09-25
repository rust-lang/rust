use std::iter;

use itertools::Itertools;
use rustc_ast::Mutability;
use rustc_const_eval::interpret;
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::{
    BasicBlock, BasicBlockData, Body, CallSource, CastKind, CoercionSource, Const, ConstOperand,
    ConstValue, Local, LocalDecl, MirSource, Operand, Place, PlaceElem, RETURN_PLACE, Rvalue,
    SourceInfo, Statement, StatementKind, Terminator, TerminatorKind, UnwindAction,
    UnwindTerminateReason,
};
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::util::{AsyncDropGlueMorphology, Discr};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_span::source_map::respan;
use rustc_span::{Span, Symbol};
use rustc_target::abi::{FieldIdx, VariantIdx};
use rustc_target::spec::PanicStrategy;
use tracing::debug;

use super::{local_decls_for_sig, new_body};

pub(super) fn build_async_destructor_ctor_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    ty: Option<Ty<'tcx>>,
) -> Body<'tcx> {
    debug!("build_drop_shim(def_id={:?}, ty={:?})", def_id, ty);

    AsyncDestructorCtorShimBuilder::new(tcx, def_id, ty).build()
}

/// Builder for async_drop_in_place shim. Functions as a stack machine
/// to build up an expression using combinators. Stack contains pairs
/// of locals and types. Combinator is a not yet instantiated pair of a
/// function and a type, is considered to be an operator which consumes
/// operands from the stack by instantiating its function and its type
/// with operand types and moving locals into the function call. Top
/// pair is considered to be the last operand.
// FIXME: add mir-opt tests
struct AsyncDestructorCtorShimBuilder<'tcx> {
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    self_ty: Option<Ty<'tcx>>,
    span: Span,
    source_info: SourceInfo,
    param_env: ty::ParamEnv<'tcx>,

    stack: Vec<Operand<'tcx>>,
    last_bb: BasicBlock,
    top_cleanup_bb: Option<BasicBlock>,

    locals: IndexVec<Local, LocalDecl<'tcx>>,
    bbs: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
}

#[derive(Clone, Copy)]
enum SurfaceDropKind {
    Async,
    Sync,
}

impl<'tcx> AsyncDestructorCtorShimBuilder<'tcx> {
    const SELF_PTR: Local = Local::from_u32(1);
    const INPUT_COUNT: usize = 1;
    const MAX_STACK_LEN: usize = 2;

    fn new(tcx: TyCtxt<'tcx>, def_id: DefId, self_ty: Option<Ty<'tcx>>) -> Self {
        let args = if let Some(ty) = self_ty {
            tcx.mk_args(&[ty.into()])
        } else {
            ty::GenericArgs::identity_for_item(tcx, def_id)
        };
        let sig = tcx.fn_sig(def_id).instantiate(tcx, args);
        let sig = tcx.instantiate_bound_regions_with_erased(sig);
        let span = tcx.def_span(def_id);

        let source_info = SourceInfo::outermost(span);

        debug_assert_eq!(sig.inputs().len(), Self::INPUT_COUNT);
        let locals = local_decls_for_sig(&sig, span);

        // Usual case: noop() + unwind resume + return
        let mut bbs = IndexVec::with_capacity(3);
        let param_env = tcx.param_env_reveal_all_normalized(def_id);
        AsyncDestructorCtorShimBuilder {
            tcx,
            def_id,
            self_ty,
            span,
            source_info,
            param_env,

            stack: Vec::with_capacity(Self::MAX_STACK_LEN),
            last_bb: bbs.push(BasicBlockData::new(None)),
            top_cleanup_bb: match tcx.sess.panic_strategy() {
                PanicStrategy::Unwind => {
                    // Don't drop input arg because it's just a pointer
                    Some(bbs.push(BasicBlockData {
                        statements: Vec::new(),
                        terminator: Some(Terminator {
                            source_info,
                            kind: TerminatorKind::UnwindResume,
                        }),
                        is_cleanup: true,
                    }))
                }
                PanicStrategy::Abort => None,
            },

            locals,
            bbs,
        }
    }

    fn build(self) -> Body<'tcx> {
        let (tcx, Some(self_ty)) = (self.tcx, self.self_ty) else {
            return self.build_zst_output();
        };
        match self_ty.async_drop_glue_morphology(tcx) {
            AsyncDropGlueMorphology::Noop => span_bug!(
                self.span,
                "async drop glue shim generator encountered type with noop async drop glue morphology"
            ),
            AsyncDropGlueMorphology::DeferredDropInPlace => {
                return self.build_deferred_drop_in_place();
            }
            AsyncDropGlueMorphology::Custom => (),
        }

        let surface_drop_kind = || {
            let adt_def = self_ty.ty_adt_def()?;
            if adt_def.async_destructor(tcx).is_some() {
                Some(SurfaceDropKind::Async)
            } else if adt_def.destructor(tcx).is_some() {
                Some(SurfaceDropKind::Sync)
            } else {
                None
            }
        };

        match self_ty.kind() {
            ty::Array(elem_ty, _) => self.build_slice(true, *elem_ty),
            ty::Slice(elem_ty) => self.build_slice(false, *elem_ty),

            ty::Tuple(elem_tys) => self.build_chain(None, elem_tys.iter()),
            ty::Adt(adt_def, args) if adt_def.is_struct() => {
                let field_tys = adt_def.non_enum_variant().fields.iter().map(|f| f.ty(tcx, args));
                self.build_chain(surface_drop_kind(), field_tys)
            }
            ty::Closure(_, args) => self.build_chain(None, args.as_closure().upvar_tys().iter()),
            ty::CoroutineClosure(_, args) => {
                self.build_chain(None, args.as_coroutine_closure().upvar_tys().iter())
            }

            ty::Adt(adt_def, args) if adt_def.is_enum() => {
                self.build_enum(*adt_def, *args, surface_drop_kind())
            }

            ty::Adt(adt_def, _) => {
                assert!(adt_def.is_union());
                match surface_drop_kind().unwrap() {
                    SurfaceDropKind::Async => self.build_fused_async_surface(),
                    SurfaceDropKind::Sync => self.build_fused_sync_surface(),
                }
            }

            ty::Bound(..)
            | ty::Foreign(_)
            | ty::Placeholder(_)
            | ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_) | ty::TyVar(_))
            | ty::Param(_)
            | ty::Alias(..) => {
                bug!("Building async destructor for unexpected type: {self_ty:?}")
            }

            _ => {
                bug!(
                    "Building async destructor constructor shim is not yet implemented for type: {self_ty:?}"
                )
            }
        }
    }

    fn build_enum(
        mut self,
        adt_def: ty::AdtDef<'tcx>,
        args: ty::GenericArgsRef<'tcx>,
        surface_drop: Option<SurfaceDropKind>,
    ) -> Body<'tcx> {
        let tcx = self.tcx;

        let surface = match surface_drop {
            None => None,
            Some(kind) => {
                self.put_self();
                Some(match kind {
                    SurfaceDropKind::Async => self.combine_async_surface(),
                    SurfaceDropKind::Sync => self.combine_sync_surface(),
                })
            }
        };

        let mut other = None;
        for (variant_idx, discr) in adt_def.discriminants(tcx) {
            let variant = adt_def.variant(variant_idx);

            let mut chain = None;
            for (field_idx, field) in variant.fields.iter_enumerated() {
                let field_ty = field.ty(tcx, args);
                self.put_variant_field(variant.name, variant_idx, field_idx, field_ty);
                let defer = self.combine_defer(field_ty);
                chain = Some(match chain {
                    None => defer,
                    Some(chain) => self.combine_chain(chain, defer),
                })
            }
            let variant_dtor = chain.unwrap_or_else(|| self.put_noop());

            other = Some(match other {
                None => variant_dtor,
                Some(other) => {
                    self.put_self();
                    self.put_discr(discr);
                    self.combine_either(other, variant_dtor)
                }
            });
        }
        let variants_dtor = other.unwrap_or_else(|| self.put_noop());

        let dtor = match surface {
            None => variants_dtor,
            Some(surface) => self.combine_chain(surface, variants_dtor),
        };
        self.combine_fuse(dtor);
        self.return_()
    }

    fn build_chain<I>(mut self, surface_drop: Option<SurfaceDropKind>, elem_tys: I) -> Body<'tcx>
    where
        I: Iterator<Item = Ty<'tcx>> + ExactSizeIterator,
    {
        let surface = match surface_drop {
            None => None,
            Some(kind) => {
                self.put_self();
                Some(match kind {
                    SurfaceDropKind::Async => self.combine_async_surface(),
                    SurfaceDropKind::Sync => self.combine_sync_surface(),
                })
            }
        };

        let mut chain = None;
        for (field_idx, field_ty) in elem_tys.enumerate().map(|(i, ty)| (FieldIdx::new(i), ty)) {
            self.put_field(field_idx, field_ty);
            let defer = self.combine_defer(field_ty);
            chain = Some(match chain {
                None => defer,
                Some(chain) => self.combine_chain(chain, defer),
            })
        }
        let chain = chain.unwrap_or_else(|| self.put_noop());

        let dtor = match surface {
            None => chain,
            Some(surface) => self.combine_chain(surface, chain),
        };
        self.combine_fuse(dtor);
        self.return_()
    }

    fn build_zst_output(mut self) -> Body<'tcx> {
        self.put_zst_output();
        self.return_()
    }

    fn build_deferred_drop_in_place(mut self) -> Body<'tcx> {
        self.put_self();
        let deferred = self.combine_deferred_drop_in_place();
        self.combine_fuse(deferred);
        self.return_()
    }

    fn build_fused_async_surface(mut self) -> Body<'tcx> {
        self.put_self();
        let surface = self.combine_async_surface();
        self.combine_fuse(surface);
        self.return_()
    }

    fn build_fused_sync_surface(mut self) -> Body<'tcx> {
        self.put_self();
        let surface = self.combine_sync_surface();
        self.combine_fuse(surface);
        self.return_()
    }

    fn build_slice(mut self, is_array: bool, elem_ty: Ty<'tcx>) -> Body<'tcx> {
        if is_array {
            self.put_array_as_slice(elem_ty)
        } else {
            self.put_self()
        }
        let dtor = self.combine_slice(elem_ty);
        self.combine_fuse(dtor);
        self.return_()
    }

    fn put_zst_output(&mut self) {
        let return_ty = self.locals[RETURN_PLACE].ty;
        self.put_operand(Operand::Constant(Box::new(ConstOperand {
            span: self.span,
            user_ty: None,
            const_: Const::zero_sized(return_ty),
        })));
    }

    /// Puts `to_drop: *mut Self` on top of the stack.
    fn put_self(&mut self) {
        self.put_operand(Operand::Copy(Self::SELF_PTR.into()))
    }

    /// Given that `Self is [ElemTy; N]` puts `to_drop: *mut [ElemTy]`
    /// on top of the stack.
    fn put_array_as_slice(&mut self, elem_ty: Ty<'tcx>) {
        let slice_ptr_ty = Ty::new_mut_ptr(self.tcx, Ty::new_slice(self.tcx, elem_ty));
        self.put_temp_rvalue(Rvalue::Cast(
            CastKind::PointerCoercion(PointerCoercion::Unsize, CoercionSource::Implicit),
            Operand::Copy(Self::SELF_PTR.into()),
            slice_ptr_ty,
        ))
    }

    /// If given Self is a struct puts `to_drop: *mut FieldTy` on top
    /// of the stack.
    fn put_field(&mut self, field: FieldIdx, field_ty: Ty<'tcx>) {
        let place = Place {
            local: Self::SELF_PTR,
            projection: self
                .tcx
                .mk_place_elems(&[PlaceElem::Deref, PlaceElem::Field(field, field_ty)]),
        };
        self.put_temp_rvalue(Rvalue::RawPtr(Mutability::Mut, place))
    }

    /// If given Self is an enum puts `to_drop: *mut FieldTy` on top of
    /// the stack.
    fn put_variant_field(
        &mut self,
        variant_sym: Symbol,
        variant: VariantIdx,
        field: FieldIdx,
        field_ty: Ty<'tcx>,
    ) {
        let place = Place {
            local: Self::SELF_PTR,
            projection: self.tcx.mk_place_elems(&[
                PlaceElem::Deref,
                PlaceElem::Downcast(Some(variant_sym), variant),
                PlaceElem::Field(field, field_ty),
            ]),
        };
        self.put_temp_rvalue(Rvalue::RawPtr(Mutability::Mut, place))
    }

    /// If given Self is an enum puts `to_drop: *mut FieldTy` on top of
    /// the stack.
    fn put_discr(&mut self, discr: Discr<'tcx>) {
        let (size, _) = discr.ty.int_size_and_signed(self.tcx);
        self.put_operand(Operand::const_from_scalar(
            self.tcx,
            discr.ty,
            interpret::Scalar::from_uint(discr.val, size),
            self.span,
        ));
    }

    /// Puts `x: RvalueType` on top of the stack.
    fn put_temp_rvalue(&mut self, rvalue: Rvalue<'tcx>) {
        let last_bb = &mut self.bbs[self.last_bb];
        debug_assert!(last_bb.terminator.is_none());
        let source_info = self.source_info;

        let local_ty = rvalue.ty(&self.locals, self.tcx);
        // We need to create a new local to be able to "consume" it with
        // a combinator
        let local = self.locals.push(LocalDecl::with_source_info(local_ty, source_info));
        last_bb.statements.extend_from_slice(&[
            Statement { source_info, kind: StatementKind::StorageLive(local) },
            Statement {
                source_info,
                kind: StatementKind::Assign(Box::new((local.into(), rvalue))),
            },
        ]);

        self.put_operand(Operand::Move(local.into()));
    }

    /// Puts operand on top of the stack.
    fn put_operand(&mut self, operand: Operand<'tcx>) {
        if let Some(top_cleanup_bb) = &mut self.top_cleanup_bb {
            let source_info = self.source_info;
            match &operand {
                Operand::Copy(_) | Operand::Constant(_) => {
                    *top_cleanup_bb = self.bbs.push(BasicBlockData {
                        statements: Vec::new(),
                        terminator: Some(Terminator {
                            source_info,
                            kind: TerminatorKind::Goto { target: *top_cleanup_bb },
                        }),
                        is_cleanup: true,
                    });
                }
                Operand::Move(place) => {
                    let local = place.as_local().unwrap();
                    *top_cleanup_bb = self.bbs.push(BasicBlockData {
                        statements: Vec::new(),
                        terminator: Some(Terminator {
                            source_info,
                            kind: if self.locals[local].ty.needs_drop(self.tcx, self.param_env) {
                                TerminatorKind::Drop {
                                    place: local.into(),
                                    target: *top_cleanup_bb,
                                    unwind: UnwindAction::Terminate(
                                        UnwindTerminateReason::InCleanup,
                                    ),
                                    replace: false,
                                }
                            } else {
                                TerminatorKind::Goto { target: *top_cleanup_bb }
                            },
                        }),
                        is_cleanup: true,
                    });
                }
            };
        }
        self.stack.push(operand);
    }

    /// Puts `noop: async_drop::Noop` on top of the stack
    fn put_noop(&mut self) -> Ty<'tcx> {
        self.apply_combinator(0, LangItem::AsyncDropNoop, &[])
    }

    fn combine_async_surface(&mut self) -> Ty<'tcx> {
        self.apply_combinator(1, LangItem::SurfaceAsyncDropInPlace, &[self.self_ty.unwrap().into()])
    }

    fn combine_sync_surface(&mut self) -> Ty<'tcx> {
        self.apply_combinator(1, LangItem::AsyncDropSurfaceDropInPlace, &[self
            .self_ty
            .unwrap()
            .into()])
    }

    fn combine_deferred_drop_in_place(&mut self) -> Ty<'tcx> {
        self.apply_combinator(1, LangItem::AsyncDropDeferredDropInPlace, &[self
            .self_ty
            .unwrap()
            .into()])
    }

    fn combine_fuse(&mut self, inner_future_ty: Ty<'tcx>) -> Ty<'tcx> {
        self.apply_combinator(1, LangItem::AsyncDropFuse, &[inner_future_ty.into()])
    }

    fn combine_slice(&mut self, elem_ty: Ty<'tcx>) -> Ty<'tcx> {
        self.apply_combinator(1, LangItem::AsyncDropSlice, &[elem_ty.into()])
    }

    fn combine_defer(&mut self, to_drop_ty: Ty<'tcx>) -> Ty<'tcx> {
        self.apply_combinator(1, LangItem::AsyncDropDefer, &[to_drop_ty.into()])
    }

    fn combine_chain(&mut self, first: Ty<'tcx>, second: Ty<'tcx>) -> Ty<'tcx> {
        self.apply_combinator(2, LangItem::AsyncDropChain, &[first.into(), second.into()])
    }

    fn combine_either(&mut self, other: Ty<'tcx>, matched: Ty<'tcx>) -> Ty<'tcx> {
        self.apply_combinator(4, LangItem::AsyncDropEither, &[
            other.into(),
            matched.into(),
            self.self_ty.unwrap().into(),
        ])
    }

    fn return_(mut self) -> Body<'tcx> {
        let last_bb = &mut self.bbs[self.last_bb];
        debug_assert!(last_bb.terminator.is_none());
        let source_info = self.source_info;

        let (1, Some(output)) = (self.stack.len(), self.stack.pop()) else {
            span_bug!(
                self.span,
                "async destructor ctor shim builder finished with invalid number of stack items: expected 1 found {}",
                self.stack.len(),
            )
        };
        #[cfg(debug_assertions)]
        if let Some(ty) = self.self_ty {
            debug_assert_eq!(
                output.ty(&self.locals, self.tcx),
                ty.async_destructor_ty(self.tcx),
                "output async destructor types did not match for type: {ty:?}",
            );
        }

        let dead_storage = match &output {
            Operand::Move(place) => Some(Statement {
                source_info,
                kind: StatementKind::StorageDead(place.as_local().unwrap()),
            }),
            _ => None,
        };

        last_bb.statements.extend(
            iter::once(Statement {
                source_info,
                kind: StatementKind::Assign(Box::new((RETURN_PLACE.into(), Rvalue::Use(output)))),
            })
            .chain(dead_storage),
        );

        last_bb.terminator = Some(Terminator { source_info, kind: TerminatorKind::Return });

        let source = MirSource::from_instance(ty::InstanceKind::AsyncDropGlueCtorShim(
            self.def_id,
            self.self_ty,
        ));
        new_body(source, self.bbs, self.locals, Self::INPUT_COUNT, self.span)
    }

    fn apply_combinator(
        &mut self,
        arity: usize,
        function: LangItem,
        args: &[ty::GenericArg<'tcx>],
    ) -> Ty<'tcx> {
        let function = self.tcx.require_lang_item(function, Some(self.span));
        let operands_split = self
            .stack
            .len()
            .checked_sub(arity)
            .expect("async destructor ctor shim combinator tried to consume too many items");
        let operands = &self.stack[operands_split..];

        let func_ty = Ty::new_fn_def(self.tcx, function, args.iter().copied());
        let func_sig = func_ty.fn_sig(self.tcx).no_bound_vars().unwrap();
        #[cfg(debug_assertions)]
        operands.iter().zip(func_sig.inputs()).for_each(|(operand, expected_ty)| {
            let operand_ty = operand.ty(&self.locals, self.tcx);
            if operand_ty == *expected_ty {
                return;
            }

            // If projection of Discriminant then compare with `Ty::discriminant_ty`
            if let ty::Alias(ty::Projection, ty::AliasTy { args, def_id, .. }) = expected_ty.kind()
                && self.tcx.is_lang_item(*def_id, LangItem::Discriminant)
                && args.first().unwrap().as_type().unwrap().discriminant_ty(self.tcx) == operand_ty
            {
                return;
            }

            span_bug!(
                self.span,
                "Operand type and combinator argument type are not equal.
    operand_ty: {:?}
    argument_ty: {:?}
",
                operand_ty,
                expected_ty
            );
        });

        let target = self.bbs.push(BasicBlockData {
            statements: operands
                .iter()
                .rev()
                .filter_map(|o| {
                    if let Operand::Move(Place { local, projection }) = o {
                        assert!(projection.is_empty());
                        Some(Statement {
                            source_info: self.source_info,
                            kind: StatementKind::StorageDead(*local),
                        })
                    } else {
                        None
                    }
                })
                .collect(),
            terminator: None,
            is_cleanup: false,
        });

        let dest_ty = func_sig.output();
        let dest =
            self.locals.push(LocalDecl::with_source_info(dest_ty, self.source_info).immutable());

        let unwind = if let Some(top_cleanup_bb) = &mut self.top_cleanup_bb {
            for _ in 0..arity {
                *top_cleanup_bb =
                    self.bbs[*top_cleanup_bb].terminator().successors().exactly_one().ok().unwrap();
            }
            UnwindAction::Cleanup(*top_cleanup_bb)
        } else {
            UnwindAction::Unreachable
        };

        let last_bb = &mut self.bbs[self.last_bb];
        debug_assert!(last_bb.terminator.is_none());
        last_bb.statements.push(Statement {
            source_info: self.source_info,
            kind: StatementKind::StorageLive(dest),
        });
        last_bb.terminator = Some(Terminator {
            source_info: self.source_info,
            kind: TerminatorKind::Call {
                func: Operand::Constant(Box::new(ConstOperand {
                    span: self.span,
                    user_ty: None,
                    const_: Const::Val(ConstValue::ZeroSized, func_ty),
                })),
                destination: dest.into(),
                target: Some(target),
                unwind,
                call_source: CallSource::Misc,
                fn_span: self.span,
                args: self.stack.drain(operands_split..).map(|o| respan(self.span, o)).collect(),
            },
        });

        self.put_operand(Operand::Move(dest.into()));
        self.last_bb = target;

        dest_ty
    }
}
