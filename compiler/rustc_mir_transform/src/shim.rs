use std::assert_matches::assert_matches;
use std::{fmt, iter};

use rustc_abi::{ExternAbi, FIRST_VARIANT, FieldIdx, VariantIdx};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::*;
use rustc_middle::query::Providers;
use rustc_middle::ty::{
    self, CoroutineArgs, CoroutineArgsExt, EarlyBinder, GenericArgs, Ty, TyCtxt,
};
use rustc_middle::{bug, span_bug};
use rustc_span::source_map::Spanned;
use rustc_span::{DUMMY_SP, Span};
use tracing::{debug, instrument};

use crate::elaborate_drop::{DropElaborator, DropFlagMode, DropStyle, Unwind, elaborate_drop};
use crate::patch::MirPatch;
use crate::{
    abort_unwinding_calls, add_call_guards, add_moves_for_packed_drops, deref_separator, inline,
    instsimplify, mentioned_items, pass_manager as pm, remove_noop_landing_pads, simplify,
};

mod async_destructor_ctor;

pub(super) fn provide(providers: &mut Providers) {
    providers.mir_shims = make_shim;
}

fn make_shim<'tcx>(tcx: TyCtxt<'tcx>, instance: ty::InstanceKind<'tcx>) -> Body<'tcx> {
    debug!("make_shim({:?})", instance);

    let mut result = match instance {
        ty::InstanceKind::Item(..) => bug!("item {:?} passed to make_shim", instance),
        ty::InstanceKind::VTableShim(def_id) => {
            let adjustment = Adjustment::Deref { source: DerefSource::MutPtr };
            build_call_shim(tcx, instance, Some(adjustment), CallKind::Direct(def_id))
        }
        ty::InstanceKind::FnPtrShim(def_id, ty) => {
            let trait_ = tcx.trait_of_item(def_id).unwrap();
            // Supports `Fn` or `async Fn` traits.
            let adjustment = match tcx
                .fn_trait_kind_from_def_id(trait_)
                .or_else(|| tcx.async_fn_trait_kind_from_def_id(trait_))
            {
                Some(ty::ClosureKind::FnOnce) => Adjustment::Identity,
                Some(ty::ClosureKind::Fn) => Adjustment::Deref { source: DerefSource::ImmRef },
                Some(ty::ClosureKind::FnMut) => Adjustment::Deref { source: DerefSource::MutRef },
                None => bug!("fn pointer {:?} is not an fn", ty),
            };

            build_call_shim(tcx, instance, Some(adjustment), CallKind::Indirect(ty))
        }
        // We are generating a call back to our def-id, which the
        // codegen backend knows to turn to an actual call, be it
        // a virtual call, or a direct call to a function for which
        // indirect calls must be codegen'd differently than direct ones
        // (such as `#[track_caller]`).
        ty::InstanceKind::ReifyShim(def_id, _) => {
            build_call_shim(tcx, instance, None, CallKind::Direct(def_id))
        }
        ty::InstanceKind::ClosureOnceShim { call_once: _, track_caller: _ } => {
            let fn_mut = tcx.require_lang_item(LangItem::FnMut, None);
            let call_mut = tcx
                .associated_items(fn_mut)
                .in_definition_order()
                .find(|it| it.is_fn())
                .unwrap()
                .def_id;

            build_call_shim(tcx, instance, Some(Adjustment::RefMut), CallKind::Direct(call_mut))
        }

        ty::InstanceKind::ConstructCoroutineInClosureShim {
            coroutine_closure_def_id,
            receiver_by_ref,
        } => build_construct_coroutine_by_move_shim(tcx, coroutine_closure_def_id, receiver_by_ref),

        ty::InstanceKind::DropGlue(def_id, ty) => {
            // FIXME(#91576): Drop shims for coroutines aren't subject to the MIR passes at the end
            // of this function. Is this intentional?
            if let Some(&ty::Coroutine(coroutine_def_id, args)) = ty.map(Ty::kind) {
                let coroutine_body = tcx.optimized_mir(coroutine_def_id);

                let ty::Coroutine(_, id_args) = *tcx.type_of(coroutine_def_id).skip_binder().kind()
                else {
                    bug!()
                };

                // If this is a regular coroutine, grab its drop shim. If this is a coroutine
                // that comes from a coroutine-closure, and the kind ty differs from the "maximum"
                // kind that it supports, then grab the appropriate drop shim. This ensures that
                // the future returned by `<[coroutine-closure] as AsyncFnOnce>::call_once` will
                // drop the coroutine-closure's upvars.
                let body = if id_args.as_coroutine().kind_ty() == args.as_coroutine().kind_ty() {
                    coroutine_body.coroutine_drop().unwrap()
                } else {
                    assert_eq!(
                        args.as_coroutine().kind_ty().to_opt_closure_kind().unwrap(),
                        ty::ClosureKind::FnOnce
                    );
                    tcx.optimized_mir(tcx.coroutine_by_move_body_def_id(coroutine_def_id))
                        .coroutine_drop()
                        .unwrap()
                };

                let mut body = EarlyBinder::bind(body.clone()).instantiate(tcx, args);
                debug!("make_shim({:?}) = {:?}", instance, body);

                pm::run_passes(
                    tcx,
                    &mut body,
                    &[
                        &mentioned_items::MentionedItems,
                        &abort_unwinding_calls::AbortUnwindingCalls,
                        &add_call_guards::CriticalCallEdges,
                    ],
                    Some(MirPhase::Runtime(RuntimePhase::Optimized)),
                    pm::Optimizations::Allowed,
                );

                return body;
            }

            build_drop_shim(tcx, def_id, ty)
        }
        ty::InstanceKind::ThreadLocalShim(..) => build_thread_local_shim(tcx, instance),
        ty::InstanceKind::CloneShim(def_id, ty) => build_clone_shim(tcx, def_id, ty),
        ty::InstanceKind::FnPtrAddrShim(def_id, ty) => build_fn_ptr_addr_shim(tcx, def_id, ty),
        ty::InstanceKind::AsyncDropGlueCtorShim(def_id, ty) => {
            async_destructor_ctor::build_async_destructor_ctor_shim(tcx, def_id, ty)
        }
        ty::InstanceKind::Virtual(..) => {
            bug!("InstanceKind::Virtual ({:?}) is for direct calls only", instance)
        }
        ty::InstanceKind::Intrinsic(_) => {
            bug!("creating shims from intrinsics ({:?}) is unsupported", instance)
        }
    };
    debug!("make_shim({:?}) = untransformed {:?}", instance, result);

    // We don't validate MIR here because the shims may generate code that's
    // only valid in a `PostAnalysis` param-env. However, since we do initial
    // validation with the MirBuilt phase, which uses a user-facing param-env.
    // This causes validation errors when TAITs are involved.
    pm::run_passes_no_validate(
        tcx,
        &mut result,
        &[
            &mentioned_items::MentionedItems,
            &add_moves_for_packed_drops::AddMovesForPackedDrops,
            &deref_separator::Derefer,
            &remove_noop_landing_pads::RemoveNoopLandingPads,
            &simplify::SimplifyCfg::MakeShim,
            &instsimplify::InstSimplify::BeforeInline,
            // Perform inlining of `#[rustc_force_inline]`-annotated callees.
            &inline::ForceInline,
            &abort_unwinding_calls::AbortUnwindingCalls,
            &add_call_guards::CriticalCallEdges,
        ],
        Some(MirPhase::Runtime(RuntimePhase::Optimized)),
    );

    debug!("make_shim({:?}) = {:?}", instance, result);

    result
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum DerefSource {
    /// `fn shim(&self) { inner(*self )}`.
    ImmRef,
    /// `fn shim(&mut self) { inner(*self )}`.
    MutRef,
    /// `fn shim(*mut self) { inner(*self )}`.
    MutPtr,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Adjustment {
    /// Pass the receiver as-is.
    Identity,

    /// We get passed a reference or a raw pointer to `self` and call the target with `*self`.
    ///
    /// This either copies `self` (if `Self: Copy`, eg. for function items), or moves out of it
    /// (for `VTableShim`, which effectively is passed `&own Self`).
    Deref { source: DerefSource },

    /// We get passed `self: Self` and call the target with `&mut self`.
    ///
    /// In this case we need to ensure that the `Self` is dropped after the call, as the callee
    /// won't do it for us.
    RefMut,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum CallKind<'tcx> {
    /// Call the `FnPtr` that was passed as the receiver.
    Indirect(Ty<'tcx>),

    /// Call a known `FnDef`.
    Direct(DefId),
}

fn local_decls_for_sig<'tcx>(
    sig: &ty::FnSig<'tcx>,
    span: Span,
) -> IndexVec<Local, LocalDecl<'tcx>> {
    iter::once(LocalDecl::new(sig.output(), span))
        .chain(sig.inputs().iter().map(|ity| LocalDecl::new(*ity, span).immutable()))
        .collect()
}

fn build_drop_shim<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, ty: Option<Ty<'tcx>>) -> Body<'tcx> {
    debug!("build_drop_shim(def_id={:?}, ty={:?})", def_id, ty);

    assert!(!matches!(ty, Some(ty) if ty.is_coroutine()));

    let args = if let Some(ty) = ty {
        tcx.mk_args(&[ty.into()])
    } else {
        GenericArgs::identity_for_item(tcx, def_id)
    };
    let sig = tcx.fn_sig(def_id).instantiate(tcx, args);
    let sig = tcx.instantiate_bound_regions_with_erased(sig);
    let span = tcx.def_span(def_id);

    let source_info = SourceInfo::outermost(span);

    let return_block = BasicBlock::new(1);
    let mut blocks = IndexVec::with_capacity(2);
    let block = |blocks: &mut IndexVec<_, _>, kind| {
        blocks.push(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator { source_info, kind }),
            is_cleanup: false,
        })
    };
    block(&mut blocks, TerminatorKind::Goto { target: return_block });
    block(&mut blocks, TerminatorKind::Return);

    let source = MirSource::from_instance(ty::InstanceKind::DropGlue(def_id, ty));
    let mut body =
        new_body(source, blocks, local_decls_for_sig(&sig, span), sig.inputs().len(), span);

    // The first argument (index 0), but add 1 for the return value.
    let mut dropee_ptr = Place::from(Local::new(1 + 0));
    if tcx.sess.opts.unstable_opts.mir_emit_retag {
        // We want to treat the function argument as if it was passed by `&mut`. As such, we
        // generate
        // ```
        // temp = &mut *arg;
        // Retag(temp, FnEntry)
        // ```
        // It's important that we do this first, before anything that depends on `dropee_ptr`
        // has been put into the body.
        let reborrow = Rvalue::Ref(
            tcx.lifetimes.re_erased,
            BorrowKind::Mut { kind: MutBorrowKind::Default },
            tcx.mk_place_deref(dropee_ptr),
        );
        let ref_ty = reborrow.ty(body.local_decls(), tcx);
        dropee_ptr = body.local_decls.push(LocalDecl::new(ref_ty, span)).into();
        let new_statements = [
            StatementKind::Assign(Box::new((dropee_ptr, reborrow))),
            StatementKind::Retag(RetagKind::FnEntry, Box::new(dropee_ptr)),
        ];
        for s in new_statements {
            body.basic_blocks_mut()[START_BLOCK]
                .statements
                .push(Statement { source_info, kind: s });
        }
    }

    if ty.is_some() {
        let patch = {
            let typing_env = ty::TypingEnv::post_analysis(tcx, def_id);
            let mut elaborator =
                DropShimElaborator { body: &body, patch: MirPatch::new(&body), tcx, typing_env };
            let dropee = tcx.mk_place_deref(dropee_ptr);
            let resume_block = elaborator.patch.resume_block();
            elaborate_drop(
                &mut elaborator,
                source_info,
                dropee,
                (),
                return_block,
                Unwind::To(resume_block),
                START_BLOCK,
            );
            elaborator.patch
        };
        patch.apply(&mut body);
    }

    body
}

fn new_body<'tcx>(
    source: MirSource<'tcx>,
    basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
    local_decls: IndexVec<Local, LocalDecl<'tcx>>,
    arg_count: usize,
    span: Span,
) -> Body<'tcx> {
    let mut body = Body::new(
        source,
        basic_blocks,
        IndexVec::from_elem_n(
            SourceScopeData {
                span,
                parent_scope: None,
                inlined: None,
                inlined_parent_scope: None,
                local_data: ClearCrossCrate::Clear,
            },
            1,
        ),
        local_decls,
        IndexVec::new(),
        arg_count,
        vec![],
        span,
        None,
        // FIXME(compiler-errors): is this correct?
        None,
    );
    // Shims do not directly mention any consts.
    body.set_required_consts(Vec::new());
    body
}

pub(super) struct DropShimElaborator<'a, 'tcx> {
    pub body: &'a Body<'tcx>,
    pub patch: MirPatch<'tcx>,
    pub tcx: TyCtxt<'tcx>,
    pub typing_env: ty::TypingEnv<'tcx>,
}

impl fmt::Debug for DropShimElaborator<'_, '_> {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        Ok(())
    }
}

impl<'a, 'tcx> DropElaborator<'a, 'tcx> for DropShimElaborator<'a, 'tcx> {
    type Path = ();

    fn patch_ref(&self) -> &MirPatch<'tcx> {
        &self.patch
    }
    fn patch(&mut self) -> &mut MirPatch<'tcx> {
        &mut self.patch
    }
    fn body(&self) -> &'a Body<'tcx> {
        self.body
    }
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        self.typing_env
    }

    fn drop_style(&self, _path: Self::Path, mode: DropFlagMode) -> DropStyle {
        match mode {
            DropFlagMode::Shallow => {
                // Drops for the contained fields are "shallow" and "static" - they will simply call
                // the field's own drop glue.
                DropStyle::Static
            }
            DropFlagMode::Deep => {
                // The top-level drop is "deep" and "open" - it will be elaborated to a drop ladder
                // dropping each field contained in the value.
                DropStyle::Open
            }
        }
    }

    fn get_drop_flag(&mut self, _path: Self::Path) -> Option<Operand<'tcx>> {
        None
    }

    fn clear_drop_flag(&mut self, _location: Location, _path: Self::Path, _mode: DropFlagMode) {}

    fn field_subpath(&self, _path: Self::Path, _field: FieldIdx) -> Option<Self::Path> {
        None
    }
    fn deref_subpath(&self, _path: Self::Path) -> Option<Self::Path> {
        None
    }
    fn downcast_subpath(&self, _path: Self::Path, _variant: VariantIdx) -> Option<Self::Path> {
        Some(())
    }
    fn array_subpath(&self, _path: Self::Path, _index: u64, _size: u64) -> Option<Self::Path> {
        None
    }
}

fn build_thread_local_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::InstanceKind<'tcx>,
) -> Body<'tcx> {
    let def_id = instance.def_id();

    let span = tcx.def_span(def_id);
    let source_info = SourceInfo::outermost(span);

    let blocks = IndexVec::from_raw(vec![BasicBlockData {
        statements: vec![Statement {
            source_info,
            kind: StatementKind::Assign(Box::new((
                Place::return_place(),
                Rvalue::ThreadLocalRef(def_id),
            ))),
        }],
        terminator: Some(Terminator { source_info, kind: TerminatorKind::Return }),
        is_cleanup: false,
    }]);

    new_body(
        MirSource::from_instance(instance),
        blocks,
        IndexVec::from_raw(vec![LocalDecl::new(tcx.thread_local_ptr_ty(def_id), span)]),
        0,
        span,
    )
}

/// Builds a `Clone::clone` shim for `self_ty`. Here, `def_id` is `Clone::clone`.
fn build_clone_shim<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, self_ty: Ty<'tcx>) -> Body<'tcx> {
    debug!("build_clone_shim(def_id={:?})", def_id);

    let mut builder = CloneShimBuilder::new(tcx, def_id, self_ty);

    let dest = Place::return_place();
    let src = tcx.mk_place_deref(Place::from(Local::new(1 + 0)));

    match self_ty.kind() {
        ty::FnDef(..) | ty::FnPtr(..) => builder.copy_shim(),
        ty::Closure(_, args) => builder.tuple_like_shim(dest, src, args.as_closure().upvar_tys()),
        ty::CoroutineClosure(_, args) => {
            builder.tuple_like_shim(dest, src, args.as_coroutine_closure().upvar_tys())
        }
        ty::Tuple(..) => builder.tuple_like_shim(dest, src, self_ty.tuple_fields()),
        ty::Coroutine(coroutine_def_id, args) => {
            assert_eq!(tcx.coroutine_movability(*coroutine_def_id), hir::Movability::Movable);
            builder.coroutine_shim(dest, src, *coroutine_def_id, args.as_coroutine())
        }
        _ => bug!("clone shim for `{:?}` which is not `Copy` and is not an aggregate", self_ty),
    };

    builder.into_mir()
}

struct CloneShimBuilder<'tcx> {
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    local_decls: IndexVec<Local, LocalDecl<'tcx>>,
    blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
    span: Span,
    sig: ty::FnSig<'tcx>,
}

impl<'tcx> CloneShimBuilder<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, def_id: DefId, self_ty: Ty<'tcx>) -> Self {
        // we must instantiate the self_ty because it's
        // otherwise going to be TySelf and we can't index
        // or access fields of a Place of type TySelf.
        let sig = tcx.fn_sig(def_id).instantiate(tcx, &[self_ty.into()]);
        let sig = tcx.instantiate_bound_regions_with_erased(sig);
        let span = tcx.def_span(def_id);

        CloneShimBuilder {
            tcx,
            def_id,
            local_decls: local_decls_for_sig(&sig, span),
            blocks: IndexVec::new(),
            span,
            sig,
        }
    }

    fn into_mir(self) -> Body<'tcx> {
        let source = MirSource::from_instance(ty::InstanceKind::CloneShim(
            self.def_id,
            self.sig.inputs_and_output[0],
        ));
        new_body(source, self.blocks, self.local_decls, self.sig.inputs().len(), self.span)
    }

    fn source_info(&self) -> SourceInfo {
        SourceInfo::outermost(self.span)
    }

    fn block(
        &mut self,
        statements: Vec<Statement<'tcx>>,
        kind: TerminatorKind<'tcx>,
        is_cleanup: bool,
    ) -> BasicBlock {
        let source_info = self.source_info();
        self.blocks.push(BasicBlockData {
            statements,
            terminator: Some(Terminator { source_info, kind }),
            is_cleanup,
        })
    }

    /// Gives the index of an upcoming BasicBlock, with an offset.
    /// offset=0 will give you the index of the next BasicBlock,
    /// offset=1 will give the index of the next-to-next block,
    /// offset=-1 will give you the index of the last-created block
    fn block_index_offset(&self, offset: usize) -> BasicBlock {
        BasicBlock::new(self.blocks.len() + offset)
    }

    fn make_statement(&self, kind: StatementKind<'tcx>) -> Statement<'tcx> {
        Statement { source_info: self.source_info(), kind }
    }

    fn copy_shim(&mut self) {
        let rcvr = self.tcx.mk_place_deref(Place::from(Local::new(1 + 0)));
        let ret_statement = self.make_statement(StatementKind::Assign(Box::new((
            Place::return_place(),
            Rvalue::Use(Operand::Copy(rcvr)),
        ))));
        self.block(vec![ret_statement], TerminatorKind::Return, false);
    }

    fn make_place(&mut self, mutability: Mutability, ty: Ty<'tcx>) -> Place<'tcx> {
        let span = self.span;
        let mut local = LocalDecl::new(ty, span);
        if mutability.is_not() {
            local = local.immutable();
        }
        Place::from(self.local_decls.push(local))
    }

    fn make_clone_call(
        &mut self,
        dest: Place<'tcx>,
        src: Place<'tcx>,
        ty: Ty<'tcx>,
        next: BasicBlock,
        cleanup: BasicBlock,
    ) {
        let tcx = self.tcx;

        // `func == Clone::clone(&ty) -> ty`
        let func_ty = Ty::new_fn_def(tcx, self.def_id, [ty]);
        let func = Operand::Constant(Box::new(ConstOperand {
            span: self.span,
            user_ty: None,
            const_: Const::zero_sized(func_ty),
        }));

        let ref_loc =
            self.make_place(Mutability::Not, Ty::new_imm_ref(tcx, tcx.lifetimes.re_erased, ty));

        // `let ref_loc: &ty = &src;`
        let statement = self.make_statement(StatementKind::Assign(Box::new((
            ref_loc,
            Rvalue::Ref(tcx.lifetimes.re_erased, BorrowKind::Shared, src),
        ))));

        // `let loc = Clone::clone(ref_loc);`
        self.block(
            vec![statement],
            TerminatorKind::Call {
                func,
                args: [Spanned { node: Operand::Move(ref_loc), span: DUMMY_SP }].into(),
                destination: dest,
                target: Some(next),
                unwind: UnwindAction::Cleanup(cleanup),
                call_source: CallSource::Normal,
                fn_span: self.span,
            },
            false,
        );
    }

    fn clone_fields<I>(
        &mut self,
        dest: Place<'tcx>,
        src: Place<'tcx>,
        target: BasicBlock,
        mut unwind: BasicBlock,
        tys: I,
    ) -> BasicBlock
    where
        I: IntoIterator<Item = Ty<'tcx>>,
    {
        // For an iterator of length n, create 2*n + 1 blocks.
        for (i, ity) in tys.into_iter().enumerate() {
            // Each iteration creates two blocks, referred to here as block 2*i and block 2*i + 1.
            //
            // Block 2*i attempts to clone the field. If successful it branches to 2*i + 2 (the
            // next clone block). If unsuccessful it branches to the previous unwind block, which
            // is initially the `unwind` argument passed to this function.
            //
            // Block 2*i + 1 is the unwind block for this iteration. It drops the cloned value
            // created by block 2*i. We store this block in `unwind` so that the next clone block
            // will unwind to it if cloning fails.

            let field = FieldIdx::new(i);
            let src_field = self.tcx.mk_place_field(src, field, ity);

            let dest_field = self.tcx.mk_place_field(dest, field, ity);

            let next_unwind = self.block_index_offset(1);
            let next_block = self.block_index_offset(2);
            self.make_clone_call(dest_field, src_field, ity, next_block, unwind);
            self.block(
                vec![],
                TerminatorKind::Drop {
                    place: dest_field,
                    target: unwind,
                    unwind: UnwindAction::Terminate(UnwindTerminateReason::InCleanup),
                    replace: false,
                },
                /* is_cleanup */ true,
            );
            unwind = next_unwind;
        }
        // If all clones succeed then we end up here.
        self.block(vec![], TerminatorKind::Goto { target }, false);
        unwind
    }

    fn tuple_like_shim<I>(&mut self, dest: Place<'tcx>, src: Place<'tcx>, tys: I)
    where
        I: IntoIterator<Item = Ty<'tcx>>,
    {
        self.block(vec![], TerminatorKind::Goto { target: self.block_index_offset(3) }, false);
        let unwind = self.block(vec![], TerminatorKind::UnwindResume, true);
        let target = self.block(vec![], TerminatorKind::Return, false);

        let _final_cleanup_block = self.clone_fields(dest, src, target, unwind, tys);
    }

    fn coroutine_shim(
        &mut self,
        dest: Place<'tcx>,
        src: Place<'tcx>,
        coroutine_def_id: DefId,
        args: CoroutineArgs<TyCtxt<'tcx>>,
    ) {
        self.block(vec![], TerminatorKind::Goto { target: self.block_index_offset(3) }, false);
        let unwind = self.block(vec![], TerminatorKind::UnwindResume, true);
        // This will get overwritten with a switch once we know the target blocks
        let switch = self.block(vec![], TerminatorKind::Unreachable, false);
        let unwind = self.clone_fields(dest, src, switch, unwind, args.upvar_tys());
        let target = self.block(vec![], TerminatorKind::Return, false);
        let unreachable = self.block(vec![], TerminatorKind::Unreachable, false);
        let mut cases = Vec::with_capacity(args.state_tys(coroutine_def_id, self.tcx).count());
        for (index, state_tys) in args.state_tys(coroutine_def_id, self.tcx).enumerate() {
            let variant_index = VariantIdx::new(index);
            let dest = self.tcx.mk_place_downcast_unnamed(dest, variant_index);
            let src = self.tcx.mk_place_downcast_unnamed(src, variant_index);
            let clone_block = self.block_index_offset(1);
            let start_block = self.block(
                vec![self.make_statement(StatementKind::SetDiscriminant {
                    place: Box::new(Place::return_place()),
                    variant_index,
                })],
                TerminatorKind::Goto { target: clone_block },
                false,
            );
            cases.push((index as u128, start_block));
            let _final_cleanup_block = self.clone_fields(dest, src, target, unwind, state_tys);
        }
        let discr_ty = args.discr_ty(self.tcx);
        let temp = self.make_place(Mutability::Mut, discr_ty);
        let rvalue = Rvalue::Discriminant(src);
        let statement = self.make_statement(StatementKind::Assign(Box::new((temp, rvalue))));
        match &mut self.blocks[switch] {
            BasicBlockData { statements, terminator: Some(Terminator { kind, .. }), .. } => {
                statements.push(statement);
                *kind = TerminatorKind::SwitchInt {
                    discr: Operand::Move(temp),
                    targets: SwitchTargets::new(cases.into_iter(), unreachable),
                };
            }
            BasicBlockData { terminator: None, .. } => unreachable!(),
        }
    }
}

/// Builds a "call" shim for `instance`. The shim calls the function specified by `call_kind`,
/// first adjusting its first argument according to `rcvr_adjustment`.
#[instrument(level = "debug", skip(tcx), ret)]
fn build_call_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::InstanceKind<'tcx>,
    rcvr_adjustment: Option<Adjustment>,
    call_kind: CallKind<'tcx>,
) -> Body<'tcx> {
    // `FnPtrShim` contains the fn pointer type that a call shim is being built for - this is used
    // to instantiate into the signature of the shim. It is not necessary for users of this
    // MIR body to perform further instantiations (see `InstanceKind::has_polymorphic_mir_body`).
    let (sig_args, untuple_args) = if let ty::InstanceKind::FnPtrShim(_, ty) = instance {
        let sig = tcx.instantiate_bound_regions_with_erased(ty.fn_sig(tcx));

        let untuple_args = sig.inputs();

        // Create substitutions for the `Self` and `Args` generic parameters of the shim body.
        let arg_tup = Ty::new_tup(tcx, untuple_args);

        (Some([ty.into(), arg_tup.into()]), Some(untuple_args))
    } else {
        (None, None)
    };

    let def_id = instance.def_id();

    let sig = tcx.fn_sig(def_id);
    let sig = sig.map_bound(|sig| tcx.instantiate_bound_regions_with_erased(sig));

    assert_eq!(sig_args.is_some(), !instance.has_polymorphic_mir_body());
    let mut sig = if let Some(sig_args) = sig_args {
        sig.instantiate(tcx, &sig_args)
    } else {
        sig.instantiate_identity()
    };

    if let CallKind::Indirect(fnty) = call_kind {
        // `sig` determines our local decls, and thus the callee type in the `Call` terminator. This
        // can only be an `FnDef` or `FnPtr`, but currently will be `Self` since the types come from
        // the implemented `FnX` trait.

        // Apply the opposite adjustment to the MIR input.
        let mut inputs_and_output = sig.inputs_and_output.to_vec();

        // Initial signature is `fn(&? Self, Args) -> Self::Output` where `Args` is a tuple of the
        // fn arguments. `Self` may be passed via (im)mutable reference or by-value.
        assert_eq!(inputs_and_output.len(), 3);

        // `Self` is always the original fn type `ty`. The MIR call terminator is only defined for
        // `FnDef` and `FnPtr` callees, not the `Self` type param.
        let self_arg = &mut inputs_and_output[0];
        *self_arg = match rcvr_adjustment.unwrap() {
            Adjustment::Identity => fnty,
            Adjustment::Deref { source } => match source {
                DerefSource::ImmRef => Ty::new_imm_ref(tcx, tcx.lifetimes.re_erased, fnty),
                DerefSource::MutRef => Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, fnty),
                DerefSource::MutPtr => Ty::new_mut_ptr(tcx, fnty),
            },
            Adjustment::RefMut => bug!("`RefMut` is never used with indirect calls: {instance:?}"),
        };
        sig.inputs_and_output = tcx.mk_type_list(&inputs_and_output);
    }

    // FIXME: Avoid having to adjust the signature both here and in
    // `fn_sig_for_fn_abi`.
    if let ty::InstanceKind::VTableShim(..) = instance {
        // Modify fn(self, ...) to fn(self: *mut Self, ...)
        let mut inputs_and_output = sig.inputs_and_output.to_vec();
        let self_arg = &mut inputs_and_output[0];
        debug_assert!(tcx.generics_of(def_id).has_self && *self_arg == tcx.types.self_param);
        *self_arg = Ty::new_mut_ptr(tcx, *self_arg);
        sig.inputs_and_output = tcx.mk_type_list(&inputs_and_output);
    }

    let span = tcx.def_span(def_id);

    debug!(?sig);

    let mut local_decls = local_decls_for_sig(&sig, span);
    let source_info = SourceInfo::outermost(span);

    let destination = Place::return_place();

    let rcvr_place = || {
        assert!(rcvr_adjustment.is_some());
        Place::from(Local::new(1))
    };
    let mut statements = vec![];

    let rcvr = rcvr_adjustment.map(|rcvr_adjustment| match rcvr_adjustment {
        Adjustment::Identity => Operand::Move(rcvr_place()),
        Adjustment::Deref { source: _ } => Operand::Move(tcx.mk_place_deref(rcvr_place())),
        Adjustment::RefMut => {
            // let rcvr = &mut rcvr;
            let ref_rcvr = local_decls.push(
                LocalDecl::new(
                    Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, sig.inputs()[0]),
                    span,
                )
                .immutable(),
            );
            let borrow_kind = BorrowKind::Mut { kind: MutBorrowKind::Default };
            statements.push(Statement {
                source_info,
                kind: StatementKind::Assign(Box::new((
                    Place::from(ref_rcvr),
                    Rvalue::Ref(tcx.lifetimes.re_erased, borrow_kind, rcvr_place()),
                ))),
            });
            Operand::Move(Place::from(ref_rcvr))
        }
    });

    let (callee, mut args) = match call_kind {
        // `FnPtr` call has no receiver. Args are untupled below.
        CallKind::Indirect(_) => (rcvr.unwrap(), vec![]),

        // `FnDef` call with optional receiver.
        CallKind::Direct(def_id) => {
            let ty = tcx.type_of(def_id).instantiate_identity();
            (
                Operand::Constant(Box::new(ConstOperand {
                    span,
                    user_ty: None,
                    const_: Const::zero_sized(ty),
                })),
                rcvr.into_iter().collect::<Vec<_>>(),
            )
        }
    };

    let mut arg_range = 0..sig.inputs().len();

    // Take the `self` ("receiver") argument out of the range (it's adjusted above).
    if rcvr_adjustment.is_some() {
        arg_range.start += 1;
    }

    // Take the last argument, if we need to untuple it (handled below).
    if untuple_args.is_some() {
        arg_range.end -= 1;
    }

    // Pass all of the non-special arguments directly.
    args.extend(arg_range.map(|i| Operand::Move(Place::from(Local::new(1 + i)))));

    // Untuple the last argument, if we have to.
    if let Some(untuple_args) = untuple_args {
        let tuple_arg = Local::new(1 + (sig.inputs().len() - 1));
        args.extend(untuple_args.iter().enumerate().map(|(i, ity)| {
            Operand::Move(tcx.mk_place_field(Place::from(tuple_arg), FieldIdx::new(i), *ity))
        }));
    }

    let n_blocks = if let Some(Adjustment::RefMut) = rcvr_adjustment { 5 } else { 2 };
    let mut blocks = IndexVec::with_capacity(n_blocks);
    let block = |blocks: &mut IndexVec<_, _>, statements, kind, is_cleanup| {
        blocks.push(BasicBlockData {
            statements,
            terminator: Some(Terminator { source_info, kind }),
            is_cleanup,
        })
    };

    // BB #0
    let args = args.into_iter().map(|a| Spanned { node: a, span: DUMMY_SP }).collect();
    block(
        &mut blocks,
        statements,
        TerminatorKind::Call {
            func: callee,
            args,
            destination,
            target: Some(BasicBlock::new(1)),
            unwind: if let Some(Adjustment::RefMut) = rcvr_adjustment {
                UnwindAction::Cleanup(BasicBlock::new(3))
            } else {
                UnwindAction::Continue
            },
            call_source: CallSource::Misc,
            fn_span: span,
        },
        false,
    );

    if let Some(Adjustment::RefMut) = rcvr_adjustment {
        // BB #1 - drop for Self
        block(
            &mut blocks,
            vec![],
            TerminatorKind::Drop {
                place: rcvr_place(),
                target: BasicBlock::new(2),
                unwind: UnwindAction::Continue,
                replace: false,
            },
            false,
        );
    }
    // BB #1/#2 - return
    let stmts = vec![];
    block(&mut blocks, stmts, TerminatorKind::Return, false);
    if let Some(Adjustment::RefMut) = rcvr_adjustment {
        // BB #3 - drop if closure panics
        block(
            &mut blocks,
            vec![],
            TerminatorKind::Drop {
                place: rcvr_place(),
                target: BasicBlock::new(4),
                unwind: UnwindAction::Terminate(UnwindTerminateReason::InCleanup),
                replace: false,
            },
            /* is_cleanup */ true,
        );

        // BB #4 - resume
        block(&mut blocks, vec![], TerminatorKind::UnwindResume, true);
    }

    let mut body =
        new_body(MirSource::from_instance(instance), blocks, local_decls, sig.inputs().len(), span);

    if let ExternAbi::RustCall = sig.abi {
        body.spread_arg = Some(Local::new(sig.inputs().len()));
    }

    body
}

pub(super) fn build_adt_ctor(tcx: TyCtxt<'_>, ctor_id: DefId) -> Body<'_> {
    debug_assert!(tcx.is_constructor(ctor_id));

    let typing_env = ty::TypingEnv::post_analysis(tcx, ctor_id);

    // Normalize the sig.
    let sig = tcx
        .fn_sig(ctor_id)
        .instantiate_identity()
        .no_bound_vars()
        .expect("LBR in ADT constructor signature");
    let sig = tcx.normalize_erasing_regions(typing_env, sig);

    let ty::Adt(adt_def, args) = sig.output().kind() else {
        bug!("unexpected type for ADT ctor {:?}", sig.output());
    };

    debug!("build_ctor: ctor_id={:?} sig={:?}", ctor_id, sig);

    let span = tcx.def_span(ctor_id);

    let local_decls = local_decls_for_sig(&sig, span);

    let source_info = SourceInfo::outermost(span);

    let variant_index =
        if adt_def.is_enum() { adt_def.variant_index_with_ctor_id(ctor_id) } else { FIRST_VARIANT };

    // Generate the following MIR:
    //
    // (return as Variant).field0 = arg0;
    // (return as Variant).field1 = arg1;
    //
    // return;
    debug!("build_ctor: variant_index={:?}", variant_index);

    let kind = AggregateKind::Adt(adt_def.did(), variant_index, args, None, None);
    let variant = adt_def.variant(variant_index);
    let statement = Statement {
        kind: StatementKind::Assign(Box::new((
            Place::return_place(),
            Rvalue::Aggregate(
                Box::new(kind),
                (0..variant.fields.len())
                    .map(|idx| Operand::Move(Place::from(Local::new(idx + 1))))
                    .collect(),
            ),
        ))),
        source_info,
    };

    let start_block = BasicBlockData {
        statements: vec![statement],
        terminator: Some(Terminator { source_info, kind: TerminatorKind::Return }),
        is_cleanup: false,
    };

    let source = MirSource::item(ctor_id);
    let mut body = new_body(
        source,
        IndexVec::from_elem_n(start_block, 1),
        local_decls,
        sig.inputs().len(),
        span,
    );
    // A constructor doesn't mention any other items (and we don't run the usual optimization passes
    // so this would otherwise not get filled).
    body.set_mentioned_items(Vec::new());

    crate::pass_manager::dump_mir_for_phase_change(tcx, &body);

    body
}

/// ```ignore (pseudo-impl)
/// impl FnPtr for fn(u32) {
///     fn addr(self) -> usize {
///         self as usize
///     }
/// }
/// ```
fn build_fn_ptr_addr_shim<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, self_ty: Ty<'tcx>) -> Body<'tcx> {
    assert_matches!(self_ty.kind(), ty::FnPtr(..), "expected fn ptr, found {self_ty}");
    let span = tcx.def_span(def_id);
    let Some(sig) = tcx.fn_sig(def_id).instantiate(tcx, &[self_ty.into()]).no_bound_vars() else {
        span_bug!(span, "FnPtr::addr with bound vars for `{self_ty}`");
    };
    let locals = local_decls_for_sig(&sig, span);

    let source_info = SourceInfo::outermost(span);
    // FIXME: use `expose_provenance` once we figure out whether function pointers have meaningful
    // provenance.
    let rvalue = Rvalue::Cast(
        CastKind::FnPtrToPtr,
        Operand::Move(Place::from(Local::new(1))),
        Ty::new_imm_ptr(tcx, tcx.types.unit),
    );
    let stmt = Statement {
        source_info,
        kind: StatementKind::Assign(Box::new((Place::return_place(), rvalue))),
    };
    let statements = vec![stmt];
    let start_block = BasicBlockData {
        statements,
        terminator: Some(Terminator { source_info, kind: TerminatorKind::Return }),
        is_cleanup: false,
    };
    let source = MirSource::from_instance(ty::InstanceKind::FnPtrAddrShim(def_id, self_ty));
    new_body(source, IndexVec::from_elem_n(start_block, 1), locals, sig.inputs().len(), span)
}

fn build_construct_coroutine_by_move_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    coroutine_closure_def_id: DefId,
    receiver_by_ref: bool,
) -> Body<'tcx> {
    let mut self_ty = tcx.type_of(coroutine_closure_def_id).instantiate_identity();
    let mut self_local: Place<'tcx> = Local::from_usize(1).into();
    let ty::CoroutineClosure(_, args) = *self_ty.kind() else {
        bug!();
    };

    // We use `&Self` here because we only need to emit an ABI-compatible shim body,
    // rather than match the signature exactly (which might take `&mut self` instead).
    //
    // We adjust the `self_local` to be a deref since we want to copy fields out of
    // a reference to the closure.
    if receiver_by_ref {
        self_local = tcx.mk_place_deref(self_local);
        self_ty = Ty::new_imm_ref(tcx, tcx.lifetimes.re_erased, self_ty);
    }

    let poly_sig = args.as_coroutine_closure().coroutine_closure_sig().map_bound(|sig| {
        tcx.mk_fn_sig(
            [self_ty].into_iter().chain(sig.tupled_inputs_ty.tuple_fields()),
            sig.to_coroutine_given_kind_and_upvars(
                tcx,
                args.as_coroutine_closure().parent_args(),
                tcx.coroutine_for_closure(coroutine_closure_def_id),
                ty::ClosureKind::FnOnce,
                tcx.lifetimes.re_erased,
                args.as_coroutine_closure().tupled_upvars_ty(),
                args.as_coroutine_closure().coroutine_captures_by_ref_ty(),
            ),
            sig.c_variadic,
            sig.safety,
            sig.abi,
        )
    });
    let sig = tcx.liberate_late_bound_regions(coroutine_closure_def_id, poly_sig);
    let ty::Coroutine(coroutine_def_id, coroutine_args) = *sig.output().kind() else {
        bug!();
    };

    let span = tcx.def_span(coroutine_closure_def_id);
    let locals = local_decls_for_sig(&sig, span);

    let mut fields = vec![];

    // Move all of the closure args.
    for idx in 1..sig.inputs().len() {
        fields.push(Operand::Move(Local::from_usize(idx + 1).into()));
    }

    for (idx, ty) in args.as_coroutine_closure().upvar_tys().iter().enumerate() {
        if receiver_by_ref {
            // The only situation where it's possible is when we capture immuatable references,
            // since those don't need to be reborrowed with the closure's env lifetime. Since
            // references are always `Copy`, just emit a copy.
            if !matches!(ty.kind(), ty::Ref(_, _, hir::Mutability::Not)) {
                // This copy is only sound if it's a `&T`. This may be
                // reachable e.g. when eagerly computing the `Fn` instance
                // of an async closure that doesn't borrowck.
                tcx.dcx().delayed_bug(format!(
                    "field should be captured by immutable ref if we have \
                    an `Fn` instance, but it was: {ty}"
                ));
            }
            fields.push(Operand::Copy(tcx.mk_place_field(
                self_local,
                FieldIdx::from_usize(idx),
                ty,
            )));
        } else {
            fields.push(Operand::Move(tcx.mk_place_field(
                self_local,
                FieldIdx::from_usize(idx),
                ty,
            )));
        }
    }

    let source_info = SourceInfo::outermost(span);
    let rvalue = Rvalue::Aggregate(
        Box::new(AggregateKind::Coroutine(coroutine_def_id, coroutine_args)),
        IndexVec::from_raw(fields),
    );
    let stmt = Statement {
        source_info,
        kind: StatementKind::Assign(Box::new((Place::return_place(), rvalue))),
    };
    let statements = vec![stmt];
    let start_block = BasicBlockData {
        statements,
        terminator: Some(Terminator { source_info, kind: TerminatorKind::Return }),
        is_cleanup: false,
    };

    let source = MirSource::from_instance(ty::InstanceKind::ConstructCoroutineInClosureShim {
        coroutine_closure_def_id,
        receiver_by_ref,
    });

    let body =
        new_body(source, IndexVec::from_elem_n(start_block, 1), locals, sig.inputs().len(), span);
    dump_mir(
        tcx,
        false,
        if receiver_by_ref { "coroutine_closure_by_ref" } else { "coroutine_closure_by_move" },
        &0,
        &body,
        |_, _| Ok(()),
    );

    body
}
