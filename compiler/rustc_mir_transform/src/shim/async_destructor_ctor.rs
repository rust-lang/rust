use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{CoroutineDesugaring, CoroutineKind, CoroutineSource, Safety};
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::{
    BasicBlock, BasicBlockData, Body, Local, LocalDecl, MirSource, Operand, Place, Rvalue,
    SourceInfo, Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::{self, EarlyBinder, Ty, TyCtxt};

use super::*;
use crate::patch::MirPatch;

// For coroutine, its async drop function layout is proxy with impl coroutine layout ref, so
// in async destructor ctor function we create
// `async_drop_in_place(*ImplCoroutine) { return ProxyCoroutine { &*ImplCoroutine } }`.
// In case of `async_drop_in_place<async_drop_in_place<ImplCoroutine>::{{closure}}>` (and further),
// we need to take impl coroutine ref from arg proxy layout and copy its to the new proxy layout
fn build_async_destructor_ctor_shim_for_coroutine<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    ty: Ty<'tcx>,
) -> Body<'tcx> {
    const INPUT_COUNT: usize = 1;
    let first_arg = Local::new(1);

    debug!("build_async_destructor_ctor_shim_for_coroutine(def_id={:?}, ty={:?})", def_id, ty);

    assert!(ty.is_coroutine());
    let args = tcx.mk_args(&[ty.into()]);

    let sig = tcx.fn_sig(def_id).instantiate(tcx, args);
    let sig = tcx.instantiate_bound_regions_with_erased(sig);
    let span = tcx.def_span(def_id);
    let source_info = SourceInfo::outermost(span);

    debug_assert_eq!(sig.inputs().len(), INPUT_COUNT);
    let locals = local_decls_for_sig(&sig, span);

    let cor_def_id = tcx.lang_items().async_drop_in_place_poll_fn().unwrap();

    let mut statements = Vec::new();
    let mut dropee_ptr = Place::from(first_arg);

    // if arg layout is a proxy layout, we need to take its ref field for final impl coroutine
    fn find_impl_coroutine<'tcx>(tcx: TyCtxt<'tcx>, mut cor_ty: Ty<'tcx>) -> Ty<'tcx> {
        let mut ty = cor_ty;
        loop {
            if let ty::Coroutine(def_id, args) = ty.kind() {
                cor_ty = ty;
                if tcx.is_templated_coroutine(*def_id) {
                    ty = args.first().unwrap().expect_ty();
                    continue;
                } else {
                    return cor_ty;
                }
            } else {
                return cor_ty;
            }
        }
    }
    let impl_cor_ty = find_impl_coroutine(tcx, ty);
    let impl_ptr_ty = Ty::new_mut_ptr(tcx, impl_cor_ty);
    if impl_cor_ty != ty {
        dropee_ptr = dropee_ptr.project_deeper(
            &[PlaceElem::Deref, PlaceElem::Field(FieldIdx::ZERO, impl_ptr_ty)],
            tcx,
        );
    }

    let resume_adt = tcx.adt_def(tcx.require_lang_item(LangItem::ResumeTy, None));
    let resume_ty = Ty::new_adt(tcx, resume_adt, ty::List::empty());

    let cor_args = ty::CoroutineArgs::new(
        tcx,
        ty::CoroutineArgsParts {
            parent_args: args,
            kind_ty: tcx.types.unit,
            resume_ty,
            yield_ty: tcx.types.unit,
            return_ty: tcx.types.unit,
            witness: Ty::new_coroutine_witness(tcx, cor_def_id, args),
            tupled_upvars_ty: Ty::new_tup(tcx, &[impl_ptr_ty]),
        },
    )
    .args;

    let mut blocks = IndexVec::with_capacity(1);
    statements.push(Statement {
        source_info,
        kind: StatementKind::Assign(Box::new((
            Place::return_place(),
            Rvalue::Aggregate(
                Box::new(AggregateKind::Coroutine(cor_def_id, cor_args)),
                [Operand::Move(dropee_ptr)].into(),
            ),
        ))),
    });
    blocks.push(BasicBlockData {
        statements,
        terminator: Some(Terminator { source_info, kind: TerminatorKind::Return }),
        is_cleanup: false,
    });

    let source = MirSource::from_instance(ty::InstanceKind::AsyncDropGlueCtorShim(def_id, ty));
    let mut body = new_body(source, blocks, locals, sig.inputs().len(), span);
    body.phase = MirPhase::Runtime(RuntimePhase::Initial);
    pm::run_passes(
        tcx,
        &mut body,
        &[
            &mentioned_items::MentionedItems,
            &simplify::SimplifyCfg::MakeShim,
            //&crate::reveal_all::RevealAll,
            &abort_unwinding_calls::AbortUnwindingCalls,
            &add_call_guards::CriticalCallEdges,
        ],
        None,
        pm::Optimizations::Allowed,
    );
    body
}

pub(super) fn build_async_destructor_ctor_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    ty: Ty<'tcx>,
) -> Body<'tcx> {
    debug!("build_async_destructor_ctor_shim(def_id={:?}, ty={:?})", def_id, ty);

    if ty.is_coroutine() {
        build_async_destructor_ctor_shim_for_coroutine(tcx, def_id, ty)
    } else {
        build_async_destructor_ctor_shim_not_coroutine(tcx, def_id, ty)
    }
}

fn build_async_destructor_ctor_shim_not_coroutine<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    ty: Ty<'tcx>,
) -> Body<'tcx> {
    debug_assert_eq!(Some(def_id), tcx.lang_items().async_drop_in_place_fn());
    let generic_body = tcx.optimized_mir(def_id);
    let args = tcx.mk_args(&[ty.into()]);
    let mut body = EarlyBinder::bind(generic_body.clone()).instantiate(tcx, args);

    pm::run_passes(
        tcx,
        &mut body,
        &[
            &simplify::SimplifyCfg::MakeShim,
            //&crate::reveal_all::RevealAll,
            &abort_unwinding_calls::AbortUnwindingCalls,
            &add_call_guards::CriticalCallEdges,
        ],
        None,
        pm::Optimizations::Allowed,
    );
    body
}

// build_drop_shim analog for async drop glue (for generated coroutine poll function)
pub(super) fn build_async_drop_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    ty: Ty<'tcx>,
) -> Body<'tcx> {
    debug!("build_async_drop_shim(def_id={:?}, ty={:?})", def_id, ty);
    let ty::Coroutine(_, parent_args) = ty.kind() else {
        bug!();
    };
    let typing_env = ty::TypingEnv::fully_monomorphized();
    //let param_env = tcx.param_env_reveal_all_normalized(def_id);

    let drop_ty = parent_args.first().unwrap().expect_ty();
    let drop_ptr_ty = Ty::new_mut_ptr(tcx, drop_ty);

    assert!(tcx.is_coroutine(def_id));
    let coroutine_kind = tcx.coroutine_kind(def_id).unwrap();

    assert!(matches!(
        coroutine_kind,
        CoroutineKind::Desugared(CoroutineDesugaring::Async, CoroutineSource::Fn)
    ));

    let needs_async_drop = drop_ty.needs_async_drop(tcx, typing_env);
    let needs_sync_drop = !needs_async_drop && drop_ty.needs_drop(tcx, typing_env);

    let resume_adt = tcx.adt_def(tcx.require_lang_item(LangItem::ResumeTy, None));
    let resume_ty = Ty::new_adt(tcx, resume_adt, ty::List::empty());

    let fn_sig = ty::Binder::dummy(tcx.mk_fn_sig(
        [ty, resume_ty],
        tcx.types.unit,
        false,
        Safety::Safe,
        ExternAbi::Rust,
    ));
    let sig = tcx.instantiate_bound_regions_with_erased(fn_sig);

    assert!(!drop_ty.is_coroutine());
    let span = tcx.def_span(def_id);
    let source_info = SourceInfo::outermost(span);

    // The first argument (index 0), but add 1 for the return value.
    let coroutine_layout = Place::from(Local::new(1 + 0));
    let coroutine_layout_dropee =
        tcx.mk_place_field(coroutine_layout, FieldIdx::new(0), drop_ptr_ty);

    let return_block = BasicBlock::new(1);
    let mut blocks = IndexVec::with_capacity(2);
    let block = |blocks: &mut IndexVec<_, _>, kind| {
        blocks.push(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator { source_info, kind }),
            is_cleanup: false,
        })
    };
    block(
        &mut blocks,
        if needs_sync_drop {
            TerminatorKind::Drop {
                place: tcx.mk_place_deref(coroutine_layout_dropee),
                target: return_block,
                unwind: UnwindAction::Continue,
                replace: false,
                drop: None,
                async_fut: None,
            }
        } else {
            TerminatorKind::Goto { target: return_block }
        },
    );
    block(&mut blocks, TerminatorKind::Return);

    let source = MirSource::from_instance(ty::InstanceKind::AsyncDropGlue(def_id, ty));
    let mut body =
        new_body(source, blocks, local_decls_for_sig(&sig, span), sig.inputs().len(), span);

    body.coroutine = Some(Box::new(CoroutineInfo::initial(
        coroutine_kind,
        parent_args.as_coroutine().yield_ty(),
        parent_args.as_coroutine().resume_ty(),
    )));
    body.phase = MirPhase::Runtime(RuntimePhase::Initial);
    if !needs_async_drop {
        // Returning noop body for types without `need async drop`
        // (or sync Drop in case of !`need async drop` && `need drop`)
        return body;
    }

    let mut dropee_ptr = Place::from(body.local_decls.push(LocalDecl::new(drop_ptr_ty, span)));
    let st_kind = StatementKind::Assign(Box::new((
        dropee_ptr,
        Rvalue::Use(Operand::Move(coroutine_layout_dropee)),
    )));
    body.basic_blocks_mut()[START_BLOCK].statements.push(Statement { source_info, kind: st_kind });

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
        let ref_ty = reborrow.ty(&body.local_decls, tcx);
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

    let dropline = body.basic_blocks.last_index();

    let patch = {
        let mut elaborator = DropShimElaborator {
            body: &body,
            patch: MirPatch::new(&body),
            tcx,
            typing_env,
            produce_async_drops: true,
        };
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
            dropline,
        );
        elaborator.patch
    };
    patch.apply(&mut body);

    body
}

pub(super) fn build_future_drop_poll_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    proxy_ty: Ty<'tcx>,
    impl_ty: Ty<'tcx>,
) -> Body<'tcx> {
    let instance = ty::InstanceKind::FutureDropPollShim(def_id, proxy_ty, impl_ty);
    let ty::Coroutine(coroutine_def_id, impl_args) = impl_ty.kind() else {
        bug!("FutureDropPollShim not for coroutine impl type: ({:?})", instance);
    };

    let span = tcx.def_span(def_id);
    let source_info = SourceInfo::outermost(span);

    let pin_proxy_layout_local = Local::new(1);
    let cor_ref = Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, impl_ty);
    let proxy_ref = Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, proxy_ty);
    // taking _1.0.0 (impl from Pin, impl from proxy)
    let proxy_ref_place = Place::from(pin_proxy_layout_local)
        .project_deeper(&[PlaceElem::Field(FieldIdx::ZERO, proxy_ref)], tcx);
    let impl_ref_place = |proxy_ref_local: Local| {
        Place::from(proxy_ref_local).project_deeper(
            &[
                PlaceElem::Deref,
                PlaceElem::Downcast(None, VariantIdx::ZERO),
                PlaceElem::Field(FieldIdx::ZERO, cor_ref),
            ],
            tcx,
        )
    };

    if tcx.is_templated_coroutine(*coroutine_def_id) {
        // ret_ty = `Poll<()>`
        let poll_adt_ref = tcx.adt_def(tcx.require_lang_item(LangItem::Poll, None));
        let ret_ty = Ty::new_adt(tcx, poll_adt_ref, tcx.mk_args(&[tcx.types.unit.into()]));
        // env_ty = `Pin<&mut proxy_ty>`
        let pin_adt_ref = tcx.adt_def(tcx.require_lang_item(LangItem::Pin, None));
        let env_ty = Ty::new_adt(tcx, pin_adt_ref, tcx.mk_args(&[proxy_ref.into()]));
        // sig = `fn (Pin<&mut proxy_ty>, &mut Context) -> Poll<()>`
        let sig = tcx.mk_fn_sig(
            [env_ty, Ty::new_task_context(tcx)],
            ret_ty,
            false,
            hir::Safety::Safe,
            ExternAbi::Rust,
        );
        let mut locals = local_decls_for_sig(&sig, span);
        let mut blocks = IndexVec::with_capacity(3);

        let proxy_ref_local = locals.push(LocalDecl::new(proxy_ref, span));
        let cor_ref_local = locals.push(LocalDecl::new(cor_ref, span));
        let cor_ref_place = Place::from(cor_ref_local);

        let call_bb = BasicBlock::new(1);
        let return_bb = BasicBlock::new(2);

        let assign1 = Statement {
            source_info,
            kind: StatementKind::Assign(Box::new((
                Place::from(proxy_ref_local),
                Rvalue::CopyForDeref(proxy_ref_place),
            ))),
        };
        let assign2 = Statement {
            source_info,
            kind: StatementKind::Assign(Box::new((
                cor_ref_place,
                Rvalue::CopyForDeref(impl_ref_place(proxy_ref_local)),
            ))),
        };

        // cor_pin_ty = `Pin<&mut cor_ref>`
        let cor_pin_ty = Ty::new_adt(tcx, pin_adt_ref, tcx.mk_args(&[cor_ref.into()]));
        let cor_pin_place = Place::from(locals.push(LocalDecl::new(cor_pin_ty, span)));

        let pin_fn = tcx.require_lang_item(LangItem::PinNewUnchecked, Some(span));
        // call Pin<FutTy>::new_unchecked(&mut impl_cor)
        blocks.push(BasicBlockData {
            statements: vec![assign1, assign2],
            terminator: Some(Terminator {
                source_info,
                kind: TerminatorKind::Call {
                    func: Operand::function_handle(tcx, pin_fn, [cor_ref.into()], span),
                    args: [dummy_spanned(Operand::Move(cor_ref_place))].into(),
                    destination: cor_pin_place,
                    target: Some(call_bb),
                    unwind: UnwindAction::Continue,
                    call_source: CallSource::Misc,
                    fn_span: span,
                },
            }),
            is_cleanup: false,
        });
        // When dropping async drop coroutine, we continue its execution:
        // we call impl::poll (impl_layout, ctx)
        let poll_fn = tcx.require_lang_item(LangItem::FuturePoll, None);
        let resume_ctx = Place::from(Local::new(2));
        blocks.push(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator {
                source_info,
                kind: TerminatorKind::Call {
                    func: Operand::function_handle(tcx, poll_fn, [impl_ty.into()], span),
                    args: [
                        dummy_spanned(Operand::Move(cor_pin_place)),
                        dummy_spanned(Operand::Move(resume_ctx)),
                    ]
                    .into(),
                    destination: Place::return_place(),
                    target: Some(return_bb),
                    unwind: UnwindAction::Continue,
                    call_source: CallSource::Misc,
                    fn_span: span,
                },
            }),
            is_cleanup: false,
        });
        blocks.push(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator { source_info, kind: TerminatorKind::Return }),
            is_cleanup: false,
        });

        let source = MirSource::from_instance(instance);
        let mut body = new_body(source, blocks, locals, sig.inputs().len(), span);
        body.phase = MirPhase::Runtime(RuntimePhase::Initial);
        return body;
    }
    // future drop poll for async drop must be resolved to standart poll (AsyncDropGlue)
    assert!(!tcx.is_templated_coroutine(*coroutine_def_id));

    // converting `(_1: Pin<&mut CorLayout>, _2: &mut Context<'_>) -> Poll<()>`
    // into `(_1: Pin<&mut ProxyLayout>, _2: &mut Context<'_>) -> Poll<()>`
    // let mut _x: &mut CorLayout = &*_1.0.0;
    // Replace old _1.0 accesses into _x accesses;
    let body = tcx.optimized_mir(*coroutine_def_id).future_drop_poll().unwrap();
    let mut body: Body<'tcx> = EarlyBinder::bind(body.clone()).instantiate(tcx, impl_args);
    body.source.instance = instance;
    body.phase = MirPhase::Runtime(RuntimePhase::Initial);
    body.var_debug_info.clear();
    let pin_adt_ref = tcx.adt_def(tcx.require_lang_item(LangItem::Pin, Some(span)));
    let args = tcx.mk_args(&[proxy_ref.into()]);
    let pin_proxy_ref = Ty::new_adt(tcx, pin_adt_ref, args);

    let proxy_ref_local = body.local_decls.push(LocalDecl::new(proxy_ref, span));
    let cor_ref_local = body.local_decls.push(LocalDecl::new(cor_ref, span));
    FixProxyFutureDropVisitor { tcx, replace_to: cor_ref_local }.visit_body(&mut body);
    // Now changing first arg from Pin<&mut ImplCoroutine> to Pin<&mut ProxyCoroutine>
    body.local_decls[pin_proxy_layout_local] = LocalDecl::new(pin_proxy_ref, span);

    {
        let bb: &mut BasicBlockData<'tcx> = &mut body.basic_blocks_mut()[START_BLOCK];
        // _tmp = _1.0 : Pin<&ProxyLayout> ==> &ProxyLayout
        bb.statements.insert(
            0,
            Statement {
                source_info,
                kind: StatementKind::Assign(Box::new((
                    Place::from(proxy_ref_local),
                    Rvalue::CopyForDeref(proxy_ref_place),
                ))),
            },
        );
        bb.statements.insert(
            1,
            Statement {
                source_info,
                kind: StatementKind::Assign(Box::new((
                    Place::from(cor_ref_local),
                    Rvalue::CopyForDeref(impl_ref_place(proxy_ref_local)),
                ))),
            },
        );
    }
    body
}
