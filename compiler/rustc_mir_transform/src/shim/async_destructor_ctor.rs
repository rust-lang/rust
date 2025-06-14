use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{CoroutineDesugaring, CoroutineKind, CoroutineSource, Safety};
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::{
    BasicBlock, BasicBlockData, Body, Local, LocalDecl, MirSource, Operand, Place, Rvalue,
    SourceInfo, Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::{self, EarlyBinder, Ty, TyCtxt, TypeVisitableExt};

use super::*;
use crate::patch::MirPatch;

pub(super) fn build_async_destructor_ctor_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    ty: Ty<'tcx>,
) -> Body<'tcx> {
    debug!("build_async_destructor_ctor_shim(def_id={:?}, ty={:?})", def_id, ty);
    debug_assert_eq!(Some(def_id), tcx.lang_items().async_drop_in_place_fn());
    let generic_body = tcx.optimized_mir(def_id);
    let args = tcx.mk_args(&[ty.into()]);
    let mut body = EarlyBinder::bind(generic_body.clone()).instantiate(tcx, args);

    // Minimal shim passes except MentionedItems,
    // it causes error "mentioned_items for DefId(...async_drop_in_place...) have already been set
    pm::run_passes(
        tcx,
        &mut body,
        &[
            &simplify::SimplifyCfg::MakeShim,
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

    let resume_adt = tcx.adt_def(tcx.require_lang_item(LangItem::ResumeTy, DUMMY_SP));
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
    if !needs_async_drop || drop_ty.references_error() {
        // Returning noop body for types without `need async drop`
        // (or sync Drop in case of !`need async drop` && `need drop`).
        // And also for error types.
        return body;
    }

    let mut dropee_ptr = Place::from(body.local_decls.push(LocalDecl::new(drop_ptr_ty, span)));
    let st_kind = StatementKind::Assign(Box::new((
        dropee_ptr,
        Rvalue::Use(Operand::Move(coroutine_layout_dropee)),
    )));
    body.basic_blocks_mut()[START_BLOCK].statements.push(Statement { source_info, kind: st_kind });
    dropee_ptr = dropee_emit_retag(tcx, &mut body, dropee_ptr, span);

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

// * For async drop a "normal" coroutine:
// `async_drop_in_place<T>::{closure}.poll()` is converted into `T.future_drop_poll()`.
// Every coroutine has its `poll` (calculate yourself a little further)
// and its `future_drop_poll` (drop yourself a little further).
//
// * For async drop of "async drop coroutine" (`async_drop_in_place<T>::{closure}`):
// Correct drop of such coroutine means normal execution of nested async drop.
// async_drop(async_drop(T))::future_drop_poll() => async_drop(T)::poll().
pub(super) fn build_future_drop_poll_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    proxy_ty: Ty<'tcx>,
    impl_ty: Ty<'tcx>,
) -> Body<'tcx> {
    let instance = ty::InstanceKind::FutureDropPollShim(def_id, proxy_ty, impl_ty);
    let ty::Coroutine(coroutine_def_id, _) = impl_ty.kind() else {
        bug!("build_future_drop_poll_shim not for coroutine impl type: ({:?})", instance);
    };

    let span = tcx.def_span(def_id);

    if tcx.is_async_drop_in_place_coroutine(*coroutine_def_id) {
        build_adrop_for_adrop_shim(tcx, proxy_ty, impl_ty, span, instance)
    } else {
        build_adrop_for_coroutine_shim(tcx, proxy_ty, impl_ty, span, instance)
    }
}

// For async drop a "normal" coroutine:
// `async_drop_in_place<T>::{closure}.poll()` is converted into `T.future_drop_poll()`.
// Every coroutine has its `poll` (calculate yourself a little further)
// and its `future_drop_poll` (drop yourself a little further).
fn build_adrop_for_coroutine_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    proxy_ty: Ty<'tcx>,
    impl_ty: Ty<'tcx>,
    span: Span,
    instance: ty::InstanceKind<'tcx>,
) -> Body<'tcx> {
    let ty::Coroutine(coroutine_def_id, impl_args) = impl_ty.kind() else {
        bug!("build_adrop_for_coroutine_shim not for coroutine impl type: ({:?})", instance);
    };
    let proxy_ref = Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, proxy_ty);
    // taking _1.0 (impl from Pin)
    let pin_proxy_layout_local = Local::new(1);
    let source_info = SourceInfo::outermost(span);
    // converting `(_1: Pin<&mut CorLayout>, _2: &mut Context<'_>) -> Poll<()>`
    // into `(_1: Pin<&mut ProxyLayout>, _2: &mut Context<'_>) -> Poll<()>`
    // let mut _x: &mut CorLayout = &*_1.0.0;
    // Replace old _1.0 accesses into _x accesses;
    let body = tcx.optimized_mir(*coroutine_def_id).future_drop_poll().unwrap();
    let mut body: Body<'tcx> = EarlyBinder::bind(body.clone()).instantiate(tcx, impl_args);
    body.source.instance = instance;
    body.phase = MirPhase::Runtime(RuntimePhase::Initial);
    body.var_debug_info.clear();
    let pin_adt_ref = tcx.adt_def(tcx.require_lang_item(LangItem::Pin, span));
    let args = tcx.mk_args(&[proxy_ref.into()]);
    let pin_proxy_ref = Ty::new_adt(tcx, pin_adt_ref, args);

    let cor_ref = Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, impl_ty);

    let proxy_ref_local = body.local_decls.push(LocalDecl::new(proxy_ref, span));
    let cor_ref_local = body.local_decls.push(LocalDecl::new(cor_ref, span));

    FixProxyFutureDropVisitor { tcx, replace_to: cor_ref_local }.visit_body(&mut body);
    // Now changing first arg from Pin<&mut ImplCoroutine> to Pin<&mut ProxyCoroutine>
    body.local_decls[pin_proxy_layout_local] = LocalDecl::new(pin_proxy_ref, span);

    {
        let mut idx: usize = 0;
        // _proxy = _1.0 : Pin<&ProxyLayout> ==> &ProxyLayout
        let proxy_ref_place = Place::from(pin_proxy_layout_local)
            .project_deeper(&[PlaceElem::Field(FieldIdx::ZERO, proxy_ref)], tcx);
        body.basic_blocks_mut()[START_BLOCK].statements.insert(
            idx,
            Statement {
                source_info,
                kind: StatementKind::Assign(Box::new((
                    Place::from(proxy_ref_local),
                    Rvalue::CopyForDeref(proxy_ref_place),
                ))),
            },
        );
        idx += 1;
        let mut cor_ptr_local = proxy_ref_local;
        proxy_ty.find_async_drop_impl_coroutine(tcx, |ty| {
            if ty != proxy_ty {
                let ty_ptr = Ty::new_mut_ptr(tcx, ty);
                let impl_ptr_place = Place::from(cor_ptr_local).project_deeper(
                    &[PlaceElem::Deref, PlaceElem::Field(FieldIdx::ZERO, ty_ptr)],
                    tcx,
                );
                cor_ptr_local = body.local_decls.push(LocalDecl::new(ty_ptr, span));
                // _cor_ptr = _proxy.0.0 (... .0)
                body.basic_blocks_mut()[START_BLOCK].statements.insert(
                    idx,
                    Statement {
                        source_info,
                        kind: StatementKind::Assign(Box::new((
                            Place::from(cor_ptr_local),
                            Rvalue::CopyForDeref(impl_ptr_place),
                        ))),
                    },
                );
                idx += 1;
            }
        });

        // _cor_ref = &*cor_ptr
        let reborrow = Rvalue::Ref(
            tcx.lifetimes.re_erased,
            BorrowKind::Mut { kind: MutBorrowKind::Default },
            tcx.mk_place_deref(Place::from(cor_ptr_local)),
        );
        body.basic_blocks_mut()[START_BLOCK].statements.insert(
            idx,
            Statement {
                source_info,
                kind: StatementKind::Assign(Box::new((Place::from(cor_ref_local), reborrow))),
            },
        );
    }
    body
}

// When dropping async drop coroutine, we continue its execution.
// async_drop(async_drop(T))::future_drop_poll() => async_drop(T)::poll()
fn build_adrop_for_adrop_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    proxy_ty: Ty<'tcx>,
    impl_ty: Ty<'tcx>,
    span: Span,
    instance: ty::InstanceKind<'tcx>,
) -> Body<'tcx> {
    let source_info = SourceInfo::outermost(span);
    let proxy_ref = Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, proxy_ty);
    // taking _1.0 (impl from Pin)
    let pin_proxy_layout_local = Local::new(1);
    let proxy_ref_place = Place::from(pin_proxy_layout_local)
        .project_deeper(&[PlaceElem::Field(FieldIdx::ZERO, proxy_ref)], tcx);
    let cor_ref = Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, impl_ty);

    // ret_ty = `Poll<()>`
    let poll_adt_ref = tcx.adt_def(tcx.require_lang_item(LangItem::Poll, span));
    let ret_ty = Ty::new_adt(tcx, poll_adt_ref, tcx.mk_args(&[tcx.types.unit.into()]));
    // env_ty = `Pin<&mut proxy_ty>`
    let pin_adt_ref = tcx.adt_def(tcx.require_lang_item(LangItem::Pin, span));
    let env_ty = Ty::new_adt(tcx, pin_adt_ref, tcx.mk_args(&[proxy_ref.into()]));
    // sig = `fn (Pin<&mut proxy_ty>, &mut Context) -> Poll<()>`
    let sig = tcx.mk_fn_sig(
        [env_ty, Ty::new_task_context(tcx)],
        ret_ty,
        false,
        hir::Safety::Safe,
        ExternAbi::Rust,
    );
    // This function will be called with pinned proxy coroutine layout.
    // We need to extract `Arg0.0` to get proxy layout, and then get `.0`
    // further to receive impl coroutine (may be needed)
    let mut locals = local_decls_for_sig(&sig, span);
    let mut blocks = IndexVec::with_capacity(3);

    let proxy_ref_local = locals.push(LocalDecl::new(proxy_ref, span));

    let call_bb = BasicBlock::new(1);
    let return_bb = BasicBlock::new(2);

    let mut statements = Vec::new();

    statements.push(Statement {
        source_info,
        kind: StatementKind::Assign(Box::new((
            Place::from(proxy_ref_local),
            Rvalue::CopyForDeref(proxy_ref_place),
        ))),
    });

    let mut cor_ptr_local = proxy_ref_local;
    proxy_ty.find_async_drop_impl_coroutine(tcx, |ty| {
        if ty != proxy_ty {
            let ty_ptr = Ty::new_mut_ptr(tcx, ty);
            let impl_ptr_place = Place::from(cor_ptr_local)
                .project_deeper(&[PlaceElem::Deref, PlaceElem::Field(FieldIdx::ZERO, ty_ptr)], tcx);
            cor_ptr_local = locals.push(LocalDecl::new(ty_ptr, span));
            // _cor_ptr = _proxy.0.0 (... .0)
            statements.push(Statement {
                source_info,
                kind: StatementKind::Assign(Box::new((
                    Place::from(cor_ptr_local),
                    Rvalue::CopyForDeref(impl_ptr_place),
                ))),
            });
        }
    });

    // convert impl coroutine ptr into ref
    let reborrow = Rvalue::Ref(
        tcx.lifetimes.re_erased,
        BorrowKind::Mut { kind: MutBorrowKind::Default },
        tcx.mk_place_deref(Place::from(cor_ptr_local)),
    );
    let cor_ref_place = Place::from(locals.push(LocalDecl::new(cor_ref, span)));
    statements.push(Statement {
        source_info,
        kind: StatementKind::Assign(Box::new((cor_ref_place, reborrow))),
    });

    // cor_pin_ty = `Pin<&mut cor_ref>`
    let cor_pin_ty = Ty::new_adt(tcx, pin_adt_ref, tcx.mk_args(&[cor_ref.into()]));
    let cor_pin_place = Place::from(locals.push(LocalDecl::new(cor_pin_ty, span)));

    let pin_fn = tcx.require_lang_item(LangItem::PinNewUnchecked, span);
    // call Pin<FutTy>::new_unchecked(&mut impl_cor)
    blocks.push(BasicBlockData {
        statements,
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
    let poll_fn = tcx.require_lang_item(LangItem::FuturePoll, span);
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
