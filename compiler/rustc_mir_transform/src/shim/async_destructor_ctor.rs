use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{CoroutineDesugaring, CoroutineKind, CoroutineSource, Safety};
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::{
    BasicBlock, BasicBlockData, Body, Local, LocalDecl, MirSource, Operand, Place, Rvalue,
    SourceInfo, Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::{self, EarlyBinder, Ty, TyCtxt};
use rustc_mir_dataflow::elaborate_drops;

use super::*;

// For coroutine, its async drop function layout is proxy with impl coroutine layout ref, so
// in async destructor ctor function we create
// `async_drop_in_place(*ImplCoroutine) { return ProxyCoroutine { &*ImplCoroutine } }`.
// In case of `async_drop_in_place<async_drop_in_place<ImplCoroutine>::{{closure}}>` (and further),
// we need to take impl coroutine ref from arg proxy layout and copy its to the new proxy layout
pub fn build_async_destructor_ctor_shim_for_coroutine<'tcx>(
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
    let mut locals = local_decls_for_sig(&sig, span);

    let cor_def_id = tcx.lang_items().async_drop_in_place_poll_fn().unwrap();

    let mut statements = Vec::new();
    let mut dropee_ptr = Place::from(first_arg);
    if tcx.sess.opts.unstable_opts.mir_emit_retag {
        let reborrow = Rvalue::Ref(
            tcx.lifetimes.re_erased,
            BorrowKind::Mut { kind: MutBorrowKind::Default },
            tcx.mk_place_deref(dropee_ptr),
        );
        let ref_ty = Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, ty);
        dropee_ptr = locals.push(LocalDecl::new(ref_ty, span)).into();
        statements.push(Statement {
            source_info,
            kind: StatementKind::Assign(Box::new((dropee_ptr, reborrow))),
        });
        statements.push(Statement {
            source_info,
            kind: StatementKind::Retag(RetagKind::FnEntry, Box::new(dropee_ptr)),
        });
    }

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
    if impl_cor_ty != ty {
        dropee_ptr = dropee_ptr
            .project_deeper(&[PlaceElem::Deref, PlaceElem::Field(FieldIdx::ZERO, ty)], tcx);
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
            tupled_upvars_ty: Ty::new_tup(tcx, &[Ty::new_mut_ptr(tcx, ty)]),
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
    pm::run_passes(
        tcx,
        &mut body,
        &[
            &mentioned_items::MentionedItems,
            &simplify::SimplifyCfg::MakeShim,
            &crate::reveal_all::RevealAll,
            &abort_unwinding_calls::AbortUnwindingCalls,
            &add_call_guards::CriticalCallEdges,
        ],
        None,
    );
    body
}

pub fn build_async_destructor_ctor_shim<'tcx>(
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

pub fn build_async_destructor_ctor_shim_not_coroutine<'tcx>(
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
            &crate::reveal_all::RevealAll,
            &abort_unwinding_calls::AbortUnwindingCalls,
            &add_call_guards::CriticalCallEdges,
        ],
        None,
    );
    body
}

// build_drop_shim analog for async drop glue (for generated coroutine poll function)
pub fn build_async_drop_shim<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, ty: Ty<'tcx>) -> Body<'tcx> {
    debug!("build_async_drop_shim(def_id={:?}, ty={:?})", def_id, ty);
    let ty::Coroutine(_, parent_args) = ty.kind() else {
        bug!();
    };
    let param_env = tcx.param_env_reveal_all_normalized(def_id);

    let drop_ty = parent_args.first().unwrap().expect_ty();
    let drop_ptr_ty = Ty::new_mut_ptr(tcx, drop_ty);

    assert!(tcx.is_coroutine(def_id));
    let coroutine_kind = tcx.coroutine_kind(def_id).unwrap();

    assert!(matches!(
        coroutine_kind,
        CoroutineKind::Desugared(CoroutineDesugaring::Async, CoroutineSource::Fn)
    ));

    let needs_async_drop = drop_ty.needs_async_drop(tcx, param_env);
    let needs_sync_drop = !needs_async_drop && drop_ty.needs_drop(tcx, param_env);

    let resume_adt = tcx.adt_def(tcx.require_lang_item(LangItem::ResumeTy, None));
    let resume_ty = Ty::new_adt(tcx, resume_adt, ty::List::empty());

    let fn_sig = ty::Binder::dummy(tcx.mk_fn_sig(
        [ty, resume_ty],
        tcx.types.unit,
        false,
        Safety::Safe,
        rustc_target::spec::abi::Abi::Rust,
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
            param_env,
            produce_async_drops: true,
        };
        let dropee = tcx.mk_place_deref(dropee_ptr);
        let resume_block = elaborator.patch.resume_block();
        elaborate_drops::elaborate_drop(
            &mut elaborator,
            source_info,
            dropee,
            (),
            return_block,
            elaborate_drops::Unwind::To(resume_block),
            START_BLOCK,
            dropline,
        );
        elaborator.patch
    };
    patch.apply(&mut body);

    body
}
