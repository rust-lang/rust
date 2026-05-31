//! Drops and async drops related logic for coroutine transformation pass

use super::*;

// Fix return Poll<Rv>::Pending statement into Poll<()>::Pending for async drop function
struct FixReturnPendingVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for FixReturnPendingVisitor<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_assign(
        &mut self,
        place: &mut Place<'tcx>,
        rvalue: &mut Rvalue<'tcx>,
        _location: Location,
    ) {
        if place.local != RETURN_PLACE {
            return;
        }

        // Converting `_0 = Poll::<Rv>::Pending` to `_0 = Poll::<()>::Pending`
        if let Rvalue::Aggregate(kind, _) = rvalue
            && let AggregateKind::Adt(_, _, ref mut args, _, _) = **kind
        {
            *args = self.tcx.mk_args(&[self.tcx.types.unit.into()]);
        }
    }
}

/// Drop elaboration has transformed all async drops into `yield` loops.
/// The resulting coroutine needs `async drop` if it yields on a path
/// reachable through 'drop' targets of a Yield terminator.
#[tracing::instrument(level = "trace", skip(body), ret)]
pub(super) fn has_async_drops<'tcx>(body: &mut Body<'tcx>) -> bool {
    let mut has_async_drops = false;

    let mut dropline: DenseBitSet<BasicBlock> = DenseBitSet::new_empty(body.basic_blocks.len());
    for (bb, data) in traversal::reverse_postorder(body) {
        // Cleanup edges are not async drops.
        if data.is_cleanup {
            continue;
        }

        if let TerminatorKind::Yield { drop, .. } = data.terminator().kind {
            if dropline.contains(bb) {
                has_async_drops = true
            }
            if let Some(v) = drop {
                dropline.insert(v);
            }
        }

        if dropline.contains(bb) {
            data.terminator().successors().for_each(|v| {
                dropline.insert(v);
            });
        }
    }

    has_async_drops
}

#[tracing::instrument(level = "trace", skip(tcx, body))]
pub(super) fn elaborate_coroutine_drops<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    use crate::elaborate_drop::{Unwind, elaborate_drop};
    use crate::patch::MirPatch;
    use crate::shim::DropShimElaborator;

    // Note that `elaborate_drops` only drops the upvars of a coroutine, and
    // this is ok because `open_drop` can only be reached within that own
    // coroutine's resume function.
    let typing_env = body.typing_env(tcx);

    let mut elaborator = DropShimElaborator {
        body,
        patch: MirPatch::new(body),
        tcx,
        typing_env,
        // FIXME(async_drop): Drops, produced by insert_clean_drop + elaborate_coroutine_drops, are
        // currently sync only. To allow async for them, flip this flag and fix the related
        // problems.
        produce_async_drops: false,
    };

    for (block, block_data) in body.basic_blocks.iter_enumerated() {
        let (target, unwind, source_info, dropline) = match block_data.terminator() {
            Terminator {
                source_info,
                kind: TerminatorKind::Drop { place, target, unwind, replace: _, drop },
            } => {
                if let Some(local) = place.as_local()
                    && local == SELF_ARG
                {
                    (target, unwind, source_info, *drop)
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
                UnwindAction::Terminate(reason) => elaborator.patch.terminate_block(reason),
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
            dropline,
        );
    }
    elaborator.patch.apply(body);
}

#[tracing::instrument(level = "trace", skip(tcx, body), ret)]
pub(super) fn insert_clean_drop<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    has_async_drops: bool,
) -> BasicBlock {
    let return_block = if has_async_drops {
        insert_poll_ready_block(tcx, body)
    } else {
        insert_term_block(body, TerminatorKind::Return)
    };

    // FIXME: When move insert_clean_drop + elaborate_coroutine_drops before async drops expand,
    // also set dropline here:
    // let dropline = if has_async_drops { Some(return_block) } else { None };
    let dropline = None;

    let term = TerminatorKind::Drop {
        place: Place::from(SELF_ARG),
        target: return_block,
        unwind: UnwindAction::Continue,
        replace: false,
        drop: dropline,
    };

    // Create a block to destroy an unresumed coroutines. This can only destroy upvars.
    insert_term_block(body, term)
}

#[tracing::instrument(level = "trace", skip(tcx, transform, body))]
pub(super) fn create_coroutine_drop_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    transform: &TransformVisitor<'tcx>,
    coroutine_ty: Ty<'tcx>,
    body: &Body<'tcx>,
    drop_clean: BasicBlock,
) -> Body<'tcx> {
    let mut body = body.clone();
    // Take the coroutine info out of the body, since the drop shim is
    // not a coroutine body itself; it just has its drop built out of it.
    let _ = body.coroutine.take();
    // Make sure the resume argument is not included here, since we're
    // building a body for `drop_glue`.
    body.arg_count = 1;

    let source_info = SourceInfo::outermost(body.span);

    let mut cases = create_cases(&mut body, transform, Operation::Drop);

    cases.insert(0, (CoroutineArgs::UNRESUMED, drop_clean));

    // The returned state and the poisoned state fall through to the default
    // case which is just to return

    let default_block = insert_term_block(&mut body, TerminatorKind::Return);
    insert_switch(&mut body, cases, transform, default_block);

    for block in body.basic_blocks_mut() {
        let kind = &mut block.terminator_mut().kind;
        if let TerminatorKind::CoroutineDrop = *kind {
            *kind = TerminatorKind::Return;
        }
    }

    // Replace the return variable
    body.local_decls[RETURN_PLACE] = LocalDecl::with_source_info(tcx.types.unit, source_info);

    make_coroutine_state_argument_indirect(tcx, &mut body);

    // Make sure we remove dead blocks to remove
    // unrelated code from the resume part of the function
    simplify::remove_dead_blocks(&mut body);

    // Run derefer to fix Derefs that are not in the first place
    deref_finder(tcx, &mut body, false);

    // Update the body's def to become the drop glue.
    let coroutine_instance = body.source.instance;
    let drop_glue = tcx.require_lang_item(LangItem::DropGlue, body.span);
    let drop_instance = InstanceKind::DropGlue(drop_glue, Some(coroutine_ty));

    // Temporary change MirSource to coroutine's instance so that dump_mir produces more sensible
    // filename.
    body.source.instance = coroutine_instance;
    if let Some(dumper) = MirDumper::new(tcx, "coroutine_drop", &body) {
        dumper.dump_mir(&body);
    }
    body.source.instance = drop_instance;

    // Creating a coroutine drop shim happens on `Analysis(PostCleanup) -> Runtime(Initial)`
    // but the pass manager doesn't update the phase of the coroutine drop shim. Update the
    // phase of the drop shim so that later on when we run the pass manager on the shim, in
    // the `mir_shims` query, we don't ICE on the intra-pass validation before we've updated
    // the phase of the body from analysis.
    body.phase = MirPhase::Runtime(RuntimePhase::Initial);

    body
}

// Create async drop shim function to drop coroutine itself
#[tracing::instrument(level = "trace", skip(tcx, transform, body))]
pub(super) fn create_coroutine_drop_shim_async<'tcx>(
    tcx: TyCtxt<'tcx>,
    transform: &TransformVisitor<'tcx>,
    body: &Body<'tcx>,
    drop_clean: BasicBlock,
    can_unwind: bool,
) -> Body<'tcx> {
    let mut body = body.clone();
    // Take the coroutine info out of the body, since the drop shim is
    // not a coroutine body itself; it just has its drop built out of it.
    let _ = body.coroutine.take();

    FixReturnPendingVisitor { tcx }.visit_body(&mut body);

    // Poison the coroutine when it unwinds
    if can_unwind {
        generate_poison_block_and_redirect_unwinds_there(transform, &mut body);
    }

    let source_info = SourceInfo::outermost(body.span);

    let mut cases = create_cases(&mut body, transform, Operation::AsyncDrop);

    cases.insert(0, (CoroutineArgs::UNRESUMED, drop_clean));

    use rustc_middle::mir::AssertKind::ResumedAfterPanic;
    // Panic when resumed on the returned or poisoned state
    if can_unwind {
        cases.insert(
            1,
            (
                CoroutineArgs::POISONED,
                insert_panic_block(tcx, &mut body, ResumedAfterPanic(transform.coroutine_kind)),
            ),
        );
    }

    // RETURNED state also goes to default_block with `return Ready<()>`.
    // For fully-polled coroutine, async drop has nothing to do.
    let default_block = insert_poll_ready_block(tcx, &mut body);
    insert_switch(&mut body, cases, transform, default_block);

    for block in body.basic_blocks_mut() {
        let kind = &mut block.terminator_mut().kind;
        if let TerminatorKind::CoroutineDrop = *kind {
            *kind = TerminatorKind::Return;
            block.statements.push(return_poll_ready_assign(tcx, source_info));
        }
    }

    // Replace the return variable: Poll<RetT> to Poll<()>
    let poll_adt_ref = tcx.adt_def(tcx.require_lang_item(LangItem::Poll, body.span));
    let poll_enum = Ty::new_adt(tcx, poll_adt_ref, tcx.mk_args(&[tcx.types.unit.into()]));
    body.local_decls[RETURN_PLACE] = LocalDecl::with_source_info(poll_enum, source_info);

    match transform.coroutine_kind {
        // Iterator::next doesn't accept a pinned argument,
        // unlike for all other coroutine kinds.
        CoroutineKind::Desugared(CoroutineDesugaring::Gen, _) => {
            make_coroutine_state_argument_indirect(tcx, &mut body);
        }

        _ => {
            make_coroutine_state_argument_pinned(tcx, &mut body);
        }
    }

    // Make sure we remove dead blocks to remove
    // unrelated code from the resume part of the function
    simplify::remove_dead_blocks(&mut body);

    pm::run_passes_no_validate(
        tcx,
        &mut body,
        &[&abort_unwinding_calls::AbortUnwindingCalls],
        None,
    );

    // Run derefer to fix Derefs that are not in the first place
    deref_finder(tcx, &mut body, false);

    if let Some(dumper) = MirDumper::new(tcx, "coroutine_drop_async", &body) {
        dumper.dump_mir(&body);
    }

    body
}

// Create async drop shim proxy function for future_drop_poll
// It is just { call coroutine_drop(); return Poll::Ready(); }
pub(super) fn create_coroutine_drop_shim_proxy_async<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
) -> Body<'tcx> {
    let mut body = body.clone();
    // Take the coroutine info out of the body, since the drop shim is
    // not a coroutine body itself; it just has its drop built out of it.
    let _ = body.coroutine.take();
    let basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>> = IndexVec::new();
    body.basic_blocks = BasicBlocks::new(basic_blocks);
    body.var_debug_info.clear();

    // Keeping return value and args
    body.local_decls.truncate(1 + body.arg_count);

    let source_info = SourceInfo::outermost(body.span);

    // Replace the return variable: Poll<RetT> to Poll<()>
    let poll_adt_ref = tcx.adt_def(tcx.require_lang_item(LangItem::Poll, body.span));
    let poll_enum = Ty::new_adt(tcx, poll_adt_ref, tcx.mk_args(&[tcx.types.unit.into()]));
    body.local_decls[RETURN_PLACE] = LocalDecl::with_source_info(poll_enum, source_info);

    // call coroutine_drop()
    let call_bb = body.basic_blocks_mut().push(BasicBlockData::new(None, false));

    // return Poll::Ready()
    let ret_bb = insert_poll_ready_block(tcx, &mut body);

    let kind = TerminatorKind::Drop {
        place: Place::from(SELF_ARG),
        target: ret_bb,
        unwind: UnwindAction::Continue,
        replace: false,
        drop: None,
    };
    body.basic_blocks_mut()[call_bb].terminator = Some(Terminator { source_info, kind });

    // Run derefer to fix Derefs that are not in the first place
    deref_finder(tcx, &mut body, false);

    if let Some(dumper) = MirDumper::new(tcx, "coroutine_drop_proxy_async", &body) {
        dumper.dump_mir(&body);
    }

    body
}
