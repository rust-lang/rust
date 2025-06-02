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
        if let Rvalue::Aggregate(kind, _) = rvalue {
            if let AggregateKind::Adt(_, _, ref mut args, _, _) = **kind {
                *args = self.tcx.mk_args(&[self.tcx.types.unit.into()]);
            }
        }
    }
}

// rv = call fut.poll()
fn build_poll_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    poll_unit_place: &Place<'tcx>,
    switch_block: BasicBlock,
    fut_pin_place: &Place<'tcx>,
    fut_ty: Ty<'tcx>,
    context_ref_place: &Place<'tcx>,
    unwind: UnwindAction,
) -> BasicBlock {
    let poll_fn = tcx.require_lang_item(LangItem::FuturePoll, None);
    let poll_fn = Ty::new_fn_def(tcx, poll_fn, [fut_ty]);
    let poll_fn = Operand::Constant(Box::new(ConstOperand {
        span: DUMMY_SP,
        user_ty: None,
        const_: Const::zero_sized(poll_fn),
    }));
    let call = TerminatorKind::Call {
        func: poll_fn.clone(),
        args: [
            dummy_spanned(Operand::Move(*fut_pin_place)),
            dummy_spanned(Operand::Move(*context_ref_place)),
        ]
        .into(),
        destination: *poll_unit_place,
        target: Some(switch_block),
        unwind,
        call_source: CallSource::Misc,
        fn_span: DUMMY_SP,
    };
    insert_term_block(body, call)
}

// pin_fut = Pin::new_unchecked(&mut fut)
fn build_pin_fut<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    fut_place: Place<'tcx>,
    unwind: UnwindAction,
) -> (BasicBlock, Place<'tcx>) {
    let span = body.span;
    let source_info = SourceInfo::outermost(span);
    let fut_ty = fut_place.ty(&body.local_decls, tcx).ty;
    let fut_ref_ty = Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, fut_ty);
    let fut_ref_place = Place::from(body.local_decls.push(LocalDecl::new(fut_ref_ty, span)));
    let pin_fut_new_unchecked_fn = Ty::new_fn_def(
        tcx,
        tcx.require_lang_item(LangItem::PinNewUnchecked, Some(span)),
        [fut_ref_ty],
    );
    let fut_pin_ty = pin_fut_new_unchecked_fn.fn_sig(tcx).output().skip_binder();
    let fut_pin_place = Place::from(body.local_decls.push(LocalDecl::new(fut_pin_ty, span)));
    let pin_fut_new_unchecked_fn = Operand::Constant(Box::new(ConstOperand {
        span,
        user_ty: None,
        const_: Const::zero_sized(pin_fut_new_unchecked_fn),
    }));

    let storage_live =
        Statement { source_info, kind: StatementKind::StorageLive(fut_pin_place.local) };

    let fut_ref_assign = Statement {
        source_info,
        kind: StatementKind::Assign(Box::new((
            fut_ref_place,
            Rvalue::Ref(
                tcx.lifetimes.re_erased,
                BorrowKind::Mut { kind: MutBorrowKind::Default },
                fut_place,
            ),
        ))),
    };

    // call Pin<FutTy>::new_unchecked(&mut fut)
    let pin_fut_bb = body.basic_blocks_mut().push(BasicBlockData {
        statements: [storage_live, fut_ref_assign].to_vec(),
        terminator: Some(Terminator {
            source_info,
            kind: TerminatorKind::Call {
                func: pin_fut_new_unchecked_fn,
                args: [dummy_spanned(Operand::Move(fut_ref_place))].into(),
                destination: fut_pin_place,
                target: None, // will be fixed later
                unwind,
                call_source: CallSource::Misc,
                fn_span: span,
            },
        }),
        is_cleanup: false,
    });
    (pin_fut_bb, fut_pin_place)
}

// Build Poll switch for async drop
// match rv {
//     Ready() => ready_block
//     Pending => yield_block
//}
fn build_poll_switch<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    poll_enum: Ty<'tcx>,
    poll_unit_place: &Place<'tcx>,
    ready_block: BasicBlock,
    yield_block: BasicBlock,
) -> BasicBlock {
    let poll_enum_adt = poll_enum.ty_adt_def().unwrap();

    let Discr { val: poll_ready_discr, ty: poll_discr_ty } = poll_enum
        .discriminant_for_variant(
            tcx,
            poll_enum_adt.variant_index_with_id(tcx.require_lang_item(LangItem::PollReady, None)),
        )
        .unwrap();
    let poll_pending_discr = poll_enum
        .discriminant_for_variant(
            tcx,
            poll_enum_adt.variant_index_with_id(tcx.require_lang_item(LangItem::PollPending, None)),
        )
        .unwrap()
        .val;
    let source_info = SourceInfo::outermost(body.span);
    let poll_discr_place =
        Place::from(body.local_decls.push(LocalDecl::new(poll_discr_ty, source_info.span)));
    let discr_assign = Statement {
        source_info,
        kind: StatementKind::Assign(Box::new((
            poll_discr_place,
            Rvalue::Discriminant(*poll_unit_place),
        ))),
    };
    let unreachable_block = insert_term_block(body, TerminatorKind::Unreachable);
    body.basic_blocks_mut().push(BasicBlockData {
        statements: [discr_assign].to_vec(),
        terminator: Some(Terminator {
            source_info,
            kind: TerminatorKind::SwitchInt {
                discr: Operand::Move(poll_discr_place),
                targets: SwitchTargets::new(
                    [(poll_ready_discr, ready_block), (poll_pending_discr, yield_block)]
                        .into_iter(),
                    unreachable_block,
                ),
            },
        }),
        is_cleanup: false,
    })
}

// Gather blocks, reachable through 'drop' targets of Yield and Drop terminators (chained)
fn gather_dropline_blocks<'tcx>(body: &mut Body<'tcx>) -> DenseBitSet<BasicBlock> {
    let mut dropline: DenseBitSet<BasicBlock> = DenseBitSet::new_empty(body.basic_blocks.len());
    for (bb, data) in traversal::reverse_postorder(body) {
        if dropline.contains(bb) {
            data.terminator().successors().for_each(|v| {
                dropline.insert(v);
            });
        } else {
            match data.terminator().kind {
                TerminatorKind::Yield { drop: Some(v), .. } => {
                    dropline.insert(v);
                }
                TerminatorKind::Drop { drop: Some(v), .. } => {
                    dropline.insert(v);
                }
                _ => (),
            }
        }
    }
    dropline
}

/// Cleanup all async drops (reset to sync)
pub(super) fn cleanup_async_drops<'tcx>(body: &mut Body<'tcx>) {
    for block in body.basic_blocks_mut() {
        if let TerminatorKind::Drop {
            place: _,
            target: _,
            unwind: _,
            replace: _,
            ref mut drop,
            ref mut async_fut,
        } = block.terminator_mut().kind
        {
            if drop.is_some() || async_fut.is_some() {
                *drop = None;
                *async_fut = None;
            }
        }
    }
}

pub(super) fn has_expandable_async_drops<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    coroutine_ty: Ty<'tcx>,
) -> bool {
    for bb in START_BLOCK..body.basic_blocks.next_index() {
        // Drops in unwind path (cleanup blocks) are not expanded to async drops, only sync drops in unwind path
        if body[bb].is_cleanup {
            continue;
        }
        let TerminatorKind::Drop { place, target: _, unwind: _, replace: _, drop: _, async_fut } =
            body[bb].terminator().kind
        else {
            continue;
        };
        let place_ty = place.ty(&body.local_decls, tcx).ty;
        if place_ty == coroutine_ty {
            continue;
        }
        if async_fut.is_none() {
            continue;
        }
        return true;
    }
    return false;
}

/// Expand Drop terminator for async drops into mainline poll-switch and dropline poll-switch
pub(super) fn expand_async_drops<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    context_mut_ref: Ty<'tcx>,
    coroutine_kind: hir::CoroutineKind,
    coroutine_ty: Ty<'tcx>,
) {
    let dropline = gather_dropline_blocks(body);
    // Clean drop and async_fut fields if potentially async drop is not expanded (stays sync)
    let remove_asyncness = |block: &mut BasicBlockData<'tcx>| {
        if let TerminatorKind::Drop {
            place: _,
            target: _,
            unwind: _,
            replace: _,
            ref mut drop,
            ref mut async_fut,
        } = block.terminator_mut().kind
        {
            *drop = None;
            *async_fut = None;
        }
    };
    for bb in START_BLOCK..body.basic_blocks.next_index() {
        // Drops in unwind path (cleanup blocks) are not expanded to async drops, only sync drops in unwind path
        if body[bb].is_cleanup {
            remove_asyncness(&mut body[bb]);
            continue;
        }
        let TerminatorKind::Drop { place, target, unwind, replace: _, drop, async_fut } =
            body[bb].terminator().kind
        else {
            continue;
        };

        let place_ty = place.ty(&body.local_decls, tcx).ty;
        if place_ty == coroutine_ty {
            remove_asyncness(&mut body[bb]);
            continue;
        }

        let Some(fut_local) = async_fut else {
            remove_asyncness(&mut body[bb]);
            continue;
        };

        let is_dropline_bb = dropline.contains(bb);

        if !is_dropline_bb && drop.is_none() {
            remove_asyncness(&mut body[bb]);
            continue;
        }

        let fut_place = Place::from(fut_local);
        let fut_ty = fut_place.ty(&body.local_decls, tcx).ty;

        // poll-code:
        // state_call_drop:
        // #bb_pin: fut_pin = Pin<FutT>::new_unchecked(&mut fut)
        // #bb_call: rv = call fut.poll() (or future_drop_poll(fut) for internal future drops)
        // #bb_check: match (rv)
        //  pending => return rv (yield)
        //  ready => *continue_bb|drop_bb*

        // Compute Poll<> (aka Poll with void return)
        let poll_adt_ref = tcx.adt_def(tcx.require_lang_item(LangItem::Poll, None));
        let poll_enum = Ty::new_adt(tcx, poll_adt_ref, tcx.mk_args(&[tcx.types.unit.into()]));
        let poll_decl = LocalDecl::new(poll_enum, body.span);
        let poll_unit_place = Place::from(body.local_decls.push(poll_decl));

        // First state-loop yield for mainline
        let context_ref_place =
            Place::from(body.local_decls.push(LocalDecl::new(context_mut_ref, body.span)));
        let source_info = body[bb].terminator.as_ref().unwrap().source_info;
        let arg = Rvalue::Use(Operand::Move(Place::from(CTX_ARG)));
        body[bb].statements.push(Statement {
            source_info,
            kind: StatementKind::Assign(Box::new((context_ref_place, arg))),
        });
        let yield_block = insert_term_block(body, TerminatorKind::Unreachable); // `kind` replaced later to yield
        let switch_block =
            build_poll_switch(tcx, body, poll_enum, &poll_unit_place, target, yield_block);
        let (pin_bb, fut_pin_place) =
            build_pin_fut(tcx, body, fut_place.clone(), UnwindAction::Continue);
        let call_bb = build_poll_call(
            tcx,
            body,
            &poll_unit_place,
            switch_block,
            &fut_pin_place,
            fut_ty,
            &context_ref_place,
            unwind,
        );

        // Second state-loop yield for transition to dropline (when coroutine async drop started)
        let mut dropline_transition_bb: Option<BasicBlock> = None;
        let mut dropline_yield_bb: Option<BasicBlock> = None;
        let mut dropline_context_ref: Option<Place<'_>> = None;
        let mut dropline_call_bb: Option<BasicBlock> = None;
        if !is_dropline_bb {
            let context_ref_place2: Place<'_> =
                Place::from(body.local_decls.push(LocalDecl::new(context_mut_ref, body.span)));
            let drop_yield_block = insert_term_block(body, TerminatorKind::Unreachable); // `kind` replaced later to yield
            let drop_switch_block = build_poll_switch(
                tcx,
                body,
                poll_enum,
                &poll_unit_place,
                drop.unwrap(),
                drop_yield_block,
            );
            let (pin_bb2, fut_pin_place2) =
                build_pin_fut(tcx, body, fut_place, UnwindAction::Continue);
            let drop_call_bb = build_poll_call(
                tcx,
                body,
                &poll_unit_place,
                drop_switch_block,
                &fut_pin_place2,
                fut_ty,
                &context_ref_place2,
                unwind,
            );
            dropline_transition_bb = Some(pin_bb2);
            dropline_yield_bb = Some(drop_yield_block);
            dropline_context_ref = Some(context_ref_place2);
            dropline_call_bb = Some(drop_call_bb);
        }

        let value =
            if matches!(coroutine_kind, CoroutineKind::Desugared(CoroutineDesugaring::AsyncGen, _))
            {
                // For AsyncGen we need `yield Poll<OptRet>::Pending`
                let full_yield_ty = body.yield_ty().unwrap();
                let ty::Adt(_poll_adt, args) = *full_yield_ty.kind() else { bug!() };
                let ty::Adt(_option_adt, args) = *args.type_at(0).kind() else { bug!() };
                let yield_ty = args.type_at(0);
                Operand::Constant(Box::new(ConstOperand {
                    span: source_info.span,
                    const_: Const::Unevaluated(
                        UnevaluatedConst::new(
                            tcx.require_lang_item(LangItem::AsyncGenPending, None),
                            tcx.mk_args(&[yield_ty.into()]),
                        ),
                        full_yield_ty,
                    ),
                    user_ty: None,
                }))
            } else {
                // value needed only for return-yields or gen-coroutines, so just const here
                Operand::Constant(Box::new(ConstOperand {
                    span: body.span,
                    user_ty: None,
                    const_: Const::from_bool(tcx, false),
                }))
            };

        use rustc_middle::mir::AssertKind::ResumedAfterDrop;
        let panic_bb = insert_panic_block(tcx, body, ResumedAfterDrop(coroutine_kind));

        if is_dropline_bb {
            body[yield_block].terminator_mut().kind = TerminatorKind::Yield {
                value: value.clone(),
                resume: panic_bb,
                resume_arg: context_ref_place,
                drop: Some(pin_bb),
            };
        } else {
            body[yield_block].terminator_mut().kind = TerminatorKind::Yield {
                value: value.clone(),
                resume: pin_bb,
                resume_arg: context_ref_place,
                drop: dropline_transition_bb,
            };
            body[dropline_yield_bb.unwrap()].terminator_mut().kind = TerminatorKind::Yield {
                value,
                resume: panic_bb,
                resume_arg: dropline_context_ref.unwrap(),
                drop: dropline_transition_bb,
            };
        }

        if let TerminatorKind::Call { ref mut target, .. } = body[pin_bb].terminator_mut().kind {
            *target = Some(call_bb);
        } else {
            bug!()
        }
        if !is_dropline_bb {
            if let TerminatorKind::Call { ref mut target, .. } =
                body[dropline_transition_bb.unwrap()].terminator_mut().kind
            {
                *target = dropline_call_bb;
            } else {
                bug!()
            }
        }

        body[bb].terminator_mut().kind = TerminatorKind::Goto { target: pin_bb };
    }
}

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
        produce_async_drops: false,
    };

    for (block, block_data) in body.basic_blocks.iter_enumerated() {
        let (target, unwind, source_info, dropline) = match block_data.terminator() {
            Terminator {
                source_info,
                kind: TerminatorKind::Drop { place, target, unwind, replace: _, drop, async_fut: _ },
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

pub(super) fn insert_clean_drop<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    has_async_drops: bool,
) -> BasicBlock {
    let source_info = SourceInfo::outermost(body.span);
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
        async_fut: None,
    };

    // Create a block to destroy an unresumed coroutines. This can only destroy upvars.
    body.basic_blocks_mut().push(BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator { source_info, kind: term }),
        is_cleanup: false,
    })
}

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
    // building a body for `drop_in_place`.
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

    // Change the coroutine argument from &mut to *mut
    body.local_decls[SELF_ARG] =
        LocalDecl::with_source_info(Ty::new_mut_ptr(tcx, coroutine_ty), source_info);

    // Make sure we remove dead blocks to remove
    // unrelated code from the resume part of the function
    simplify::remove_dead_blocks(&mut body);

    // Update the body's def to become the drop glue.
    let coroutine_instance = body.source.instance;
    let drop_in_place = tcx.require_lang_item(LangItem::DropInPlace, None);
    let drop_instance = InstanceKind::DropGlue(drop_in_place, Some(coroutine_ty));

    // Temporary change MirSource to coroutine's instance so that dump_mir produces more sensible
    // filename.
    body.source.instance = coroutine_instance;
    dump_mir(tcx, false, "coroutine_drop", &0, &body, |_, _| Ok(()));
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

    let mut cases = create_cases(&mut body, transform, Operation::Drop);

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
    let poll_adt_ref = tcx.adt_def(tcx.require_lang_item(LangItem::Poll, None));
    let poll_enum = Ty::new_adt(tcx, poll_adt_ref, tcx.mk_args(&[tcx.types.unit.into()]));
    body.local_decls[RETURN_PLACE] = LocalDecl::with_source_info(poll_enum, source_info);

    make_coroutine_state_argument_indirect(tcx, &mut body);

    match transform.coroutine_kind {
        // Iterator::next doesn't accept a pinned argument,
        // unlike for all other coroutine kinds.
        CoroutineKind::Desugared(CoroutineDesugaring::Gen, _) => {}
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

    dump_mir(tcx, false, "coroutine_drop_async", &0, &body, |_, _| Ok(()));

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
    let poll_adt_ref = tcx.adt_def(tcx.require_lang_item(LangItem::Poll, None));
    let poll_enum = Ty::new_adt(tcx, poll_adt_ref, tcx.mk_args(&[tcx.types.unit.into()]));
    body.local_decls[RETURN_PLACE] = LocalDecl::with_source_info(poll_enum, source_info);

    // call coroutine_drop()
    let call_bb = body.basic_blocks_mut().push(BasicBlockData {
        statements: Vec::new(),
        terminator: None,
        is_cleanup: false,
    });

    // return Poll::Ready()
    let ret_bb = insert_poll_ready_block(tcx, &mut body);

    let kind = TerminatorKind::Drop {
        place: Place::from(SELF_ARG),
        target: ret_bb,
        unwind: UnwindAction::Continue,
        replace: false,
        drop: None,
        async_fut: None,
    };
    body.basic_blocks_mut()[call_bb].terminator = Some(Terminator { source_info, kind });

    dump_mir(tcx, false, "coroutine_drop_proxy_async", &0, &body, |_, _| Ok(()));

    body
}
