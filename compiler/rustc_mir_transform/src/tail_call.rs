use rustc_attr_ir::lang_items::LangItem;
use rustc_hir::Safety;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_index::IndexVec;
use rustc_middle::mir::*;
use rustc_middle::query::Providers;
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::{self, InstanceKind, ShimKind, Ty, TyCtxt};
use rustc_span::{Span, Spanned};

/// Returns `true` if the body of `def_id` contains a guaranteed tail call (`become`).
///
/// Cross-crate results are read from metadata; this (local) provider only runs for local defs.
fn contains_tail_call(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    // `become` can only appear in function-like bodies.
    if !tcx.def_kind(def_id).is_fn_like() {
        return false;
    }

    // We need a body to check (skip trait methods without a default and
    // functions declared in extern blocks).
    if tcx.hir_maybe_body_owned_by(def_id).is_none() {
        return false;
    }

    tcx.mir_built(def_id)
        .borrow()
        .basic_blocks
        .iter()
        .filter_map(|block| block.terminator.as_ref())
        .any(|term| matches!(term.kind, TerminatorKind::TailCall { .. }))
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { contains_tail_call, ..*providers };
}

pub(crate) fn reify_tail_call_shim<'tcx>(
    tcx: TyCtxt<'tcx>,
    local_decls: &mut IndexVec<Local, LocalDecl<'tcx>>,
    callee: DefId,
    callee_args: ty::GenericArgsRef<'tcx>,
    args_tuple: Ty<'tcx>,
    ret: Ty<'tcx>,
    span: Span,
) -> (Local, Statement<'tcx>) {
    // `fn(Args) -> TailNext<Args, Ret>`, the type of every tail-call shim.
    let ptr_ty = Ty::new_fn_ptr(
        tcx,
        ty::Binder::dummy(
            tcx.mk_fn_sig_safe_rust_abi([args_tuple], Ty::new_tail_next(tcx, args_tuple, ret)),
        ),
    );

    // `tail_shim::<callee, Args, Ret>`, which resolves to the callee's tail-call shim via some
    // custom logic in `resolve_instance_raw`.
    let callee_fn_ty = Ty::new_fn_def(tcx, callee, ty::Binder::dummy(callee_args));
    let handle = Operand::function_handle(
        tcx,
        tcx.require_lang_item(LangItem::TailShim, span),
        &[callee_fn_ty.into(), args_tuple.into(), ret.into()],
        span,
    );

    // Turn the function item type into a function pointer type.
    let cast = Rvalue::Cast(
        CastKind::PointerCoercion(
            PointerCoercion::ReifyFnPointer(Safety::Safe),
            CoercionSource::Implicit,
        ),
        handle,
        ptr_ty,
    );

    let local = local_decls.push(LocalDecl::new(ptr_ty, span));
    let stmt = Statement::new(
        SourceInfo::outermost(span),
        StatementKind::Assign(Box::new((Place::from(local), cast))),
    );
    (local, stmt)
}

/// Replace the body of a function that uses guaranteed tail calls (`become`) with a body that calls
/// its TailCall shim repeatedly using.
pub(super) struct LowerTailCall;

impl<'tcx> crate::MirPass<'tcx> for LowerTailCall {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();

        // Don't lower tail calls if this is itself a tail call shim.
        if let InstanceKind::Shim(ShimKind::TailCall(..)) = body.source.instance {
            return;
        }

        // Skip this pass when not using the tail call fallback mechanism.
        if !tcx.sess.use_tail_call_fallback() || !tcx.contains_tail_call(def_id) {
            return;
        }

        // This span and source info are used throughout the custom MIR that is built here.
        // They are not very accurate, but the best location we have.
        let span = body.span;
        let source_info = SourceInfo::outermost(span);

        // Arguments are tupled so we only need one generic parameter for them.
        let arguments: Vec<Ty<'tcx>> = body.args_iter().map(|l| body.local_decls[l].ty).collect();
        let arguments_tuple = Ty::new_tup(tcx, &arguments);
        let ret_ty = body.return_ty();

        // The `tail_eval::<Args, Ret>` function.
        let tail_eval = Operand::function_handle(
            tcx,
            tcx.require_lang_item(LangItem::TailEval, span),
            &[arguments_tuple.into(), ret_ty.into()],
            span,
        );

        // Clear all locals except the return place and the argument locals.
        body.local_decls.truncate(body.arg_count + 1);
        body.var_debug_info.clear();

        // Build the argument tuple to start the call chain.
        let args_tuple = Ty::new_tup(tcx, &arguments);

        // Create the function pointer type from the `tail_shim` item type.
        let (shim_ptr_local, shim_stmt) = reify_tail_call_shim(
            tcx,
            &mut body.local_decls,
            def_id,
            ty::GenericArgs::identity_for_item(tcx, def_id),
            args_tuple,
            ret_ty,
            span,
        );

        // Contruct the argument tuple.
        let arg_locals: Vec<Local> = body.args_iter().collect();
        let tuple_local = body.local_decls.push(LocalDecl::new(args_tuple, span));
        let tuple_stmt = Statement::new(
            source_info,
            StatementKind::Assign(Box::new((
                Place::from(tuple_local),
                Rvalue::Aggregate(
                    Box::new(AggregateKind::Tuple),
                    arg_locals.iter().map(|&l| Operand::Move(Place::from(l))).collect(),
                ),
            ))),
        );

        // The call to tail_eval.
        let bb0 = BasicBlockData::new_stmts(
            vec![shim_stmt, tuple_stmt],
            Some(Terminator {
                source_info,
                kind: TerminatorKind::Call {
                    func: tail_eval,
                    args: Box::from([
                        Spanned { node: Operand::Move(Place::from(shim_ptr_local)), span },
                        Spanned { node: Operand::Move(Place::from(tuple_local)), span },
                    ]),
                    destination: Place::return_place(),
                    target: Some(BasicBlock::from_usize(1)),
                    unwind: UnwindAction::Continue,
                    call_source: CallSource::Misc,
                    fn_span: span,
                },
            }),
            false,
        );

        // Return.
        let bb1 = BasicBlockData::new(
            Some(Terminator { source_info, kind: TerminatorKind::Return }),
            false,
        );

        *body.basic_blocks_mut() = IndexVec::from([bb0, bb1]);

        // The new call to `tail_eval` is created after `AbortUnwindingCalls` already ran, so run
        // it again to pick the right unwind action for this body's ABI and panic strategy.
        crate::abort_unwinding_calls::AbortUnwindingCalls.run_pass(tcx, body);
    }

    /// This pass must run, otherwise `become` cannot be lowered.
    fn policy(&self, _ctx: &crate::PassCtx<'_>) -> crate::PassPolicy {
        crate::PassPolicy::Required
    }
}
