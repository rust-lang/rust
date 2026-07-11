//! Elides a temporary `Vec` between a slice iterator pipeline and an integer fold.
//!
//! This pass intentionally runs before MIR inlining, while calls to the iterator methods are
//! still recognizable. It turns
//!
//! ```text
//! slice.iter().map(f).collect::<Vec<I>>().into_iter().fold(init, integer_op)
//! ```
//!
//! into a fold over the original iterator directly. The narrow recognition rules matter: the
//! accepted wrapping and saturating integer reducers cannot panic, integer elements have no drop
//! glue, and the accepted standard iterator adapters have no observable `size_hint` behavior.
//! Therefore interleaving the reduction with calls to non-capturing adapter closures does not
//! change panic or drop ordering.

use rustc_hir::LangItem;
use rustc_middle::mir::{
    BasicBlock, Body, Const, ConstOperand, Local, Operand, SourceInfo, Statement, StatementKind,
    TerminatorKind,
};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::sym;
use tracing::debug;

use crate::ssa::SsaLocals;

pub(super) struct TemporaryVecFoldElision;

struct Replacement<'tcx> {
    terminator: TerminatorKind<'tcx>,
    storage_target: BasicBlock,
    storage_dead: Vec<Local>,
    source_info: SourceInfo,
}

impl<'tcx> crate::MirPass<'tcx> for TemporaryVecFoldElision {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let ssa = SsaLocals::new(tcx, body, body.typing_env(tcx));
        let replacements: Vec<_> = body
            .basic_blocks
            .indices()
            .filter_map(|bb| {
                replacement_for(tcx, body, &ssa, bb).map(|replacement| (bb, replacement))
            })
            .collect();

        if replacements.is_empty() {
            return;
        }

        debug!(def_id = ?body.source.def_id(), count = replacements.len(), "eliding temporary Vec folds");
        let blocks = body.basic_blocks.as_mut();
        for (bb, replacement) in replacements {
            blocks[bb].terminator_mut().kind = replacement.terminator;
            for local in replacement.storage_dead.into_iter().rev() {
                blocks[replacement.storage_target].statements.insert(
                    0,
                    Statement::new(replacement.source_info, StatementKind::StorageDead(local)),
                );
            }
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}

fn replacement_for<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    ssa: &SsaLocals,
    collect_bb: BasicBlock,
) -> Option<Replacement<'tcx>> {
    let source_info = body[collect_bb].terminator().source_info;
    let TerminatorKind::Call {
        func: collect_func,
        args: collect_args,
        destination: vec_destination,
        target: Some(into_iter_bb),
        unwind: collect_unwind,
        ..
    } = &body[collect_bb].terminator().kind
    else {
        return None;
    };
    let (collect_def_id, _) = collect_func.const_fn_def()?;
    if !is_diagnostic_method(tcx, collect_def_id, sym::iterator_collect_fn) {
        return None;
    }
    let [producer_arg] = &**collect_args else { return None };
    let producer_local = producer_arg.node.place()?.as_local()?;
    let vec_local = vec_destination.as_local()?;
    let into_iter_bb = *into_iter_bb;

    let vec_ty = body.local_decls[vec_local].ty;
    let ty::Adt(vec_def, vec_args) = *vec_ty.kind() else { return None };
    if !tcx.is_diagnostic_item(sym::Vec, vec_def.did()) {
        return None;
    }
    let item_ty = vec_args.type_at(0);
    if !matches!(item_ty.kind(), ty::Int(_) | ty::Uint(_)) {
        return None;
    }

    if !single_use_class(ssa, vec_local)
        || !supported_slice_pipeline(tcx, body, ssa, producer_local, collect_bb, item_ty)
    {
        return None;
    }

    if !only_predecessor(body, into_iter_bb, collect_bb) {
        return None;
    }
    let TerminatorKind::Call {
        func: into_iter_func,
        args: into_iter_args,
        destination: into_iter_destination,
        target: Some(fold_bb),
        ..
    } = &body[into_iter_bb].terminator().kind
    else {
        return None;
    };
    let (into_iter_def_id, _) = into_iter_func.const_fn_def()?;
    if !is_lang_method(tcx, into_iter_def_id, LangItem::IntoIterIntoIter) {
        return None;
    }
    let [vec_arg] = &**into_iter_args else { return None };
    let vec_arg_local = vec_arg.node.place()?.as_local()?;
    if !same_class(ssa, vec_arg_local, vec_local)
        || vec_arg.node.ty(&body.local_decls, tcx) != vec_ty
    {
        return None;
    }
    let into_iter_local = into_iter_destination.as_local()?;
    let fold_bb = *fold_bb;
    if !single_use_class(ssa, into_iter_local) || !only_predecessor(body, fold_bb, into_iter_bb) {
        return None;
    }

    let TerminatorKind::Call {
        func: fold_func,
        args: fold_args,
        destination: fold_destination,
        target: Some(fold_target),
        call_source,
        fn_span,
        ..
    } = &body[fold_bb].terminator().kind
    else {
        return None;
    };
    if !only_predecessor(body, *fold_target, fold_bb) {
        return None;
    }
    let (fold_def_id, _) = fold_func.const_fn_def()?;
    let fold_trait_item = trait_item(tcx, fold_def_id);
    if tcx.item_name(fold_trait_item) != sym::fold
        || tcx
            .trait_of_assoc(fold_trait_item)
            .is_none_or(|trait_id| !tcx.is_lang_item(trait_id, LangItem::Iterator))
    {
        return None;
    }
    let [fold_iter_arg, init_arg, reducer_arg] = &**fold_args else { return None };
    let fold_iter_local = fold_iter_arg.node.place()?.as_local()?;
    if !same_class(ssa, fold_iter_local, into_iter_local) {
        return None;
    }
    if !matches!(init_arg.node, Operand::Constant(_))
        || init_arg.node.ty(&body.local_decls, tcx) != item_ty
        || fold_destination.ty(&body.local_decls, tcx).ty != item_ty
        || !is_non_panicking_integer_reducer(tcx, &reducer_arg.node, item_ty)
    {
        return None;
    }

    let producer_ty = producer_arg.node.ty(&body.local_decls, tcx);
    let reducer_ty = reducer_arg.node.ty(&body.local_decls, tcx);
    let direct_fold_args = tcx.mk_args(&[producer_ty.into(), item_ty.into(), reducer_ty.into()]);
    if tcx.generics_of(fold_trait_item).count() != direct_fold_args.len() {
        return None;
    }
    let direct_fold_ty = Ty::new_fn_def(tcx, fold_trait_item, direct_fold_args);
    let direct_fold_func = Operand::Constant(Box::new(ConstOperand {
        span: fold_func.span(&body.local_decls),
        user_ty: None,
        const_: Const::zero_sized(direct_fold_ty),
    }));

    let producer_head = ssa.copy_classes()[producer_local];
    let storage_dead =
        ssa.locals().filter(|&local| ssa.copy_classes()[local] == producer_head).collect();

    Some(Replacement {
        terminator: TerminatorKind::Call {
            func: direct_fold_func,
            args: [producer_arg.clone(), init_arg.clone(), reducer_arg.clone()].into(),
            destination: *fold_destination,
            target: Some(*fold_target),
            unwind: *collect_unwind,
            call_source: *call_source,
            fn_span: *fn_span,
        },
        storage_target: *fold_target,
        storage_dead,
        source_info,
    })
}

fn trait_item(tcx: TyCtxt<'_>, def_id: rustc_hir::def_id::DefId) -> rustc_hir::def_id::DefId {
    tcx.trait_item_of(def_id).unwrap_or(def_id)
}

fn is_diagnostic_method(
    tcx: TyCtxt<'_>,
    def_id: rustc_hir::def_id::DefId,
    name: rustc_span::Symbol,
) -> bool {
    tcx.is_diagnostic_item(name, trait_item(tcx, def_id))
}

fn is_lang_method(tcx: TyCtxt<'_>, def_id: rustc_hir::def_id::DefId, item: LangItem) -> bool {
    tcx.is_lang_item(trait_item(tcx, def_id), item)
}

fn supported_slice_pipeline<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    ssa: &SsaLocals,
    local: Local,
    consumer_bb: BasicBlock,
    item_ty: Ty<'tcx>,
) -> bool {
    if !single_use_class(ssa, local) {
        return false;
    }
    let Some((producer_bb, destination_local)) = assigning_call(body, ssa, local) else {
        return false;
    };
    if !same_class(ssa, destination_local, local)
        || !only_predecessor(body, consumer_bb, producer_bb)
    {
        return false;
    }
    let TerminatorKind::Call { func, args, destination, target: Some(target), .. } =
        &body[producer_bb].terminator().kind
    else {
        return false;
    };
    if *target != consumer_bb || destination.as_local() != Some(destination_local) {
        return false;
    }
    let Some((def_id, _)) = func.const_fn_def() else { return false };

    if tcx.is_diagnostic_item(sym::slice_iter, def_id) {
        let [slice_arg] = &**args else { return false };
        let slice_ty = slice_arg.node.ty(&body.local_decls, tcx);
        let ty::Ref(_, pointee, _) = *slice_ty.kind() else { return false };
        let ty::Slice(slice_item_ty) = *pointee.kind() else { return false };
        return slice_item_ty == item_ty;
    }

    let adapter = trait_item(tcx, def_id);
    let (input_arg, closure_arg) = if tcx.is_diagnostic_item(sym::IteratorMap, adapter)
        || tcx.is_diagnostic_item(sym::iter_filter, adapter)
    {
        let [input_arg, closure_arg] = &**args else { return false };
        (input_arg, Some(closure_arg))
    } else if tcx.is_diagnostic_item(sym::iter_copied, adapter)
        || tcx.is_diagnostic_item(sym::iter_cloned, adapter)
        || tcx.is_diagnostic_item(sym::enumerate_method, adapter)
    {
        let [input_arg] = &**args else { return false };
        (input_arg, None)
    } else {
        return false;
    };

    if closure_arg.is_some_and(|arg| !is_non_capturing_closure(tcx, body, &arg.node)) {
        return false;
    }
    let Some(input_local) = input_arg.node.place().and_then(|place| place.as_local()) else {
        return false;
    };
    supported_slice_pipeline(tcx, body, ssa, input_local, producer_bb, item_ty)
}

fn is_non_capturing_closure<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    operand: &Operand<'tcx>,
) -> bool {
    let closure_ty = operand.ty(&body.local_decls, tcx);
    let ty::Closure(_, closure_args) = *closure_ty.kind() else { return false };
    closure_args.as_closure().upvar_tys().is_empty()
}

fn is_non_panicking_integer_reducer<'tcx>(
    tcx: TyCtxt<'tcx>,
    reducer: &Operand<'tcx>,
    item_ty: Ty<'tcx>,
) -> bool {
    let Some((def_id, args)) = reducer.const_fn_def() else { return false };
    if !matches!(
        tcx.item_name(def_id),
        sym::saturating_add
            | sym::saturating_mul
            | sym::saturating_sub
            | sym::wrapping_add
            | sym::wrapping_mul
            | sym::wrapping_sub
    ) {
        return false;
    }
    let Some(impl_id) = tcx.inherent_impl_of_assoc(def_id) else { return false };
    let self_ty = tcx.type_of(impl_id).instantiate(tcx, args).skip_norm_wip();
    if self_ty != item_ty {
        return false;
    }
    let sig = tcx.fn_sig(def_id).instantiate(tcx, args).skip_binder();
    sig.inputs() == [item_ty, item_ty] && sig.output() == item_ty
}

fn assigning_call(body: &Body<'_>, ssa: &SsaLocals, local: Local) -> Option<(BasicBlock, Local)> {
    let head = ssa.copy_classes()[local];
    let mut result = None;
    for (bb, data) in body.basic_blocks.iter_enumerated() {
        let TerminatorKind::Call { destination, .. } = &data.terminator().kind else { continue };
        let Some(destination) = destination.as_local() else { continue };
        if ssa.copy_classes()[destination] != head {
            continue;
        }
        if result.replace((bb, destination)).is_some() {
            return None;
        }
    }
    result
}

fn same_class(ssa: &SsaLocals, left: Local, right: Local) -> bool {
    ssa.copy_classes()[left] == ssa.copy_classes()[right]
}

fn single_use_class(ssa: &SsaLocals, local: Local) -> bool {
    let head = ssa.copy_classes()[local];
    let mut uses = 0_u32;
    for candidate in ssa.locals().filter(|&candidate| ssa.copy_classes()[candidate] == head) {
        if !ssa.is_ssa(candidate) {
            return false;
        }
        uses = uses.saturating_add(ssa.num_direct_uses(candidate));
    }
    uses == 1
}

fn only_predecessor(body: &Body<'_>, block: BasicBlock, predecessor: BasicBlock) -> bool {
    body.basic_blocks.predecessors()[block].as_slice() == [predecessor]
}
