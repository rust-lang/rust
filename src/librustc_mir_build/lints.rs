use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::FnKind;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_middle::hir::map::blocks::FnLikeNode;
use rustc_middle::mir::{BasicBlock, Body, ReadOnlyBodyAndCache, TerminatorKind, START_BLOCK};
use rustc_middle::ty::subst::InternalSubsts;
use rustc_middle::ty::{self, AssocItem, AssocItemContainer, Instance, TyCtxt};
use rustc_session::lint::builtin::UNCONDITIONAL_RECURSION;
use std::collections::VecDeque;

crate fn check<'tcx>(tcx: TyCtxt<'tcx>, body: &ReadOnlyBodyAndCache<'_, 'tcx>, def_id: DefId) {
    let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();

    if let Some(fn_like_node) = FnLikeNode::from_node(tcx.hir().get(hir_id)) {
        if let FnKind::Closure(_) = fn_like_node.kind() {
            // closures can't recur, so they don't matter.
            return;
        }

        check_fn_for_unconditional_recursion(tcx, body, def_id);
    }
}

fn check_fn_for_unconditional_recursion<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &ReadOnlyBodyAndCache<'_, 'tcx>,
    def_id: DefId,
) {
    let self_calls = find_blocks_calling_self(tcx, &body, def_id);
    let mut results = IndexVec::from_elem_n(vec![], body.basic_blocks().len());
    let mut queue: VecDeque<_> = self_calls.iter().collect();

    while let Some(bb) = queue.pop_front() {
        if !results[bb].is_empty() {
            // Already propagated.
            continue;
        }

        let locations = if self_calls.contains(bb) {
            // `bb` *is* a self-call.
            vec![bb]
        } else {
            // If *all* successors of `bb` lead to a self-call, emit notes at all of their
            // locations.

            // Converging successors without unwind paths.
            let terminator = body[bb].terminator();
            let relevant_successors = match &terminator.kind {
                TerminatorKind::Call { destination: Some((_, dest)), .. } => {
                    Some(dest).into_iter().chain(&[])
                }
                TerminatorKind::Call { destination: None, .. } => None.into_iter().chain(&[]),
                TerminatorKind::SwitchInt { targets, .. } => None.into_iter().chain(targets),
                TerminatorKind::Goto { target }
                | TerminatorKind::Drop { target, .. }
                | TerminatorKind::DropAndReplace { target, .. }
                | TerminatorKind::Assert { target, .. } => Some(target).into_iter().chain(&[]),
                TerminatorKind::Yield { .. } | TerminatorKind::GeneratorDrop => {
                    None.into_iter().chain(&[])
                }
                TerminatorKind::FalseEdges { real_target, .. }
                | TerminatorKind::FalseUnwind { real_target, .. } => {
                    Some(real_target).into_iter().chain(&[])
                }
                TerminatorKind::Resume
                | TerminatorKind::Abort
                | TerminatorKind::Return
                | TerminatorKind::Unreachable => {
                    unreachable!("unexpected terminator {:?}", terminator.kind)
                }
            };

            let all_are_self_calls =
                relevant_successors.clone().all(|&succ| !results[succ].is_empty());

            if all_are_self_calls {
                relevant_successors.flat_map(|&succ| results[succ].iter().copied()).collect()
            } else {
                vec![]
            }
        };

        if !locations.is_empty() {
            // This is a newly confirmed-to-always-reach-self-call block.
            results[bb] = locations;

            // Propagate backwards through the CFG.
            debug!("propagate loc={:?} in {:?} -> {:?}", results[bb], bb, body.predecessors()[bb]);
            queue.extend(body.predecessors()[bb].iter().copied());
        }
    }

    debug!("unconditional recursion results: {:?}", results);

    let self_call_locations = &mut results[START_BLOCK];
    self_call_locations.sort();
    self_call_locations.dedup();

    if !self_call_locations.is_empty() {
        let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
        let sp = tcx.sess.source_map().guess_head_span(tcx.hir().span(hir_id));
        tcx.struct_span_lint_hir(UNCONDITIONAL_RECURSION, hir_id, sp, |lint| {
            let mut db = lint.build("function cannot return without recursing");
            db.span_label(sp, "cannot return without recursing");
            // offer some help to the programmer.
            for bb in self_call_locations {
                let span = body.basic_blocks()[*bb].terminator().source_info.span;
                db.span_label(span, "recursive call site");
            }
            db.help("a `loop` may express intention better if this is on purpose");
            db.emit();
        });
    }
}

/// Finds blocks with `Call` terminators that would end up calling back into the same method.
fn find_blocks_calling_self<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    def_id: DefId,
) -> BitSet<BasicBlock> {
    let param_env = tcx.param_env(def_id);
    let trait_substs_count = match tcx.opt_associated_item(def_id) {
        Some(AssocItem { container: AssocItemContainer::TraitContainer(trait_def_id), .. }) => {
            tcx.generics_of(trait_def_id).count()
        }
        _ => 0,
    };
    let caller_substs = &InternalSubsts::identity_for_item(tcx, def_id)[..trait_substs_count];

    let mut self_calls = BitSet::new_empty(body.basic_blocks().len());

    for (bb, data) in body.basic_blocks().iter_enumerated() {
        if let TerminatorKind::Call { func, .. } = &data.terminator().kind {
            let func_ty = func.ty(body, tcx);

            if let ty::FnDef(fn_def_id, substs) = func_ty.kind {
                let (call_fn_id, call_substs) =
                    if let Some(instance) = Instance::resolve(tcx, param_env, fn_def_id, substs) {
                        (instance.def_id(), instance.substs)
                    } else {
                        (fn_def_id, substs)
                    };

                // FIXME(#57965): Make this work across function boundaries

                let is_self_call =
                    call_fn_id == def_id && &call_substs[..caller_substs.len()] == caller_substs;

                if is_self_call {
                    self_calls.insert(bb);
                }
            }
        }
    }

    self_calls
}
