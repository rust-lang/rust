use rustc_data_structures::bit_set::BitSet;
use rustc::hir::def_id::DefId;
use rustc::hir::intravisit::FnKind;
use rustc::hir::map::blocks::FnLikeNode;
use rustc::lint::builtin::UNCONDITIONAL_RECURSION;
use rustc::mir::{self, Body, TerminatorKind};
use rustc::ty::{self, AssocItem, AssocItemContainer, Instance, TyCtxt};
use rustc::ty::subst::InternalSubsts;

pub fn check(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, def_id: DefId) {
    let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();

    if let Some(fn_like_node) = FnLikeNode::from_node(tcx.hir().get(hir_id)) {
        check_fn_for_unconditional_recursion(tcx, fn_like_node.kind(), body, def_id);
    }
}

fn check_fn_for_unconditional_recursion(
    tcx: TyCtxt<'tcx>,
    fn_kind: FnKind<'_>,
    body: &Body<'tcx>,
    def_id: DefId,
) {
    if let FnKind::Closure(_) = fn_kind {
        // closures can't recur, so they don't matter.
        return;
    }

    //FIXME(#54444) rewrite this lint to use the dataflow framework

    // Walk through this function (say `f`) looking to see if
    // every possible path references itself, i.e., the function is
    // called recursively unconditionally. This is done by trying
    // to find a path from the entry node to the exit node that
    // *doesn't* call `f` by traversing from the entry while
    // pretending that calls of `f` are sinks (i.e., ignoring any
    // exit edges from them).
    //
    // NB. this has an edge case with non-returning statements,
    // like `loop {}` or `panic!()`: control flow never reaches
    // the exit node through these, so one can have a function
    // that never actually calls itself but is still picked up by
    // this lint:
    //
    //     fn f(cond: bool) {
    //         if !cond { panic!() } // could come from `assert!(cond)`
    //         f(false)
    //     }
    //
    // In general, functions of that form may be able to call
    // itself a finite number of times and then diverge. The lint
    // considers this to be an error for two reasons, (a) it is
    // easier to implement, and (b) it seems rare to actually want
    // to have behaviour like the above, rather than
    // e.g., accidentally recursing after an assert.

    let basic_blocks = body.basic_blocks();
    let mut reachable_without_self_call_queue = vec![mir::START_BLOCK];
    let mut reached_exit_without_self_call = false;
    let mut self_call_locations = vec![];
    let mut visited = BitSet::new_empty(basic_blocks.len());

    let param_env = tcx.param_env(def_id);
    let trait_substs_count =
        match tcx.opt_associated_item(def_id) {
            Some(AssocItem {
                container: AssocItemContainer::TraitContainer(trait_def_id),
                ..
            }) => tcx.generics_of(trait_def_id).count(),
            _ => 0
        };
    let caller_substs = &InternalSubsts::identity_for_item(tcx, def_id)[..trait_substs_count];

    while let Some(bb) = reachable_without_self_call_queue.pop() {
        if visited.contains(bb) {
            //already done
            continue;
        }

        visited.insert(bb);

        let block = &basic_blocks[bb];

        if let Some(ref terminator) = block.terminator {
            match terminator.kind {
                TerminatorKind::Call { ref func, .. } => {
                    let func_ty = func.ty(body, tcx);

                    if let ty::FnDef(fn_def_id, substs) = func_ty.sty {
                        let (call_fn_id, call_substs) =
                            if let Some(instance) = Instance::resolve(tcx,
                                                                        param_env,
                                                                        fn_def_id,
                                                                        substs) {
                                (instance.def_id(), instance.substs)
                            } else {
                                (fn_def_id, substs)
                            };

                        let is_self_call =
                            call_fn_id == def_id &&
                                &call_substs[..caller_substs.len()] == caller_substs;

                        if is_self_call {
                            self_call_locations.push(terminator.source_info);

                            //this is a self call so we shouldn't explore
                            //further down this path
                            continue;
                        }
                    }
                },
                TerminatorKind::Abort | TerminatorKind::Return => {
                    //found a path!
                    reached_exit_without_self_call = true;
                    break;
                }
                _ => {}
            }

            for successor in terminator.successors() {
                reachable_without_self_call_queue.push(*successor);
            }
        }
    }

    // Check the number of self calls because a function that
    // doesn't return (e.g., calls a `-> !` function or `loop { /*
    // no break */ }`) shouldn't be linted unless it actually
    // recurs.
    if !reached_exit_without_self_call && !self_call_locations.is_empty() {
        let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
        let sp = tcx.sess.source_map().def_span(tcx.hir().span(hir_id));
        let mut db = tcx.struct_span_lint_hir(UNCONDITIONAL_RECURSION,
                                              hir_id,
                                              sp,
                                              "function cannot return without recursing");
        db.span_label(sp, "cannot return without recursing");
        // offer some help to the programmer.
        for location in &self_call_locations {
            db.span_label(location.span, "recursive call site");
        }
        db.help("a `loop` may express intention better if this is on purpose");
        db.emit();
    }
}
