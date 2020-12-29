use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::mir::TerminatorKind;
use rustc_middle::ty::TypeFoldable;
use rustc_middle::ty::{self, subst::SubstsRef, InstanceDef, TyCtxt};

// FIXME: check whether it is cheaper to precompute the entire call graph instead of invoking
// this query riddiculously often.
#[instrument(skip(tcx, root, target))]
crate fn mir_callgraph_reachable(
    tcx: TyCtxt<'tcx>,
    (root, target): (ty::Instance<'tcx>, LocalDefId),
) -> bool {
    trace!(%root, target = %tcx.def_path_str(target.to_def_id()));
    let param_env = tcx.param_env_reveal_all_normalized(target);
    assert_ne!(
        root.def_id().expect_local(),
        target,
        "you should not call `mir_callgraph_reachable` on immediate self recursion"
    );
    assert!(
        matches!(root.def, InstanceDef::Item(_)),
        "you should not call `mir_callgraph_reachable` on shims"
    );
    assert!(
        !tcx.is_constructor(root.def_id()),
        "you should not call `mir_callgraph_reachable` on enum/struct constructor functions"
    );
    #[instrument(skip(tcx, param_env, target, stack, seen, recursion_limiter, caller))]
    fn process(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        caller: ty::Instance<'tcx>,
        target: LocalDefId,
        stack: &mut Vec<ty::Instance<'tcx>>,
        seen: &mut FxHashSet<ty::Instance<'tcx>>,
        recursion_limiter: &mut FxHashMap<DefId, usize>,
    ) -> bool {
        trace!(%caller);
        for &(callee, substs) in tcx.mir_inliner_callees(caller.def) {
            let substs = caller.subst_mir_and_normalize_erasing_regions(tcx, param_env, substs);
            let callee = match ty::Instance::resolve(tcx, param_env, callee, substs).unwrap() {
                Some(callee) => callee,
                None => {
                    trace!(?callee, "cannot resolve, skipping");
                    continue;
                }
            };

            // Found a path.
            if callee.def_id() == target.to_def_id() {
                return true;
            }

            if tcx.is_constructor(callee.def_id()) {
                trace!("constructors always have MIR");
                // Constructor functions cannot cause a query cycle.
                continue;
            }

            match callee.def {
                InstanceDef::Item(_) => {
                    // If there is no MIR available (either because it was not in metadata or
                    // because it has no MIR because it's an extern function), then the inliner
                    // won't cause cycles on this.
                    if !tcx.is_mir_available(callee.def_id()) {
                        trace!(?callee, "no mir available, skipping");
                        continue;
                    }
                }
                // These have no own callable MIR.
                InstanceDef::Intrinsic(_) | InstanceDef::Virtual(..) => continue,
                // These have MIR and if that MIR is inlined, substituted and then inlining is run
                // again, a function item can end up getting inlined. Thus we'll be able to cause
                // a cycle that way
                InstanceDef::VtableShim(_)
                | InstanceDef::ReifyShim(_)
                | InstanceDef::FnPtrShim(..)
                | InstanceDef::ClosureOnceShim { .. }
                | InstanceDef::CloneShim(..) => {}
                InstanceDef::DropGlue(..) => {
                    // FIXME: A not fully substituted drop shim can cause ICEs if one attempts to
                    // have its MIR built. Likely oli-obk just screwed up the `ParamEnv`s, so this
                    // needs some more analysis.
                    if callee.needs_subst() {
                        continue;
                    }
                }
            }

            if seen.insert(callee) {
                let recursion = recursion_limiter.entry(callee.def_id()).or_default();
                trace!(?callee, recursion = *recursion);
                if tcx.sess.recursion_limit().value_within_limit(*recursion) {
                    *recursion += 1;
                    stack.push(callee);
                    let found_recursion = ensure_sufficient_stack(|| {
                        process(tcx, param_env, callee, target, stack, seen, recursion_limiter)
                    });
                    if found_recursion {
                        return true;
                    }
                    stack.pop();
                } else {
                    // Pessimistically assume that there could be recursion.
                    return true;
                }
            }
        }
        false
    }
    process(
        tcx,
        param_env,
        root,
        target,
        &mut Vec::new(),
        &mut FxHashSet::default(),
        &mut FxHashMap::default(),
    )
}

crate fn mir_inliner_callees<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::InstanceDef<'tcx>,
) -> &'tcx [(DefId, SubstsRef<'tcx>)] {
    let steal;
    let guard;
    let body = match (instance, instance.def_id().as_local()) {
        (InstanceDef::Item(_), Some(def_id)) => {
            let def = ty::WithOptConstParam::unknown(def_id);
            steal = tcx.mir_promoted(def).0;
            guard = steal.borrow();
            &*guard
        }
        // Functions from other crates and MIR shims
        _ => tcx.instance_mir(instance),
    };
    let mut calls = Vec::new();
    for bb_data in body.basic_blocks() {
        let terminator = bb_data.terminator();
        if let TerminatorKind::Call { func, .. } = &terminator.kind {
            let ty = func.ty(&body.local_decls, tcx);
            let call = match ty.kind() {
                ty::FnDef(def_id, substs) => (*def_id, *substs),
                _ => continue,
            };
            // We've seen this before
            if calls.contains(&call) {
                continue;
            }
            calls.push(call);
        }
    }
    tcx.arena.alloc_slice(&calls)
}
