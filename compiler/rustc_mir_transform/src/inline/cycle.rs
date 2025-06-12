use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::mir::TerminatorKind;
use rustc_middle::ty::{self, GenericArgsRef, InstanceKind, TyCtxt, TypeVisitableExt};
use rustc_session::Limit;
use rustc_span::sym;
use tracing::{instrument, trace};

// FIXME: check whether it is cheaper to precompute the entire call graph instead of invoking
// this query ridiculously often.
#[instrument(level = "debug", skip(tcx, root, target))]
pub(crate) fn mir_callgraph_reachable<'tcx>(
    tcx: TyCtxt<'tcx>,
    (root, target): (ty::Instance<'tcx>, LocalDefId),
) -> bool {
    trace!(%root, target = %tcx.def_path_str(target));
    assert_ne!(
        root.def_id().expect_local(),
        target,
        "you should not call `mir_callgraph_reachable` on immediate self recursion"
    );
    assert!(
        matches!(root.def, InstanceKind::Item(_)),
        "you should not call `mir_callgraph_reachable` on shims"
    );
    assert!(
        !tcx.is_constructor(root.def_id()),
        "you should not call `mir_callgraph_reachable` on enum/struct constructor functions"
    );
    #[instrument(
        level = "debug",
        skip(tcx, typing_env, target, stack, seen, recursion_limiter, caller, recursion_limit)
    )]
    fn process<'tcx>(
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        caller: ty::Instance<'tcx>,
        target: LocalDefId,
        stack: &mut Vec<ty::Instance<'tcx>>,
        seen: &mut FxHashSet<ty::Instance<'tcx>>,
        recursion_limiter: &mut FxHashMap<DefId, usize>,
        recursion_limit: Limit,
    ) -> bool {
        trace!(%caller);
        for &(callee, args) in tcx.mir_inliner_callees(caller.def) {
            let Ok(args) = caller.try_instantiate_mir_and_normalize_erasing_regions(
                tcx,
                typing_env,
                ty::EarlyBinder::bind(args),
            ) else {
                trace!(?caller, ?typing_env, ?args, "cannot normalize, skipping");
                continue;
            };
            let Ok(Some(callee)) = ty::Instance::try_resolve(tcx, typing_env, callee, args) else {
                trace!(?callee, "cannot resolve, skipping");
                continue;
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
                InstanceKind::Item(_) => {
                    // If there is no MIR available (either because it was not in metadata or
                    // because it has no MIR because it's an extern function), then the inliner
                    // won't cause cycles on this.
                    if !tcx.is_mir_available(callee.def_id()) {
                        trace!(?callee, "no mir available, skipping");
                        continue;
                    }
                }
                // These have no own callable MIR.
                InstanceKind::Intrinsic(_) | InstanceKind::Virtual(..) => continue,
                // These have MIR and if that MIR is inlined, instantiated and then inlining is run
                // again, a function item can end up getting inlined. Thus we'll be able to cause
                // a cycle that way
                InstanceKind::VTableShim(_)
                | InstanceKind::ReifyShim(..)
                | InstanceKind::FnPtrShim(..)
                | InstanceKind::ClosureOnceShim { .. }
                | InstanceKind::ConstructCoroutineInClosureShim { .. }
                | InstanceKind::ThreadLocalShim { .. }
                | InstanceKind::CloneShim(..) => {}

                // This shim does not call any other functions, thus there can be no recursion.
                InstanceKind::FnPtrAddrShim(..) => {
                    continue;
                }
                InstanceKind::DropGlue(..)
                | InstanceKind::FutureDropPollShim(..)
                | InstanceKind::AsyncDropGlue(..)
                | InstanceKind::AsyncDropGlueCtorShim(..) => {
                    // FIXME: A not fully instantiated drop shim can cause ICEs if one attempts to
                    // have its MIR built. Likely oli-obk just screwed up the `ParamEnv`s, so this
                    // needs some more analysis.
                    if callee.has_param() {
                        continue;
                    }
                }
            }

            if seen.insert(callee) {
                let recursion = recursion_limiter.entry(callee.def_id()).or_default();
                trace!(?callee, recursion = *recursion);
                if recursion_limit.value_within_limit(*recursion) {
                    *recursion += 1;
                    stack.push(callee);
                    let found_recursion = ensure_sufficient_stack(|| {
                        process(
                            tcx,
                            typing_env,
                            callee,
                            target,
                            stack,
                            seen,
                            recursion_limiter,
                            recursion_limit,
                        )
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
    // FIXME(-Znext-solver=no): Remove this hack when trait solver overflow can return an error.
    // In code like that pointed out in #128887, the type complexity we ask the solver to deal with
    // grows as we recurse into the call graph. If we use the same recursion limit here and in the
    // solver, the solver hits the limit first and emits a fatal error. But if we use a reduced
    // limit, we will hit the limit first and give up on looking for inlining. And in any case,
    // the default recursion limits are quite generous for us. If we need to recurse 64 times
    // into the call graph, we're probably not going to find any useful MIR inlining.
    let recursion_limit = tcx.recursion_limit() / 2;
    process(
        tcx,
        ty::TypingEnv::post_analysis(tcx, target),
        root,
        target,
        &mut Vec::new(),
        &mut FxHashSet::default(),
        &mut FxHashMap::default(),
        recursion_limit,
    )
}

pub(crate) fn mir_inliner_callees<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::InstanceKind<'tcx>,
) -> &'tcx [(DefId, GenericArgsRef<'tcx>)] {
    let steal;
    let guard;
    let body = match (instance, instance.def_id().as_local()) {
        (InstanceKind::Item(_), Some(def_id)) => {
            steal = tcx.mir_promoted(def_id).0;
            guard = steal.borrow();
            &*guard
        }
        // Functions from other crates and MIR shims
        _ => tcx.instance_mir(instance),
    };
    let mut calls = FxIndexSet::default();
    for bb_data in body.basic_blocks.iter() {
        let terminator = bb_data.terminator();
        if let TerminatorKind::Call { func, args: call_args, .. } = &terminator.kind {
            let ty = func.ty(&body.local_decls, tcx);
            let ty::FnDef(def_id, generic_args) = ty.kind() else {
                continue;
            };
            let call = if tcx.is_intrinsic(*def_id, sym::const_eval_select) {
                let func = &call_args[2].node;
                let ty = func.ty(&body.local_decls, tcx);
                let ty::FnDef(def_id, generic_args) = ty.kind() else {
                    continue;
                };
                (*def_id, *generic_args)
            } else {
                (*def_id, *generic_args)
            };
            calls.insert(call);
        }
    }
    tcx.arena.alloc_from_iter(calls.iter().copied())
}
