use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_data_structures::unord::UnordSet;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::mir::TerminatorKind;
use rustc_middle::ty::{self, GenericArgsRef, InstanceKind, TyCtxt, TypeVisitableExt};
use rustc_session::Limit;
use rustc_span::sym;
use tracing::{instrument, trace};

#[instrument(level = "debug", skip(tcx), ret)]
fn should_recurse<'tcx>(tcx: TyCtxt<'tcx>, callee: ty::Instance<'tcx>) -> bool {
    match callee.def {
        // If there is no MIR available (either because it was not in metadata or
        // because it has no MIR because it's an extern function), then the inliner
        // won't cause cycles on this.
        InstanceKind::Item(_) => {
            if !tcx.is_mir_available(callee.def_id()) {
                return false;
            }
        }

        // These have no own callable MIR.
        InstanceKind::Intrinsic(_) | InstanceKind::Virtual(..) => return false,

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
        InstanceKind::FnPtrAddrShim(..) => return false,

        // FIXME: A not fully instantiated drop shim can cause ICEs if one attempts to
        // have its MIR built. Likely oli-obk just screwed up the `ParamEnv`s, so this
        // needs some more analysis.
        InstanceKind::DropGlue(..)
        | InstanceKind::FutureDropPollShim(..)
        | InstanceKind::AsyncDropGlue(..)
        | InstanceKind::AsyncDropGlueCtorShim(..) => {
            if callee.has_param() {
                return false;
            }
        }
    }

    crate::pm::should_run_pass(tcx, &crate::inline::Inline, crate::pm::Optimizations::Allowed)
        || crate::inline::ForceInline::should_run_pass_for_callee(tcx, callee.def.def_id())
}

#[instrument(
    level = "debug",
    skip(tcx, typing_env, seen, involved, recursion_limiter, recursion_limit),
    ret
)]
fn process<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    caller: ty::Instance<'tcx>,
    target: LocalDefId,
    seen: &mut FxHashMap<ty::Instance<'tcx>, bool>,
    involved: &mut FxHashSet<LocalDefId>,
    recursion_limiter: &mut FxHashMap<DefId, usize>,
    recursion_limit: Limit,
) -> bool {
    trace!(%caller);
    let mut reaches_root = false;

    for &(callee_def_id, args) in tcx.mir_inliner_callees(caller.def) {
        let Ok(args) = caller.try_instantiate_mir_and_normalize_erasing_regions(
            tcx,
            typing_env,
            ty::EarlyBinder::bind(args),
        ) else {
            trace!(?caller, ?typing_env, ?args, "cannot normalize, skipping");
            continue;
        };
        let Ok(Some(callee)) = ty::Instance::try_resolve(tcx, typing_env, callee_def_id, args)
        else {
            trace!(?callee_def_id, "cannot resolve, skipping");
            continue;
        };

        // Found a path.
        if callee.def_id() == target.to_def_id() {
            reaches_root = true;
            seen.insert(callee, true);
            continue;
        }

        if tcx.is_constructor(callee.def_id()) {
            trace!("constructors always have MIR");
            // Constructor functions cannot cause a query cycle.
            continue;
        }

        if !should_recurse(tcx, callee) {
            continue;
        }

        let callee_reaches_root = if let Some(&c) = seen.get(&callee) {
            // Even if we have seen this callee before, and thus don't need
            // to recurse into it, we still need to propagate whether it reaches
            // the root so that we can mark all the involved callers, in case we
            // end up reaching that same recursive callee through some *other* cycle.
            c
        } else {
            seen.insert(callee, false);
            let recursion = recursion_limiter.entry(callee.def_id()).or_default();
            trace!(?callee, recursion = *recursion);
            let callee_reaches_root = if recursion_limit.value_within_limit(*recursion) {
                *recursion += 1;
                ensure_sufficient_stack(|| {
                    process(
                        tcx,
                        typing_env,
                        callee,
                        target,
                        seen,
                        involved,
                        recursion_limiter,
                        recursion_limit,
                    )
                })
            } else {
                // Pessimistically assume that there could be recursion.
                true
            };
            seen.insert(callee, callee_reaches_root);
            callee_reaches_root
        };
        if callee_reaches_root {
            if let Some(callee_def_id) = callee.def_id().as_local() {
                // Calling `optimized_mir` of a non-local definition cannot cycle.
                involved.insert(callee_def_id);
            }
            reaches_root = true;
        }
    }

    reaches_root
}

#[instrument(level = "debug", skip(tcx), ret)]
pub(crate) fn mir_callgraph_cyclic<'tcx>(
    tcx: TyCtxt<'tcx>,
    root: LocalDefId,
) -> UnordSet<LocalDefId> {
    assert!(
        !tcx.is_constructor(root.to_def_id()),
        "you should not call `mir_callgraph_reachable` on enum/struct constructor functions"
    );

    // FIXME(-Znext-solver=no): Remove this hack when trait solver overflow can return an error.
    // In code like that pointed out in #128887, the type complexity we ask the solver to deal with
    // grows as we recurse into the call graph. If we use the same recursion limit here and in the
    // solver, the solver hits the limit first and emits a fatal error. But if we use a reduced
    // limit, we will hit the limit first and give up on looking for inlining. And in any case,
    // the default recursion limits are quite generous for us. If we need to recurse 64 times
    // into the call graph, we're probably not going to find any useful MIR inlining.
    let recursion_limit = tcx.recursion_limit() / 2;
    let mut involved = FxHashSet::default();
    let typing_env = ty::TypingEnv::post_analysis(tcx, root);
    let root_instance =
        ty::Instance::new_raw(root.to_def_id(), ty::GenericArgs::identity_for_item(tcx, root));
    if !should_recurse(tcx, root_instance) {
        trace!("cannot walk, skipping");
        return involved.into();
    }
    process(
        tcx,
        typing_env,
        root_instance,
        root,
        &mut FxHashMap::default(),
        &mut involved,
        &mut FxHashMap::default(),
        recursion_limit,
    );
    involved.into()
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
