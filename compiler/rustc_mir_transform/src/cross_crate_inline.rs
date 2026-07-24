use rustc_hir::attrs::InlineAttr;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{self as hir, find_attr};
use rustc_middle::bug;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, GenericArgsRef, Instance, InstanceKind, TyCtxt, Unnormalized};
use rustc_session::config::{InliningThreshold, OptLevel};

use crate::{inline, pass_manager as pm};

pub(super) fn provide(providers: &mut Providers) {
    providers.cross_crate_inlinable = cross_crate_inlinable;
}

fn cross_crate_inlinable(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    let codegen_fn_attrs = tcx.codegen_fn_attrs(def_id);
    // If this has an extern indicator, then this function is globally shared and thus will not
    // generate cgu-internal copies which would make it cross-crate inlinable.
    if codegen_fn_attrs.contains_extern_indicator() {
        return false;
    }

    // This just reproduces the logic from Instance::requires_inline.
    match tcx.def_kind(def_id) {
        DefKind::Ctor(..) | DefKind::Closure | DefKind::SyntheticCoroutineBody => return true,
        DefKind::Fn | DefKind::AssocFn => {}
        _ => return false,
    }

    // From this point on, it is valid to return true or false.
    if tcx.sess.opts.unstable_opts.cross_crate_inline_threshold == InliningThreshold::Always {
        return true;
    }

    if find_attr!(tcx, def_id, RustcIntrinsic) {
        // Intrinsic fallback bodies are always cross-crate inlineable.
        // To ensure that the MIR inliner doesn't cluelessly try to inline fallback
        // bodies even when the backend would implement something better, we stop
        // the MIR inliner from ever inlining an intrinsic.
        return true;
    }

    if let hir::Constness::Const { always: true } = tcx.constness(def_id) {
        // Comptime functions only exist during const eval and can never be passed
        // to codegen. The const eval MIR pipeline also doesn't inline anything at all.
        return false;
    }

    // Obey source annotations first; this is important because it means we can use
    // #[inline(never)] to force code generation.
    match codegen_fn_attrs.inline {
        InlineAttr::Never => return false,
        InlineAttr::Hint | InlineAttr::Always | InlineAttr::Force { .. } => return true,
        _ => {}
    }

    // If the crate is likely to be mostly unused, use cross-crate inlining to defer codegen until
    // the function is referenced, in order to skip codegen for unused functions. This is
    // intentionally after the check for `inline(never)`, so that `inline(never)` wins.
    if tcx.sess.opts.unstable_opts.hint_mostly_unused {
        return true;
    }

    let sig = tcx.fn_sig(def_id).instantiate_identity().skip_norm_wip();
    for ty in sig.inputs().skip_binder().iter().chain(std::iter::once(&sig.output().skip_binder()))
    {
        // FIXME(f16_f128): in order to avoid crashes building `core`, always inline to skip
        // codegen if the function is not used.
        if ty == &tcx.types.f16 || ty == &tcx.types.f128 {
            return true;
        }
    }

    // Don't do any inference when incremental compilation is enabled; the additional inlining that
    // inference permits also creates more work for small edits.
    if tcx.sess.opts.incremental.is_some() {
        return false;
    }

    // Don't do any inference if codegen optimizations are disabled and also MIR inlining is not
    // enabled. This ensures that we do inference even if someone only passes -Zinline-mir,
    // which is less confusing than having to also enable -Copt-level=1.
    let inliner_will_run = pm::should_run_pass(tcx, &inline::Inline, pm::Optimizations::Allowed)
        || inline::ForceInline::should_run_pass_for_callee(tcx, def_id.to_def_id());
    if matches!(tcx.sess.opts.optimize, OptLevel::No) && !inliner_will_run {
        return false;
    }

    if !tcx.is_mir_available(def_id) {
        return false;
    }

    let threshold = match tcx.sess.opts.unstable_opts.cross_crate_inline_threshold {
        InliningThreshold::Always => return true,
        InliningThreshold::Sometimes(threshold) => threshold,
        InliningThreshold::Never => return false,
    };

    let mir = tcx.optimized_mir(def_id);
    let mut checker = CostChecker {
        tcx,
        typing_env: mir.typing_env(tcx),
        callee_body: mir,
        calls: 0,
        statements: 0,
        landing_pads: 0,
        resumes: 0,
    };
    checker.visit_body(mir);
    checker.calls == 0
        && checker.resumes == 0
        && checker.landing_pads == 0
        && checker.statements <= threshold
}

// The threshold that CostChecker computes is balancing the desire to make more things
// inlinable cross crates against the growth in incremental CGU size that happens when too many
// things in the sysroot are made inlinable.
// Permitting calls causes the size of some incremental CGUs to grow, because more functions are
// made inlinable out of the sysroot or dependencies.
// Assert terminators are similar to calls, but do not have the same impact on compile time, so
// those are just treated as statements.
// A threshold exists at all because we don't want to blindly mark a huge function as inlinable.

struct CostChecker<'b, 'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    callee_body: &'b Body<'tcx>,
    calls: usize,
    statements: usize,
    landing_pads: usize,
    resumes: usize,
}

impl<'tcx> CostChecker<'_, 'tcx> {
    fn trait_call_resolves_to_inline_item(
        &self,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
    ) -> bool {
        let tcx = self.tcx;
        if tcx.trait_of_assoc(def_id).is_none() {
            return false;
        }

        let Ok(args) =
            tcx.try_normalize_erasing_regions(self.typing_env, Unnormalized::new_wip(args))
        else {
            return false;
        };
        let Ok(Some(instance)) = Instance::try_resolve(tcx, self.typing_env, def_id, args) else {
            return false;
        };
        let InstanceKind::Item(_) = instance.def else {
            return false;
        };

        // `#[inline]` is ignored on externally exported functions.
        let codegen_fn_attrs = tcx.codegen_instance_attrs(instance.def);
        if codegen_fn_attrs.contains_extern_indicator() {
            return false;
        }

        match codegen_fn_attrs.inline {
            InlineAttr::Always | InlineAttr::Force { .. } => true,
            // At opt-level 0, an ordinary hint does not make the enclosing body
            // cross-crate inlinable, even when the MIR inliner was explicitly enabled.
            InlineAttr::Hint => !matches!(tcx.sess.opts.optimize, OptLevel::No),
            InlineAttr::None | InlineAttr::Never => false,
        }
    }
}

impl<'tcx> Visitor<'tcx> for CostChecker<'_, 'tcx> {
    fn visit_statement(&mut self, statement: &Statement<'tcx>, _: Location) {
        // Don't count StorageLive/StorageDead in the inlining cost.
        match statement.kind {
            StatementKind::StorageLive(_) | StatementKind::StorageDead(_) | StatementKind::Nop => {}
            _ => self.statements += 1,
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, _: Location) {
        self.statements += 1;
        let tcx = self.tcx;
        match &terminator.kind {
            TerminatorKind::Drop { place, unwind, .. } => {
                let ty = place.ty(self.callee_body, tcx).ty;
                if !ty.is_trivially_pure_clone_copy() {
                    self.calls += 1;
                    if let UnwindAction::Cleanup(_) = unwind {
                        self.landing_pads += 1;
                    }
                }
            }
            TerminatorKind::Call { func, unwind, .. } => {
                // We track calls because they make our function not a leaf (and in theory, the
                // number of calls indicates how likely this function is to perturb other CGUs).
                // But there are a handful of intrinsics such as raw_eq that should not block
                // cross-crate-inlining. Adding a broad exception for all intrinsics benchmarks well
                // and seems more sustainable than an ever-growing list of intrinsics to ignore.
                // Explicitly inline functions already have MIR encoded for downstream crates.
                // A statically dispatched trait call can hide an inline attribute.
                // Resolve its selected implementation before charging for the call.
                // Direct calls and inferred cross-crate inlining remain outside this exception.
                // Exposing a caller can still increase downstream work,
                // even when its callee is available.
                if let Some((fn_def_id, args)) = func.const_fn_def() {
                    if find_attr!(tcx, fn_def_id, RustcIntrinsic) {
                        return;
                    }

                    if self.trait_call_resolves_to_inline_item(fn_def_id, args) {
                        if let UnwindAction::Cleanup(_) = unwind {
                            self.landing_pads += 1;
                        }
                        return;
                    }
                }
                self.calls += 1;
                if let UnwindAction::Cleanup(_) = unwind {
                    self.landing_pads += 1;
                }
            }
            TerminatorKind::TailCall { .. } => {
                self.calls += 1;
            }
            TerminatorKind::Assert { unwind, .. } => {
                if let UnwindAction::Cleanup(_) = unwind {
                    self.landing_pads += 1;
                }
            }
            TerminatorKind::UnwindResume => self.resumes += 1,
            TerminatorKind::InlineAsm { unwind, .. } => {
                if let UnwindAction::Cleanup(_) = unwind {
                    self.landing_pads += 1;
                }
            }
            TerminatorKind::Return
            | TerminatorKind::Goto { .. }
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Unreachable
            | TerminatorKind::UnwindTerminate(_) => {}
            kind @ (TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::Yield { .. }
            | TerminatorKind::CoroutineDrop) => {
                bug!("{kind:?} should not be in runtime MIR");
            }
        }
    }
}
