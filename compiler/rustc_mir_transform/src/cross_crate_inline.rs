use rustc_attr::InlineAttr;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::OptLevel;

pub fn provide(providers: &mut Providers) {
    providers.cross_crate_inlinable = cross_crate_inlinable;
}

fn cross_crate_inlinable(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    let codegen_fn_attrs = tcx.codegen_fn_attrs(def_id);
    // If this has an extern indicator, then this function is globally shared and thus will not
    // generate cgu-internal copies which would make it cross-crate inlinable.
    if codegen_fn_attrs.contains_extern_indicator() {
        return false;
    }

    // Obey source annotations first; this is important because it means we can use
    // #[inline(never)] to force code generation.
    match codegen_fn_attrs.inline {
        InlineAttr::Never => return false,
        InlineAttr::Hint | InlineAttr::Always => return true,
        _ => {}
    }

    // This just reproduces the logic from Instance::requires_inline.
    match tcx.def_kind(def_id) {
        DefKind::Ctor(..) | DefKind::Closure => return true,
        DefKind::Fn | DefKind::AssocFn => {}
        _ => return false,
    }

    // Don't do any inference when incremental compilation is enabled; the additional inlining that
    // inference permits also creates more work for small edits.
    if tcx.sess.opts.incremental.is_some() {
        return false;
    }

    // Don't do any inference unless optimizations are enabled.
    if matches!(tcx.sess.opts.optimize, OptLevel::No) {
        return false;
    }

    if !tcx.is_mir_available(def_id) {
        return false;
    }

    let mir = tcx.optimized_mir(def_id);
    let mut checker =
        CostChecker { tcx, callee_body: mir, calls: 0, statements: 0, landing_pads: 0, resumes: 0 };
    checker.visit_body(mir);
    checker.calls == 0
        && checker.resumes == 0
        && checker.landing_pads == 0
        && checker.statements
            <= tcx.sess.opts.unstable_opts.cross_crate_inline_threshold.unwrap_or(100)
}

struct CostChecker<'b, 'tcx> {
    tcx: TyCtxt<'tcx>,
    callee_body: &'b Body<'tcx>,
    calls: usize,
    statements: usize,
    landing_pads: usize,
    resumes: usize,
}

impl<'tcx> Visitor<'tcx> for CostChecker<'_, 'tcx> {
    fn visit_statement(&mut self, statement: &Statement<'tcx>, _: Location) {
        // Don't count StorageLive/StorageDead in the inlining cost.
        match statement.kind {
            StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Deinit(_)
            | StatementKind::Nop => {}
            _ => self.statements += 1,
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, _: Location) {
        let tcx = self.tcx;
        match terminator.kind {
            TerminatorKind::Drop { ref place, unwind, .. } => {
                let ty = place.ty(self.callee_body, tcx).ty;
                if !ty.is_trivially_pure_clone_copy() {
                    self.calls += 1;
                    if let UnwindAction::Cleanup(_) = unwind {
                        self.landing_pads += 1;
                    }
                }
            }
            TerminatorKind::Call { unwind, .. } => {
                self.calls += 1;
                if let UnwindAction::Cleanup(_) = unwind {
                    self.landing_pads += 1;
                }
            }
            TerminatorKind::Assert { unwind, .. } => {
                self.calls += 1;
                if let UnwindAction::Cleanup(_) = unwind {
                    self.landing_pads += 1;
                }
            }
            TerminatorKind::UnwindResume => self.resumes += 1,
            TerminatorKind::InlineAsm { unwind, .. } => {
                self.statements += 1;
                if let UnwindAction::Cleanup(_) = unwind {
                    self.landing_pads += 1;
                }
            }
            TerminatorKind::Return => {}
            _ => self.statements += 1,
        }
    }
}
