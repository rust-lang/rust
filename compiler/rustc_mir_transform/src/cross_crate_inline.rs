use rustc_attr::InlineAttr;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::LangItem;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::InliningThreshold;
use rustc_span::sym;

pub fn provide(providers: &mut Providers) {
    providers.cross_crate_inlinable = cross_crate_inlinable;
}

fn cross_crate_inlinable(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    // Bail quickly for DefIds that wouldn't be inlinable anyway, such as statics.
    match tcx.def_kind(def_id) {
        DefKind::Ctor(..) => return true,
        DefKind::Fn | DefKind::AssocFn | DefKind::Closure => {}
        _ => return false,
    }

    // Drop glue is special-cased to be always inlinable.
    if tcx.is_lang_item(def_id.into(), LangItem::DropInPlace)
        || tcx.is_lang_item(def_id.into(), LangItem::AsyncDropInPlace)
    {
        return true;
    }

    // The program entrypoint is never inlinable in any sense.
    if let Some((entry_fn, _)) = tcx.entry_fn(()) {
        if def_id == entry_fn.expect_local() {
            return false;
        }
    }

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
        InlineAttr::None => {}
    }

    // From this point on, it is technically valid to return true or false.
    let threshold = match tcx.sess.opts.unstable_opts.cross_crate_inline_threshold {
        InliningThreshold::Always => return true,
        InliningThreshold::Sometimes(threshold) => threshold,
        InliningThreshold::Never => return false,
    };

    if tcx.has_attr(def_id, sym::rustc_intrinsic) {
        // Intrinsic fallback bodies are always cross-crate inlineable.
        // To ensure that the MIR inliner doesn't cluelessly try to inline fallback
        // bodies even when the backend would implement something better, we stop
        // the MIR inliner from ever inlining an intrinsic.
        return true;
    }

    // If there is no MIR for the DefId, we can't analyze the body. But also, this only arises in
    // two relevant cases: extern functions and MIR shims. So here we recognize the MIR shims by a
    // DefId that has no MIR and whose parent is one of the shimmed traits.
    // Everything else is extern functions, and thus not a candidate for inlining.
    if !tcx.is_mir_available(def_id) {
        let parent = tcx.parent(def_id.into());
        match tcx.lang_items().from_def_id(parent.into()) {
            Some(LangItem::Clone | LangItem::FnOnce | LangItem::Fn | LangItem::FnMut) => {
                return true;
            }
            _ => return false,
        }
    }

    if tcx.sess.opts.incremental.is_some() {
        return false;
    }

    if matches!(tcx.sess.opts.optimize, rustc_session::config::OptLevel::No) {
        return false;
    }

    let mir = tcx.optimized_mir(def_id);
    let mut checker =
        CostChecker { tcx, callee_body: mir, calls: 0, statements: 0, landing_pads: 0, resumes: 0 };
    checker.visit_body(mir);

    checker.calls == 0
        && checker.resumes == 0
        && checker.landing_pads == 0
        && checker.statements <= threshold
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
