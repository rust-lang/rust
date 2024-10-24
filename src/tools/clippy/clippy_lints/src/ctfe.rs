use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl};
use rustc_lint::Level::Deny;
use rustc_lint::{LateContext, LateLintPass, Lint};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

/// Ensures that Constant-time Function Evaluation is being done (specifically, MIR lint passes).
/// As Clippy deactivates codegen, this lint ensures that CTFE (used in hard errors) is still ran.
pub static CLIPPY_CTFE: &Lint = &Lint {
    name: &"clippy::CLIPPY_CTFE",
    default_level: Deny,
    desc: "Ensure CTFE is being made",
    edition_lint_opts: None,
    report_in_external_macro: true,
    future_incompatible: None,
    is_externally_loaded: true,
    crate_level_only: false,
    eval_always: true,
    ..Lint::default_fields_for_macro()
};

// No static CLIPPY_CTFE_INFO because we want this lint to be invisible

declare_lint_pass! { ClippyCtfe => [CLIPPY_CTFE] }

impl<'tcx> LateLintPass<'tcx> for ClippyCtfe {
    fn check_fn(
        &mut self,
        cx: &LateContext<'_>,
        _: FnKind<'tcx>,
        _: &'tcx FnDecl<'tcx>,
        _: &'tcx Body<'tcx>,
        _: Span,
        defid: LocalDefId,
    ) {
        cx.tcx.ensure().mir_drops_elaborated_and_const_checked(defid); // Lint
    }
}
