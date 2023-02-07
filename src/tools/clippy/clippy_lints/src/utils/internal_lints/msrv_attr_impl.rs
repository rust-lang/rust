use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::match_type;
use clippy_utils::{match_def_path, paths};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir_analysis::hir_ty_to_ty;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::{self, subst::GenericArgKind};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Check that the `extract_msrv_attr!` macro is used, when a lint has a MSRV.
    ///
    pub MISSING_MSRV_ATTR_IMPL,
    internal,
    "checking if all necessary steps were taken when adding a MSRV to a lint"
}

declare_lint_pass!(MsrvAttrImpl => [MISSING_MSRV_ATTR_IMPL]);

impl LateLintPass<'_> for MsrvAttrImpl {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &hir::Item<'_>) {
        if_chain! {
            if let hir::ItemKind::Impl(hir::Impl {
                of_trait: Some(lint_pass_trait_ref),
                self_ty,
                items,
                ..
            }) = &item.kind;
            if let Some(lint_pass_trait_def_id) = lint_pass_trait_ref.trait_def_id();
            let is_late_pass = match_def_path(cx, lint_pass_trait_def_id, &paths::LATE_LINT_PASS);
            if is_late_pass || match_def_path(cx, lint_pass_trait_def_id, &paths::EARLY_LINT_PASS);
            let self_ty = hir_ty_to_ty(cx.tcx, self_ty);
            if let ty::Adt(self_ty_def, _) = self_ty.kind();
            if self_ty_def.is_struct();
            if self_ty_def.all_fields().any(|f| {
                cx.tcx
                    .type_of(f.did)
                    .subst_identity()
                    .walk()
                    .filter(|t| matches!(t.unpack(), GenericArgKind::Type(_)))
                    .any(|t| match_type(cx, t.expect_ty(), &paths::MSRV))
            });
            if !items.iter().any(|item| item.ident.name == sym!(enter_lint_attrs));
            then {
                let context = if is_late_pass { "LateContext" } else { "EarlyContext" };
                let lint_pass = if is_late_pass { "LateLintPass" } else { "EarlyLintPass" };
                let span = cx.sess().source_map().span_through_char(item.span, '{');
                span_lint_and_sugg(
                    cx,
                    MISSING_MSRV_ATTR_IMPL,
                    span,
                    &format!("`extract_msrv_attr!` macro missing from `{lint_pass}` implementation"),
                    &format!("add `extract_msrv_attr!({context})` to the `{lint_pass}` implementation"),
                    format!("{}\n    extract_msrv_attr!({context});", snippet(cx, span, "..")),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}
