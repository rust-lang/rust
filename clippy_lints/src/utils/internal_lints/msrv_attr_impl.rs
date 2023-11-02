use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::match_type;
use clippy_utils::{match_def_path, paths};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::{self, EarlyBinder, GenericArgKind};
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
        if let hir::ItemKind::Impl(hir::Impl {
            of_trait: Some(_),
            items,
            ..
        }) = &item.kind
            && let Some(trait_ref) = cx
                .tcx
                .impl_trait_ref(item.owner_id)
                .map(EarlyBinder::instantiate_identity)
            && let is_late_pass = match_def_path(cx, trait_ref.def_id, &paths::LATE_LINT_PASS)
            && (is_late_pass || match_def_path(cx, trait_ref.def_id, &paths::EARLY_LINT_PASS))
            && let ty::Adt(self_ty_def, _) = trait_ref.self_ty().kind()
            && self_ty_def.is_struct()
            && self_ty_def.all_fields().any(|f| {
                cx.tcx
                    .type_of(f.did)
                    .instantiate_identity()
                    .walk()
                    .filter(|t| matches!(t.unpack(), GenericArgKind::Type(_)))
                    .any(|t| match_type(cx, t.expect_ty(), &paths::MSRV))
            })
            && !items.iter().any(|item| item.ident.name == sym!(enter_lint_attrs))
        {
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
