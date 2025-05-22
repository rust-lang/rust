use crate::internal_paths;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::{self, EarlyBinder, GenericArgKind};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_tool_lint! {
    /// ### What it does
    /// Check that the `extract_msrv_attr!` macro is used, when a lint has a MSRV.
    pub clippy::MISSING_MSRV_ATTR_IMPL,
    Warn,
    "checking if all necessary steps were taken when adding a MSRV to a lint",
    report_in_external_macro: true
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
            && internal_paths::EARLY_LINT_PASS.matches(cx, trait_ref.def_id)
            && let ty::Adt(self_ty_def, _) = trait_ref.self_ty().kind()
            && self_ty_def.is_struct()
            && self_ty_def.all_fields().any(|f| {
                cx.tcx
                    .type_of(f.did)
                    .instantiate_identity()
                    .walk()
                    .filter(|t| matches!(t.unpack(), GenericArgKind::Type(_)))
                    .any(|t| internal_paths::MSRV_STACK.matches_ty(cx, t.expect_ty()))
            })
            && !items.iter().any(|item| item.ident.name.as_str() == "check_attributes")
        {
            let span = cx.sess().source_map().span_through_char(item.span, '{');
            span_lint_and_sugg(
                cx,
                MISSING_MSRV_ATTR_IMPL,
                span,
                "`extract_msrv_attr!` macro missing from `EarlyLintPass` implementation",
                "add `extract_msrv_attr!()` to the `EarlyLintPass` implementation",
                format!("{}\n    extract_msrv_attr!();", snippet(cx, span, "..")),
                Applicability::MachineApplicable,
            );
        }
    }
}
