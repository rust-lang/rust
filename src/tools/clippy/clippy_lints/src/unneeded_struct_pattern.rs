use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_from_proc_macro;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Pat, PatKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for struct patterns that match against unit variant.
    ///
    /// ### Why is this bad?
    /// Struct pattern `{ }` or `{ .. }` is not needed for unit variant.
    ///
    /// ### Example
    /// ```no_run
    /// match Some(42) {
    ///     Some(v) => v,
    ///     None { .. } => 0,
    /// };
    /// // Or
    /// match Some(42) {
    ///     Some(v) => v,
    ///     None { } => 0,
    /// };
    /// ```
    /// Use instead:
    /// ```no_run
    /// match Some(42) {
    ///     Some(v) => v,
    ///     None => 0,
    /// };
    /// ```
    #[clippy::version = "1.86.0"]
    pub UNNEEDED_STRUCT_PATTERN,
    style,
    "using struct pattern to match against unit variant"
}

declare_lint_pass!(UnneededStructPattern => [UNNEEDED_STRUCT_PATTERN]);

impl LateLintPass<'_> for UnneededStructPattern {
    fn check_pat(&mut self, cx: &LateContext<'_>, pat: &Pat<'_>) {
        if !pat.span.from_expansion()
            && let PatKind::Struct(path, [], _) = &pat.kind
            && let QPath::Resolved(_, path) = path
            && let Res::Def(DefKind::Variant, did) = path.res
        {
            let enum_did = cx.tcx.parent(did);
            let variant = cx.tcx.adt_def(enum_did).variant_with_id(did);

            let has_only_fields_brackets = variant.ctor.is_some() && variant.fields.is_empty();
            let non_exhaustive_activated = variant.field_list_has_applicable_non_exhaustive();
            if !has_only_fields_brackets || non_exhaustive_activated {
                return;
            }

            if is_from_proc_macro(cx, *path) {
                return;
            }

            if let Some(brackets_span) = pat.span.trim_start(path.span) {
                span_lint_and_sugg(
                    cx,
                    UNNEEDED_STRUCT_PATTERN,
                    brackets_span,
                    "struct pattern is not needed for a unit variant",
                    "remove the struct pattern",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}
