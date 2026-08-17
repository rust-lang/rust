use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_from_proc_macro;
use rustc_errors::Applicability;
use rustc_lint::LateLintPass;
use rustc_middle::ty;
use rustc_session::declare_lint_pass;

use std::fmt::Write as _;

declare_clippy_lint! {
    /// ### What it does
    /// Disallow the use of rest patterns for accessible fields
    ///
    /// ### Why restrict this?
    /// It might lead to unhandled fields when the struct changes.
    ///
    /// ### Example
    /// ```no_run
    /// struct S {
    ///     a: u8,
    ///     b: u8,
    ///     c: u8,
    /// }
    ///
    /// let s = S { a: 1, b: 2, c: 3 };
    ///
    /// let S { a, b, .. } = s;
    /// ```
    /// Use instead:
    /// ```no_run
    /// struct S {
    ///     a: u8,
    ///     b: u8,
    ///     c: u8,
    /// }
    ///
    /// let s = S { a: 1, b: 2, c: 3 };
    ///
    /// let S { a, b, c: _ } = s;
    /// ```
    #[clippy::version = "1.99.0"]
    pub REST_PATTERN_ACCESSIBLE_FIELD,
    restriction,
    "rest pattern (`..`) used for accessible field"
}

declare_clippy_lint! {
    /// ### What it does
    /// Disallow the use of rest patterns that don't capture any fields.
    ///
    /// ### Why restrict this?
    /// It might lead to unhandled fields when the struct changes.
    ///
    /// ### Example
    /// ```no_run
    /// struct S {
    ///     a: u8,
    ///     b: u8,
    ///     c: u8,
    /// }
    ///
    /// let s = S { a: 1, b: 2, c: 3 };
    ///
    /// let S { a, b, c, .. } = s;
    /// ```
    /// Use instead:
    /// ```no_run
    /// struct S {
    ///     a: u8,
    ///     b: u8,
    ///     c: u8,
    /// }
    ///
    /// let s = S { a: 1, b: 2, c: 3 };
    ///
    /// let S { a, b, c } = s;
    /// ```
    #[clippy::version = "1.99.0"]
    pub UNNECESSARY_REST_PATTERN,
    restriction,
    "unnecessary rest pattern (`..`) in destructuring expression"
}

declare_lint_pass!(RestWhenDestructuringStruct => [
    REST_PATTERN_ACCESSIBLE_FIELD,
    UNNECESSARY_REST_PATTERN,
]);

impl<'tcx> LateLintPass<'tcx> for RestWhenDestructuringStruct {
    fn check_pat(&mut self, cx: &rustc_lint::LateContext<'tcx>, pat: &'tcx rustc_hir::Pat<'tcx>) {
        if let rustc_hir::PatKind::Struct(path, fields, Some(dotdot)) = pat.kind
            && let qty = cx.typeck_results().qpath_res(&path, pat.hir_id)
            && let ty = cx.typeck_results().pat_ty(pat)
            && let ty::Adt(a, _) = ty.kind()
            && let Some(vid) = qty.opt_def_id().map(|x| a.variant_index_with_id(x))
            && let Some(variant) = a.variants().get(vid)
        {
            let mut missing_suggestions = String::new();
            let mut needs_dotdot = variant.field_list_has_applicable_non_exhaustive();

            for field in &variant.fields {
                if field.vis.is_accessible_from(cx.tcx.parent_module(pat.hir_id), cx.tcx) {
                    if !fields.iter().any(|x| x.ident.name == field.name) {
                        if !missing_suggestions.is_empty() {
                            missing_suggestions.push_str(", ");
                        }
                        let _ = write!(missing_suggestions, "{}: _", field.name.as_str());
                    }
                } else {
                    needs_dotdot = true;
                }
            }

            // Filter out results from macros
            if (missing_suggestions.is_empty() && needs_dotdot)
                || pat.span.in_external_macro(cx.tcx.sess.source_map())
                || is_from_proc_macro(cx, pat)
            {
                return;
            }

            if !missing_suggestions.is_empty() {
                let suggestion_span = if needs_dotdot {
                    missing_suggestions.push_str(", ");
                    dotdot.shrink_to_lo()
                } else {
                    dotdot
                };

                let message = if fields.is_empty() {
                    "consider explicitly ignoring fields with wildcard patterns (`x: _`)"
                } else {
                    "consider explicitly ignoring remaining fields with wildcard patterns (`x: _`)"
                };

                span_lint_and_then(
                    cx,
                    REST_PATTERN_ACCESSIBLE_FIELD,
                    pat.span,
                    "struct destructuring with rest (`..`)",
                    |diag| {
                        diag.span_suggestion_verbose(
                            suggestion_span,
                            message,
                            &missing_suggestions,
                            Applicability::MachineApplicable,
                        );
                    },
                );
            } else if !needs_dotdot {
                let message = "consider removing the unnecessary rest pattern (`..`)";
                span_lint_and_then(
                    cx,
                    UNNECESSARY_REST_PATTERN,
                    pat.span,
                    "unnecessary rest pattern (`..`)",
                    |diag| {
                        diag.span_suggestion_verbose(dotdot, message, "", Applicability::MachineApplicable);
                    },
                );
            }
        }
    }
}
