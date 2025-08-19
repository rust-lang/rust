use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::VecArgs;
use clippy_utils::macros::root_macro_call_first_node;
use clippy_utils::source::SpanRangeExt;
use clippy_utils::ty::implements_trait;
use clippy_utils::{is_no_std_crate, sym};
use rustc_ast::{LitIntType, LitKind, UintTy};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem, QPath, StructTailExpr};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use std::fmt::{self, Display, Formatter};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `Vec` or array initializations that contain only one range.
    ///
    /// ### Why is this bad?
    /// This is almost always incorrect, as it will result in a `Vec` that has only one element.
    /// Almost always, the programmer intended for it to include all elements in the range or for
    /// the end of the range to be the length instead.
    ///
    /// ### Example
    /// ```no_run
    /// let x = [0..200];
    /// ```
    /// Use instead:
    /// ```no_run
    /// // If it was intended to include every element in the range...
    /// let x = (0..200).collect::<Vec<i32>>();
    /// // ...Or if 200 was meant to be the len
    /// let x = [0; 200];
    /// ```
    #[clippy::version = "1.72.0"]
    pub SINGLE_RANGE_IN_VEC_INIT,
    suspicious,
    "checks for initialization of `Vec` or arrays which consist of a single range"
}
declare_lint_pass!(SingleRangeInVecInit => [SINGLE_RANGE_IN_VEC_INIT]);

enum SuggestedType {
    Vec,
    Array,
}

impl SuggestedType {
    fn starts_with(&self) -> &'static str {
        if matches!(self, SuggestedType::Vec) {
            "vec!"
        } else {
            "["
        }
    }

    fn ends_with(&self) -> &'static str {
        if matches!(self, SuggestedType::Vec) { "" } else { "]" }
    }
}

impl Display for SuggestedType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if matches!(&self, SuggestedType::Vec) {
            write!(f, "a `Vec`")
        } else {
            write!(f, "an array")
        }
    }
}

impl LateLintPass<'_> for SingleRangeInVecInit {
    fn check_expr<'tcx>(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        // inner_expr: `vec![0..200]` or `[0..200]`
        //                   ^^^^^^       ^^^^^^^
        // span: `vec![0..200]` or `[0..200]`
        //        ^^^^^^^^^^^^      ^^^^^^^^
        // suggested_type: What to print, "an array" or "a `Vec`"
        let (inner_expr, span, suggested_type) = if let ExprKind::Array([inner_expr]) = expr.kind
            && !expr.span.from_expansion()
        {
            (inner_expr, expr.span, SuggestedType::Array)
        } else if let Some(macro_call) = root_macro_call_first_node(cx, expr)
            && let Some(VecArgs::Vec([expr])) = VecArgs::hir(cx, expr)
        {
            (expr, macro_call.span, SuggestedType::Vec)
        } else {
            return;
        };

        let ExprKind::Struct(QPath::LangItem(lang_item, ..), [start, end], StructTailExpr::None) = inner_expr.kind
        else {
            return;
        };

        if matches!(lang_item, LangItem::Range)
            && let ty = cx.typeck_results().expr_ty(start.expr)
            && let Some(snippet) = span.get_source_text(cx)
            // `is_from_proc_macro` will skip any `vec![]`. Let's not!
            && snippet.starts_with(suggested_type.starts_with())
            && snippet.ends_with(suggested_type.ends_with())
            && let Some(start_snippet) = start.span.get_source_text(cx)
            && let Some(end_snippet) = end.span.get_source_text(cx)
        {
            let should_emit_every_value = if let Some(step_def_id) = cx.tcx.get_diagnostic_item(sym::range_step)
                && implements_trait(cx, ty, step_def_id, &[])
            {
                true
            } else {
                false
            };
            let should_emit_of_len = if let Some(copy_def_id) = cx.tcx.lang_items().copy_trait()
                && implements_trait(cx, ty, copy_def_id, &[])
                && let ExprKind::Lit(lit_kind) = end.expr.kind
                && let LitKind::Int(.., suffix_type) = lit_kind.node
                && let LitIntType::Unsigned(UintTy::Usize) | LitIntType::Unsuffixed = suffix_type
            {
                true
            } else {
                false
            };

            if should_emit_every_value || should_emit_of_len {
                span_lint_and_then(
                    cx,
                    SINGLE_RANGE_IN_VEC_INIT,
                    span,
                    format!("{suggested_type} of `Range` that is only one element"),
                    |diag| {
                        if should_emit_every_value && !is_no_std_crate(cx) {
                            diag.span_suggestion(
                                span,
                                "if you wanted a `Vec` that contains the entire range, try",
                                format!("({start_snippet}..{end_snippet}).collect::<std::vec::Vec<{ty}>>()"),
                                Applicability::MaybeIncorrect,
                            );
                        }

                        if should_emit_of_len {
                            diag.span_suggestion(
                                inner_expr.span,
                                format!("if you wanted {suggested_type} of len {end_snippet}, try"),
                                format!("{start_snippet}; {end_snippet}"),
                                Applicability::MaybeIncorrect,
                            );
                        }
                    },
                );
            }
        }
    }
}
