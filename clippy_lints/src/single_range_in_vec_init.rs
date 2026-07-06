use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::{Range, VecArgs};
use clippy_utils::macros::root_macro_call_first_node;
use clippy_utils::source::{SpanExt as _, snippet_with_context};
use clippy_utils::ty::implements_trait;
use clippy_utils::{is_no_std_crate, sym};
use rustc_ast::{LitIntType, LitKind, RangeLimits, UintTy};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
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
    #[expect(clippy::too_many_lines)]
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

        let Some(range) = Range::hir(cx, inner_expr) else {
            return;
        };

        let Some(snippet) = span.get_text(cx) else {
            return;
        };
        // `is_from_proc_macro` will skip any `vec![]`. Let's not!
        if !snippet.starts_with(suggested_type.starts_with()) || !snippet.ends_with(suggested_type.ends_with()) {
            return;
        }

        let mut applicability = Applicability::MaybeIncorrect;
        let suggestion = match (range.start, range.end, range.ty.limits()) {
            (Some(start), Some(end), limits) => {
                let element_ty = cx.typeck_results().expr_ty(start);
                let (start_snippet, _) = snippet_with_context(cx, start.span, span.ctxt(), "..", &mut applicability);
                let (end_snippet, _) = snippet_with_context(cx, end.span, span.ctxt(), "..", &mut applicability);

                let should_emit_every_value = if let Some(step_def_id) = cx.tcx.get_diagnostic_item(sym::range_step)
                    && implements_trait(cx, element_ty, step_def_id, &[])
                    && range.ty.implements_into_iterator()
                {
                    Some(range.ty)
                } else {
                    None
                };
                let should_emit_of_len = if limits == RangeLimits::HalfOpen
                    && let Some(copy_def_id) = cx.tcx.lang_items().copy_trait()
                    && implements_trait(cx, element_ty, copy_def_id, &[])
                    && let ExprKind::Lit(lit_kind) = end.kind
                    && let LitKind::Int(.., suffix_type) = lit_kind.node
                    && let LitIntType::Unsigned(UintTy::Usize) | LitIntType::Unsuffixed = suffix_type
                {
                    true
                } else {
                    false
                };

                if should_emit_every_value.is_some() || should_emit_of_len {
                    Some((
                        element_ty,
                        start_snippet,
                        end_snippet,
                        should_emit_every_value,
                        should_emit_of_len,
                    ))
                } else {
                    return;
                }
            },
            (None, Some(_) | None, _) | (Some(_), None, RangeLimits::HalfOpen) => None,
            (Some(_), None, RangeLimits::Closed) => return,
        };

        span_lint_and_then(
            cx,
            SINGLE_RANGE_IN_VEC_INIT,
            span,
            format!(
                "{suggested_type} of `{}` that is only one element",
                cx.typeck_results()
                    .expr_ty(inner_expr)
                    .ty_adt_def()
                    .map_or(sym::Range, |adt_def| cx.tcx.item_name(adt_def.did()))
            ),
            |diag| {
                if let Some((ty, start_snippet, end_snippet, should_emit_every_value, should_emit_of_len)) = suggestion
                {
                    if let Some(range_ty) = should_emit_every_value
                        && !is_no_std_crate(cx)
                    {
                        let range_op = match range_ty.limits() {
                            RangeLimits::HalfOpen => "..",
                            RangeLimits::Closed => "..=",
                        };

                        let collect_code = if range_ty.implements_iterator() {
                            format!("({start_snippet}{range_op}{end_snippet}).collect::<std::vec::Vec<{ty}>>()")
                        } else {
                            // If the range type does not implement `Iterator` then we cannot just call
                            // `.collect()`. In that case, we use `from_iter()` rather than
                            // `.into_iter().collect::<...>()` because it is more concise.
                            format!("std::vec::Vec::<{ty}>::from_iter({start_snippet}{range_op}{end_snippet})")
                        };

                        diag.span_suggestion(
                            span,
                            "if you wanted a `Vec` that contains the entire range, try",
                            collect_code,
                            applicability,
                        );
                    }

                    if should_emit_of_len {
                        diag.span_suggestion(
                            inner_expr.span,
                            format!("if you wanted {suggested_type} of len {end_snippet}, try"),
                            format!("{start_snippet}; {end_snippet}"),
                            applicability,
                        );
                    }
                }
            },
        );
    }
}
