use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::in_macro;
use clippy_utils::source::snippet_with_applicability;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for tuple structs initialized with field syntax.
    /// It will however not lint if a base initializer is present.
    /// The lint will also ignore code in macros.
    ///
    /// ### Why is this bad?
    /// This may be confusing to the uninitiated and adds no
    /// benefit as opposed to tuple initializers
    ///
    /// ### Example
    /// ```rust
    /// struct TupleStruct(u8, u16);
    ///
    /// let _ = TupleStruct {
    ///     0: 1,
    ///     1: 23,
    /// };
    ///
    /// // should be written as
    /// let base = TupleStruct(1, 23);
    ///
    /// // This is OK however
    /// let _ = TupleStruct { 0: 42, ..base };
    /// ```
    #[clippy::version = "1.59.0"]
    pub INIT_NUMBERED_FIELDS,
    style,
    "numbered fields in tuple struct initializer"
}

declare_lint_pass!(NumberedFields => [INIT_NUMBERED_FIELDS]);

impl<'tcx> LateLintPass<'tcx> for NumberedFields {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::Struct(path, fields, None) = e.kind {
            if !fields.is_empty()
                && !in_macro(e.span)
                && fields
                    .iter()
                    .all(|f| f.ident.as_str().as_bytes().iter().all(u8::is_ascii_digit))
            {
                let expr_spans = fields
                    .iter()
                    .map(|f| (Reverse(f.ident.as_str().parse::<usize>().unwrap()), f.expr.span))
                    .collect::<BinaryHeap<_>>();
                let mut appl = Applicability::MachineApplicable;
                let snippet = format!(
                    "{}({})",
                    snippet_with_applicability(cx, path.span(), "..", &mut appl),
                    expr_spans
                        .into_iter_sorted()
                        .map(|(_, span)| snippet_with_applicability(cx, span, "..", &mut appl))
                        .intersperse(Cow::Borrowed(", "))
                        .collect::<String>()
                );
                span_lint_and_sugg(
                    cx,
                    INIT_NUMBERED_FIELDS,
                    e.span,
                    "used a field initializer for a tuple struct",
                    "try this instead",
                    snippet,
                    appl,
                );
            }
        }
    }
}
