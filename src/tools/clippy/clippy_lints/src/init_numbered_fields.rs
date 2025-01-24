use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{snippet_with_applicability, snippet_with_context};
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Expr, ExprKind, StructTailExpr};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::SyntaxContext;
use std::borrow::Cow;

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
    /// ```no_run
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
        if let ExprKind::Struct(path, fields @ [field, ..], StructTailExpr::None) = e.kind
            // If the first character of any field is a digit it has to be a tuple.
            && field.ident.as_str().as_bytes().first().is_some_and(u8::is_ascii_digit)
            // Type aliases can't be used as functions.
            && !matches!(
                cx.qpath_res(path, e.hir_id),
                Res::Def(DefKind::TyAlias | DefKind::AssocTy, _)
            )
            // This is the only syntax macros can use that works for all struct types.
            && !e.span.from_expansion()
            && let mut has_side_effects = false
            && let Ok(mut expr_spans) = fields
                .iter()
                .map(|f| {
                    has_side_effects |= f.expr.can_have_side_effects();
                    f.ident.as_str().parse::<usize>().map(|x| (x, f.expr.span))
                })
                .collect::<Result<Vec<_>, _>>()
            // We can only reorder the expressions if there are no side effects.
            && (!has_side_effects || expr_spans.is_sorted_by_key(|&(idx, _)| idx))
        {
            span_lint_and_then(
                cx,
                INIT_NUMBERED_FIELDS,
                e.span,
                "used a field initializer for a tuple struct",
                |diag| {
                    if !has_side_effects {
                        // We already checked the order if there are side effects.
                        expr_spans.sort_by_key(|&(idx, _)| idx);
                    }
                    let mut app = Applicability::MachineApplicable;
                    diag.span_suggestion(
                        e.span,
                        "use tuple initialization",
                        format!(
                            "{}({})",
                            snippet_with_applicability(cx, path.span(), "..", &mut app),
                            expr_spans
                                .into_iter()
                                .map(
                                    |(_, span)| snippet_with_context(cx, span, SyntaxContext::root(), "..", &mut app).0
                                )
                                .intersperse(Cow::Borrowed(", "))
                                .collect::<String>()
                        ),
                        app,
                    );
                },
            );
        }
    }
}
