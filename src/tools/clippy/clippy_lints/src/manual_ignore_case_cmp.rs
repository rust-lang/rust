use crate::manual_ignore_case_cmp::MatchType::{Literal, ToAscii};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sym;
use clippy_utils::ty::{get_type_diagnostic_name, is_type_diagnostic_item, is_type_lang_item};
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::ExprKind::{Binary, Lit, MethodCall};
use rustc_hir::{BinOpKind, Expr, LangItem};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_middle::ty::{Ty, UintTy};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual case-insensitive ASCII comparison.
    ///
    /// ### Why is this bad?
    /// The `eq_ignore_ascii_case` method is faster because it does not allocate
    /// memory for the new strings, and it is more readable.
    ///
    /// ### Example
    /// ```no_run
    /// fn compare(a: &str, b: &str) -> bool {
    ///     a.to_ascii_lowercase() == b.to_ascii_lowercase() || a.to_ascii_lowercase() == "abc"
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn compare(a: &str, b: &str) -> bool {
    ///     a.eq_ignore_ascii_case(b) || a.eq_ignore_ascii_case("abc")
    /// }
    /// ```
    #[clippy::version = "1.84.0"]
    pub MANUAL_IGNORE_CASE_CMP,
    perf,
    "manual case-insensitive ASCII comparison"
}

declare_lint_pass!(ManualIgnoreCaseCmp => [MANUAL_IGNORE_CASE_CMP]);

enum MatchType<'a> {
    ToAscii(bool, Ty<'a>),
    Literal(LitKind),
}

fn get_ascii_type<'a>(cx: &LateContext<'a>, kind: rustc_hir::ExprKind<'_>) -> Option<(Span, MatchType<'a>)> {
    if let MethodCall(path, expr, _, _) = kind {
        let is_lower = match path.ident.name {
            sym::to_ascii_lowercase => true,
            sym::to_ascii_uppercase => false,
            _ => return None,
        };
        let ty_raw = cx.typeck_results().expr_ty(expr);
        let ty = ty_raw.peel_refs();
        if needs_ref_to_cmp(cx, ty)
            || ty.is_str()
            || ty.is_slice()
            || matches!(get_type_diagnostic_name(cx, ty), Some(sym::OsStr | sym::OsString))
        {
            return Some((expr.span, ToAscii(is_lower, ty_raw)));
        }
    } else if let Lit(expr) = kind {
        return Some((expr.span, Literal(expr.node)));
    }
    None
}

/// Returns true if the type needs to be dereferenced to be compared
fn needs_ref_to_cmp(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
    ty.is_char()
        || *ty.kind() == ty::Uint(UintTy::U8)
        || is_type_diagnostic_item(cx, ty, sym::Vec)
        || is_type_lang_item(cx, ty, LangItem::String)
}

impl LateLintPass<'_> for ManualIgnoreCaseCmp {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'_ Expr<'_>) {
        // check if expression represents a comparison of two strings
        // using .to_ascii_lowercase() or .to_ascii_uppercase() methods,
        // or one of the sides is a literal
        // Offer to replace it with .eq_ignore_ascii_case() method
        if let Binary(op, left, right) = &expr.kind
            && (op.node == BinOpKind::Eq || op.node == BinOpKind::Ne)
            && let Some((left_span, left_val)) = get_ascii_type(cx, left.kind)
            && let Some((right_span, right_val)) = get_ascii_type(cx, right.kind)
            && match (&left_val, &right_val) {
                (ToAscii(l_lower, ..), ToAscii(r_lower, ..)) if l_lower == r_lower => true,
                (ToAscii(..), Literal(..)) | (Literal(..), ToAscii(..)) => true,
                _ => false,
            }
        {
            let deref = match right_val {
                ToAscii(_, ty) if needs_ref_to_cmp(cx, ty) => "&",
                ToAscii(..) => "",
                Literal(ty) => {
                    if let LitKind::Char(_) | LitKind::Byte(_) = ty {
                        "&"
                    } else {
                        ""
                    }
                },
            };
            let neg = if op.node == BinOpKind::Ne { "!" } else { "" };
            span_lint_and_then(
                cx,
                MANUAL_IGNORE_CASE_CMP,
                expr.span,
                "manual case-insensitive ASCII comparison",
                |diag| {
                    let mut app = Applicability::MachineApplicable;
                    diag.span_suggestion_verbose(
                        expr.span,
                        "consider using `.eq_ignore_ascii_case()` instead",
                        format!(
                            "{neg}{}.eq_ignore_ascii_case({deref}{})",
                            snippet_with_applicability(cx, left_span, "_", &mut app),
                            snippet_with_applicability(cx, right_span, "_", &mut app)
                        ),
                        app,
                    );
                },
            );
        }
    }
}
