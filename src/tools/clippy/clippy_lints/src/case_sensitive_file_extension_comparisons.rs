use clippy_utils::diagnostics::span_lint_and_help;
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_hir::{Expr, ExprKind, PathSegment};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{source_map::Spanned, symbol::sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `ends_with` with possible file extensions
    /// and suggests to use a case-insensitive approach instead.
    ///
    /// ### Why is this bad?
    /// `ends_with` is case-sensitive and may not detect files with a valid extension.
    ///
    /// ### Example
    /// ```rust
    /// fn is_rust_file(filename: &str) -> bool {
    ///     filename.ends_with(".rs")
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// fn is_rust_file(filename: &str) -> bool {
    ///     filename.rsplit('.').next().map(|ext| ext.eq_ignore_ascii_case("rs")) == Some(true)
    /// }
    /// ```
    #[clippy::version = "1.51.0"]
    pub CASE_SENSITIVE_FILE_EXTENSION_COMPARISONS,
    pedantic,
    "Checks for calls to ends_with with case-sensitive file extensions"
}

declare_lint_pass!(CaseSensitiveFileExtensionComparisons => [CASE_SENSITIVE_FILE_EXTENSION_COMPARISONS]);

fn check_case_sensitive_file_extension_comparison(ctx: &LateContext<'_>, expr: &Expr<'_>) -> Option<Span> {
    if_chain! {
        if let ExprKind::MethodCall(PathSegment { ident, .. }, _, [obj, extension, ..], span) = expr.kind;
        if ident.as_str() == "ends_with";
        if let ExprKind::Lit(Spanned { node: LitKind::Str(ext_literal, ..), ..}) = extension.kind;
        if (2..=6).contains(&ext_literal.as_str().len());
        if ext_literal.as_str().starts_with('.');
        if ext_literal.as_str().chars().skip(1).all(|c| c.is_uppercase() || c.is_digit(10))
            || ext_literal.as_str().chars().skip(1).all(|c| c.is_lowercase() || c.is_digit(10));
        then {
            let mut ty = ctx.typeck_results().expr_ty(obj);
            ty = match ty.kind() {
                ty::Ref(_, ty, ..) => ty,
                _ => ty
            };

            match ty.kind() {
                ty::Str => {
                    return Some(span);
                },
                ty::Adt(&ty::AdtDef { did, .. }, _) => {
                    if ctx.tcx.is_diagnostic_item(sym::String, did) {
                        return Some(span);
                    }
                },
                _ => { return None; }
            }
        }
    }
    None
}

impl LateLintPass<'tcx> for CaseSensitiveFileExtensionComparisons {
    fn check_expr(&mut self, ctx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let Some(span) = check_case_sensitive_file_extension_comparison(ctx, expr) {
            span_lint_and_help(
                ctx,
                CASE_SENSITIVE_FILE_EXTENSION_COMPARISONS,
                span,
                "case-sensitive file extension comparison",
                None,
                "consider using a case-insensitive comparison instead",
            );
        }
    }
}
