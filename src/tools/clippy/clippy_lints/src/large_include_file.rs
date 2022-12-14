use clippy_utils::diagnostics::span_lint_and_note;
use clippy_utils::is_lint_allowed;
use clippy_utils::macros::root_macro_call_first_node;
use rustc_ast::LitKind;
use rustc_hir::Expr;
use rustc_hir::ExprKind;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the inclusion of large files via `include_bytes!()`
    /// and `include_str!()`
    ///
    /// ### Why is this bad?
    /// Including large files can increase the size of the binary
    ///
    /// ### Example
    /// ```rust,ignore
    /// let included_str = include_str!("very_large_file.txt");
    /// let included_bytes = include_bytes!("very_large_file.txt");
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// use std::fs;
    ///
    /// // You can load the file at runtime
    /// let string = fs::read_to_string("very_large_file.txt")?;
    /// let bytes = fs::read("very_large_file.txt")?;
    /// ```
    #[clippy::version = "1.62.0"]
    pub LARGE_INCLUDE_FILE,
    restriction,
    "including a large file"
}

pub struct LargeIncludeFile {
    max_file_size: u64,
}

impl LargeIncludeFile {
    #[must_use]
    pub fn new(max_file_size: u64) -> Self {
        Self { max_file_size }
    }
}

impl_lint_pass!(LargeIncludeFile => [LARGE_INCLUDE_FILE]);

impl LateLintPass<'_> for LargeIncludeFile {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'_ Expr<'_>) {
        if_chain! {
            if let Some(macro_call) = root_macro_call_first_node(cx, expr);
            if !is_lint_allowed(cx, LARGE_INCLUDE_FILE, expr.hir_id);
            if cx.tcx.is_diagnostic_item(sym::include_bytes_macro, macro_call.def_id)
            || cx.tcx.is_diagnostic_item(sym::include_str_macro, macro_call.def_id);
            if let ExprKind::Lit(lit) = &expr.kind;
            then {
                let len = match &lit.node {
                    // include_bytes
                    LitKind::ByteStr(bstr, _) => bstr.len(),
                    // include_str
                    LitKind::Str(sym, _) => sym.as_str().len(),
                    _ => return,
                };

                if len as u64 <= self.max_file_size {
                    return;
                }

                span_lint_and_note(
                    cx,
                    LARGE_INCLUDE_FILE,
                    expr.span,
                    "attempted to include a large file",
                    None,
                    &format!(
                        "the configuration allows a maximum size of {} bytes",
                        self.max_file_size
                    ),
                );
            }
        }
    }
}
