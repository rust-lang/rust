use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::root_macro_call_first_node;
use rustc_ast::LitKind;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the inclusion of large files via `include_bytes!()`
    /// or `include_str!()`.
    ///
    /// ### Why restrict this?
    /// Including large files can undesirably increase the size of the binary produced by the compiler.
    /// This lint may be used to catch mistakes where an unexpectedly large file is included, or
    /// temporarily to obtain a list of all large files.
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
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            max_file_size: conf.max_include_file_size,
        }
    }
}

impl_lint_pass!(LargeIncludeFile => [LARGE_INCLUDE_FILE]);

impl LateLintPass<'_> for LargeIncludeFile {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'_ Expr<'_>) {
        if let ExprKind::Lit(lit) = &expr.kind
            && let len = match &lit.node {
                // include_bytes
                LitKind::ByteStr(bstr, _) => bstr.len(),
                // include_str
                LitKind::Str(sym, _) => sym.as_str().len(),
                _ => return,
            }
            && len as u64 > self.max_file_size
            && let Some(macro_call) = root_macro_call_first_node(cx, expr)
            && (cx.tcx.is_diagnostic_item(sym::include_bytes_macro, macro_call.def_id)
                || cx.tcx.is_diagnostic_item(sym::include_str_macro, macro_call.def_id))
        {
            #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
            span_lint_and_then(
                cx,
                LARGE_INCLUDE_FILE,
                expr.span.source_callsite(),
                "attempted to include a large file",
                |diag| {
                    diag.note(format!(
                        "the configuration allows a maximum size of {} bytes",
                        self.max_file_size
                    ));
                },
            );
        }
    }
}
