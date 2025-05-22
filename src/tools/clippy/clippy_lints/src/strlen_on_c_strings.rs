use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::{is_type_diagnostic_item, is_type_lang_item};
use clippy_utils::visitors::is_expr_unsafe;
use clippy_utils::{match_libc_symbol, sym};
use rustc_errors::Applicability;
use rustc_hir::{Block, BlockCheckMode, Expr, ExprKind, LangItem, Node, UnsafeSource};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `libc::strlen` on a `CString` or `CStr` value,
    /// and suggest calling `as_bytes().len()` or `to_bytes().len()` respectively instead.
    ///
    /// ### Why is this bad?
    /// This avoids calling an unsafe `libc` function.
    /// Currently, it also avoids calculating the length.
    ///
    /// ### Example
    /// ```rust, ignore
    /// use std::ffi::CString;
    /// let cstring = CString::new("foo").expect("CString::new failed");
    /// let len = unsafe { libc::strlen(cstring.as_ptr()) };
    /// ```
    /// Use instead:
    /// ```rust, no_run
    /// use std::ffi::CString;
    /// let cstring = CString::new("foo").expect("CString::new failed");
    /// let len = cstring.as_bytes().len();
    /// ```
    #[clippy::version = "1.55.0"]
    pub STRLEN_ON_C_STRINGS,
    complexity,
    "using `libc::strlen` on a `CString` or `CStr` value, while `as_bytes().len()` or `to_bytes().len()` respectively can be used instead"
}

declare_lint_pass!(StrlenOnCStrings => [STRLEN_ON_C_STRINGS]);

impl<'tcx> LateLintPass<'tcx> for StrlenOnCStrings {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !expr.span.from_expansion()
            && let ExprKind::Call(func, [recv]) = expr.kind
            && let ExprKind::Path(path) = &func.kind
            && let Some(did) = cx.qpath_res(path, func.hir_id).opt_def_id()
            && match_libc_symbol(cx, did, sym::strlen)
            && let ExprKind::MethodCall(path, self_arg, [], _) = recv.kind
            && !recv.span.from_expansion()
            && path.ident.name == sym::as_ptr
        {
            let ctxt = expr.span.ctxt();
            let span = match cx.tcx.parent_hir_node(expr.hir_id) {
                Node::Block(&Block {
                    rules: BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided),
                    span,
                    ..
                }) if span.ctxt() == ctxt && !is_expr_unsafe(cx, self_arg) => span,
                _ => expr.span,
            };

            let ty = cx.typeck_results().expr_ty(self_arg).peel_refs();
            let mut app = Applicability::MachineApplicable;
            let val_name = snippet_with_context(cx, self_arg.span, ctxt, "..", &mut app).0;
            let method_name = if is_type_diagnostic_item(cx, ty, sym::cstring_type) {
                "as_bytes"
            } else if is_type_lang_item(cx, ty, LangItem::CStr) {
                "to_bytes"
            } else {
                return;
            };

            span_lint_and_sugg(
                cx,
                STRLEN_ON_C_STRINGS,
                span,
                "using `libc::strlen` on a `CString` or `CStr` value",
                "try",
                format!("{val_name}.{method_name}().len()"),
                app,
            );
        }
    }
}
