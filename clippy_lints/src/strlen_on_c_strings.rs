use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::in_macro;
use clippy_utils::paths;
use clippy_utils::source::snippet_with_macro_callsite;
use clippy_utils::ty::{is_type_diagnostic_item, is_type_ref_to_diagnostic_item};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::{sym, Symbol};

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
    pub STRLEN_ON_C_STRINGS,
    complexity,
    "using `libc::strlen` on a `CString` or `CStr` value, while `as_bytes().len()` or `to_bytes().len()` respectively can be used instead"
}

declare_lint_pass!(StrlenOnCStrings => [STRLEN_ON_C_STRINGS]);

impl LateLintPass<'tcx> for StrlenOnCStrings {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if in_macro(expr.span) {
            return;
        }

        if_chain! {
            if let hir::ExprKind::Call(func, [recv]) = expr.kind;
            if let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = func.kind;

            if (&paths::LIBC_STRLEN).iter().map(|x| Symbol::intern(x)).eq(
                path.segments.iter().map(|seg| seg.ident.name));
            if let hir::ExprKind::MethodCall(path, _, args, _) = recv.kind;
            if args.len() == 1;
            if !args.iter().any(|e| e.span.from_expansion());
            if path.ident.name == sym::as_ptr;
            then {
                let cstring = &args[0];
                let ty = cx.typeck_results().expr_ty(cstring);
                let val_name = snippet_with_macro_callsite(cx, cstring.span, "..");
                let sugg = if is_type_diagnostic_item(cx, ty, sym::cstring_type){
                    format!("{}.as_bytes().len()", val_name)
                } else if is_type_ref_to_diagnostic_item(cx, ty, sym::CStr){
                    format!("{}.to_bytes().len()", val_name)
                } else {
                    return;
                };

                span_lint_and_sugg(
                    cx,
                    STRLEN_ON_C_STRINGS,
                    expr.span,
                    "using `libc::strlen` on a `CString` or `CStr` value",
                    "try this (you might also need to get rid of `unsafe` block in some cases):",
                    sugg,
                    Applicability::Unspecified // Sometimes unnecessary `unsafe` block
                );
            }
        }
    }
}
