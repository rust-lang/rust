use clippy_utils::{
    diagnostics::{span_lint_and_help, span_lint_and_then},
    is_lint_allowed, match_def_path, path_def_id,
};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    ///
    /// ### Why is this bad?
    ///
    /// ### Example
    /// ```rust
    /// // example code where clippy issues a warning
    /// ```
    /// Use instead:
    /// ```rust
    /// // example code which does not raise clippy warning
    /// ```
    #[clippy::version = "1.71.0"]
    pub HOST_ENDIAN_BYTES,
    restriction,
    "disallows usage of the `to_ne_bytes` method"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of the `to_ne_bytes` method.
    ///
    /// ### Why is this bad?
    /// It's not, but some may prefer to specify the target endianness explicitly.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let _x = 2i32.to_ne_bytes();
    /// let _y = 2i64.to_ne_bytes();
    /// ```
    #[clippy::version = "1.71.0"]
    pub LITTLE_ENDIAN_BYTES,
    restriction,
    "disallows usage of the `to_le_bytes` method"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of the `to_le_bytes` method.
    ///
    /// ### Why is this bad?
    ///
    /// ### Example
    /// ```rust,ignore
    /// // example code where clippy issues a warning
    /// ```
    #[clippy::version = "1.71.0"]
    pub BIG_ENDIAN_BYTES,
    restriction,
    "disallows usage of the `to_be_bytes` method"
}

declare_lint_pass!(EndianBytes => [HOST_ENDIAN_BYTES, LITTLE_ENDIAN_BYTES, BIG_ENDIAN_BYTES]);

impl LateLintPass<'_> for EndianBytes {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if_chain! {
            if let ExprKind::MethodCall(method_name, receiver, args, ..) = expr.kind;
            if let ExprKind::Lit(..) = receiver.kind;
            if args.is_empty();
            then {
                if method_name.ident.name == sym!(to_ne_bytes) {
                    span_lint_and_help(
                        cx,
                        HOST_ENDIAN_BYTES,
                        expr.span,
                        "use of the method `to_ne_bytes`",
                        None,
                        "consider specifying the desired endianness",
                    );
                } else if method_name.ident.name == sym!(to_le_bytes) {
                    span_lint_and_then(cx, LITTLE_ENDIAN_BYTES, expr.span, "use of the method `to_le_bytes`", |diag| {
                        if is_lint_allowed(cx, BIG_ENDIAN_BYTES, expr.hir_id) {
                            diag.help("use `to_be_bytes` instead");
                        }
                    });
                } else if method_name.ident.name == sym!(to_be_bytes) {
                    span_lint_and_then(cx, BIG_ENDIAN_BYTES, expr.span, "use of the method `to_be_bytes`", |diag| {
                        if is_lint_allowed(cx, LITTLE_ENDIAN_BYTES, expr.hir_id) {
                            diag.help("use `to_le_bytes` instead");
                        }
                    });
                }

                // don't waste time also checking from_**_bytes
                return;
            }
        }

        span_lint_and_help(
            cx,
            HOST_ENDIAN_BYTES,
            expr.span,
            "use of the method `from_ne_bytes`",
            None,
            &format!("consider specifying the desired endianness: {expr:?}"),
        );

        if_chain! {
            if let ExprKind::Call(function, args) = expr.kind;
            if let Some(function_def_id) = path_def_id(cx, function);
            if args.len() == 1;
            then {
                if match_def_path(cx, function_def_id, &["from_ne_bytes"]) {
                    span_lint_and_help(
                        cx,
                        HOST_ENDIAN_BYTES,
                        expr.span,
                        "use of the method `from_ne_bytes`",
                        None,
                        "consider specifying the desired endianness",
                    );
                } else if match_def_path(cx, function_def_id, &["from_le_bytes"]) {
                    span_lint_and_then(cx, LITTLE_ENDIAN_BYTES, expr.span, "use of the method `from_le_bytes`", |diag| {
                        if is_lint_allowed(cx, BIG_ENDIAN_BYTES, expr.hir_id) {
                            diag.help("use `from_be_bytes` instead");
                        }
                    });
                } else if match_def_path(cx, function_def_id, &["from_be_bytes"]) {
                    span_lint_and_then(cx, BIG_ENDIAN_BYTES, expr.span, "use of the method `from_be_bytes`", |diag| {
                        if is_lint_allowed(cx, LITTLE_ENDIAN_BYTES, expr.hir_id) {
                            diag.help("use `from_le_bytes` instead");
                        }
                    });
                }
            }
        }
    }
}
