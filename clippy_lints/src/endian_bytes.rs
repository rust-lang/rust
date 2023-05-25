use crate::Lint;
use clippy_utils::{diagnostics::span_lint_and_then, is_lint_allowed};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Symbol;
use std::borrow::Cow;

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
    pub HOST_ENDIAN_BYTES,
    restriction,
    "disallows usage of the `to_ne_bytes` method"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of the `to_le_bytes` method.
    ///
    /// ### Why is this bad?
    /// It's not, but some may wish to lint usages of this method, either to suggest using the host
    /// endianness or big endian.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let _x = 2i32.to_le_bytes();
    /// let _y = 2i64.to_le_bytes();
    /// ```
    #[clippy::version = "1.71.0"]
    pub LITTLE_ENDIAN_BYTES,
    restriction,
    "disallows usage of the `to_le_bytes` method"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of the `to_be_bytes` method.
    ///
    /// ### Why is this bad?
    /// It's not, but some may wish to lint usages of this method, either to suggest using the host
    /// endianness or little endian.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let _x = 2i32.to_be_bytes();
    /// let _y = 2i64.to_be_bytes();
    /// ```
    #[clippy::version = "1.71.0"]
    pub BIG_ENDIAN_BYTES,
    restriction,
    "disallows usage of the `to_be_bytes` method"
}

declare_lint_pass!(EndianBytes => [HOST_ENDIAN_BYTES, LITTLE_ENDIAN_BYTES, BIG_ENDIAN_BYTES]);

#[derive(Clone, Debug)]
enum LintKind {
    Host,
    Little,
    Big,
}

impl LintKind {
    fn allowed(&self, cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
        is_lint_allowed(cx, self.as_lint(), expr.hir_id)
    }

    fn as_lint(&self) -> &'static Lint {
        match self {
            LintKind::Host => HOST_ENDIAN_BYTES,
            LintKind::Little => LITTLE_ENDIAN_BYTES,
            LintKind::Big => BIG_ENDIAN_BYTES,
        }
    }

    fn to_name(&self, prefix: &str) -> String {
        match self {
            LintKind::Host => format!("{prefix}_ne_bytes"),
            LintKind::Little => format!("{prefix}_le_bytes"),
            LintKind::Big => format!("{prefix}_be_bytes"),
        }
    }
}

impl LateLintPass<'_> for EndianBytes {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if in_external_macro(cx.sess(), expr.span) {
            return;
        }

        if_chain! {
            if let ExprKind::MethodCall(method_name, receiver, args, ..) = expr.kind;
            if let ExprKind::Lit(..) = receiver.kind;
            if args.is_empty();
            if try_lint_endian_bytes(cx, expr, "to", method_name.ident.name);
            then {
                return;
            }
        }

        if_chain! {
            if let ExprKind::Call(function, ..) = expr.kind;
            if let ExprKind::Path(qpath) = function.kind;
            if let Some(def_id) = cx.qpath_res(&qpath, function.hir_id).opt_def_id();
            if let Some(function_name) = cx.get_def_path(def_id).last();
            if cx.typeck_results().expr_ty(expr).is_primitive_ty();
            then {
                try_lint_endian_bytes(cx, expr, "from", *function_name);
            }
        }
    }
}

fn try_lint_endian_bytes(cx: &LateContext<'_>, expr: &Expr<'_>, prefix: &str, name: Symbol) -> bool {
    let ne = format!("{prefix}_ne_bytes");
    let le = format!("{prefix}_le_bytes");
    let be = format!("{prefix}_be_bytes");

    let (lint, other_lints) = match name.as_str() {
        name if name == ne => ((&LintKind::Host), [(&LintKind::Little), (&LintKind::Big)]),
        name if name == le => ((&LintKind::Little), [(&LintKind::Host), (&LintKind::Big)]),
        name if name == be => ((&LintKind::Big), [(&LintKind::Host), (&LintKind::Little)]),
        _ => return false,
    };

    let mut help = None;

    'build_help: {
        // all lints disallowed, don't give help here
        if [&[lint], other_lints.as_slice()]
            .concat()
            .iter()
            .all(|lint| !lint.allowed(cx, expr))
        {
            break 'build_help;
        }

        // ne_bytes and all other lints allowed
        if lint.to_name(prefix) == ne && other_lints.iter().all(|lint| lint.allowed(cx, expr)) {
            help = Some(Cow::Borrowed("specify the desired endianness explicitly"));
            break 'build_help;
        }

        // le_bytes where ne_bytes allowed but be_bytes is not, or le_bytes where ne_bytes allowed but
        // le_bytes is not
        if (lint.to_name(prefix) == le || lint.to_name(prefix) == be) && LintKind::Host.allowed(cx, expr) {
            help = Some(Cow::Borrowed("use the native endianness instead"));
            break 'build_help;
        }

        let allowed_lints = other_lints.iter().filter(|lint| lint.allowed(cx, expr));
        let len = allowed_lints.clone().count();

        let mut help_str = "use ".to_owned();

        for (i, lint) in allowed_lints.enumerate() {
            let only_one = len == 1;
            if !only_one {
                help_str.push_str("either of ");
            }

            help_str.push_str(&format!("`{}` ", lint.to_name(prefix)));

            if i != len && !only_one {
                help_str.push_str("or ");
            }
        }

        help = Some(Cow::Owned(help_str + "instead"));
    }

    span_lint_and_then(
        cx,
        lint.as_lint(),
        expr.span,
        &format!("usage of the method `{}`", lint.to_name(prefix)),
        move |diag| {
            if let Some(help) = help {
                diag.help(help);
            }
        },
    );

    true
}
