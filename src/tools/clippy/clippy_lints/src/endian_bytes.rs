use crate::Lint;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_lint_allowed;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::Ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Symbol;
use std::borrow::Cow;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of the `to_ne_bytes` method and/or the function `from_ne_bytes`.
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
    /// Checks for the usage of the `to_le_bytes` method and/or the function `from_le_bytes`.
    ///
    /// ### Why is this bad?
    /// It's not, but some may wish to lint usage of this method, either to suggest using the host
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
    /// Checks for the usage of the `to_be_bytes` method and/or the function `from_be_bytes`.
    ///
    /// ### Why is this bad?
    /// It's not, but some may wish to lint usage of this method, either to suggest using the host
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

const HOST_NAMES: [&str; 2] = ["from_ne_bytes", "to_ne_bytes"];
const LITTLE_NAMES: [&str; 2] = ["from_le_bytes", "to_le_bytes"];
const BIG_NAMES: [&str; 2] = ["from_be_bytes", "to_be_bytes"];

#[derive(Clone, Debug)]
enum LintKind {
    Host,
    Little,
    Big,
}

#[derive(Clone, Copy, PartialEq)]
enum Prefix {
    From,
    To,
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

    fn as_name(&self, prefix: Prefix) -> &str {
        let index = usize::from(prefix == Prefix::To);

        match self {
            LintKind::Host => HOST_NAMES[index],
            LintKind::Little => LITTLE_NAMES[index],
            LintKind::Big => BIG_NAMES[index],
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
            if args.is_empty();
            let ty = cx.typeck_results().expr_ty(receiver);
            if ty.is_primitive_ty();
            if maybe_lint_endian_bytes(cx, expr, Prefix::To, method_name.ident.name, ty);
            then {
                return;
            }
        }

        if_chain! {
            if let ExprKind::Call(function, ..) = expr.kind;
            if let ExprKind::Path(qpath) = function.kind;
            if let Some(def_id) = cx.qpath_res(&qpath, function.hir_id).opt_def_id();
            if let Some(function_name) = cx.get_def_path(def_id).last();
            let ty = cx.typeck_results().expr_ty(expr);
            if ty.is_primitive_ty();
            then {
                maybe_lint_endian_bytes(cx, expr, Prefix::From, *function_name, ty);
            }
        }
    }
}

fn maybe_lint_endian_bytes(cx: &LateContext<'_>, expr: &Expr<'_>, prefix: Prefix, name: Symbol, ty: Ty<'_>) -> bool {
    let ne = LintKind::Host.as_name(prefix);
    let le = LintKind::Little.as_name(prefix);
    let be = LintKind::Big.as_name(prefix);

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
        if lint.as_name(prefix) == ne && other_lints.iter().all(|lint| lint.allowed(cx, expr)) {
            help = Some(Cow::Borrowed("specify the desired endianness explicitly"));
            break 'build_help;
        }

        // le_bytes where ne_bytes allowed but be_bytes is not, or le_bytes where ne_bytes allowed but
        // le_bytes is not
        if (lint.as_name(prefix) == le || lint.as_name(prefix) == be) && LintKind::Host.allowed(cx, expr) {
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

            help_str.push_str(&format!("`{ty}::{}` ", lint.as_name(prefix)));

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
        &format!(
            "usage of the {}`{ty}::{}`{}",
            if prefix == Prefix::From { "function " } else { "" },
            lint.as_name(prefix),
            if prefix == Prefix::To { " method" } else { "" },
        ),
        move |diag| {
            if let Some(help) = help {
                diag.help(help);
            }
        },
    );

    true
}
