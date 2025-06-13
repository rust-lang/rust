use crate::Lint;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{is_lint_allowed, sym};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::Ty;
use rustc_session::declare_lint_pass;
use rustc_span::Symbol;
use std::fmt::Write;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of the `to_ne_bytes` method and/or the function `from_ne_bytes`.
    ///
    /// ### Why restrict this?
    /// To ensure use of explicitly chosen endianness rather than the target’s endianness,
    /// such as when implementing network protocols or file formats rather than FFI.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let _x = 2i32.to_ne_bytes();
    /// let _y = 2i64.to_ne_bytes();
    /// ```
    #[clippy::version = "1.72.0"]
    pub HOST_ENDIAN_BYTES,
    restriction,
    "disallows usage of the `to_ne_bytes` method"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of the `to_le_bytes` method and/or the function `from_le_bytes`.
    ///
    /// ### Why restrict this?
    /// To ensure use of big-endian or the target’s endianness rather than little-endian.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let _x = 2i32.to_le_bytes();
    /// let _y = 2i64.to_le_bytes();
    /// ```
    #[clippy::version = "1.72.0"]
    pub LITTLE_ENDIAN_BYTES,
    restriction,
    "disallows usage of the `to_le_bytes` method"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of the `to_be_bytes` method and/or the function `from_be_bytes`.
    ///
    /// ### Why restrict this?
    /// To ensure use of little-endian or the target’s endianness rather than big-endian.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let _x = 2i32.to_be_bytes();
    /// let _y = 2i64.to_be_bytes();
    /// ```
    #[clippy::version = "1.72.0"]
    pub BIG_ENDIAN_BYTES,
    restriction,
    "disallows usage of the `to_be_bytes` method"
}

declare_lint_pass!(EndianBytes => [HOST_ENDIAN_BYTES, LITTLE_ENDIAN_BYTES, BIG_ENDIAN_BYTES]);

const HOST_NAMES: [Symbol; 2] = [sym::from_ne_bytes, sym::to_ne_bytes];
const LITTLE_NAMES: [Symbol; 2] = [sym::from_le_bytes, sym::to_le_bytes];
const BIG_NAMES: [Symbol; 2] = [sym::from_be_bytes, sym::to_be_bytes];

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

    fn as_name(&self, prefix: Prefix) -> Symbol {
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
        let (prefix, name, ty_expr) = match expr.kind {
            ExprKind::MethodCall(method_name, receiver, [], ..) => (Prefix::To, method_name.ident.name, receiver),
            ExprKind::Call(function, ..)
                if let ExprKind::Path(qpath) = function.kind
                    && let Some(def_id) = cx.qpath_res(&qpath, function.hir_id).opt_def_id()
                    && let Some(function_name) = cx.get_def_path(def_id).last() =>
            {
                (Prefix::From, *function_name, expr)
            },
            _ => return,
        };
        if !expr.span.in_external_macro(cx.sess().source_map())
            && let ty = cx.typeck_results().expr_ty(ty_expr)
            && ty.is_primitive_ty()
        {
            maybe_lint_endian_bytes(cx, expr, prefix, name, ty);
        }
    }
}

fn maybe_lint_endian_bytes(cx: &LateContext<'_>, expr: &Expr<'_>, prefix: Prefix, name: Symbol, ty: Ty<'_>) {
    let ne = LintKind::Host.as_name(prefix);
    let le = LintKind::Little.as_name(prefix);
    let be = LintKind::Big.as_name(prefix);

    let (lint, other_lints) = match name {
        name if name == ne => ((&LintKind::Host), [(&LintKind::Little), (&LintKind::Big)]),
        name if name == le => ((&LintKind::Little), [(&LintKind::Host), (&LintKind::Big)]),
        name if name == be => ((&LintKind::Big), [(&LintKind::Host), (&LintKind::Little)]),
        _ => return,
    };

    span_lint_and_then(
        cx,
        lint.as_lint(),
        expr.span,
        format!(
            "usage of the {}`{ty}::{}`{}",
            if prefix == Prefix::From { "function " } else { "" },
            lint.as_name(prefix),
            if prefix == Prefix::To { " method" } else { "" },
        ),
        move |diag| {
            // all lints disallowed, don't give help here
            if [&[lint], other_lints.as_slice()]
                .concat()
                .iter()
                .all(|lint| !lint.allowed(cx, expr))
            {
                return;
            }

            // ne_bytes and all other lints allowed
            if lint.as_name(prefix) == ne && other_lints.iter().all(|lint| lint.allowed(cx, expr)) {
                diag.help("specify the desired endianness explicitly");
                return;
            }

            // le_bytes where ne_bytes allowed but be_bytes is not, or le_bytes where ne_bytes allowed but
            // le_bytes is not
            if (lint.as_name(prefix) == le || lint.as_name(prefix) == be) && LintKind::Host.allowed(cx, expr) {
                diag.help("use the native endianness instead");
                return;
            }

            let allowed_lints = other_lints.iter().filter(|lint| lint.allowed(cx, expr));
            let len = allowed_lints.clone().count();

            let mut help_str = "use ".to_owned();

            for (i, lint) in allowed_lints.enumerate() {
                let only_one = len == 1;
                if !only_one {
                    help_str.push_str("either of ");
                }

                write!(help_str, "`{ty}::{}` ", lint.as_name(prefix)).unwrap();

                if i != len && !only_one {
                    help_str.push_str("or ");
                }
            }
            help_str.push_str("instead");
            diag.help(help_str);
        },
    );
}
