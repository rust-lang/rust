use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{is_in_cfg_test, is_in_test_function, is_lint_allowed};
use rustc_hir::Expr;
use rustc_lint::{LateContext, Lint};
use rustc_middle::ty;
use rustc_span::sym;

use super::{EXPECT_USED, UNWRAP_USED};

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum Variant {
    Unwrap,
    Expect,
}

impl Variant {
    fn method_name(self, is_err: bool) -> &'static str {
        match (self, is_err) {
            (Variant::Unwrap, true) => "unwrap_err",
            (Variant::Unwrap, false) => "unwrap",
            (Variant::Expect, true) => "expect_err",
            (Variant::Expect, false) => "expect",
        }
    }

    fn lint(self) -> &'static Lint {
        match self {
            Variant::Unwrap => UNWRAP_USED,
            Variant::Expect => EXPECT_USED,
        }
    }
}

/// lint use of `unwrap()` or `unwrap_err` for `Result` and `unwrap()` for `Option`.
pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    recv: &Expr<'_>,
    is_err: bool,
    allow_unwrap_in_tests: bool,
    variant: Variant,
) {
    let ty = cx.typeck_results().expr_ty(recv).peel_refs();

    let (kind, none_value, none_prefix) = if is_type_diagnostic_item(cx, ty, sym::Option) && !is_err {
        ("an `Option`", "None", "")
    } else if is_type_diagnostic_item(cx, ty, sym::Result)
        && let ty::Adt(_, substs) = ty.kind()
        && let Some(t_or_e_ty) = substs[usize::from(!is_err)].as_type()
    {
        // Issue #11245: Do not lint `!` or never-like enums
        if t_or_e_ty.is_never()
            || (t_or_e_ty.is_enum() && t_or_e_ty.ty_adt_def().is_some_and(|def| def.variants().is_empty()))
        {
            return;
        }

        ("a `Result`", if is_err { "Ok" } else { "Err" }, "an ")
    } else {
        return;
    };

    let method_suffix = if is_err { "_err" } else { "" };

    if allow_unwrap_in_tests && (is_in_test_function(cx.tcx, expr.hir_id) || is_in_cfg_test(cx.tcx, expr.hir_id)) {
        return;
    }

    let help = if variant == Variant::Unwrap && is_lint_allowed(cx, EXPECT_USED, expr.hir_id) {
        format!(
            "if you don't want to handle the `{none_value}` case gracefully, consider \
                using `expect{method_suffix}()` to provide a better panic message"
        )
    } else {
        format!("if this value is {none_prefix}`{none_value}`, it will panic")
    };

    span_lint_and_help(
        cx,
        variant.lint(),
        expr.span,
        &format!("used `{}()` on {kind} value", variant.method_name(is_err)),
        None,
        &help,
    );
}
