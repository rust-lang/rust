use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::MaybeDef;
use clippy_utils::ty::is_never_like;
use clippy_utils::{is_in_test, is_inside_always_const_context, is_lint_allowed};
use rustc_hir::Expr;
use rustc_hir::def::DefKind;
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

/// Lint usage of `unwrap` or `unwrap_err` for `Result` and `unwrap()` for `Option` (and their
/// `expect` counterparts).
#[allow(clippy::too_many_arguments)]
pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    recv: &Expr<'_>,
    is_err: bool,
    allow_unwrap_in_consts: bool,
    allow_unwrap_in_tests: bool,
    allow_unwrap_types: &[String],
    variant: Variant,
) {
    let ty = cx.typeck_results().expr_ty(recv).peel_refs();

    let (kind, none_value, none_prefix) = match ty.opt_diag_name(cx) {
        Some(sym::Option) if !is_err => ("an `Option`", "None", ""),
        Some(sym::Result)
            if let ty::Adt(_, substs) = ty.kind()
                && let Some(t_or_e_ty) = substs[usize::from(!is_err)].as_type() =>
        {
            if is_never_like(t_or_e_ty) {
                return;
            }

            ("a `Result`", if is_err { "Ok" } else { "Err" }, "an ")
        },
        _ => return,
    };

    let method_suffix = if is_err { "_err" } else { "" };

    let ty_name = ty.to_string();
    if allow_unwrap_types
        .iter()
        .any(|allowed_type| ty_name.starts_with(allowed_type) || ty_name == *allowed_type)
    {
        return;
    }

    for s in allow_unwrap_types {
        let def_ids = clippy_utils::paths::lookup_path_str(cx.tcx, clippy_utils::paths::PathNS::Type, s);
        for def_id in def_ids {
            if let ty::Adt(adt, _) = ty.kind()
                && adt.did() == def_id
            {
                return;
            }
            if cx.tcx.def_kind(def_id) == DefKind::TyAlias {
                let alias_ty = cx.tcx.type_of(def_id).instantiate_identity();
                if let (ty::Adt(adt, substs), ty::Adt(alias_adt, alias_substs)) = (ty.kind(), alias_ty.kind())
                    && adt.did() == alias_adt.did()
                {
                    let mut all_match = true;
                    for (arg, alias_arg) in substs.iter().zip(alias_substs.iter()) {
                        if let (Some(arg_ty), Some(alias_arg_ty)) = (arg.as_type(), alias_arg.as_type()) {
                            if matches!(alias_arg_ty.kind(), ty::Param(_)) {
                                continue;
                            }
                            if let (ty::Adt(arg_adt, _), ty::Adt(alias_arg_adt, _)) =
                                (arg_ty.peel_refs().kind(), alias_arg_ty.peel_refs().kind())
                            {
                                if arg_adt.did() != alias_arg_adt.did() {
                                    all_match = false;
                                    break;
                                }
                            } else if arg_ty != alias_arg_ty {
                                all_match = false;
                                break;
                            }
                        }
                    }
                    if all_match {
                        return;
                    }
                }
            }
        }
    }

    if allow_unwrap_in_tests && is_in_test(cx.tcx, expr.hir_id) {
        return;
    }

    if allow_unwrap_in_consts && is_inside_always_const_context(cx.tcx, expr.hir_id) {
        return;
    }

    span_lint_and_then(
        cx,
        variant.lint(),
        expr.span,
        format!("used `{}()` on {kind} value", variant.method_name(is_err)),
        |diag| {
            diag.note(format!("if this value is {none_prefix}`{none_value}`, it will panic"));

            if variant == Variant::Unwrap && is_lint_allowed(cx, EXPECT_USED, expr.hir_id) {
                diag.help(format!(
                    "consider using `expect{method_suffix}()` to provide a better panic message"
                ));
            }
        },
    );
}

#[expect(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
pub(super) fn check_call(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    func: &Expr<'_>,
    args: &[Expr<'_>],
    allow_unwrap_in_consts: bool,
    allow_unwrap_in_tests: bool,
    allow_expect_in_consts: bool,
    allow_expect_in_tests: bool,
    allow_unwrap_types: &[String],
) {
    let Some(recv) = args.first() else {
        return;
    };
    let Some((DefKind::AssocFn, def_id)) = cx.typeck_results().type_dependent_def(func.hir_id) else {
        return;
    };

    match cx.tcx.item_name(def_id) {
        sym::unwrap => {
            check(
                cx,
                expr,
                recv,
                false,
                allow_unwrap_in_consts,
                allow_unwrap_in_tests,
                allow_unwrap_types,
                Variant::Unwrap,
            );
        },
        sym::expect => {
            check(
                cx,
                expr,
                recv,
                false,
                allow_expect_in_consts,
                allow_expect_in_tests,
                allow_unwrap_types,
                Variant::Expect,
            );
        },
        clippy_utils::sym::unwrap_err => {
            check(
                cx,
                expr,
                recv,
                true,
                allow_unwrap_in_consts,
                allow_unwrap_in_tests,
                allow_unwrap_types,
                Variant::Unwrap,
            );
        },
        clippy_utils::sym::expect_err => {
            check(
                cx,
                expr,
                recv,
                true,
                allow_expect_in_consts,
                allow_expect_in_tests,
                allow_unwrap_types,
                Variant::Expect,
            );
        },
        _ => (),
    }
}
