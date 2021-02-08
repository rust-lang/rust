use crate::utils::{
    is_type_diagnostic_item, match_trait_method, match_type, paths, snippet, snippet_with_applicability, span_lint,
    span_lint_and_help, span_lint_and_sugg,
};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Mutability, Pat};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty, TyS};
use rustc_span::symbol::sym;

pub(super) fn check_for_loop_arg(cx: &LateContext<'_>, pat: &Pat<'_>, arg: &Expr<'_>, expr: &Expr<'_>) {
    let mut next_loop_linted = false; // whether or not ITER_NEXT_LOOP lint was used
    if let ExprKind::MethodCall(ref method, _, ref args, _) = arg.kind {
        // just the receiver, no arguments
        if args.len() == 1 {
            let method_name = &*method.ident.as_str();
            // check for looping over x.iter() or x.iter_mut(), could use &x or &mut x
            if method_name == "iter" || method_name == "iter_mut" {
                if is_ref_iterable_type(cx, &args[0]) {
                    lint_iter_method(cx, args, arg, method_name);
                }
            } else if method_name == "into_iter" && match_trait_method(cx, arg, &paths::INTO_ITERATOR) {
                let receiver_ty = cx.typeck_results().expr_ty(&args[0]);
                let receiver_ty_adjusted = cx.typeck_results().expr_ty_adjusted(&args[0]);
                if TyS::same_type(receiver_ty, receiver_ty_adjusted) {
                    let mut applicability = Applicability::MachineApplicable;
                    let object = snippet_with_applicability(cx, args[0].span, "_", &mut applicability);
                    span_lint_and_sugg(
                        cx,
                        super::EXPLICIT_INTO_ITER_LOOP,
                        arg.span,
                        "it is more concise to loop over containers instead of using explicit \
                         iteration methods",
                        "to write this more concisely, try",
                        object.to_string(),
                        applicability,
                    );
                } else {
                    let ref_receiver_ty = cx.tcx.mk_ref(
                        cx.tcx.lifetimes.re_erased,
                        ty::TypeAndMut {
                            ty: receiver_ty,
                            mutbl: Mutability::Not,
                        },
                    );
                    if TyS::same_type(receiver_ty_adjusted, ref_receiver_ty) {
                        lint_iter_method(cx, args, arg, method_name)
                    }
                }
            } else if method_name == "next" && match_trait_method(cx, arg, &paths::ITERATOR) {
                span_lint(
                    cx,
                    super::ITER_NEXT_LOOP,
                    expr.span,
                    "you are iterating over `Iterator::next()` which is an Option; this will compile but is \
                    probably not what you want",
                );
                next_loop_linted = true;
            }
        }
    }
    if !next_loop_linted {
        check_arg_type(cx, pat, arg);
    }
}

/// Checks for `for` loops over `Option`s and `Result`s.
fn check_arg_type(cx: &LateContext<'_>, pat: &Pat<'_>, arg: &Expr<'_>) {
    let ty = cx.typeck_results().expr_ty(arg);
    if is_type_diagnostic_item(cx, ty, sym::option_type) {
        span_lint_and_help(
            cx,
            super::FOR_LOOPS_OVER_FALLIBLES,
            arg.span,
            &format!(
                "for loop over `{0}`, which is an `Option`. This is more readably written as an \
                `if let` statement",
                snippet(cx, arg.span, "_")
            ),
            None,
            &format!(
                "consider replacing `for {0} in {1}` with `if let Some({0}) = {1}`",
                snippet(cx, pat.span, "_"),
                snippet(cx, arg.span, "_")
            ),
        );
    } else if is_type_diagnostic_item(cx, ty, sym::result_type) {
        span_lint_and_help(
            cx,
            super::FOR_LOOPS_OVER_FALLIBLES,
            arg.span,
            &format!(
                "for loop over `{0}`, which is a `Result`. This is more readably written as an \
                `if let` statement",
                snippet(cx, arg.span, "_")
            ),
            None,
            &format!(
                "consider replacing `for {0} in {1}` with `if let Ok({0}) = {1}`",
                snippet(cx, pat.span, "_"),
                snippet(cx, arg.span, "_")
            ),
        );
    }
}

fn lint_iter_method(cx: &LateContext<'_>, args: &[Expr<'_>], arg: &Expr<'_>, method_name: &str) {
    let mut applicability = Applicability::MachineApplicable;
    let object = snippet_with_applicability(cx, args[0].span, "_", &mut applicability);
    let muta = if method_name == "iter_mut" { "mut " } else { "" };
    span_lint_and_sugg(
        cx,
        super::EXPLICIT_ITER_LOOP,
        arg.span,
        "it is more concise to loop over references to containers instead of using explicit \
         iteration methods",
        "to write this more concisely, try",
        format!("&{}{}", muta, object),
        applicability,
    )
}

/// Returns `true` if the type of expr is one that provides `IntoIterator` impls
/// for `&T` and `&mut T`, such as `Vec`.
#[rustfmt::skip]
fn is_ref_iterable_type(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    // no walk_ptrs_ty: calling iter() on a reference can make sense because it
    // will allow further borrows afterwards
    let ty = cx.typeck_results().expr_ty(e);
    is_iterable_array(ty, cx) ||
    is_type_diagnostic_item(cx, ty, sym::vec_type) ||
    match_type(cx, ty, &paths::LINKED_LIST) ||
    is_type_diagnostic_item(cx, ty, sym!(hashmap_type)) ||
    is_type_diagnostic_item(cx, ty, sym!(hashset_type)) ||
    is_type_diagnostic_item(cx, ty, sym!(vecdeque_type)) ||
    match_type(cx, ty, &paths::BINARY_HEAP) ||
    match_type(cx, ty, &paths::BTREEMAP) ||
    match_type(cx, ty, &paths::BTREESET)
}

fn is_iterable_array<'tcx>(ty: Ty<'tcx>, cx: &LateContext<'tcx>) -> bool {
    // IntoIterator is currently only implemented for array sizes <= 32 in rustc
    match ty.kind() {
        ty::Array(_, n) => n
            .try_eval_usize(cx.tcx, cx.param_env)
            .map_or(false, |val| (0..=32).contains(&val)),
        _ => false,
    }
}
