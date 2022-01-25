use super::EXPLICIT_ITER_LOOP;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_trait_method;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_errors::Applicability;
use rustc_hir::{Expr, Mutability};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::sym;

pub(super) fn check(cx: &LateContext<'_>, self_arg: &Expr<'_>, arg: &Expr<'_>, method_name: &str) {
    let should_lint = match method_name {
        "iter" | "iter_mut" => is_ref_iterable_type(cx, self_arg),
        "into_iter" if is_trait_method(cx, arg, sym::IntoIterator) => {
            let receiver_ty = cx.typeck_results().expr_ty(self_arg);
            let receiver_ty_adjusted = cx.typeck_results().expr_ty_adjusted(self_arg);
            let ref_receiver_ty = cx.tcx.mk_ref(
                cx.tcx.lifetimes.re_erased,
                ty::TypeAndMut {
                    ty: receiver_ty,
                    mutbl: Mutability::Not,
                },
            );
            receiver_ty_adjusted == ref_receiver_ty
        },
        _ => false,
    };

    if !should_lint {
        return;
    }

    let mut applicability = Applicability::MachineApplicable;
    let object = snippet_with_applicability(cx, self_arg.span, "_", &mut applicability);
    let muta = if method_name == "iter_mut" { "mut " } else { "" };
    span_lint_and_sugg(
        cx,
        EXPLICIT_ITER_LOOP,
        arg.span,
        "it is more concise to loop over references to containers instead of using explicit \
         iteration methods",
        "to write this more concisely, try",
        format!("&{}{}", muta, object),
        applicability,
    );
}

/// Returns `true` if the type of expr is one that provides `IntoIterator` impls
/// for `&T` and `&mut T`, such as `Vec`.
#[rustfmt::skip]
fn is_ref_iterable_type(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    // no walk_ptrs_ty: calling iter() on a reference can make sense because it
    // will allow further borrows afterwards
    let ty = cx.typeck_results().expr_ty(e);
    is_iterable_array(ty, cx) ||
    is_type_diagnostic_item(cx, ty, sym::Vec) ||
    is_type_diagnostic_item(cx, ty, sym::LinkedList) ||
    is_type_diagnostic_item(cx, ty, sym::HashMap) ||
    is_type_diagnostic_item(cx, ty, sym::HashSet) ||
    is_type_diagnostic_item(cx, ty, sym::VecDeque) ||
    is_type_diagnostic_item(cx, ty, sym::BinaryHeap) ||
    is_type_diagnostic_item(cx, ty, sym::BTreeMap) ||
    is_type_diagnostic_item(cx, ty, sym::BTreeSet)
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
