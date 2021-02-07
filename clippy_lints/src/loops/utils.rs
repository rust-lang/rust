use crate::utils::{get_trait_def_id, has_iter_method, implements_trait, paths, sugg};
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, Mutability};
use rustc_lint::LateContext;

/// If `arg` was the argument to a `for` loop, return the "cleanest" way of writing the
/// actual `Iterator` that the loop uses.
pub(super) fn make_iterator_snippet(cx: &LateContext<'_>, arg: &Expr<'_>, applic_ref: &mut Applicability) -> String {
    let impls_iterator = get_trait_def_id(cx, &paths::ITERATOR).map_or(false, |id| {
        implements_trait(cx, cx.typeck_results().expr_ty(arg), id, &[])
    });
    if impls_iterator {
        format!(
            "{}",
            sugg::Sugg::hir_with_applicability(cx, arg, "_", applic_ref).maybe_par()
        )
    } else {
        // (&x).into_iter() ==> x.iter()
        // (&mut x).into_iter() ==> x.iter_mut()
        match &arg.kind {
            ExprKind::AddrOf(BorrowKind::Ref, mutability, arg_inner)
                if has_iter_method(cx, cx.typeck_results().expr_ty(&arg_inner)).is_some() =>
            {
                let meth_name = match mutability {
                    Mutability::Mut => "iter_mut",
                    Mutability::Not => "iter",
                };
                format!(
                    "{}.{}()",
                    sugg::Sugg::hir_with_applicability(cx, &arg_inner, "_", applic_ref).maybe_par(),
                    meth_name,
                )
            }
            _ => format!(
                "{}.into_iter()",
                sugg::Sugg::hir_with_applicability(cx, arg, "_", applic_ref).maybe_par()
            ),
        }
    }
}
