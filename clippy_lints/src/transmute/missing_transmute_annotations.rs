use rustc_errors::Applicability;
use rustc_hir::{GenericArg, HirId, Node, Path, TyKind};
use rustc_lint::LateContext;
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::Ty;

use clippy_utils::diagnostics::span_lint_and_sugg;

use crate::transmute::MISSING_TRANSMUTE_ANNOTATIONS;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    path: &Path<'tcx>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
    expr_hir_id: HirId,
) -> bool {
    let last = path.segments.last().unwrap();
    if in_external_macro(cx.tcx.sess, last.ident.span) {
        // If it comes from a non-local macro, we ignore it.
        return false;
    }
    let args = last.args;
    let missing_generic = match args {
        Some(args) if !args.args.is_empty() => args.args.iter().any(|arg| match arg {
            GenericArg::Infer(_) => true,
            GenericArg::Type(ty) => matches!(ty.kind, TyKind::Infer),
            _ => false,
        }),
        _ => true,
    };
    if !missing_generic {
        return false;
    }
    // If it's being set as a local variable value...
    if let Some((_, node)) = cx.tcx.hir().parent_iter(expr_hir_id).next()
        && let Node::Local(local) = node
        // ... which does have type annotations.
        && local.ty.is_some()
    {
        return false;
    }
    span_lint_and_sugg(
        cx,
        MISSING_TRANSMUTE_ANNOTATIONS,
        last.ident.span.with_hi(path.span.hi()),
        "transmute used without annotations",
        "consider adding missing annotations",
        format!("{}::<{from_ty}, {to_ty}>", last.ident.as_str()),
        Applicability::MaybeIncorrect,
    );
    true
}
