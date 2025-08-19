use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_errors::Applicability;
use rustc_hir::{GenericArg, HirId, LetStmt, Node, Path, TyKind};
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;

use crate::transmute::MISSING_TRANSMUTE_ANNOTATIONS;

fn get_parent_local_binding_ty<'tcx>(cx: &LateContext<'tcx>, expr_hir_id: HirId) -> Option<LetStmt<'tcx>> {
    let mut parent_iter = cx.tcx.hir_parent_iter(expr_hir_id);
    if let Some((_, node)) = parent_iter.next() {
        match node {
            Node::LetStmt(local) => Some(*local),
            Node::Block(_) => {
                if let Some((parent_hir_id, Node::Expr(expr))) = parent_iter.next()
                    && matches!(expr.kind, rustc_hir::ExprKind::Block(_, _))
                {
                    get_parent_local_binding_ty(cx, parent_hir_id)
                } else {
                    None
                }
            },
            _ => None,
        }
    } else {
        None
    }
}

fn is_function_block(cx: &LateContext<'_>, expr_hir_id: HirId) -> bool {
    let def_id = cx.tcx.hir_enclosing_body_owner(expr_hir_id);
    if let Some(body) = cx.tcx.hir_maybe_body_owned_by(def_id) {
        return body.value.peel_blocks().hir_id == expr_hir_id;
    }
    false
}

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    path: &Path<'tcx>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
    expr_hir_id: HirId,
) -> bool {
    let last = path.segments.last().unwrap();
    if last.ident.span.in_external_macro(cx.tcx.sess.source_map()) {
        // If it comes from a non-local macro, we ignore it.
        return false;
    }
    let args = last.args;
    let missing_generic = match args {
        Some(args) if !args.args.is_empty() => args.args.iter().any(|arg| matches!(arg, GenericArg::Infer(_))),
        _ => true,
    };
    if !missing_generic {
        return false;
    }
    // If it's being set as a local variable value...
    if let Some(local) = get_parent_local_binding_ty(cx, expr_hir_id) {
        // ... which does have type annotations.
        if let Some(ty) = local.ty
            // If this is a `let x: _ =`, we should lint.
            && !matches!(ty.kind, TyKind::Infer(()))
        {
            return false;
        }
    // We check if this transmute is not the only element in the function
    } else if is_function_block(cx, expr_hir_id) {
        return false;
    }
    span_lint_and_sugg(
        cx,
        MISSING_TRANSMUTE_ANNOTATIONS,
        last.ident.span.with_hi(path.span.hi()),
        "transmute used without annotations",
        "consider adding missing annotations",
        format!("{}::<{from_ty}, {to_ty}>", last.ident),
        Applicability::MaybeIncorrect,
    );
    true
}
