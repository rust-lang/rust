use rustc_hir as hir;
use rustc_hir_pretty::qpath_to_string;
use rustc_lint_defs::builtin::STATIC_MUT_REF;
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;
use rustc_type_ir::Mutability;

use crate::errors;

/// Check for shared or mutable references of `static mut` inside expression
pub fn maybe_expr_static_mut(tcx: TyCtxt<'_>, expr: hir::Expr<'_>) {
    let span = expr.span;
    let hir_id = expr.hir_id;
    if let hir::ExprKind::AddrOf(borrow_kind, m, expr) = expr.kind
        && matches!(borrow_kind, hir::BorrowKind::Ref)
        && let Some(var) = is_path_static_mut(*expr)
    {
        handle_static_mut_ref(
            tcx,
            span,
            var,
            span.edition().at_least_rust_2024(),
            matches!(m, Mutability::Mut),
            hir_id,
        );
    }
}

fn is_path_static_mut(expr: hir::Expr<'_>) -> Option<String> {
    if let hir::ExprKind::Path(qpath) = expr.kind
        && let hir::QPath::Resolved(_, path) = qpath
        && let hir::def::Res::Def(def_kind, _) = path.res
        && let hir::def::DefKind::Static(mt) = def_kind
        && matches!(mt, Mutability::Mut)
    {
        return Some(qpath_to_string(&qpath));
    }
    None
}

fn handle_static_mut_ref(
    tcx: TyCtxt<'_>,
    span: Span,
    var: String,
    e2024: bool,
    mutable: bool,
    hir_id: hir::HirId,
) {
    if e2024 {
        let sugg = if mutable {
            errors::StaticMutRefSugg::Mut { span, var }
        } else {
            errors::StaticMutRefSugg::Shared { span, var }
        };
        tcx.sess.parse_sess.dcx.emit_err(errors::StaticMutRef { span, sugg });
        return;
    }

    let (label, sugg, shared) = if mutable {
        (
            errors::RefOfMutStaticLabel::Mut { span },
            errors::RefOfMutStaticSugg::Mut { span, var },
            "mutable ",
        )
    } else {
        (
            errors::RefOfMutStaticLabel::Shared { span },
            errors::RefOfMutStaticSugg::Shared { span, var },
            "shared ",
        )
    };
    tcx.emit_spanned_lint(
        STATIC_MUT_REF,
        hir_id,
        span,
        errors::RefOfMutStatic { shared, why_note: (), label, sugg },
    );
}
