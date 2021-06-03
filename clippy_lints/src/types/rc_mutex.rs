use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{get_qpath_generic_tys, is_ty_param_diagnostic_item};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{self as hir, def_id::DefId, QPath, TyKind};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::RC_MUTEX;

pub(super) fn check(cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>, qpath: &QPath<'_>, def_id: DefId) -> bool {
    if_chain! {
        if cx.tcx.is_diagnostic_item(sym::Rc, def_id) ;
        if let Some(ty) = is_ty_param_diagnostic_item(cx, qpath, sym!(mutex_type)) ;
        if let TyKind::Path(ref qpath_inner)=ty.kind;

        then{
            let mut applicability = Applicability::MachineApplicable;

            let inner_span = match get_qpath_generic_tys(qpath_inner).next() {
                Some(ty) => ty.span,
                None => return false,
            };

            span_lint_and_sugg(
                cx,
                RC_MUTEX,
                hir_ty.span,
                "you seem to be trying to use `Rc<Mutex<T>>`. Consider using `Rc<RefCell<T>>`",
                "try",
                format!(
                    "Rc<RefCell<{}>>",
                    snippet_with_applicability(cx, inner_span, "..", &mut applicability)
                ),
                applicability,
            );
            return true;
        }
    }

    false
}
