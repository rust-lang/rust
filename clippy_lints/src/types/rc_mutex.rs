use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::is_ty_param_diagnostic_item;
use if_chain::if_chain;
use rustc_hir::{self as hir, def_id::DefId, QPath};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::RC_MUTEX;

pub(super) fn check(cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>, qpath: &QPath<'_>, def_id: DefId) -> bool {
    if_chain! {
        if cx.tcx.is_diagnostic_item(sym::Rc, def_id) ;
        if let Some(_) = is_ty_param_diagnostic_item(cx, qpath, sym::Mutex) ;
        then {
            span_lint_and_help(
                cx,
                RC_MUTEX,
                hir_ty.span,
                "usage of `Rc<Mutex<_>>`",
                None,
                "consider using `Rc<RefCell<_>>` or `Arc<Mutex<_>>` instead",
            );
            return true;
        }
    }

    false
}
