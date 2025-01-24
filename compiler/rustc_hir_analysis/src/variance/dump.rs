use std::fmt::Write;

use rustc_hir::def_id::{CRATE_DEF_ID, LocalDefId};
use rustc_middle::ty::{GenericArgs, TyCtxt};
use rustc_span::sym;

fn format_variances(tcx: TyCtxt<'_>, def_id: LocalDefId) -> String {
    let variances = tcx.variances_of(def_id);
    let generics = GenericArgs::identity_for_item(tcx, def_id);
    // 7 = 2-letter parameter + ": " + 1-letter variance + ", "
    let mut ret = String::with_capacity(2 + 7 * variances.len());
    ret.push('[');
    for (arg, variance) in generics.iter().zip(variances.iter()) {
        write!(ret, "{arg}: {variance:?}, ").unwrap();
    }
    // Remove trailing `, `.
    if !variances.is_empty() {
        ret.pop();
        ret.pop();
    }
    ret.push(']');
    ret
}

pub(crate) fn variances(tcx: TyCtxt<'_>) {
    let crate_items = tcx.hir_crate_items(());

    if tcx.has_attr(CRATE_DEF_ID, sym::rustc_variance_of_opaques) {
        for id in crate_items.opaques() {
            tcx.dcx().emit_err(crate::errors::VariancesOf {
                span: tcx.def_span(id),
                variances: format_variances(tcx, id),
            });
        }
    }

    for id in crate_items.free_items() {
        if !tcx.has_attr(id.owner_id, sym::rustc_variance) {
            continue;
        }

        tcx.dcx().emit_err(crate::errors::VariancesOf {
            span: tcx.def_span(id.owner_id),
            variances: format_variances(tcx, id.owner_id.def_id),
        });
    }
}
