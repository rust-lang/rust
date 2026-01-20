use std::fmt::Write;

use rustc_hir::attrs::AttributeKind;
use rustc_hir::def_id::{CRATE_DEF_ID, LocalDefId};
use rustc_hir::find_attr;
use rustc_middle::ty::{GenericArgs, TyCtxt};

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

    if find_attr!(tcx.get_all_attrs(CRATE_DEF_ID), AttributeKind::RustcVarianceOfOpaques) {
        for id in crate_items.opaques() {
            tcx.dcx().emit_err(crate::errors::VariancesOf {
                span: tcx.def_span(id),
                variances: format_variances(tcx, id),
            });
        }
    }

    for id in crate_items.free_items() {
        if !find_attr!(tcx.get_all_attrs(id.owner_id), AttributeKind::RustcVariance) {
            continue;
        }

        tcx.dcx().emit_err(crate::errors::VariancesOf {
            span: tcx.def_span(id.owner_id),
            variances: format_variances(tcx, id.owner_id.def_id),
        });
    }
}
