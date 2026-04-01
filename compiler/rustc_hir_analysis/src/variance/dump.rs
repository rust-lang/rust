use std::fmt::Write;

use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
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

    if find_attr!(tcx, crate, RustcDumpVariancesOfOpaques) {
        for id in crate_items.opaques() {
            tcx.dcx().span_err(tcx.def_span(id), format_variances(tcx, id));
        }
    }

    for id in crate_items.owners() {
        if !find_attr!(tcx, id, RustcDumpVariances) {
            continue;
        }

        match tcx.def_kind(id) {
            DefKind::AssocFn | DefKind::Fn | DefKind::Enum | DefKind::Struct | DefKind::Union => {}
            DefKind::TyAlias if tcx.type_alias_is_lazy(id) => {}
            kind => {
                let message = format!(
                    "attr parsing didn't report an error for `#[{}]` on {kind:?}",
                    rustc_span::sym::rustc_dump_variances,
                );
                tcx.dcx().span_delayed_bug(tcx.def_span(id), message);
                continue;
            }
        }

        tcx.dcx().span_err(tcx.def_span(id), format_variances(tcx, id.def_id));
    }
}
