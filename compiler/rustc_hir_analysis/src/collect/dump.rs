use rustc_hir::def::DefKind;
use rustc_hir::def_id::CRATE_DEF_ID;
use rustc_middle::ty::TyCtxt;
use rustc_span::sym;

pub(crate) fn opaque_hidden_types(tcx: TyCtxt<'_>) {
    if !tcx.has_attr(CRATE_DEF_ID, sym::rustc_hidden_type_of_opaques) {
        return;
    }

    for id in tcx.hir().items() {
        let DefKind::OpaqueTy = tcx.def_kind(id.owner_id) else { continue };

        let ty = tcx.type_of(id.owner_id).instantiate_identity();

        tcx.dcx().emit_err(crate::errors::TypeOf { span: tcx.def_span(id.owner_id), ty });
    }
}

pub(crate) fn predicates_and_item_bounds(tcx: TyCtxt<'_>) {
    for id in tcx.hir_crate_items(()).owners() {
        if tcx.has_attr(id, sym::rustc_dump_predicates) {
            let preds = tcx.predicates_of(id).instantiate_identity(tcx).predicates;
            let span = tcx.def_span(id);

            let mut diag = tcx.dcx().struct_span_err(span, sym::rustc_dump_predicates.as_str());
            for pred in preds {
                diag.note(format!("{pred:?}"));
            }
            diag.emit();
        }
        if tcx.has_attr(id, sym::rustc_dump_item_bounds) {
            let bounds = tcx.item_bounds(id).instantiate_identity();
            let span = tcx.def_span(id);

            let mut diag = tcx.dcx().struct_span_err(span, sym::rustc_dump_item_bounds.as_str());
            for bound in bounds {
                diag.note(format!("{bound:?}"));
            }
            diag.emit();
        }
    }
}
