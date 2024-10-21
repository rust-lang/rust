use rustc_hir::def_id::{CRATE_DEF_ID, LocalDefId};
use rustc_hir::intravisit;
use rustc_middle::hir::nested_filter::OnlyBodies;
use rustc_middle::ty::TyCtxt;
use rustc_span::sym;

pub(crate) fn opaque_hidden_types(tcx: TyCtxt<'_>) {
    if !tcx.has_attr(CRATE_DEF_ID, sym::rustc_hidden_type_of_opaques) {
        return;
    }

    for id in tcx.hir_crate_items(()).opaques() {
        let ty = tcx.type_of(id).instantiate_identity();
        let span = tcx.def_span(id);
        tcx.dcx().emit_err(crate::errors::TypeOf { span, ty });
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

pub(crate) fn def_parents(tcx: TyCtxt<'_>) {
    for did in tcx.hir().body_owners() {
        if tcx.has_attr(did, sym::rustc_dump_def_parents) {
            struct AnonConstFinder<'tcx> {
                tcx: TyCtxt<'tcx>,
                anon_consts: Vec<LocalDefId>,
            }

            impl<'tcx> intravisit::Visitor<'tcx> for AnonConstFinder<'tcx> {
                type NestedFilter = OnlyBodies;

                fn nested_visit_map(&mut self) -> Self::Map {
                    self.tcx.hir()
                }

                fn visit_anon_const(&mut self, c: &'tcx rustc_hir::AnonConst) {
                    self.anon_consts.push(c.def_id);
                    intravisit::walk_anon_const(self, c)
                }
            }

            // Look for any anon consts inside of this body owner as there is no way to apply
            // the `rustc_dump_def_parents` attribute to the anon const so it would not be possible
            // to see what its def parent is.
            let mut anon_ct_finder = AnonConstFinder { tcx, anon_consts: vec![] };
            intravisit::walk_expr(&mut anon_ct_finder, tcx.hir().body_owned_by(did).value);

            for did in [did].into_iter().chain(anon_ct_finder.anon_consts) {
                let span = tcx.def_span(did);

                let mut diag = tcx.dcx().struct_span_err(
                    span,
                    format!("{}: {did:?}", sym::rustc_dump_def_parents.as_str()),
                );

                let mut current_did = did.to_def_id();
                while let Some(parent_did) = tcx.opt_parent(current_did) {
                    current_did = parent_did;
                    diag.span_note(tcx.def_span(parent_did), format!("{parent_did:?}"));
                }
                diag.emit();
            }
        }
    }
}
