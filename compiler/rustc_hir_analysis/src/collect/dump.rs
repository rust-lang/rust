use rustc_hir as hir;
use rustc_hir::def_id::{CRATE_DEF_ID, LocalDefId};
use rustc_hir::intravisit;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt};
use rustc_span::sym;

pub(crate) fn opaque_hidden_types(tcx: TyCtxt<'_>) {
    if !tcx.has_attr(CRATE_DEF_ID, sym::rustc_hidden_type_of_opaques) {
        return;
    }

    for id in tcx.hir_crate_items(()).opaques() {
        if let hir::OpaqueTyOrigin::FnReturn { parent: fn_def_id, .. }
        | hir::OpaqueTyOrigin::AsyncFn { parent: fn_def_id, .. } =
            tcx.hir_expect_opaque_ty(id).origin
            && let hir::Node::TraitItem(trait_item) = tcx.hir_node_by_def_id(fn_def_id)
            && let (_, hir::TraitFn::Required(..)) = trait_item.expect_fn()
        {
            continue;
        }

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
    for iid in tcx.hir_free_items() {
        let did = iid.owner_id.def_id;
        if tcx.has_attr(did, sym::rustc_dump_def_parents) {
            struct AnonConstFinder<'tcx> {
                tcx: TyCtxt<'tcx>,
                anon_consts: Vec<LocalDefId>,
            }

            impl<'tcx> intravisit::Visitor<'tcx> for AnonConstFinder<'tcx> {
                type NestedFilter = nested_filter::All;

                fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
                    self.tcx
                }

                fn visit_anon_const(&mut self, c: &'tcx rustc_hir::AnonConst) {
                    self.anon_consts.push(c.def_id);
                    intravisit::walk_anon_const(self, c)
                }
            }

            // Look for any anon consts inside of this item as there is no way to apply
            // the `rustc_dump_def_parents` attribute to the anon const so it would not be possible
            // to see what its def parent is.
            let mut anon_ct_finder = AnonConstFinder { tcx, anon_consts: vec![] };
            intravisit::walk_item(&mut anon_ct_finder, tcx.hir_item(iid));

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

pub(crate) fn vtables<'tcx>(tcx: TyCtxt<'tcx>) {
    for id in tcx.hir_free_items() {
        let def_id = id.owner_id.def_id;

        let Some(attr) = tcx.get_attr(def_id, sym::rustc_dump_vtable) else {
            continue;
        };

        let vtable_entries = match tcx.hir_item(id).kind {
            hir::ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }) => {
                let trait_ref = tcx.impl_trait_ref(def_id).unwrap().instantiate_identity();
                if trait_ref.has_non_region_param() {
                    tcx.dcx().span_err(
                        attr.span(),
                        "`rustc_dump_vtable` must be applied to non-generic impl",
                    );
                    continue;
                }
                if !tcx.is_dyn_compatible(trait_ref.def_id) {
                    tcx.dcx().span_err(
                        attr.span(),
                        "`rustc_dump_vtable` must be applied to dyn-compatible trait",
                    );
                    continue;
                }
                let Ok(trait_ref) = tcx
                    .try_normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), trait_ref)
                else {
                    tcx.dcx().span_err(
                        attr.span(),
                        "`rustc_dump_vtable` applied to impl header that cannot be normalized",
                    );
                    continue;
                };
                tcx.vtable_entries(trait_ref)
            }
            hir::ItemKind::TyAlias(..) => {
                let ty = tcx.type_of(def_id).instantiate_identity();
                if ty.has_non_region_param() {
                    tcx.dcx().span_err(
                        attr.span(),
                        "`rustc_dump_vtable` must be applied to non-generic type",
                    );
                    continue;
                }
                let Ok(ty) =
                    tcx.try_normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), ty)
                else {
                    tcx.dcx().span_err(
                        attr.span(),
                        "`rustc_dump_vtable` applied to type alias that cannot be normalized",
                    );
                    continue;
                };
                let ty::Dynamic(data, _) = *ty.kind() else {
                    tcx.dcx()
                        .span_err(attr.span(), "`rustc_dump_vtable` to type alias of dyn type");
                    continue;
                };
                if let Some(principal) = data.principal() {
                    tcx.vtable_entries(
                        tcx.instantiate_bound_regions_with_erased(principal).with_self_ty(tcx, ty),
                    )
                } else {
                    TyCtxt::COMMON_VTABLE_ENTRIES
                }
            }
            _ => {
                tcx.dcx().span_err(
                    attr.span(),
                    "`rustc_dump_vtable` only applies to impl, or type alias of dyn type",
                );
                continue;
            }
        };

        tcx.dcx().span_err(tcx.def_span(def_id), format!("vtable entries: {vtable_entries:#?}"));
    }
}
