use rustc::ty::TyCtxt;
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_trait_selection::traits::{self, SkipLeakCheck};

pub fn crate_inherent_impls_overlap_check(tcx: TyCtxt<'_>, crate_num: CrateNum) {
    assert_eq!(crate_num, LOCAL_CRATE);
    let krate = tcx.hir().krate();
    krate.visit_all_item_likes(&mut InherentOverlapChecker { tcx });
}

struct InherentOverlapChecker<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl InherentOverlapChecker<'tcx> {
    /// Checks whether any associated items in impls 1 and 2 share the same identifier and
    /// namespace.
    fn impls_have_common_items(&self, impl1: DefId, impl2: DefId) -> bool {
        let impl_items1 = self.tcx.associated_items(impl1);
        let impl_items2 = self.tcx.associated_items(impl2);

        for item1 in impl_items1.in_definition_order() {
            let collision = impl_items2.filter_by_name_unhygienic(item1.ident.name).any(|item2| {
                // Symbols and namespace match, compare hygienically.
                item1.kind.namespace() == item2.kind.namespace()
                    && item1.ident.modern() == item2.ident.modern()
            });

            if collision {
                return true;
            }
        }

        false
    }

    fn check_for_common_items_in_impls(
        &self,
        impl1: DefId,
        impl2: DefId,
        overlap: traits::OverlapResult<'_>,
    ) {
        let impl_items1 = self.tcx.associated_items(impl1);
        let impl_items2 = self.tcx.associated_items(impl2);

        for item1 in impl_items1.in_definition_order() {
            let collision = impl_items2.filter_by_name_unhygienic(item1.ident.name).find(|item2| {
                // Symbols and namespace match, compare hygienically.
                item1.kind.namespace() == item2.kind.namespace()
                    && item1.ident.modern() == item2.ident.modern()
            });

            if let Some(item2) = collision {
                let name = item1.ident.modern();
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    self.tcx.span_of_impl(item1.def_id).unwrap(),
                    E0592,
                    "duplicate definitions with name `{}`",
                    name
                );
                err.span_label(
                    self.tcx.span_of_impl(item1.def_id).unwrap(),
                    format!("duplicate definitions for `{}`", name),
                );
                err.span_label(
                    self.tcx.span_of_impl(item2.def_id).unwrap(),
                    format!("other definition for `{}`", name),
                );

                for cause in &overlap.intercrate_ambiguity_causes {
                    cause.add_intercrate_ambiguity_hint(&mut err);
                }

                if overlap.involves_placeholder {
                    traits::add_placeholder_note(&mut err);
                }

                err.emit();
            }
        }
    }

    fn check_for_overlapping_inherent_impls(&self, impl1_def_id: DefId, impl2_def_id: DefId) {
        traits::overlapping_impls(
            self.tcx,
            impl1_def_id,
            impl2_def_id,
            // We go ahead and just skip the leak check for
            // inherent impls without warning.
            SkipLeakCheck::Yes,
            |overlap| {
                self.check_for_common_items_in_impls(impl1_def_id, impl2_def_id, overlap);
                false
            },
            || true,
        );
    }
}

impl ItemLikeVisitor<'v> for InherentOverlapChecker<'tcx> {
    fn visit_item(&mut self, item: &'v hir::Item<'v>) {
        match item.kind {
            hir::ItemKind::Enum(..)
            | hir::ItemKind::Struct(..)
            | hir::ItemKind::Trait(..)
            | hir::ItemKind::Union(..) => {
                let ty_def_id = self.tcx.hir().local_def_id(item.hir_id);
                let impls = self.tcx.inherent_impls(ty_def_id);

                for (i, &impl1_def_id) in impls.iter().enumerate() {
                    for &impl2_def_id in &impls[(i + 1)..] {
                        if self.impls_have_common_items(impl1_def_id, impl2_def_id) {
                            self.check_for_overlapping_inherent_impls(impl1_def_id, impl2_def_id);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem<'v>) {}

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem<'v>) {}
}
