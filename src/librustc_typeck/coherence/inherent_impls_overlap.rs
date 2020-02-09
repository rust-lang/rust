use crate::namespace::Namespace;
use rustc::traits::{self, IntercrateMode, SkipLeakCheck};
use rustc::ty::{AssocItem, TyCtxt};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use smallvec::SmallVec;
use std::collections::hash_map::Entry;

pub fn crate_inherent_impls_overlap_check(tcx: TyCtxt<'_>, crate_num: CrateNum) {
    assert_eq!(crate_num, LOCAL_CRATE);
    let krate = tcx.hir().krate();
    krate.visit_all_item_likes(&mut InherentOverlapChecker { tcx });
}

struct InherentOverlapChecker<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl InherentOverlapChecker<'tcx> {
    /// Emits an error if the impls defining `item1` and `item2` overlap.
    fn forbid_overlap(&self, item1: &AssocItem, item2: &AssocItem) {
        let impl1_def_id = item1.container.id();
        let impl2_def_id = item2.container.id();
        let name = item1.ident;

        traits::overlapping_impls(
            self.tcx,
            impl1_def_id,
            impl2_def_id,
            IntercrateMode::Issue43355,
            // We go ahead and just skip the leak check for
            // inherent impls without warning.
            SkipLeakCheck::Yes,
            |overlap| {
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
            },
            || {},
        );
    }

    fn process_local_ty(&self, ty_def_id: DefId) {
        let impls = self.tcx.inherent_impls(ty_def_id);

        // Create a map from `Symbol`s to the list of `&'tcx AssocItem`s with that name, across all
        // inherent impls.
        let mut item_map: FxHashMap<_, SmallVec<[&AssocItem; 1]>> = FxHashMap::default();
        for impl_def_id in impls {
            let impl_items = self.tcx.associated_items(*impl_def_id);

            for impl_item in impl_items {
                match item_map.entry(impl_item.ident.name) {
                    Entry::Occupied(mut occupied) => {
                        // Do a proper name check respecting namespaces and hygiene against all
                        // items in this map entry.
                        let (ident1, ns1) =
                            (impl_item.ident.modern(), Namespace::from(impl_item.kind));

                        for impl_item2 in occupied.get() {
                            let (ident2, ns2) =
                                (impl_item2.ident.modern(), Namespace::from(impl_item2.kind));

                            if ns1 == ns2 && ident1 == ident2 {
                                // Items with same name. Their containing impls must not overlap.

                                self.forbid_overlap(impl_item2, impl_item);
                            }
                        }

                        occupied.get_mut().push(impl_item);
                    }
                    Entry::Vacant(vacant) => {
                        vacant.insert(SmallVec::from([impl_item]));
                    }
                }
            }
        }
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
                self.process_local_ty(ty_def_id);
            }
            _ => {}
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem<'v>) {}

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem<'v>) {}
}
