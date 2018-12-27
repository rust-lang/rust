use namespace::Namespace;
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::hir;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::traits::{self, IntercrateMode, FutureCompatOverlapErrorKind};
use rustc::ty::TyCtxt;
use rustc::ty::relate::TraitObjectMode;

use lint;

pub fn crate_inherent_impls_overlap_check<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                                    crate_num: CrateNum) {
    assert_eq!(crate_num, LOCAL_CRATE);
    let krate = tcx.hir().krate();
    krate.visit_all_item_likes(&mut InherentOverlapChecker { tcx });
}

struct InherentOverlapChecker<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>
}

impl<'a, 'tcx> InherentOverlapChecker<'a, 'tcx> {
    fn check_for_common_items_in_impls(
        &self, impl1: DefId, impl2: DefId,
        overlap: traits::OverlapResult,
        used_to_be_allowed: Option<FutureCompatOverlapErrorKind>)
    {

        let name_and_namespace = |def_id| {
            let item = self.tcx.associated_item(def_id);
            (item.ident, Namespace::from(item.kind))
        };

        let impl_items1 = self.tcx.associated_item_def_ids(impl1);
        let impl_items2 = self.tcx.associated_item_def_ids(impl2);

        for &item1 in &impl_items1[..] {
            let (name, namespace) = name_and_namespace(item1);

            for &item2 in &impl_items2[..] {
                if (name, namespace) == name_and_namespace(item2) {
                    let node_id = self.tcx.hir().as_local_node_id(impl1);
                    let mut err = match used_to_be_allowed {
                        Some(kind) if node_id.is_some() => {
                            let lint = match kind {
                                FutureCompatOverlapErrorKind::Issue43355 =>
                                    lint::builtin::INCOHERENT_FUNDAMENTAL_IMPLS,
                                FutureCompatOverlapErrorKind::Issue33140 =>
                                    lint::builtin::ORDER_DEPENDENT_TRAIT_OBJECTS,
                            };
                            self.tcx.struct_span_lint_node(
                                lint,
                                node_id.unwrap(),
                                self.tcx.span_of_impl(item1).unwrap(),
                                &format!("duplicate definitions with name `{}` (E0592)", name)
                            )
                        }
                        _ => {
                            struct_span_err!(self.tcx.sess,
                                             self.tcx.span_of_impl(item1).unwrap(),
                                             E0592,
                                             "duplicate definitions with name `{}`",
                                             name)
                        }
                    };

                    err.span_label(self.tcx.span_of_impl(item1).unwrap(),
                                   format!("duplicate definitions for `{}`", name));
                    err.span_label(self.tcx.span_of_impl(item2).unwrap(),
                                   format!("other definition for `{}`", name));

                    for cause in &overlap.intercrate_ambiguity_causes {
                        cause.add_intercrate_ambiguity_hint(&mut err);
                    }

                    err.emit();
                }
            }
        }
    }

    fn check_for_overlapping_inherent_impls(&self, ty_def_id: DefId) {
        let impls = self.tcx.inherent_impls(ty_def_id);

        for (i, &impl1_def_id) in impls.iter().enumerate() {
            for &impl2_def_id in &impls[(i + 1)..] {
                // First, check if the impl was forbidden under the
                // old rules. In that case, just have an error.
                let used_to_be_allowed = traits::overlapping_impls(
                    self.tcx,
                    impl1_def_id,
                    impl2_def_id,
                    IntercrateMode::Issue43355,
                    TraitObjectMode::NoSquash,
                    |overlap| {
                        self.check_for_common_items_in_impls(
                            impl1_def_id,
                            impl2_def_id,
                            overlap,
                            None,
                        );
                        false
                    },
                    || true,
                );

                if !used_to_be_allowed {
                    continue;
                }

                // Then, check if the impl was forbidden under only
                // #43355. In that case, emit an #43355 error.
                let used_to_be_allowed = traits::overlapping_impls(
                    self.tcx,
                    impl1_def_id,
                    impl2_def_id,
                    IntercrateMode::Fixed,
                    TraitObjectMode::NoSquash,
                    |overlap| {
                        self.check_for_common_items_in_impls(
                            impl1_def_id,
                            impl2_def_id,
                            overlap,
                            Some(FutureCompatOverlapErrorKind::Issue43355),
                        );
                        false
                    },
                    || true,
                );

                if !used_to_be_allowed {
                    continue;
                }

                // Then, check if the impl was forbidden under
                // #33140. In that case, emit a #33140 error.
                traits::overlapping_impls(
                    self.tcx,
                    impl1_def_id,
                    impl2_def_id,
                    IntercrateMode::Fixed,
                    TraitObjectMode::SquashAutoTraitsIssue33140,
                    |overlap| {
                        self.check_for_common_items_in_impls(
                            impl1_def_id,
                            impl2_def_id,
                            overlap,
                            Some(FutureCompatOverlapErrorKind::Issue33140),
                        );
                        false
                    },
                    || true,
                );
            }
        }
    }
}

impl<'a, 'tcx, 'v> ItemLikeVisitor<'v> for InherentOverlapChecker<'a, 'tcx> {
    fn visit_item(&mut self, item: &'v hir::Item) {
        match item.node {
            hir::ItemKind::Enum(..) |
            hir::ItemKind::Struct(..) |
            hir::ItemKind::Trait(..) |
            hir::ItemKind::Union(..) => {
                let type_def_id = self.tcx.hir().local_def_id(item.id);
                self.check_for_overlapping_inherent_impls(type_def_id);
            }
            _ => {}
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}
