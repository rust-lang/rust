// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::hir;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::traits;
use rustc::traits::IntercrateAmbiguityCause;
use rustc::ty::{self, TyCtxt};

pub fn crate_inherent_impls_overlap_check<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                                    crate_num: CrateNum) {
    assert_eq!(crate_num, LOCAL_CRATE);
    let krate = tcx.hir.krate();
    krate.visit_all_item_likes(&mut InherentOverlapChecker { tcx });
}

struct InherentOverlapChecker<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>
}

impl<'a, 'tcx> InherentOverlapChecker<'a, 'tcx> {
    fn check_for_common_items_in_impls(&self, impl1: DefId, impl2: DefId,
                                       overlap: traits::OverlapResult) {
        #[derive(Copy, Clone, PartialEq)]
        enum Namespace {
            Type,
            Value,
        }

        let name_and_namespace = |def_id| {
            let item = self.tcx.associated_item(def_id);
            (item.name, match item.kind {
                ty::AssociatedKind::Type => Namespace::Type,
                ty::AssociatedKind::Const |
                ty::AssociatedKind::Method => Namespace::Value,
            })
        };

        let impl_items1 = self.tcx.associated_item_def_ids(impl1);
        let impl_items2 = self.tcx.associated_item_def_ids(impl2);

        for &item1 in &impl_items1[..] {
            let (name, namespace) = name_and_namespace(item1);

            for &item2 in &impl_items2[..] {
                if (name, namespace) == name_and_namespace(item2) {
                    let mut err = struct_span_err!(self.tcx.sess,
                                                   self.tcx.span_of_impl(item1).unwrap(),
                                                   E0592,
                                                   "duplicate definitions with name `{}`",
                                                   name);

                    err.span_label(self.tcx.span_of_impl(item1).unwrap(),
                                   format!("duplicate definitions for `{}`", name));
                    err.span_label(self.tcx.span_of_impl(item2).unwrap(),
                                   format!("other definition for `{}`", name));

                    for cause in &overlap.intercrate_ambiguity_causes {
                        match cause {
                            &IntercrateAmbiguityCause::DownstreamCrate(def_id) => {
                                err.note(&format!("downstream crates may implement `{}`",
                                                  self.tcx.item_path_str(def_id)));
                            }
                            &IntercrateAmbiguityCause::UpstreamCrateUpdate(def_id) => {
                                err.note(&format!("upstream crates may add new impl for `{}` \
                                                  in future versions",
                                                  self.tcx.item_path_str(def_id)));
                            }
                        }
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
                self.tcx.infer_ctxt().enter(|infcx| {
                    if let Some(overlap) =
                            traits::overlapping_impls(&infcx, impl1_def_id, impl2_def_id) {
                        self.check_for_common_items_in_impls(impl1_def_id, impl2_def_id, overlap)
                    }
                });
            }
        }
    }
}

impl<'a, 'tcx, 'v> ItemLikeVisitor<'v> for InherentOverlapChecker<'a, 'tcx> {
    fn visit_item(&mut self, item: &'v hir::Item) {
        match item.node {
            hir::ItemEnum(..) |
            hir::ItemStruct(..) |
            hir::ItemTrait(..) |
            hir::ItemUnion(..) => {
                let type_def_id = self.tcx.hir.local_def_id(item.id);
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

