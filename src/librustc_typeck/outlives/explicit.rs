// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::map as hir_map;
use rustc::hir;
use rustc::hir::def_id::{CrateNum, DefId, LocalDefId, LOCAL_CRATE};
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::ty::maps::Providers;
use rustc::ty::{self, CratePredicatesMap, TyCtxt};
use rustc_data_structures::sync::Lrc;
use util::nodemap::FxHashMap;

pub fn explicit_predicates<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    crate_num: CrateNum,
) -> FxHashMap<DefId, Lrc<Vec<ty::Predicate<'tcx>>>> {
    assert_eq!(crate_num, LOCAL_CRATE);
    let mut predicates: FxHashMap<DefId, Lrc<Vec<ty::Predicate<'tcx>>>> = FxHashMap();

    // iterate over the entire crate
    tcx.hir.krate().visit_all_item_likes(&mut ExplicitVisitor {
        tcx: tcx,
        explicit_predicates: &mut predicates,
        crate_num: crate_num,
    });

    predicates
}

pub struct ExplicitVisitor<'cx, 'tcx: 'cx> {
    tcx: TyCtxt<'cx, 'tcx, 'tcx>,
    explicit_predicates: &'cx mut FxHashMap<DefId, Lrc<Vec<ty::Predicate<'tcx>>>>,
    crate_num: CrateNum,
}

impl<'cx, 'tcx> ItemLikeVisitor<'tcx> for ExplicitVisitor<'cx, 'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let def_id = DefId {
            krate: self.crate_num,
            index: item.hir_id.owner,
        };

        let local_explicit_predicate = self.tcx.explicit_predicates_of(def_id);

        let filtered_predicates = local_explicit_predicate
            .predicates
            .into_iter()
            .filter(|pred| match pred {
                ty::Predicate::TypeOutlives(..) | ty::Predicate::RegionOutlives(..) => true,

                ty::Predicate::Trait(..)
                | ty::Predicate::Projection(..)
                | ty::Predicate::WellFormed(..)
                | ty::Predicate::ObjectSafe(..)
                | ty::Predicate::ClosureKind(..)
                | ty::Predicate::Subtype(..)
                | ty::Predicate::ConstEvaluatable(..) => false,
            })
            .collect();

        match item.node {
            hir::ItemStruct(..) | hir::ItemEnum(..) => {
                self.tcx.adt_def(def_id);
            }
            _ => {}
        }

        self.explicit_predicates
            .insert(def_id, Lrc::new(filtered_predicates));
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem) {}

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem) {}
}
