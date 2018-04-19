// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir;
use rustc::hir::def_id::{CrateNum, DefId};
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::ty::{self, TyCtxt};
use util::nodemap::FxHashMap;

use super::utils::*;

pub fn explicit_predicates<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    crate_num: CrateNum,
) -> FxHashMap<DefId, RequiredPredicates<'tcx>> {
    let mut predicates = FxHashMap::default();

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
    explicit_predicates: &'cx mut FxHashMap<DefId, RequiredPredicates<'tcx>>,
    crate_num: CrateNum,
}

impl<'cx, 'tcx> ItemLikeVisitor<'tcx> for ExplicitVisitor<'cx, 'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        let def_id = DefId {
            krate: self.crate_num,
            index: item.hir_id.owner,
        };

        let mut required_predicates = RequiredPredicates::default();
        let local_explicit_predicate = self.tcx.explicit_predicates_of(def_id).predicates;

        for pred in local_explicit_predicate.into_iter() {
            match pred {
                ty::Predicate::TypeOutlives(predicate) => {
                    let ty::OutlivesPredicate(ref ty, ref reg) = predicate.skip_binder();
                    insert_outlives_predicate(self.tcx, (*ty).into(), reg, &mut required_predicates)
                }

                ty::Predicate::RegionOutlives(predicate) => {
                    let ty::OutlivesPredicate(ref reg1, ref reg2) = predicate.skip_binder();
                    insert_outlives_predicate(
                        self.tcx,
                        (*reg1).into(),
                        reg2,
                        &mut required_predicates,
                    )
                }

                ty::Predicate::Trait(..)
                | ty::Predicate::Projection(..)
                | ty::Predicate::WellFormed(..)
                | ty::Predicate::ObjectSafe(..)
                | ty::Predicate::ClosureKind(..)
                | ty::Predicate::Subtype(..)
                | ty::Predicate::ConstEvaluatable(..) => (),
            }
        }

        self.explicit_predicates.insert(def_id, required_predicates);
    }

    fn visit_trait_item(&mut self, _trait_item: &'tcx hir::TraitItem) {}

    fn visit_impl_item(&mut self, _impl_item: &'tcx hir::ImplItem) {}
}
